#!/usr/bin/env python
"""Evaluation harness for the sparse transformer STAGE (TemporalSparseTransformerBlock).

Replays the real per-window stage inputs captured from a production run
(default: /tmp/sample2_transformer_inputs/, sample 2 @ 720p) and reports, for a
chosen stage implementation:
  - aggregate latency over all windows (the end-to-end stage cost to optimize)
  - per-window latency
  - correctness vs the EAGER reference (max-abs-err + PSNR)

This is the canonical evaluation an optimizer (kernel agent) should drive:
faster total latency at iso-correctness (PSNR vs eager) wins.

Usage:
    python scripts/experiments/eval_transformer_stage.py --impl eager-fp16
    python scripts/experiments/eval_transformer_stage.py --impl fast-fp16
    python scripts/experiments/eval_transformer_stage.py --impl all
    # pick GPU:  CUDA_VISIBLE_DEVICES=1 python ... --impl all
"""
import argparse, glob, json, os, sys, time

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)

import torch

CKPT = os.path.join(_REPO, 'weights', 'ProPainter.pth')


def build_block(cls, half):
    """Build a stage block with checkpoint weights, ctor args inferred from the
    loaded model (so shapes always match the trained checkpoint)."""
    from model.propainter import InpaintGenerator
    model = InpaintGenerator(model_path=CKPT).cuda().eval()
    sd = model.transformers.state_dict()
    at0 = model.transformers.transformer[0].attention
    t2t = model.transformers.transformer[0].mlp.t2t_params
    pw = at0.pool_layer.weight
    blk = cls(dim=at0.query.weight.shape[0], n_head=at0.n_head,
              window_size=at0.window_size, pool_size=(pw.shape[2], pw.shape[3]),
              depths=model.transformers.depths, t2t_params=t2t).cuda().eval()
    blk.load_state_dict(sd)
    del model
    torch.cuda.empty_cache()
    return blk.half() if half else blk.float()


def load_windows(fixture_dir, limit=None):
    paths = sorted(glob.glob(os.path.join(fixture_dir, 'window_*.pt')))
    if limit:
        paths = paths[:limit]
    wins = []
    for p in paths:
        d = torch.load(p)
        wins.append(d)
    return wins


def psnr(a, b):
    mse = (a.float() - b.float()).pow(2).mean().item()
    if mse == 0:
        return float('inf')
    # signal range estimated from the reference
    rng = (b.float().max() - b.float().min()).item()
    return 10.0 * torch.log10(torch.tensor(rng * rng / mse)).item()


@torch.no_grad()
def run_impl(name, blk, wins, dtype, warmup=2, reps=3, ref_outs=None):
    dev = 'cuda'

    def cast(d):
        x = d['x'].to(dev).to(dtype)
        lm = None if d['l_mask'] is None else d['l_mask'].to(dev).to(dtype)
        return x, d['fold_x_size'], lm, d['t_dilation']

    cached = [cast(d) for d in wins]

    # correctness: collect outputs once
    outs = []
    for (x, fold, lm, td) in cached:
        outs.append(blk(x, fold, lm, t_dilation=td).float().cpu())

    # latency: warmup then timed reps over the full window set
    for _ in range(warmup):
        for (x, fold, lm, td) in cached:
            blk(x, fold, lm, t_dilation=td)
    torch.cuda.synchronize()

    per_win = [0.0] * len(cached)
    totals = []
    for _ in range(reps):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for i, (x, fold, lm, td) in enumerate(cached):
            torch.cuda.synchronize(); s = time.perf_counter()
            blk(x, fold, lm, t_dilation=td)
            torch.cuda.synchronize(); per_win[i] += (time.perf_counter() - s) * 1e3
        torch.cuda.synchronize(); totals.append((time.perf_counter() - t0) * 1e3)
    per_win = [v / reps for v in per_win]
    total = sorted(totals)[len(totals) // 2]

    corr = ''
    if ref_outs is not None:
        maxerr = max((o - r).abs().max().item() for o, r in zip(outs, ref_outs))
        p = min(psnr(o, r) for o, r in zip(outs, ref_outs))
        corr = f"  maxerr={maxerr:.4f}  minPSNR={p:.1f}dB"
    print(f"[{name:14s}] stage total = {total:8.1f} ms over {len(cached)} windows"
          f"  ({total/len(cached):.2f} ms/win){corr}")
    return {'name': name, 'total_ms': total, 'per_win_ms': per_win, 'outs': outs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fixture', default='/tmp/sample2_transformer_inputs')
    ap.add_argument('--impl', default='all',
                    choices=['eager-fp16', 'eager-fp32', 'fast-fp16', 'fast-fp32', 'all'])
    ap.add_argument('--windows', type=int, default=None, help='limit #windows (debug)')
    args = ap.parse_args()

    wins = load_windows(args.fixture, args.windows)
    print(f"loaded {len(wins)} windows from {args.fixture}")
    print(f"  x[0]={tuple(wins[0]['x'].shape)} fold={wins[0]['fold_x_size']}")

    from model.modules.sparse_transformer import TemporalSparseTransformerBlock as Eager
    from model.propainter_fast import FastTemporalSparseTransformerBlock as Fast

    # eager-fp32 is the correctness reference
    print("\n=== building eager-fp32 reference ===")
    ref_blk = build_block(Eager, half=False)
    ref = run_impl('eager-fp32(ref)', ref_blk, wins, torch.float32, ref_outs=None)
    ref_outs = ref['outs']
    del ref_blk; torch.cuda.empty_cache()

    results = [ref]
    todo = []
    if args.impl in ('eager-fp16', 'all'):
        todo.append(('eager-fp16', Eager, True, torch.float16))
    if args.impl in ('fast-fp16', 'all'):
        todo.append(('fast-fp16', Fast, True, torch.float16))
    if args.impl in ('fast-fp32', 'all'):
        todo.append(('fast-fp32', Fast, False, torch.float32))

    for name, cls, half, dtype in todo:
        blk = build_block(cls, half=half)
        if cls is Fast:
            blk._fast_enabled = True
            blk._strict = True  # raise if the fast path can't engage
        results.append(run_impl(name, blk, wins, dtype, ref_outs=ref_outs))
        del blk; torch.cuda.empty_cache()

    print("\n=== summary (lower total = better; PSNR vs eager-fp32) ===")
    base = next((r for r in results if r['name'] == 'eager-fp16'), ref)
    for r in results:
        if r['name'] == 'eager-fp32(ref)':
            continue
        spd = base['total_ms'] / r['total_ms']
        print(f"  {r['name']:14s} {r['total_ms']:8.1f} ms   {spd:.2f}x vs eager-fp16")


if __name__ == '__main__':
    main()
