#!/usr/bin/env python
"""Capture the real per-window input to the sparse transformer STAGE
(TemporalSparseTransformerBlock.forward) from a real inference_propainter.py run,
to a DURABLE location (default: <repo>/eval_fixtures/sample2_720p/).

Monkeypatches the stage forward, saves {x, fold_x_size, l_mask, t_dilation} per
window (input-only, fp16) + manifest.json, then runs the normal inference.

Usage (pick a free GPU):
    CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python scripts/experiments/capture_stage_inputs.py \
        --video inputs/samples/2/input.mp4 --mask inputs/samples/2/masks.mp4 \
        --height 720 --width 1280 --frames 200 --fp16 \
        --neighbor_length 10 --subvideo_length 80 --outdir eval_fixtures/sample2_720p
"""
import argparse, json, os, sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
os.environ.setdefault('PROPAINTER_FAST', '0')  # eager so the stage module executes

# ---- parse our extra args, leave the rest for inference_propainter.py ----
pre = argparse.ArgumentParser(add_help=False)
pre.add_argument('--outdir', default=os.path.join(_REPO, 'eval_fixtures', 'sample2_720p'))
ns, remaining = pre.parse_known_args()
OUTDIR = ns.outdir if os.path.isabs(ns.outdir) else os.path.join(_REPO, ns.outdir)
os.makedirs(OUTDIR, exist_ok=True)

import torch
import model.modules.sparse_transformer as st

_calls = []
_orig = st.TemporalSparseTransformerBlock.forward


def _meta(t):
    if t is None:
        return None
    d = {'shape': tuple(t.shape), 'dtype': str(t.dtype)}
    if t.is_floating_point():
        tf = t.float()
        d.update(min=round(tf.min().item(), 4), max=round(tf.max().item(), 4),
                 mean=round(tf.mean().item(), 5), std=round(tf.std().item(), 5))
    return d


def _hook(self, x, fold_x_size, l_mask=None, t_dilation=2):
    idx = len(_calls)
    out = _orig(self, x, fold_x_size, l_mask, t_dilation)
    rec = {'window': idx, 'T': int(x.size(1)), 'x': _meta(x),
           'fold_x_size': list(fold_x_size), 'l_mask': _meta(l_mask),
           't_dilation': int(t_dilation), 'out_ref': _meta(out)}
    if l_mask is not None:
        rec['mask_coverage'] = round((l_mask > 0).float().mean().item(), 4)
    _calls.append(rec)
    torch.save({'x': x.detach().half().cpu(), 'fold_x_size': tuple(fold_x_size),
                'l_mask': None if l_mask is None else l_mask.detach().half().cpu(),
                't_dilation': int(t_dilation)},
               os.path.join(OUTDIR, f'window_{idx:03d}.pt'))
    return out


st.TemporalSparseTransformerBlock.forward = _hook

import atexit


@atexit.register
def _dump():
    with open(os.path.join(OUTDIR, 'manifest.json'), 'w') as f:
        json.dump(_calls, f, indent=1)
    from collections import Counter
    print(f"\n[CAPTURE] {len(_calls)} windows -> {OUTDIR}/window_*.pt")
    if _calls:
        g = _calls[0]['x']['shape']
        print(f"[CAPTURE] token grid h,w={g[2]},{g[3]} C={g[4]} fold={_calls[0]['fold_x_size']}")
        print("[CAPTURE] T dist:", dict(sorted(Counter(c['T'] for c in _calls).items())))


# ---- run the real inference with the remaining argv ----
sys.argv = ['inference_propainter.py'] + remaining
exec(open(os.path.join(_REPO, 'inference_propainter.py')).read())
