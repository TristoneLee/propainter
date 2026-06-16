# -*- coding: utf-8 -*-
"""Unified ProPainter end-to-end benchmark: optimized fork vs pristine original.

Drives the REAL inference entry points (not a stand-in bench harness), so the
numbers reflect the actual pipeline a user runs — GPU compositing, no stray
empty_cache, production defaults, and the full data_loading -> ... -> write_video
span.

  optimized = repo-root  inference_propainter.py   (PROPAINTER_FAST=1 + --fp16)
  original  = ProPainter/inference_propainter.py    (pristine model code; fp16)

Two numbers per (variant, shape) cell:
  * TOTAL (pure)  — zero added synchronize(); one perf_counter span from
                    data_loading start to output_write end. This is the headline,
                    most faithful end-to-end wall. (`--bench_json` default mode.)
  * Per-stage     — a SEPARATE diagnostic run with PROPAINTER_TIME_STAGES=1, which
                    adds ~6 CUDA syncs (so its own total is ~1-2% high). Shown as
                    reference detail only; never mixed into the pure TOTAL.

Methodology: 1 warmup (full pipeline incl. video read + write, discarded) +
N timed runs per cell, each a fresh subprocess (clean CUDA context / allocator /
L2). Report median + min/max. Pin one GPU via --gpu.

Run from repo root:
    python scripts/bench/run_benchmark.py --gpu 1
    python scripts/bench/run_benchmark.py --cells 720p80 --variants optimized --gpu 1
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# label -> (source sample under inputs/samples, width, height, frames)
CELLS = {
    '720p80':  ('1', 1280, 720, 80),
    '720p200': ('2', 1280, 720, 200),
    '540p200': ('3', 960, 540, 200),
}

# stage tags emitted by inference_propainter.py under PROPAINTER_TIME_STAGES=1
STAGES = ['data_loading', 'flow_estimation', 'flow_completion',
          'image_propagation', 'feat_prop_transformer', 'output_write']

SUBVIDEO = 80     # upstream ProPainter default, applied to both variants
NEIGHBOR = 10     # upstream default (transformer temporal receptive field)
WINDOW_STRIDE = 9 # fork default sliding-window step (optimized variant only)


def build_cmd(variant, label, json_out, out_dir, stage_mode):
    """argv, cwd, env for one real-inference run of a cell.

    stage_mode=False -> pure (zero-sync) end-to-end; True -> add per-stage timing.
    """
    sample, w, h, n = CELLS[label]
    env = dict(os.environ)
    chunk = ['--subvideo_length', str(SUBVIDEO), '--neighbor_length', str(NEIGHBOR)]
    if variant == 'optimized':
        env['PROPAINTER_FAST'] = '1'
        env['PROPAINTER_FLEX_ATTN'] = '0'
        env['PROPAINTER_NVTX'] = '0'
        env['PROPAINTER_TIME_STAGES'] = '1' if stage_mode else '0'
        src = os.path.join(_REPO, 'inputs', 'samples', sample)
        argv = [sys.executable, 'inference_propainter.py',
                '-i', os.path.join(src, 'input.mp4'),
                '-m', os.path.join(src, 'masks.mp4'),
                '--height', str(h), '--width', str(w), '--frames', str(n),
                '--fp16', *chunk, '--window_stride', str(WINDOW_STRIDE),
                '--output', out_dir, '--bench_json', json_out]
        return argv, _REPO, env
    elif variant == 'original':
        env.pop('PROPAINTER_FAST', None)
        env.pop('PROPAINTER_FLEX_ATTN', None)
        # pristine original only emits the pure stamp (no _tick instrumentation),
        # so stage_mode is a no-op for it — it always returns pure total only.
        src = os.path.join(_REPO, 'inputs', 'samples', sample)
        cwd = os.path.join(_REPO, 'ProPainter')
        argv = [sys.executable, 'inference_propainter.py',
                '-i', os.path.join(src, 'input.mp4'),
                '-m', os.path.join(src, 'masks.mp4'),
                '--height', str(h), '--width', str(w), '--frames', str(n),
                '--fp16', *chunk, '--output', out_dir, '--bench_json', json_out]
        return argv, cwd, env
    raise ValueError(variant)


def run_once(variant, label, gpu, tag, stage_mode):
    fd, json_out = tempfile.mkstemp(suffix='.json', prefix='ppbench_')
    os.close(fd)
    out_dir = tempfile.mkdtemp(prefix='ppout_')
    argv, cwd, env = build_cmd(variant, label, json_out, out_dir, stage_mode)
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    print(f'  [{tag}] {variant}/{label}{" +stages" if stage_mode else ""} ...', flush=True)
    r = subprocess.run(argv, cwd=cwd, env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    data = None
    if r.returncode != 0:
        out = r.stdout.decode(errors='replace')
        kind = 'OOM' if 'OutOfMemoryError' in out else 'FAIL'
        print(f'    {kind} (rc={r.returncode}):\n{out[-1200:]}', flush=True)
        data = {'__error__': kind}
    elif os.path.exists(json_out):
        with open(json_out) as fh:
            data = json.load(fh)
    if os.path.exists(json_out):
        os.remove(json_out)
    # clean the throwaway output videos
    try:
        import shutil
        shutil.rmtree(out_dir, ignore_errors=True)
    except Exception:
        pass
    return data


def run_cell(variant, label, gpu, n_timed):
    # --- warmup (full pipeline, discarded) ---
    warm = run_once(variant, label, gpu, 'warmup', stage_mode=False)
    if warm is None or '__error__' in warm:
        return {'variant': variant, 'label': label,
                'status': warm['__error__'] if warm else 'FAIL',
                'frames': CELLS[label][3]}

    # --- N timed PURE runs (headline) ---
    pure = []
    for i in range(n_timed):
        d = run_once(variant, label, gpu, f'pure {i + 1}/{n_timed}', stage_mode=False)
        if d and '__error__' not in d and d.get('total_pure_s'):
            pure.append(d)
    if not pure:
        return {'variant': variant, 'label': label, 'status': 'OOM',
                'frames': CELLS[label][3]}

    # --- 1 diagnostic STAGE run (both variants now carry stage tags) ---
    stages = None
    s = run_once(variant, label, gpu, 'stage-diag', stage_mode=True)
    if s and '__error__' not in s and s.get('stages_ms'):
        stages = s['stages_ms']

    def agg(key):
        vals = [d[key] for d in pure]
        return {'median': statistics.median(vals), 'min': min(vals), 'max': max(vals)}

    return {
        'variant': variant, 'label': label, 'status': 'ok',
        'frames': pure[0].get('frames'), 'w': pure[0].get('w'), 'h': pure[0].get('h'),
        'n_timed': len(pure),
        'warmup_pure_s': warm.get('total_pure_s'),
        'total_pure_s': agg('total_pure_s'),
        'throughput_fps': agg('throughput_fps'),
        'peak_gpu_gb': agg('peak_gpu_gb'),
        'stages_ms': stages,   # diagnostic, sync-inflated; optimized only
        'raw': pure,
    }


def write_report(results, path):
    ok = lambda x: x is not None and x.get('status') == 'ok'
    by = {}
    for r in results:
        by.setdefault(r['label'], {})[r['variant']] = r

    L = ['# ProPainter End-to-End Benchmark — Optimized Fork vs Pristine Original', '']
    L.append('Drives the **real** inference entry points (`inference_propainter.py` '
             'for the fork; `ProPainter/inference_propainter.py`, pristine model '
             'code, for the original). Protocol: 1 warmup (full pipeline incl. '
             'video read + write, discarded) + N timed runs per cell, fresh '
             'subprocess each. Chunking (both): `subvideo_length=80`, '
             '`neighbor_length=10` (upstream defaults). Optimized = '
             '`PROPAINTER_FAST=1` + fp16 + `window_stride=9`.')
    L.append('')
    L.append('**TOTAL (pure)** = zero-added-synchronize wall from data_loading '
             'start to output_write end (the full per-video pipeline; model load '
             'excluded). Reported **median (min–max)** in seconds. The per-stage '
             'table below is a SEPARATE diagnostic run with `PROPAINTER_TIME_STAGES=1` '
             '(adds ~6 CUDA syncs, so its own total runs ~1-2% high) — reference '
             'detail only, never mixed into the pure TOTAL. Stages are optimized-only '
             '(the pristine original has no timing instrumentation).')
    L.append('')
    L.append('## End-to-end total + speedup')
    L.append('')
    L.append('| Shape | Frames | Optimized (s) | Original (s) | Speedup | '
             'Opt fps | Opt peak GB | Orig peak GB |')
    L.append('|---|---|---|---|---|---|---|---|')
    for label in CELLS:
        c = by.get(label, {})
        o, b = c.get('optimized'), c.get('original')
        def cell(x):
            if ok(x):
                t = x['total_pure_s']
                return f"{t['median']:.2f} ({t['min']:.2f}–{t['max']:.2f})"
            return f"**{x.get('status', 'FAIL')}**" if x is not None else '-'
        ot = o['total_pure_s']['median'] if ok(o) else None
        bt = b['total_pure_s']['median'] if ok(b) else None
        spd = f'{bt / ot:.2f}×' if (ot and bt) else '-'
        fps = f"{o['throughput_fps']['median']:.2f}" if ok(o) else '-'
        opg = f"{o['peak_gpu_gb']['median']:.2f}" if ok(o) else '-'
        bpg = f"{b['peak_gpu_gb']['median']:.2f}" if ok(b) else '-'
        frames = (o or b or {}).get('frames', '-')
        L.append(f'| {label} | {frames} | {cell(o)} | {cell(b)} | {spd} | {fps} | {opg} | {bpg} |')
    L.append('')

    GPU_STAGES = ['flow_estimation', 'flow_completion', 'image_propagation',
                  'feat_prop_transformer']
    IO_STAGES = ['data_loading', 'output_write']

    # GPU-only summary (sum of the 4 GPU stages, excl. data_loading + output_write)
    L.append('## GPU-only time (sum of the 4 GPU stages, excl. data_loading + write)')
    L.append('')
    L.append('_From the stage-diagnostic run (CUDA-synced, ~1-2% high). Isolates '
             'pure GPU compute from host I/O._')
    L.append('')
    L.append('| Shape | Opt GPU-only (s) | Orig GPU-only (s) | Speedup | '
             'Opt I/O (s) | Orig I/O (s) |')
    L.append('|---|---|---|---|---|---|')
    for label in CELLS:
        c = by.get(label, {})
        def gpu_io(x):
            if not (ok(x) and x.get('stages_ms')):
                return None, None
            sm = x['stages_ms']
            return (sum(sm.get(k, 0) for k in GPU_STAGES) / 1000,
                    sum(sm.get(k, 0) for k in IO_STAGES) / 1000)
        og, oio = gpu_io(c.get('optimized'))
        bg, bio = gpu_io(c.get('original'))
        spd = f'{bg / og:.2f}×' if (og and bg) else '-'
        f = lambda v: f'{v:.2f}' if v is not None else '-'
        L.append(f'| {label} | {f(og)} | {f(bg)} | {spd} | {f(oio)} | {f(bio)} |')
    L.append('')

    L.append('## Per-stage diagnostic (ms — CUDA-synced, ~1-2% high)')
    L.append('')
    for label in CELLS:
        c = by.get(label, {})
        o, b = c.get('optimized'), c.get('original')
        osm = o.get('stages_ms') if ok(o) else None
        bsm = b.get('stages_ms') if ok(b) else None
        if not osm and not bsm:
            continue
        L.append(f'### {label}')
        L.append('')
        L.append('| Stage | Optimized ms | Original ms |')
        L.append('|---|---|---|')
        for st in STAGES:
            ov = f'{osm[st]:.1f}' if (osm and st in osm) else '-'
            bv = f'{bsm[st]:.1f}' if (bsm and st in bsm) else '-'
            L.append(f'| {st} | {ov} | {bv} |')
        og = sum(osm.get(k, 0) for k in GPU_STAGES) if osm else None
        bg = sum(bsm.get(k, 0) for k in GPU_STAGES) if bsm else None
        L.append(f'| **GPU-only sum** | **{og:.1f}** | **{bg:.1f}** |'
                 if (og and bg) else
                 f'| **GPU-only sum** | **{og:.1f}** | - |' if og else
                 f'| **GPU-only sum** | - | **{bg:.1f}** |' if bg else
                 '| **GPU-only sum** | - | - |')
        if ok(o):
            wp, t0 = o['warmup_pure_s'], o['raw'][0]['total_pure_s']
            L.append('')
            L.append(f'_opt warmup pure {wp:.2f}s vs first timed pure {t0:.2f}s '
                     f'(warmup ≥ timed confirms cold-start absorbed)_')
        L.append('')

    with open(path, 'w') as fh:
        fh.write('\n'.join(L) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variants', nargs='+', default=['optimized', 'original'],
                    choices=['optimized', 'original'])
    ap.add_argument('--cells', nargs='+', default=list(CELLS), choices=list(CELLS))
    ap.add_argument('--timed', type=int, default=3)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--out-json', default=os.path.join(_REPO, 'results', 'bench', 'benchmark.json'))
    ap.add_argument('--out-md', default=os.path.join(_REPO, 'results', 'report', 'BENCHMARK.md'))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_md), exist_ok=True)

    # merge with prior results: keep cells not re-run this invocation
    results = []
    if os.path.exists(args.out_json):
        with open(args.out_json) as fh:
            prior = json.load(fh)
        rerun = {(v, c) for c in args.cells for v in args.variants}
        results = [r for r in prior if (r['variant'], r['label']) not in rerun]

    for label in args.cells:
        for variant in args.variants:
            print(f'== {variant} / {label} (GPU {args.gpu}) ==', flush=True)
            agg = run_cell(variant, label, args.gpu, args.timed)
            results.append(agg)
            if agg.get('status') == 'ok':
                print(f'   median pure total = {agg["total_pure_s"]["median"]:.2f}s', flush=True)
            else:
                print(f'   {agg.get("status", "FAIL")} (recorded)', flush=True)
            with open(args.out_json, 'w') as fh:
                json.dump(results, fh, indent=2)
            write_report(results, args.out_md)

    print(f'\nWrote {args.out_json}\nWrote {args.out_md}')


if __name__ == '__main__':
    main()
