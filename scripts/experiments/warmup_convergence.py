# -*- coding: utf-8 -*-
"""Experiment: how many warmup (fresh-subprocess) runs until end-to-end time converges?

The benchmark (scripts/bench/run_benchmark.py) does 1 warmup + N timed, each a
FRESH subprocess (clean CUDA context / allocator / L2). So an in-process JIT/cache
warmup is N/A — the only thing a discarded run could smooth is machine-level cold
state: page cache for weights/input video, GPU clock ramp, driver/context warmth.

This script runs the SAME cell K times back-to-back (fresh subprocess each, exactly
like the bench) and dumps every run's total_pure_s, so we can see at which index the
series stabilizes (i.e. how many leading runs should be discarded as warmup).

Usage:
    python scripts/experiments/warmup_convergence.py --gpu 1 --cell 720p80 --runs 12
"""
import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.run_benchmark import run_once, CELLS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=1)
    ap.add_argument('--cell', default='720p80', choices=list(CELLS))
    ap.add_argument('--variant', default='optimized', choices=['optimized', 'original'])
    ap.add_argument('--runs', type=int, default=12)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    print(f'== warmup-convergence: {args.variant}/{args.cell} x{args.runs} (GPU {args.gpu}) ==',
          flush=True)
    series = []
    for i in range(args.runs):
        d = run_once(args.variant, args.cell, args.gpu, f'run {i + 1}/{args.runs}',
                     stage_mode=False)
        t = d.get('total_pure_s') if (d and '__error__' not in d) else None
        series.append(t)
        print(f'    run {i + 1:2d}: total_pure_s = {t:.3f}s' if t else
              f'    run {i + 1:2d}: FAIL', flush=True)

    vals = [t for t in series if t]
    print('\n--- summary ---', flush=True)
    for i, t in enumerate(series):
        delta = ''
        if i > 0 and t and series[i - 1]:
            delta = f'  ({(t - series[i - 1]) / series[i - 1] * 100:+.1f}% vs prev)'
        print(f'  run {i + 1:2d}: {t:.3f}s{delta}' if t else f'  run {i + 1:2d}: FAIL',
              flush=True)
    if len(vals) > 1:
        print(f'\n  overall: min={min(vals):.3f} max={max(vals):.3f} '
              f'median={statistics.median(vals):.3f} '
              f'stdev={statistics.pstdev(vals):.3f} '
              f'spread={(max(vals) - min(vals)) / min(vals) * 100:.1f}%', flush=True)
        # convergence: median of tail after discarding k leading runs
        print('\n  tail stats (discard first k runs):', flush=True)
        for k in range(0, min(5, len(vals))):
            tail = vals[k:]
            if len(tail) >= 2:
                print(f'    k={k}: n={len(tail)} median={statistics.median(tail):.3f} '
                      f'stdev={statistics.pstdev(tail):.3f} '
                      f'spread={(max(tail) - min(tail)) / min(tail) * 100:.1f}%', flush=True)

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   f'warmup_conv_{args.variant}_{args.cell}.json')
    with open(out, 'w') as fh:
        json.dump({'variant': args.variant, 'cell': args.cell, 'gpu': args.gpu,
                   'series': series}, fh, indent=2)
    print(f'\nWrote {out}', flush=True)


if __name__ == '__main__':
    main()
