#!/usr/bin/env python3
"""Per-kernel breakdown of GPU work *inside* a given NVTX range.

The top-level analyze_nsys_metrics.py renders the pp/->model/->tst/ hierarchy but
stops at the range level. For ranges with no sub-NVTX markers (e.g.
`pp/image_propagation`, whose BidirectionalPropagation.forward has no inner
ranges) we need to look at which CUDA kernels actually ran in that window and
bucket them (grid_sample / elementwise / copy / conv / other).

Usage:
  python scripts/experiments/breakdown_nvtx_kernels.py <export.sqlite> [range_text]
  (range_text default: pp/image_propagation)
"""
import sqlite3
import sys
import re
from collections import defaultdict

SQLITE = sys.argv[1]
RANGE = sys.argv[2] if len(sys.argv) > 2 else 'pp/image_propagation'


def classify(name):
    n = name.lower()
    if 'grid_sampler' in n or 'gridsampler' in n or 'grid_sample' in n:
        return 'grid_sample (flow_warp)'
    if 'upsample' in n or 'interpolat' in n:
        return 'interpolate'
    if any(k in n for k in ('elementwise', 'vectorized_elementwise', 'unrolled_elementwise',
                            'functor', 'binary', 'unary', 'compare', 'where', 'fill',
                            'mul', 'add', 'sub', 'div', 'pow', 'abs', 'exp', 'sum',
                            'reduce', 'clamp', 'sigmoid', 'tanh', 'relu', 'cat', 'index')):
        return 'elementwise/reduce/index'
    if 'copy' in n or 'direct_copy' in n or 'memcpy' in n:
        return 'copy/reshape'
    if any(k in n for k in ('conv', 'cudnn', 'gemm', 'cutlass', 'wgrad', 'implicit')):
        return 'conv/gemm'
    return 'other'


con = sqlite3.connect(SQLITE)
cur = con.cursor()

# 1) range windows
rows = cur.execute(
    "SELECT start, end FROM NVTX_EVENTS WHERE text = ? AND end IS NOT NULL ORDER BY start",
    (RANGE,)).fetchall()
if not rows:
    # try LIKE in case of prefix differences
    rows = cur.execute(
        "SELECT start, end FROM NVTX_EVENTS WHERE text LIKE ? AND end IS NOT NULL ORDER BY start",
        ('%' + RANGE + '%',)).fetchall()
if not rows:
    print(f'[breakdown] no NVTX range matching "{RANGE}" found')
    sys.exit(1)

win_lo = min(r[0] for r in rows)
win_hi = max(r[1] for r in rows)
total_wall_ms = sum((e - s) for s, e in rows) / 1e6
print(f'[breakdown] range="{RANGE}"  instances={len(rows)}  wall={total_wall_ms:.2f} ms '
      f'(window {win_lo}..{win_hi})')

# 2) kernels within the union of range windows. Build an OR of time predicates.
preds = ' OR '.join(['(k.start >= ? AND k.start < ?)'] * len(rows))
params = []
for s, e in rows:
    params += [s, e]

# kernel name: prefer shortName -> StringIds
q = f"""
SELECT s.value AS name, COUNT(*) AS n, SUM(k.end - k.start) AS dur_ns
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN StringIds s ON s.id = k.shortName
WHERE {preds}
GROUP BY s.value
ORDER BY dur_ns DESC
"""
krows = cur.execute(q, params).fetchall()

if not krows:
    print('[breakdown] no kernels found in range window (check table names)')
    sys.exit(1)

total_gpu_ms = sum(r[2] for r in krows) / 1e6
bucket_ms = defaultdict(float)
bucket_n = defaultdict(int)
for name, n, dur in krows:
    b = classify(name)
    bucket_ms[b] += dur / 1e6
    bucket_n[b] += n

print(f'\n[breakdown] total GPU-kernel time in range = {total_gpu_ms:.2f} ms '
      f'(busy {100*total_gpu_ms/total_wall_ms:.0f}% of wall; rest = launch/host gaps)\n')

print('=== by category ===')
print(f'{"category":<28}{"GPU ms":>10}{"% gpu":>8}{"launches":>10}')
for b in sorted(bucket_ms, key=bucket_ms.get, reverse=True):
    print(f'{b:<28}{bucket_ms[b]:>10.2f}{100*bucket_ms[b]/total_gpu_ms:>7.1f}%{bucket_n[b]:>10}')

print('\n=== top 15 kernels ===')
print(f'{"kernel":<60}{"GPU ms":>9}{"%":>6}{"calls":>8}')
for name, n, dur in krows[:15]:
    ms = dur / 1e6
    short = name if len(name) <= 58 else name[:55] + '...'
    print(f'{short:<60}{ms:>9.2f}{100*ms/total_gpu_ms:>5.0f}%{n:>8}')

con.close()
