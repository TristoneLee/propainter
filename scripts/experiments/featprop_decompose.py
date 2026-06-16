#!/usr/bin/env python3
"""Launch-bound vs compute-bound decomposition of an NVTX range.

For a target range (default model/feat_prop) reports, summed over all instances:
  - wall (sum of range durations) and GPU-busy time + busy%
  - kernel category split (conv/gemm, grid_sample, deform_conv, elementwise, copy, other)
  - CUDA runtime API call counts (cudaMalloc/cudaFree/cudaLaunchKernel/etc)
  - wall time of nested fp/* sub-ranges (conv_offset / deform_conv2d / backbone / warp_check / fuse)

Usage:
  python scripts/experiments/featprop_decompose.py <export.sqlite> [range_text]
"""
import sqlite3
import sys
from collections import defaultdict

SQLITE = sys.argv[1]
RANGE = sys.argv[2] if len(sys.argv) > 2 else 'model/feat_prop'


def classify(name):
    n = name.lower()
    if 'deform' in n or 'modulated' in n:
        return 'deform_conv2d'
    if 'grid_sampler' in n or 'gridsampler' in n or 'grid_sample' in n:
        return 'grid_sample (flow_warp)'
    if any(k in n for k in ('conv', 'cudnn', 'gemm', 'cutlass', 'wgrad', 'implicit',
                            'sgemm', 'hgemm', 'volta', 'ampere', 'turing', 'xmma',
                            'cask', 'winograd')):
        return 'conv/gemm'
    if 'upsample' in n or 'interpolat' in n:
        return 'interpolate'
    if 'copy' in n or 'memcpy' in n:
        return 'copy/reshape'
    if any(k in n for k in ('elementwise', 'functor', 'binary', 'unary', 'compare',
                            'where', 'fill', 'mul', 'add', 'sub', 'div', 'pow', 'abs',
                            'exp', 'sum', 'reduce', 'clamp', 'sigmoid', 'tanh', 'relu',
                            'leaky', 'cat', 'index', 'permute', 'stack', 'chunk', 'flip',
                            'tensor', 'arange', 'meshgrid')):
        return 'elementwise/reduce/index'
    return 'other'


con = sqlite3.connect(SQLITE)
cur = con.cursor()

rows = cur.execute(
    "SELECT start, end FROM NVTX_EVENTS WHERE text = ? AND end IS NOT NULL ORDER BY start",
    (RANGE,)).fetchall()
if not rows:
    rows = cur.execute(
        "SELECT start, end FROM NVTX_EVENTS WHERE text LIKE ? AND end IS NOT NULL ORDER BY start",
        ('%' + RANGE + '%',)).fetchall()
if not rows:
    print(f'[decompose] no NVTX range matching "{RANGE}"')
    sys.exit(1)

total_wall_ms = sum((e - s) for s, e in rows) / 1e6
print(f'[decompose] range="{RANGE}"  instances={len(rows)}  wall={total_wall_ms:.2f} ms')

preds = ' OR '.join(['(k.start >= ? AND k.start < ?)'] * len(rows))
params = []
for s, e in rows:
    params += [s, e]

# --- kernels ---
q = f"""
SELECT s.value AS name, COUNT(*) AS n, SUM(k.end - k.start) AS dur_ns
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN StringIds s ON s.id = k.shortName
WHERE {preds}
GROUP BY s.value
ORDER BY dur_ns DESC
"""
krows = cur.execute(q, params).fetchall()
total_gpu_ms = sum(r[2] for r in krows) / 1e6 if krows else 0.0
bucket_ms = defaultdict(float)
bucket_n = defaultdict(int)
for name, n, dur in krows:
    b = classify(name)
    bucket_ms[b] += dur / 1e6
    bucket_n[b] += n

busy = 100 * total_gpu_ms / total_wall_ms if total_wall_ms else 0
print(f'\n[decompose] GPU-busy = {total_gpu_ms:.2f} ms  ({busy:.0f}% of wall; '
      f'idle/launch-gap = {total_wall_ms-total_gpu_ms:.2f} ms = {100-busy:.0f}%)\n')

print('=== kernel category split ===')
print(f'{"category":<28}{"GPU ms":>10}{"% gpu":>8}{"launches":>10}')
for b in sorted(bucket_ms, key=bucket_ms.get, reverse=True):
    print(f'{b:<28}{bucket_ms[b]:>10.2f}{100*bucket_ms[b]/total_gpu_ms:>7.1f}%{bucket_n[b]:>10}')
total_launches = sum(bucket_n.values())
print(f'{"TOTAL":<28}{total_gpu_ms:>10.2f}{100.0:>7.1f}%{total_launches:>10}')

print('\n=== top 18 kernels ===')
print(f'{"kernel":<58}{"GPU ms":>9}{"%":>6}{"calls":>8}')
for name, n, dur in krows[:18]:
    ms = dur / 1e6
    short = name if len(name) <= 56 else name[:53] + '...'
    print(f'{short:<58}{ms:>9.2f}{100*ms/total_gpu_ms:>5.0f}%{n:>8}')

# --- runtime API counts in window ---
print('\n=== CUDA runtime API calls in window ===')
q2 = f"""
SELECT s.value AS name, COUNT(*) AS n, SUM(r.end - r.start) AS dur_ns
FROM CUPTI_ACTIVITY_KIND_RUNTIME r
JOIN StringIds s ON s.id = r.nameId
WHERE {preds.replace('k.', 'r.')}
GROUP BY s.value
ORDER BY n DESC
"""
try:
    arows = cur.execute(q2, params).fetchall()
    print(f'{"api":<34}{"calls":>10}{"total ms":>12}')
    for name, n, dur in arows:
        print(f'{name:<34}{n:>10}{(dur or 0)/1e6:>12.2f}')
except Exception as e:
    print('runtime api query failed:', e)

# --- nested fp/* sub-ranges (wall) ---
print('\n=== nested fp/* sub-range wall (summed over instances) ===')
sub = cur.execute(
    "SELECT text, COUNT(*) n, SUM(end-start) dur FROM NVTX_EVENTS "
    "WHERE text LIKE 'fp/%' AND end IS NOT NULL GROUP BY text ORDER BY dur DESC").fetchall()
if sub:
    print(f'{"sub-range":<22}{"instances":>11}{"wall ms":>10}{"% featprop":>12}')
    for text, n, dur in sub:
        ms = (dur or 0) / 1e6
        print(f'{text:<22}{n:>11}{ms:>10.2f}{100*ms/total_wall_ms:>11.1f}%')
else:
    print('(no fp/* sub-ranges found)')

con.close()
