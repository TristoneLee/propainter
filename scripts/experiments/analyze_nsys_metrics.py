#!/usr/bin/env python
"""Unified ProPainter profile report from a single nsys sqlite.

Renders the NVTX hierarchy as a top-down bottleneck view — one capture, one report:

    pp/*     pipeline stages          (inference_propainter.py)
    model/*  InpaintGenerator.forward (model/propainter.py)
    tst/*    transformer segments     (model/modules/sparse_transformer.py)

Per range it reports: summed wall-ms (over all calls), % of total run, % of its
parent range, peak GPU memory (GB, from the --cuda-memory-usage trace), and
time-averaged GPU counters (SMs/Tensor/DRAM active). Reading the tree top-down,
the largest %-of-parent at each level traces the bottleneck path.

Usage:
    python analyze_nsys_metrics.py results/nsys/<run>.sqlite --out results/report/PROFILE.md
"""
import argparse
import bisect
import sqlite3
import sys
from collections import defaultdict

METRIC_IDS = {7: 'SMs%', 9: 'Tensor%', 19: 'DRAM_Rd%', 20: 'DRAM_Wr%'}

# (tag, depth, parent). parent=None => top-level stage (child of the whole run).
# Order is the render order; tags absent from the capture are skipped.
HIERARCHY = [
    ('pp/data_loading',            0, None),
    ('pp/flow_estimation',         0, None),
    ('pp/flow_completion',         0, None),
    ('pp/image_propagation',       0, None),
    ('pp/feat_prop_transformer',   0, None),
    ('pp/fpt/indexing',            1, 'pp/feat_prop_transformer'),
    ('pp/fpt/model_fwd',           1, 'pp/feat_prop_transformer'),
    ('model/encoder',              2, 'pp/fpt/model_fwd'),
    ('model/feat_prop',            2, 'pp/fpt/model_fwd'),
    ('model/softsplit',            2, 'pp/fpt/model_fwd'),
    ('model/transformer',          2, 'pp/fpt/model_fwd'),
    ('tst/block',                  3, 'model/transformer'),
    ('tst/norm1',                  4, 'tst/block'),
    ('tst/attn/qkv_proj',          4, 'tst/block'),
    ('tst/attn/win_partition_qkv', 4, 'tst/block'),
    ('tst/attn/roll_expand',       4, 'tst/block'),
    ('tst/attn/pool_kv',           4, 'tst/block'),
    ('tst/attn/mask_prep',         4, 'tst/block'),
    ('tst/attn/sdpa_loop',         4, 'tst/block'),
    ('tst/attn/unshuffle_proj',    4, 'tst/block'),
    ('tst/norm2',                  4, 'tst/block'),
    ('tst/ffn/fc1',                4, 'tst/block'),
    ('tst/ffn/fold_unfold',        4, 'tst/block'),
    ('tst/ffn/fc2',                4, 'tst/block'),
    ('model/softcomp',             2, 'pp/fpt/model_fwd'),
    ('model/decoder',              2, 'pp/fpt/model_fwd'),
    ('pp/fpt/d2h',                 1, 'pp/feat_prop_transformer'),
    ('pp/fpt/cpu_post',            1, 'pp/feat_prop_transformer'),
]


def load_nvtx(cur):
    """All pp/* model/* tst/* ranges -> {tag: [(start,end), ...]}."""
    rows = cur.execute("""
        SELECT text, start, end FROM NVTX_EVENTS
         WHERE end IS NOT NULL
           AND (text LIKE 'pp/%' OR text LIKE 'model/%' OR text LIKE 'tst/%')
    """).fetchall()
    by_tag = defaultdict(list)
    for txt, s, e in rows:
        if e > s:
            by_tag[txt].append((s, e))
    return by_tag


def build_mem_timeline(con):
    """Reconstruct cumulative GPU bytes-allocated over time from the nsys
    --cuda-memory-usage trace. Returns (sorted_ts, cumulative_bytes) arrays, or
    None if the capture has no memory trace. Schema varies by nsys version, so
    detect the table/columns defensively."""
    cur = con.cursor()
    tbl = None
    for cand in ('CUDA_GPU_MEMORY_USAGE_EVENTS',):
        try:
            cur.execute(f'SELECT 1 FROM {cand} LIMIT 1')
            tbl = cand
            break
        except sqlite3.OperationalError:
            continue
    if tbl is None:
        return None
    cols = {r[1] for r in cur.execute(f'PRAGMA table_info({tbl})')}
    tcol = 'start' if 'start' in cols else ('timestamp' if 'timestamp' in cols else None)
    if tcol is None or 'bytes' not in cols:
        return None
    # memoryOperationType: 0 = allocation, 1 = deallocation (nsys convention)
    opcol = 'memoryOperationType' if 'memoryOperationType' in cols else None
    rows = cur.execute(f'SELECT {tcol}, bytes' + (f', {opcol}' if opcol else '') +
                       f' FROM {tbl} ORDER BY {tcol}').fetchall()
    if not rows:
        return None
    ts, cum = [], []
    running = 0
    for r in rows:
        t, b = r[0], r[1]
        op = r[2] if opcol else 0
        running += b if op == 0 else -b
        ts.append(t)
        cum.append(running)
    return ts, cum


def peak_mem_gb(timeline, s, e):
    """Max cumulative allocated bytes within [s, e] -> GB."""
    if timeline is None:
        return None
    ts, cum = timeline
    lo = bisect.bisect_left(ts, s)
    hi = bisect.bisect_right(ts, e)
    window = cum[lo:hi]
    base = cum[lo - 1] if lo > 0 else 0  # level entering the range
    peak = max(window) if window else base
    return max(peak, base) / 1024**3


def choose_gpu_typeid(cur):
    """The active GPU's typeId = the one with the most GPU_METRICS samples.
    (Don't gate on a probe window — CPU-bound stages like data_loading have
    SMs Active = 0 and would falsely reject the GPU.)"""
    row = cur.execute("""SELECT typeId, COUNT(*) c FROM GPU_METRICS
                          GROUP BY typeId ORDER BY c DESC LIMIT 1""").fetchone()
    return row[0] if row else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sqlite_path')
    ap.add_argument('--out', default=None, help='Write the markdown report here.')
    args = ap.parse_args()

    con = sqlite3.connect(args.sqlite_path)
    cur = con.cursor()

    by_tag = load_nvtx(cur)
    if not by_tag:
        print('[analyze] no pp/ model/ tst/ NVTX ranges found — was PROPAINTER_NVTX=1 set '
              'and --trace=nvtx passed to nsys?', file=sys.stderr)
        sys.exit(1)
    print(f'[analyze] {sum(len(v) for v in by_tag.values())} NVTX ranges, '
          f'{len(by_tag)} distinct tags')

    # per-tag summed wall time
    wall_ns = {tag: sum(e - s for s, e in spans) for tag, spans in by_tag.items()}

    # GPU hardware counters (optional — needs --gpu-metrics-devices AND admin
    # profiling permission; SM/Tensor counters read as 0 for non-admin users).
    metrics = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))  # tag -> mid -> [sum,n]
    has_gpu_metrics = False
    metrics_note = ''
    try:
        tid = choose_gpu_typeid(cur)
        mids = ','.join(str(m) for m in METRIC_IDS)
        # Gate on the compute counters (SMs=7, Tensor=9): restricted (non-admin)
        # profiling reads these as exactly 0, which is the signal that the whole
        # HW-counter capture is unusable here.
        gmax = cur.execute("""SELECT MAX(value) FROM GPU_METRICS
                               WHERE typeId=? AND metricId IN (7,9)""",
                           (tid,)).fetchone()[0] if tid is not None else None
        if tid is not None and gmax and gmax > 0:
            has_gpu_metrics = True
            for tag, spans in by_tag.items():
                for s, e in spans:
                    for mid, val in cur.execute(
                            f"""SELECT metricId, value FROM GPU_METRICS
                                 WHERE typeId=? AND timestamp BETWEEN ? AND ?
                                   AND metricId IN ({mids})""", (tid, s, e)):
                        metrics[tag][mid][0] += val
                        metrics[tag][mid][1] += 1
        else:
            metrics_note = ('GPU HW counters unavailable (SM/Tensor read 0 — needs '
                            'admin profiling permission); wall-time + memory only.')
    except sqlite3.OperationalError:
        metrics_note = 'GPU metrics tables absent in capture.'

    # GPU memory timeline (optional — needs --cuda-memory-usage)
    timeline = build_mem_timeline(con)
    peak = {}
    if timeline is not None:
        for tag, spans in by_tag.items():
            peak[tag] = max((peak_mem_gb(timeline, s, e) for s, e in spans), default=0.0)

    total_ns = sum(wall_ns.get(t, 0) for t, d, p in HIERARCHY if d == 0)
    total_ns = total_ns or 1

    # ---- render ----
    mcols = list(METRIC_IDS.values()) if has_gpu_metrics else []
    head = ['Stage', 'Calls', 'Wall(ms)', '%tot', '%par']
    if timeline is not None:
        head.append('Peak(GB)')
    head += mcols
    md = ['| ' + ' | '.join(head) + ' |',
          '|' + '|'.join(['---'] + ['---:'] * (len(head) - 1)) + '|']
    print('  '.join(f'{h:>10}' if i else f'{h:<34}' for i, h in enumerate(head)))

    seen = set()
    rows_to_show = [r for r in HIERARCHY if r[0] in by_tag]
    rows_to_show += [(t, 0, None) for t in sorted(by_tag) if t not in {r[0] for r in HIERARCHY}]
    for tag, depth, parent in rows_to_show:
        seen.add(tag)
        w = wall_ns.get(tag, 0)
        pct_tot = 100.0 * w / total_ns
        pct_par = 100.0 * w / wall_ns[parent] if parent and wall_ns.get(parent) else float('nan')
        label = ('  ' * depth) + tag.split('/')[-1]
        cells = [label, str(len(by_tag[tag])), f'{w/1e6:.2f}',
                 f'{pct_tot:.1f}', ('' if parent is None else f'{pct_par:.1f}')]
        if timeline is not None:
            cells.append(f'{peak.get(tag, 0):.2f}')
        for mid in METRIC_IDS:
            if has_gpu_metrics:
                s, n = metrics[tag][mid]
                cells.append(f'{(s/n):.1f}' if n else '-')
        md.append('| ' + ' | '.join(cells) + ' |')
        print(f'{label:<34}' + '  '.join(f'{c:>10}' for c in cells[1:]))

    if args.out:
        with open(args.out, 'w') as f:
            f.write(f'# ProPainter profile — `{args.sqlite_path}`\n\n')
            f.write(f'Total accounted (sum of top-level `pp/*` stages): '
                    f'**{total_ns/1e6:.1f} ms**. Columns: summed wall time over all '
                    f'calls, % of total, % of parent range, peak GPU mem, '
                    f'time-averaged GPU counters.\n\n')
            if metrics_note:
                f.write(f'> Note: {metrics_note}\n\n')
            f.write('\n'.join(md) + '\n')
        print(f'[analyze] wrote {args.out}')


if __name__ == '__main__':
    main()
