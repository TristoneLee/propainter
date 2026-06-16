# ProPainter Benchmarking & Profiling Guide

Two distinct tools, used for two different questions:

1. **Latency benchmarking** (this section) — *"how fast is the optimized fork vs
   the pristine original, end-to-end, at each shape?"* A reproducible
   wall-clock harness with warmup, repeats, and variance.
2. **nsys profiling** (further below) — *"where inside one run does the time
   go?"* A single NVTX-instrumented run turned into a hierarchical kernel/stage
   bottleneck report.

---

# Part 1 — Latency Benchmarking

## What it measures

It drives the **real inference entry points** (not a stand-in harness), so the
numbers reflect the actual pipeline a user runs — GPU-resident compositing, no
stray `empty_cache`, production defaults, and the full `data_loading → … →
output_write` span:

- **optimized** = repo-root `inference_propainter.py` with `PROPAINTER_FAST=1`
  (fused CUDA transformer kernels) + `--fp16` + SEA_RAFT fp16 flow.
- **original** = `ProPainter/inference_propainter.py`, **pristine model code**
  (only the CLI wrapper gains `--frames`/`--bench_json`), fp16.

Shapes (3): 720p80 (sample 1), 720p200 (sample 2), 540p200 (sample 3). 1080p was
dropped — the original's fp32 RAFT OOMs at full HD and the fork only fits 1080p80
with a reduced flow clip, so it isn't a clean apples-to-apples cell on 32 GB.

## Two numbers per cell

- **TOTAL (pure)** — the headline. A **zero-added-`synchronize()`** `perf_counter`
  span from `data_loading` start to `output_write` end (full per-video pipeline;
  model load excluded). No stage instrumentation, so nothing serializes the
  CPU/GPU overlap — this is the most faithful end-to-end wall.
- **Per-stage diagnostic** — a *separate* run with `PROPAINTER_TIME_STAGES=1`,
  which inserts ~6 CUDA syncs at stage boundaries (so its own total runs ~1–2%
  high). Shown as reference detail only, **never mixed into the pure TOTAL**.
  Optimized-only (the pristine original has no timing hooks). Includes the
  `data_loading` sub-stages (read_video / resize / read_mask / to_tensor) and
  `output_write`.

## Methodology — why each piece matters

- **Warmup (1 untimed run per cell, discarded).** The first run pays one-time
  costs that have nothing to do with steady-state latency: cuDNN/cuBLAS algorithm
  selection, CUDA context creation, first-touch `cudaMalloc`, weight H2D copy, and
  the OS file cache / video decoder warming. The warmup runs the **whole pipeline
  including video read + write**, so the I/O path is warm too. Discarding it
  removes cold-start bias.
- **Median of 3 timed runs + min/max.** Even after warmup, the RTX 5090 clocks
  drift with temperature/power (sustained runs throttle) and the OS adds jitter, so
  a single number can be off 5–15%. The median resists spikes; the min/max spread
  **exposes throttling**.
- **Fresh subprocess per run.** Each run is its own process → clean CUDA context /
  allocator / L2, statistically independent.
- **Single pinned GPU.** `--gpu` pins one idle GPU. Never set
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` (inflates ProPainter ~25×); never
  run a side probe on the same GPU (steals VRAM → spurious OOM).

## Timing scope

`TOTAL (pure)` covers, in order: data_loading (video decode + resize + mask read +
to-tensor + H2D) → flow_estimation → flow_completion → image_propagation →
feat_prop_transformer (**including** the per-frame compositing + final D2H) →
output_write (encode + write mp4). It **excludes** model load + weight download
(one-time, not per-video latency). Boundary note: the pristine original always
also writes a `masked_in.mp4` preview (its native behavior) and defers its
`.half()` to after RAFT — both kept as-is.

## How to run

```bash
# Full matrix on an idle GPU (1 warmup + 3 pure timed + 1 stage-diag per cell).
python scripts/bench/run_benchmark.py --gpu 1

# subsets while iterating:
python scripts/bench/run_benchmark.py --cells 720p80 --variants optimized --gpu 1
python scripts/bench/run_benchmark.py --timed 5 --gpu 1
```

Outputs (written incrementally, so a late failure keeps earlier cells):
- `results/report/BENCHMARK.md` — pure TOTAL + speedup + per-stage diagnostic.
- `results/bench/benchmark.json` — aggregates incl. every raw timed run.

## Reading the report

- **Speedup** = original pure median ÷ optimized pure median.
- **min–max** on the pure TOTAL is the throttling/jitter check: tight = trustworthy
  median; wide = re-run on a cooler/idle GPU.
- Each cell prints **warmup vs first-timed pure** — warmup should be ≥ timed,
  confirming cold-start was absorbed.
- Resize caveat: `resize_frames()` snaps to a multiple of 8; the fork uses cv2
  bicubic vs the original's PIL bicubic — outputs match to >40 dB PSNR, not
  bit-exact vs upstream.

## Files

- `scripts/bench/run_benchmark.py` — orchestrates warmup/repeats, pure + stage
  runs, aggregation, report.
- `inference_propainter.py` — fork CLI; `--frames`, `--bench_json` (pure default;
  `PROPAINTER_TIME_STAGES=1` → stage mode).
- `ProPainter/inference_propainter.py` — original CLI; model code pristine, wrapper
  gains `--frames` + `--bench_json` (pure only).
- `bench_fork.py` / `ProPainter/bench_original.py` — older stand-alone stage-timed
  benches (compute-only, no I/O); superseded by the real-CLI path above but kept.

---

# Part 2 — nsys Profiling (per-run bottleneck breakdown)

All profiling goes through **one nsys-driven path**. A single
NVTX-instrumented inference run is captured by Nsight Systems and turned into one
hierarchical bottleneck report — no separate benchmark harness, no perf_counter
timers, no torch.profiler pass.

## Quick start

```bash
# capture -> export sqlite -> analyze, in one command. Pick a free GPU.
CUDA_VISIBLE_DEVICES=1 scripts/profile_nsys.sh myrun -- \
    -i inputs/object_removal/bmx-trees -m inputs/object_removal/bmx-trees_mask \
    --fp16 --height 720 --width 1280 --frames 80
```

Produces:
- `results/nsys/myrun.nsys-rep` — open in the Nsight Systems GUI for the timeline
- `results/nsys/myrun.sqlite`   — exported trace
- `results/report/PROFILE.md`   — the unified hierarchical report

## What you get: a top-down bottleneck tree

The instrumentation marks the whole process as nested NVTX ranges, so the report
shows where time goes at every level:

```
pp/*      pipeline stages (data_loading, flow_estimation, flow_completion,
          image_propagation, feat_prop_transformer + its fpt/ sub-phases)
model/*   InpaintGenerator.forward (encoder, feat_prop, softsplit, transformer,
          softcomp, decoder)
tst/*     transformer segments (norm1, attn/*, norm2, ffn/*)  [eager path only]
```

Each row reports: **summed wall-ms** (over all calls), **% of total**, **% of
parent**, **peak GPU memory (GB)**, and (when available) **SMs/Tensor/DRAM**
counters. Read it top-down — the biggest `%par` at each level traces the
bottleneck path (e.g. `feat_prop_transformer → model_fwd → model/transformer`).

## Notes / requirements

- **`tst/*` drill-down needs the eager path.** The default fast CUDA kernels
  (`PROPAINTER_FAST=1`) replace the transformer block wholesale, so `model/transformer`
  shows as one range. To see the per-segment `tst/*` breakdown, capture with
  `PROPAINTER_FAST=0` (the launcher passes the env through):
  ```bash
  CUDA_VISIBLE_DEVICES=1 PROPAINTER_FAST=0 scripts/profile_nsys.sh eager -- <args>
  ```
- **GPU hardware counters (SMs/Tensor/DRAM) require admin profiling permission.**
  Without it (`NVreg_RestrictProfilingToAdminUsers=0` not set / non-root), nsys
  reads them as 0; the analyzer detects this and prints wall-time + memory only.
- **Instrumentation overhead is zero when not profiling.** All NVTX ranges are
  gated by `PROPAINTER_NVTX=1` (set only by the launcher); a normal
  `python inference_propainter.py ...` run is unaffected.
- Re-analyze an existing capture without re-running:
  `python scripts/experiments/analyze_nsys_metrics.py results/nsys/myrun.sqlite --out results/report/PROFILE.md`

## Implementation

- `model/nvtx.py` — the single `nvtx(tag)` primitive (gated, no-op off).
- `inference_propainter.py` — `pp/*` stage + `pp/fpt/*` sub-phase ranges.
- `model/propainter.py` — `model/*` ranges in `InpaintGenerator.forward`.
- `model/modules/sparse_transformer.py` — `tst/*` segment ranges.
- `scripts/profile_nsys.sh` — capture→export→analyze launcher.
- `scripts/experiments/analyze_nsys_metrics.py` — sqlite → hierarchical `PROFILE.md`.
