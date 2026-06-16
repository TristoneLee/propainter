# Experiment scripts

Run all from the repo root so the relative `inputs/` / `results/` / `weights/`
paths resolve.

## Profiling — use the unified nsys path

Benchmarking and profiling are consolidated into a single nsys-driven path; see
[BENCHMARK_GUIDE.md](../../BENCHMARK_GUIDE.md). Capture → export → analyze:

```
CUDA_VISIBLE_DEVICES=1 scripts/profile_nsys.sh myrun -- \
    -i inputs/object_removal/bmx-trees -m inputs/object_removal/bmx-trees_mask \
    --fp16 --height 720 --width 1280 --frames 80
```

### `analyze_nsys_metrics.py`
Turns an nsys sqlite into the hierarchical bottleneck report
`results/report/PROFILE.md` (`pp/` → `model/` → `tst/` ranges with wall-ms,
%-of-total, %-of-parent, peak GPU memory, and GPU counters when permitted).
Invoked by `profile_nsys.sh`; can also be run standalone on an existing sqlite.

## Transformer-stage correctness eval (kernel optimization)

### `capture_stage_inputs.py`
Captures the real per-window input to the sparse transformer stage
(`TemporalSparseTransformerBlock.forward`) from a production run, to
`eval_fixtures/<name>/`. Builds a replayable fixture for offline work.

### `eval_transformer_stage.py`
Replays a captured fixture through a stage implementation and reports aggregate
latency + correctness (max-abs-err / PSNR vs the eager reference). This is the
**correctness** check nsys can't do; kept separate from the profiling path.

## Misc

### `test_fullgraph.py`
Smoke-tests `torch.compile(fullgraph=True)` on `InpaintGenerator` with
representative dummy shapes (no video decode needed).
