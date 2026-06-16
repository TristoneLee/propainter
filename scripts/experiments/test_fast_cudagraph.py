"""CUDA-graph capture/replay of the fused transformer block: correctness + speed.

Validates that `FastTemporalSparseTransformerBlock` with PROPAINTER_FAST_CUDAGRAPH
- captures the per-block loop and replays it bit-identically to the eager fast
  path (the window partition stays eager, outside the graph),
- picks up fresh input data on replay (not stale captured data),
and reports the per-call wall-time reduction from collapsing the per-op launches
into a single graph replay.

Run from repo root:  python scripts/experiments/test_fast_cudagraph.py
Exit 0 on pass, 1 on numerical mismatch, 2 if CUDA / extension unavailable.
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch


def _time_forward(blk, x, fold, lm, t_dil, iters=50):
    torch.cuda.synchronize()
    # warm up (captures the graph on the first call for the graph path)
    for _ in range(5):
        blk.forward(x, fold, lm.clone(), t_dil)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        blk.forward(x, fold, lm.clone(), t_dil)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms/call


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return 2

    from model.propainter_fast import FastTemporalSparseTransformerBlock, _load_ext
    from model import fast_ops

    if _load_ext() is None or fast_ops.ops_namespace() is None:
        print("SKIP: cuda_extension / custom ops unavailable")
        return 2
    if not hasattr(_load_ext(), "build_rtab"):
        print("SKIP: extension predates build_rtab (rebuild model/cuda_extension.so)")
        return 2

    dim, n_head, window, pool, depths = 512, 4, (5, 9), (4, 4), 8
    t2t = {"kernel_size": (7, 7), "stride": (3, 3), "padding": (3, 3)}
    blk = FastTemporalSparseTransformerBlock(
        dim, n_head, window, pool, depths, t2t_params=t2t).cuda().half().eval()

    B, T, H, W, C = 1, 10, 60, 107, 512
    fold = (180, 320)
    torch.manual_seed(0)
    x1 = torch.randn(B, T, H, W, C, device="cuda", dtype=torch.float16) * 0.1
    x2 = torch.randn(B, T, H, W, C, device="cuda", dtype=torch.float16) * 0.1
    lm = torch.zeros(B, T, H, W, 1, device="cuda", dtype=torch.float32)
    lm[:, :, : H // 2] = 1.0   # mixed pattern -> both attention branches active

    ok = True
    with torch.no_grad():
        # Eager fast path (production default: direct pybind) as reference.
        blk._cudagraph = False
        blk._use_ops = False
        y_ref1 = blk.forward(x1, fold, lm.clone(), 2).clone()
        y_ref2 = blk.forward(x2, fold, lm.clone(), 2).clone()

        # Graph path: first call captures, replays; second call (fresh x) is a
        # cache hit that must reflect the new input, not stale captured data.
        blk._cudagraph = True
        y_g1 = blk.forward(x1, fold, lm.clone(), 2).clone()
        y_g2 = blk.forward(x2, fold, lm.clone(), 2).clone()

    n_graphs = sum(1 for v in blk._graphs.values() if v not in (None, False))
    print(f"  captured graphs: {n_graphs}  (keys: {list(blk._graphs.keys())})")

    d1 = (y_ref1.float() - y_g1.float()).abs().max().item()
    d2 = (y_ref2.float() - y_g2.float()).abs().max().item()
    # Cross-check the graph actually re-ran on x2 (outputs for x1 vs x2 differ).
    cross = (y_g1.float() - y_g2.float()).abs().max().item()
    print(f"  [capture ] graph vs eager (x1)  max_abs_diff = {d1:.3e}  "
          f"{'OK' if d1 == 0.0 else 'MISMATCH'}")
    print(f"  [replay  ] graph vs eager (x2)  max_abs_diff = {d2:.3e}  "
          f"{'OK' if d2 == 0.0 else 'MISMATCH'}")
    print(f"  [freshness] |graph(x1)-graph(x2)| = {cross:.3e}  "
          f"{'OK' if cross > 0.0 else 'STALE!'}")
    ok = ok and d1 == 0.0 and d2 == 0.0 and cross > 0.0

    # Timing: eager direct-pybind vs eager custom-op vs graph replay.
    blk._cudagraph = False
    blk._use_ops = False
    t_ext = _time_forward(blk, x1, fold, lm, 2)
    blk._use_ops = True
    t_ops = _time_forward(blk, x1, fold, lm, 2)
    blk._use_ops = False
    blk._cudagraph = True
    t_graph = _time_forward(blk, x1, fold, lm, 2)

    print(f"\n  per-call wall (ms):")
    print(f"    eager direct-pybind : {t_ext:8.3f}")
    print(f"    eager custom-op     : {t_ops:8.3f}")
    print(f"    cuda-graph replay   : {t_graph:8.3f}")
    print(f"    speedup vs direct   : {t_ext / t_graph:6.2f}x  "
          f"({t_ext - t_graph:+.3f} ms/call saved)")

    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
