"""Equivalence check: torch.library custom-op path vs direct-pybind path.

The custom ops registered in `model/fast_ops.py` wrap the *same* cuda_extension
kernels, so routing `_forward_fast` through `torch.ops.propainter_fast.*` must be
bit-identical to calling `cuda_extension.*` directly (same plans, same algos,
deterministic for fixed shapes). This asserts max_abs_diff == 0 on three mask
patterns, and also exercises the full `forward()` integration via the
PROPAINTER_FAST_OPS path.

Run from repo root:  python scripts/experiments/test_fast_ops.py
Exit 0 on pass, 1 on numerical mismatch, 2 if CUDA / extension unavailable.
"""
import os
import sys

# Allow `python scripts/experiments/test_fast_ops.py` from the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return 2

    from model.propainter_fast import FastTemporalSparseTransformerBlock, _load_ext
    from model import fast_ops

    if _load_ext() is None:
        print("SKIP: cuda_extension not importable")
        return 2

    ns = fast_ops.ops_namespace()
    if ns is None:
        print("SKIP: custom ops not registered (torch.library.custom_op missing?)")
        return 2

    dim, n_head, window, pool, depths = 512, 4, (5, 9), (4, 4), 8
    t2t = {"kernel_size": (7, 7), "stride": (3, 3), "padding": (3, 3)}
    blk = FastTemporalSparseTransformerBlock(
        dim, n_head, window, pool, depths, t2t_params=t2t).cuda().half().eval()

    B, T, H, W, C = 1, 10, 60, 107, 512        # post-encoder grid at 720p
    fold = (180, 320)
    torch.manual_seed(0)
    x = torch.randn(B, T, H, W, C, device="cuda", dtype=torch.float16) * 0.1

    patterns = {
        "zero": 0.0,    # no masked windows  -> attn_unmasked-only
        "one": 1.0,     # all masked         -> attn_masked-only
        "mixed": 0.5,   # both branches active
    }

    ok = True
    for name, lvl in patterns.items():
        lm = torch.zeros(B, T, H, W, 1, device="cuda", dtype=torch.float32)
        if lvl >= 1.0:
            lm[:] = 1.0
        elif lvl > 0.0:
            lm[:, :, : int(H * lvl)] = 1.0
        with torch.no_grad():
            y_ext = blk._forward_fast(x, fold, lm.clone(), 2, ops=None)
            y_ops = blk._forward_fast(x, fold, lm.clone(), 2, ops=ns)
        diff = (y_ext.float() - y_ops.float()).abs().max().item()
        status = "OK" if diff == 0.0 else "MISMATCH"
        print(f"  [{name:5s}] custom-op vs ext  max_abs_diff = {diff:.3e}  {status}")
        ok = ok and (diff == 0.0)

    # Precomputed device roll-table (graph-safe path) must match the C++
    # host-built table (legacy path). Force _rtab to None to take the fallback.
    lm = torch.zeros(B, T, H, W, 1, device="cuda", dtype=torch.float32)
    lm[:, :, : H // 2] = 1.0
    with torch.no_grad():
        y_rtab = blk._forward_fast(x, fold, lm.clone(), 2, ops=None)
        orig_rtab = blk._rtab
        blk._rtab = lambda *a, **k: None        # force C++ host-build fallback
        try:
            y_legacy = blk._forward_fast(x, fold, lm.clone(), 2, ops=None)
        finally:
            blk._rtab = orig_rtab
    rdiff = (y_rtab.float() - y_legacy.float()).abs().max().item()
    rstatus = "OK" if rdiff == 0.0 else "MISMATCH"
    print(f"  [rtab ] precomputed vs host-built max_abs_diff = {rdiff:.3e}  {rstatus}")
    ok = ok and (rdiff == 0.0)

    # Full forward() integration through the PROPAINTER_FAST_OPS toggle.
    lm = torch.zeros(B, T, H, W, 1, device="cuda", dtype=torch.float32)
    lm[:, :, : H // 2] = 1.0
    with torch.no_grad():
        blk._use_ops = False
        y_direct = blk.forward(x, fold, lm.clone(), 2)
        blk._use_ops = True
        y_viaops = blk.forward(x, fold, lm.clone(), 2)
    fdiff = (y_direct.float() - y_viaops.float()).abs().max().item()
    fstatus = "OK" if fdiff == 0.0 else "MISMATCH"
    print(f"  [fwd  ] forward() direct vs ops   max_abs_diff = {fdiff:.3e}  {fstatus}")
    ok = ok and (fdiff == 0.0)

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
