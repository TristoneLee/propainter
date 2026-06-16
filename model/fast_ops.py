"""Register the fused `cuda_extension` kernels as torch.library custom ops.

Why
---
`cuda_extension` exposes its kernels as raw pybind11 functions. To the PyTorch
dispatcher those are *opaque Python calls*: torch.compile graph-breaks on them
and the CUDA-graph machinery has no schema/fake-tensor view of them. Wrapping
each kernel in a `torch.library.custom_op` (with a `register_fake` meta impl)
turns them into first-class operators under the `propainter_fast::` namespace.
That is the prerequisite for capturing the fused transformer block into a CUDA
graph (manual capture or `torch.compile(mode="reduce-overhead")`) so the ~40
small kernel launches per block × N blocks collapse into a single replay,
eliminating the per-op launch overhead.

What is and isn't wrapped
-------------------------
The per-block compute kernels (linear / layernorm / residual_norm / add_residual
/ dwconv_pool / ffn_* / attn_unmasked / attn_masked) are wrapped — they are pure
"allocate output, launch on the current stream" calls and are capture-safe once
the cuBLASLt/cuDNN plans + workspaces are warmed up.

`window_mask_partition` is deliberately NOT wrapped: it does a D2H copy +
`cudaStreamSynchronize` and returns data-dependent-size tensors, so it must run
*outside* any captured region (it is already called once before the per-block
loop). The weight-prep helpers (`to_half`, `w1_filter`, `w2_filter`) likewise
run once and are cached, so they stay as direct `ext.*` calls.

The op signatures (names, argument order, defaults) mirror the pybind functions
exactly, so `propainter_fast` can pass either `cuda_extension` or
`torch.ops.propainter_fast` as its `ops` namespace and the body is identical.
"""
from typing import List, Optional, Tuple

import torch

NAMESPACE = "propainter_fast"

# Loaded lazily on first op invocation (fake/meta impls never touch it).
_EXT = None


def _ext():
    global _EXT
    if _EXT is None:
        # Lazy import avoids a circular import at module load time
        # (propainter_fast imports this module to trigger registration).
        from . import propainter_fast as _pf
        _EXT = _pf._load_ext()
        if _EXT is None:
            raise RuntimeError(
                "cuda_extension is unavailable; cannot run propainter_fast custom ops. "
                f"Last load error: {_pf._EXT_ERR!r}")
    return _EXT


_HAS_CUSTOM_OP = hasattr(torch.library, "custom_op")


def _registered() -> bool:
    """True if the ops were already registered (idempotent re-import guard)."""
    ns = getattr(torch.ops, NAMESPACE, None)
    return ns is not None and hasattr(ns, "linear")


if _HAS_CUSTOM_OP and not _registered():
    _custom_op = torch.library.custom_op

    # ------------------------------------------------------------------ gemm
    @_custom_op(f"{NAMESPACE}::linear", mutates_args=(), device_types="cuda")
    def linear(input: torch.Tensor, weight: torch.Tensor,
               bias: torch.Tensor) -> torch.Tensor:
        return _ext().linear(input, weight, bias)

    @linear.register_fake
    def _(input, weight, bias):
        shape = list(input.shape)
        shape[-1] = weight.shape[0]
        return input.new_empty(shape, dtype=torch.float16)

    # ------------------------------------------------------------- layernorm
    @_custom_op(f"{NAMESPACE}::layernorm", mutates_args=(), device_types="cuda")
    def layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                  pad_w: int = 0, pad_h: int = 0) -> torch.Tensor:
        return _ext().layernorm(x, weight, bias, pad_w, pad_h)

    @layernorm.register_fake
    def _(x, weight, bias, pad_w=0, pad_h=0):
        b, t, h, w, c = x.shape
        if pad_h > 0 or pad_w > 0:
            return x.new_empty((b * t, h + pad_h, w + pad_w, c), dtype=torch.float16)
        return x.new_empty((b, t, h, w, c), dtype=torch.float16)

    # ------------------------------------------------------- residual_norm
    @_custom_op(f"{NAMESPACE}::residual_norm", mutates_args=(), device_types="cuda")
    def residual_norm(x: torch.Tensor, branch: torch.Tensor, weight: torch.Tensor,
                      bias: torch.Tensor, pad_w: int = 0,
                      bias2: Optional[torch.Tensor] = None,
                      pad_h: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        return _ext().residual_norm(x, branch, weight, bias, pad_w, bias2, pad_h)

    @residual_norm.register_fake
    def _(x, branch, weight, bias, pad_w=0, bias2=None, pad_h=0):
        b, t, h, w, c = x.shape
        x_new = x.new_empty((b, t, h, w, c), dtype=torch.float16)
        if pad_h > 0 or pad_w > 0:
            norm_out = x.new_empty((b * t, h + pad_h, w + pad_w, c), dtype=torch.float16)
        else:
            norm_out = x.new_empty((b, t, h, w, c), dtype=torch.float16)
        return x_new, norm_out

    # --------------------------------------------------------- add_residual
    @_custom_op(f"{NAMESPACE}::add_residual", mutates_args=(), device_types="cuda")
    def add_residual(x: torch.Tensor, y: torch.Tensor,
                     bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        return _ext().add_residual(x, y, bias)

    @add_residual.register_fake
    def _(x, y, bias=None):
        return x.new_empty(x.shape, dtype=torch.float16)

    # ----------------------------------------------------------- dwconv_pool
    @_custom_op(f"{NAMESPACE}::dwconv_pool", mutates_args=(), device_types="cuda")
    def dwconv_pool(x: torch.Tensor, weight: torch.Tensor,
                    bias: torch.Tensor) -> torch.Tensor:
        return _ext().dwconv_pool(x, weight, bias)

    @dwconv_pool.register_fake
    def _(x, weight, bias):
        bt, H, W, C = x.shape
        kh, kw = int(weight.shape[2]), int(weight.shape[3])
        OH = (H - kh) // kh + 1
        OW = (W - kw) // kw + 1
        return x.new_empty((bt, OH, OW, C), dtype=torch.float16)

    # --------------------------------------------------------- attn_unmasked
    @_custom_op(f"{NAMESPACE}::attn_unmasked", mutates_args=(), device_types="cuda")
    def attn_unmasked(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                      unmask_ind: torch.Tensor, n_head: int, wh: int,
                      ww: int) -> torch.Tensor:
        return _ext().attn_unmasked(q, k, v, unmask_ind, n_head, wh, ww)

    @attn_unmasked.register_fake
    def _(q, k, v, unmask_ind, n_head, wh, ww):
        return q.new_empty(q.shape, dtype=torch.float16)

    # ----------------------------------------------------------- attn_masked
    # Mutates `out` in place (accumulates the masked-window results computed by
    # attn_unmasked) and returns nothing.
    @_custom_op(f"{NAMESPACE}::attn_masked", mutates_args=("out",), device_types="cuda")
    def attn_masked(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    pk: torch.Tensor, pv: torch.Tensor, mask_ind: torch.Tensor,
                    out: torch.Tensor, n_head: int, t_off: int, t_dil: int,
                    wh: int, ww: int, eh: int, ew: int,
                    rtab: Optional[torch.Tensor] = None) -> None:
        # `rtab` is the precomputed device roll-table (constant per window
        # config); passing it keeps this op CUDA-graph capturable. None falls
        # back to the C++ host build (legacy eager path).
        _ext().attn_masked(q, k, v, pk, pv, mask_ind, out, n_head, t_off,
                           t_dil, wh, ww, eh, ew, rtab)

    @attn_masked.register_fake
    def _(q, k, v, pk, pv, mask_ind, out, n_head, t_off, t_dil, wh, ww, eh, ew,
          rtab=None):
        return None

    # ------------------------------------------------------------- ffn_fold
    @_custom_op(f"{NAMESPACE}::ffn_fold_conv", mutates_args=(), device_types="cuda")
    def ffn_fold_conv(y: torch.Tensor, K1: torch.Tensor, OH: int, OW: int,
                      kh: int, kw: int, sh: int, sw: int, ph: int,
                      pw: int) -> torch.Tensor:
        return _ext().ffn_fold_conv(y, K1, OH, OW, kh, kw, sh, sw, ph, pw)

    @ffn_fold_conv.register_fake
    def _(y, K1, OH, OW, kh, kw, sh, sw, ph, pw):
        b, t = int(y.shape[0]), int(y.shape[1])
        CO = int(K1.shape[3])
        return y.new_empty((b * t, OH, OW, CO), dtype=torch.float16)

    @_custom_op(f"{NAMESPACE}::ffn_gelu_div", mutates_args=(), device_types="cuda")
    def ffn_gelu_div(raw: torch.Tensor, bias1: torch.Tensor, fh: int, fw: int,
                     kh: int, kw: int, sh: int, sw: int, ph: int,
                     pw: int) -> torch.Tensor:
        return _ext().ffn_gelu_div(raw, bias1, fh, fw, kh, kw, sh, sw, ph, pw)

    @ffn_gelu_div.register_fake
    def _(raw, bias1, fh, fw, kh, kw, sh, sw, ph, pw):
        return raw.new_empty(raw.shape, dtype=torch.float16)

    @_custom_op(f"{NAMESPACE}::ffn_unfold_conv", mutates_args=(), device_types="cuda")
    def ffn_unfold_conv(fd2: torch.Tensor, K2: torch.Tensor, b: int, kh: int,
                        kw: int, sh: int, sw: int, ph: int,
                        pw: int) -> torch.Tensor:
        return _ext().ffn_unfold_conv(fd2, K2, b, kh, kw, sh, sw, ph, pw)

    @ffn_unfold_conv.register_fake
    def _(fd2, K2, b, kh, kw, sh, sw, ph, pw):
        bt, OH, OW, CO = (int(s) for s in fd2.shape)
        Cm = int(K2.shape[0])
        fh = (OH + 2 * ph - kh) // sh + 1
        fw = (OW + 2 * pw - kw) // sw + 1
        return fd2.new_empty((b, bt // b, fh, fw, Cm), dtype=torch.float16)


def ops_namespace():
    """Return the registered op namespace (`torch.ops.propainter_fast`) or None."""
    if _HAS_CUSTOM_OP and _registered():
        return getattr(torch.ops, NAMESPACE)
    return None
