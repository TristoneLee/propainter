"""Drop-in accelerated TemporalSparseTransformerBlock for ProPainter.

Production integration
----------------------
In ProPainter's `model/propainter.py`, the transformer is built around line 297:

    from .misc import TemporalSparseTransformerBlock        # original
    self.transformers = TemporalSparseTransformerBlock(...)

Replace that import with:

    from .propainter_fast import FastTemporalSparseTransformerBlock \
        as TemporalSparseTransformerBlock

Nothing else changes: the module name, __init__ signature, forward signature,
parameter/buffer names and therefore the checkpoint state_dict are all identical
to the original, so an existing ProPainter checkpoint loads unchanged.

Safety
------
`forward` runs the CUDA extension only when every kernel assumption holds
(CUDA fp32 input, batch==1, C divisible by 8, extension importable). Otherwise
it transparently falls back to the original PyTorch path, so the model never
fails because of the accelerator. Set env `PROPAINTER_FAST=0` to force fallback,
or `PROPAINTER_FAST_STRICT=1` to raise instead of silently falling back (useful
in tests/CI to catch shapes that should have been accelerated).
"""
import ctypes
import os
import sys
import warnings

import torch
import torch.nn as nn

# --- original implementation (the module we are wrapping) ---
# Import path matches ProPainter's package layout; adjust if vendored elsewhere.
try:
    from .modules.sparse_transformer import TemporalSparseTransformerBlock as _OrigBlock  # type: ignore
except Exception:  # pragma: no cover - allows standalone use next to model.py
    from model.modules.sparse_transformer import TemporalSparseTransformerBlock as _OrigBlock

# Importing fast_ops registers the fused kernels as torch.library custom ops
# (torch.ops.propainter_fast.*). The per-block loop can then run through either
# the raw pybind module (`_EXT`, lowest eager overhead) or the custom-op
# namespace (capturable into a CUDA graph / traceable by torch.compile).
try:
    from . import fast_ops as _fast_ops  # type: ignore
except Exception:  # pragma: no cover
    import model.fast_ops as _fast_ops


_EXT = None
_EXT_ERR = None


def _load_ext():
    """Import the compiled cuda_extension once, preloading cuDNN for it."""
    global _EXT, _EXT_ERR
    if _EXT is not None or _EXT_ERR is not None:
        return _EXT
    try:
        sp = os.path.dirname(os.path.dirname(torch.__file__))
        for cand in (os.path.join(sp, 'nvidia', 'cudnn', 'lib', 'libcudnn.so.9'),
                     os.path.join(os.path.dirname(torch.__file__), 'lib', 'libcudnn.so.9'),
                     'libcudnn.so.9'):
            try:
                ctypes.CDLL(cand, mode=ctypes.RTLD_GLOBAL)
                break
            except OSError:
                continue
        # cuda_extension.so ships next to this module; ensure it's importable
        # regardless of the process CWD (inference runs from the repo root).
        _here = os.path.dirname(os.path.abspath(__file__))
        if _here not in sys.path:
            sys.path.insert(0, _here)
        import cuda_extension as ext  # built from kernels/ via build_ext.py
        _EXT = ext
    except Exception as e:  # noqa: BLE001
        _EXT_ERR = e
        warnings.warn(f"cuda_extension unavailable, using PyTorch fallback: {e}")
    return _EXT


class FastTemporalSparseTransformerBlock(_OrigBlock):
    """Same module, with an accelerated forward and a PyTorch safety net."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._half_cache = {}
        self._rtab_cache = {}
        self._fast_enabled = os.environ.get("PROPAINTER_FAST", "1") != "0"
        self._strict = os.environ.get("PROPAINTER_FAST_STRICT", "0") == "1"
        # Route the per-block compute kernels through the torch.library custom-op
        # namespace instead of the raw pybind module. Adds a little eager dispatch
        # overhead, but makes the block capturable into a CUDA graph. Off by
        # default so plain eager keeps the lowest-overhead direct-pybind path.
        self._use_ops = os.environ.get("PROPAINTER_FAST_OPS", "0") == "1"
        # Capture the per-block loop into a CUDA graph and replay it, collapsing
        # the ~40-launches-per-block × depths host launches into one replay. The
        # window partition (host sync) stays outside the graph. Implies the
        # custom-op path. Graphs are cached per (T, mask/unmask counts, grid).
        self._cudagraph = os.environ.get("PROPAINTER_FAST_CUDAGRAPH", "0") == "1"
        self._graphs = {}

    # ---- weight conversion caches (invalidated on in-place param updates) ----
    def _f32(self, p):
        """fp32 view of a param (cached). The kernels take small per-channel
        params (LN gamma/beta, biases, conv weights) in fp32, but model.half()
        turns them fp16; this restores fp32 so the fp16-input model still works."""
        if p.dtype == torch.float32:
            return p
        key = (id(p), 'f32')
        ver = p._version
        ent = self._half_cache.get(key)
        if ent is None or ent[0] != ver:
            ent = (ver, p.float().contiguous())
            self._half_cache[key] = ent
        return ent[1]

    def _h(self, p):
        ext = _EXT
        key = (id(p), 'h')
        ver = p._version
        ent = self._half_cache.get(key)
        if ent is None or ent[0] != ver:
            ent = (ver, ext.to_half(self._f32(p)))
            self._half_cache[key] = ent
        return ent[1]

    def _rtab(self, wh, ww, eh, ew):
        """Cached device roll-table for attn_masked (constant per window config).

        Built once via the extension and reused so the captured per-block loop
        carries no host build + synchronous H2D copy. Returns None if the loaded
        extension predates build_rtab (then attn_masked falls back to its C++
        host build — non-capturable but still correct in eager mode)."""
        if not hasattr(_EXT, "build_rtab"):
            return None
        key = (wh, ww, eh, ew)
        t = self._rtab_cache.get(key)
        if t is None:
            t = _EXT.build_rtab(wh, ww, eh, ew)
            self._rtab_cache[key] = t
        return t

    def _filt(self, p, which, kh, kw):
        ext = _EXT
        key = (id(p), which)
        ver = p._version
        ent = self._half_cache.get(key)
        if ent is None or ent[0] != ver:
            pf = self._f32(p)
            t = ext.w1_filter(pf, kh, kw) if which == 'w1' else ext.w2_filter(pf, kh, kw)
            ent = (ver, t)
            self._half_cache[key] = ent
        return ent[1]

    def _can_accelerate(self, x):
        if not self._fast_enabled:
            return False
        if _load_ext() is None:
            return False
        c = x.size(-1)
        n_head = self.transformer[0].attention.n_head
        return (x.is_cuda and x.dtype in (torch.float32, torch.float16)
                and x.dim() == 5
                and x.size(0) == 1 and c % 8 == 0 and (c // n_head) % 8 == 0)

    @torch.no_grad()
    def prime_plans(self, fold_x_size, h, w, c, T_range, mask_levels=(0.0, 0.5),
                    t_dilation=2, device='cuda', dtype=torch.float16):
        """Pre-build cuDNN/cuBLASLt plans for every (T, mask-coverage) shape that
        serving will hit, so real requests are all cache hits (no per-shape
        autotune spike inside the loop).

        fold_x_size, h, w, c: the fixed token-grid dims for this deployment.
        T_range: iterable of frame counts to cover (e.g. range(T_min, T_max+1)).
        mask_levels: fraction of windows to mark masked, to probe whether the
            masked-attention plan key also depends on mask coverage.
        Returns the number of (T, level) shapes primed.
        """
        if _load_ext() is None:
            return 0
        n_head = self.transformer[0].attention.n_head
        wh, ww = self.transformer[0].attention.window_size
        import math as _m
        n_wh, n_ww = _m.ceil(h / wh), _m.ceil(w / ww)
        n_primed = 0
        for T in T_range:
            for lvl in mask_levels:
                x = torch.zeros(1, T, h, w, c, device=device, dtype=dtype)
                # l_mask is [1, l_t, h, w, 1]; mark a fraction of rows masked so
                # window_mask_partition yields a representative masked/unmasked split.
                lm = torch.zeros(1, T, h, w, 1, device=device, dtype=torch.float32)
                if lvl > 0:
                    n_mask_rows = max(1, int(round(h * lvl)))
                    lm[:, :, :n_mask_rows] = 1.0
                try:
                    self._forward_fast(x, fold_x_size, lm, t_dilation)
                    n_primed += 1
                except Exception as e:  # noqa: BLE001 - priming is best-effort
                    if self._strict:
                        raise
                    warnings.warn(f"prime_plans skipped T={T} lvl={lvl}: {e}")
        torch.cuda.synchronize()
        return n_primed

    def forward(self, x, fold_x_size, l_mask=None, t_dilation=2):
        if self._can_accelerate(x):
            try:
                if self._cudagraph:
                    return self._forward_fast_graph(x, fold_x_size, l_mask, t_dilation)
                ops = _fast_ops.ops_namespace() if self._use_ops else None
                return self._forward_fast(x, fold_x_size, l_mask, t_dilation, ops=ops)
            except Exception as e:  # noqa: BLE001
                if self._strict:
                    raise
                warnings.warn(f"fast path failed ({e}); falling back to PyTorch")
        return super().forward(x, fold_x_size, l_mask, t_dilation)

    def _forward_fast(self, x, fold_x_size, l_mask, t_dilation, ops=None):
        ext = _EXT
        # `ops` selects the compute-kernel namespace for the per-block loop:
        #   None  -> raw pybind module (_EXT), lowest eager overhead
        #   torch.ops.propainter_fast -> custom ops (CUDA-graph capturable)
        if ops is None:
            ops = ext
        import math
        assert self.depths % t_dilation == 0, 'wrong t_dilation input.'
        # The residual stream is fp16 throughout the kernels now. Accept either
        # fp32 or fp16 input; convert to fp16 once here and cast the result back
        # to the caller's dtype at the end so the block is dtype-transparent.
        in_dtype = x.dtype
        if x.dtype != torch.float16:
            x = x.half()
        if l_mask is not None and l_mask.dtype != torch.float32:
            l_mask = l_mask.float()
        at0 = self.transformer[0].attention
        wh, ww = at0.window_size
        eh, ew = (wh + 1) // 2, (ww + 1) // 2
        h, w = x.size(2), x.size(3)
        n_wh = math.ceil(h / wh)
        n_ww = math.ceil(w / ww)
        # Host-syncing partition + one-time weight conversion stay on the raw
        # pybind module: they must run OUTSIDE any CUDA-graph captured region.
        mask_ind, unmask_ind = ext.window_mask_partition(l_mask, wh, ww, n_wh, n_ww)
        rtab = self._rtab(wh, ww, eh, ew)
        x = self._run_blocks(x, fold_x_size, mask_ind, unmask_ind, rtab, t_dilation, ops)
        return x if in_dtype == torch.float16 else x.to(in_dtype)

    def _run_blocks(self, x, fold_x_size, mask_ind, unmask_ind, rtab, t_dilation, ops):
        """Capturable per-block compute region.

        Assumes `x` is fp16 [b,t,h,w,c] and the window partition
        (mask_ind/unmask_ind) and roll-table (rtab) are already computed. Every
        op here is a stream-ordered kernel launch with no host sync and no
        data-dependent allocation (allocations depend only on the partition
        counts, which are fixed for a given capture), so the whole region can be
        captured into a CUDA graph and replayed. Returns the fp16 output."""
        import math
        b, t, h, w, _ = x.shape
        at0 = self.transformer[0].attention
        n_head = at0.n_head
        wh, ww = at0.window_size
        eh, ew = (wh + 1) // 2, (ww + 1) // 2
        n_wh = math.ceil(h / wh)
        n_ww = math.ceil(w / ww)
        pad_w = n_ww * ww - w
        pad_h = n_wh * wh - h

        OH, OW = fold_x_size
        t2t = self.transformer[0].mlp.t2t_params
        kh, kw = t2t['kernel_size']
        sh, sw = t2t['stride']
        ph, pw = t2t['padding']

        # ext.layernorm / residual_norm return 4D [b*t, H, W, C] when there is
        # padding, but 5D [b, t, h, w, c] when pad_w == pad_h == 0. The rest of
        # the fast loop (dwconv_pool, attn_*) requires the 4D layout, so collapse
        # the leading b,t dims here (b == 1 is guaranteed by _can_accelerate).
        def _as4d(xn_):
            return xn_.reshape(b * t, xn_.size(-3), xn_.size(-2), xn_.size(-1)) \
                if xn_.dim() == 5 else xn_

        _f32 = self._f32
        xn = _as4d(ops.layernorm(x, _f32(self.transformer[0].norm1.weight),
                                 _f32(self.transformer[0].norm1.bias), pad_w, pad_h))
        for i in range(self.depths):
            blk = self.transformer[i]
            at = blk.attention
            kw_h, kb_h = self._h(at.key.weight), self._h(at.key.bias)
            vw_h, vb_h = self._h(at.value.weight), self._h(at.value.bias)
            q = ops.linear(xn, self._h(at.query.weight), self._h(at.query.bias))
            k = ops.linear(xn, kw_h, kb_h)
            v = ops.linear(xn, vw_h, vb_h)
            px = ops.dwconv_pool(xn, _f32(at.pool_layer.weight), _f32(at.pool_layer.bias))
            pk = ops.linear(px, kw_h, kb_h)
            pv = ops.linear(px, vw_h, vb_h)

            ob = ops.attn_unmasked(q, k, v, unmask_ind, n_head, wh, ww)
            ops.attn_masked(q, k, v, pk, pv, mask_ind, ob, n_head,
                            i % t_dilation, t_dilation, wh, ww, eh, ew, rtab)
            pr = ops.linear(ob, self._h(at.proj.weight), self._h(at.proj.bias))
            x, y = ops.residual_norm(x, pr, _f32(blk.norm2.weight), _f32(blk.norm2.bias), 0)

            K1 = self._filt(blk.mlp.fc1[0].weight, 'w1', kh, kw)
            K2 = self._filt(blk.mlp.fc2[1].weight, 'w2', kh, kw)
            raw = ops.ffn_fold_conv(y, K1, OH, OW, kh, kw, sh, sw, ph, pw)
            fd2 = ops.ffn_gelu_div(raw, _f32(blk.mlp.fc1[0].bias), h, w, kh, kw, sh, sw, ph, pw)
            f2 = ops.ffn_unfold_conv(fd2, K2, b, kh, kw, sh, sw, ph, pw)
            b2 = _f32(blk.mlp.fc2[1].bias)
            if i + 1 < self.depths:
                nxt = self.transformer[i + 1]
                x, xn = ops.residual_norm(x, f2, _f32(nxt.norm1.weight),
                                          _f32(nxt.norm1.bias), pad_w, b2, pad_h)
                xn = _as4d(xn)
            else:
                x = ops.add_residual(x, f2, b2)
        return x

    def _forward_fast_graph(self, x, fold_x_size, l_mask, t_dilation):
        """CUDA-graph capture/replay of the per-block loop.

        The window partition (host sync) and weight prep stay eager/outside the
        graph; only `_run_blocks` is captured. Graphs are cached by a shape key
        that includes the partition counts (they fix the per-block launch
        structure). On a cache miss we warm up plans/workspaces on a side stream
        then capture; on a hit we copy the live inputs into the static buffers
        and replay. Falls back to the eager fast path if capture fails."""
        import math
        ext = _EXT
        ops = _fast_ops.ops_namespace()
        if ops is None or not hasattr(ext, "build_rtab"):
            # Custom ops or capturable attn_masked unavailable: no graph path.
            return self._forward_fast(x, fold_x_size, l_mask, t_dilation, ops=ops)

        in_dtype = x.dtype
        x16 = x if x.dtype == torch.float16 else x.half()
        x16 = x16.contiguous()
        if l_mask is not None and l_mask.dtype != torch.float32:
            l_mask = l_mask.float()
        at0 = self.transformer[0].attention
        wh, ww = at0.window_size
        eh, ew = (wh + 1) // 2, (ww + 1) // 2
        b, t, h, w, c = x16.shape
        n_wh = math.ceil(h / wh)
        n_ww = math.ceil(w / ww)
        mask_ind, unmask_ind = ext.window_mask_partition(l_mask, wh, ww, n_wh, n_ww)
        rtab = self._rtab(wh, ww, eh, ew)

        key = (int(t), int(h), int(w), int(c), int(mask_ind.numel()),
               int(unmask_ind.numel()), int(t_dilation), tuple(fold_x_size))
        g = self._graphs.get(key)
        if g is None:
            try:
                g = self._capture_graph(x16, fold_x_size, mask_ind, unmask_ind,
                                        rtab, t_dilation, ops)
            except Exception as e:  # noqa: BLE001
                if self._strict:
                    raise
                warnings.warn(f"CUDA-graph capture failed ({e}); eager fast path")
                g = False  # remember the failure; don't retry this shape
            self._graphs[key] = g
        if g is False:
            return self._forward_fast(x, fold_x_size, l_mask, t_dilation, ops=ops)

        g["x"].copy_(x16)
        if g["mask_ind"].numel():
            g["mask_ind"].copy_(mask_ind)
        if g["unmask_ind"].numel():
            g["unmask_ind"].copy_(unmask_ind)
        g["graph"].replay()
        out = g["out"]
        return out.clone() if in_dtype == torch.float16 else out.to(in_dtype)

    def _capture_graph(self, x_ex, fold_x_size, mask_ind, unmask_ind, rtab,
                       t_dilation, ops):
        """Warm up then capture `_run_blocks` into a CUDA graph with static IO."""
        static_x = x_ex.clone()
        static_mask = mask_ind.clone()
        static_unmask = unmask_ind.clone()

        # Warm up on a side stream: triggers cuBLASLt/cuDNN plan selection +
        # workspace growth and fills the weight-prep cache, so nothing allocates
        # or autotunes during the capture itself.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._run_blocks(static_x, fold_x_size, static_mask,
                                 static_unmask, rtab, t_dilation, ops)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = self._run_blocks(static_x, fold_x_size, static_mask,
                                          static_unmask, rtab, t_dilation, ops)
        return {"graph": graph, "x": static_x, "mask_ind": static_mask,
                "unmask_ind": static_unmask, "out": static_out}
