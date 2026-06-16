"""Single NVTX instrumentation primitive for ProPainter profiling.

All performance instrumentation goes through `nvtx(tag)` so one `nsys` capture
sees the whole hierarchy in one timeline:

    pp/*     pipeline stages          (inference_propainter.py)
    model/*  InpaintGenerator.forward (model/propainter.py)
    tst/*    transformer segments     (model/modules/sparse_transformer.py)

Gated by the PROPAINTER_NVTX=1 env var (set by scripts/profile_nsys.sh). When
unset, `nvtx()` is a zero-overhead no-op so normal inference is unaffected.
"""
import contextlib
import os

import torch

NVTX_ENABLED = os.environ.get('PROPAINTER_NVTX', '0') == '1'


class _NvtxRange:
    __slots__ = ('tag',)

    def __init__(self, tag):
        self.tag = tag

    def __enter__(self):
        torch.cuda.nvtx.range_push(self.tag)
        return self

    def __exit__(self, *exc):
        torch.cuda.nvtx.range_pop()
        return False


def nvtx(tag):
    """Context manager emitting an NVTX range `tag` when PROPAINTER_NVTX=1, else no-op."""
    if NVTX_ENABLED:
        return _NvtxRange(tag)
    return contextlib.nullcontext()
