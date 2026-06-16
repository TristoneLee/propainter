#!/usr/bin/env python
"""Probe torch.compile compatibility of each InpaintGenerator submodule.

On the first real model() call (after the full flow/completion/img-prop pipeline
has produced genuine inputs), capture each submodule's real input via a
forward-pre-hook, then run torch._dynamo.explain on it to count graph breaks and
list reasons. For modules with 0 breaks, also attempt a real torch.compile()
forward to confirm Inductor can lower + execute it. Prints a report and exits
before the heavy window loop.

Usage (pick a free GPU):
    CUDA_VISIBLE_DEVICES=1 PROPAINTER_FAST=1 \
    python scripts/experiments/probe_compile.py \
        --video inputs/samples/2/input.mp4 --mask inputs/samples/2/masks.mp4 \
        --height 720 --width 1280 --frames 24 --fp16 --neighbor_length 10
"""
import argparse, os, sys, traceback

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)

import torch
from model.propainter import InpaintGenerator

# Submodule attribute names on InpaintGenerator to probe, in pipeline order.
PROBE = ['encoder', 'feat_prop_module', 'ss', 'transformers', 'sc', 'decoder']

_captured = {}


def _mk_prehook(name):
    def hook(module, args, kwargs):
        if name not in _captured:
            _captured[name] = (args, kwargs)
    return hook


def _shape(x):
    if torch.is_tensor(x):
        return f'{tuple(x.shape)}:{str(x.dtype).replace("torch.","")}'
    return repr(x)


def _explain_report(name, mod, args, kwargs):
    torch._dynamo.reset()
    line = f'\n### {name}  inputs=[{", ".join(_shape(a) for a in args)}]'
    if kwargs:
        line += f' kwargs={{{", ".join(f"{k}={_shape(v)}" for k,v in kwargs.items())}}}'
    print(line)
    try:
        exp = torch._dynamo.explain(mod)(*args, **kwargs)
    except Exception as e:  # noqa: BLE001
        print(f'  EXPLAIN FAILED: {type(e).__name__}: {str(e)[:300]}')
        return
    gc = getattr(exp, 'graph_count', '?')
    gbc = getattr(exp, 'graph_break_count', '?')
    oc = getattr(exp, 'op_count', '?')
    print(f'  graphs={gc}  graph_breaks={gbc}  ops={oc}')
    reasons = getattr(exp, 'break_reasons', []) or []
    seen = set()
    for i, r in enumerate(reasons):
        rs = getattr(r, 'reason', str(r))
        if rs in seen:
            continue
        seen.add(rs)
        print(f'    break[{len(seen)}]: {rs[:160]}')
        if len(seen) >= 6:
            print(f'    ... (+{len(reasons)-i-1} more break records)')
            break
    # If clean, confirm Inductor actually lowers + runs it.
    if gbc == 0:
        torch._dynamo.reset()
        try:
            cmod = torch.compile(mod, fullgraph=True)
            with torch.no_grad():
                cmod(*args, **kwargs)
            print('  fullgraph compile+run: OK (Inductor lowered & executed)')
        except Exception as e:  # noqa: BLE001
            print(f'  fullgraph compile+run: FAILED {type(e).__name__}: {str(e)[:240]}')


_orig_fwd = InpaintGenerator.forward


def _patched_fwd(self, *a, **k):
    handles = [getattr(self, n).register_forward_pre_hook(_mk_prehook(n), with_kwargs=True)
               for n in PROBE]
    try:
        with torch.no_grad():
            out = _orig_fwd(self, *a, **k)
    finally:
        for h in handles:
            h.remove()
    print('\n' + '=' * 78)
    print('torch.compile compatibility probe  (PROPAINTER_FAST=%s, torch %s)'
          % (os.environ.get('PROPAINTER_FAST', '1'), torch.__version__))
    print('=' * 78)
    for n in PROBE:
        if n not in _captured:
            print(f'\n### {n}: NOT CALLED this window (skipped)')
            continue
        args, kwargs = _captured[n]
        try:
            _explain_report(n, getattr(self, n), args, kwargs)
        except Exception:  # noqa: BLE001
            print(f'### {n}: probe crashed:\n{traceback.format_exc()[:600]}')
    print('\n' + '=' * 78)
    sys.stdout.flush()
    os._exit(0)


InpaintGenerator.forward = _patched_fwd

# ---- parse our extra args, hand the rest to inference_propainter.py ----
pre = argparse.ArgumentParser(add_help=False)
_, remaining = pre.parse_known_args()
sys.argv = ['inference_propainter.py'] + remaining
exec(open(os.path.join(_REPO, 'inference_propainter.py')).read())
