"""Isolated steady-state micro-bench: eager vs torch.compile for the encoder
conv stack at representative window-batch sizes (b*t) and 720p. Run from repo
root. TEST_COMPILE=1 to compile (dynamic=True). Confirms whether Inductor
fusion beats cuDNN NCHW fp16 on sm_120 in steady state (compile cost excluded
via warmup)."""
import os, sys, time, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault('PROPAINTER_FAST', '1')
from utils.download_util import load_file_from_url
from model.propainter import InpaintGenerator
from model.misc import get_device
import warnings; warnings.filterwarnings('ignore')

COMPILE = os.environ.get('TEST_COMPILE', '0') == '1'
URL = 'https://github.com/sczhou/ProPainter/releases/download/v0.1.0/'
dev = get_device()
ck = load_file_from_url(url=os.path.join(URL, 'ProPainter.pth'), model_dir='weights')
m = InpaintGenerator(model_path=ck).to(dev).eval().half()
enc = m.encoder
cin = [mm.in_channels for mm in m.encoder.modules() if isinstance(mm, nn.Conv2d)][0]
print('encoder in_channels:', cin)
if COMPILE:
    enc = torch.compile(enc, dynamic=True)


def bench(fn, x, n=40, w=10):
    with torch.no_grad():
        for _ in range(w):
            fn(x)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(n):
            fn(x)
        torch.cuda.synchronize(); return (time.perf_counter() - t0) / n * 1000


for bt in [10, 14]:
    x = torch.randn(bt, cin, 720, 1280, device=dev, dtype=torch.float16)
    print(f'  encoder bt={bt}: {bench(lambda z: enc(z), x):.2f} ms/call compile={COMPILE}')

# --- per-process warmup cost: time first compiled call (retrace+guard, cache warm) ---
if COMPILE:
    import gc; gc.collect(); torch.cuda.empty_cache()
    enc2 = torch.compile(m.encoder, dynamic=True)
    x = torch.randn(14, cin, 720, 1280, device=dev, dtype=torch.float16)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.no_grad(): enc2(x)
    torch.cuda.synchronize()
    print(f'  FIRST compiled call (warm disk cache, fresh process): {(time.perf_counter()-t0)*1000:.0f} ms')
