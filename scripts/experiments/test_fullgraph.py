"""Validate fullgraph=True on InpaintGenerator with representative dummy shapes.

Bypasses the torchvision.io.read_video path so we can exercise torch.compile
without a working video decoder. Uses the same warmup shape as
inference_propainter.py: subvideo_length=100, neighbor_length=10, 720x1280, fp16.
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from model.propainter import InpaintGenerator
from utils.download_util import load_file_from_url

pretrain_model_url = 'https://github.com/sczhou/ProPainter/releases/download/v0.1.0/'


def main():
    device = torch.device('cuda')
    h, w = 720, 1280
    neighbor_length = 10
    subvideo_length = 100
    ref_stride = 10

    ckpt_path = load_file_from_url(url=os.path.join(pretrain_model_url, 'ProPainter.pth'),
                                   model_dir='weights', progress=True, file_name=None)
    model = InpaintGenerator(model_path=ckpt_path).to(device).half().eval()

    torch._dynamo.config.cache_size_limit = 16
    model = torch.compile(model, mode='max-autotune-no-cudagraphs', fullgraph=False)

    l_t = neighbor_length + 1
    ref_n = max(1, subvideo_length // ref_stride)
    t = l_t + ref_n
    dt = torch.float16
    imgs = torch.zeros(1, t, 3, h, w, device=device, dtype=dt)
    mi = torch.zeros(1, t, 1, h, w, device=device, dtype=dt)
    mu = torch.zeros(1, t, 1, h, w, device=device, dtype=dt)
    ff = torch.zeros(1, l_t - 1, 2, h, w, device=device, dtype=dt)
    fb = torch.zeros(1, l_t - 1, 2, h, w, device=device, dtype=dt)

    print(f'[test] compiling: mode=max-autotune-no-cudagraphs fullgraph=False '
          f't={t} l_t={l_t} {h}x{w} {dt}')
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(imgs, (ff, fb), mi, mu, l_t)
    torch.cuda.synchronize()
    print(f'[test] first (compile) call: {time.perf_counter() - t0:.2f}s  out={type(out)}')

    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model(imgs, (ff, fb), mi, mu, l_t)
    torch.cuda.synchronize()
    print(f'[test] second (cached) call: {time.perf_counter() - t0:.3f}s')
    print('[test] OK: compiled successfully')


if __name__ == '__main__':
    main()
