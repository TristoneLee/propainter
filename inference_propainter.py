# -*- coding: utf-8 -*-
import os
import sys
import cv2
import argparse
import imageio
import numpy as np
import scipy.ndimage
from PIL import Image
from tqdm import tqdm

import torch
import torchvision

from model.modules.flow_comp_raft import RAFT_bi
from model.recurrent_flow_completion import RecurrentFlowCompleteNet
from model.propainter import InpaintGenerator
from utils.download_util import load_file_from_url
from core.utils import to_tensors
from model.misc import get_device

import warnings
warnings.filterwarnings("ignore")

from model.nvtx import nvtx as _nvtx, NVTX_ENABLED as _NVTX

# Set PROPAINTER_NSYS_CAPTURE_ITER=N to fence the cudaProfilerApi capture window
# around iteration N of the feat_prop+transformer loop (0-indexed). nsys otherwise
# captures the whole run; use this only to isolate one heavy iteration.
_NSYS_CAPTURE_ITER = int(os.environ.get('PROPAINTER_NSYS_CAPTURE_ITER', '-1'))

# Pipeline-stage markers. NVTX push/pop (paired tick/tock) so a single nsys capture
# sees pp/* alongside the model/* and tst/* ranges. No-op without PROPAINTER_NVTX=1.
# PROPAINTER_TIME_STAGES=1 additionally records CUDA-synced wall per stage (NO nsys,
# so I/O isn't osrt-inflated) and prints a summary at the end — used to measure the
# decode/encode I/O share vs compute.
import time as _time_mod
_TIME_STAGES = os.environ.get('PROPAINTER_TIME_STAGES', '0') == '1'
_stage_times = {}
_stage_stack = []
_run_t0 = [None]

def _tick(name):
    if _NVTX:
        torch.cuda.nvtx.range_push('pp/' + name)
    if _TIME_STAGES:
        torch.cuda.synchronize()
        t = _time_mod.perf_counter()
        if _run_t0[0] is None:
            _run_t0[0] = t
        _stage_stack.append((name, t))

def _tock(name):
    if _NVTX:
        torch.cuda.nvtx.range_pop()
    if _TIME_STAGES:
        torch.cuda.synchronize()
        nm, t0 = _stage_stack.pop()
        _stage_times[nm] = _stage_times.get(nm, 0.0) + (_time_mod.perf_counter() - t0)

def _print_stage_times():
    if not _TIME_STAGES or not _stage_times:
        return
    total = _time_mod.perf_counter() - (_run_t0[0] or _time_mod.perf_counter())
    # Only top-level stages (no '/') count toward the accounted total; 'dl/*' are
    # nested sub-stages of data_loading shown as detail (would double-count).
    acct = sum(v for k, v in _stage_times.items() if '/' not in k)
    print('\n=== stage wall (CUDA-synced, no nsys) ===')
    for nm, s in sorted(_stage_times.items(), key=lambda kv: -kv[1]):
        tag = '   (sub)' if '/' in nm else ''
        print(f'  {nm:<24}{s*1000:>10.1f} ms  {100*s/total:>5.1f}%{tag}')
    print(f'  {"(unaccounted)":<24}{(total-acct)*1000:>10.1f} ms  {100*(total-acct)/total:>5.1f}%')
    print(f'  {"TOTAL":<24}{total*1000:>10.1f} ms')

def _dump_bench_json(path, meta):
    """Dump end-to-end + per-stage wall as JSON. TOTAL spans the first _tick
    (data_loading) to now (after output_write) — the full per-video pipeline,
    model load excluded. Top-level stages (no '/') sum toward the accounted
    total; 'dl/*' and 'fpt/*' are nested detail."""
    import json as _json
    out = dict(meta)
    # stage-mode accounted total (sync-inflated); absent in pure mode.
    if _run_t0[0] is not None:
        out['total_stage_s'] = _time_mod.perf_counter() - _run_t0[0]
        out['stages_ms'] = {k: v * 1000 for k, v in _stage_times.items()}
    with open(path, 'w') as fh:
        _json.dump(out, fh, indent=2)
    print(f'wrote bench json {path}')

pretrain_model_url = 'https://github.com/sczhou/ProPainter/releases/download/v0.1.0/'

def imwrite(img, file_path, params=None, auto_mkdir=True):
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    return cv2.imwrite(file_path, img, params)


# resize frames
def resize_frames(frames, size=None):
    """Resize a list of (H,W,3) uint8 RGB ndarrays to (W, H). Uses cv2.resize
    because PIL.Image.resize is single-threaded
    Python and ~10× slower than cv2 on uint8 RGB at 1080p->720p scale.

    Note: cv2.INTER_CUBIC uses a=-0.75 while PIL bicubic uses a=-0.5, so the
    resampled values differ by up to ~30 luma units on edges. End-to-end
    inpainting PSNR vs the PIL path stays > 40 dB which is visually
    indistinguishable; users who need bit-exact reproducibility against
    upstream ProPainter outputs should swap back to PIL.resize."""
    if size is not None:
        out_size = size
        process_size = (out_size[0]-out_size[0]%8, out_size[1]-out_size[1]%8)
        frames = _cv2_resize_pil_list(frames, process_size)
    else:
        out_size = (frames[0].shape[1], frames[0].shape[0])  # (W, H)
        process_size = (out_size[0]-out_size[0]%8, out_size[1]-out_size[1]%8)
        if not out_size == process_size:
            frames = _cv2_resize_pil_list(frames, process_size)

    return frames, process_size, out_size


def _cv2_resize_pil_list(frames, size):
    """Resize a list of (H,W,3) uint8 RGB ndarrays to (W, H) via cv2.resize and
    return ndarrays (no PIL round-trip). INTER_CUBIC to roughly match PIL's
    default bicubic (close but not bit-exact — see resize_frames docstring)."""
    target_w, target_h = size
    out = []
    for f in frames:
        arr = np.asarray(f)  # (H, W, 3) uint8 RGB (no-op if already ndarray)
        if arr.shape[1] == target_w and arr.shape[0] == target_h:
            out.append(arr)
            continue
        out.append(cv2.resize(arr, (target_w, target_h), interpolation=cv2.INTER_CUBIC))
    return out


def _read_video_rgb(path):
    """Return (RGB uint8 ndarray [T,H,W,3], fps). Falls back to imageio when
    torchvision.io.read_video is unavailable (removed in torchvision >= 0.26)."""
    if hasattr(torchvision.io, 'read_video'):
        vframes, _, info = torchvision.io.read_video(filename=path, pts_unit='sec')
        return vframes.numpy(), info.get('video_fps')
    import imageio.v3 as _iio
    arr = _iio.imread(path)
    meta = _iio.immeta(path)
    return arr, meta.get('fps')


#  read frames from video
def read_frame_from_videos(frame_root):
    if frame_root.endswith(('mp4', 'mov', 'avi', 'MP4', 'MOV', 'AVI')): # input video path
        video_name = os.path.basename(frame_root)[:-4]
        vframes, fps = _read_video_rgb(frame_root)  # RGB [T,H,W,3] uint8
        frames = list(vframes)  # list of (H,W,3) uint8 RGB ndarrays — no PIL round-trip
    else:
        video_name = os.path.basename(frame_root)
        frames = []
        fr_lst = sorted(os.listdir(frame_root))
        for fr in fr_lst:
            frame = cv2.imread(os.path.join(frame_root, fr))
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # (H,W,3) uint8 RGB
        fps = None
    size = (frames[0].shape[1], frames[0].shape[0])  # (W, H)

    return frames, fps, size, video_name


def binary_mask(mask, th=0.1):
    mask[mask>th] = 1
    mask[mask<=th] = 0
    return mask
  
  
# 4-connected (Manhattan / city-block) structuring element. scipy's default
# binary_dilation uses this, applied N times. cv2.dilate with this kernel and
# `iterations=N` produces bit-exact identical output (verified against
# scipy.ndimage.binary_dilation on real ProPainter masks: 0 pixel diff out of
# millions across all 200 frames of sample 2).
_CV2_CROSS3 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)


# read frame-wise masks
def read_mask(mpath, length, size, flow_mask_dilates=8, mask_dilates=5):
    """Vectorized mask reader. Replaces:
      - PIL.resize per-frame                with one cv2.resize-loop (5-10× faster)
      - scipy.ndimage.binary_dilation       with cv2.dilate (10-20× faster)
      - per-frame Image.fromarray wrap     with a single np stack at the end
    Returns the same (flow_masks, masks_dilated) list-of-PIL-images shape that
    downstream `to_tensors()` expects.
    """
    if mpath.endswith(('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')):
        # Single-image path: keep PIL flow (length is provided to broadcast).
        m = Image.open(mpath)
        if size is not None:
            m = m.resize(size, Image.NEAREST)
        m = np.array(m.convert('L'))
        masks_np = m[None]
    elif mpath.endswith(('mp4', 'mov', 'avi', 'MP4', 'MOV', 'AVI')):
        vframes, _ = _read_video_rgb(mpath)  # [T, H, W, 3] uint8 RGB
        T = vframes.shape[0]
        # Convert RGB→grayscale with the same ITU-R 601-2 weights as PIL's
        # convert('L'): Y = 0.299*R + 0.587*G + 0.114*B, rounded.
        # cv2.cvtColor with COLOR_RGB2GRAY uses the same coefficients.
        gray = np.empty((T,) + vframes.shape[1:3], dtype=np.uint8)
        for i in range(T):
            gray[i] = cv2.cvtColor(vframes[i], cv2.COLOR_RGB2GRAY)
        if size is not None:
            target_w, target_h = size
            masks_np = np.empty((T, target_h, target_w), dtype=np.uint8)
            for i in range(T):
                masks_np[i] = cv2.resize(gray[i], (target_w, target_h),
                                         interpolation=cv2.INTER_NEAREST)
        else:
            masks_np = gray
    else:
        mnames = sorted(os.listdir(mpath))
        masks_list = []
        for mp in mnames:
            m = Image.open(os.path.join(mpath, mp))
            if size is not None:
                m = m.resize(size, Image.NEAREST)
            masks_list.append(np.array(m.convert('L')))
        masks_np = np.stack(masks_list, axis=0)

    # Threshold to binary (mirrors binary_mask: th=0.1 -> any nonzero)
    binary = (masks_np > 0).astype(np.uint8)

    # Per-frame dilation via cv2.dilate with cross-3 kernel + iterations=N.
    # Bit-exact identical to scipy.ndimage.binary_dilation(iterations=N).
    if flow_mask_dilates > 0:
        flow_arr = np.empty_like(binary)
        for i in range(binary.shape[0]):
            flow_arr[i] = cv2.dilate(binary[i], _CV2_CROSS3, iterations=flow_mask_dilates)
    else:
        flow_arr = binary
    if mask_dilates > 0:
        dilated_arr = np.empty_like(binary)
        for i in range(binary.shape[0]):
            dilated_arr[i] = cv2.dilate(binary[i], _CV2_CROSS3, iterations=mask_dilates)
    else:
        dilated_arr = binary

    # Return ndarrays (no PIL round-trip): list of (H,W) uint8 0/255.
    flow_masks = list(flow_arr * 255)
    masks_dilated = list(dilated_arr * 255)

    if len(masks_dilated) == 1:
        flow_masks = flow_masks * length
        masks_dilated = masks_dilated * length

    return flow_masks, masks_dilated


def extrapolation(video_ori, scale):
    """Prepares the data for video outpainting.
    """
    nFrame = len(video_ori)
    imgW, imgH = video_ori[0].size

    # Defines new FOV.
    imgH_extr = int(scale[0] * imgH)
    imgW_extr = int(scale[1] * imgW)
    imgH_extr = imgH_extr - imgH_extr % 8
    imgW_extr = imgW_extr - imgW_extr % 8
    H_start = int((imgH_extr - imgH) / 2)
    W_start = int((imgW_extr - imgW) / 2)

    # Extrapolates the FOV for video.
    frames = []
    for v in video_ori:
        frame = np.zeros(((imgH_extr, imgW_extr, 3)), dtype=np.uint8)
        frame[H_start: H_start + imgH, W_start: W_start + imgW, :] = v
        frames.append(Image.fromarray(frame))

    # Generates the mask for missing region.
    masks_dilated = []
    flow_masks = []
    
    dilate_h = 4 if H_start > 10 else 0
    dilate_w = 4 if W_start > 10 else 0
    mask = np.ones(((imgH_extr, imgW_extr)), dtype=np.uint8)
    
    mask[H_start+dilate_h: H_start+imgH-dilate_h, 
         W_start+dilate_w: W_start+imgW-dilate_w] = 0
    flow_masks.append(Image.fromarray(mask * 255))

    mask[H_start: H_start+imgH, W_start: W_start+imgW] = 0
    masks_dilated.append(Image.fromarray(mask * 255))
  
    flow_masks = flow_masks * nFrame
    masks_dilated = masks_dilated * nFrame
    
    return frames, flow_masks, masks_dilated, (imgW_extr, imgH_extr)


def get_ref_index(mid_neighbor_id, neighbor_ids, length, ref_stride=10, ref_num=-1):
    ref_index = []
    if ref_num == -1:
        for i in range(0, length, ref_stride):
            if i not in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, mid_neighbor_id - ref_stride * (ref_num // 2))
        end_idx = min(length, mid_neighbor_id + ref_stride * (ref_num // 2))
        for i in range(start_idx, end_idx, ref_stride):
            if i not in neighbor_ids:
                if len(ref_index) > ref_num:
                    break
                ref_index.append(i)
    return ref_index



if __name__ == '__main__':
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = get_device()
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--video', type=str, default='inputs/object_removal/bmx-trees', help='Path of the input video or image folder.')
    parser.add_argument(
        '-m', '--mask', type=str, default='inputs/object_removal/bmx-trees_mask', help='Path of the mask(s) or mask folder.')
    parser.add_argument(
        '-o', '--output', type=str, default='results', help='Output folder. Default: results')
    parser.add_argument(
        "--resize_ratio", type=float, default=1.0, help='Resize scale for processing video.')
    parser.add_argument(
        '--height', type=int, default=-1, help='Height of the processing video.')
    parser.add_argument(
        '--width', type=int, default=-1, help='Width of the processing video.')
    parser.add_argument(
        '--mask_dilation', type=int, default=4, help='Mask dilation for video and flow masking.')
    parser.add_argument(
        "--ref_stride", type=int, default=10, help='Stride of global reference frames.')
    parser.add_argument(
        # 10 matches the training setting — this is the temporal receptive field of
        # the transformer/feat_prop (window half-width = neighbor_length//2), so
        # keep it faithful to how the model was trained. To trade speed for fidelity
        # don't enlarge this (it changes cross-frame semantics); instead enlarge
        # --window_stride, which keeps the trained window but skips redundant
        # overlapping windows (measured ~47% off the transformer stage).
        "--neighbor_length", type=int, default=10, help='Length of local neighboring frames (transformer temporal receptive field).')
    parser.add_argument(
        # Loop step for the feat-prop/transformer sliding window, decoupled from the
        # window size. Original behavior = neighbor_length//2 (windows overlap ~50%).
        # -1 keeps that. A larger stride runs fewer windows (less recompute of the
        # same frames) WITHOUT changing each window's trained temporal field, so it
        # speeds up the transformer stage while preserving cross-frame semantics.
        # Must be <= neighbor_length (else gaps between windows -> unwritten frames);
        # the loop also forces a final window flush against the video end so the tail
        # is always covered. Some overlap (stride < neighbor_length) is kept so the
        # 0.5-average still blends window seams.
        "--window_stride", type=int, default=-1,
        help='Sliding-window step for transformer stage (-1 = neighbor_length//2, the original). '
             'Larger = faster, fewer windows; must be <= neighbor_length.')
    parser.add_argument(
        "--subvideo_length", type=int, default=60, help='Length of sub-video for long video inference.')
    parser.add_argument(
        "--raft_iter", type=int, default=20, help='Iterations for RAFT inference.')
    parser.add_argument(
        '--mode', default='video_inpainting', choices=['video_inpainting', 'video_outpainting'], help="Modes: video_inpainting / video_outpainting")
    parser.add_argument(
        '--scale_h', type=float, default=1.0, help='Outpainting scale of height for video_outpainting mode.')
    parser.add_argument(
        '--scale_w', type=float, default=1.2, help='Outpainting scale of width for video_outpainting mode.')
    parser.add_argument(
        '--save_fps', type=int, default=24, help='Frame per second. Default: 24')
    parser.add_argument(
        '--save_frames', action='store_true', help='Save output frames. Default: False')
    parser.add_argument(
        '--save_masked_in', action='store_true',
        help='Save the green-overlay preview video (masked_in.mp4). Off by '
             'default — the overlay is debug-only and dominates data_loading '
             'wall time on long videos.')
    parser.add_argument(
        '--fp16', action='store_true', help='Use fp16 (half precision) during inference. Default: fp32 (single precision).')
    parser.add_argument(
        '--frames', type=int, default=-1, help='Number of frames to process. Default: -1 (all frames).')
    parser.add_argument(
        '--compile', action='store_true',
        help='Wrap selected submodules (see --compile-targets) with torch.compile.')
    parser.add_argument(
        '--compile-mode', type=str, default='default',
        choices=['default', 'reduce-overhead', 'max-autotune', 'max-autotune-no-cudagraphs'],
        help='Mode forwarded to torch.compile when --compile is set. Default: "default".')
    parser.add_argument(
        '--compile-fullgraph', action='store_true',
        help='Pass fullgraph=True to torch.compile. Fails on any graph break.')
    parser.add_argument(
        '--compile-targets', type=str, default='encoder,decoder',
        help='Comma-separated submodules to compile when --compile is set. '
             'Choices: encoder, decoder, transformers, flow_complete. '
             'Default "encoder,decoder" — NOTE: compiling "transformers" disables the '
             'fused-kernel fast path (PROPAINTER_FAST); the two are mutually-exclusive '
             'routes to the same block, and the fused path wins at real shapes.')
    parser.add_argument(
        '--bench_json', type=str, default='',
        help='If set, dump end-to-end + per-stage wall (CUDA-synced) as JSON to '
             'this path. Implies PROPAINTER_TIME_STAGES. TOTAL spans '
             'data_loading -> output_write (one full pipeline, model load excluded).')

    args = parser.parse_args()
    # --bench_json has two modes:
    #   pure  (default): zero added synchronize() — TOTAL is one perf_counter span
    #                    from data_loading start to output_write end. No per-stage.
    #   stage (PROPAINTER_TIME_STAGES=1): also records CUDA-synced per-stage wall
    #                    (adds ~6 sync points; TOTAL then ~1-2% high). Diagnostic.
    _bench_pure = bool(args.bench_json) and not _TIME_STAGES
    _e2e = [None, None]  # [start, end] perf_counter stamps, set with NO sync

    # Use fp16 precision during inference to reduce running memory cost
    use_half = True if args.fp16 else False 
    if device == torch.device('cpu'):
        use_half = False

    if args.bench_json and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    if args.bench_json:
        _e2e[0] = _time_mod.perf_counter()  # pure start stamp, NO synchronize
    _tick('data_loading')
    _tick('dl/read_video'); frames, fps, size, video_name = read_frame_from_videos(args.video); _tock('dl/read_video')
    if not args.width == -1 and not args.height == -1:
        size = (args.width, args.height)
    if not args.resize_ratio == 1.0:
        size = (int(args.resize_ratio * size[0]), int(args.resize_ratio * size[1]))
    
    if args.frames > 0:
        frames = frames[:args.frames]
    _tick('dl/resize'); frames, size, out_size = resize_frames(frames, size); _tock('dl/resize')
    
    fps = args.save_fps if fps is None else fps
    save_root = os.path.join(args.output, video_name)
    if not os.path.exists(save_root):
        os.makedirs(save_root, exist_ok=True)

    if args.mode == 'video_inpainting':
        frames_len = len(frames)
        _tick('dl/read_mask')
        flow_masks, masks_dilated = read_mask(args.mask, frames_len, size,
                                              flow_mask_dilates=args.mask_dilation,
                                              mask_dilates=args.mask_dilation)
        _tock('dl/read_mask')
        if args.frames > 0:
            flow_masks = flow_masks[:frames_len]
            masks_dilated = masks_dilated[:frames_len]
        w, h = size
    elif args.mode == 'video_outpainting':
        assert args.scale_h is not None and args.scale_w is not None, 'Please provide a outpainting scale (s_h, s_w).'
        frames = [Image.fromarray(f) for f in frames]  # extrapolation() operates on PIL
        frames, flow_masks, masks_dilated, size = extrapolation(frames, (args.scale_h, args.scale_w))
        w, h = size
    else:
        raise NotImplementedError
    
    # Build the green-overlay preview only when explicitly requested; it is
    # debug-only and skipping it shaves several seconds off data_loading at
    # 720p/200 frames.
    masked_frame_for_save = None
    if args.save_masked_in:
        _imgs = np.stack([np.asarray(f, dtype=np.uint8) for f in frames], axis=0)  # (T, H, W, 3)
        _msk = np.stack([np.asarray(m, dtype=np.uint8) for m in masks_dilated], axis=0)  # (T, H, W)
        _alpha = 0.6
        _mask3 = (_msk > 0)[..., None]  # (T, H, W, 1) bool broadcast
        # green(0,255,0): inside mask, fused = 0.4*img + 0.6*green
        _g_r = (_imgs[..., 0:1] * (1 - _alpha)).astype(np.uint8)
        _g_g = (_imgs[..., 1:2] * (1 - _alpha) + _alpha * 255).astype(np.uint8)
        _g_b = (_imgs[..., 2:3] * (1 - _alpha)).astype(np.uint8)
        _fused = np.concatenate([_g_r, _g_g, _g_b], axis=-1)
        masked_frame_for_save = list(np.where(_mask3, _fused, _imgs))
        del _imgs, _msk, _mask3, _g_r, _g_g, _g_b, _fused

    _tick('dl/to_tensor')
    # Vectorized PIL->tensor. The old path = to_tensors() = Compose([Stack,
    # ToTorchFormatTensor]): a big np.stack(axis=2) + permute().contiguous() + a
    # CPU .float().div(255) over 200x720p frames (~5.2s, 17% of the whole run; the
    # class even comments "this transpose takes 80% of the loading time"). Instead
    # stack once to uint8, move uint8 to GPU (6x less PCIe than float), and do the
    # permute/normalize on-device. Bit-exact: identical uint8->/255 (->*2-1) math,
    # IEEE division is device-agnostic.
    _frames_np = np.stack([np.asarray(f, dtype=np.uint8) for f in frames], axis=0)   # (L,H,W,3) RGB
    frames_inp = list(_frames_np)
    frames = (torch.from_numpy(_frames_np).to(device)
              .permute(0, 3, 1, 2).float().div_(255).mul_(2).sub_(1)).unsqueeze(0)   # (1,L,3,H,W) [-1,1]
    _fm_np = np.stack([np.asarray(m, dtype=np.uint8) for m in flow_masks], axis=0)    # (L,H,W) 0/255
    flow_masks = torch.from_numpy(_fm_np).to(device).float().div_(255).unsqueeze(1).unsqueeze(0)   # (1,L,1,H,W)
    _md_np = np.stack([np.asarray(m, dtype=np.uint8) for m in masks_dilated], axis=0)
    masks_dilated = torch.from_numpy(_md_np).to(device).float().div_(255).unsqueeze(1).unsqueeze(0)
    _tock('dl/to_tensor')
    _tock('data_loading')


    ##############################################
    # set up RAFT and flow competition model
    ##############################################
    ckpt_path = load_file_from_url(url=os.path.join(pretrain_model_url, 'raft-things.pth'), 
                                    model_dir='weights', progress=True, file_name=None)
    fix_raft = RAFT_bi(ckpt_path, device)
    
    ckpt_path = load_file_from_url(url=os.path.join(pretrain_model_url, 'recurrent_flow_completion.pth'), 
                                    model_dir='weights', progress=True, file_name=None)
    fix_flow_complete = RecurrentFlowCompleteNet(ckpt_path)
    for p in fix_flow_complete.parameters():
        p.requires_grad = False
    fix_flow_complete.to(device)
    fix_flow_complete.eval()


    ##############################################
    # set up ProPainter model
    ##############################################
    ckpt_path = load_file_from_url(url=os.path.join(pretrain_model_url, 'ProPainter.pth'), 
                                    model_dir='weights', progress=True, file_name=None)
    model = InpaintGenerator(model_path=ckpt_path).to(device)
    model.eval()

    if use_half:
        frames, flow_masks, masks_dilated = frames.half(), flow_masks.half(), masks_dilated.half()
        fix_flow_complete = fix_flow_complete.half()
        model = model.half()
        fix_raft = fix_raft.half()

    if args.compile:
        # dynamic=True so the varying batch (b*t, where t = local frames + a
        # variable number of reference frames per window) does NOT trigger a
        # per-window recompile — that churn would eat any fusion win. The conv
        # stacks (encoder/decoder/flow_complete) stay NCHW: channels_last
        # regressed +19-37% on sm_120 here, so do not combine the two.
        targets = {t.strip() for t in args.compile_targets.split(',') if t.strip()}
        compile_kwargs = dict(mode=args.compile_mode, fullgraph=args.compile_fullgraph,
                              dynamic=True)
        print(f'[compile] targets={sorted(targets)} torch.compile {compile_kwargs}')
        if 'encoder' in targets:
            model.encoder = torch.compile(model.encoder, **compile_kwargs)
        if 'decoder' in targets:
            model.decoder = torch.compile(model.decoder, **compile_kwargs)
        if 'transformers' in targets:
            # Compiling the transformer takes it OFF the fused-kernel fast path.
            model.transformers = torch.compile(model.transformers, **compile_kwargs)
        if 'flow_complete' in targets:
            fix_flow_complete = torch.compile(fix_flow_complete, **compile_kwargs)

    ##############################################
    # ProPainter inference
    ##############################################
    video_length = frames.size(1)
    print(f'\nProcessing: {video_name} [{video_length} frames]...')
    with torch.no_grad():
        # ---- compute flow ----
        _tick('flow_estimation')
        # Chunk RAFT over long videos to bound flow-estimation memory. The only
        # distinction that matters is width <= 640 (narrow → 96-frame clips fit)
        # vs wider (→ 64). The old >720 / >1280 / else branches all returned 64.
        # 64 is the max that fits at 720p/200f on a 32 GB card: RAFT's correlation
        # activations grow ~linearly with clip length, and flow_estimation is itself
        # a peak point (~24.5 GB allocated at clip=64); clip=80 OOMs. Do not raise.
        short_clip_len = 96 if frames.size(-1) <= 640 else 64

        if frames.size(1) > short_clip_len:
            gt_flows_f_list, gt_flows_b_list = [], []
            for f in range(0, video_length, short_clip_len):
                end_f = min(video_length, f + short_clip_len)
                if f == 0:
                    flows_f, flows_b = fix_raft(frames[:,f:end_f], iters=args.raft_iter)
                else:
                    flows_f, flows_b = fix_raft(frames[:,f-1:end_f], iters=args.raft_iter)
                
                gt_flows_f_list.append(flows_f)
                gt_flows_b_list.append(flows_b)
                
            gt_flows_f = torch.cat(gt_flows_f_list, dim=1)
            gt_flows_b = torch.cat(gt_flows_b_list, dim=1)
            gt_flows_bi = (gt_flows_f, gt_flows_b)
        else:
            gt_flows_bi = fix_raft(frames, iters=args.raft_iter)
        _tock('flow_estimation')
        del fix_raft
        torch.cuda.empty_cache()


        # ---- complete flow ----
        _tick('flow_completion')
        flow_length = gt_flows_bi[0].size(1)
        if flow_length > args.subvideo_length:
            print(f'Flow completion for long video, processing in sub-videos with length {args.subvideo_length}...')
            pred_flows_f, pred_flows_b = [], []
            pad_len = 5
            for f in range(0, flow_length, args.subvideo_length):
                s_f = max(0, f - pad_len)
                e_f = min(flow_length, f + args.subvideo_length + pad_len)
                pad_len_s = max(0, f) - s_f
                pad_len_e = e_f - min(flow_length, f + args.subvideo_length)
                pred_flows_bi_sub, _ = fix_flow_complete.forward_bidirect_flow(
                    (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]), 
                    flow_masks[:, s_f:e_f+1])
                pred_flows_bi_sub = fix_flow_complete.combine_flow(
                    (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]), 
                    pred_flows_bi_sub, 
                    flow_masks[:, s_f:e_f+1])

                pred_flows_f.append(pred_flows_bi_sub[0][:, pad_len_s:e_f-s_f-pad_len_e])
                pred_flows_b.append(pred_flows_bi_sub[1][:, pad_len_s:e_f-s_f-pad_len_e])
                torch.cuda.empty_cache()
                
            pred_flows_f = torch.cat(pred_flows_f, dim=1)
            pred_flows_b = torch.cat(pred_flows_b, dim=1)
            pred_flows_bi = (pred_flows_f, pred_flows_b)
        else:
            pred_flows_bi, _ = fix_flow_complete.forward_bidirect_flow(gt_flows_bi, flow_masks)
            pred_flows_bi = fix_flow_complete.combine_flow(gt_flows_bi, pred_flows_bi, flow_masks)
        _tock('flow_completion')
        del fix_flow_complete, gt_flows_bi
        torch.cuda.empty_cache()


        # ---- image propagation ----
        _tick('image_propagation')
        masked_frames = frames * (1 - masks_dilated)
        subvideo_length_img_prop = min(100, args.subvideo_length) # ensure a minimum of 100 frames for image propagation
        if video_length > subvideo_length_img_prop:
            updated_frames, updated_masks = [], []
            pad_len = 10
            for f in range(0, video_length, subvideo_length_img_prop):
                s_f = max(0, f - pad_len)
                e_f = min(video_length, f + subvideo_length_img_prop + pad_len)
                pad_len_s = max(0, f) - s_f
                pad_len_e = e_f - min(video_length, f + subvideo_length_img_prop)

                b, t, _, _, _ = masks_dilated[:, s_f:e_f].size()
                pred_flows_bi_sub = (pred_flows_bi[0][:, s_f:e_f-1], pred_flows_bi[1][:, s_f:e_f-1])
                prop_imgs_sub, updated_local_masks_sub = model.img_propagation(masked_frames[:, s_f:e_f], 
                                                                       pred_flows_bi_sub, 
                                                                       masks_dilated[:, s_f:e_f], 
                                                                       'nearest')
                updated_frames_sub = frames[:, s_f:e_f] * (1 - masks_dilated[:, s_f:e_f]) + \
                                    prop_imgs_sub.view(b, t, 3, h, w) * masks_dilated[:, s_f:e_f]
                updated_masks_sub = updated_local_masks_sub.view(b, t, 1, h, w)
                
                updated_frames.append(updated_frames_sub[:, pad_len_s:e_f-s_f-pad_len_e])
                updated_masks.append(updated_masks_sub[:, pad_len_s:e_f-s_f-pad_len_e])
                # Per-chunk empty_cache() removed: it dumped the allocator cache so the
                # next chunk's allocations all hit cudaMalloc (sync). Profiling showed
                # 244 cudaMalloc+cudaFree = ~6 s (81% of image_propagation) at 200f.
                # Same fix already applied to the main loop; memory stays bounded by chunking.
                # torch.cuda.empty_cache()

            updated_frames = torch.cat(updated_frames, dim=1)
            updated_masks = torch.cat(updated_masks, dim=1)
        else:
            b, t, _, _, _ = masks_dilated.size()
            prop_imgs, updated_local_masks = model.img_propagation(masked_frames, pred_flows_bi, masks_dilated, 'nearest')
            updated_frames = frames * (1 - masks_dilated) + prop_imgs.view(b, t, 3, h, w) * masks_dilated
            updated_masks = updated_local_masks.view(b, t, 1, h, w)
            # torch.cuda.empty_cache()  # removed (allocator-cache dump → cudaMalloc storm)
        _tock('image_propagation')
    # img_prop_module is parameter-free (~0 MB), so the old _offload_prop() CPU
    # round-trip saved no memory — removed. Instead, free the two full-length
    # inputs that are now dead: `frames` and `masked_frames` (each (1,T,3,H,W)
    # fp16, ~0.44 GB at 720p/80f). The transformer loop below only needs
    # updated_frames / updated_masks / masks_dilated / pred_flows_bi.
    del masked_frames, frames
    # One-shot empty_cache() at the image_prop -> transformer boundary. This is
    # the single biggest memory lever in the pipeline: it returns the fragmented
    # reserved pool left by image_propagation before the transformer phase.
    # Measured at 720p/80f (torch.cuda peak counters, deterministic):
    #   with:    max_allocated 14.1 GB / max_reserved 16.4 GB
    #   without: max_allocated 23.4 GB / max_reserved 30.5 GB  (near OOM on 32 GB)
    # Costs no measurable time — it's one call at a stage boundary, NOT in a loop
    # (the per-chunk/per-window empty_cache() calls removed elsewhere are the
    # dangerous ones: they trigger a cudaMalloc storm). Do not remove this.
    torch.cuda.empty_cache()

    ori_frames = frames_inp
    comp_frames = [None] * video_length

    neighbor_stride = args.neighbor_length // 2
    if video_length > args.subvideo_length:
        ref_num = args.subvideo_length // args.ref_stride
    else:
        ref_num = -1

    # Window half-width is the trained temporal field; the loop step is decoupled
    # (--window_stride) so we can skip redundant overlapping windows without
    # changing per-window semantics. Default step = neighbor_stride (original).
    win_step = args.window_stride if args.window_stride > 0 else neighbor_stride
    win_step = max(1, min(win_step, args.neighbor_length))  # <= neighbor_length: keep >=1-frame overlap, no gaps
    # Explicit center list so we can guarantee tail coverage: if the last window
    # [c-half, c+half] doesn't reach the final frame, append one more center placed
    # so its window ends at video_length-1. Without this, frames past the last
    # center are never written and read back as uninitialized garbage.
    window_centers = list(range(0, video_length, win_step))
    last_covered = (window_centers[-1] + neighbor_stride) if window_centers else -1
    if last_covered < video_length - 1:
        tail_center = max(0, video_length - 1 - neighbor_stride)
        if tail_center != window_centers[-1]:
            window_centers.append(tail_center)

    # GPU-resident compositing (replaces the per-window float D2H + numpy blend).
    # Originals as exact uint8 on device: ori_frames[idx] is HWC uint8 -> CHW uint8.
    # comp_gpu holds the running composite (uint8, CHW); a per-frame write count
    # reproduces the original sequential 0.5-average semantics bit-exactly.
    ori_u8 = torch.from_numpy(np.stack(ori_frames, axis=0)).to(device).permute(0, 3, 1, 2).contiguous()  # [T,3,H,W] uint8
    comp_gpu = torch.empty((video_length, 3, h, w), dtype=torch.uint8, device=device)
    comp_written = [False] * video_length
    # Pinned host staging for a single, overlappable uint8 D2H at the end.
    comp_host = torch.empty((video_length, h, w, 3), dtype=torch.uint8, pin_memory=True)
    _d2h_stream = torch.cuda.Stream()

    # ---- feature propagation + transformer ----
    _tick('feat_prop_transformer')
    _nsys_active = False
    for _it, f in enumerate(tqdm(window_centers)):
        if _NSYS_CAPTURE_ITER >= 0 and _it == _NSYS_CAPTURE_ITER:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            torch.cuda.nvtx.range_push(f'capture/iter{_it}')
            _nsys_active = True
        with _nvtx('pp/fpt/indexing'):
            neighbor_ids = [
                i for i in range(max(0, f - neighbor_stride),
                                    min(video_length, f + neighbor_stride + 1))
            ]
            ref_ids = get_ref_index(f, neighbor_ids, video_length, args.ref_stride, ref_num)
            selected_imgs = updated_frames[:, neighbor_ids + ref_ids, :, :, :]
            selected_masks = masks_dilated[:, neighbor_ids + ref_ids, :, :, :]
            selected_update_masks = updated_masks[:, neighbor_ids + ref_ids, :, :, :]
            selected_pred_flows_bi = (pred_flows_bi[0][:, neighbor_ids[:-1], :, :, :], pred_flows_bi[1][:, neighbor_ids[:-1], :, :, :])

        with torch.no_grad():
            # 1.0 indicates mask
            l_t = len(neighbor_ids)

            with _nvtx('pp/fpt/model_fwd'):
                # pred_img = selected_imgs # results of image propagation
                pred_img = model(selected_imgs, selected_pred_flows_bi, selected_masks, selected_update_masks, l_t)
                pred_img = pred_img.view(-1, 3, h, w)
                pred_img = (pred_img + 1) / 2

            # The per-window float D2H is gone: the blend now stays on device and
            # only a single uint8 D2H runs after the loop. The range is kept (now
            # near-empty) so the nsys hierarchy stays comparable to the baseline.
            with _nvtx('pp/fpt/d2h'):
                pass

            # GPU compositing. Equivalent to the old numpy path: trunc(pred*255)
            # inside the mask, exact original uint8 outside, then a running
            # 0.5-average (fp32) with uint8 truncation between overlapping windows.
            # Fully vectorized across the window's neighbour frames (one set of
            # kernels per window instead of one per frame).
            with _nvtx('pp/fpt/cpu_post'):
                idx_t = torch.as_tensor(neighbor_ids, device=device)            # [l_t]
                # *255 in fp16 to match (pred_img.cpu().numpy()*255) dtype, then trunc.
                pred_u8 = (pred_img[:l_t] * 255).clamp_(0, 255).to(torch.uint8)  # [l_t,3,h,w]
                mask_b = masks_dilated[0, neighbor_ids].to(torch.bool)           # [l_t,1,h,w]
                blended = torch.where(mask_b, pred_u8, ori_u8[idx_t])            # [l_t,3,h,w] uint8
                written = torch.as_tensor([comp_written[j] for j in neighbor_ids],
                                          device=device).view(l_t, 1, 1, 1)
                prev = comp_gpu[idx_t]                                           # [l_t,3,h,w] uint8
                averaged = (prev.float() * 0.5 + blended.float() * 0.5).clamp_(0, 255).to(torch.uint8)
                comp_gpu[idx_t] = torch.where(written, averaged, blended)
                for j in neighbor_ids:
                    comp_written[j] = True

        if _nsys_active:
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
            torch.cuda.cudart().cudaProfilerStop()
            _nsys_active = False
        # Per-iteration empty_cache() removed: it cost ~30-50s over the loop
        # (device sync + cudaFree of all cached blocks, re-malloc'd next iter)
        # with no peak-memory benefit. See bench_fork.py for the A/B.
    _tock('feat_prop_transformer')

    # Single uint8 D2H (1 byte/elem vs the old per-window float transfer, ~6x less
    # volume overall) on a side stream with pinned memory so it can overlap. Then
    # materialize the comp_frames list (HWC uint8) the rest of the pipeline expects.
    with _nvtx('pp/fpt/d2h_final'):
        # The side stream must observe all comp_gpu writes (issued on the default
        # stream during the loop) before copying, else it could race ahead.
        _d2h_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(_d2h_stream):
            comp_host.copy_(comp_gpu.permute(0, 2, 3, 1), non_blocking=True)
        _d2h_stream.synchronize()
        comp_np = comp_host.numpy()
        comp_frames = [comp_np[idx] for idx in range(video_length)]

    # Optional parity dump: raw uint8 comp_frames (pre-resize/codec) for A/B checks.
    _dump_path = os.environ.get('PROPAINTER_DUMP_COMP', '')
    if _dump_path:
        np.save(_dump_path, np.stack(comp_frames, axis=0))
        print(f'[dump] wrote comp_frames stack -> {_dump_path}')

    # save each frame
    if args.save_frames:
        for idx in range(video_length):
            f = comp_frames[idx]
            f = cv2.resize(f, out_size, interpolation = cv2.INTER_CUBIC)
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            img_save_root = os.path.join(save_root, 'frames', str(idx).zfill(4)+'.png')
            imwrite(f, img_save_root)
                    

    # if args.mode == 'video_outpainting':
    #     comp_frames = [i[10:-10,10:-10] for i in comp_frames]
    #     masked_frame_for_save = [i[10:-10,10:-10] for i in masked_frame_for_save]
    
    # save videos frame
    _tick('output_write')
    comp_frames = [cv2.resize(f, out_size) for f in comp_frames]
    imageio.mimwrite(os.path.join(save_root, 'inpaint_out.mp4'), comp_frames, fps=fps, quality=7)
    if masked_frame_for_save is not None:
        masked_frame_for_save = [cv2.resize(f, out_size) for f in masked_frame_for_save]
        imageio.mimwrite(os.path.join(save_root, 'masked_in.mp4'), masked_frame_for_save, fps=fps, quality=7)
    _tock('output_write')
    # mimwrite forces the final D2H to complete, so the GPU is drained here; the
    # pure end stamp needs no extra synchronize.
    if args.bench_json:
        _e2e[1] = _time_mod.perf_counter()

    print(f'\nAll results are saved in {save_root}')
    _print_stage_times()
    if args.bench_json:
        peak_gb = (torch.cuda.max_memory_allocated() / 1024**3
                   if torch.cuda.is_available() else 0.0)
        total_pure = (_e2e[1] - _e2e[0]) if (_e2e[0] and _e2e[1]) else None
        _dump_bench_json(args.bench_json, {
            'video': video_name, 'w': w, 'h': h, 'frames': video_length,
            'half': use_half, 'fast': os.environ.get('PROPAINTER_FAST', '1'),
            'peak_gpu_gb': peak_gb,
            'total_pure_s': total_pure,  # zero-sync end-to-end (data_load->write)
            'mode': 'stage' if _TIME_STAGES else 'pure',
            'throughput_fps': (video_length / total_pure) if total_pure else None,
        })

    torch.cuda.empty_cache()