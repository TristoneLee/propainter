# -*- coding: utf-8 -*-
"""Stage-timed benchmark for the optimized FORK ProPainter.

Mirrors the fork's inference_propainter.py body (SEA_RAFT flow in fp16, larger
RAFT clip lengths, eager fp16 transformer) with CUDA-synced
timers, so it is directly comparable to bench_original.py in the ProPainter/
subdir. Run with PROPAINTER_FAST=0 PROPAINTER_FLEX_ATTN=0 for the clean eager
fp16 path (the fork's best end-to-end config at real shapes).
"""
import os
import time
import json
import argparse

import numpy as np
import torch

from inference_propainter import (
    read_frame_from_videos, read_mask, resize_frames, get_ref_index,
    pretrain_model_url,
)
from core.utils import to_tensors
from utils.download_util import load_file_from_url
from model.modules.flow_comp_raft import RAFT_bi
from model.recurrent_flow_completion import RecurrentFlowCompleteNet
from model.propainter import InpaintGenerator
from model.misc import get_device

import warnings
warnings.filterwarnings("ignore")


def sync():
    torch.cuda.synchronize()


def flush_l2(device):
    """Evict the GPU L2 cache by streaming a buffer larger than it through a
    write, so the timed region starts from a cold L2 (like a real first call).
    RTX 5090 (sm_120) L2 is ~96 MB; 128 MB of fp32 scratch comfortably exceeds
    it. Cheap (~a few hundred us) and freed immediately."""
    n = 128 * 1024 * 1024 // 4  # 128 MB of float32
    scratch = torch.empty(n, dtype=torch.float32, device=device)
    scratch.zero_()
    sync()
    del scratch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-i', '--video', type=str, default='inputs/object_removal/bmx-trees')
    p.add_argument('-m', '--mask', type=str, default='inputs/object_removal/bmx-trees_mask')
    p.add_argument('--height', type=int, default=-1)
    p.add_argument('--width', type=int, default=-1)
    p.add_argument('--mask_dilation', type=int, default=4)
    p.add_argument('--ref_stride', type=int, default=10)
    p.add_argument('--neighbor_length', type=int, default=10)
    p.add_argument('--subvideo_length', type=int, default=80)
    p.add_argument('--raft_iter', type=int, default=20)
    p.add_argument('--fp16', action='store_true')
    p.add_argument('--frames', type=int, default=-1,
                   help='Truncate to N frames (-1 = all). Truncates masks too.')
    p.add_argument('--short_clip_len', type=int, default=-1,
                   help='Override SEA_RAFT flow clip length (-1 = auto by res). '
                        'Lower it to fit 1080p flow on limited VRAM.')
    p.add_argument('--flush-l2', dest='flush_l2', action='store_true',
                   help='Evict GPU L2 cache right before the timed region.')
    p.add_argument('--compile', action='store_true',
                   help='torch.compile the submodules in --compile-targets.')
    p.add_argument('--compile-mode', type=str, default='default',
                   choices=['default', 'reduce-overhead', 'max-autotune',
                            'max-autotune-no-cudagraphs'])
    p.add_argument('--compile-targets', type=str, default='encoder,decoder',
                   help='Comma-separated: encoder,decoder,transformers,flow_complete. '
                        'Compiling transformers disables the fused-kernel fast path.')
    p.add_argument('--json_out', type=str, default='')
    args = p.parse_args()

    device = get_device()
    use_half = bool(args.fp16) and device != torch.device('cpu')
    st = {}

    # ---- preprocessing ----
    frames, fps, size, video_name = read_frame_from_videos(args.video)
    if args.width != -1 and args.height != -1:
        size = (args.width, args.height)
    if args.frames > 0:
        frames = frames[:args.frames]
    frames, size, out_size = resize_frames(frames, size)
    w, h = size
    frames_len = len(frames)
    flow_masks, masks_dilated = read_mask(args.mask, frames_len, size,
                                          flow_mask_dilates=args.mask_dilation,
                                          mask_dilates=args.mask_dilation)
    if args.frames > 0:
        flow_masks, masks_dilated = flow_masks[:frames_len], masks_dilated[:frames_len]
    # read_frame_from_videos/read_mask now return ndarrays (no PIL); vectorize the
    # tensor conversion the same way inference_propainter.py does (uint8 -> GPU -> normalize).
    _frames_np = np.stack([np.asarray(f, dtype=np.uint8) for f in frames], axis=0)   # (L,H,W,3)
    frames_inp = list(_frames_np)
    frames = (torch.from_numpy(_frames_np).to(device)
              .permute(0, 3, 1, 2).float().div_(255).mul_(2).sub_(1)).unsqueeze(0)
    _fm = np.stack([np.asarray(m, dtype=np.uint8) for m in flow_masks], axis=0)       # (L,H,W)
    flow_masks = torch.from_numpy(_fm).to(device).float().div_(255).unsqueeze(1).unsqueeze(0)
    _md = np.stack([np.asarray(m, dtype=np.uint8) for m in masks_dilated], axis=0)
    masks_dilated = torch.from_numpy(_md).to(device).float().div_(255).unsqueeze(1).unsqueeze(0)

    # ---- model setup ----
    t0 = time.perf_counter()
    ckpt = load_file_from_url(url=os.path.join(pretrain_model_url, 'raft-things.pth'),
                              model_dir='weights', progress=True, file_name=None)
    fix_raft = RAFT_bi(ckpt, device)
    ckpt = load_file_from_url(url=os.path.join(pretrain_model_url, 'recurrent_flow_completion.pth'),
                              model_dir='weights', progress=True, file_name=None)
    fix_flow_complete = RecurrentFlowCompleteNet(ckpt)
    for pp in fix_flow_complete.parameters():
        pp.requires_grad = False
    fix_flow_complete.to(device).eval()
    ckpt = load_file_from_url(url=os.path.join(pretrain_model_url, 'ProPainter.pth'),
                              model_dir='weights', progress=True, file_name=None)
    model = InpaintGenerator(model_path=ckpt).to(device).eval()
    if use_half:
        frames, flow_masks, masks_dilated = frames.half(), flow_masks.half(), masks_dilated.half()
        fix_flow_complete = fix_flow_complete.half()
        model = model.half()
        fix_raft = fix_raft.half()
    if args.compile:
        targets = {t.strip() for t in args.compile_targets.split(',') if t.strip()}
        ckw = dict(mode=args.compile_mode, dynamic=True)
        print(f'[compile] targets={sorted(targets)} {ckw}')
        if 'encoder' in targets:
            model.encoder = torch.compile(model.encoder, **ckw)
        if 'decoder' in targets:
            model.decoder = torch.compile(model.decoder, **ckw)
        if 'transformers' in targets:
            model.transformers = torch.compile(model.transformers, **ckw)
        if 'flow_complete' in targets:
            fix_flow_complete = torch.compile(fix_flow_complete, **ckw)
    sync()
    st['setup'] = time.perf_counter() - t0

    video_length = frames.size(1)
    print(f'Processing: {video_name} [{video_length} frames] {w}x{h} half={use_half}')
    torch.cuda.reset_peak_memory_stats()
    if args.flush_l2:
        flush_l2(device)
    sync()
    bench_t0 = time.perf_counter()

    with torch.no_grad():
        # ---- flow estimation (SEA_RAFT, fp16) ----
        sync(); t0 = time.perf_counter()
        if args.short_clip_len > 0:
            short_clip_len = args.short_clip_len
        elif frames.size(-1) <= 640:
            short_clip_len = 96
        else:
            short_clip_len = 64
        if frames.size(1) > short_clip_len:
            gt_flows_f_list, gt_flows_b_list = [], []
            for f in range(0, video_length, short_clip_len):
                end_f = min(video_length, f + short_clip_len)
                if f == 0:
                    flows_f, flows_b = fix_raft(frames[:, f:end_f], iters=args.raft_iter)
                else:
                    flows_f, flows_b = fix_raft(frames[:, f - 1:end_f], iters=args.raft_iter)
                gt_flows_f_list.append(flows_f)
                gt_flows_b_list.append(flows_b)
                torch.cuda.empty_cache()
            gt_flows_bi = (torch.cat(gt_flows_f_list, dim=1), torch.cat(gt_flows_b_list, dim=1))
        else:
            gt_flows_bi = fix_raft(frames, iters=args.raft_iter)
        sync(); st['flow_estimation'] = time.perf_counter() - t0
        del fix_raft
        torch.cuda.empty_cache()

        # ---- flow completion ----
        sync(); t0 = time.perf_counter()
        flow_length = gt_flows_bi[0].size(1)
        if flow_length > args.subvideo_length:
            pred_flows_f, pred_flows_b = [], []
            pad_len = 5
            for f in range(0, flow_length, args.subvideo_length):
                s_f = max(0, f - pad_len)
                e_f = min(flow_length, f + args.subvideo_length + pad_len)
                pad_len_s = max(0, f) - s_f
                pad_len_e = e_f - min(flow_length, f + args.subvideo_length)
                sub, _ = fix_flow_complete.forward_bidirect_flow(
                    (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]), flow_masks[:, s_f:e_f + 1])
                sub = fix_flow_complete.combine_flow(
                    (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]), sub, flow_masks[:, s_f:e_f + 1])
                pred_flows_f.append(sub[0][:, pad_len_s:e_f - s_f - pad_len_e])
                pred_flows_b.append(sub[1][:, pad_len_s:e_f - s_f - pad_len_e])
                torch.cuda.empty_cache()
            pred_flows_bi = (torch.cat(pred_flows_f, dim=1), torch.cat(pred_flows_b, dim=1))
        else:
            pred_flows_bi, _ = fix_flow_complete.forward_bidirect_flow(gt_flows_bi, flow_masks)
            pred_flows_bi = fix_flow_complete.combine_flow(gt_flows_bi, pred_flows_bi, flow_masks)
        sync(); st['flow_completion'] = time.perf_counter() - t0
        del fix_flow_complete, gt_flows_bi
        torch.cuda.empty_cache()

        # ---- image propagation ----
        sync(); t0 = time.perf_counter()
        masked_frames = frames * (1 - masks_dilated)
        subvideo_length_img_prop = min(100, args.subvideo_length)
        if video_length > subvideo_length_img_prop:
            updated_frames, updated_masks = [], []
            pad_len = 10
            for f in range(0, video_length, subvideo_length_img_prop):
                s_f = max(0, f - pad_len)
                e_f = min(video_length, f + subvideo_length_img_prop + pad_len)
                pad_len_s = max(0, f) - s_f
                pad_len_e = e_f - min(video_length, f + subvideo_length_img_prop)
                b, t, _, _, _ = masks_dilated[:, s_f:e_f].size()
                sub_flows = (pred_flows_bi[0][:, s_f:e_f - 1], pred_flows_bi[1][:, s_f:e_f - 1])
                prop_imgs_sub, upd_masks_sub = model.img_propagation(
                    masked_frames[:, s_f:e_f], sub_flows, masks_dilated[:, s_f:e_f], 'nearest')
                upd_frames_sub = frames[:, s_f:e_f] * (1 - masks_dilated[:, s_f:e_f]) + \
                    prop_imgs_sub.view(b, t, 3, h, w) * masks_dilated[:, s_f:e_f]
                upd_masks_sub = upd_masks_sub.view(b, t, 1, h, w)
                updated_frames.append(upd_frames_sub[:, pad_len_s:e_f - s_f - pad_len_e])
                updated_masks.append(upd_masks_sub[:, pad_len_s:e_f - s_f - pad_len_e])
                torch.cuda.empty_cache()
            updated_frames = torch.cat(updated_frames, dim=1)
            updated_masks = torch.cat(updated_masks, dim=1)
        else:
            b, t, _, _, _ = masks_dilated.size()
            prop_imgs, upd_local = model.img_propagation(masked_frames, pred_flows_bi, masks_dilated, 'nearest')
            updated_frames = frames * (1 - masks_dilated) + prop_imgs.view(b, t, 3, h, w) * masks_dilated
            updated_masks = upd_local.view(b, t, 1, h, w)
            torch.cuda.empty_cache()
        sync(); st['image_propagation'] = time.perf_counter() - t0

    torch.cuda.empty_cache()

    # ---- feature propagation + transformer ----
    ori_frames = frames_inp
    comp_frames = [None] * video_length
    neighbor_stride = args.neighbor_length // 2
    ref_num = args.subvideo_length // args.ref_stride if video_length > args.subvideo_length else -1

    sync(); t0 = time.perf_counter()
    n_iters = 0
    for f in range(0, video_length, neighbor_stride):
        n_iters += 1
        neighbor_ids = [i for i in range(max(0, f - neighbor_stride), min(video_length, f + neighbor_stride + 1))]
        ref_ids = get_ref_index(f, neighbor_ids, video_length, args.ref_stride, ref_num)
        selected_imgs = updated_frames[:, neighbor_ids + ref_ids, :, :, :]
        selected_masks = masks_dilated[:, neighbor_ids + ref_ids, :, :, :]
        selected_update_masks = updated_masks[:, neighbor_ids + ref_ids, :, :, :]
        selected_pred_flows_bi = (pred_flows_bi[0][:, neighbor_ids[:-1], :, :, :],
                                  pred_flows_bi[1][:, neighbor_ids[:-1], :, :, :])
        with torch.no_grad():
            l_t = len(neighbor_ids)
            pred_img = model(selected_imgs, selected_pred_flows_bi, selected_masks, selected_update_masks, l_t)
            pred_img = pred_img.view(-1, 3, h, w)
            pred_img = (pred_img + 1) / 2
            pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
            binary_masks = masks_dilated[0, neighbor_ids, :, :, :].cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8)
            for i in range(len(neighbor_ids)):
                idx = neighbor_ids[i]
                img = np.array(pred_img[i]).astype(np.uint8) * binary_masks[i] + ori_frames[idx] * (1 - binary_masks[i])
                if comp_frames[idx] is None:
                    comp_frames[idx] = img
                else:
                    comp_frames[idx] = comp_frames[idx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5
                comp_frames[idx] = comp_frames[idx].astype(np.uint8)
        # NOTE: a per-iteration torch.cuda.empty_cache() used to live here. It
        # cost ~30-50s (sync + cudaFree of all cached blocks, forcing cudaMalloc
        # again next iter) with zero peak-memory benefit. Removed.
    sync(); st['feat_prop_transformer'] = time.perf_counter() - t0

    total = time.perf_counter() - bench_t0
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3

    print("\n================ FORK ProPainter benchmark ================")
    print(f"FAST={os.environ.get('PROPAINTER_FAST','1')} FLEX={os.environ.get('PROPAINTER_FLEX_ATTN','0')}")
    print(f"input            : {video_name}  {w}x{h}  {video_length} frames  half={use_half}")
    print(f"transformer iters: {n_iters}")
    print(f"peak GPU mem     : {peak_gb:.2f} GB")
    print(f"{'stage':<24}{'ms':>12}{'%':>8}")
    inf_stages = ['flow_estimation', 'flow_completion', 'image_propagation', 'feat_prop_transformer']
    for k in inf_stages:
        print(f"{k:<24}{st[k]*1000:>12.1f}{100*st[k]/total:>8.1f}")
    print(f"{'-'*44}")
    print(f"{'TOTAL inference':<24}{total*1000:>12.1f}{100:>8.1f}")
    print(f"{'(setup, excluded)':<24}{st['setup']*1000:>12.1f}")
    print(f"throughput       : {video_length/total:.2f} frames/s")
    print("==========================================================")

    if args.json_out:
        out = {'video': video_name, 'w': w, 'h': h, 'frames': video_length, 'half': use_half,
               'fast': os.environ.get('PROPAINTER_FAST', '1'), 'flex': os.environ.get('PROPAINTER_FLEX_ATTN', '0'),
               'transformer_iters': n_iters, 'peak_gpu_gb': peak_gb, 'total_inference_s': total,
               'throughput_fps': video_length / total,
               'stages_ms': {k: st[k] * 1000 for k in inf_stages}, 'setup_s': st['setup']}
        with open(args.json_out, 'w') as fh:
            json.dump(out, fh, indent=2)
        print(f"wrote {args.json_out}")


if __name__ == '__main__':
    main()
