#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ProPainter Performance Benchmark Script
Comprehensive performance testing and analysis for ProPainter model
"""

import os
import sys
import json
import csv
import time
import argparse
from datetime import datetime
import torch
import numpy as np
from PIL import Image
import scipy.ndimage

# Import ProPainter modules
from model.modules.flow_comp_raft import RAFT_bi
from model.recurrent_flow_completion import RecurrentFlowCompleteNet
from model.propainter import InpaintGenerator
from model.misc import get_device
from core.utils import to_tensors
from utils.download_util import load_file_from_url

pretrain_model_url = 'https://github.com/sczhou/ProPainter/releases/download/v0.1.0/'


class ModelProfiler:
    """Model parameter and performance profiler"""
    def __init__(self):
        self.timings = {}
        self.memory = {}
        self.param_counts = {}

    def count_parameters(self, model, name):
        """Count model parameters"""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.param_counts[name] = {
            'total': total,
            'trainable': trainable,
            'total_M': total / 1e6,
            'trainable_M': trainable / 1e6
        }
        print(f"[{name}] Total: {total/1e6:.2f}M params, Trainable: {trainable/1e6:.2f}M params")

    def tick(self, stage):
        """Start timing"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.timings[stage] = {'start': time.perf_counter()}

    def tock(self, stage):
        """End timing and record memory"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.timings[stage]['start']
        self.timings[stage]['elapsed'] = elapsed

        if torch.cuda.is_available():
            self.timings[stage]['memory'] = {
                'peak_gb': torch.cuda.max_memory_allocated() / 1024**3,
                'current_gb': torch.cuda.memory_allocated() / 1024**3,
                'reserved_gb': torch.cuda.memory_reserved() / 1024**3
            }
        print(f"[{stage}] Time: {elapsed:.2f}s, Peak Memory: {self.timings[stage].get('memory', {}).get('peak_gb', 0):.2f}GB")

    def get_summary(self):
        """Get summary of all timings and memory"""
        total_time = sum(t.get('elapsed', 0) for t in self.timings.values())
        max_memory = max((t.get('memory', {}).get('peak_gb', 0) for t in self.timings.values()), default=0)
        return {
            'total_time': total_time,
            'max_memory': max_memory,
            'stages': self.timings,
            'parameters': self.param_counts
        }


class BenchmarkRunner:
    """Benchmark executor"""
    def __init__(self, config):
        self.config = config
        self.profiler = ModelProfiler()
        self.device = get_device()
        self.results = {}

    def setup_models(self):
        """Initialize models and count parameters"""
        print(f"\n{'='*60}")
        print(f"Setting up models for: {self.config['name']}")
        print(f"{'='*60}")

        # Initialize RAFT
        ckpt_path = load_file_from_url(
            url=os.path.join(pretrain_model_url, 'raft-things.pth'),
            model_dir='weights', progress=True, file_name=None
        )
        self.fix_raft = RAFT_bi(ckpt_path, self.device)
        self.profiler.count_parameters(self.fix_raft, 'RAFT_bi')

        # Initialize Flow Completion
        ckpt_path = load_file_from_url(
            url=os.path.join(pretrain_model_url, 'recurrent_flow_completion.pth'),
            model_dir='weights', progress=True, file_name=None
        )
        self.fix_flow_complete = RecurrentFlowCompleteNet(ckpt_path)
        for p in self.fix_flow_complete.parameters():
            p.requires_grad = False
        self.fix_flow_complete.to(self.device)
        self.fix_flow_complete.eval()
        self.profiler.count_parameters(self.fix_flow_complete, 'RecurrentFlowCompleteNet')

        # Initialize ProPainter
        ckpt_path = load_file_from_url(
            url=os.path.join(pretrain_model_url, 'ProPainter.pth'),
            model_dir='weights', progress=True, file_name=None
        )
        self.model = InpaintGenerator(model_path=ckpt_path).to(self.device)
        self.model.eval()
        self.profiler.count_parameters(self.model, 'InpaintGenerator')

        # Apply fp16 if needed
        if self.config['fp16']:
            print("Converting models to FP16...")
            self.fix_flow_complete = self.fix_flow_complete.half()
            self.model = self.model.half()

        print(f"{'='*60}\n")

    def run_inference(self):
        """Run inference with profiling"""
        import torchvision

        print(f"Running inference: {self.config['name']}")
        print(f"Resolution: {self.config['width']}x{self.config['height']}, Frames: {self.config['frames']}")

        # Load data
        self.profiler.tick('data_loading')

        # Read video
        video_path = self.config['video']
        if video_path.endswith(('mp4', 'mov', 'avi', 'MP4', 'MOV', 'AVI')):
            vframes, aframes, info = torchvision.io.read_video(filename=video_path, pts_unit='sec')
            frames = list(vframes.numpy())
            frames = [Image.fromarray(f) for f in frames]
            fps = info['video_fps']
        else:
            raise ValueError("Video file not found or unsupported format")

        # Limit frames
        if self.config['frames'] > 0:
            frames = frames[:self.config['frames']]

        # Resize frames
        size = (self.config['width'], self.config['height'])
        process_size = (size[0] - size[0] % 8, size[1] - size[1] % 8)
        frames = [f.resize(process_size) for f in frames]
        w, h = process_size

        # Read masks
        mask_path = self.config['mask']
        if mask_path.endswith(('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')):
            mask_img = Image.open(mask_path).resize(process_size, Image.NEAREST)
            mask_img = np.array(mask_img.convert('L'))
            mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=4).astype(np.uint8)
            masks_dilated = [Image.fromarray(mask_img * 255)] * len(frames)
            flow_masks = [Image.fromarray(mask_img * 255)] * len(frames)

        # Convert to tensors
        frames_inp = [np.array(f).astype(np.uint8) for f in frames]
        frames = to_tensors()(frames).unsqueeze(0) * 2 - 1
        flow_masks = to_tensors()(flow_masks).unsqueeze(0)
        masks_dilated = to_tensors()(masks_dilated).unsqueeze(0)
        frames = frames.to(self.device)
        flow_masks = flow_masks.to(self.device)
        masks_dilated = masks_dilated.to(self.device)

        video_length = frames.size(1)

        self.profiler.tock('data_loading')

        with torch.no_grad():
            # Flow estimation
            self.profiler.tick('flow_estimation')

            if frames.size(-1) <= 640:
                short_clip_len = 12
            elif frames.size(-1) <= 720:
                short_clip_len = 8
            elif frames.size(-1) <= 1280:
                short_clip_len = 4
            else:
                short_clip_len = 2

            if frames.size(1) > short_clip_len:
                gt_flows_f_list, gt_flows_b_list = [], []
                for f in range(0, video_length, short_clip_len):
                    end_f = min(video_length, f + short_clip_len)
                    if f == 0:
                        flows_f, flows_b = self.fix_raft(frames[:, f:end_f], iters=20)
                    else:
                        flows_f, flows_b = self.fix_raft(frames[:, f-1:end_f], iters=20)
                    gt_flows_f_list.append(flows_f)
                    gt_flows_b_list.append(flows_b)
                    torch.cuda.empty_cache()
                gt_flows_f = torch.cat(gt_flows_f_list, dim=1)
                gt_flows_b = torch.cat(gt_flows_b_list, dim=1)
                gt_flows_bi = (gt_flows_f, gt_flows_b)
            else:
                gt_flows_bi = self.fix_raft(frames, iters=20)
                torch.cuda.empty_cache()

            self.profiler.tock('flow_estimation')

            # Apply fp16 to data if needed
            if self.config['fp16']:
                frames = frames.half()
                flow_masks = flow_masks.half()
                masks_dilated = masks_dilated.half()
                gt_flows_bi = (gt_flows_bi[0].half(), gt_flows_bi[1].half())

            # Flow completion
            self.profiler.tick('flow_completion')

            flow_length = gt_flows_bi[0].size(1)
            subvideo_length = self.config['subvideo_length']

            if flow_length > subvideo_length:
                pred_flows_f, pred_flows_b = [], []
                pad_len = 5
                for f in range(0, flow_length, subvideo_length):
                    s_f = max(0, f - pad_len)
                    e_f = min(flow_length, f + subvideo_length + pad_len)
                    pad_len_s = max(0, f) - s_f
                    pad_len_e = e_f - min(flow_length, f + subvideo_length)

                    pred_flows_bi_sub, _ = self.fix_flow_complete.forward_bidirect_flow(
                        (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]),
                        flow_masks[:, s_f:e_f+1]
                    )
                    pred_flows_bi_sub = self.fix_flow_complete.combine_flow(
                        (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]),
                        pred_flows_bi_sub,
                        flow_masks[:, s_f:e_f+1]
                    )
                    pred_flows_f.append(pred_flows_bi_sub[0][:, pad_len_s:e_f-s_f-pad_len_e])
                    pred_flows_b.append(pred_flows_bi_sub[1][:, pad_len_s:e_f-s_f-pad_len_e])
                    torch.cuda.empty_cache()
                pred_flows_f = torch.cat(pred_flows_f, dim=1)
                pred_flows_b = torch.cat(pred_flows_b, dim=1)
                pred_flows_bi = (pred_flows_f, pred_flows_b)
            else:
                pred_flows_bi, _ = self.fix_flow_complete.forward_bidirect_flow(gt_flows_bi, flow_masks)
                pred_flows_bi = self.fix_flow_complete.combine_flow(gt_flows_bi, pred_flows_bi, flow_masks)
                torch.cuda.empty_cache()

            self.profiler.tock('flow_completion')

            # Image propagation
            self.profiler.tick('image_propagation')

            masked_frames = frames * (1 - masks_dilated)
            subvideo_length_img_prop = min(100, subvideo_length)

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
                    prop_imgs_sub, updated_local_masks_sub = self.model.img_propagation(
                        masked_frames[:, s_f:e_f],
                        pred_flows_bi_sub,
                        masks_dilated[:, s_f:e_f],
                        'nearest'
                    )
                    updated_frames_sub = frames[:, s_f:e_f] * (1 - masks_dilated[:, s_f:e_f]) + \
                                        prop_imgs_sub.view(b, t, 3, h, w) * masks_dilated[:, s_f:e_f]
                    updated_masks_sub = updated_local_masks_sub.view(b, t, 1, h, w)

                    updated_frames.append(updated_frames_sub[:, pad_len_s:e_f-s_f-pad_len_e])
                    updated_masks.append(updated_masks_sub[:, pad_len_s:e_f-s_f-pad_len_e])
                    torch.cuda.empty_cache()
                updated_frames = torch.cat(updated_frames, dim=1)
                updated_masks = torch.cat(updated_masks, dim=1)
            else:
                b, t, _, _, _ = masks_dilated.size()
                prop_imgs, updated_local_masks = self.model.img_propagation(
                    masked_frames, pred_flows_bi, masks_dilated, 'nearest'
                )
                updated_frames = frames * (1 - masks_dilated) + prop_imgs.view(b, t, 3, h, w) * masks_dilated
                updated_masks = updated_local_masks.view(b, t, 1, h, w)
                torch.cuda.empty_cache()

            self.profiler.tock('image_propagation')

            # Feature propagation + transformer
            self.profiler.tick('feat_prop_transformer')

            neighbor_stride = self.config['neighbor_length'] // 2
            ref_stride = self.config['ref_stride']

            if video_length > subvideo_length:
                ref_num = subvideo_length // ref_stride
            else:
                ref_num = -1

            # Just run one iteration for benchmarking
            neighbor_ids = list(range(0, min(video_length, self.config['neighbor_length'])))
            ref_ids = [i for i in range(0, video_length, ref_stride) if i not in neighbor_ids][:ref_num] if ref_num > 0 else []

            selected_imgs = updated_frames[:, neighbor_ids + ref_ids, :, :, :]
            selected_masks = masks_dilated[:, neighbor_ids + ref_ids, :, :, :]
            selected_update_masks = updated_masks[:, neighbor_ids + ref_ids, :, :, :]
            selected_pred_flows_bi = (
                pred_flows_bi[0][:, neighbor_ids[:-1], :, :, :],
                pred_flows_bi[1][:, neighbor_ids[:-1], :, :, :]
            )

            l_t = len(neighbor_ids)
            pred_img = self.model(selected_imgs, selected_pred_flows_bi, selected_masks, selected_update_masks, l_t)
            torch.cuda.empty_cache()

            self.profiler.tock('feat_prop_transformer')

        # Store results
        self.results = self.profiler.get_summary()
        self.results['config'] = self.config
        self.results['video_length'] = video_length
        self.results['resolution'] = f"{w}x{h}"
        self.results['fps'] = video_length / self.results['total_time'] if self.results['total_time'] > 0 else 0

        print(f"\nCompleted: {self.config['name']}")
        print(f"Total Time: {self.results['total_time']:.2f}s")
        print(f"FPS: {self.results['fps']:.3f}")
        print(f"Peak Memory: {self.results['max_memory']:.2f}GB\n")

    def generate_reports(self, output_dir):
        """Generate reports in multiple formats"""
        os.makedirs(output_dir, exist_ok=True)

        # Generate JSON report
        json_path = os.path.join(output_dir, f"{self.config['name']}_results.json")
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"JSON report saved to: {json_path}")

        return self.results


def generate_comparison_report(results_list, output_dir):
    """Generate comparison report for multiple configurations"""
    os.makedirs(output_dir, exist_ok=True)

    # Generate Markdown report
    md_path = os.path.join(output_dir, 'benchmark_report.md')
    with open(md_path, 'w') as f:
        f.write("# ProPainter Performance Benchmark Report\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # System info
        f.write("## System Information\n\n")
        if torch.cuda.is_available():
            f.write(f"- **GPU:** {torch.cuda.get_device_name(0)}\n")
            f.write(f"- **CUDA Version:** {torch.version.cuda}\n")
        f.write(f"- **PyTorch Version:** {torch.__version__}\n\n")

        # Model parameters
        if results_list:
            f.write("## Model Parameters\n\n")
            f.write("| Model | Total Params (M) | Trainable Params (M) |\n")
            f.write("|-------|------------------|----------------------|\n")
            params = results_list[0]['parameters']
            total_params = 0
            total_trainable = 0
            for name, counts in params.items():
                f.write(f"| {name} | {counts['total_M']:.2f} | {counts['trainable_M']:.2f} |\n")
                total_params += counts['total_M']
                total_trainable += counts['trainable_M']
            f.write(f"| **Total** | **{total_params:.2f}** | **{total_trainable:.2f}** |\n\n")

        # Performance comparison
        if len(results_list) >= 2:
            f.write("## Performance Comparison: FP32 vs FP16\n\n")

            fp32_results = next((r for r in results_list if not r['config']['fp16']), None)
            fp16_results = next((r for r in results_list if r['config']['fp16']), None)

            if fp32_results and fp16_results:
                f.write("### Execution Time (seconds)\n\n")
                f.write("| Stage | FP32 | FP16 | Speedup |\n")
                f.write("|-------|------|------|---------||\n")

                stages = ['data_loading', 'flow_estimation', 'flow_completion', 'image_propagation', 'feat_prop_transformer']
                stage_names = ['Data Loading', 'Flow Estimation', 'Flow Completion', 'Image Propagation', 'Feat Prop + Transformer']

                for stage, name in zip(stages, stage_names):
                    fp32_time = fp32_results['stages'].get(stage, {}).get('elapsed', 0)
                    fp16_time = fp16_results['stages'].get(stage, {}).get('elapsed', 0)
                    speedup = fp32_time / fp16_time if fp16_time > 0 else 0
                    f.write(f"| {name} | {fp32_time:.2f} | {fp16_time:.2f} | {speedup:.2f}x |\n")

                total_speedup = fp32_results['total_time'] / fp16_results['total_time'] if fp16_results['total_time'] > 0 else 0
                f.write(f"| **Total** | **{fp32_results['total_time']:.2f}** | **{fp16_results['total_time']:.2f}** | **{total_speedup:.2f}x** |\n\n")

                f.write("### Memory Usage (GB)\n\n")
                f.write("| Stage | FP32 Peak | FP16 Peak | Reduction |\n")
                f.write("|-------|-----------|-----------|-----------||\n")

                for stage, name in zip(stages, stage_names):
                    fp32_mem = fp32_results['stages'].get(stage, {}).get('memory', {}).get('peak_gb', 0)
                    fp16_mem = fp16_results['stages'].get(stage, {}).get('memory', {}).get('peak_gb', 0)
                    reduction = ((fp32_mem - fp16_mem) / fp32_mem * 100) if fp32_mem > 0 else 0
                    f.write(f"| {name} | {fp32_mem:.2f} | {fp16_mem:.2f} | {reduction:.1f}% |\n")

                mem_reduction = ((fp32_results['max_memory'] - fp16_results['max_memory']) / fp32_results['max_memory'] * 100) if fp32_results['max_memory'] > 0 else 0
                f.write(f"| **Maximum** | **{fp32_results['max_memory']:.2f}** | **{fp16_results['max_memory']:.2f}** | **{mem_reduction:.1f}%** |\n\n")

                f.write("### Throughput\n\n")
                f.write("| Metric | FP32 | FP16 |\n")
                f.write("|--------|------|------|\n")
                f.write(f"| FPS | {fp32_results['fps']:.3f} | {fp16_results['fps']:.3f} |\n")
                f.write(f"| Seconds per frame | {1/fp32_results['fps']:.3f} | {1/fp16_results['fps']:.3f} |\n")
                f.write(f"| Total processing time | {fp32_results['total_time']:.2f}s | {fp16_results['total_time']:.2f}s |\n\n")

    print(f"Markdown report saved to: {md_path}")

    # Generate CSV report
    csv_path = os.path.join(output_dir, 'benchmark_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Config', 'Precision', 'Resolution', 'Frames', 'Total Time (s)', 'FPS', 'Peak Memory (GB)'])
        for result in results_list:
            writer.writerow([
                result['config']['name'],
                'FP16' if result['config']['fp16'] else 'FP32',
                result['resolution'],
                result['video_length'],
                f"{result['total_time']:.2f}",
                f"{result['fps']:.3f}",
                f"{result['max_memory']:.2f}"
            ])

    print(f"CSV report saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='ProPainter Performance Benchmark')
    parser.add_argument('--config', type=str, help='Test config name (fp32_720p_80frames or fp16_720p_80frames)')
    parser.add_argument('--all', action='store_true', help='Run all configs')
    parser.add_argument('--output-dir', default='results', help='Output directory')
    args = parser.parse_args()

    # Test configurations
    TEST_CONFIGS = [
        {
            'name': 'fp32_720p_80frames',
            'video': 'inputs/video_completion/running_car.mp4',
            'mask': 'inputs/video_completion/mask_square.png',
            'height': 720,
            'width': 1280,
            'frames': 80,
            'fp16': False,
            'subvideo_length': 80,
            'neighbor_length': 10,
            'ref_stride': 10
        },
        {
            'name': 'fp16_720p_80frames',
            'video': 'inputs/video_completion/running_car.mp4',
            'mask': 'inputs/video_completion/mask_square.png',
            'height': 720,
            'width': 1280,
            'frames': 80,
            'fp16': True,
            'subvideo_length': 80,
            'neighbor_length': 10,
            'ref_stride': 10
        }
    ]

    results_list = []

    if args.all:
        # Run all test configurations
        for config in TEST_CONFIGS:
            runner = BenchmarkRunner(config)
            runner.setup_models()
            runner.run_inference()
            results = runner.generate_reports(args.output_dir)
            results_list.append(results)
    elif args.config:
        # Run specific configuration
        config = next((c for c in TEST_CONFIGS if c['name'] == args.config), None)
        if config is None:
            print(f"Error: Config '{args.config}' not found")
            print(f"Available configs: {', '.join(c['name'] for c in TEST_CONFIGS)}")
            return
        runner = BenchmarkRunner(config)
        runner.setup_models()
        runner.run_inference()
        results = runner.generate_reports(args.output_dir)
        results_list.append(results)
    else:
        print("Please specify --config <name> or --all")
        print(f"Available configs: {', '.join(c['name'] for c in TEST_CONFIGS)}")
        return

    # Generate comparison report if multiple results
    if len(results_list) > 0:
        generate_comparison_report(results_list, args.output_dir)

    print("\nBenchmark completed successfully!")


if __name__ == '__main__':
    main()
