#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test Data Preparation Script for ProPainter Benchmark
Prepares 720p 80-frame test videos from existing data
"""

import os
import cv2
import numpy as np
from PIL import Image
import torchvision


def prepare_test_video(input_video, output_video, target_width=1280, target_height=720, max_frames=80):
    """
    Prepare test video with specific resolution and frame count

    Args:
        input_video: Path to input video
        output_video: Path to output video
        target_width: Target width
        target_height: Target height
        max_frames: Maximum number of frames
    """
    print(f"Preparing test video: {output_video}")
    print(f"Target resolution: {target_width}x{target_height}")
    print(f"Max frames: {max_frames}")

    # Read video
    vframes, aframes, info = torchvision.io.read_video(filename=input_video, pts_unit='sec')
    fps = info['video_fps']

    print(f"Original video: {vframes.shape[0]} frames, {vframes.shape[2]}x{vframes.shape[1]}, {fps} fps")

    # Limit frames
    frames = vframes[:max_frames]

    # Resize frames
    resized_frames = []
    for frame in frames:
        frame_np = frame.numpy()
        frame_resized = cv2.resize(frame_np, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        resized_frames.append(frame_resized)

    # Save video
    os.makedirs(os.path.dirname(output_video), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (target_width, target_height))

    for frame in resized_frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()

    print(f"Saved: {output_video} ({len(resized_frames)} frames)")
    return len(resized_frames)


def prepare_test_mask(input_mask, output_mask, target_width=1280, target_height=720):
    """
    Prepare test mask with specific resolution

    Args:
        input_mask: Path to input mask
        output_mask: Path to output mask
        target_width: Target width
        target_height: Target height
    """
    print(f"Preparing test mask: {output_mask}")

    # Read mask
    mask = Image.open(input_mask).convert('L')

    # Resize mask
    mask_resized = mask.resize((target_width, target_height), Image.NEAREST)

    # Save mask
    os.makedirs(os.path.dirname(output_mask), exist_ok=True)
    mask_resized.save(output_mask)

    print(f"Saved: {output_mask} ({target_width}x{target_height})")


def verify_test_data(video_path, mask_path, expected_frames=80):
    """
    Verify test data is correctly prepared

    Args:
        video_path: Path to video
        mask_path: Path to mask
        expected_frames: Expected number of frames
    """
    print(f"\nVerifying test data...")

    # Check video
    if not os.path.exists(video_path):
        print(f"ERROR: Video not found: {video_path}")
        return False

    vframes, _, info = torchvision.io.read_video(filename=video_path, pts_unit='sec')
    print(f"Video: {vframes.shape[0]} frames, {vframes.shape[2]}x{vframes.shape[1]}")

    if vframes.shape[0] < expected_frames:
        print(f"WARNING: Video has fewer frames than expected ({vframes.shape[0]} < {expected_frames})")

    # Check mask
    if not os.path.exists(mask_path):
        print(f"ERROR: Mask not found: {mask_path}")
        return False

    mask = Image.open(mask_path)
    print(f"Mask: {mask.size[0]}x{mask.size[1]}")

    if mask.size != (vframes.shape[2], vframes.shape[1]):
        print(f"ERROR: Mask size doesn't match video size")
        return False

    print("Verification passed!")
    return True


def main():
    # Prepare 720p test data
    input_video = 'inputs/video_completion/running_car.mp4'
    input_mask = 'inputs/video_completion/mask_square.png'

    output_dir = 'inputs/benchmark_test'
    output_video = os.path.join(output_dir, 'test_720p_80frames.mp4')
    output_mask = os.path.join(output_dir, 'mask_720p.png')

    # Check if input files exist
    if not os.path.exists(input_video):
        print(f"ERROR: Input video not found: {input_video}")
        return

    if not os.path.exists(input_mask):
        print(f"ERROR: Input mask not found: {input_mask}")
        return

    # Prepare test data
    print("="*60)
    print("Preparing ProPainter Benchmark Test Data")
    print("="*60)
    print()

    num_frames = prepare_test_video(input_video, output_video,
                                     target_width=1280, target_height=720, max_frames=80)
    print()

    prepare_test_mask(input_mask, output_mask,
                      target_width=1280, target_height=720)
    print()

    # Verify
    verify_test_data(output_video, output_mask, expected_frames=80)

    print()
    print("="*60)
    print("Test data preparation completed!")
    print("="*60)
    print(f"\nTest video: {output_video}")
    print(f"Test mask: {output_mask}")
    print(f"\nYou can now run the benchmark with:")
    print(f"  python benchmark_propainter.py --all")


if __name__ == '__main__':
    main()
