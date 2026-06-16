#!/usr/bin/env python
"""
End-to-end PSNR comparison: legacy SDPA loop vs FlexAttention path.

Runs inference_propainter.py twice on a sample directory (default
inputs/samples/1) at 1280x720 fp16 — once with --use-flex-attn off and
once on — reads both inpaint_out.mp4 outputs frame by frame using
imageio.v3, computes per-frame PSNR with a simple numpy implementation,
and reports mean PSNR + worst-frame PSNR/index.

Acceptance (US-005): exit 0 if mean PSNR > 40 dB; non-zero otherwise.
"""

import argparse
import os
import subprocess
import sys

import imageio.v3 as iio
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _psnr_uint8(a: np.ndarray, b: np.ndarray) -> float:
    """PSNR in dB for two uint8 frame arrays of equal shape."""
    diff = a.astype(np.float64) - b.astype(np.float64)
    mse = float(np.mean(diff * diff))
    if mse == 0.0:
        return float('inf')
    return 10.0 * np.log10((255.0 ** 2) / mse)


def _run_inference(video, mask, output_root, use_flex_attn, frames, fp16, height, width,
                   compile_mode=None, compile_fullgraph=False):
    cmd = [
        sys.executable,
        os.path.join(_REPO_ROOT, 'inference_propainter.py'),
        '--video', video,
        '--mask', mask,
        '--output', output_root,
        '--height', str(height),
        '--width', str(width),
    ]
    if fp16:
        cmd.append('--fp16')
    if use_flex_attn:
        cmd.append('--use-flex-attn')
    if frames > 0:
        cmd.extend(['--frames', str(frames)])
    if compile_mode is not None:
        cmd.extend(['--compile', '--compile-mode', compile_mode])
        if compile_fullgraph:
            cmd.append('--compile-fullgraph')
    label = 'flex' if use_flex_attn else 'legacy'
    print(f'[compare_outputs] running {label}: {" ".join(cmd)}', flush=True)
    subprocess.run(cmd, check=True, cwd=_REPO_ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample', default='inputs/samples/1',
                        help='Sample dir holding input.mp4 + masks.mp4 (default: inputs/samples/1)')
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--fp16', action='store_true', default=True,
                        help='Run inference in fp16 (default true; --no-fp16 disables)')
    parser.add_argument('--no-fp16', dest='fp16', action='store_false')
    parser.add_argument('--frames', type=int, default=30,
                        help='Limit number of frames (-1 = all). Default 30: eager '
                             'flex_attention materializes the full scores matrix and '
                             'OOMs on long videos at 720p; 30 is a safe smoke size '
                             'and is enough to validate end-to-end visual fidelity.')
    parser.add_argument('--output-root', default='results/compare',
                        help='Root for the two intermediate inference outputs')
    parser.add_argument('--threshold', type=float, default=40.0,
                        help='Mean PSNR threshold (dB) for the pass/fail check')
    parser.add_argument('--skip-legacy', action='store_true',
                        help='Reuse a previously generated legacy run')
    parser.add_argument('--skip-flex', action='store_true',
                        help='Reuse a previously generated flex run')
    parser.add_argument('--compile-mode', default=None,
                        help='If set (e.g. "default", "max-autotune-no-cudagraphs"), '
                             'forward --compile --compile-mode <value> to the FLEX run only. '
                             'The legacy run is always eager so PSNR validates compiled flex '
                             'against eager legacy ground truth.')
    parser.add_argument('--compile-fullgraph', action='store_true',
                        help='Pass --compile-fullgraph to the flex run when --compile-mode is set.')
    args = parser.parse_args()

    sample_dir = args.sample if os.path.isabs(args.sample) else os.path.join(_REPO_ROOT, args.sample)
    video = os.path.join(sample_dir, 'input.mp4')
    mask = os.path.join(sample_dir, 'masks.mp4')
    if not os.path.exists(video) or not os.path.exists(mask):
        print(f'[compare_outputs] missing input.mp4 or masks.mp4 under {sample_dir}', file=sys.stderr)
        return 2

    output_root = args.output_root if os.path.isabs(args.output_root) else os.path.join(_REPO_ROOT, args.output_root)
    legacy_root = os.path.join(output_root, 'legacy')
    flex_root = os.path.join(output_root, 'flex')

    # `inference_propainter.read_frame_from_videos` derives video_name from the basename minus the last 4 chars.
    video_name = os.path.basename(video)[:-4]
    legacy_out = os.path.join(legacy_root, video_name, 'inpaint_out.mp4')
    flex_out = os.path.join(flex_root, video_name, 'inpaint_out.mp4')

    if not args.skip_legacy:
        _run_inference(video, mask, legacy_root, use_flex_attn=False,
                       frames=args.frames, fp16=args.fp16,
                       height=args.height, width=args.width)
    if not args.skip_flex:
        _run_inference(video, mask, flex_root, use_flex_attn=True,
                       frames=args.frames, fp16=args.fp16,
                       height=args.height, width=args.width,
                       compile_mode=args.compile_mode,
                       compile_fullgraph=args.compile_fullgraph)

    for p in (legacy_out, flex_out):
        if not os.path.exists(p):
            print(f'[compare_outputs] expected output missing: {p}', file=sys.stderr)
            return 2

    legacy_frames = iio.imread(legacy_out)
    flex_frames = iio.imread(flex_out)
    if legacy_frames.shape != flex_frames.shape:
        print(f'[compare_outputs] shape mismatch: legacy={legacy_frames.shape}  flex={flex_frames.shape}',
              file=sys.stderr)
        return 1

    n = legacy_frames.shape[0]
    psnrs = np.array([_psnr_uint8(legacy_frames[i], flex_frames[i]) for i in range(n)])
    finite = psnrs[np.isfinite(psnrs)]
    if finite.size == 0:
        # all frames bit-identical
        mean_psnr = float('inf')
    else:
        mean_psnr = float(finite.mean())
    worst_idx = int(np.argmin(psnrs))
    worst_psnr = float(psnrs[worst_idx])

    print(f'[compare_outputs] frames={n}  mean_psnr={mean_psnr:.2f} dB  '
          f'worst={worst_psnr:.2f} dB @ frame {worst_idx}')

    if mean_psnr > args.threshold:
        print(f'[compare_outputs] PASS (mean PSNR {mean_psnr:.2f} dB > {args.threshold:.1f} dB)')
        return 0
    print(f'[compare_outputs] FAIL (mean PSNR {mean_psnr:.2f} dB <= {args.threshold:.1f} dB)')
    return 1


if __name__ == '__main__':
    sys.exit(main())
