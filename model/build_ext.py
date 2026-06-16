"""Build cuda_extension.so for production (ahead-of-time, no runtime JIT).

Usage:
    python build_ext.py                 # build into ./ (cuda_extension.so)
    TORCH_CUDA_ARCH_LIST=9.0 python build_ext.py    # target a specific arch

The kernels live in kernels/*.cu / *.cpp plus binding.cpp. This mirrors what
utils/compile.sh does but writes a standalone .so you can ship alongside
propainter_fast.py (put it on PYTHONPATH or next to the package).
"""
import os
import shutil
from pathlib import Path

import torch
import torch.utils.cpp_extension as cpp_ext

HERE = Path(__file__).parent


def find_cudnn_include():
    """Locate cudnn.h (torch's bundled nvidia-cudnn, or a system install)."""
    sp = Path(torch.__file__).parent.parent
    cands = [
        sp / 'nvidia' / 'cudnn' / 'include',
        Path(torch.__file__).parent / 'include',
        Path('/usr/local/cuda/include'),
        Path('/usr/include'),
    ]
    env = os.environ.get('CUDNN_INCLUDE_DIR')
    if env:
        cands.insert(0, Path(env))
    for c in cands:
        if (c / 'cudnn.h').exists():
            return str(c)
    raise FileNotFoundError(
        "cudnn.h not found. Set CUDNN_INCLUDE_DIR to the dir containing it "
        f"(searched: {[str(c) for c in cands]})")


def main():
    srcs = sorted(set(
        [str(p) for p in HERE.glob('*.cu')] + [str(p) for p in HERE.glob('*.cpp')] +
        [str(p) for p in (HERE / 'kernels').glob('*.cu')] +
        [str(p) for p in (HERE / 'kernels').glob('*.cpp')]
    ))
    build_dir = HERE / 'build' / 'aot'
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    cudnn_inc = find_cudnn_include()
    print(f"cudnn.h include dir: {cudnn_inc}")
    print(f"Building {len(srcs)} sources -> cuda_extension.so")
    cpp_ext.load(
        name='cuda_extension',
        sources=srcs,
        build_directory=str(build_dir),
        verbose=True,
        with_cuda=True,
        extra_include_paths=[cudnn_inc],
        extra_cflags=['-O3', '-std=c++17'],
        extra_cuda_cflags=['-O3', '--use_fast_math'],
    )
    out = HERE / 'cuda_extension.so'
    shutil.copy2(build_dir / 'cuda_extension.so', out)
    print(f"OK -> {out}")


if __name__ == '__main__':
    main()
