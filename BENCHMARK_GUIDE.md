# ProPainter Benchmark Usage Guide

## Quick Start

### 1. Run Basic Benchmark (FP32 vs FP16)

```bash
# Run all configurations
bash run_benchmark.sh

# Or run individually
python benchmark_propainter.py --config fp32_720p_80frames
python benchmark_propainter.py --config fp16_720p_80frames

# Run both and generate comparison report
python benchmark_propainter.py --all
```

### 2. View Results

Results are saved in the `results/` directory:
- `benchmark_report.md` - Comprehensive comparison report
- `fp32_720p_80frames_results.json` - Detailed FP32 results
- `fp16_720p_80frames_results.json` - Detailed FP16 results
- `benchmark_results.csv` - CSV format for Excel

## Output Format

### Markdown Report Structure

The benchmark generates a detailed Markdown report with:

1. **System Information**
   - GPU model
   - CUDA version
   - PyTorch version

2. **Model Parameters**
   - Parameter count for each model (RAFT_bi, RecurrentFlowCompleteNet, InpaintGenerator)
   - Total and trainable parameters

3. **Performance Comparison (FP32 vs FP16)**
   - Execution time for each stage
   - Speedup factors
   - Memory usage (peak, current, reserved)
   - Memory reduction percentages
   - Throughput (FPS, seconds per frame)

4. **Stage Breakdown**
   - Data Loading
   - Flow Estimation (RAFT)
   - Flow Completion
   - Image Propagation
   - Feature Propagation + Transformer

### JSON Report Structure

```json
{
  "total_time": 123.45,
  "max_memory": 25.6,
  "fps": 0.65,
  "video_length": 80,
  "resolution": "1280x720",
  "config": {...},
  "stages": {
    "data_loading": {
      "elapsed": 5.2,
      "memory": {
        "peak_gb": 2.1,
        "current_gb": 1.8,
        "reserved_gb": 2.5
      }
    },
    ...
  },
  "parameters": {
    "RAFT_bi": {
      "total": 5300000,
      "trainable": 0,
      "total_M": 5.3,
      "trainable_M": 0.0
    },
    ...
  }
}
```

## Additional Experiments

The `experiments_config.json` file defines additional experiments you can run:

### 1. Resolution Scaling
Test performance at different resolutions (480p, 720p, 1080p)

### 2. Frame Count Scaling
Test with different frame counts (40, 80, 120, 160 frames)

### 3. Parameter Tuning
Test impact of:
- `subvideo_length` (40, 80, 120)
- `neighbor_length` (5, 10, 15)
- `ref_stride` (5, 10, 20)

### 4. Ablation Study
Analyze contribution of each module:
- Flow estimation only
- Flow completion
- Image propagation
- Full pipeline

## Expected Results (720p, 80 frames)

### FP32
- Total time: ~60-120 seconds
- Peak memory: ~20-30 GB
- FPS: ~0.6-1.3

### FP16
- Total time: ~40-80 seconds (1.5-2x speedup)
- Peak memory: ~12-18 GB (40-50% reduction)
- FPS: ~1.0-2.0

### Model Parameters
- RAFT_bi: ~5M parameters (frozen)
- RecurrentFlowCompleteNet: ~10-20M parameters (frozen)
- InpaintGenerator: ~80-100M parameters (trainable)
- Total: ~100-130M parameters

## Troubleshooting

### Out of Memory
If you encounter OOM errors:
1. Use FP16: `--config fp16_720p_80frames`
2. Reduce resolution
3. Reduce `subvideo_length` parameter
4. Reduce `neighbor_length` parameter

### Slow Performance
- Ensure CUDA is available: `torch.cuda.is_available()`
- Check GPU utilization: `nvidia-smi`
- Verify model weights are downloaded to `weights/` directory

## Notes

- The benchmark uses the existing ProPainter profiling infrastructure
- All stages match the original `inference_propainter.py` implementation
- Results may vary based on GPU model and system configuration
- First run may be slower due to model weight downloads
