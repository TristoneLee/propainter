#!/bin/bash
# ProPainter Benchmark Execution Script

echo "=========================================="
echo "ProPainter Performance Benchmark"
echo "=========================================="
echo ""

# Create results directory
mkdir -p results

# Run FP32 benchmark
echo "1. Running FP32 benchmark (720p, 80 frames)..."
python benchmark_propainter.py --config fp32_720p_80frames --output-dir results

echo ""
echo "Waiting 10 seconds before next test..."
sleep 10

# Run FP16 benchmark
echo "2. Running FP16 benchmark (720p, 80 frames)..."
python benchmark_propainter.py --config fp16_720p_80frames --output-dir results

echo ""
echo "=========================================="
echo "Benchmark completed!"
echo "=========================================="
echo ""
echo "Results saved in: results/"
echo "- benchmark_report.md (comparison report)"
echo "- fp32_720p_80frames_results.json"
echo "- fp16_720p_80frames_results.json"
echo "- benchmark_results.csv"
echo ""
