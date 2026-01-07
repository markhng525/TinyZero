#!/bin/bash
# Throughput Verification Test
# Tests if increasing gpu_memory_utilization improves tokens/sec
#
# This validates the claim that:
#   gpu_memory_utilization ↑ → KV cache ↑ → concurrent sequences ↑ → tokens/sec ↑
#
# Usage: bash test_throughput.sh
set -x

cd /workspace

echo "=========================================="
echo "GPU Memory Utilization Throughput Test"
echo "=========================================="
echo ""
echo "This test measures generation throughput at different"
echo "gpu_memory_utilization levels (0.5, 0.7, 0.8) to verify"
echo "that larger KV cache improves tokens/sec."
echo ""

# Run the test with Qwen2.5-1.5B
/workspace/.venv.linux-aarch64/bin/python /workspace/experiments/test_throughput.py \
    --model Qwen/Qwen2.5-1.5B \
    --batch-size 64 \
    --max-tokens 128 \
    --warmup 1 \
    --test-batches 3 \
    --utils 0.5 0.7 0.8 \
    2>&1 | tee /workspace/experiments/logs/throughput_test.log

echo ""
echo "Results saved to /workspace/experiments/logs/throughput_test.log"
