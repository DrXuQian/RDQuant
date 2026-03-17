#!/usr/bin/env bash
# Accuracy benchmark: quantize a model at multiple budgets and evaluate on
# WikiText-2 perplexity and zero-shot tasks.
#
# Usage:
#   bash benchmarks/accuracy/run_eval.sh [MODEL] [OUTPUT_DIR]
#
# Example:
#   bash benchmarks/accuracy/run_eval.sh meta-llama/Llama-2-7b-hf ./results/llama2-7b

set -euo pipefail

MODEL="${1:-sshleifer/tiny-gpt2}"
OUTPUT_DIR="${2:-./results/benchmark}"
BUDGETS="4.0 5.0 5.3 6.0 7.0 8.0"

echo "=============================="
echo " RDQuant Accuracy Benchmark"
echo "=============================="
echo "Model      : ${MODEL}"
echo "Output dir : ${OUTPUT_DIR}"
echo "Budgets    : ${BUDGETS}"
echo ""

mkdir -p "${OUTPUT_DIR}"

# Sweep budgets and collect R-D curve
python examples/sweep_budget.py \
    --model "${MODEL}" \
    --budgets ${BUDGETS} \
    --eval \
    --plot "${OUTPUT_DIR}/rd_curve.png" \
    2>&1 | tee "${OUTPUT_DIR}/sweep.log"

echo ""
echo "Results written to ${OUTPUT_DIR}/"
echo "Done."
