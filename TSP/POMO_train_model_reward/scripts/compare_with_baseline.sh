#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/compare_with_baseline.sh <new_eval_json> [output_markdown]" >&2
  exit 1
fi

NEW_EVAL_JSON="$1"
OUTPUT_MARKDOWN="${2:-compare_with_baseline.md}"

python compare_eval.py \
  --baseline ./result_lib/baseline_eval_cpu.json \
  --new "${NEW_EVAL_JSON}" \
  --output "${OUTPUT_MARKDOWN}"
