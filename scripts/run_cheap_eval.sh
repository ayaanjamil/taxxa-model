#!/usr/bin/env bash
# Run baseline + graph eval with cheap-mode model overrides.
# Output streams to your terminal AND to data/eval_runs/*.log via tee.
# Usage:
#   ./scripts/run_cheap_eval.sh              # full 83-question sweep
#   ./scripts/run_cheap_eval.sh --sample 20  # cost-bounded dry run
set -euo pipefail

cd "$(dirname "$0")/.."

set -a
source .env
source .env.cheap
set +a

mkdir -p data/eval_runs
ts=$(date +%Y%m%d_%H%M%S)

echo "Models:"
.venv/bin/python3 -c "from answerer import ANSWER_MODEL, PLANNER_MODEL; import os; print('  answer :', ANSWER_MODEL); print('  planner:', PLANNER_MODEL); print('  judge  :', os.getenv('TAXXA_JUDGE_MODEL', 'anthropic/claude-haiku-4-5'))"
echo

echo "=== BASELINE ==="
.venv/bin/python3 -u eval_harness.py --mode baseline \
  --output "data/eval_runs/baseline_${ts}.json" "$@" \
  2>&1 | tee "data/eval_runs/baseline_${ts}.log"

echo
echo "=== GRAPH ==="
.venv/bin/python3 -u eval_harness.py --mode graph \
  --output "data/eval_runs/graph_${ts}.json" "$@" \
  2>&1 | tee "data/eval_runs/graph_${ts}.log"

echo
echo "Done. Outputs:"
echo "  data/eval_runs/baseline_${ts}.{log,json}"
echo "  data/eval_runs/graph_${ts}.{log,json}"
