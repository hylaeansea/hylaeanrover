#!/bin/bash
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python

run_one () {
  local tag=$1 model=$2 vn=$3 cov=$4
  $PY examples/evaluate.py --stage minerals --load "$model" --vecnorm "$vn" \
    --scenario sparse_game --horizon long --episodes 5 --seed 42 --frame-skip 4 \
    --power-capacity 1000 --mission-supervisor $cov \
    > "runs/reports/coverage_gate_${tag}_sparse_game_long_1kwh.txt" 2>&1
  echo "done $tag"
}

run_one baseline ../models/minerals/model.zip ../models/minerals/vecnorm.pkl ""
run_one coverage_best runs/stage2_minerals_coverage/best/best_model.zip runs/stage2_minerals_coverage/best/vecnorm.pkl --coverage-observation
run_one coverage_final runs/stage2_minerals_coverage/model.zip runs/stage2_minerals_coverage/vecnorm.pkl --coverage-observation
echo "LONG EVALS COMPLETE"
