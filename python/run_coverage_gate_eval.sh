#!/bin/zsh
# Matched-seed gate evaluation: promoted minerals baseline vs coverage candidate.
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python
SEED=42
FRAME_SKIP=4
POWER_CAPACITY=100
EPISODES=20
SCENARIOS=(minerals_sparse minerals_transition sparse_game no_cube_control cube_visible_low_power)

run_model () {
  local tag=$1 model=$2 vecnorm=$3 covflag=$4
  for scenario in "${SCENARIOS[@]}"; do
    echo "=== $tag / $scenario ==="
    $PY examples/evaluate.py \
      --stage minerals \
      --load "$model" \
      --vecnorm "$vecnorm" \
      --scenario "$scenario" \
      --horizon medium \
      --episodes $EPISODES \
      --seed $SEED \
      --frame-skip $FRAME_SKIP \
      --power-capacity $POWER_CAPACITY \
      --mission-supervisor $covflag \
      > "runs/reports/coverage_gate_${tag}_${scenario}_medium.txt" 2>&1
    echo "done: runs/reports/coverage_gate_${tag}_${scenario}_medium.txt"
  done
}

run_model baseline ../models/minerals/model.zip ../models/minerals/vecnorm.pkl ""
run_model coverage_best runs/stage2_minerals_coverage/best/best_model.zip runs/stage2_minerals_coverage/best/vecnorm.pkl "--coverage-observation"
run_model coverage_final runs/stage2_minerals_coverage/model.zip runs/stage2_minerals_coverage/vecnorm.pkl "--coverage-observation"
echo "ALL EVALS COMPLETE"
