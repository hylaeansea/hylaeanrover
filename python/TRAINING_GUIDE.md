# Step-by-Step RL Training Guide

This guide starts from no trained model and ends with a promoted `full`
policy. Setup blocks that start with `cd python` assume you are at the
repo root. After setup, stay in `python/` unless a step says otherwise.

Use this as the operating checklist. For Stage 0/1 acceptance details,
also keep `../docs/rl_stage0_stage1_hardening_plan.md` open.

## 0. Pick Run Settings

Use one seed, one frame-skip, and one power capacity through the whole
curriculum.

```bash
cd python

export SEED=42
export FRAME_SKIP=4
export N_ENVS=8
export POWER_CAPACITY=100
export EVAL_EPISODES=20

export STAGE0_STEPS=2000000
export STAGE1_STEPS=2000000
export STAGE2_STEPS=2000000
export STAGE3_STEPS=2000000
```

Use fewer `N_ENVS` if the machine has fewer physical cores. Keep
`FRAME_SKIP` identical for training, evaluation, promotion, and export.

## 1. Create the Python Environment

Skip this section only if `python/.venv` already exists and is current.

```bash
cd python
uv venv --python 3.13
source .venv/bin/activate
uv pip install 'maturin[patchelf]>=1.4'
maturin develop --release --uv --extras sb3
```

Plain `pip` alternative:

```bash
cd python
python -m venv .venv
source .venv/bin/activate
pip install 'maturin[patchelf]>=1.4'
pip install -e '.[sb3]'
maturin develop --release
```

## 2. Verify the Environment

```bash
cd python
source .venv/bin/activate

python -c "import hylaeanrover, os; p = hylaeanrover._native.__file__; print(p, os.path.getmtime(p))"
python examples/train_ppo.py
python -m unittest discover tests
```

From the repo root, run the Rust checks when Rust code changed:

```bash
cargo test --workspace
cargo clippy --workspace --tests
```

## 3. Create Output Folders

```bash
cd python
mkdir -p runs/reports
```

Use these run directories:

```text
runs/stage0_locomotion
runs/stage1_cube_intercept
runs/stage1_power_idle_bc
runs/stage1_power_cubes
runs/stage1_power_cubes_bridge
runs/stage2_minerals
runs/stage3_full
```

## 4. Record Random Baselines

### Stage 0 Terrain Baselines

```bash
for scenario in terrain_fixed_1_0 terrain_fixed_1_5 terrain_fixed_2_0 terrain_mixed_1_2; do
  python examples/evaluate.py \
    --stage locomotion \
    --random \
    --scenario "$scenario" \
    --horizon medium \
    --episodes "$EVAL_EPISODES" \
    --seed "$SEED" \
    --frame-skip "$FRAME_SKIP" \
    --power-capacity "$POWER_CAPACITY" \
    | tee "runs/reports/stage0_random_${scenario}.txt"
done
```

### Full-Stage Random Baseline

```bash
python examples/evaluate.py \
  --stage full \
  --random \
  --scenario terrain_mixed_1_2 \
  --horizon medium \
  --episodes "$EVAL_EPISODES" \
  --seed "$SEED" \
  --frame-skip "$FRAME_SKIP" \
  --power-capacity "$POWER_CAPACITY" \
  | tee runs/reports/stage3_random_full.txt
```

## 5. Train Stage 0: Locomotion

Train from scratch on mixed terrain.

```bash
python examples/train.py \
  --stage locomotion \
  --timesteps "$STAGE0_STEPS" \
  --save runs/stage0_locomotion \
  --seed "$SEED" \
  --n-envs "$N_ENVS" \
  --frame-skip "$FRAME_SKIP" \
  --power-capacity "$POWER_CAPACITY" \
  --horizon short \
  --scenario terrain_mixed_1_2 \
  --locomotion-shaping power_efficiency \
  --extra-eval-scenarios terrain_fixed_1_0,terrain_fixed_1_5,terrain_fixed_2_0 \
  --eval-freq 25000 \
  --n-eval-episodes 10 \
  --checkpoint-freq 100000 \
  --ent-coef 0.01 \
  --target-kl 0.02
```

Watch the run:

```bash
tensorboard --logdir runs/stage0_locomotion/tb
```

## 6. Evaluate Stage 0

Evaluate short and medium horizons first.

```bash
for horizon in short medium; do
  for scenario in terrain_fixed_1_0 terrain_fixed_1_5 terrain_fixed_2_0 terrain_mixed_1_2; do
    python examples/evaluate.py \
      --stage locomotion \
      --load runs/stage0_locomotion/best/best_model.zip \
      --vecnorm runs/stage0_locomotion/best/vecnorm.pkl \
      --scenario "$scenario" \
      --horizon "$horizon" \
      --episodes "$EVAL_EPISODES" \
      --seed "$SEED" \
      --frame-skip "$FRAME_SKIP" \
      --power-capacity "$POWER_CAPACITY" \
      | tee "runs/reports/stage0_best_${scenario}_${horizon}.txt"
  done
done
```

Run one long-horizon sample before promotion.

```bash
python examples/evaluate.py \
  --stage locomotion \
  --load runs/stage0_locomotion/best/best_model.zip \
  --vecnorm runs/stage0_locomotion/best/vecnorm.pkl \
  --scenario terrain_mixed_1_2 \
  --horizon long \
  --episodes 5 \
  --seed "$SEED" \
  --frame-skip "$FRAME_SKIP" \
  --power-capacity "$POWER_CAPACITY" \
  | tee runs/reports/stage0_best_mixed_long.txt
```

Summarize the gate:

```bash
python examples/check_stage0_gate.py
```

If Stage 0 does not pass, continue from the current best checkpoint
instead of starting over:

```bash
python examples/train.py \
  --stage locomotion \
  --timesteps "$STAGE0_STEPS" \
  --load runs/stage0_locomotion/best/best_model.zip \
  --vecnorm runs/stage0_locomotion/best/vecnorm.pkl \
  --reset-reward-stats \
  --save runs/stage0_locomotion_continue1 \
  --seed "$SEED" \
  --n-envs "$N_ENVS" \
  --frame-skip "$FRAME_SKIP" \
  --power-capacity "$POWER_CAPACITY" \
  --horizon short \
  --scenario terrain_mixed_1_2 \
  --locomotion-shaping power_efficiency \
  --extra-eval-scenarios terrain_fixed_1_0,terrain_fixed_1_5,terrain_fixed_2_0 \
  --eval-freq 25000 \
  --n-eval-episodes 10 \
  --checkpoint-freq 100000 \
  --learning-rate 0.0001 \
  --ent-coef 0.01 \
  --target-kl 0.02
```

Evaluate the continuation into a separate report directory:

```bash
export STAGE0_REPORT_DIR=runs/reports/stage0_continue1
mkdir -p "$STAGE0_REPORT_DIR"
cp runs/reports/stage0_random_*.txt "$STAGE0_REPORT_DIR"/

for horizon in short medium; do
  for scenario in terrain_fixed_1_0 terrain_fixed_1_5 terrain_fixed_2_0 terrain_mixed_1_2; do
    python examples/evaluate.py \
      --stage locomotion \
      --load runs/stage0_locomotion_continue1/best/best_model.zip \
      --vecnorm runs/stage0_locomotion_continue1/best/vecnorm.pkl \
      --scenario "$scenario" \
      --horizon "$horizon" \
      --episodes "$EVAL_EPISODES" \
      --seed "$SEED" \
      --frame-skip "$FRAME_SKIP" \
      --power-capacity "$POWER_CAPACITY" \
      | tee "$STAGE0_REPORT_DIR/stage0_best_${scenario}_${horizon}.txt"
  done
done

python examples/evaluate.py \
  --stage locomotion \
  --load runs/stage0_locomotion_continue1/best/best_model.zip \
  --vecnorm runs/stage0_locomotion_continue1/best/vecnorm.pkl \
  --scenario terrain_mixed_1_2 \
  --horizon long \
  --episodes 5 \
  --seed "$SEED" \
  --frame-skip "$FRAME_SKIP" \
  --power-capacity "$POWER_CAPACITY" \
  | tee "$STAGE0_REPORT_DIR/stage0_best_mixed_long.txt"

python examples/check_stage0_gate.py --reports-dir "$STAGE0_REPORT_DIR"
```

If a short-horizon continuation still misses the medium gate, run one
medium-horizon fine-tune from the best continuation checkpoint:

```bash
python examples/train.py \
  --stage locomotion \
  --timesteps 1000000 \
  --load runs/stage0_locomotion_continue1/best/best_model.zip \
  --vecnorm runs/stage0_locomotion_continue1/best/vecnorm.pkl \
  --preserve-reward-stats \
  --save runs/stage0_locomotion_medium_ft \
  --seed "$SEED" \
  --n-envs "$N_ENVS" \
  --frame-skip "$FRAME_SKIP" \
  --power-capacity "$POWER_CAPACITY" \
  --horizon medium \
  --scenario terrain_mixed_1_2 \
  --locomotion-shaping power_efficiency \
  --extra-eval-scenarios terrain_fixed_1_0,terrain_fixed_1_5,terrain_fixed_2_0 \
  --eval-freq 25000 \
  --n-eval-episodes 10 \
  --checkpoint-freq 100000 \
  --learning-rate 0.00005 \
  --ent-coef 0.01 \
  --target-kl 0.02
```

If the medium-horizon fine-tune still learns an out-of-power sprint, run
one power-efficiency recovery pass from that checkpoint:

```bash
python examples/train.py \
  --stage locomotion \
  --timesteps 1000000 \
  --load runs/stage0_locomotion_medium_ft/best/best_model.zip \
  --vecnorm runs/stage0_locomotion_medium_ft/best/vecnorm.pkl \
  --preserve-reward-stats \
  --save runs/stage0_locomotion_power_efficiency \
  --seed "$SEED" \
  --n-envs "$N_ENVS" \
  --frame-skip "$FRAME_SKIP" \
  --power-capacity "$POWER_CAPACITY" \
  --horizon medium \
  --scenario terrain_mixed_1_2 \
  --locomotion-shaping power_efficiency \
  --extra-eval-scenarios terrain_fixed_1_0,terrain_fixed_1_5,terrain_fixed_2_0 \
  --eval-freq 25000 \
  --n-eval-episodes 10 \
  --checkpoint-freq 100000 \
  --learning-rate 0.00005 \
  --n-epochs 5 \
  --ent-coef 0.02 \
  --target-kl 0.02
```

If that recovery pass still only misses one terrain, fine-tune on the
failing terrain with stronger power shaping and keep the other terrains
as side evals:

```bash
python examples/train.py \
  --stage locomotion \
  --timesteps 750000 \
  --load runs/stage0_locomotion_power_efficiency/best/best_model.zip \
  --vecnorm runs/stage0_locomotion_power_efficiency/best/vecnorm.pkl \
  --preserve-reward-stats \
  --save runs/stage0_locomotion_fixed_2_recovery \
  --seed "$SEED" \
  --n-envs "$N_ENVS" \
  --frame-skip "$FRAME_SKIP" \
  --power-capacity "$POWER_CAPACITY" \
  --horizon medium \
  --scenario terrain_fixed_2_0 \
  --locomotion-shaping power_efficiency \
  --locomotion-coast-bonus 0.50 \
  --locomotion-power-draw-penalty 80 \
  --locomotion-power-recovery-reward 40 \
  --locomotion-out-of-power-penalty 150 \
  --extra-eval-scenarios terrain_mixed_1_2,terrain_fixed_1_0,terrain_fixed_1_5 \
  --eval-freq 25000 \
  --n-eval-episodes 10 \
  --checkpoint-freq 100000 \
  --learning-rate 0.00003 \
  --n-epochs 5 \
  --ent-coef 0.02 \
  --target-kl 0.02
```

Stage 0 can pass when:

- trained distance is at least 2-3x the random baseline;
- flip rate is low on fixed `1.0`, `1.5`, `2.0`, and mixed terrain;
- failures are understood from terminal reasons;
- observation shape is still 41 and action space is still `Discrete(10)`.

## 7. Promote Stage 0

Only promote after Stage 0 passes.

```bash
python examples/promote_model.py \
  --stage locomotion \
  --run runs/stage0_locomotion \
  --source best \
  --frame-skip "$FRAME_SKIP"
```

Use the promoted bundle for Stage 1:

```text
../models/locomotion/model.zip
../models/locomotion/vecnorm.pkl
../models/locomotion/model.onnx
../models/locomotion/model.norm.json
```

Optional visual check:

```bash
cargo run -p hylaeanrover_game --release -- \
  --policy ../models/locomotion/model.onnx
```

## 8. Record Stage 1 Locomotion Baselines

Evaluate the promoted locomotion policy under the Stage 1 rewards and
scenarios. These are baselines, not promotion evidence for Stage 1.

```bash
for scenario in cube_intercept cube_intercept_low_power sparse_visible_low_power; do
  python examples/evaluate.py \
    --stage cube_intercept \
    --load ../models/locomotion/model.zip \
    --vecnorm ../models/locomotion/vecnorm.pkl \
    --scenario "$scenario" \
    --horizon medium \
    --episodes "$EVAL_EPISODES" \
    --seed "$SEED" \
    --frame-skip "$FRAME_SKIP" \
    --power-capacity "$POWER_CAPACITY" \
    --cube-shaping off \
    | tee "runs/reports/stage1_locomotion_baseline_${scenario}.txt"
done

for scenario in dense_training transition sparse_game sparse_visible_low_power low_power_start cube_visible_low_power no_cube_control; do
  python examples/evaluate.py \
    --stage power_cubes \
    --load ../models/locomotion/model.zip \
    --vecnorm ../models/locomotion/vecnorm.pkl \
    --scenario "$scenario" \
    --horizon medium \
    --episodes "$EVAL_EPISODES" \
    --seed "$SEED" \
    --frame-skip "$FRAME_SKIP" \
    --power-capacity "$POWER_CAPACITY" \
    --cube-shaping off \
    | tee "runs/reports/stage1_locomotion_baseline_${scenario}.txt"
done
```

## 9. Train Stage 1A: Cube Intercept

Stage 1A trains the missing sensor-to-action skill in isolation: one
settled, actionable forced cube, no random spawns, fixed observation and
action spaces. First create and evaluate a behavior-cloned checkpoint.
Do not start PPO until the BC-only checkpoint can pick up the forced
cubes with shaping off.

```bash
python examples/train.py \
  --stage cube_intercept \
  --timesteps 0 \
  --load ../models/locomotion/model.zip \
  --vecnorm ../models/locomotion/vecnorm.pkl \
  --reset-reward-stats \
  --save runs/stage1_cube_intercept_bc \
  --seed "$SEED" \
  --n-envs 1 \
  --frame-skip "$FRAME_SKIP" \
  --power-capacity "$POWER_CAPACITY" \
  --horizon short \
  --scenario cube_intercept \
  --terrain-height fixed_1_0 \
  --teacher-pretrain-samples 20000 \
  --teacher-pretrain-epochs 30 \
  --teacher-pretrain-batch-size 512 \
  --teacher-scenarios cube_intercept \
  --teacher-pretrain-only \
  --cube-shaping auto \
  --n-steps 512 \
  --batch-size 128 \
  --learning-rate 0.00003 \
  --clip-range 0.05 \
  --n-epochs 2 \
  --target-kl 0.005
```

If the BC-only checkpoint passes, optionally run a short conservative
PPO preservation pass. Compare it against the BC checkpoint; use the
better eval result, not the later checkpoint by default.

```bash
python examples/train.py \
  --stage cube_intercept \
  --timesteps 50000 \
  --load runs/stage1_cube_intercept_bc/model.zip \
  --vecnorm runs/stage1_cube_intercept_bc/vecnorm.pkl \
  --preserve-reward-stats \
  --save runs/stage1_cube_intercept \
  --seed "$SEED" \
  --n-envs 1 \
  --frame-skip "$FRAME_SKIP" \
  --power-capacity "$POWER_CAPACITY" \
  --horizon short \
  --scenario cube_intercept \
  --terrain-height fixed_1_0 \
  --train-scenarios cube_intercept,cube_intercept_low_power \
  --extra-eval-scenarios cube_intercept_low_power,sparse_visible_low_power,no_cube_control \
  --cube-shaping auto \
  --eval-freq 10000 \
  --n-eval-episodes 10 \
  --checkpoint-freq 25000 \
  --learning-rate 0.00001 \
  --n-steps 512 \
  --batch-size 128 \
  --n-epochs 2 \
  --clip-range 0.05 \
  --target-kl 0.005
```

Watch the run:

```bash
tensorboard --logdir runs/stage1_cube_intercept_bc/tb runs/stage1_cube_intercept/tb
```

## 10. Evaluate Stage 1A Without Shaping

Acceptance evals must use `--cube-shaping off`.

```bash
for scenario in cube_intercept cube_intercept_low_power sparse_visible_low_power no_cube_control; do
  python examples/evaluate.py \
    --stage cube_intercept \
    --load runs/stage1_cube_intercept/best/best_model.zip \
    --vecnorm runs/stage1_cube_intercept/best/vecnorm.pkl \
    --scenario "$scenario" \
    --horizon medium \
    --episodes "$EVAL_EPISODES" \
    --seed "$SEED" \
    --frame-skip "$FRAME_SKIP" \
    --power-capacity "$POWER_CAPACITY" \
    --cube-shaping off \
    | tee "runs/reports/stage1_cube_intercept_${scenario}_medium.txt"
done
```

Stage 1A can pass only when:

- `cube_intercept` reaches >=0.8 pickups/episode on forced 15-60 m
  visible cubes.
- `cube_intercept_low_power` reaches >=0.7 pickups/episode and
  out-of-power rate <=20%.
- `sparse_visible_low_power` succeeds with shaping off.
- Action traces show sustained approach instead of reverse/turn drift.

## 11. Train Stage 1B: Power Idle

Only start this after Stage 1A passes. Warm-start from the intercept
checkpoint and train low-power no-target discipline with no random cubes
and no pickup reward. This isolates the broad Stage 1 failure mode where
the policy keeps throttling after the actionable cube sensor is empty.
Use teacher pretraining first: it gives direct coast labels for no-cube
low-power observations and visible-cube intercept labels to avoid
forgetting the Stage 1A skill.

```bash
python examples/train.py \
  --stage power_idle \
  --timesteps 0 \
  --load runs/stage1_cube_intercept/best/best_model.zip \
  --vecnorm runs/stage1_cube_intercept/best/vecnorm.pkl \
  --reset-reward-stats \
  --save runs/stage1_power_idle_bc \
  --seed "$SEED" \
  --n-envs 1 \
  --frame-skip "$FRAME_SKIP" \
  --power-capacity "$POWER_CAPACITY" \
  --horizon short \
  --scenario power_idle \
  --teacher-pretrain-samples 20000 \
  --teacher-pretrain-epochs 10 \
  --teacher-pretrain-batch-size 512 \
  --teacher-scenarios power_idle,cube_intercept_low_power \
  --teacher-pretrain-only
```

Evaluate the idle gate:

```bash
for scenario in power_idle no_cube_control; do
  python examples/evaluate.py \
    --stage power_idle \
    --load runs/stage1_power_idle_bc/best/best_model.zip \
    --vecnorm runs/stage1_power_idle_bc/best/vecnorm.pkl \
    --scenario "$scenario" \
    --horizon medium \
    --episodes "$EVAL_EPISODES" \
    --seed "$SEED" \
    --frame-skip "$FRAME_SKIP" \
    --power-capacity "$POWER_CAPACITY" \
    --cube-shaping off \
    | tee "runs/reports/stage1_power_idle_bc_${scenario}_medium.txt"
done
```

Stage 1B can pass only when `power_idle` and `no_cube_control` have
out-of-power rate <=20% over a medium-horizon batch.

## 12. Train Stage 1C: Power Cubes

Only start broad power-cube training after Stage 1A and Stage 1B pass.
Warm-start from the idle BC checkpoint and mix dense, bridge, transition,
sparse-visible, and sparse-game scenarios. Dense pickup success alone is
not promotable. Stage 1C uses zero distance reward in `power_cubes`; the
travel objective returns in later stages after visible-cube pickup is
reliable. `--train-scenarios` assigns scenarios to vector-env workers, so
set `N_ENVS` to at least the number of listed scenarios.

```bash
python examples/train.py \
  --stage power_cubes \
  --timesteps "$STAGE1_STEPS" \
  --load runs/stage1_power_idle_bc/best/best_model.zip \
  --vecnorm runs/stage1_power_idle_bc/best/vecnorm.pkl \
  --reset-reward-stats \
  --save runs/stage1_power_cubes \
  --seed "$SEED" \
  --n-envs "$N_ENVS" \
  --frame-skip "$FRAME_SKIP" \
  --power-capacity "$POWER_CAPACITY" \
  --horizon short \
  --scenario sparse_visible_low_power \
  --train-scenarios dense_training,bridge_low_power,transition,sparse_visible_low_power,sparse_low_power,no_cube_control \
  --extra-eval-scenarios sparse_visible_low_power,sparse_visible_reset,sparse_game,transition,dense_training,no_cube_control \
  --cube-spawn-seed "$SEED" \
  --cube-shaping intercept \
  --eval-freq 25000 \
  --n-eval-episodes 10 \
  --checkpoint-freq 100000 \
  --learning-rate 0.00002 \
  --n-epochs 4 \
  --ent-coef 0.01 \
  --target-kl 0.015
```

Evaluate Stage 1C with shaping off:

```bash
for scenario in sparse_visible_low_power sparse_visible_reset sparse_game transition bridge_low_power dense_training no_cube_control; do
  python examples/evaluate.py \
    --stage power_cubes \
    --load runs/stage1_power_cubes/best/best_model.zip \
    --vecnorm runs/stage1_power_cubes/best/vecnorm.pkl \
    --scenario "$scenario" \
    --horizon medium \
    --episodes "$EVAL_EPISODES" \
    --seed "$SEED" \
    --frame-skip "$FRAME_SKIP" \
    --power-capacity "$POWER_CAPACITY" \
    --cube-shaping off \
    | tee "runs/reports/stage1_power_cubes_${scenario}_medium.txt"
done
```

Run sparse long-horizon samples before promotion:

```bash
python examples/evaluate.py \
  --stage power_cubes \
  --load runs/stage1_power_cubes/best/best_model.zip \
  --vecnorm runs/stage1_power_cubes/best/vecnorm.pkl \
  --scenario sparse_game \
  --horizon long \
  --episodes 5 \
  --seed "$SEED" \
  --frame-skip "$FRAME_SKIP" \
  --power-capacity "$POWER_CAPACITY" \
  --cube-shaping off \
  | tee runs/reports/stage1_power_cubes_sparse_game_long.txt
```

Stage 1C can pass when it beats the Stage 1 locomotion baselines on:

- no-shaping pickups in `sparse_visible_low_power`, `dense_training`, and
  `transition`;
- nonzero sparse-game pickups over a meaningful medium-horizon batch;
- end power and out-of-power rate in power-pressure scenarios;
- no-cube control behavior without fake pickup reward.

## 13. Promote Stage 1

Only promote after Stage 1A, Stage 1B, and Stage 1C gates pass and the
reports are recorded.

```bash
python examples/promote_model.py \
  --stage power_cubes \
  --run runs/stage1_power_cubes \
  --source best \
  --frame-skip "$FRAME_SKIP"
```

If a later Stage 1C continuation is the accepted run, promote that run
instead. Do not start minerals until `../models/power_cubes/` comes from
a run that passed the Stage 1 gates.

## 14. Record Stage 2 Baselines

Evaluate the promoted power-cube policy under the Stage 2 reward. Stage
2 must improve mineral integral without badly regressing cube behavior.

```bash
python examples/evaluate.py \
  --stage minerals \
  --load ../models/power_cubes/model.zip \
  --vecnorm ../models/power_cubes/vecnorm.pkl \
  --scenario terrain_mixed_1_2 \
  --horizon medium \
  --episodes "$EVAL_EPISODES" \
  --seed "$SEED" \
  --frame-skip "$FRAME_SKIP" \
  --power-capacity "$POWER_CAPACITY" \
  | tee runs/reports/stage2_power_cubes_baseline_medium.txt
```

Also record sparse and transition cube-density checks.

```bash
for scenario in transition sparse_game; do
  python examples/evaluate.py \
    --stage minerals \
    --load ../models/power_cubes/model.zip \
    --vecnorm ../models/power_cubes/vecnorm.pkl \
    --scenario "$scenario" \
    --horizon medium \
    --episodes "$EVAL_EPISODES" \
    --seed "$SEED" \
    --frame-skip "$FRAME_SKIP" \
    --power-capacity "$POWER_CAPACITY" \
    | tee "runs/reports/stage2_power_cubes_baseline_${scenario}.txt"
done
```

## 15. Bootstrap Stage 2 Exploration

Warm-start from the accepted power-cube model and add the missing
exploration motor prior before PPO. The mineral observation is local
concentration, not a directional gradient, so this teacher simply drives
while power is healthy, coasts during low-power recovery, and preserves
visible-cube interception only as a survival behavior.

```bash
python examples/train.py \
  --stage minerals \
  --timesteps 0 \
  --load ../models/power_cubes/model.zip \
  --vecnorm ../models/power_cubes/vecnorm.pkl \
  --reset-reward-stats \
  --save runs/stage2_minerals_bc \
  --seed "$SEED" \
  --n-envs 1 \
  --frame-skip "$FRAME_SKIP" \
  --power-capacity "$POWER_CAPACITY" \
  --horizon short \
  --scenario minerals_explore \
  --low-power-threshold 0.35 \
  --teacher-pretrain-samples 75000 \
  --teacher-pretrain-epochs 5 \
  --teacher-pretrain-batch-size 512 \
  --teacher-pretrain-learning-rate 0.0003 \
  --teacher-scenarios minerals_explore,minerals_sparse,minerals_transition,transition,no_cube_control \
  --teacher-pretrain-only
```

## 16. Train Stage 2: Minerals

Warm-start from the exploration-prior checkpoint.

```bash
python examples/train.py \
  --stage minerals \
  --timesteps "$STAGE2_STEPS" \
  --load runs/stage2_minerals_bc/model.zip \
  --vecnorm runs/stage2_minerals_bc/vecnorm.pkl \
  --reset-reward-stats \
  --save runs/stage2_minerals \
  --seed "$SEED" \
  --n-envs "$N_ENVS" \
  --frame-skip "$FRAME_SKIP" \
  --power-capacity "$POWER_CAPACITY" \
  --horizon medium \
  --scenario minerals_sparse \
  --low-power-threshold 0.35 \
  --locomotion-out-of-power-penalty 1500 \
  --flip-penalty 1500 \
  --tilt-penalty 5 \
  --tilt-threshold-deg 45 \
  --eval-selection-stat median \
  --eval-failure-penalty 10000 \
  --selection-extra-scenarios transition \
  --ignored-cube-penalty 500 \
  --train-scenarios minerals_explore,minerals_sparse,minerals_sparse,minerals_transition,minerals_transition,minerals_fixed_2_sparse,no_cube_control,no_cube_control,cube_intercept_low_power \
  --extra-eval-scenarios minerals_explore,minerals_transition,sparse_visible_low_power,sparse_game,no_cube_control,transition,terrain_fixed_1_0,terrain_fixed_2_0 \
  --eval-freq 25000 \
  --n-eval-episodes 10 \
  --checkpoint-freq 100000 \
  --ent-coef 0.01 \
  --target-kl 0.02
```

Stage 2 still has to preserve Stage 1 survival behavior. `auto`
locomotion shaping enables the power-efficiency guard for minerals, but
cube pickups are no longer directly rewarded in the minerals stage. The
mixed train scenarios make no-cube and sparse-cube mineral exploration
the main task; low-power cube shaping keeps the Stage 1 survival skill
available without teaching the exported rover to wait for cube drops.
Stage 2 trains on the medium horizon because the power-exhaustion and
rollover tail appears after the short horizon has already truncated.
Do not use an oversized global battery-draw penalty as a generic safety
fix. A medium-horizon trial at `1000` reduced flips but increased
out-of-power failures in plain transition, sparse-game, and forced
low-power evaluation. Keep the default draw cost until low-power visible
cube progress can be shaped separately from ordinary exploration.
The visible-cube progress guard (`--low-power-visible-stall-throttle-penalty`)
improved forced intercepts but did not clear transition survival gates; keep
it disabled in the canonical run until a cube-reachability signal is added.

## 17. Evaluate Stage 2

```bash
for scenario in minerals_explore minerals_sparse minerals_transition transition sparse_game terrain_fixed_1_0 terrain_fixed_1_5 terrain_fixed_2_0; do
  python examples/evaluate.py \
    --stage minerals \
    --load runs/stage2_minerals/best/best_model.zip \
    --vecnorm runs/stage2_minerals/best/vecnorm.pkl \
    --scenario "$scenario" \
    --horizon medium \
    --episodes "$EVAL_EPISODES" \
    --seed "$SEED" \
    --frame-skip "$FRAME_SKIP" \
    --power-capacity "$POWER_CAPACITY" \
    | tee "runs/reports/stage2_best_${scenario}_medium.txt"
done
```

Run one long sample.

```bash
python examples/evaluate.py \
  --stage minerals \
  --load runs/stage2_minerals/best/best_model.zip \
  --vecnorm runs/stage2_minerals/best/vecnorm.pkl \
  --scenario terrain_mixed_1_2 \
  --horizon long \
  --episodes 5 \
  --seed "$SEED" \
  --frame-skip "$FRAME_SKIP" \
  --power-capacity "$POWER_CAPACITY" \
  | tee runs/reports/stage2_best_mixed_long.txt
```

Stage 2 can pass when:

- mineral mean beats the Stage 2 power-cube baseline;
- cube pickups and end power do not regress badly;
- `sparse_visible_low_power`, `sparse_game`, and `no_cube_control` remain
  within the accepted Stage 1 envelope;
- distance and flip rate remain acceptable;
- medium and sampled long horizons do not expose a new terminal failure.

## 18. Promote Stage 2

### Coverage-aware Stage 2 candidate

The accepted non-coverage minerals bundle remains the fallback. Create a
coverage candidate by warm-starting it once with the repurposed observation
columns reset:

```bash
python examples/train.py \
  --stage minerals --timesteps 1000000 \
  --load ../models/minerals/model.zip \
  --vecnorm ../models/minerals/vecnorm.pkl \
  --scenario minerals_coverage \
  --train-scenarios minerals_coverage,minerals_sparse,minerals_transition,minerals_fixed_2_sparse,no_cube_control \
  --n-envs 5 --frame-skip 4 \
  --mission-supervisor --coverage-observation \
  --reset-coverage-inputs --reset-reward-stats \
  --save runs/stage2_minerals_coverage
```

Matched evaluation must improve unique cells covered per episode by at least
30% at a matched power budget, keep revisit rate at or below 30%, retain at
least 90% of mineral score, keep forced-visible pickups at or above 0.70, and
stay at 0% out-of-power with no more than 5% flips. Promote only after those
gates pass, adding `--coverage-observation` to the normal promotion command.

Do not gate on `novel_distance_per_100m`: the ratio is capped near 100 and a
baseline that drives short distances without retracing already scores ~94, so
a 30% relative improvement is unreachable for any policy. The coverage
benefit appears as total novel ground covered per episode, not as per-meter
novelty efficiency. (2026-07-12 matched-seed evaluation: the coverage `best`
checkpoint improved unique cells +32-107% at medium horizon and +88% on the
long 1 kWh sparse-game horizon while all other gates passed; the final 1M
checkpoint exceeded the flip gate and was rejected.)

```bash
python examples/promote_model.py \
  --stage minerals \
  --run runs/stage2_minerals \
  --source best \
  --frame-skip "$FRAME_SKIP" \
  --mission-supervisor \
  --runtime-power-capacity 1000 \
  --runtime-supervisor-low-power-enter-fraction 0.15 \
  --runtime-supervisor-low-power-exit-fraction 0.20
```

## 19. Record Stage 3 Baselines

Evaluate the promoted minerals policy under the full reward with beacons
enabled.

```bash
python examples/evaluate.py \
  --stage full \
  --load ../models/minerals/model.zip \
  --vecnorm ../models/minerals/vecnorm.pkl \
  --scenario terrain_mixed_1_2 \
  --horizon medium \
  --episodes "$EVAL_EPISODES" \
  --seed "$SEED" \
  --frame-skip "$FRAME_SKIP" \
  --power-capacity "$POWER_CAPACITY" \
  | tee runs/reports/stage3_minerals_baseline_medium.txt
```

## 20. Train Stage 3: Full

Warm-start from the accepted minerals model.

First evaluate the hierarchical full-mission controller before changing PPO.
With `--mission-supervisor`, the accepted mineral policy remains responsible
for exploration while the shared controller automatically deploys beacons only
after 100 m of travel, at least 75 m apart, and when the scarcity-weighted
surface score is at least 150. This same logic runs in the game.

The July 2026 direct PPO probe was rejected: it learned beacon reward but cut
sparse mineral reward by more than half. The frozen Stage 2 policy plus the
hierarchical controller passed instead:

| Scenario | Minerals | Pickups | Beacons | Flip | Out of power |
|---|---:|---:|---:|---:|---:|
| `minerals_transition` | 7468 | 4.15 | 0.79 | 2% | 0% |
| `sparse_game` | 4884 | 0.37 | 0.36 | 0% | 0% |

These are matched 100-episode, medium-horizon results at seed 2000 and frame
skip 4. Do not fine-tune the exploration policy unless a new run matches or
beats both rows. If PPO training is still required, keep the beacon guard and
use transition plus sparse scenarios for safety-adjusted checkpoint selection.

```bash
python examples/train.py \
  --stage full \
  --timesteps "$STAGE3_STEPS" \
  --load ../models/minerals/model.zip \
  --vecnorm ../models/minerals/vecnorm.pkl \
  --reset-reward-stats \
  --save runs/stage3_full \
  --seed "$SEED" \
  --n-envs "$N_ENVS" \
  --frame-skip "$FRAME_SKIP" \
  --power-capacity "$POWER_CAPACITY" \
  --horizon medium \
  --scenario minerals_transition \
  --train-scenarios minerals_transition,minerals_sparse \
  --extra-eval-scenarios sparse_game \
  --selection-extra-scenarios sparse_game \
  --eval-selection-stat median \
  --eval-failure-penalty 10000 \
  --eval-freq 25000 \
  --n-eval-episodes 10 \
  --checkpoint-freq 100000 \
  --ent-coef 0.01 \
  --target-kl 0.02 \
  --mission-supervisor \
  --supervisor-beacon-first-distance-m 100 \
  --supervisor-beacon-spacing-m 75 \
  --supervisor-beacon-surface-score-threshold 150
```

## 21. Evaluate Stage 3

```bash
for scenario in terrain_mixed_1_2 transition sparse_game terrain_fixed_1_0 terrain_fixed_1_5 terrain_fixed_2_0; do
  python examples/evaluate.py \
    --stage full \
    --load runs/stage3_full/best/best_model.zip \
    --vecnorm runs/stage3_full/best/vecnorm.pkl \
    --scenario "$scenario" \
    --horizon medium \
    --episodes "$EVAL_EPISODES" \
    --seed "$SEED" \
    --frame-skip "$FRAME_SKIP" \
    --power-capacity "$POWER_CAPACITY" \
    | tee "runs/reports/stage3_best_${scenario}_medium.txt"
done
```

Run one long full-stage sample.

```bash
python examples/evaluate.py \
  --stage full \
  --load runs/stage3_full/best/best_model.zip \
  --vecnorm runs/stage3_full/best/vecnorm.pkl \
  --scenario terrain_mixed_1_2 \
  --horizon long \
  --episodes 5 \
  --seed "$SEED" \
  --frame-skip "$FRAME_SKIP" \
  --power-capacity "$POWER_CAPACITY" \
  | tee runs/reports/stage3_best_mixed_long.txt
```

Stage 3 can pass when:

- beacon bonus beats the random and minerals-policy baselines;
- beacons are used intentionally instead of immediately burned;
- mineral and cube metrics do not collapse;
- distance, power survival, and flip rate remain acceptable.

## 22. Promote Stage 3

```bash
python examples/promote_model.py \
  --stage full \
  --run runs/stage3_full \
  --source best \
  --frame-skip "$FRAME_SKIP" \
  --mission-supervisor \
  --runtime-power-capacity 1000 \
  --runtime-supervisor-low-power-enter-fraction 0.15 \
  --runtime-supervisor-low-power-exit-fraction 0.20
```

## 22. Export and Watch Policies

Promotion already creates `model.onnx` and `model.norm.json` in
`../models/<stage>/`. Run the game with a promoted model:

```bash
cargo run -p hylaeanrover_game --release -- \
  --policy ../models/full/model.onnx \
  --mission-supervisor
```

Export an unpromoted checkpoint for inspection:

```bash
python examples/export_policy.py \
  --stage power_cubes \
  --model runs/stage1_power_cubes/best/best_model.zip \
  --vecnorm runs/stage1_power_cubes/best/vecnorm.pkl \
  --out runs/stage1_power_cubes/best/best_model.onnx \
  --frame-skip "$FRAME_SKIP"
```

Then:

```bash
cargo run -p hylaeanrover_game --release -- \
  --policy runs/stage1_power_cubes/best/best_model.onnx \
  --mission-supervisor
```

In the game, press `P` to toggle autopilot and `O` to reload the ONNX
file after exporting a newer checkpoint.

The mission supervisor is the deployment controller for Stage 2 and later.
PPO explores for minerals while power and attitude are healthy. The supervisor
masks cube slots from PPO so visible cubes cannot divert the healthy-power
mineral mission. It also presents constant healthy power to PPO so the learned
100 Wh reserve behavior cannot park the 1 kWh rover before the runtime gate.
The supervisor retains raw cube and power observations for its own decisions.
At 35% training power it switches to a shared deterministic cube intercept when
the nearest visible cube is reachable, or coasts to preserve reserve when no
viable cube exists.
Recovery normally releases at 40%. A detected cube recharge below 40% also
releases recovery and grants up to 75 Wh of exploration before recovery is
re-armed; this prevents the rover from parking immediately after a useful
pickup on the 1 kWh runtime battery. Deterministic intercepts are limited to
the validated 120 m envelope.
Its kinetic tilt guard brakes above 20 degrees while speed exceeds 1 m/s and
releases after the rover slows, so ordinary slopes do not permanently suppress
exploration. Training and acceptance evals must pass `--mission-supervisor` so
they exercise the same controller used by the game.

```bash
python examples/evaluate.py \
  --stage minerals \
  --load runs/stage2_minerals_explore_ppo_v3/best/best_model.zip \
  --vecnorm runs/stage2_minerals_explore_ppo_v3/best/vecnorm.pkl \
  --scenario minerals_transition \
  --horizon medium --episodes 100 --seed 2000 --frame-skip 4 \
  --low-power-threshold 0.35 \
  --mission-supervisor
```

The default target-loss grace is zero. Continuing forward after a cube leaves
the actionable sensor caused rollovers in calibration and is not permitted by
the deployment controller.

Matched 100-episode acceptance results for the `ppo_v3` Stage 2 candidate
(seed 2000, medium horizon, frame skip 4, shaping off):

| Scenario | Minerals | Pickups | Flip | Out of power |
|---|---:|---:|---:|---:|
| `minerals_transition` | 7805 | 3.81 | 3% | 0% |
| `sparse_game` | 3889 | 0.31 | 0% | 0% |
| `sparse_visible_low_power` | 3024 | 1.13 | 0% | 0% |

The unsupervised matched baselines were 8775 minerals with 11% flips and 6%
out of power on `minerals_transition`, and 4679 minerals, 0.29 pickups, and 6%
out of power on `sparse_game`. The supervisor is therefore a deliberate small
throughput trade for a large reduction in terminal failures, while retaining
nonzero sparse pickups and the required forced-visible intercept behavior.

## 23. Final Verification Checklist

Run this before committing promoted models:

```bash
cd python
source .venv/bin/activate

python examples/train_ppo.py
python -m unittest discover tests
python -m black --check \
  hylaeanrover/__init__.py \
  hylaeanrover/wrappers.py \
  hylaeanrover/export.py \
  examples/train.py \
  examples/evaluate.py \
  examples/train_ppo.py \
  examples/export_policy.py \
  examples/promote_model.py \
  tests/test_stage_hardening.py
```

From the repo root:

```bash
cargo fmt --all --check
cargo test --workspace
cargo clippy --workspace --tests
```

## 24. Commit the Accepted Bundles

After all gates pass:

```bash
cd ..
git add models/locomotion models/power_cubes models/minerals models/full
git add python/TRAINING_GUIDE.md docs/rl_stage0_stage1_hardening_plan.md
git status --short
```

Only commit `python/runs/reports` if those reports are intentionally
kept in the repo. The `runs/` directory is normally scratch output and
may require `git add -f python/runs/reports`.

## Scenario Reference

Use these with `--scenario`:

| Scenario | Primary use |
|---|---|
| `terrain_fixed_1_0` | Flat/reference terrain eval |
| `terrain_fixed_1_5` | Mid-height terrain eval |
| `terrain_fixed_2_0` | High terrain eval |
| `terrain_mixed_1_2` | Reset-time terrain randomization |
| `dense_training` | Stage 1 positive pickup signal |
| `bridge_training` | Intermediate density bridge eval |
| `bridge_low_power` | Low-power training bridge before transition |
| `transition` | Bridge from dense training to sparse game |
| `sparse_game` | Deployment-like cube density |
| `sparse_low_power` | Sparse cubes with low-power start |
| `sparse_visible_reset` | Sparse spawn plus one reset-time visible cube |
| `sparse_visible_low_power` | Low-power sparse-visible intercept diagnostic |
| `low_power_start` | Starts near low power with transition cubes |
| `cube_visible_low_power` | Low power with likely visible cube |
| `no_cube_control` | Low-power negative control with no cubes |

Use these with `--cube-shaping`:

| Mode | Main use |
|---|---|
| `off` | Acceptance evals |
| `low_power` | Reward visible-cube range reduction only under low power |
| `intercept` | Reward visible-cube alignment/range reduction before low power |

Use these with `--horizon`:

| Horizon | Physics ticks | Main use |
|---|---:|---|
| `short` | 2000 | Main training horizon |
| `medium` | 7200 | Required acceptance eval horizon |
| `long` | 21600 | Sampled long-run failure check |

Use these with `--cube-spawn-preset` when overriding a scenario:

| Preset | Main use |
|---|---|
| `dense_training` | Frequent reachable cubes |
| `bridge_training` | Intermediate density and search radius |
| `transition` | Near-sparse bridge |
| `sparse_game` | Game-like sparse lifelines |
| `none` | No-cube control |

## Stop Rules

Stop and fix the environment or curriculum before continuing if any of
these happen:

- Stage 0 does not beat random distance by at least 2x on medium mixed
  terrain.
- Stage 0 flips frequently on fixed `1.5` or `2.0` terrain.
- Stage 1 only improves dense pickups and fails transition/sparse evals.
- Stage 1 low-power visible-cube approach rate stays near zero.
- Stage 1 has better shaped return but worse out-of-power rate than
  locomotion.
- Stage 2 improves minerals but loses the Stage 1 power-cube behavior.
- Stage 3 earns beacon bonus by immediately spending all beacons.
- Eval uses a different `FRAME_SKIP`, `POWER_CAPACITY`, model, or
  VecNormalize file than training.
