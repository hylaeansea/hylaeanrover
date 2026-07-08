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

## 15. Train Stage 2: Minerals

Warm-start from the accepted power-cube model.

```bash
python examples/train.py \
  --stage minerals \
  --timesteps "$STAGE2_STEPS" \
  --load ../models/power_cubes/model.zip \
  --vecnorm ../models/power_cubes/vecnorm.pkl \
  --reset-reward-stats \
  --save runs/stage2_minerals \
  --seed "$SEED" \
  --n-envs "$N_ENVS" \
  --frame-skip "$FRAME_SKIP" \
  --power-capacity "$POWER_CAPACITY" \
  --horizon short \
  --scenario terrain_mixed_1_2 \
  --extra-eval-scenarios transition,sparse_game,terrain_fixed_1_0,terrain_fixed_2_0 \
  --eval-freq 25000 \
  --n-eval-episodes 10 \
  --checkpoint-freq 100000 \
  --ent-coef 0.01 \
  --target-kl 0.02
```

## 16. Evaluate Stage 2

```bash
for scenario in terrain_mixed_1_2 transition sparse_game terrain_fixed_1_0 terrain_fixed_1_5 terrain_fixed_2_0; do
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
- distance and flip rate remain acceptable;
- medium and sampled long horizons do not expose a new terminal failure.

## 17. Promote Stage 2

```bash
python examples/promote_model.py \
  --stage minerals \
  --run runs/stage2_minerals \
  --source best \
  --frame-skip "$FRAME_SKIP"
```

## 18. Record Stage 3 Baselines

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

## 19. Train Stage 3: Full

Warm-start from the accepted minerals model.

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
  --horizon short \
  --scenario terrain_mixed_1_2 \
  --extra-eval-scenarios transition,sparse_game,terrain_fixed_1_0,terrain_fixed_2_0 \
  --eval-freq 25000 \
  --n-eval-episodes 10 \
  --checkpoint-freq 100000 \
  --ent-coef 0.01 \
  --target-kl 0.02
```

## 20. Evaluate Stage 3

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

## 21. Promote Stage 3

```bash
python examples/promote_model.py \
  --stage full \
  --run runs/stage3_full \
  --source best \
  --frame-skip "$FRAME_SKIP"
```

## 22. Export and Watch Policies

Promotion already creates `model.onnx` and `model.norm.json` in
`../models/<stage>/`. Run the game with a promoted model:

```bash
cargo run -p hylaeanrover_game --release -- \
  --policy ../models/full/model.onnx
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
  --policy runs/stage1_power_cubes/best/best_model.onnx
```

In the game, press `P` to toggle autopilot and `O` to reload the ONNX
file after exporting a newer checkpoint.

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
