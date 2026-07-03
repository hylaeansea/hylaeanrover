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
runs/stage1_power_cubes
runs/stage1_power_cubes_transition
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
  --preserve-reward-stats \
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

Evaluate the promoted locomotion policy under the Stage 1 reward and
the Stage 1 scenarios. These are the baselines the power-cube policy
must beat.

```bash
for scenario in dense_training transition sparse_game low_power_start cube_visible_low_power no_cube_control; do
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

## 9. Train Stage 1: Power Cubes

Start dense. Keep low-power shaping on during training.

```bash
python examples/train.py \
  --stage power_cubes \
  --timesteps "$STAGE1_STEPS" \
  --load ../models/locomotion/model.zip \
  --vecnorm ../models/locomotion/vecnorm.pkl \
  --reset-reward-stats \
  --save runs/stage1_power_cubes \
  --seed "$SEED" \
  --n-envs "$N_ENVS" \
  --frame-skip "$FRAME_SKIP" \
  --power-capacity "$POWER_CAPACITY" \
  --horizon short \
  --scenario dense_training \
  --extra-eval-scenarios transition,sparse_game,low_power_start,cube_visible_low_power,no_cube_control \
  --cube-spawn-seed "$SEED" \
  --cube-shaping auto \
  --eval-freq 25000 \
  --n-eval-episodes 10 \
  --checkpoint-freq 100000 \
  --ent-coef 0.01 \
  --target-kl 0.02
```

Watch the run:

```bash
tensorboard --logdir runs/stage1_power_cubes/tb
```

## 10. Evaluate Stage 1 Without Shaping

Acceptance evals should use `--cube-shaping off`.

```bash
for scenario in dense_training transition sparse_game low_power_start cube_visible_low_power no_cube_control; do
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
    | tee "runs/reports/stage1_best_${scenario}_medium.txt"
done
```

Run sparse long-horizon samples.

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
  | tee runs/reports/stage1_best_sparse_game_long.txt
```

Stage 1 can pass when it beats the Stage 1 locomotion baselines on:

- cube pickups in `dense_training` and `transition`;
- end power fraction in `transition`, `low_power_start`, and
  `cube_visible_low_power`;
- out-of-power rate in power-pressure scenarios;
- nonzero sparse-game pickups over a meaningful eval batch;
- visible-cube approach rate when power is low;
- no-cube control behavior without fake pickup reward.

## 11. Continue Stage 1 on Transition if Needed

Use this if dense training improves but transition or sparse evals are
still weak.

```bash
python examples/train.py \
  --stage power_cubes \
  --timesteps "$STAGE1_STEPS" \
  --load runs/stage1_power_cubes/best/best_model.zip \
  --vecnorm runs/stage1_power_cubes/best/vecnorm.pkl \
  --preserve-reward-stats \
  --save runs/stage1_power_cubes_transition \
  --seed "$SEED" \
  --n-envs "$N_ENVS" \
  --frame-skip "$FRAME_SKIP" \
  --power-capacity "$POWER_CAPACITY" \
  --horizon short \
  --scenario transition \
  --extra-eval-scenarios dense_training,sparse_game,low_power_start,cube_visible_low_power,no_cube_control \
  --cube-spawn-seed "$SEED" \
  --cube-shaping auto \
  --eval-freq 25000 \
  --n-eval-episodes 10 \
  --checkpoint-freq 100000 \
  --ent-coef 0.01 \
  --target-kl 0.02
```

Evaluate the transition continuation with the same commands from Step 10,
replacing `runs/stage1_power_cubes` with
`runs/stage1_power_cubes_transition`.

## 12. Promote Stage 1

Only promote after the Stage 1 gates pass and the reports are recorded.

```bash
python examples/promote_model.py \
  --stage power_cubes \
  --run runs/stage1_power_cubes \
  --source best \
  --frame-skip "$FRAME_SKIP"
```

If the transition continuation is the accepted run, promote that run
instead:

```bash
python examples/promote_model.py \
  --stage power_cubes \
  --run runs/stage1_power_cubes_transition \
  --source best \
  --frame-skip "$FRAME_SKIP"
```

Do not start minerals until `../models/power_cubes/` comes from a run
that passed the Stage 1 gates.

## 13. Record Stage 2 Baselines

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

## 14. Train Stage 2: Minerals

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

## 15. Evaluate Stage 2

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

## 16. Promote Stage 2

```bash
python examples/promote_model.py \
  --stage minerals \
  --run runs/stage2_minerals \
  --source best \
  --frame-skip "$FRAME_SKIP"
```

## 17. Record Stage 3 Baselines

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

## 18. Train Stage 3: Full

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

## 19. Evaluate Stage 3

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

## 20. Promote Stage 3

```bash
python examples/promote_model.py \
  --stage full \
  --run runs/stage3_full \
  --source best \
  --frame-skip "$FRAME_SKIP"
```

## 21. Export and Watch Policies

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

## 22. Final Verification Checklist

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

## 23. Commit the Accepted Bundles

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
| `transition` | Bridge from dense training to sparse game |
| `sparse_game` | Deployment-like cube density |
| `low_power_start` | Starts near low power with transition cubes |
| `cube_visible_low_power` | Low power with likely visible cube |
| `no_cube_control` | Low-power negative control with no cubes |

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
