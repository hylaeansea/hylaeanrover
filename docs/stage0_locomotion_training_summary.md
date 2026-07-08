# Stage 0 Locomotion Training Summary

_Created: 2026-07-03_

This document records how the first-pass Stage 0 locomotion model was
trained, evaluated, and promoted for pipeline validation. The promoted
model is good enough to unblock Stage 1 power-cube training, but it is
not final-quality locomotion.

## Outcome

Promoted bundle:

```text
models/locomotion/model.zip
models/locomotion/vecnorm.pkl
models/locomotion/model.onnx
models/locomotion/model.norm.json
```

Promotion source:

```text
python/runs/stage0_locomotion_short_rebalance/model.zip
python/runs/stage0_locomotion_short_rebalance/vecnorm.pkl
```

The promoted SB3 model and VecNormalize stats match the final
short-rebalance run:

```text
2147037133bb480d7c797fdd825205fb5805e6db132f07d3d88971a6da9a1977  models/locomotion/model.zip
2147037133bb480d7c797fdd825205fb5805e6db132f07d3d88971a6da9a1977  python/runs/stage0_locomotion_short_rebalance/model.zip
55b227d13f4859675d686d956138bbe2204cc76c21e6d82b13dce04be87ad197  models/locomotion/vecnorm.pkl
55b227d13f4859675d686d956138bbe2204cc76c21e6d82b13dce04be87ad197  python/runs/stage0_locomotion_short_rebalance/vecnorm.pkl
```

Promotion command:

```bash
cd python
python examples/promote_model.py \
  --stage locomotion \
  --run runs/stage0_locomotion_short_rebalance \
  --source final \
  --frame-skip "$FRAME_SKIP"
```

## Acceptance Decision

The strict Stage 0 checker still reports `FAIL` because short-horizon
distance is slightly below the 2x random-distance floor on three
terrains. We accepted this model as a first-pass pipeline-validation
checkpoint because:

- medium-horizon distance clears the 2x random baseline on all terrains;
- short-horizon distance is close to the 2x floor;
- flip rate is within the 10% cap;
- sampled long-horizon eval had no flips;
- Stage 1 is specifically intended to improve power behavior with
  sensor-driven power-cube collection.

Known caveats:

- short-horizon flat/mixed distance is not quite at the strict gate;
- out-of-power remains high on medium and long horizons;
- this should be revisited if Stage 1 cannot improve power survival.

## Fixed Training Settings

Use the same settings when reproducing this training path:

```bash
cd python
source .venv/bin/activate

export SEED=42
export FRAME_SKIP=4
export N_ENVS=8
export POWER_CAPACITY=100
export EVAL_EPISODES=20
export STAGE0_STEPS=2000000
```

Keep `FRAME_SKIP` and `POWER_CAPACITY` consistent across training,
evaluation, export, promotion, and Stage 1 warm-start.

## Setup and Verification

```bash
cd python
source .venv/bin/activate
maturin develop --release
python examples/train_ppo.py
python -m unittest discover tests
```

From the repo root:

```bash
cargo test --workspace
cargo clippy --workspace --tests
```

## Baseline Reports

Random baselines were recorded once and copied into each Stage 0 report
directory before running `check_stage0_gate.py`.

```bash
cd python
mkdir -p runs/reports

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

Recorded random distances:

| Scenario | Random Distance |
|---|---:|
| `terrain_fixed_1_0` | 126.56 |
| `terrain_fixed_1_5` | 126.94 |
| `terrain_fixed_2_0` | 130.13 |
| `terrain_mixed_1_2` | 128.63 |

## Training History

The sequence below is the actual path used to reach the promoted
checkpoint. Early runs are retained here because they explain why the
later power-efficiency shaping and short rebalance were added.

| Step | Run Directory | Source | Main Change | Result |
|---:|---|---|---|---|
| 1 | `runs/stage0_locomotion` | scratch | Short mixed-terrain locomotion, no power-efficiency shaping | Close but below gate; medium ratios around 1.9x and high out-of-power |
| 2 | `runs/stage0_locomotion_continue1` | `stage0_locomotion/best` | Continued with lower LR after frequent KL early stops | Did not clear gate; still sprinted to out-of-power |
| 3 | `runs/stage0_locomotion_medium_ft` | `stage0_locomotion_continue1/best` | Medium-horizon fine-tune | Medium improved on some terrains, but fixed 1.5 flip and ratios failed |
| 4 | `runs/stage0_locomotion_power_efficiency` | `stage0_locomotion_medium_ft/best` | Added `power_efficiency` shaping | Three medium terrains crossed 2x; fixed 2.0 still failed |
| 5 | `runs/stage0_locomotion_fixed_2_recovery` | `stage0_locomotion_power_efficiency/best` | Stronger power shaping on fixed 2.0 | All medium terrains crossed 2x; short ratios fell below 2x |
| 6 | `runs/stage0_locomotion_short_rebalance` | `stage0_locomotion_fixed_2_recovery/best` | Short-horizon rebalance with softer power shaping | Final checkpoint accepted provisionally for pipeline validation |

## Commands Used

### 1. Initial Scratch Run

Under the current CLI, pass `--locomotion-shaping off` to reproduce the
original unshaped behavior.

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
  --locomotion-shaping off \
  --extra-eval-scenarios terrain_fixed_1_0,terrain_fixed_1_5,terrain_fixed_2_0 \
  --eval-freq 25000 \
  --n-eval-episodes 10 \
  --checkpoint-freq 100000 \
  --ent-coef 0.01 \
  --target-kl 0.02
```

### 2. Continue With Lower Learning Rate

```bash
python examples/train.py \
  --stage locomotion \
  --timesteps "$STAGE0_STEPS" \
  --load runs/stage0_locomotion/best/best_model.zip \
  --vecnorm runs/stage0_locomotion/best/vecnorm.pkl \
  --save runs/stage0_locomotion_continue1 \
  --seed "$SEED" \
  --n-envs "$N_ENVS" \
  --frame-skip "$FRAME_SKIP" \
  --power-capacity "$POWER_CAPACITY" \
  --horizon short \
  --scenario terrain_mixed_1_2 \
  --locomotion-shaping off \
  --extra-eval-scenarios terrain_fixed_1_0,terrain_fixed_1_5,terrain_fixed_2_0 \
  --eval-freq 25000 \
  --n-eval-episodes 10 \
  --checkpoint-freq 100000 \
  --learning-rate 0.0001 \
  --ent-coef 0.01 \
  --target-kl 0.02
```

### 3. Medium-Horizon Fine-Tune

```bash
python examples/train.py \
  --stage locomotion \
  --timesteps 1000000 \
  --load runs/stage0_locomotion_continue1/best/best_model.zip \
  --vecnorm runs/stage0_locomotion_continue1/best/vecnorm.pkl \
  --save runs/stage0_locomotion_medium_ft \
  --seed "$SEED" \
  --n-envs "$N_ENVS" \
  --frame-skip "$FRAME_SKIP" \
  --power-capacity "$POWER_CAPACITY" \
  --horizon medium \
  --scenario terrain_mixed_1_2 \
  --locomotion-shaping off \
  --extra-eval-scenarios terrain_fixed_1_0,terrain_fixed_1_5,terrain_fixed_2_0 \
  --eval-freq 25000 \
  --n-eval-episodes 10 \
  --checkpoint-freq 100000 \
  --learning-rate 0.00005 \
  --n-epochs 5 \
  --ent-coef 0.01 \
  --target-kl 0.02
```

### 4. Power-Efficiency Recovery

This introduced explicit shaping for coasting distance, power draw,
power recovery, and out-of-power termination.

```bash
python examples/train.py \
  --stage locomotion \
  --timesteps 1000000 \
  --load runs/stage0_locomotion_medium_ft/best/best_model.zip \
  --vecnorm runs/stage0_locomotion_medium_ft/best/vecnorm.pkl \
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

### 5. Fixed 2.0 Recovery

This focused the policy on the terrain that was still failing after the
first power-efficiency recovery.

```bash
python examples/train.py \
  --stage locomotion \
  --timesteps 750000 \
  --load runs/stage0_locomotion_power_efficiency/best/best_model.zip \
  --vecnorm runs/stage0_locomotion_power_efficiency/best/vecnorm.pkl \
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

### 6. Short-Horizon Rebalance

This softened the power shaping and moved back to short horizon so the
policy did not become too conservative for short evaluation.

```bash
python examples/train.py \
  --stage locomotion \
  --timesteps 500000 \
  --load runs/stage0_locomotion_fixed_2_recovery/best/best_model.zip \
  --vecnorm runs/stage0_locomotion_fixed_2_recovery/best/vecnorm.pkl \
  --save runs/stage0_locomotion_short_rebalance \
  --seed "$SEED" \
  --n-envs "$N_ENVS" \
  --frame-skip "$FRAME_SKIP" \
  --power-capacity "$POWER_CAPACITY" \
  --horizon short \
  --scenario terrain_mixed_1_2 \
  --locomotion-shaping power_efficiency \
  --locomotion-coast-bonus 0.35 \
  --locomotion-power-draw-penalty 60 \
  --locomotion-power-recovery-reward 30 \
  --locomotion-out-of-power-penalty 120 \
  --extra-eval-scenarios terrain_fixed_1_0,terrain_fixed_1_5,terrain_fixed_2_0 \
  --eval-freq 25000 \
  --n-eval-episodes 10 \
  --checkpoint-freq 100000 \
  --learning-rate 0.00003 \
  --n-epochs 5 \
  --ent-coef 0.02 \
  --target-kl 0.02
```

## Final Evaluation

Evaluate the final checkpoint, not the eval-best checkpoint:

```bash
export STAGE0_REPORT_DIR=runs/reports/stage0_short_rebalance_final
mkdir -p "$STAGE0_REPORT_DIR"
cp runs/reports/stage0_random_*.txt "$STAGE0_REPORT_DIR"/

for horizon in short medium; do
  for scenario in terrain_fixed_1_0 terrain_fixed_1_5 terrain_fixed_2_0 terrain_mixed_1_2; do
    python examples/evaluate.py \
      --stage locomotion \
      --load runs/stage0_locomotion_short_rebalance/model.zip \
      --vecnorm runs/stage0_locomotion_short_rebalance/vecnorm.pkl \
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
  --load runs/stage0_locomotion_short_rebalance/model.zip \
  --vecnorm runs/stage0_locomotion_short_rebalance/vecnorm.pkl \
  --scenario terrain_mixed_1_2 \
  --horizon long \
  --episodes 5 \
  --seed "$SEED" \
  --frame-skip "$FRAME_SKIP" \
  --power-capacity "$POWER_CAPACITY" \
  | tee "$STAGE0_REPORT_DIR/stage0_best_mixed_long.txt"

python examples/check_stage0_gate.py --reports-dir "$STAGE0_REPORT_DIR"
```

Final strict-check output:

| Scenario | Random | Short Distance | Short Ratio | Medium Distance | Medium Ratio | Medium Flip | Medium Out-of-Power |
|---|---:|---:|---:|---:|---:|---:|---:|
| `terrain_fixed_1_0` | 126.56 | 231.81 | 1.83x | 257.54 | 2.03x | 0% | 100% |
| `terrain_fixed_1_5` | 126.94 | 244.88 | 1.93x | 273.34 | 2.15x | 10% | 85% |
| `terrain_fixed_2_0` | 130.13 | 261.47 | 2.01x | 299.20 | 2.30x | 0% | 95% |
| `terrain_mixed_1_2` | 128.63 | 250.59 | 1.95x | 281.27 | 2.19x | 10% | 90% |

Long sample:

```text
distance=278.43
flip=0.0%
out_of_power=100.0%
terminal reasons={'out_of_power': 5}
```

Strict checker result:

```text
OVERALL: FAIL - do not promote Stage 0 yet.
```

Human decision:

```text
Promote as first-pass Stage 0 for pipeline validation.
Do not treat this as final-quality locomotion.
```

## Reproduction Notes

- The original scratch and continuation runs were done before
  `--locomotion-shaping` existed. Under current code, pass
  `--locomotion-shaping off` for those early runs to reproduce the same
  training condition.
- The current `train.py --locomotion-shaping auto` enables
  `power_efficiency` for locomotion by default. This is the desired
  behavior for new Stage 0 work, but it does not match the earliest
  exploratory runs.
- Current same-stage continuation should preserve reward normalization.
  The updated `train.py` auto-detects obvious same-stage paths, but pass
  `--preserve-reward-stats` when reproducing continuation commands to make
  the intent explicit.
- The final promoted model uses the final checkpoint from
  `runs/stage0_locomotion_short_rebalance`, not
  `runs/stage0_locomotion_short_rebalance/best/best_model.zip`.
- Stage 1 should begin from the promoted `models/locomotion/` bundle and
  should explicitly test whether power-cube seeking improves survival.
