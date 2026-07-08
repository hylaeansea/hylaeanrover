# RL Stage 0/1 Hardening Plan

_Created: 2026-07-02_

This plan is the gate before minerals work continues. Stage 0 locomotion
and Stage 1 power cubes must be proven against the actual failure modes
found in review: weak Stage 1 improvement over locomotion, no proof of
low-power sensor use, dense-training vs sparse-game mismatch, fixed
terrain height, short episode horizons, unseeded cube spawns, and stale
tooling/docs.

## Goals

- Keep the policy interface fixed: 41-float observation, `Discrete(10)`
  actions, same PPO/SB3 stack.
- Make Stage 0 robust to terrain height variation.
- Make Stage 1 prove sensor-driven power-cube seeking under power
  pressure, not just opportunistic pickup in a dense cube field.
- Do not promote `models/power_cubes/` or start minerals until the gates
  below pass and the command outputs are recorded.

## Runtime Knobs

Implemented knobs:

- `terrain_height_scale`: native env/core terrain multiplier, fixed at
  construction or varied per reset.
- `power_start_fraction`: native env reset starts the battery partially
  charged for low-power scenarios.
- `cube_spawn_seed`: seeded power-cube spawn RNG, combined with the reset
  seed for reproducible per-episode cube timing and placement.
- `cube_spawn_preset`: Python scenario preset for cube density:
  `dense_training`, `bridge_training`, `transition`, `sparse_game`, or
  `none`.
- Forced diagnostic/training cubes are spawned settled on or near terrain
  instead of at the normal falling spawn height, so a visible forced cube is
  also physically collectible.
- The RL-visible cube sensor filters out non-actionable cubes and keeps the
  fixed 41-float observation interface. Debug `info` fields report nearest
  raw cube height above ground and actionability for diagnostics.
- `--train-scenarios`: optional mixed-curriculum training list assigned
  across vectorized PPO workers. `--n-envs` must be at least the number
  of listed scenarios.
- `--cube-shaping intercept`: training-only visible-cube interception
  shaping for sparse scenarios; acceptance evals still use
  `--cube-shaping off`.
- `cube_intercept`, `cube_intercept_close`, and
  `cube_intercept_low_power`: no-random-spawn, one-forced-cube scenarios
  for learning and evaluating direct sensor-to-action pickup behavior
  before broad power-cube training resumes. `cube_intercept` terminates
  successfully when the native episode pickup count increments.
- `power_idle`: no-random-spawn, low-power no-target scenario used after
  `cube_intercept` to teach the policy not to spend a nearly empty
  battery when the actionable cube sensor is empty.
- Low-power no-target shaping: in `power_idle` and `power_cubes`, motor
  actions get an extra penalty below the low-power threshold when no
  actionable cube is visible; coast/no-op actions get a small reward.
- `sparse_visible_reset` and `sparse_visible_low_power`: sparse-game spawn
  scenarios with one reset-time cube forced into the actual cube sensor
  cone, used to diagnose and train visible-cube interception without
  changing observation/action dimensions.
- `--teacher-pretrain-samples`, `--teacher-scenarios`, and
  `--teacher-pretrain-only`: cross-entropy bootstrap for
  `cube_intercept` and `power_idle`. The BC-only checkpoint must pass
  before PPO fine-tuning is attempted.
- `horizon`: named episode length: `short=2000`, `medium=7200`,
  `long=21600`.

The in-game export path still does not carry dense cube spawn settings
into `.norm.json`. Dense spawns are a training curriculum tool; sparse
performance must be proven by evaluation.

## Stage 0 Gate

Current first-pass Stage 0 promotion history is recorded in
`docs/stage0_locomotion_training_summary.md`.

Stage 0 may advance only when the promoted locomotion policy:

- beats random distance by at least 2-3x on short and medium horizons;
- keeps flip rate low across `fixed_1_0`, `fixed_1_5`, `fixed_2_0`, and
  `mixed_1_2` terrain-height evaluations;
- does not rely on stale observation/action dimensions or incompatible
  VecNormalize stats.
- does not simply sprint until out-of-power; if it plateaus there, train
  with `--locomotion-shaping power_efficiency` before promotion.

Suggested commands:

```bash
cd python && source .venv/bin/activate

python examples/evaluate.py --stage locomotion --random \
  --scenario terrain_mixed_1_2 --horizon medium --episodes 20

python examples/evaluate.py --stage locomotion \
  --load ../models/locomotion/model.zip \
  --vecnorm ../models/locomotion/vecnorm.pkl \
  --scenario terrain_mixed_1_2 --horizon medium --episodes 20
```

## Stage 1 Gate

Stage 1 is now split into `cube_intercept`, `power_idle`, and
`power_cubes`.
The existing Stage 0 checkpoint can bootstrap these runs, but it is not
considered a final locomotion promotion unless the Stage 0 gate above is
met.

The `cube_intercept` policy must first prove that a visible, settled cube
causes committed approach and pickup behavior:

- >=0.8 pickups/episode on forced 15-60 m visible cubes.
- >=0.7 pickups/episode in `cube_intercept_low_power` and out-of-power
  rate <=20%.
- Action traces show commitment toward the cube instead of backing away,
  oscillating, or losing visibility.
- The forced 15 m, 0 degree diagnostic cube is visible and collectible
  with `frame_skip=4`.

Only after those checks pass should `power_idle` training resume from the
`cube_intercept` checkpoint. `power_idle` must show low out-of-power rates
on no-cube low-power episodes before broad `power_cubes` training starts.
Stage 1 may advance only when the trained `power_cubes` policy beats the
promoted locomotion policy under identical seeds and scenarios.

Required comparisons:

- `dense_training`: higher pickups and no catastrophic distance/flip
  regression.
- `transition`: higher pickups, higher end power, lower out-of-power
  rate.
- `sparse_game`: nonzero pickups over a meaningful batch and acceptable
  distance retention.
- `low_power_start`: higher end power and distance after first low-power
  event.
- `cube_visible_low_power`: visible-cube approach behavior, measured by
  `approach_rate`, not just shaped return.
- `sparse_visible_low_power`: reliably approaches and picks up the
  reset-time visible cube with `--cube-shaping off` before sparse-game
  promotion is considered.
- `no_cube_control`: no artificial pickup reward and no dependency on
  cube-shaping artifacts when cubes are absent. During the `power_idle`
  gate, out-of-power should be <=20% on a medium-horizon batch.

Required metrics are printed by `evaluate.py`: pickups, end power,
distance after first low-power event, visible low-power steps,
visible-cube approach rate, flip rate, out-of-power rate, and terminal
reason counts.

Suggested comparison loop:

```bash
cd python && source .venv/bin/activate

for scenario in cube_intercept cube_intercept_low_power sparse_visible_low_power; do
  python examples/evaluate.py --stage cube_intercept \
    --load runs/stage1_cube_intercept/best/best_model.zip \
    --vecnorm runs/stage1_cube_intercept/best/vecnorm.pkl \
    --scenario "$scenario" --horizon medium --episodes 20 \
    --cube-shaping off
done

for scenario in dense_training transition sparse_game low_power_start cube_visible_low_power no_cube_control; do
  python examples/evaluate.py --stage power_cubes \
    --load ../models/locomotion/model.zip \
    --vecnorm ../models/locomotion/vecnorm.pkl \
    --scenario "$scenario" --horizon medium --episodes 20

  python examples/evaluate.py --stage power_cubes \
    --load runs/power_cubes/best/best_model.zip \
    --vecnorm runs/power_cubes/best/vecnorm.pkl \
    --scenario "$scenario" --horizon medium --episodes 20
done
```

Run at least one sampled long-horizon check before promotion:

```bash
python examples/evaluate.py --stage power_cubes \
  --load runs/power_cubes/best/best_model.zip \
  --vecnorm runs/power_cubes/best/vecnorm.pkl \
  --scenario sparse_game --horizon long --episodes 5
```

## Training Guidance

Stop broad Stage 1 training until `cube_intercept` works. Start with one
settled forced cube and no random cube spawns:

```bash
python examples/train.py --stage cube_intercept --timesteps 0 \
  --load ../models/locomotion/model.zip \
  --vecnorm ../models/locomotion/vecnorm.pkl \
  --reset-reward-stats \
  --save runs/stage1_cube_intercept_bc \
  --scenario cube_intercept \
  --terrain-height fixed_1_0 \
  --horizon short \
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

If the BC-only checkpoint passes, optionally run a short conservative PPO
preservation pass and keep whichever checkpoint evaluates better:

```bash
python examples/train.py --stage cube_intercept --timesteps 50000 \
  --load runs/stage1_cube_intercept_bc/model.zip \
  --vecnorm runs/stage1_cube_intercept_bc/vecnorm.pkl \
  --preserve-reward-stats \
  --save runs/stage1_cube_intercept \
  --scenario cube_intercept \
  --terrain-height fixed_1_0 \
  --train-scenarios cube_intercept,cube_intercept_low_power \
  --extra-eval-scenarios cube_intercept_low_power,sparse_visible_low_power,no_cube_control \
  --horizon short \
  --cube-shaping auto \
  --eval-freq 10000 \
  --learning-rate 0.00001 \
  --n-steps 512 \
  --batch-size 128 \
  --n-epochs 2 \
  --clip-range 0.05 \
  --target-kl 0.005
```

Then train low-power no-target discipline from the accepted intercept
checkpoint. This stage has no random cubes and no pickup reward; it exists
only to stop the current broad-stage failure where the rover burns down an
empty sensor view. Use teacher pretraining first: the simple reward-shaped
PPO probe did not reliably move the deterministic motor habit, while the
teacher dataset gives direct no-cube coast labels and visible-cube
intercept labels.

```bash
python examples/train.py --stage power_idle --timesteps 0 \
  --load runs/stage1_cube_intercept/best/best_model.zip \
  --vecnorm runs/stage1_cube_intercept/best/vecnorm.pkl \
  --reset-reward-stats \
  --save runs/stage1_power_idle_bc \
  --scenario power_idle \
  --horizon short \
  --teacher-pretrain-samples 20000 \
  --teacher-pretrain-epochs 10 \
  --teacher-pretrain-batch-size 512 \
  --teacher-scenarios power_idle,cube_intercept_low_power \
  --teacher-pretrain-only
```

Then resume broad `power_cubes` training from the accepted `power_idle`
checkpoint. `power_cubes` uses zero distance reward while this gate is
being hardened, so broad Stage 1 cannot pass by driving until the battery
dies; travel reward returns in later stages.

```bash
python examples/train.py --stage power_cubes --timesteps 1000000 \
  --load runs/stage1_power_idle_bc/best/best_model.zip \
  --vecnorm runs/stage1_power_idle_bc/best/vecnorm.pkl \
  --reset-reward-stats \
  --save runs/power_cubes \
  --n-envs 6 \
  --scenario sparse_visible_low_power \
  --train-scenarios dense_training,bridge_low_power,transition,sparse_visible_low_power,sparse_low_power,no_cube_control \
  --extra-eval-scenarios sparse_visible_low_power,sparse_game,no_cube_control \
  --horizon short \
  --cube-shaping intercept
```

If `power_cubes` improves only under `dense_training`, continue curriculum
training with bridge and sparse-visible mixes before attempting
sparse-game acceptance.

## Verification Checklist

- `cargo test --workspace`
- Rebuild the extension after Rust changes:
  `cd python && source .venv/bin/activate && maturin develop --release`
- `python -m unittest discover tests`
- `python -m unittest tests.test_stage_hardening`
- Targeted no-shaping evals for `cube_intercept`,
  `cube_intercept_low_power`, `power_idle`, `sparse_visible_low_power`,
  `sparse_game`, and `no_cube_control`.
- `python examples/train_ppo.py`
- Stage 0 and Stage 1 eval tables recorded in this file or
  `docs/rl_training_plan.md`.

## Promotion Rule

Only promote the power-cube bundle after Stage 1 gates pass:

```bash
python examples/promote_model.py --stage power_cubes --run runs/power_cubes
git add ../models/power_cubes
```

Do not promote or train minerals from a Stage 1 run that merely improves
shaped return or dense-preset pickups. The accepted policy must show
power-preserving, sensor-driven behavior under low-power and sparse/near
sparse conditions.
