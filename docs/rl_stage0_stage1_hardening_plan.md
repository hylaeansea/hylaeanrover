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
  `dense_training`, `transition`, `sparse_game`, or `none`.
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
- `no_cube_control`: no artificial pickup reward and no dependency on
  cube-shaping artifacts when cubes are absent.

Required metrics are printed by `evaluate.py`: pickups, end power,
distance after first low-power event, visible low-power steps,
visible-cube approach rate, flip rate, out-of-power rate, and terminal
reason counts.

Suggested comparison loop:

```bash
cd python && source .venv/bin/activate

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

Start Stage 1 dense enough to expose the skill, but always evaluate
transition and sparse scenarios during the run:

```bash
python examples/train.py --stage power_cubes --timesteps 1000000 \
  --load ../models/locomotion/model.zip \
  --vecnorm ../models/locomotion/vecnorm.pkl \
  --reset-reward-stats \
  --save runs/power_cubes \
  --scenario dense_training \
  --extra-eval-scenarios transition,sparse_game,low_power_start,cube_visible_low_power \
  --horizon short \
  --cube-shaping auto
```

If Stage 1 improves only under `dense_training`, continue curriculum
training with `transition` before attempting sparse-game acceptance.

## Verification Checklist

- `cargo test --workspace`
- Rebuild the extension after Rust changes:
  `cd python && source .venv/bin/activate && maturin develop --release`
- `python -m unittest discover tests`
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
