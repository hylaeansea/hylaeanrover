# Curated models

The best-so-far policy for each curriculum stage, committed to the repo
so anyone can resume training, run the in-game autopilot, or compare
against it. The bundles are tiny (~240 KB/stage), so plain git is fine
(no LFS needed).

## Layout

```
models/<stage>/
├── model.zip        SB3 PPO policy — resume training or re-export
├── vecnorm.pkl      matching VecNormalize observation stats
├── model.onnx       policy network for the in-game autopilot
└── model.norm.json  obs stats + frame-skip for the autopilot
```

`<stage>` is one of `locomotion`, `power_cubes`, `minerals`, `full`.

## Promoting a run into here

After a training run you're happy with (see `python/README.md`):

```bash
cd python && source .venv/bin/activate
python examples/promote_model.py --stage locomotion --run runs/stage0
# add --frame-skip K if you trained with frame-skip; --source final to
# promote the last model instead of the eval-best one.
git add ../models/locomotion && git commit -m "Promote locomotion model"
```

`promote_model.py` copies the chosen `model.zip` + `vecnorm.pkl` here and
regenerates `model.onnx` + `model.norm.json` from them.

Hierarchical mineral/full candidates must also pass `--mission-supervisor`.
That requirement is written into `model.norm.json`, and the game enables the
shared controller automatically when loading the promoted ONNX bundle. The
sidecar also keeps beacon behavior stage-correct: disabled for `minerals`,
enabled for `full`.

```bash
python examples/promote_model.py \
  --stage minerals --run runs/stage2_minerals_explore_ppo_v3 \
  --source best --frame-skip 4 --mission-supervisor \
  --runtime-power-capacity 1000 \
  --runtime-supervisor-low-power-enter-fraction 0.15 \
  --runtime-supervisor-low-power-exit-fraction 0.20
python examples/promote_model.py \
  --stage full --run runs/stage2_minerals_explore_ppo_v3 \
  --source best --frame-skip 4 --mission-supervisor \
  --runtime-power-capacity 1000 \
  --runtime-supervisor-low-power-enter-fraction 0.15 \
  --runtime-supervisor-low-power-exit-fraction 0.20
```

The sidecar records both `training_power_capacity_wh=100` and
`runtime_power_capacity_wh=1000`. The policy still observes only power
fraction; the split keeps power pressure useful during training without
shrinking the normal game battery. At runtime, a detected cube recharge can
release low-power preservation even when the battery remains below the runtime
20% recovery exit. The promoted 1 kWh bundles enter recovery at 15% (150 Wh),
while 100 Wh training retains its 35%/40% curriculum thresholds. The supervisor
then grants PPO up to 75 Wh of exploration before re-arming preservation, so a
pickup funds continued mineral search
instead of leaving the rover parked. The supervisor also masks cube slots from
the deployed PPO observation and presents constant healthy power. This keeps
PPO focused on minerals instead of applying its 100 Wh reserve behavior to the
1 kWh battery, while the supervisor retains raw cube/power state for recovery.

Do not promote `models/power_cubes/` until the Stage 1 gates in
[`../docs/rl_stage0_stage1_hardening_plan.md`](../docs/rl_stage0_stage1_hardening_plan.md)
pass. That means the policy must beat the promoted locomotion policy on
pickup rate, end power, out-of-power rate, sparse-game behavior, and
low-power visible-cube approach metrics.

Coverage-aware mineral/full candidates must be promoted with
`--coverage-observation`. This writes `coverage_version=1` to the sidecar so
the game replaces PPO's hidden cube slots with the same 18 frontier features
used in training. Existing promoted bundles intentionally omit that key and
remain unchanged until a coverage candidate passes its acceptance gates.

## Using a promoted model

Resume training from the stage's best (e.g. continue locomotion, or warm-
start the next stage):

```bash
python examples/train.py --stage power_cubes --timesteps 1000000 \
    --load models/locomotion/model.zip --vecnorm models/locomotion/vecnorm.pkl \
    --reset-reward-stats \
    --save runs/power_cubes
```

Use `--preserve-reward-stats` instead when continuing the same stage from
one of its own checkpoints.

Evaluate it:

```bash
python examples/evaluate.py --stage locomotion \
    --load models/locomotion/model.zip --vecnorm models/locomotion/vecnorm.pkl
```

Watch it in the game:

```bash
cargo run -p hylaeanrover_game --release -- --policy models/full/model.onnx
```

Promoted bundles that record `mission_supervisor: true` do not require a
separate CLI flag. `--mission-supervisor` remains available for older exports.

## Note on portability

`model.zip` / `vecnorm.pkl` are Python pickles — resuming training needs
compatible Stable-Baselines3 / PyTorch / NumPy versions. The `model.onnx`
bundle is self-contained and version-independent, so the autopilot works
regardless.
