# hylaeanrover — Python / RL bindings

Gymnasium-compatible RL environment wrapping the Hylaean Rover Bevy
simulator. Drives a headless physics + sensor stack from Python so
Stable Baselines 3 (or any other RL library) can `reset()` / `step()`
episodes against it.

## Setup

You need a Rust toolchain (`rustup`) and Python ≥ 3.9.

Keep the virtualenv **in this `python/` directory** (`python/.venv`). That
one rule avoids a whole class of confusion: `maturin develop` auto-finds a
`.venv` in the current or a parent folder, and everything — the extension,
SB3, your `runs/` — lives next to the code that uses it. Don't train from a
global interpreter or a second checkout; see "Which binary am I running?"
below for why.

### Recommended: `uv`

```bash
cd python
uv venv --python 3.13
source .venv/bin/activate
uv pip install 'maturin[patchelf]>=1.4'
# builds the extension and installs the [sb3] extra (sb3, torch,
# tensorboard, onnx) + base deps into .venv. --uv is needed because
# `uv venv` makes a pip-less venv; drop it after `uv pip install pip`.
maturin develop --release --uv --extras sb3
```

### Plain `pip` alternative

```bash
cd python
python -m venv .venv && source .venv/bin/activate
pip install 'maturin[patchelf]>=1.4'
pip install -e '.[sb3]'      # gymnasium + numpy + sb3 + torch + tensorboard + onnx
maturin develop --release
```

`maturin develop` compiles `crates/hylaeanrover_py` and drops the resulting
`_native.abi3.so` into `python/hylaeanrover/` (the `hylaeanrover._native`
submodule), then installs the `hylaeanrover` package editable into the
active venv. Re-run it after any change to the Rust crates.

### Which binary am I running?

The Rust extension is a compiled `.so`; editing the Rust source does nothing
until you rebuild, and a `python` from the wrong venv/checkout will silently
import a stale one. To check what's actually loaded:

```bash
python -c "import hylaeanrover, os; p = hylaeanrover._native.__file__; print(p, os.path.getmtime(p))"
```

The path should be *this* checkout's `python/hylaeanrover/_native.abi3.so`
with a recent mtime. `examples/train.py` prints the same line at startup, so
every training log records which binary produced it.

## Quick start

```python
from hylaeanrover import RoverEnv

env = RoverEnv(seed=42, max_steps=2000)
obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()  # 0..9
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
env.close()
```

End-to-end smoke test (gymnasium contract check + tiny PPO run):

```bash
python examples/train_ppo.py
```

## Staged training (curriculum)

The intended way to actually train is in stages that share one fixed
observation/action space and differ only in the reward, so each stage's
policy weights warm-start the next. See
[`docs/rl_training_plan.md`](../docs/rl_training_plan.md) for the full
design. In short:

```bash
# Stage 0 — locomotion (drive far, stay upright)
python examples/evaluate.py --stage locomotion --random      # baseline
python examples/train.py --stage locomotion --timesteps 1000000 \
    --scenario terrain_mixed_1_2 --save runs/stage0
python examples/evaluate.py --stage locomotion --load runs/stage0/model.zip \
    --vecnorm runs/stage0/vecnorm.pkl

# Stage 1A — cube intercept (one actionable forced cube, warm-started from stage 0)
python examples/train.py --stage cube_intercept --timesteps 0 \
    --load runs/stage0/model.zip --vecnorm runs/stage0/vecnorm.pkl \
    --reset-reward-stats \
    --scenario cube_intercept \
    --teacher-pretrain-samples 20000 \
    --teacher-pretrain-epochs 30 \
    --teacher-scenarios cube_intercept \
    --teacher-pretrain-only \
    --save runs/stage1_cube_intercept_bc

# Optional conservative PPO preservation pass after the BC-only checkpoint passes
python examples/train.py --stage cube_intercept --timesteps 50000 \
    --load runs/stage1_cube_intercept_bc/model.zip \
    --vecnorm runs/stage1_cube_intercept_bc/vecnorm.pkl \
    --preserve-reward-stats \
    --scenario cube_intercept \
    --train-scenarios cube_intercept,cube_intercept_low_power \
    --extra-eval-scenarios cube_intercept_low_power,sparse_visible_low_power,no_cube_control \
    --learning-rate 0.00001 --n-steps 512 --batch-size 128 \
    --n-epochs 2 --clip-range 0.05 --target-kl 0.005 \
    --save runs/stage1_cube_intercept

# Stage 1B — power idle (low-power no-target discipline)
python examples/train.py --stage power_idle --timesteps 0 \
    --load runs/stage1_cube_intercept/model.zip \
    --vecnorm runs/stage1_cube_intercept/vecnorm.pkl \
    --reset-reward-stats \
    --scenario power_idle \
    --teacher-pretrain-samples 20000 \
    --teacher-pretrain-epochs 10 \
    --teacher-pretrain-batch-size 512 \
    --teacher-scenarios power_idle,cube_intercept_low_power \
    --teacher-pretrain-only \
    --save runs/stage1_power_idle_bc

# Stage 1C — power cubes (mixed dense/bridge/sparse, warm-started from power idle)
python examples/train.py --stage power_cubes --timesteps 1000000 \
    --load runs/stage1_power_idle_bc/model.zip \
    --vecnorm runs/stage1_power_idle_bc/vecnorm.pkl \
    --reset-reward-stats \
    --n-envs 6 \
    --scenario sparse_visible_low_power \
    --train-scenarios dense_training,bridge_low_power,transition,sparse_visible_low_power,sparse_low_power,no_cube_control \
    --extra-eval-scenarios sparse_visible_low_power,sparse_game,no_cube_control \
    --cube-shaping intercept \
    --save runs/stage1_power_cubes

# Stage 2A — minerals exploration prior (drive while powered, survive low-power)
python examples/train.py --stage minerals --timesteps 0 \
    --load runs/stage1_power_cubes/model.zip \
    --vecnorm runs/stage1_power_cubes/vecnorm.pkl \
    --reset-reward-stats \
    --scenario minerals_explore \
    --low-power-threshold 0.35 \
    --teacher-pretrain-samples 75000 \
    --teacher-pretrain-epochs 5 \
    --teacher-pretrain-batch-size 512 \
    --teacher-scenarios minerals_explore,minerals_sparse,minerals_transition,transition,no_cube_control \
    --teacher-pretrain-only \
    --save runs/stage2_minerals_bc

# Stage 2B — drive + minerals (warm-started from the exploration prior)
python examples/train.py --stage minerals --timesteps 1000000 \
    --load runs/stage2_minerals_bc/model.zip \
    --vecnorm runs/stage2_minerals_bc/vecnorm.pkl \
    --reset-reward-stats \
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
    --extra-eval-scenarios minerals_explore,minerals_transition,sparse_visible_low_power,sparse_game,no_cube_control,transition \
    --save runs/stage2_minerals

# Stage 3 — full mission incl. beacons (warm-started from stage 2)
python examples/train.py --stage full --timesteps 1000000 \
    --load runs/stage2/model.zip --vecnorm runs/stage2/vecnorm.pkl \
    --reset-reward-stats --save runs/stage3

tensorboard --logdir runs/
```

`RoverEnv(..., beacons_enabled=False)` makes action index 9 an inert
no-op (and disables the `beacons_deployed` game-over) — used by every
stage but `full` so the action space stays `Discrete(10)` throughout.
Forced diagnostic/training cubes are spawned settled near terrain so the
XZ cube sensor and 3D pickup rule agree. Airborne or otherwise
non-actionable cubes are hidden from the RL-visible cube slots; debug
`info` fields still report the nearest raw cube height/actionable state
for diagnostics without changing `OBS_DIM`.

The `cube_intercept` stage disables random cube spawns and trains on one
visible, settled cube. The `power_idle` stage then trains low-power
no-target behavior with no random cubes and no pickup reward, so the policy
does not burn down a nearly empty battery when the actionable cube sensor
is empty. The `power_cubes` stage then mixes dense, bridge, transition,
sparse-visible, and sparse-game scenarios. In `power_cubes`, distance
reward is off so visible-cube pickup and power behavior are hardened
before travel reward returns in later stages. Dense spawns are
training-only: the game keeps its own sparse, periodic spawn rate, and the
exported autopilot bundle deliberately does not carry the training density
over ("train dense, deploy sparse"). The reward shaping lives in
`hylaeanrover.wrappers.StagedRewardWrapper`.

Before moving from `power_cubes` to `minerals`, run the Stage 0/1 gates
in [`../docs/rl_stage0_stage1_hardening_plan.md`](../docs/rl_stage0_stage1_hardening_plan.md).
The key scenario knobs are:

- `--scenario`: named eval/train presets such as `dense_training`,
  `bridge_training`, `bridge_low_power`, `transition`,
  `cube_intercept`, `cube_intercept_close`, `power_idle`,
  `cube_intercept_low_power`, `sparse_game`, `sparse_low_power`,
  `sparse_visible_reset`, `sparse_visible_low_power`, `low_power_start`,
  `cube_visible_low_power`, and `no_cube_control`.
- `--train-scenarios`: comma-separated scenario names assigned across
  vectorized training workers for mixed-curriculum PPO runs. `--n-envs`
  must be at least the number of listed scenarios.
- `--horizon`: `short` (2000), `medium` (7200), or `long` (21600)
  physics ticks.
- `--terrain-height`: fixed value, `min:max` range, or preset such as
  `mixed_1_2`.
- `--cube-spawn-preset`: `dense_training`, `bridge_training`,
  `transition`, `sparse_game`, or `none`.
- `--power-start-fraction`: starts an episode with a partial battery for
  low-power checks.
- `--forced-cube-distance` and `--forced-cube-bearing-deg`: add one
  reset-time cube in the cube sensor cone for sparse-visible diagnostics.
- `--cube-shaping`: `low_power` rewards closing on visible cubes after
  the battery is low; `intercept` also rewards range reduction and
  heading commitment before low power. Use `off` for acceptance evals.
- `--cube-approach-reward`, `--cube-heading-reward`, and
  `--ignored-cube-penalty`: tune Stage 1 training-only cube approach
  shaping.
- `--loss-of-sight-penalty` and `--intercept-failure-penalty`: tune the
  `cube_intercept` penalties for abandoning a visible cube or timing out
  without pickup.
- `--teacher-pretrain-samples`, `--teacher-pretrain-epochs`,
  `--teacher-pretrain-batch-size`, `--teacher-scenario`, and
  `--teacher-scenarios`: optional
  teacher-assisted policy-head bootstrap for `cube_intercept`, low-power
  idle bootstrap for `power_idle`, and mineral exploration bootstrap for
  `minerals`. It is a bootstrap only; promotion still depends on
  no-shaping eval gates and any later PPO fine-tuning.
- `--locomotion-shaping`: `power_efficiency` adds pacing shaping so
  coasting/regen is learnable. `auto` enables it for `locomotion`,
  `power_idle`, `power_cubes`, and `minerals`.
- `--locomotion-coast-bonus`, `--locomotion-power-draw-penalty`,
  `--locomotion-power-recovery-reward`, and
  `--locomotion-out-of-power-penalty`: tune Stage 0 pacing pressure when
  the policy plateaus as an out-of-power sprint. Stage 2 uses a larger
  out-of-power penalty during PPO so high-mineral terminal episodes do
  not dominate checkpoint selection.
- `--flip-penalty`: tune the terminal penalty when the rover flips; use a
  larger value for Stage 2 PPO when transition-density mineral runs trade
  safety for reward.
- `--low-power-no-target-throttle-penalty` and
  `--low-power-no-target-coast-reward`: tune the extra `power_idle` /
  `power_cubes` / `minerals` pressure for low-power steps where no
  actionable cube is visible.
- `--low-power-visible-stall-throttle-penalty`: experimental low-power
  penalty for motor actions that neither close on nor align with a visible
  cube. It improves forced intercepts but is not yet a transition-safe
  Stage 2 default.
- `--eval-selection-stat`, `--eval-failure-penalty`, and
  `--selection-extra-scenarios`: select the saved checkpoint on robust
  return after failure cost, using the worst score across the named
  transition distributions.
- `--reset-reward-stats` / `--preserve-reward-stats`: reset reward
  normalization for stage transitions, preserve it for same-stage
  continuation.

### Speeding up training

The bottleneck is the physics sim (CPU), not the tiny MLP policy — keep
PPO on CPU. Two `train.py` flags help:

- `--n-envs N` runs N simulations in parallel via `SubprocVecEnv`
  (separate processes — the Bevy `App` is `!Send`, so thread-based
  vectorization can't run sims concurrently). Near-linear speedup up to
  your physical core count. Each worker gets a distinct terrain.
- `--frame-skip K` holds each action for K physics ticks, so the policy
  makes 1/K as many decisions per second of sim (fewer network/obs
  passes, faster credit assignment). Episodes stay the same sim-duration.

```bash
python examples/train.py --stage locomotion --timesteps 1000000 \
    --n-envs 8 --frame-skip 4 --save runs/stage0
```

**`--frame-skip` must match across training, eval, and the autopilot.**
Pass the same value to `examples/evaluate.py --frame-skip K` and to
`examples/export_policy.py --frame-skip K` (it's recorded in the
`.norm.json` so the in-game autopilot replays at the right cadence).
And make sure the extension is a **release** build (`maturin develop
--release`) — a debug build is many times slower.

Picking values:

- **`--n-envs` ≈ your physical core count** (e.g. 8). This is the big
  win — roughly Nx throughput.
- **`--frame-skip 2–4`** is a reasonable start for driving. Higher =
  faster but coarser control; if the rover gets twitchy or can't react
  in time, dial it back.
- **Keep PPO on CPU.** The policy is a tiny MLP — a GPU adds transfer
  overhead for negligible compute and is usually slower. The two flags
  above are where the speedup is.

### Watching a trained policy in the game

Export the policy to ONNX and run the native game with it (full HUD,
overlays, camera — no Python in the loop):

```bash
python examples/export_policy.py \
    --stage locomotion \
    --model runs/stage0/model.zip --vecnorm runs/stage0/vecnorm.pkl
cargo run -p hylaeanrover_game --release -- --policy runs/stage0/model.onnx
```

In-game: **P** toggles autopilot, **O** hot-reloads the policy file (so
you can re-export a newer checkpoint mid-training and watch it improve).

### Saving & sharing the best model per stage

Training writes to `runs/` (git-ignored scratch). To keep a curated
"best so far" for a stage — committed to the repo so you can resume
training from it, run the autopilot, or share it — promote a run into
the tracked `models/<stage>/` bundle:

```bash
python examples/promote_model.py --stage locomotion --run runs/stage0
git add ../models/locomotion && git commit -m "Promote locomotion model"
```

This copies the run's eval-best `model.zip` + `vecnorm.pkl` into
`models/locomotion/` and regenerates `model.onnx` + `model.norm.json`
from them (add `--frame-skip K` if you trained with frame-skip, or
`--source final` to promote the last model instead of the eval-best
one). The bundles are ~240 KB/stage — small enough for plain git.

Then everything just points at the bundle:

```bash
# resume / warm-start the next stage from the stage's best
python examples/train.py --stage cube_intercept --timesteps 0 \
    --load models/locomotion/model.zip --vecnorm models/locomotion/vecnorm.pkl \
    --reset-reward-stats \
    --scenario cube_intercept \
    --teacher-pretrain-samples 20000 \
    --teacher-pretrain-only \
    --save runs/stage1_cube_intercept_bc
# watch the best
cargo run -p hylaeanrover_game --release -- \
    --policy models/locomotion/model.onnx --mission-supervisor
```

See [`../models/README.md`](../models/README.md) for the bundle layout.
For a command-first walkthrough from no models through a promoted full
policy, use [`TRAINING_GUIDE.md`](TRAINING_GUIDE.md).

## Action space

`Discrete(10)`:

| index | throttle | steering | beacon |
|------:|---------:|---------:|:------:|
| 0     | -1       | -1       |        |
| 1     | -1       |  0       |        |
| 2     | -1       | +1       |        |
| 3     |  0       | -1       |        |
| 4     |  0       |  0       |        |
| 5     |  0       | +1       |        |
| 6     | +1       | -1       |        |
| 7     | +1       |  0       |        |
| 8     | +1       | +1       |        |
| 9     |  0       |  0       | drop   |

Beacon is edge-triggered: action 9 fires one beacon if the budget
isn't exhausted. The 6th beacon (or beyond) is a no-op and immediately
ends the episode via `GameOverReason::BeaconsDeployed`. When the env is
built with `beacons_enabled=False`, action 9 is an inert no-op and that
game-over never fires (see Staged training, above).

## Observation space

`Box(low=-inf, high=+inf, shape=(41,), dtype=float32)`. The vector is
the in-game JSON telemetry, flattened. Slot ranges (matches
`crates/hylaeanrover_core/src/observation.rs`):

| slot | length | content |
|-----:|-------:|---------|
| 0..7 | 7 | speed, heading, pitch, roll, yaw rate, accel fwd, accel lat |
| 7..15 | 8 | lidar fan, meters (200 = no hit) |
| 15..33 | 18 | 6 visible cubes × (bearing, range, valid_flag) |
| 33..34 | 1 | power remaining, fraction of capacity 0..1 |
| 34..40 | 6 | mineral concentrations under the rover (Si, Al, Fe, Ti, H2O, He-3) |
| 40..41 | 1 | beacons remaining |

Power is exposed only as a fraction of capacity (no raw Wh slot) so the
observation is invariant to the configured battery size — a policy
trained on the RL env's small battery (`power_capacity`, see
`hylaeanrover.wrappers.DEFAULT_POWER_CAPACITY_WH`) reads the same
signal when driving the game's 1 kWh battery via the in-game autopilot.

The cumulative reward breakdown and the game-over flag are deliberately
**not** in the observation: the reward components grow unbounded within
an episode and are non-Markovian (bad policy inputs), and the game-over
flag is always 0 on the steps the agent acts on. The full per-component
reward is still available in the `info` dict (see Reward, below) for
shaping and logging. Keeping the observation shape fixed at 41 lets a
policy trained on one curriculum stage transfer cleanly to the next.

## Mission supervisor

Stage 2 and later use a shared Rust mission supervisor around PPO. The policy
owns mineral exploration while power and attitude are healthy. Below 35%
power, the supervisor intercepts only a visible cube that the remaining energy
can reach under the real drain model; otherwise it coasts. A speed-gated tilt
guard brakes dangerous motion without treating a stationary slope as a
rollover. The same state machine is exposed through `MissionSupervisorWrapper`
for training/evaluation and through `--mission-supervisor` in the game.

Enable it explicitly in Python with `--mission-supervisor`. Its decisions and
override rate are reported through the `supervisor_*` `info` fields and the
evaluation summary. Target-loss grace defaults to zero, so a cube leaving the
actionable sensor does not cause blind forward throttle.

In the `full` stage, the same supervisor protects beacon use. It requires
100 m of exploration before the first placement, 75 m between placements, and
a scarcity-weighted surface-mineral score of at least 150. The supervisor can
therefore keep the accepted mineral-exploration policy frozen while adding
strategic beacon placement. Early or crowded policy requests are converted to
coast and receive a small training penalty; in-game deployment uses the same
guard and automatic placement rule.

## Reward

`step_reward = total_reward_now − total_reward_last_step` — purely
incremental, so SB3 sees positive reward when the rover makes useful
progress and 0 (or negative on game-over events) otherwise.

See `crates/hylaeanrover_core/src/reward.rs` for the breakdown.

## Termination

`terminated` is true the step the game reaches any `GameOver` state:

- **out_of_power**: power = 0 AND at-rest 5 s
- **flipped**: |pitch|/|roll| > 100° AND at-rest 5 s
- **beacons_deployed**: 5th beacon dropped (immediate)

The reason is in `info['game_over']` as a string.

`truncated` fires when `step_count >= max_steps` without termination.

## Limitations

- **No live render this branch.** `render_mode='human'` would need
  winit to own the OS main thread, which conflicts with Python doing
  the same. Either an offscreen-render-to-RGB-array path (next branch)
  or an out-of-process viewer reading shared state.
- **Single-thread only.** The Rust env is `unsendable` (Bevy `App` is
  `!Send`), so SB3's `SubprocVecEnv` works (separate processes) but
  `DummyVecEnv` with thread workers does not.
- **Action seed is not exposed.** PPO uses its own RNG for the policy;
  env's `seed` only controls terrain + minerals.
