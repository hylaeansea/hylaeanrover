# hylaeanrover — Python / RL bindings

Gymnasium-compatible RL environment wrapping the Hylaean Rover Bevy
simulator. Drives a headless physics + sensor stack from Python so
Stable Baselines 3 (or any other RL library) can `reset()` / `step()`
episodes against it.

## Setup

You need a Rust toolchain (`rustup`) and Python ≥ 3.9.

```bash
# from the project root (one level up from this README)
cd python

# create + activate a venv
python -m venv .venv && source .venv/bin/activate

# install maturin and project deps
pip install 'maturin[patchelf]>=1.4'
pip install -e '.[sb3]'   # gymnasium + numpy + sb3 + torch

# build the Rust cdylib in-place
maturin develop --release
```

`maturin develop` compiles `crates/hylaeanrover_py` and drops the
resulting `hylaeanrover_py.*.so` into the active venv's
`site-packages/`. The Python package `hylaeanrover` (this directory)
imports that extension module.

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
ends the episode via `GameOverReason::BeaconsDeployed`.

## Observation space

`Box(low=-inf, high=+inf, shape=(47,), dtype=float32)`. The vector is
the in-game JSON telemetry, flattened. Slot ranges (matches
`crates/hylaeanrover_py/src/lib.rs::observation`):

| slot | length | content |
|-----:|-------:|---------|
| 0..7 | 7 | speed, heading, pitch, roll, yaw rate, accel fwd, accel lat |
| 7..15 | 8 | lidar fan, meters (200 = no hit) |
| 15..33 | 18 | 6 visible cubes × (bearing, range, valid_flag) |
| 33..35 | 2 | power normalized 0..1, power Wh |
| 35..41 | 6 | mineral concentrations under the rover (Si, Al, Fe, Ti, H2O, He-3) |
| 41..45 | 4 | reward total, distance, mineral_integral, beacon_bonus |
| 45..46 | 1 | beacons remaining |
| 46..47 | 1 | game_over flag (0/1) |

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
