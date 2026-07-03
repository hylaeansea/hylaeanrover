# Hylaean Rover

A small Bevy game and a headless OpenAI-Gymnasium environment built on
the same simulation — a rover roams a lunar landscape, sniffs mineral
deposits, places survey beacons, and tries to run a clean mission
before it tips over or runs out of power.

The repo contains:

- a **playable game** built with [Bevy 0.18](https://bevyengine.org/) +
  [Rapier 0.33](https://rapier.rs/) (`crates/hylaeanrover_game/`)
- a **headless RL environment** for Stable Baselines 3 wrapped via PyO3
  (`crates/hylaeanrover_py/` + `python/hylaeanrover/`)
- the **shared game logic** both use (`crates/hylaeanrover_core/`)

![Rover sitting on the landing pad at game start, with the full HUD visible.](https://hylaeansea.org/assets/images/hylaeanrover_screenshot_begingame_readme.png)

<video src="https://raw.githubusercontent.com/hylaeansea/hylaeansea.github.io/main/assets/images/hylaeanrover_readme.mp4" controls muted loop width="100%"></video>

The HUD shows everything the simulation is tracking. Left column:
**POWER** reserve in kWh, **MINERAL SURVEY** with surface
concentrations of six elements directly under the rover,
**IMU / TELEMETRY** (speed, heading, pitch, roll, yaw rate, lateral
and longitudinal acceleration, an 8-ray forward lidar histogram), and
**VISIBLE CUBES** — bearings and ranges of any visible-from-here
power cubes. Top bar: beacons remaining and the live reward
breakdown. Right column: terrain regeneration controls.

## What you do

Drive around a procedurally generated lunar terrain (with craters)
collecting glowing blue power cubes for energy, surveying mineral
distributions, and placing **5 survey beacons** on what you think are
high-value subsurface deposits. The reward function combines:

- distance driven
- a line-integral over the terrain weighted by element scarcity
  (Si=1, Al=2, Fe=3, Ti=10, H₂O=20, ³He=50)
- **50× the subsurface concentration at each beacon location** —
  finding a water or helium-3 jackpot is worth orders of magnitude
  more than mileage

A run ends when one of three conditions hits:

- you exhaust your beacon budget (mission complete)
- you tip the rover over and it comes to rest (failure)
- you run out of power and come to rest (failure)

![Game-over screen showing "BEACONS DEPLOYED — all five beacons placed — survey complete." with three orange-tipped beacons visible behind the rover.](https://hylaeansea.org/assets/images/hylaeanrover_screenshot_endgame_readme.png)

## The mineral overlay system

Clicking a row in the **MINERAL SURVEY** panel colour-codes the
terrain by that element's surface concentration. Drives the
intuition for where to spend beacons:

![Iron overlay — terrain shaded in browns and oranges.](https://hylaeansea.org/assets/images/hylaeanrover_screenshot_fe_readme.png)
![Titanium overlay — magenta hotspots are ilmenite-rich pockets.](https://hylaeansea.org/assets/images/hylaeanrover_screenshot_ti_readme.png)
![Water overlay — cyan patches mark permanently-shadowed reservoirs.](https://hylaeansea.org/assets/images/hylaeanrover_screenshot_h2o_readme.png)

The trick is that the *surface* readings the HUD shows are only
loosely correlated with the *subsurface* deposits beacons actually
score against. The agent (human or RL) has to infer where the
deposits are from surface trends.

---

## Playing the game

Requires a Rust toolchain (`rustup`).

```bash
cargo run -p hylaeanrover_game --release
```

The first build takes a few minutes (Bevy is big). After that, `cargo
run` is fast.

### Controls

| Key | Action |
|---|---|
| **W / S** | Forward / reverse throttle |
| **A / D** | Steer left / right (opposite-phase 4WS) |
| **B** | Drop a beacon behind the rover (limit 5) |
| **R** | Respawn upright at your current XZ |
| **Shift+R** | Respawn at the world origin (landing pad) |
| **C** | Toggle between follow camera and free orbit camera |
| **F1** | Toggle Rapier collision wireframes |
| *(click)* in orbit mode | Drag to orbit, right-drag to pan, scroll to zoom |

When launched with a trained policy (`--policy`, see *Watching a trained
agent* below) two more keys are active: **P** toggles autopilot on/off,
**O** hot-reloads the policy file from disk.

The bottom of the window shows a continuously-updated JSON dump of
the **same observation an RL agent sees** — useful for debugging or
just understanding what's in the agent's input space.

---

## Training an RL agent

The headless environment is exposed as a `gymnasium.Env` (via PyO3)
that drives the same Bevy simulation without rendering. Stable
Baselines 3 can train on it directly.

### One-time setup

Requires Python ≥ 3.9 and a Rust toolchain.

```bash
cd python
python3 -m venv .venv
source .venv/bin/activate

pip install 'maturin[patchelf]>=1.4'
pip install -e '.[sb3]'           # gymnasium, numpy, stable-baselines3, torch

maturin develop --release          # builds and installs the Rust cdylib
```

`maturin develop` rebuilds the `hylaeanrover._native` extension into
the active venv. **Rerun it after any Rust change**; pure-Python
edits in `python/hylaeanrover/` are picked up automatically.

### Smoke test

```bash
source .venv/bin/activate
python examples/train_ppo.py
```

This runs:

1. `gymnasium.utils.env_checker.check_env(env)` — verifies the env
   satisfies the gymnasium contract (correct dtypes, finite reward,
   resettable, etc.)
2. A 200-step random rollout — sanity-check that `step()` returns
   sensible values and the env terminates within `max_steps`
3. `PPO.learn(total_timesteps=1024)` — confirms SB3 can actually
   train against the env end-to-end without errors

Expected output ends with `✓ PPO.learn() completed without errors`
in about 30 seconds on a modern laptop CPU.

### Using the env in your own code

```python
from hylaeanrover import RoverEnv

env = RoverEnv(seed=42, max_steps=2000)
obs, info = env.reset(seed=42)
for _ in range(1000):
    action = env.action_space.sample()  # Discrete(10): see python/README.md
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
env.close()
```

Action layout (10 discrete) and the 41-float observation schema — plus
the staged training curriculum (locomotion → power_cubes → minerals → full) — are
documented in detail in [`python/README.md`](python/README.md) and
[`docs/rl_training_plan.md`](docs/rl_training_plan.md).

### Watching a trained agent

Training is headless. To watch a trained policy drive in the **full
rendered game** (HUD, lidar, overlays, camera), export it to ONNX and
launch the game with `--policy`:

```bash
# from python/, with the venv active
python examples/export_policy.py \
    --stage locomotion \
    --model runs/stage0/model.zip --vecnorm runs/stage0/vecnorm.pkl

cargo run -p hylaeanrover_game --release -- --policy runs/stage0/model.onnx
```

The policy runs natively in the game via `tract-onnx` (no Python in the
loop). Press **P** to toggle autopilot vs. manual driving, and **O** to
hot-reload the policy — so you can re-export a newer checkpoint mid-run
and watch progress. See [`docs/rl_training_plan.md`](docs/rl_training_plan.md)
for details.

---

## Repo layout

```
hylaeanrover/
├── Cargo.toml                            workspace + shared dep versions
├── .cargo/config.toml                    macOS linker flags for the cdylib
├── crates/
│   ├── hylaeanrover_core/                Bevy plugins, physics, reward, telemetry
│   │   └── src/
│   │       ├── lib.rs                    RoverCorePlugin (Gltf|Primitives, with_ui|headless)
│   │       ├── rover.rs                  chassis/wheels/drive/respawn + RoverAction
│   │       ├── beacons.rs                5-beacon budget, scarcity-weighted bonus
│   │       ├── game_state.rs             dual game-over triggers + at-rest timer
│   │       ├── imu.rs                    IMU panel + 8-ray lidar + visible-cubes sensor
│   │       ├── minerals.rs               6-element procedural concentration maps
│   │       ├── power_cubes.rs            cubes, energy reserve, game-over modal
│   │       ├── reward.rs                 RewardState + scoring + top-bar UI
│   │       ├── observation.rs            shared 41-float obs builder (env + autopilot)
│   │       ├── telemetry.rs              RL observation resource + JSON readout
│   │       ├── terrain.rs                heightfield + crater stamping
│   │       ├── terrain_controls.rs       right-side terrain panel + regen
│   │       └── ui.rs                     LeftSidebar container + UiFont resource
│   ├── hylaeanrover_game/                playable binary
│   │   ├── assets/                       glTF rover + beacon + fonts
│   │   └── src/
│   │       ├── main.rs                   DefaultPlugins + camera + RoverCorePlugin
│   │       └── autopilot.rs              ONNX-policy autopilot (tract-onnx)
│   └── hylaeanrover_py/                  cdylib for Python / SB3
│       └── src/lib.rs                    RoverEnv pyclass, fixed-timestep stepping
├── python/
│   ├── pyproject.toml                    maturin packaging
│   ├── README.md                         action/obs schema + training/eval docs
│   ├── hylaeanrover/
│   │   ├── __init__.py                   gymnasium.Env wrapper
│   │   ├── wrappers.py                   staged reward, action-repeat, env factory
│   │   └── export.py                     SB3 policy → ONNX bundle
│   └── examples/                         train / evaluate / export / promote / smoke
├── models/                               curated best-per-stage bundles (tracked)
└── docs/rl_training_plan.md              RL curriculum design + progress log
```

The same `RoverCorePlugin` is wired into both the game's
`DefaultPlugins` App and the RL env's `MinimalPlugins` App. UI-spawning
systems gracefully no-op when the `UiFont` resource is absent, and the
rover is spawned either from the glTF model (game) or as bare named
entities the physics setup picks up the same way (RL).

---

## Project status

- **Game**: playable. Drive around, drop beacons, try to maximise
  reward. The procedural terrain reseeds with the *Randomize seed*
  button (terrain + mineral distribution share a single seed).
- **RL env**: end-to-end working with SB3. The smoke test passes;
  reward shaping and longer-horizon training have not been tuned.
- **Live rendering during training (`render_mode='human'`)**: not
  implemented. Bevy's window backend (winit) needs to own the OS main
  thread, which conflicts with Python doing the same. A future branch
  may add `render_mode='rgb_array'` via an offscreen render pass.

## License

Source code: dual-licensed under MIT or Apache-2.0 (your choice).
Bundled fonts in `crates/hylaeanrover_game/assets/fonts/` are
DejaVu Sans (Bitstream Vera derivative, see `LICENSE.txt`).
