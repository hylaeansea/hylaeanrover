# RL Training Plan — Hylaean Rover

_Last updated: 2026-06-15_

This document records the plan and the decisions behind how we train an RL
agent to drive the rover. It is meant to be updated as we make progress.

## Context

We want to train an RL agent to drive the rover. The simulation and a
headless Gymnasium/SB3 env already exist and the plumbing works end-to-end
(`check_env` passes, `PPO.learn(1024)` runs). The gap is going from
"PPO doesn't crash" to "PPO measurably learns," and doing it **in stages
without wasting training time**.

The central design question: if we start with a simple locomotion task, how
do we get to the full mission (strategic beacon placement) without throwing
away the simple policy?

**Answer — staged curriculum with weight transfer.** Hold the observation
space and action space *fixed* across every stage, and change **only the
reward**. Because the policy network's input and output shapes never change,
each stage's trained weights initialize the next stage. The agent's learned
motor control (drive forward, turn, don't flip, manage power) carries
forward; later stages only need to learn the *new* part of the reward. This
is reward annealing / curriculum learning — nothing is discarded.

## Current state (verified)

- Shared sim: `crates/hylaeanrover_core/` (Bevy 0.18 + Rapier), headless-safe.
- Env: `crates/hylaeanrover_py/src/lib.rs` — `RoverEnv` pyclass, fixed 1/60s
  `step()`, deterministic.
- Wrapper: `python/hylaeanrover/__init__.py` — `Discrete(10)` actions,
  `Box(47,)` observations.
- Reward: `crates/hylaeanrover_core/src/reward.rs` — distance + mineral
  line-integral + beacon bonus (50×).
- `make_info()` already emits **per-component** cumulative reward
  (`reward_distance`, `reward_mineral_integral`, `reward_beacon_bonus`) — so
  stage-specific reward can be recomputed in Python from info deltas with no
  Rust reward changes.

### Blockers for real training (not yet addressed)

1. No observation normalization (obs mix degrees, meters, and unbounded
   cumulative totals up to ~50k) → needs `VecNormalize`.
2. Cumulative reward totals (obs slots 41–45) fed back as inputs — unbounded,
   non-Markovian.
3. Full task is sparse/hard; the dense learnable signal is distance +
   survival.
4. No training harness: no logging, checkpoints, eval, or baseline
   comparison.

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| First milestone | Staged curriculum, starting with locomotion | Dense signal first; transfer weights forward so nothing is wasted |
| Change scope | Python harness + small Rust knobs | Keep Rust changes minimal and one-time; iterate reward shaping in Python |
| Scaling | Single env, CPU, ~1–2M steps/stage | Prove learning before optimizing throughput |
| Observation | Freeze at 42 dims (trim slots 41–46) | Unbounded/non-Markovian inputs removed; fixed shape enables transfer |
| Action space | Freeze `Discrete(10)` across all stages | Fixed output head enables transfer; gate beacons via a knob |

## The staging design (decisions fixed up front, once)

These two things are decided **now and never change again**, so weights
transfer across all stages:

- **Observation (fixed):** trim the 4 cumulative reward components
  (slots 41–45) and the game_over flag (slot 46) out of `observation()`.
  Keep `beacons_remaining` (Markovian, useful). New `OBS_DIM = 42`.
  The Python `observation_space` is built from `RoverEnv.obs_dim()`
  dynamically, so it tracks automatically.
- **Action (fixed): `Discrete(10)` in every stage.** In stages where
  beacons aren't part of the objective, action 9 must become a true no-op
  rather than silently ending the episode after 5 presses (see the
  `beacons_enabled` knob below). Keeping the 10-way action head fixed means
  the policy's output layer transfers cleanly into the beacon stage.

### Stages (only the reward changes)

- **Stage 0 — Locomotion.** Reward = distance delta (+ small flip penalty,
  small alive bonus). Beacons & mineral integral excluded. Agent learns to
  drive far, steer, manage power (power cubes are in the obs), not flip.
  Densest signal; fastest proof of learning.
- **Stage 1 — Drive + minerals.** Reward = distance + scarcity-weighted
  mineral integral. Load Stage 0 weights, continue. Motor skills transfer;
  agent now also steers toward scarce ground.
- **Stage 2 — Full mission.** Reward adds the beacon bonus; `beacons_enabled`
  on so action 9 places beacons and `BeaconsDeployed` can end the run. Load
  Stage 1 weights, continue. Agent already drives and seeks minerals — it
  only has to learn *when* to drop beacons.

Reward staging itself lives in **Python** (a `gym.Wrapper`) computing each
stage's reward from the per-component info deltas. This keeps shaping fast to
iterate and leaves the Rust reward math untouched.

## Changes

### Rust (minimal, one-time, then frozen)

File: `crates/hylaeanrover_py/src/lib.rs`

- In `observation()`: drop the 4 reward-component pushes (slots 41–45) and
  the game_over flag push; keep `beacons_remaining`. Update `OBS_DIM` to 42
  and the `debug_assert`/doc comment.
- Add a `beacons_enabled: bool` kwarg to `RoverEnv::new` (default `true`),
  stored on `EnvInner`, applied to the sim config at construction.

Files: `crates/hylaeanrover_core/src/lib.rs` + `beacons.rs` + `game_state.rs`

- Add `beacons_enabled: bool` to `RoverCoreConfig` (default `true`;
  `headless()` leaves it `true`, env overrides per stage).
- Gate `place_beacon_on_input` (`beacons.rs:82`) and the
  `beacons_remaining == 0` termination (`game_state.rs:114`) on the flag.
  When disabled, action 9 = pure no-op (identical to action 4) and never
  ends the episode — so the 10-way action space stays valid in Stages 0/1.
- Thread the flag from `RoverCoreConfig` to the systems via a small
  `Resource` (e.g. `BeaconsEnabled(bool)`) so the existing plugin wiring
  stays simple.

File: `python/README.md` — update the obs table (47 → 42).

### Python (the actual training work)

Dir: `python/` (add extras as needed: `tensorboard`).

- `python/hylaeanrover/wrappers.py` — `StagedRewardWrapper(gym.Wrapper)`:
  takes `stage` ∈ {`locomotion`, `minerals`, `full`}, recomputes `reward`
  each step from info-field deltas (`reward_distance`,
  `reward_mineral_integral`, `reward_beacon_bonus`), adds a small flip
  penalty when `info["game_over"] == "flipped"`. For `locomotion`/`minerals`
  it constructs `RoverEnv(beacons_enabled=False)`.
- `python/examples/train.py` — single-env harness:
  `RoverEnv` → `StagedRewardWrapper` → `Monitor` → `DummyVecEnv` (one env;
  `SubprocVecEnv` not needed yet, `DummyVecEnv` is fine for one) →
  `VecNormalize(norm_obs=True, norm_reward=True)`. `PPO("MlpPolicy", …,
  tensorboard_log=…)` with `CheckpointCallback` + `EvalCallback`.
  CLI: `--stage`, `--timesteps` (default ~1e6–2e6), `--load <prev .zip>`,
  `--vecnorm <prev .pkl>`, `--save <dir>`.
  **Transfer handling on `--load`:** load PPO weights and **carry the obs
  normalization stats** (load the saved `VecNormalize`, keep its `obs_rms`)
  but **reset reward normalization** (`ret_rms`) at each stage boundary,
  since reward scale jumps between stages.
- `python/examples/evaluate.py` — run **random baseline vs trained** over N
  episodes, report mean episode return, mean length, flip-rate,
  out-of-power-rate, and (Stage 2) beacons used / beacon bonus. This is how
  we decide a stage actually learned before advancing.

## Why no training time is wasted

- Obs/action shapes are frozen → `PPO.load(prev).set_env(new_env)` transfers
  every layer; no re-init of input/output heads.
- Each stage starts from a policy that already solves the previous stage's
  sub-skill, so gradient updates target only the incremental objective.
- VecNormalize obs stats carry over (obs distribution is stable across
  stages); only reward stats reset.

## Go / no-go gates (don't advance until met)

- **Stage 0 → 1:** trained mean episode distance clearly beats the random
  baseline and flip-rate drops (e.g. distance ≳ 2–3× random, flip-rate
  trending down) over an eval batch.
- **Stage 1 → 2:** mineral-integral component per episode beats a
  distance-only Stage 0 policy evaluated under the Stage 1 reward.
- **Stage 2 done:** agent places beacons and beacon-bonus per episode beats
  a random-beacon-timing baseline.

## Verification (end-to-end)

1. `cd python && source .venv/bin/activate && maturin develop --release`
   (rebuilds the cdylib after the Rust obs/knob changes).
2. `python examples/train_ppo.py` — confirm `check_env` still passes with
   `OBS_DIM == 42` and the smoke PPO run completes.
3. `python examples/evaluate.py --random` — record baseline metrics.
4. `python examples/train.py --stage locomotion --timesteps 1000000 --save runs/stage0`
   then `evaluate.py --load runs/stage0` — confirm Stage 0 gate.
5. `train.py --stage minerals --load runs/stage0/model.zip --vecnorm runs/stage0/vecnorm.pkl --save runs/stage1`
   → evaluate → gate. Repeat for `--stage full` from Stage 1.
6. Watch `tensorboard --logdir runs/` for `rollout/ep_rew_mean` and
   `ep_len_mean` rising.

## Files to modify

- `crates/hylaeanrover_py/src/lib.rs` — trim obs, `beacons_enabled` kwarg.
- `crates/hylaeanrover_core/src/lib.rs`, `beacons.rs`, `game_state.rs` —
  `beacons_enabled` config + gating.
- `python/hylaeanrover/__init__.py` — pass `beacons_enabled` through.
- `python/hylaeanrover/wrappers.py` *(new)* — `StagedRewardWrapper`.
- `python/examples/train.py` *(new)*, `python/examples/evaluate.py` *(new)*.
- `python/pyproject.toml` — add `tensorboard` to the `sb3` extra.
- `python/README.md` — obs table + staged-training usage.

## Visualizing the trained agent (in-game autopilot)

Training is headless (fast). To *watch* a policy drive in the full
rendered game — HUD, lidar, mineral overlays, follow camera — export it
and run the game with `--policy`:

```bash
# 1. Export the trained policy to ONNX (+ a .norm.json with the
#    VecNormalize obs stats) for the in-game runtime.
python examples/export_policy.py \
    --model runs/stage0/model.zip --vecnorm runs/stage0/vecnorm.pkl

# 2. Watch it in the real game.
cargo run -p hylaeanrover_game --release -- --policy runs/stage0/model.onnx
```

In-game keys: **P** toggles autopilot on/off (off → keyboard takes
over), **O** hot-reloads the policy file from disk — so you can watch
progress *during* a long training run by periodically re-exporting the
latest checkpoint and pressing O.

How it works: the policy runs natively in the Rust game via
`tract-onnx` (pure Rust, no Python in the loop). Each frame the game
builds the *same* 42-dim observation the env uses
(`hylaeanrover_core::observation::build_observation`, shared by both),
normalizes it with the exported stats, runs the network, and argmaxes
the logits into a `RoverAction`. Files:
`crates/hylaeanrover_game/src/autopilot.rs`,
`python/examples/export_policy.py`. Live rendering *inside* the Python
training loop is still intentionally not done (winit/Python main-thread
conflict, and it would cripple training throughput) — the
export-and-watch path is the practical substitute and covers checkpoints
mid-training via the O key.

## Progress log

- 2026-06-15 — Plan drafted.
- 2026-06-15 — Implemented:
  - Rust: trimmed observation to 42 dims (dropped cumulative reward
    components + game-over flag); added `beacons_enabled` to
    `RoverCoreConfig` + `BeaconsEnabled` resource, gating beacon
    placement (`beacons.rs`) and the `BeaconsDeployed` game-over
    (`game_state.rs`); exposed `beacons_enabled` as a `RoverEnv` kwarg.
    `cargo check --workspace` passes.
  - Python: `StagedRewardWrapper` + `make_staged_env`
    (`hylaeanrover/wrappers.py`); `examples/train.py` (VecNormalize,
    tensorboard, checkpoints, eval, warm-start transfer);
    `examples/evaluate.py` (random vs trained metrics). `tensorboard`
    added to the `sb3` extra; docs updated.
  - Not yet done: actual multi-stage training runs (the go/no-go gates).
- 2026-06-15 — Added in-game autopilot for watching trained policies:
  - Shared observation builder moved to
    `hylaeanrover_core::observation::build_observation` (env + game now
    produce identical inputs); `AutopilotActive` resource added so the
    `drive` system honours a zero-throttle policy action.
  - Game: `crates/hylaeanrover_game/src/autopilot.rs` loads an ONNX
    policy via `tract-onnx`, normalizes obs with the exported stats, and
    drives `RoverAction` each frame. `--policy <path>` CLI; P toggles,
    O hot-reloads.
  - Python: `examples/export_policy.py` exports the SB3 policy to ONNX
    (+ `.norm.json`). `onnx` added to the `sb3` extra.
  - Verified end-to-end: export → tract load → in-game inference, no
    errors. (Tested with a toy 4096-step model; quality not meaningful.)
- 2026-06-15 — Code-review fixes:
  - `RoverEnv.reset` now draws a fresh terrain seed from the gym RNG each
    episode (deterministic when seeded, varied otherwise). Previously SB3
    autoreset (seed=None) reused one terrain, so training overfit a single
    map and `evaluate.py` compared the trained policy (one terrain) against
    the random baseline (many terrains).
  - `train.py` / `evaluate.py` now require the matching `VecNormalize`
    stats whenever `--load` is given (auto-resolving the sibling
    `vecnorm.pkl`), via `wrappers.resolve_vecnorm` — a model loaded
    without them silently saw un-normalized observations.
- 2026-06-15 — Throughput levers (CPU stays the right place for the MLP):
  - `train.py --n-envs N` runs N sims in parallel via `SubprocVecEnv`
    (separate processes; the Bevy `App` is `!Send`). Per-worker seeds.
    `start_method="spawn"`.
  - Frame-skip / action-repeat: `wrappers.ActionRepeat` holds each action
    K physics ticks; `--frame-skip K` on `train.py` / `evaluate.py`.
    `export_policy.py --frame-skip K` records it in `.norm.json`, and the
    in-game autopilot reads it and replays at the same cadence (beacon
    fires once per decision, then coasts). All three must use the same K.
  - Verified: 2-process SubprocVecEnv (spawn) trains; frame-skip=2 halves
    the policy-step episode length as expected; game crate compiles.
- 2026-06-15 — Curated per-stage model bundles (tracked, shareable):
  - `models/<stage>/` holds `model.zip` + `vecnorm.pkl` (resume training)
    and `model.onnx` + `model.norm.json` (autopilot) — ~240 KB/stage,
    committed to git. See `models/README.md`.
  - `examples/promote_model.py` copies a run's best (or final) into the
    bundle and regenerates the ONNX pair. Export logic refactored into
    `hylaeanrover/export.py` (shared with `export_policy.py`).
  - `train.py` now saves the VecNormalize stats next to
    `best/best_model.zip` on each new eval-best (`SaveBestVecNormalize`
    callback), so the best checkpoint is a complete, loadable bundle —
    previously `best/` had no matching vecnorm.
  - Verified: best/ now contains best_model.zip + vecnorm.pkl; promote
    produces all four files; refactored export still works.
