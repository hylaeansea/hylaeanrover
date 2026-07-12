# RL Training Plan — Hylaean Rover

_Last updated: 2026-07-01_

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
  `Box(41,)` observations (see the 2026-07-01 log entry below).
- Reward: `crates/hylaeanrover_core/src/reward.rs` — distance + cube pickup
  bonus (flat) + mineral line-integral + beacon bonus (50×).
- `make_info()` already emits **per-component** cumulative reward
  (`reward_distance`, `reward_cube_bonus`, `reward_mineral_integral`,
  `reward_beacon_bonus`) — so stage-specific reward can be recomputed in
  Python from info deltas with no Rust reward changes.

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
| Observation | Freeze at 42 dims (trim slots 41–46); later cut to 41 (raw-Wh power slot dropped, see 2026-07-01 log) | Unbounded/non-Markovian inputs removed; fixed shape enables transfer |
| Action space | Freeze `Discrete(10)` across all stages | Fixed output head enables transfer; gate beacons via a knob |

## The staging design (decisions fixed up front, once)

These two things are decided **now and never change again**, so weights
transfer across all stages:

- **Observation (fixed):** trim the cumulative reward components and the
  game_over flag out of `observation()`. Keep `beacons_remaining`
  (Markovian, useful). `OBS_DIM = 41` (see the 2026-07-01 log entry: the
  raw-Wh power slot was later dropped too, keeping only the 0..1
  fraction). The Python `observation_space` is built from
  `RoverEnv.obs_dim()` dynamically, so it tracks automatically.
- **Action (fixed): `Discrete(10)` in every stage.** In stages where
  beacons aren't part of the objective, action 9 must become a true no-op
  rather than silently ending the episode after 5 presses (see the
  `beacons_enabled` knob below). Keeping the 10-way action head fixed means
  the policy's output layer transfers cleanly into the beacon stage.

### Stages (only the reward changes)

- **Stage 0 — Locomotion.** Reward = distance delta (+ small flip penalty,
  small alive bonus). Cube/mineral/beacon bonuses excluded. Agent learns to
  drive far, steer, manage power (power cubes are in the obs but not yet
  rewarded), not flip. Densest signal; fastest proof of learning.
- **Stage 1A — Cube intercept.** Reward = cube-intercept shaping +
  pickup success, with no random cube spawns and one settled forced cube.
  Load Stage 0 weights, optionally pretrain the policy head from the
  built-in close-range teacher, then PPO fine-tune. This isolates the
  missing visible-cube sensor-to-action behavior before broad cube
  training resumes.
- **Stage 1B — Power idle.** Teacher-pretrained low-power no-target
  behavior, with no random cubes and low starting power. Load Stage 1A
  weights and teach the policy not to spend a nearly empty battery when
  the actionable cube sensor is empty.
- **Stage 1C — Power cubes.** Reward = pickup bonus plus mixed
  dense/bridge/sparse-visible scenarios. Load Stage 1B weights, continue.
  Dense spawns remain a curriculum tool, but no-shaping
  `sparse_visible_low_power` and `sparse_game` evals are required before
  promotion. Forced diagnostic cubes are settled on/near terrain and the
  RL-visible cube sensor hides non-actionable airborne cubes so `OBS_DIM`
  stays fixed and the visible cube slots remain actionable.
- **Stage 2 — Drive + minerals.** Reward = distance +
  scarcity-weighted mineral integral. Load Stage 1 weights, continue.
  Motor + seek skills transfer; a short mineral-explore teacher bootstrap
  first restores the missing "cover ground while powered" prior, then PPO
  learns the mineral reward. Power cubes are survival support, not a paid
  objective: keep power-efficiency and low-power cube shaping active, but
  train primarily on no-cube / sparse-cube mineral exploration scenarios
  so the exported policy searches terrain instead of waiting for cube
  drops. Stage 2 trains and evaluates on the medium horizon, adds dense
  excessive-tilt shaping, and selects checkpoints by a robust reward
  statistic with an explicit terminal-failure cost. A short-horizon mean
  reward hid both late failures and rare mineral-return outliers.
- **Stage 3 — Full mission.** Reward adds the beacon bonus; `beacons_enabled`
  on so action 9 places beacons and `BeaconsDeployed` can end the run. Load
  Stage 2 weights, continue. Agent already drives, seeks cubes, and finds
  minerals — it only has to learn *when* to drop beacons.

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
  takes `stage` ∈ {`locomotion`, `cube_intercept`, `power_idle`,
  `power_cubes`, `minerals`, `full`}, recomputes `reward`
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
  normalization stats** (load the saved `VecNormalize`, keep its `obs_rms`).
  Reset reward normalization (`ret_rms`) at each stage boundary because reward
  scale jumps between stages; preserve it for same-stage continuation. The
  trainer auto-detects obvious path names, and `--reset-reward-stats` /
  `--preserve-reward-stats` make the choice explicit.
- `python/examples/evaluate.py` — run **random baseline vs trained** over N
  episodes, report mean episode return, mean length, flip-rate,
  out-of-power-rate, end power, low-power distance, visible-cube approach
  rate, pickups, and beacons used / beacon bonus. This is how we decide a
  stage actually learned before advancing.

## Why no training time is wasted

- Obs/action shapes are frozen → `PPO.load(prev).set_env(new_env)` transfers
  every layer; no re-init of input/output heads.
- Each stage starts from a policy that already solves the previous stage's
  sub-skill, so gradient updates target only the incremental objective.
- VecNormalize obs stats carry over (obs distribution is stable across
  stages). Reward stats reset across stage boundaries and are preserved for
  same-stage continuation.

## Go / no-go gates (don't advance until met)

- **Stage 0 → 1:** trained mean episode distance clearly beats the random
  baseline and flip-rate drops (e.g. distance ≳ 2–3× random, flip-rate
  trending down) over an eval batch.
- **Stage 1A → 1B:** do not start `power_idle` training until
  `cube_intercept` reaches the pickup and low-power gates in
  `docs/rl_stage0_stage1_hardening_plan.md`.
- **Stage 1B → 1C:** do not start broad `power_cubes` training until
  `power_idle` / `no_cube_control` evals show low out-of-power behavior
  when no actionable cube is visible.
- **Stage 1C → 2:** do not advance until the hardening gates in
  `docs/rl_stage0_stage1_hardening_plan.md` pass. The `power_cubes`
  policy must beat the promoted locomotion policy on pickups, end power,
  low-power behavior, and out-of-power rate across dense, transition,
  sparse-visible, sparse-game, and no-cube-control scenarios.
- **Stage 2 → 3:** mineral-integral component per episode beats a
  Stage 1 policy evaluated under the Stage 2 reward, without the
  cube-pickup rate regressing badly from Stage 1.
- **Stage 3 done:** agent places beacons and beacon-bonus per episode beats
  a random-beacon-timing baseline.

## Verification (end-to-end)

1. `cd python && source .venv/bin/activate && maturin develop --release`
   (rebuilds the cdylib after the Rust obs/knob changes).
2. `python examples/train_ppo.py` — confirm `check_env` still passes with
   `OBS_DIM == 41` and the smoke PPO run completes.
3. `python examples/evaluate.py --random` — record baseline metrics.
4. `python examples/train.py --stage locomotion --timesteps 1000000 --save runs/stage0`
   then `evaluate.py --load runs/stage0` — confirm Stage 0 gate.
5. `train.py --stage cube_intercept --load runs/stage0/model.zip --vecnorm runs/stage0/vecnorm.pkl --reset-reward-stats --teacher-pretrain-samples 20000 --save runs/stage1_cube_intercept`
   → evaluate the forced-visible intercept gates with shaping off.
6. `train.py --stage power_idle --load runs/stage1_cube_intercept/best/best_model.zip --vecnorm runs/stage1_cube_intercept/best/vecnorm.pkl --reset-reward-stats --scenario power_idle --teacher-pretrain-samples 20000 --teacher-scenarios power_idle,cube_intercept_low_power --teacher-pretrain-only --save runs/stage1_power_idle_bc`
7. `train.py --stage power_cubes --load runs/stage1_power_idle_bc/best/best_model.zip --vecnorm runs/stage1_power_idle_bc/best/vecnorm.pkl --reset-reward-stats --save runs/stage1_power_cubes`
   → evaluate against `docs/rl_stage0_stage1_hardening_plan.md` gates.
8. `train.py --stage minerals --load runs/stage1_power_cubes/best/best_model.zip --vecnorm runs/stage1_power_cubes/best/vecnorm.pkl --reset-reward-stats --scenario minerals_explore --low-power-threshold 0.35 --teacher-pretrain-samples 75000 --teacher-scenarios minerals_explore,minerals_sparse,minerals_transition,transition,no_cube_control --teacher-pretrain-only --save runs/stage2_minerals_bc`
9. `train.py --stage minerals --load runs/stage2_minerals_bc/model.zip --vecnorm runs/stage2_minerals_bc/vecnorm.pkl --reset-reward-stats --horizon medium --scenario minerals_sparse --low-power-threshold 0.35 --locomotion-out-of-power-penalty 1500 --flip-penalty 1500 --tilt-penalty 5 --tilt-threshold-deg 45 --eval-selection-stat median --eval-failure-penalty 10000 --selection-extra-scenarios transition --ignored-cube-penalty 500 --mission-supervisor --save runs/stage2_minerals`
   → evaluate mineral exploration, sparse-cube survival, and Stage 1
   regression gates before promoting.
10. After those pass, evaluate the accepted minerals policy with `--stage full --mission-supervisor`.
   The shared controller requires 100 m before the first
   beacon, 75 m spacing, and a scarcity-weighted surface score of 150. Keep the
   exploration policy frozen when this hierarchical candidate passes; direct
   full-stage PPO is optional and must beat the transition and sparse mineral,
   pickup, beacon, and terminal-failure gates before replacing it.
11. Promote mineral/full candidates with `--runtime-power-capacity 1000`,
   `--runtime-supervisor-low-power-enter-fraction 0.15`, and
   `--runtime-supervisor-low-power-exit-fraction 0.20`.
   Sidecars retain `training_power_capacity_wh=100` for provenance while the
   game applies `runtime_power_capacity_wh=1000`. Training keeps the 35%/40%
   curriculum gates, but the promoted runtime supervisor enters recovery at
   15% (150 Wh), exits at 20% (200 Wh), and rejects deterministic low-power
   intercepts beyond the trained 120 m envelope. A cube recharge below the 20%
   exit threshold releases recovery with a temporary 75 Wh exploration budget,
   after which the supervisor re-arms preservation at the lower of the
   post-charge budget floor and the normal 15% runtime threshold.
   While the supervisor is enabled, PPO receives zero-filled cube slots and a
   constant healthy-power value, remaining a pure mineral-exploration policy.
   The supervisor alone receives raw cube/power state and owns survival.
12. Watch `tensorboard --logdir runs/` for `rollout/ep_rew_mean` and
   `ep_len_mean` rising.
13. Train map-aware mineral exploration as a separate `coverage_v1` candidate.
   The core keeps a fixed 990 x 990 byte grid of 5 m cells per game and emits
   18 egocentric frontier features through PPO's supervisor-hidden cube slots,
   preserving `OBS_DIM=41`. Warm-start the accepted minerals policy, zero the
   repurposed input columns once, reset only their normalization plus reward
   normalization, and keep the promoted baseline until matched coverage,
   mineral, pickup, and terminal-failure gates pass.

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
    --stage locomotion \
    --model runs/stage0/model.zip --vecnorm runs/stage0/vecnorm.pkl

# 2. Watch it in the real game.
cargo run -p hylaeanrover_game --release -- \
  --policy runs/stage0/model.onnx --mission-supervisor
```

In-game keys: **P** toggles autopilot on/off (off → keyboard takes
over), **O** hot-reloads the policy file from disk — so you can watch
progress *during* a long training run by periodically re-exporting the
latest checkpoint and pressing O.

How it works: the policy runs natively in the Rust game via
`tract-onnx` (pure Rust, no Python in the loop). Each frame the game
builds the *same* 41-dim observation the env uses
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
- 2026-07-01 — Root-caused the ~200k-step "policy collapse": the battery
  was never reset between episodes. `RelaunchEvent` reset reward, game
  state, beacons, cubes, and the rover — but power was only refilled by
  the UI Relaunch button, which the RL env's `reset()` bypasses. Each
  training process therefore had one 1 kWh battery for its whole life
  (~2000 m of driving at 0.5 Wh/m); once a policy's cumulative distance
  crossed that, every episode started dead (throttle no-op, ~7.7 reward
  from spawn-settle, out_of_power after the 5 s at-rest gate) and no PPO
  knob could recover it. Better policies died sooner — the "climbs then
  collapses" pattern was environmental, not optimizer instability.
  - Rust: `reset_power_on_relaunch` in `power_cubes.rs` refills the
    battery on `RelaunchEvent` (button handler no longer does it
    manually); env `reset()` also clears the previous episode's
    `RoverAction` so stale throttle can't drain the fresh battery during
    settle frames.
  - Power management is now part of the curriculum: battery capacity is
    configurable (`RoverCoreConfig.power_capacity_wh`, `RoverEnv
    power_capacity=`, `train.py`/`evaluate.py --power-capacity`).
    Training defaults to `wrappers.DEFAULT_POWER_CAPACITY_WH = 100` Wh —
    measured: a flat-out 2000-tick episode covers ~320 m and costs
    ~158 Wh, so 100 Wh dies ~63% in and caps naive driving at ~230 m;
    pacing + regen braking extend it. Same value should be used across
    all stages (like frame-skip). The game keeps 1 kWh.
  - Observation is now 41 dims: dropped the raw-Wh power slot, keeping
    the 0..1 fraction, so the obs is capacity-invariant and a policy
    trained on the small battery transfers to the game's 1 kWh
    autopilot. Old checkpoints/vecnorm stats are incompatible (they were
    trained under the battery bug anyway).
  - `train.py` logs episode outcomes per rollout to TensorBoard
    (`episodes/frac_time_limit|out_of_power|flipped|...` and
    `episodes/end_power_frac` via `EpisodeOutcomeCallback` on the new
    `info["power_frac"]`), so env-driven failures no longer masquerade
    as RL instability.
  - Verified: battery refills across resets (obs slot 33 back to 1.0),
    out-of-power binds mid-episode at 100 Wh, smoke PPO run logs the new
    metrics end-to-end.
- 2026-07-02 — Added the `power_cubes` stage and root-caused two failures
  found while actually training it, plus a train/deploy config mismatch
  found while watching the result in the game:
  - **Stage added**: `locomotion → power_cubes → minerals → full`. New
    `RewardState.cube_bonus` component (`reward.rs`), credited from
    `power_cubes.rs::advance_charging_cubes`. `STAGE_WEIGHTS` in
    `wrappers.py` carries the `cube` weight forward from `power_cubes`
    onward, same as `distance`.
  - **Root-caused near-zero learning signal**: cube spawns were anchored
    to world origin (`SPAWN_EXTENT` around `(0,0)`), but the
    `locomotion`-warm-started policy already drives ~200 m away fairly
    directly — an origin-anchored box stopped overlapping its path almost
    immediately, so no density tuning could fix it. `spawn_power_cubes`
    now samples an annulus `[10m, extent)` around the rover's *current*
    position instead — rover-anchored so cubes stay reachable, with a
    minimum spawn distance so every cube requires real navigation (a
    plain rover-centred box lets cubes land within grabbing range of a
    near-stationary policy). Density/extent are stage-configurable
    (`RoverCoreConfig.cube_spawn_lambda` / `cube_spawn_extent`, mirroring
    `power_capacity_wh`), **training-only** — the game keeps its sparse
    0.05/s "periodic lifeline" feel; the trained seek skill is a local
    reactive behavior and transfers across densities ("train dense,
    deploy sparse"). `power_cubes` uses `λ=1.5/s, extent=30m` —
    calibrated by measuring pickups/episode against the actual warm-start
    policy (not a random baseline, which is a poor proxy): ~0.5
    pickups/episode, a positive example about every other episode. An
    earlier `λ=3.5/s` pick maximized signal but visually "rained" cubes
    (~117 spawns/episode, ~99% never collected); `MAX_ALIVE_CUBES = 150`
    in `power_cubes.rs` now also hard-caps uncollected pileup in
    unbounded (non-episode) game sessions regardless of configured rate.
  - **Root-caused a ~200-300k-step PPO collapse** (`approx_kl` spike,
    `explained_variance` crash, reward climbing then permanently
    dropping): `CUBE_PICKUP_BONUS=100` was credited as a single-tick spike
    on charge completion — ~100-1000x the surrounding per-tick distance
    signal, landing in only about half of episodes. That variance wrecked
    GAE advantage estimates and blew up a policy update. Fix:
    `RewardState::credit_cube_charge` now pays the same total out
    smoothly across the ~0.5s charge (~3.3/tick instead of +100 once);
    per-episode totals are unchanged (the deltas telescope to exactly
    `CUBE_PICKUP_BONUS`).
  - **Root-caused a train/deploy config mismatch**: a `power_cubes`
    checkpoint watched via the in-game autopilot drove to ~200m and
    rapid-fired all 5 beacons, ending the run. Training sets
    `beacons_enabled=false`, so action 9 is an inert no-op there
    (identical to coasting, never differentiated from it) — but the game
    always runs with beacons live, and at ~200m the rover is past
    everything the 100 Wh training battery ever let it reach (out of
    training distribution). `export_policy.py` now records the stage's
    `beacons_enabled` / `power_capacity_wh` in `.norm.json`; `main.rs`
    reads them back via `autopilot::resolve_core_config` *before*
    constructing `RoverCorePlugin`, so watching a non-`full`-stage
    checkpoint replays it under the same battery/beacon conditions it
    trained in. The training cube-spawn config is deliberately *not*
    carried over: it calibrates a spawn rate for a bounded ~33s episode
    that gets wiped every reset, and applying it to the game's unbounded
    session piled up cubes without limit (first version of this fix did
    exactly that). Old-format sidecars (missing these keys) fall back to
    the game's defaults unchanged — see `autopilot.rs`'s unit tests for
    the fallback matrix. The committed `models/locomotion/model.norm.json`
    was re-exported (from the existing `model.zip`/`vecnorm.pkl`, no
    retraining) to pick this up.
  - Verified: `cargo test --workspace` (autopilot unit tests) and
    `cargo clippy --workspace --tests` pass; re-exported both
    `locomotion` and a `power_cubes` checkpoint and confirmed the correct
    fields land in `.norm.json` (and the cube-spawn fields don't);
    launched the game with a `power_cubes` `.norm.json` and confirmed the
    startup log shows `beacons_enabled=false, power_capacity_wh=100`.
- 2026-07-02 — Stage 0/1 hardening gate added before minerals:
  - New plan: `docs/rl_stage0_stage1_hardening_plan.md`.
  - Rust/Python now expose terrain height scale, reset-time power start
    fraction, seeded cube spawning, named cube spawn presets, named
    horizons, low-power Stage 1 shaping, and acceptance metrics for
    low-power visible-cube behavior.
  - `export_policy.py --stage` is required so non-full exports do not
    accidentally write full-stage runtime config.
- 2026-07-05 — Stage 1 framework reset after forced-visible diagnostics:
  - Forced/training cubes now spawn settled on/near terrain, and the
    RL-visible cube sensor filters to actionable cubes while preserving
    `OBS_DIM == 41`. Debug `info` fields expose nearest raw cube height
    and actionability without feeding them to the policy.
  - Stage 1 is split into `cube_intercept`, `power_idle`, and `power_cubes`.
    `cube_intercept` uses one forced cube, no random spawns, intercept
    shaping, loss-of-sight/timeout penalties, and an optional close-range
    teacher policy-head pretrain before PPO. `power_idle` then isolates
    low-power no-target conservation with a teacher dataset that mixes
    no-cube coast labels and visible-cube intercept labels. `power_cubes`
    resumes only after the intercept and idle gates pass. During broad
    power-cube training, `power_cubes` carries cube reward but zero
    distance reward so pickup and power behavior are hardened before
    travel reward returns in later stages.
  - Promotion is blocked until no-shaping `sparse_visible_low_power` and
    `sparse_game` checks show pickup behavior; dense pickup success alone
    is not promotable.
- 2026-07-12 — Coverage candidate trained, gate re-anchored, promoted:
  - 1M-step `minerals_coverage` run warm-started from the promoted
    minerals bundle (columns 15..32 reset, reward normalization reset).
    Checkpoint selection picked the 500k `best`; the 1M final exceeded
    the 5% flip gate (10% medium forced-cube, 20% long) and was rejected.
  - Re-anchored the novelty gate from `novel_distance_per_100m` (+30%)
    to unique cells per episode (+30% at matched power budget). The
    per-100m ratio caps near 100 and the baseline already scores ~94 by
    driving short non-retracing paths, so the old gate was unreachable
    for any policy; the coverage benefit is total novel ground covered.
  - Matched-seed results for `best` vs promoted baseline: unique cells
    +32-107% (medium) and +88% (long 1 kWh sparse-game, 193 vs 103
    cells), mineral score 155% of baseline (medium), revisit rate <=17%,
    forced-visible pickups 8.1/episode, 0% out-of-power everywhere,
    flips 5% medium ceiling / 0% long. Long no-cube behavior is
    supervisor-owned and byte-identical between models.
  - Promoted `runs/stage2_minerals_coverage/best` with
    `--coverage-observation`; sidecar exports `coverage_version: 1`.
- 2026-07-12 — Full-stage promotion of the coverage policy (no Stage 3 PPO):
  - Re-baselined the full stage at seed 2000 (100 episodes, medium,
    frame-skip 4): the July hierarchical table is stale post framework
    reset (old full bundle now measures 6537/3696 minerals and 0.85/0.18
    pickups on transition/sparse, vs the recorded 7468/4884, 4.15/0.37).
  - Coverage policy + hierarchical beacon controller beats the old full
    bundle on the same seeds: minerals +10%/+12%, unique cells +63%/+25%,
    pickups and beacon usage up, flips <=2%, 0% out-of-power. Known
    trade: transition beacon bonus -17% (beacons deployed on fresher,
    lower-scoring ground); address via the supervisor surface-score
    threshold, not PPO fine-tuning.
  - Promoted the coverage best checkpoint as `models/full/` with
    `coverage_version: 1` in the sidecar; staged under
    `runs/stage3_full_coverage_hierarchical/`. The full bundle and the
    minerals bundle now share the same policy weights by design — the
    full stage adds only the supervisor's beacon logic.
- 2026-07-12 — Supervisor stuck-recovery guard:
  - In-game, deployed beacons are fixed colliders, but headless training
    skips the collider, so the policy never learns them as obstacles and
    could wedge against one indefinitely (observed in play).
  - The supervisor now detects sustained throttle with speed below
    0.2 m/s for 20 consecutive decisions and issues 10 decisions of
    reverse-with-steering (`stuck_recovery` mode), alternating turn
    direction across recoveries. Tilt guard retains priority; coasting
    at rest never triggers it. Mode-based novelty suppression already
    excludes recovery motion from PPO coverage credit.
  - Matched eval (minerals_transition, seed 42, 20 episodes) is
    unchanged with the guard active; unit tests cover trigger, release,
    alternation, coast/moving non-triggers, and reset.
