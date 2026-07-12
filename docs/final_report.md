# Hylaean Rover — Draft Final Report

**A Bevy simulation and reinforcement-learning testbed for autonomous lunar surveying**

_Prepared by: Kyle Johnson, Jason Helland_

_Aided by: Claude Fable/Opus/Sonnet, OpenAI Sol, gpt5.5_

_Zeta Internal Research & Development MOONSHOT effort_

_Draft — 2026-07-12_


---

## Executive summary

This IR&D effort set out to build hands-on institutional experience in two
areas we expect to matter for future autonomy work: (1) building physics-based
simulations in the [Bevy](https://bevyengine.org/) game engine (Rust), and
(2) wrapping such a simulation as a reinforcement-learning (RL) environment and
training an agent to perform a non-trivial task inside it.

We built **Hylaean Rover**: a six-wheeled rover on a procedurally generated
lunar landscape that must drive without flipping, manage a finite power budget
by collecting energy cubes, survey a subsurface mineral field, and place a
small number of survey beacons on the most valuable deposits it can infer. The
same simulation code runs in two forms — a **playable game** with a full
telemetry HUD, and a **headless Gymnasium environment** that Stable
Baselines 3 (SB3) trains against with no rendering.

We trained a PPO agent through a **staged curriculum** (locomotion → power
management → mineral exploration → full mission with beacons), holding the
observation and action spaces fixed so each stage's learned weights warm-start
the next. Trained policies export to ONNX and run natively back in the game via
`tract-onnx`, so a learned policy can be watched driving in the full rendered
world with no Python in the loop.

The headline technical outcome is not a single benchmark number but a working
end-to-end pipeline and a catalog of hard-won lessons: most of our engineering
time went into diagnosing failures that *looked* like RL-optimizer instability
but were actually **environment bugs and train/deploy mismatches**. Those
lessons — documented stage by stage below — are the main deliverable.

---

## 1. Background

### 1.1 Motivation

Reinforcement learning is attractive for robotics and autonomy because it can
discover control and decision policies directly from interaction, without a
hand-written controller for every situation. But RL is notoriously
finicky: reward shaping, observation design, environment determinism, and the
sim-to-deployment gap all have to be right before the learning algorithm itself
matters. The cheapest place to build that intuition is a simulation we fully
control.

Two capabilities drove the choice of project:

- **Simulation in Bevy.** Bevy is a modern, data-oriented (ECS) Rust game
  engine. Rust's performance and safety are attractive for simulation, and an
  ECS architecture maps cleanly onto "many entities with physics and sensors."
  We wanted to learn whether Bevy is a viable base for building custom RL
  environments rather than reaching for an off-the-shelf simulator.
- **RL against a custom environment.** Rather than train on a canned Gym task,
  we wanted the full loop: design the observation and action spaces, write the
  reward, expose the simulator to Python, and confront the practical failure
  modes that only appear with a real, non-toy environment.

### 1.2 Why a lunar rover

A planetary surveying rover is a good vehicle for these goals because it layers
several distinct sub-problems that can be introduced one at a time:

- **Locomotion** — drive over rough terrain without tipping over.
- **Resource management** — a finite battery that depletes with driving and
  recharges from collectible power cubes.
- **Sensing and inference** — a mineral field whose *surface* readings are only
  loosely correlated with the *subsurface* deposits that actually score, so the
  agent must infer value from partial information.
- **Strategic decision-making** — a scarce budget of five survey beacons that
  should be spent on the highest-value deposits.

The lunar setting also let us ground the reward in real planetary-science
intuition (element abundances, ilmenite/titanium hotspots, water ice in
permanently shadowed regions, solar-wind-implanted helium-3), which made the
reward function's scarcity weighting more principled than arbitrary.

---

## 2. Goals

The effort had one meta-goal — **learn by building** — decomposed into concrete
objectives:

1. **Build a physics-based rover simulation in Bevy** with real wheel physics,
   suspension, steering, and procedurally generated terrain.
2. **Make the simulation dual-use**: a playable, inspectable game *and* a
   headless RL environment sharing one body of simulation code, so what the
   agent learns on is exactly what a human plays.
3. **Expose the environment to the Python RL ecosystem** via a standard
   `gymnasium.Env` interface, so Stable Baselines 3 (and any other library)
   can train against it.
4. **Train an agent to perform the full mission** using a curriculum that
   builds up from simple locomotion to strategic beacon placement without
   discarding earlier learning.
5. **Close the sim-to-deployment loop**: export a trained policy and run it
   natively back in the rendered game, so learning can be visually verified.
6. **Document the decisions and failure modes** thoroughly enough that the next
   effort starts from our lessons rather than rediscovering them.

Explicitly *out of scope*: photorealistic rendering, hardware-in-the-loop,
state-of-the-art sample efficiency, or beating a published benchmark. This was
a capability-building exercise, and the reward-shaping and long-horizon tuning
were deliberately treated as a means to surface lessons, not as an end.

---

## 3. Architecture

### 3.1 Guiding principle: one simulation, two front ends

The central architectural decision was to put all simulation logic in a shared
library crate and wire it into two different applications. This guarantees the
RL agent trains on the identical physics, sensors, and reward that a human sees
when playing — eliminating a whole class of sim-to-sim mismatch before it can
start.

```
hylaeanrover/  (Cargo workspace, Rust edition 2024)
├── crates/
│   ├── hylaeanrover_core/   shared sim: Bevy plugins, physics, sensors, reward
│   ├── hylaeanrover_game/   playable binary (DefaultPlugins + camera + HUD)
│   └── hylaeanrover_py/     PyO3 cdylib exposing RoverEnv to Python
├── python/
│   └── hylaeanrover/        gymnasium.Env wrapper, staged-reward wrappers, export
├── models/                  curated per-stage policy bundles (tracked in git)
└── docs/                    curriculum design + progress logs
```

The same `RoverCorePlugin` is added to both apps. UI-spawning systems no-op
gracefully when the font resource is absent (headless mode), and the rover is
spawned either from a glTF model (game) or as bare named entities the physics
setup picks up identically (RL). This "graceful degradation" pattern is what
makes one plugin serve both a rendered and a headless host.

### 3.2 Technology stack

| Layer | Choice | Notes |
|---|---|---|
| Engine / ECS | **Bevy 0.18** | Data-oriented Rust engine; systems + queries |
| Physics | **bevy_rapier3d 0.33** | Rigid bodies, colliders, joints, heightfield |
| Camera | bevy_panorbit_camera 0.34 | Orbit/free camera for inspection |
| Python bridge | **PyO3 + maturin** | Compiles the core into a Python C-extension |
| RL library | **Stable Baselines 3 (PPO)** + PyTorch | CPU training |
| Env interface | **Gymnasium** | `Discrete(10)` actions, `Box(41,)` observations |
| Policy deployment | **tract-onnx** (pure Rust) | Runs the exported ONNX policy in-game |

### 3.3 The simulation (`hylaeanrover_core`)

Key subsystems, each a Bevy module:

- **Terrain** (`terrain.rs`) — a procedural heightfield stamped with ~1000
  parabolic impact craters, forming a single ~5 km "mega-bowl" arena with a
  Rapier heightfield collider. Terrain height scale is configurable so the
  agent can be trained across a range of roughness.
- **Rover** (`rover.rs`) — a chassis with four wheels on prismatic
  spring/damper suspension joints, real wheel-pivot steering (opposite-phase
  four-wheel steer), throttle, and respawn. Actions arrive as a `RoverAction`
  (throttle, steering, beacon-drop).
- **Power** (`power_cubes.rs`) — a battery that drains with driving and
  regenerates slightly on coast; glowing power cubes spawn via a Poisson
  process and recharge the battery when collected. Running the battery to zero
  is a failure condition.
- **Minerals** (`minerals.rs`) — six-element procedural concentration maps
  (Si, Al, Fe, Ti, H₂O, ³He) grounded in lunar geochemistry, with rare deposits
  (Ti, water, helium-3) modelled as Gaussian mixtures. Crucially, *surface*
  readings are only loosely correlated with *subsurface* value.
- **Beacons** (`beacons.rs`) — a five-beacon budget; each placement scores the
  scarcity-weighted subsurface concentration at that spot.
- **Sensors / IMU** (`imu.rs`, `observation.rs`) — speed, heading, pitch/roll,
  yaw rate, accelerations, an 8-ray forward lidar fan, and a "visible cubes"
  sensor (bearing/range to nearby cubes). A single shared
  `build_observation()` produces the 41-float vector consumed by both the RL
  env and the in-game autopilot, guaranteeing they see identical inputs.
- **Reward** (`reward.rs`) — the scoring function (Section 5).
- **Game state** (`game_state.rs`) — three terminal conditions with an
  "at-rest" debounce so momentary events don't end a run.
- **Mission supervisor** (`mission_supervisor.rs`) — a hand-written
  hierarchical safety/decision controller wrapped around the learned policy
  (Section 6.4).
- **Survey coverage** (`survey_coverage.rs`) — a per-game visited-cell grid
  enabling coverage-aware exploration (Section 6.5).

### 3.4 The RL environment (`hylaeanrover_py` + `python/hylaeanrover`)

The Rust `RoverEnv` pyclass drives the Bevy `App` with a fixed 1/60 s timestep
and a deterministic `step()`. The Python layer wraps it as a `gymnasium.Env`,
adds staged-reward wrappers, `VecNormalize`, action-repeat/frame-skip, and the
SB3 training harness.

**Notable constraints that shaped the design:**

- The Bevy `App` is `!Send`, so the env is `unsendable`. Parallel training uses
  SB3's `SubprocVecEnv` (separate processes), not thread-based vectorization.
- No live rendering during training: Bevy's windowing (winit) needs the OS main
  thread, which conflicts with Python owning it. The chosen substitute is
  **export-to-ONNX and watch in the native game** — which also exercises the
  real deployment path.

### 3.5 The deployment loop

A trained SB3 policy is exported to ONNX plus a `.norm.json` sidecar carrying
the `VecNormalize` observation statistics, frame-skip, and stage config
(battery capacity, whether beacons are enabled, whether the mission supervisor
and coverage observation are active). The game's `autopilot.rs` loads the ONNX
network with `tract-onnx`, builds the *same* 41-float observation each frame,
normalizes it with the exported stats, and argmaxes the policy logits into a
`RoverAction`. In-game, **P** toggles autopilot and **O** hot-reloads the
policy file, so a checkpoint can be re-exported mid-training and watched
improving.

---

## 4. RL algorithm and why we chose it

### 4.1 PPO

We used **Proximal Policy Optimization (PPO)** from Stable Baselines 3 as the
single learning algorithm throughout. Rationale:

- **Robust default for continuous-control-style tasks.** PPO's clipped
  surrogate objective makes it forgiving of hyperparameter choices relative to,
  say, vanilla policy gradients or value-based methods, which matters when the
  environment itself is still being debugged.
- **On-policy stability with a warm-start story.** Because we planned a
  curriculum with weight transfer between stages, we wanted an algorithm where
  loading a previous policy and continuing is natural. PPO's on-policy updates
  plus a low clip range and optional target-KL early-stopping let us make
  *conservative* updates that preserve a warm-started policy.
- **Mature, well-supported implementation.** SB3's PPO comes with
  `VecNormalize`, vectorized envs, checkpointing, TensorBoard logging, and an
  `EvalCallback` — the training *harness* we would otherwise have had to build.
- **Small policy, CPU-friendly.** The policy is a small MLP (`MlpPolicy`). The
  bottleneck is the physics sim, not the network, so we kept everything on CPU
  and scaled throughput with parallel sim processes and frame-skip rather than
  a GPU.

**Discrete action space.** We chose `Discrete(10)` (nine throttle×steering
combinations plus a beacon-drop) rather than a continuous action space. A small
discrete space is easier to learn and to reason about, is trivially compatible
across all curriculum stages, and made the "freeze the action head" transfer
strategy clean.

### 4.2 Observation design

The 41-float observation (`Box(41,)`) is: IMU/telemetry (7) + 8-ray lidar (8) +
six visible cubes × (bearing, range, valid) (18) + power fraction (1) + six
surface mineral concentrations (6) + beacons remaining (1).

Two deliberate design decisions:

- **Power as a fraction of capacity, not raw watt-hours.** This makes the
  observation invariant to battery size, so a policy trained on a small 100 Wh
  training battery reads the same signal when deployed on the game's 1 kWh
  battery.
- **Cumulative reward components and the game-over flag are excluded.** They are
  unbounded within an episode and non-Markovian — bad policy inputs. They
  remain available in the `info` dict for reward shaping and logging.

Keeping the observation shape fixed at 41 across every stage is what makes
weight transfer between curriculum stages possible.

### 4.3 Supporting techniques

- **`VecNormalize`** for running observation (and reward) normalization — the
  raw observation mixes degrees, meters, and fractions on wildly different
  scales.
- **Frame-skip / action-repeat** — hold each action K physics ticks (typically
  K=4) so the policy makes fewer, coarser decisions; faster credit assignment
  and throughput. The same K must be used in training, eval, and the autopilot.
- **Behavior-cloning "teacher" bootstraps** — for the harder sub-skills we
  pretrain the policy head with cross-entropy against a scripted teacher (e.g.
  drive toward a visible cube, coast when the battery is low and nothing is
  actionable) before letting PPO fine-tune. A reward-shaped PPO probe alone did
  not reliably move entrenched motor habits.

---

## 5. The reward function

The reward is purely **incremental** — SB3 sees `total_now − total_last_step` —
and combines four components (`reward.rs`):

| Component | Weight | Intent |
|---|---|---|
| Distance driven | 1.0 / m | Dense locomotion signal |
| Mineral line integral | 0.1 × scarcity-weighted | Reward covering valuable ground |
| Power cube pickup | 100 per cube (paid smoothly over the ~0.5 s charge) | Reward energy management |
| Beacon placement | **50× scarcity-weighted subsurface score** | Reward strategic guessing |

The per-element **scarcity weights** — Si=1, Al=2, Fe=3, Ti=10, H₂O=20, ³He=50
— encode that finding a water or helium-3 deposit is worth orders of magnitude
more than mileage. This makes the terminal beacon decision the dominant lever
in the full mission, exactly as intended.

A key enabler: the reward is emitted **per component** in the `info` dict, so
each curriculum stage recomputes its own effective reward in Python (a
`StagedRewardWrapper`) from the component deltas — **without changing the Rust
reward math**. Reward shaping stayed fast to iterate in Python while the
simulation stayed frozen.

---

## 6. Training stages and lessons learned

The core methodology is a **staged curriculum with weight transfer**. The
observation space (41) and action space (`Discrete(10)`) are fixed *once, up
front, and never change*. Only the reward changes between stages. Because the
policy network's input and output shapes never change,
`PPO.load(prev).set_env(new_env)` transfers every layer, and each stage starts
from a policy that already solves the previous stage's sub-skill. Nothing is
discarded. `VecNormalize` observation stats carry across stages; reward
normalization is reset at each stage boundary (reward scale jumps) and
preserved for same-stage continuation.

The go/no-go gate between every stage is an explicit, recorded evaluation
against a random baseline and the previous stage's policy. We did not advance on
"the loss went down" — we advanced on measured behavior.

### Stage 0 — Locomotion

**Objective.** Drive far and stay upright. Reward = distance delta (plus a small
flip penalty and alive bonus); cube/mineral/beacon bonuses excluded. This is the
densest, fastest-to-learn signal and proves the pipeline learns at all.

**What happened.** The agent learned to drive 2–2.3× the random-baseline
distance on medium horizons across a range of terrain roughness, with low flip
rates. But the strict short-horizon gate (≥2× random) was only *just* missed on
some terrains, and the policy tended to **sprint until it ran out of power**
(out-of-power rates of 85–100% on medium/long horizons).

**Lessons.**

- A naive distance reward produces a "floor it until the battery dies" policy.
  We added explicit **power-efficiency shaping** (coast bonus, power-draw
  penalty, regen reward, out-of-power penalty) to make pacing learnable — this
  was the difference between passing and failing several terrain gates.
- Reaching the gate took a *sequence* of fine-tunes (continue at lower LR →
  medium-horizon fine-tune → power-efficiency recovery → single-terrain
  recovery → short-horizon rebalance), each warm-started from the last. The
  documented training history is a realistic picture of how much iteration a
  "simple" first stage actually needs.
- We made a pragmatic engineering call: the first-pass Stage 0 model was
  promoted for *pipeline validation* while still short of the strict gate,
  explicitly flagged as "not final-quality locomotion," so downstream stages
  weren't blocked. Being honest in the record about that trade-off matters.

### Stage 1 — Power management

Stage 1 turned out to need splitting into three sub-stages, driven by observed
failure modes:

- **1A — Cube intercept.** One settled, visible cube, no random spawns, with
  intercept shaping and (critically) a behavior-cloning teacher bootstrap. This
  isolates the missing "see a cube → commit to it → pick it up" skill.
- **1B — Power idle.** No cubes, low starting battery. Teaches the policy *not*
  to burn a nearly empty battery when nothing actionable is visible — a
  discipline that reward-shaped PPO alone could not instill, but a teacher
  dataset of coast-when-idle labels could.
- **1C — Power cubes.** Broad training over mixed dense/bridge/transition/
  sparse cube densities, warm-started from 1B. Distance reward is turned *off*
  here so pickup and power behavior are hardened before travel reward returns.

**Lessons — this stage produced the most instructive bugs of the project:**

- **"Policy collapse" was an environment bug, not the optimizer.** Policies
  would climb for ~200k steps then permanently collapse. Root cause: **the
  battery was never reset between episodes.** The env's `reset()` bypassed the
  UI "relaunch" button that was the only thing refilling power, so each training
  process effectively had *one battery for its entire life*. Once a policy's
  cumulative distance exhausted it, every subsequent episode started dead.
  Better policies died sooner — which is exactly why it looked like "learns then
  collapses" instability. The fix was a one-line-conceptually,
  deep-in-consequences change to refill power on the reset event.
- **A single-tick reward spike wrecks PPO.** A `+100` cube-pickup bonus credited
  on a single frame was 100–1000× the surrounding per-tick signal and landed in
  only ~half of episodes. That variance destroyed GAE advantage estimates and
  blew up updates (KL spike, explained-variance crash). The fix: **pay the same
  total smoothly over the ~0.5 s charge** (~3.3/tick). Per-episode totals are
  identical; the learning signal is night-and-day. *Reward variance, not just
  reward magnitude, is a first-class design concern.*
- **Curriculum spawn geometry has to track the agent.** Cubes anchored to the
  world origin stopped intersecting the path of a policy that already drives
  200 m away, yielding near-zero learning signal no matter how we tuned density.
  Fixing it required spawning cubes in an annulus around the rover's *current*
  position. "Train dense, deploy sparse" became an explicit principle: the
  training cube density is a curriculum tool and is deliberately *not* carried
  into the game.
- **Behavior cloning unblocks what shaping can't.** For both cube intercept and
  power-idle discipline, a scripted-teacher pretrain of the policy head was
  necessary before PPO could fine-tune; pure reward shaping did not reliably
  overcome entrenched habits from the warm-started policy.
- **Acceptance must be measured with shaping OFF.** Dense-field pickup success
  is not promotable on its own; we required nonzero pickups in genuinely sparse,
  low-power scenarios with all training shaping disabled before advancing.

### Stage 2 — Mineral exploration

**Objective.** Reward = distance + scarcity-weighted mineral line integral,
warm-started from Stage 1. The motor and survival skills transfer; the new task
is to cover valuable ground.

**Lessons.**

- **The mineral signal is local, not directional.** The observation gives
  concentration *under* the rover, not a gradient pointing at richer ground, so
  there is nothing to "climb." The learnable behavior is therefore *explore
  broadly while powered* — which we had to restore with a short exploration
  teacher bootstrap because Stage 1 had trained a cube-seeking prior that sat
  and waited for cube drops.
- **Checkpoint selection has to price in terminal failure.** Short-horizon mean
  reward hid both late-episode failures and rare high-mineral outliers. We moved
  to a **median return with an explicit terminal-failure penalty**, evaluated on
  the medium horizon (where the power-exhaustion and rollover tail actually
  appears), selecting on the worst score across transition distributions.
- **Beware the oversized global safety knob.** A large blanket battery-draw
  penalty reduced flips but *increased* out-of-power failures — trading one
  terminal failure for another. The lesson: safety pressure has to be targeted
  (e.g. dense excessive-tilt shaping) rather than applied as a single global
  cost.

### Stage 3 — Full mission (beacons)

**Objective.** Add the beacon bonus and enable beacon placement so the agent
learns *when* to spend its five beacons.

**Lessons — the biggest strategic decision of the project:**

- **We did not train beacon placement with PPO.** A direct PPO probe learned the
  beacon reward but cut sparse-terrain mineral reward by more than half — it
  optimized the new bonus at the expense of the exploration it was supposed to
  build on. Instead we **froze the accepted Stage 2 exploration policy and
  wrapped it in a hand-written hierarchical controller** (the mission
  supervisor) that owns beacon placement: it requires 100 m of exploration
  before the first beacon, 75 m spacing, and a minimum scarcity-weighted surface
  score. This is a deliberate "learn what's learnable, script what's scriptable"
  boundary — a legitimate and often *preferable* architecture over end-to-end RL
  for sparse, safety-relevant, rule-like decisions.
- **A learned policy needs a deployment guardian.** The same supervisor also
  handles survival concerns that don't belong in the learned policy: it masks
  cube slots and presents constant healthy power to PPO while the battery is
  fine (so the policy stays a pure mineral-explorer), takes over with a
  deterministic, energy-feasibility-checked cube intercept only when power is
  genuinely low, brakes dangerous tilt with a speed-gated guard, and even runs a
  stuck-recovery maneuver — because deployed beacons are solid colliders in the
  game that the headless-trained policy never learned to avoid.
- **Train/deploy config mismatches are their own failure class.** A `power_cubes`
  checkpoint watched in-game drove out to 200 m and rapid-fired all five
  beacons. Cause: training ran with beacons disabled (action 9 an inert no-op),
  but the game always has beacons live, and at 200 m the rover was far outside
  anything its 100 Wh training battery ever reached. The fix was to record the
  stage's battery/beacon config in the export sidecar and replay a checkpoint
  under the *same* conditions it trained in. **What you train under and what you
  deploy under must be reconciled explicitly.**

### Extension — Coverage-aware exploration

A later refinement gave mineral/full policies a **per-game survey memory**: a
saturating byte per 5 m cell over the ~5 km arena (a 990×990 grid), reset each
run, feeding 18 rover-relative "frontier" features into the policy through the
supervisor-hidden cube slots — **without changing `OBS_DIM=41`.** The reward
pays first-visit distance and first-visit mineral integral, so retracing old
ground earns nothing. Warm-started from the promoted minerals policy (repurposed
input columns reset once), the coverage candidate improved unique cells covered
per episode by **+32–107%** at medium horizon and **+88%** on the long sparse
horizon, while holding flips, out-of-power, and mineral score within gates. It
was promoted as the mineral and full policy.

**Lesson.** Re-anchoring an evaluation metric mid-project is sometimes correct:
the original novelty metric (`novel_distance_per_100m`) was mathematically
capped such that a short-path baseline already scored near the ceiling, making a
30% improvement *unreachable for any policy*. The real benefit — total novel
ground covered — needed a different metric. **A gate is only as good as the
metric behind it.**

---

## 7. Cross-cutting lessons

Distilling the stage-by-stage record into transferable takeaways:

1. **Most "RL instability" was environmental.** The two worst-looking optimizer
   pathologies (200k-step collapses) were a battery that never reset and a
   single-tick reward spike. Before blaming PPO, instrument the *environment* —
   we added per-rollout logging of terminal reasons and end-power specifically
   so env-driven failures stop masquerading as learning failures.
2. **Reward variance is as important as reward shape.** Smearing a bonus over
   time changed nothing about the objective and everything about trainability.
3. **Freeze the interface, vary the reward.** Fixing observation and action
   shapes up front is what made a four-stage curriculum with weight transfer
   possible; it cost some up-front discipline (e.g. keeping a 10-way action head
   with an inert beacon action in early stages) and repaid it many times over.
4. **Behavior cloning is a practical unblock.** Scripted-teacher pretraining
   moved skills that reward shaping alone could not.
5. **Hierarchy beats end-to-end for sparse, rule-like, safety-critical
   decisions.** A frozen learned explorer under a hand-written supervisor
   outperformed direct PPO on the full mission and was far easier to reason
   about and make safe.
6. **The sim-to-deployment gap is concrete and debuggable.** Sharing one
   observation builder between env and game, and recording training config in
   the export sidecar, turned "why does it behave differently in the game" from
   a mystery into a checklist.
7. **Gate on measured behavior, not training curves.** Every stage advanced only
   after explicit, recorded, shaping-off evaluation against baselines.

---

## 8. What we built (deliverables)

- A **playable Bevy rover game** with full physics (Rapier suspension, wheel
  steering), procedural lunar terrain, six-element mineral overlays, power
  cubes, survey beacons, a live telemetry HUD, and a survey-coverage
  mini-heatmap.
- A **headless Gymnasium RL environment** wrapping the identical simulation via
  PyO3, with `VecNormalize`, subprocess vectorization, frame-skip, staged-reward
  wrappers, teacher-bootstrap tooling, and a full SB3 training/evaluation/export
  harness.
- A **four-stage trained policy curriculum** (locomotion → power cubes →
  minerals → full) with curated, git-tracked model bundles per stage.
- A **native ONNX autopilot** (`tract-onnx`) that runs any trained policy back
  in the rendered game with hot-reload, plus a hierarchical **mission
  supervisor** shared between training and deployment.
- **Thorough documentation**: a curriculum design doc with a dated progress log,
  a Stage 0/1 hardening plan, a Stage 0 training summary, and a step-by-step
  operating guide — the record that makes the lessons above reusable.

---

## 9. Limitations and future work

- **No live rendering during training** (winit/Python main-thread conflict). The
  export-and-watch loop is the substitute; an offscreen `rgb_array` render pass
  is the natural next step.
- **Locomotion never cleared the strict short-horizon gate** and was promoted as
  a pipeline-validation checkpoint. Final-quality locomotion is unfinished work.
- **Beacon strategy is scripted, not learned.** This was a deliberate and, we
  argue, correct choice — but learning the strategic layer end-to-end remains an
  open problem if richer beacon-value observations were provided.
- **Sample efficiency was not a goal.** Training relied on CPU parallelism and
  frame-skip; no attempt was made to minimize timesteps.
- **The mineral observation is local, not directional**, which fundamentally
  limits how targeted exploration can be. A directional gradient or a learned
  belief map over the subsurface field would change the problem substantially.
- **Single-process env constraint** (`!Send` Bevy `App`) means only
  process-based parallelism is available.

Natural next directions: an offscreen render path; a proper belief-state
representation of the subsurface mineral field; learning the beacon policy with
a supervisor-shaped curriculum rather than a fully scripted controller; and
porting the shared-core, dual-front-end pattern to a second, different task to
test how reusable the architecture really is.

---

## 10. Conclusion

The effort met its primary goal: we now have direct, documented experience
building a physics simulation in Bevy and training an RL agent against it
end-to-end, from custom environment design through curriculum training to native
in-game deployment. The most valuable output is not the trained rover but the
catalog of failure modes and design principles — especially the repeated finding
that what presents as RL-optimizer trouble is usually an environment,
reward-variance, or train/deploy-config problem hiding behind it. Those lessons,
and the reusable shared-core/dual-front-end architecture that surfaced them, are
what we carry forward.

---

_Appendix pointers: `docs/rl_training_plan.md` (curriculum design + full dated
progress log), `docs/rl_stage0_stage1_hardening_plan.md` (Stage 0/1 acceptance
gates), `docs/stage0_locomotion_training_summary.md` (Stage 0 training history),
`python/TRAINING_GUIDE.md` (step-by-step operating checklist),
`python/README.md` (action/observation schema, supervisor, coverage)._
