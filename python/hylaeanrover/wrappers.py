"""Staged reward shaping for the RL curriculum.

The training plan (see `docs/rl_training_plan.md`) trains the rover in
stages that share one fixed observation/action space and differ *only*
in the reward, so each stage's policy weights initialize the next:

    locomotion  →  power_cubes  →  minerals  →  full

`StagedRewardWrapper` recomputes the per-step reward from the cumulative
reward components the Rust env already exposes in its `info` dict
(`reward_distance`, `reward_cube_bonus`, `reward_mineral_integral`,
`reward_beacon_bonus`), weighting them per stage. Keeping the shaping in
Python means we can retune it without rebuilding the Rust extension.

`make_staged_env` is the convenience constructor: it picks the right
`beacons_enabled` for the stage (off for locomotion/minerals so action
index 9 is an inert no-op) and wraps the env.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import gymnasium as gym

from hylaeanrover import RoverEnv

# Per-stage weights on the four reward components. The component values
# themselves come from the Rust reward (distance in meters, the flat
# per-pickup cube bonus, the scarcity-weighted mineral line-integral, and
# the 50x beacon bonus). Once a component's weight turns on for a stage it
# stays on for every later stage too — nothing already learned is
# discarded, only new objectives are layered on top (reward annealing).
STAGE_WEIGHTS: dict[str, dict[str, float]] = {
    # Drive far, stay upright, manage power. Densest signal.
    "locomotion": {"distance": 1.0, "cube": 0.0, "mineral": 0.0, "beacon": 0.0},
    # Also reward actively seeking out and collecting power cubes.
    "power_cubes": {"distance": 1.0, "cube": 1.0, "mineral": 0.0, "beacon": 0.0},
    # Also reward crossing scarce-mineral ground.
    "minerals": {"distance": 1.0, "cube": 1.0, "mineral": 1.0, "beacon": 0.0},
    # Full mission, including strategic beacon placement.
    "full": {"distance": 1.0, "cube": 1.0, "mineral": 1.0, "beacon": 1.0},
}

STAGES = tuple(STAGE_WEIGHTS.keys())

# Battery capacity (Wh) for training episodes, all stages. The game's
# 1 kWh battery lasts ~2 km of driving — far more than one episode can
# use — so with it, power never affects an episode and the policy can't
# learn power management. This value is sized so a full-throttle episode
# runs dry before the step limit (measured: flat-out over a 2000-tick
# episode covers ~320 m and costs ~158 Wh, so 100 Wh dies ~63% in and
# caps naive driving at ~230 m), making pacing / regen braking (coasting
# recharges at 20% efficiency) part of the learned behavior.
# The observation exposes power only as a fraction-of-capacity, so a
# policy trained at this capacity transfers to the game's 1 kWh battery.
# Applied to every stage so later stages don't unlearn it.
DEFAULT_POWER_CAPACITY_WH = 100.0

# Power-cube spawn rate (cubes/sec) and spawn-region max radius (m) for
# the power_cubes stage onward — TRAINING ONLY. These are deliberately
# NOT exported to the game (see export.py): they exist to give PPO enough
# positive seek-and-collect examples per episode to learn from, not to
# define the gameplay feel. The game keeps its own sparse, periodic rate
# (0.05/sec) where a cube is a scarce lifeline, and the learned skill —
# steer toward a sensed cube when one exists — transfers across densities
# because it's a local reactive behavior, not a density-dependent one
# ("train dense, deploy sparse").
#
# Spawns are drawn in an annulus [10m, extent) around the rover's
# *current* position (see power_cubes::spawn_power_cubes): rover-anchored
# because an origin-anchored region stops overlapping a competent
# driver's path almost immediately (the warm-started policy covers
# ~200m/episode), and with a minimum distance so every cube requires real
# navigation — without it, a near-stationary policy could wait for cubes
# to land within reach.
#
# Values calibrated against the actual `models/locomotion` warm-start
# policy (a random policy is a poor proxy for a driver that covers
# ~200m/episode): measured sweep over the annulus mechanic gave
# 1.0/30m -> 0.20, 1.0/40m -> 0.30, 1.5/30m -> 0.52, 2.0/30m -> 0.70
# pickups/episode. 1.5/30m is the chosen balance: a positive example
# roughly every other episode (enough signal for PPO) at less than half
# the spawn volume of denser settings that visually "rained" cubes.
# Re-measure with `PPO.load(...)` + a few dozen eval episodes (see git
# history for the calibration script) if retuning.
DEFAULT_CUBE_SPAWN_LAMBDA = 1.5
DEFAULT_CUBE_SPAWN_EXTENT = 30.0

CUBE_SPAWN_PRESETS: dict[str, dict[str, float]] = {
    # Current Stage 1 training density: enough positives for gradient signal.
    "dense_training": {
        "lambda": DEFAULT_CUBE_SPAWN_LAMBDA,
        "extent": DEFAULT_CUBE_SPAWN_EXTENT,
    },
    # Bridge between training density and the game's sparse lifeline cadence.
    "transition": {"lambda": 0.30, "extent": 120.0},
    # Matches the game's current defaults.
    "sparse_game": {"lambda": 0.05, "extent": 500.0},
    # Negative-control scenario: no power cubes should appear.
    "none": {"lambda": 0.0, "extent": DEFAULT_CUBE_SPAWN_EXTENT},
}

HORIZON_STEPS: dict[str, int] = {
    "short": 2_000,
    "medium": 7_200,
    "long": 21_600,
}

TERRAIN_HEIGHT_PRESETS: dict[
    str, tuple[Optional[float], Optional[tuple[float, float]]]
] = {
    "fixed_1_0": (1.0, None),
    "fixed_1_5": (1.5, None),
    "fixed_2_0": (2.0, None),
    "mixed_1_2": (None, (1.0, 2.0)),
}

EVAL_SCENARIOS: dict[str, dict[str, Any]] = {
    "dense_training": {"cube_spawn_preset": "dense_training"},
    "transition": {"cube_spawn_preset": "transition"},
    "sparse_game": {"cube_spawn_preset": "sparse_game"},
    "low_power_start": {
        "cube_spawn_preset": "transition",
        "power_start_fraction": 0.35,
        "cube_shaping": "off",
    },
    "cube_visible_low_power": {
        "cube_spawn_preset": "dense_training",
        "power_start_fraction": 0.35,
        "cube_shaping": "off",
    },
    "no_cube_control": {
        "cube_spawn_preset": "none",
        "power_start_fraction": 0.35,
        "cube_shaping": "off",
    },
    "terrain_fixed_1_0": {"terrain_height": "fixed_1_0"},
    "terrain_fixed_1_5": {"terrain_height": "fixed_1_5"},
    "terrain_fixed_2_0": {"terrain_height": "fixed_2_0"},
    "terrain_mixed_1_2": {"terrain_height": "mixed_1_2"},
}

SCENARIOS = tuple(EVAL_SCENARIOS.keys())
CUBE_SHAPING_MODES = ("off", "low_power")
LOCOMOTION_SHAPING_MODES = ("off", "power_efficiency")


def apply_scenario_defaults(
    scenario: Optional[str],
    *,
    cube_spawn_preset: Optional[str] = None,
    power_start_fraction: Optional[float] = None,
    terrain_height: Optional[str] = None,
    cube_shaping: Optional[str] = None,
) -> dict[str, Any]:
    """Resolve scenario + explicit CLI overrides into env kwargs."""
    cfg: dict[str, Any] = {}
    if scenario:
        if scenario not in EVAL_SCENARIOS:
            raise ValueError(f"unknown scenario {scenario!r}; choose from {SCENARIOS}")
        cfg.update(EVAL_SCENARIOS[scenario])
    if cube_spawn_preset is not None:
        cfg["cube_spawn_preset"] = cube_spawn_preset
    if power_start_fraction is not None:
        cfg["power_start_fraction"] = power_start_fraction
    if terrain_height is not None:
        cfg["terrain_height"] = terrain_height
    if cube_shaping is not None:
        cfg["cube_shaping"] = cube_shaping
    return cfg


def parse_terrain_height(
    value: Optional[str],
) -> tuple[Optional[float], Optional[tuple[float, float]]]:
    if value is None:
        return None, None
    if value in TERRAIN_HEIGHT_PRESETS:
        return TERRAIN_HEIGHT_PRESETS[value]
    if ":" in value:
        lo_s, hi_s = value.split(":", 1)
        lo, hi = float(lo_s), float(hi_s)
        if lo <= 0.0 or hi <= 0.0 or hi < lo:
            raise ValueError(
                "terrain height range must be positive and ordered as min:max"
            )
        return None, (lo, hi)
    scale = float(value)
    if scale <= 0.0:
        raise ValueError("terrain height scale must be positive")
    return scale, None


class ActionRepeat(gym.Wrapper):
    """Hold each chosen action for `k` physics ticks (frame-skip).

    The simulator runs at a fixed 1/60 s tick. Driving doesn't need a
    fresh decision every tick, so repeating each action `k` times means
    the policy makes 1/k as many decisions per second of sim — fewer
    network/observation passes and faster credit assignment, at the cost
    of coarser control. Episodes stay the same length in *sim-time*
    because the underlying env truncates on its own tick count.

    The beacon action (index 9) is edge-triggered: it fires only on the
    first sub-tick and the remaining sub-ticks coast (action 4), so one
    decision drops at most one beacon — matching the in-game autopilot.
    """

    def __init__(self, env: gym.Env, k: int) -> None:
        super().__init__(env)
        self.k = max(1, int(k))

    def step(self, action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        total_reward = 0.0
        terminated = truncated = False
        obs: Any = None
        info: dict[str, Any] = {}
        for i in range(self.k):
            # Repeat motor actions; fire a beacon only on the first tick.
            sub = action if i == 0 else (4 if action == 9 else action)
            obs, reward, terminated, truncated, info = self.env.step(sub)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class StagedRewardWrapper(gym.Wrapper):
    """Recompute reward from cumulative info components, weighted by stage.

    Adds an optional one-off ``flip_penalty`` when the episode terminates
    with the rover flipped, to discourage tipping over.
    """

    def __init__(
        self,
        env: gym.Env,
        stage: str,
        flip_penalty: float = 50.0,
        scenario: Optional[str] = None,
        cube_spawn_preset: Optional[str] = None,
        locomotion_shaping: str = "off",
        cube_shaping: str = "off",
        low_power_threshold: float = 0.45,
        cube_approach_reward: float = 0.25,
        ignored_cube_penalty: float = 25.0,
        coast_distance_bonus: float = 0.35,
        power_draw_penalty: float = 40.0,
        power_recovery_reward: float = 20.0,
        out_of_power_penalty: float = 75.0,
    ) -> None:
        if stage not in STAGE_WEIGHTS:
            raise ValueError(f"unknown stage {stage!r}; choose from {STAGES}")
        if locomotion_shaping not in LOCOMOTION_SHAPING_MODES:
            raise ValueError(f"unknown locomotion_shaping {locomotion_shaping!r}")
        if cube_shaping not in CUBE_SHAPING_MODES:
            raise ValueError(f"unknown cube_shaping {cube_shaping!r}")
        super().__init__(env)
        self.stage = stage
        self._w = STAGE_WEIGHTS[stage]
        self.flip_penalty = flip_penalty
        self.scenario = scenario
        self.cube_spawn_preset = cube_spawn_preset
        self.locomotion_shaping = locomotion_shaping
        self.cube_shaping = cube_shaping
        self.low_power_threshold = low_power_threshold
        self.cube_approach_reward = cube_approach_reward
        self.ignored_cube_penalty = ignored_cube_penalty
        self.coast_distance_bonus = coast_distance_bonus
        self.power_draw_penalty = power_draw_penalty
        self.power_recovery_reward = power_recovery_reward
        self.out_of_power_penalty = out_of_power_penalty
        # Cumulative component totals from the previous step, so we can
        # take deltas. Seeded in reset().
        self._prev = {"distance": 0.0, "cube": 0.0, "mineral": 0.0, "beacon": 0.0}
        self._prev_power_frac = 1.0
        self._prev_visible_range: Optional[float] = None
        self._low_power_visible_steps = 0

    @staticmethod
    def _components(info: dict[str, Any]) -> dict[str, float]:
        return {
            "distance": float(info.get("reward_distance", 0.0)),
            "cube": float(info.get("reward_cube_bonus", 0.0)),
            "mineral": float(info.get("reward_mineral_integral", 0.0)),
            "beacon": float(info.get("reward_beacon_bonus", 0.0)),
        }

    def _annotate_info(self, info: dict[str, Any]) -> None:
        if self.scenario is not None:
            info["scenario"] = self.scenario
        if self.cube_spawn_preset is not None:
            info["cube_spawn_preset"] = self.cube_spawn_preset
        info["locomotion_shaping"] = self.locomotion_shaping
        info["cube_shaping"] = self.cube_shaping

    @staticmethod
    def _is_coast_action(action: int) -> bool:
        # 3, 4, 5 are zero-throttle steering/coast actions. In non-full
        # stages action 9 is also a no-op and ActionRepeat coasts after
        # the first sub-tick, so treat it as coast-shaped too.
        return int(action) in (3, 4, 5, 9)

    def _locomotion_power_shaping(
        self,
        action: int,
        info: dict[str, Any],
        terminated: bool,
        distance_delta: float,
    ) -> float:
        if self.stage != "locomotion" or self.locomotion_shaping == "off":
            return 0.0

        power_frac = float(info.get("power_frac", self._prev_power_frac))
        power_delta = power_frac - self._prev_power_frac
        shaped = 0.0

        if distance_delta > 0.0 and self._is_coast_action(action):
            shaped += distance_delta * self.coast_distance_bonus
        if power_delta < 0.0:
            shaped -= (-power_delta) * self.power_draw_penalty
        else:
            shaped += power_delta * self.power_recovery_reward
        if terminated and info.get("game_over") == "out_of_power":
            shaped -= self.out_of_power_penalty
        return shaped

    def _cube_approach_shaping(
        self,
        info: dict[str, Any],
        terminated: bool,
    ) -> float:
        if self.stage != "power_cubes" or self.cube_shaping == "off":
            return 0.0
        power_frac = float(info.get("power_frac", 1.0))
        visible = int(info.get("visible_cube_count", 0) or 0) > 0
        nearest_raw = info.get("nearest_visible_cube_range")
        nearest = float(nearest_raw) if nearest_raw is not None else None
        if power_frac > self.low_power_threshold or not visible or nearest is None:
            self._prev_visible_range = nearest if visible else None
            return 0.0

        self._low_power_visible_steps += 1
        shaped = 0.0
        if self._prev_visible_range is not None:
            # Positive when the policy moves toward the visible cube,
            # negative when it moves away while power is binding.
            shaped += (self._prev_visible_range - nearest) * self.cube_approach_reward
        self._prev_visible_range = nearest

        if terminated and info.get("game_over") == "out_of_power":
            shaped -= self.ignored_cube_penalty
        return shaped

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[Any, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        self._prev = self._components(info)
        self._prev_power_frac = float(info.get("power_frac", 1.0))
        self._prev_visible_range = None
        self._low_power_visible_steps = 0
        self._annotate_info(info)
        return obs, info

    def step(self, action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        obs, _reward, terminated, truncated, info = self.env.step(action)

        cur = self._components(info)
        deltas = {
            key: cur[key] - self._prev[key]
            for key in ("distance", "cube", "mineral", "beacon")
        }
        shaped = (
            self._w["distance"] * deltas["distance"]
            + self._w["cube"] * deltas["cube"]
            + self._w["mineral"] * deltas["mineral"]
            + self._w["beacon"] * deltas["beacon"]
        )
        shaped += self._locomotion_power_shaping(
            int(action), info, terminated, deltas["distance"]
        )
        shaped += self._cube_approach_shaping(info, terminated)
        self._prev = cur
        self._prev_power_frac = float(info.get("power_frac", self._prev_power_frac))

        if terminated and info.get("game_over") == "flipped":
            shaped -= self.flip_penalty

        self._annotate_info(info)
        info["low_power_visible_steps"] = self._low_power_visible_steps
        return obs, float(shaped), terminated, truncated, info


def resolve_vecnorm(model_path: str, vecnorm_arg: Optional[str]) -> str:
    """Resolve the VecNormalize stats to apply alongside a loaded model.

    A policy loaded *without* its `VecNormalize` stats silently receives
    un-normalized observations (wildly different scale from training) and
    produces near-random actions — so train.py / evaluate.py must always
    pair `--load` with the matching stats. `train.py` saves them as
    `vecnorm.pkl` next to `model.zip`, so when `--vecnorm` is omitted we
    fall back to that sibling and fail loudly if it's missing.
    """
    if vecnorm_arg:
        return vecnorm_arg
    sibling = Path(model_path).with_name("vecnorm.pkl")
    if sibling.exists():
        print(f"Using {sibling} (pass --vecnorm to override)")
        return str(sibling)
    raise SystemExit(
        f"--load needs the matching VecNormalize stats, but no --vecnorm was "
        f"given and {sibling} does not exist. A policy run without them sees "
        f"un-normalized observations. Re-run with --vecnorm <path>."
    )


def make_staged_env(
    stage: str,
    seed: int = 42,
    max_steps: int = 2000,
    flip_penalty: float = 50.0,
    frame_skip: int = 1,
    power_capacity: Optional[float] = None,
    power_start_fraction: Optional[float] = None,
    cube_spawn_preset: Optional[str] = None,
    cube_spawn_lambda: Optional[float] = None,
    cube_spawn_extent: Optional[float] = None,
    cube_spawn_seed: Optional[int] = None,
    terrain_height_scale: Optional[float] = None,
    terrain_height_scale_range: Optional[tuple[float, float]] = None,
    scenario: Optional[str] = None,
    locomotion_shaping: str = "off",
    locomotion_coast_bonus: float = 0.35,
    locomotion_power_draw_penalty: float = 40.0,
    locomotion_power_recovery_reward: float = 20.0,
    locomotion_out_of_power_penalty: float = 75.0,
    cube_shaping: str = "off",
) -> gym.Env:
    """Construct a `RoverEnv` configured for `stage` and wrap its reward.

    Beacons are enabled only in the ``full`` stage; in ``locomotion`` and
    ``minerals`` action index 9 is an inert no-op so the action space
    stays `Discrete(10)` across every stage.

    `frame_skip` > 1 holds each action for that many physics ticks
    (`ActionRepeat`). The same value must be used at training, eval, and
    in-game-autopilot time so the policy acts at the cadence it learned.
    Wrapping order is RoverEnv → ActionRepeat → StagedRewardWrapper, so
    the staged reward's per-step delta naturally spans all skipped ticks.

    `power_capacity` (Wh) defaults to `DEFAULT_POWER_CAPACITY_WH` so the
    battery binds within an episode; pass a larger value (e.g. 1000 for
    the game's battery) to loosen it.

    `cube_spawn_lambda` / `cube_spawn_extent` default to the game's values
    for `locomotion` (unchanged behavior, so the committed `locomotion`
    model bundle stays a valid warm-start base) and to
    `DEFAULT_CUBE_SPAWN_LAMBDA` / `DEFAULT_CUBE_SPAWN_EXTENT` from
    `power_cubes` onward, matching how the `cube` reward weight is carried
    forward — pass explicit values to override either.
    """
    if stage not in STAGE_WEIGHTS:
        raise ValueError(f"unknown stage {stage!r}; choose from {STAGES}")
    if power_capacity is None:
        power_capacity = DEFAULT_POWER_CAPACITY_WH
    if power_start_fraction is None:
        power_start_fraction = 1.0
    if cube_spawn_preset is not None:
        if cube_spawn_preset not in CUBE_SPAWN_PRESETS:
            raise ValueError(
                f"unknown cube_spawn_preset {cube_spawn_preset!r}; "
                f"choose from {tuple(CUBE_SPAWN_PRESETS)}"
            )
        preset = CUBE_SPAWN_PRESETS[cube_spawn_preset]
        if cube_spawn_lambda is None:
            cube_spawn_lambda = preset["lambda"]
        if cube_spawn_extent is None:
            cube_spawn_extent = preset["extent"]
    elif stage != "locomotion":
        cube_spawn_preset = "dense_training"
        if cube_spawn_lambda is None:
            cube_spawn_lambda = DEFAULT_CUBE_SPAWN_LAMBDA
        if cube_spawn_extent is None:
            cube_spawn_extent = DEFAULT_CUBE_SPAWN_EXTENT
    else:
        cube_spawn_preset = "sparse_game"
    env: gym.Env = RoverEnv(
        seed=seed,
        max_steps=max_steps,
        beacons_enabled=(stage == "full"),
        power_capacity=power_capacity,
        power_start_fraction=power_start_fraction,
        cube_spawn_lambda=cube_spawn_lambda,
        cube_spawn_extent=cube_spawn_extent,
        cube_spawn_seed=cube_spawn_seed,
        terrain_height_scale=terrain_height_scale,
        terrain_height_scale_range=terrain_height_scale_range,
    )
    if frame_skip > 1:
        env = ActionRepeat(env, frame_skip)
    return StagedRewardWrapper(
        env,
        stage=stage,
        flip_penalty=flip_penalty,
        scenario=scenario,
        cube_spawn_preset=cube_spawn_preset,
        locomotion_shaping=locomotion_shaping,
        coast_distance_bonus=locomotion_coast_bonus,
        power_draw_penalty=locomotion_power_draw_penalty,
        power_recovery_reward=locomotion_power_recovery_reward,
        out_of_power_penalty=locomotion_out_of_power_penalty,
        cube_shaping=cube_shaping,
    )
