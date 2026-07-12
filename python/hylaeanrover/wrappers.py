"""Staged reward shaping for the RL curriculum.

The training plan (see `docs/rl_training_plan.md`) trains the rover in
stages that share one fixed observation/action space and differ *only*
in the reward, so each stage's policy weights initialize the next:

    locomotion  →  cube_intercept  →  power_idle  →  power_cubes  →  minerals  →  full

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

from hylaeanrover import MissionSupervisorCore, RoverEnv

# Per-stage weights on the four reward components. The component values
# themselves come from the Rust reward (distance in meters, the flat
# per-pickup cube bonus, the scarcity-weighted mineral line-integral, and
# the 50x beacon bonus). Once a component's weight turns on for a stage it
# can be retuned per stage. Stage 1 keeps distance deliberately weaker
# than cube pickup so broad power-cube training cannot pass by merely
# driving far while ignoring visible cubes.
STAGE_WEIGHTS: dict[str, dict[str, float]] = {
    # Drive far, stay upright, manage power. Densest signal.
    "locomotion": {"distance": 1.0, "cube": 0.0, "mineral": 0.0, "beacon": 0.0},
    # Isolated visible-cube intercept. No distance reward: pickup is the task.
    "cube_intercept": {
        "distance": 0.0,
        "cube": 1.0,
        "mineral": 0.0,
        "beacon": 0.0,
    },
    # Low-power no-target discipline. This teaches the policy not to burn a
    # nearly empty battery when the actionable cube sensor is empty.
    "power_idle": {"distance": 0.0, "cube": 0.0, "mineral": 0.0, "beacon": 0.0},
    # Also reward actively seeking out and collecting power cubes. Distance is
    # off in this stage so the policy cannot pass by driving until the battery
    # dies; later stages reintroduce travel objectives after pickup behavior
    # is reliable.
    "power_cubes": {"distance": 0.0, "cube": 1.0, "mineral": 0.0, "beacon": 0.0},
    # Also reward crossing scarce-mineral ground. Power cubes are survival
    # tools in Stage 2, not the objective: the policy should pick them up
    # because power enables more distance/mineral reward, not because the
    # pickup itself is paid like it was in Stage 1.
    "minerals": {"distance": 1.0, "cube": 0.0, "mineral": 1.0, "beacon": 0.0},
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
    # Intermediate curriculum step after dense low-power training. Keeps
    # pickups frequent enough for PPO while widening the search area before
    # the much sparser transition setting.
    "bridge_training": {"lambda": 0.75, "extent": 75.0},
    # Near-sparse bridge between training density and the game's cadence.
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
    "bridge_training": {"cube_spawn_preset": "bridge_training"},
    "bridge_low_power": {
        "cube_spawn_preset": "bridge_training",
        "power_start_fraction": 0.35,
        "cube_shaping": "off",
    },
    "transition": {"cube_spawn_preset": "transition"},
    "sparse_game": {"cube_spawn_preset": "sparse_game"},
    "sparse_low_power": {
        "cube_spawn_preset": "sparse_game",
        "power_start_fraction": 0.35,
        "cube_shaping": "off",
    },
    "sparse_visible_reset": {
        "cube_spawn_preset": "sparse_game",
        "forced_cube_distance_range": (30.0, 60.0),
        "forced_cube_bearing_range": (-35.0, 35.0),
    },
    "sparse_visible_low_power": {
        "cube_spawn_preset": "sparse_game",
        "power_start_fraction": 0.35,
        "forced_cube_distance_range": (30.0, 60.0),
        "forced_cube_bearing_range": (-35.0, 35.0),
        "cube_shaping": "off",
    },
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
    "power_idle": {
        "cube_spawn_preset": "none",
        "power_start_fraction": 0.35,
        "cube_shaping": "off",
    },
    "cube_intercept": {
        "cube_spawn_preset": "none",
        "forced_cube_distance_range": (15.0, 60.0),
        "forced_cube_bearing_range": (-35.0, 35.0),
        "cube_shaping": "intercept",
    },
    "cube_intercept_close": {
        "cube_spawn_preset": "none",
        "forced_cube_distance_range": (15.0, 20.0),
        "forced_cube_bearing_range": (0.0, 5.0),
        "cube_shaping": "intercept",
    },
    "cube_intercept_low_power": {
        "cube_spawn_preset": "none",
        "power_start_fraction": 0.35,
        "forced_cube_distance_range": (15.0, 60.0),
        "forced_cube_bearing_range": (-35.0, 35.0),
        "cube_shaping": "intercept",
    },
    "minerals_explore": {
        "cube_spawn_preset": "none",
        "terrain_height": "mixed_1_2",
        "cube_shaping": "off",
    },
    "minerals_sparse": {
        "cube_spawn_preset": "sparse_game",
        "terrain_height": "mixed_1_2",
        "cube_shaping": "low_power",
    },
    "minerals_transition": {
        "cube_spawn_preset": "transition",
        "terrain_height": "mixed_1_2",
        "cube_shaping": "low_power",
    },
    "minerals_fixed_2_sparse": {
        "cube_spawn_preset": "sparse_game",
        "terrain_height": "fixed_2_0",
        "cube_shaping": "low_power",
    },
    "terrain_fixed_1_0": {"terrain_height": "fixed_1_0"},
    "terrain_fixed_1_5": {"terrain_height": "fixed_1_5"},
    "terrain_fixed_2_0": {"terrain_height": "fixed_2_0"},
    "terrain_mixed_1_2": {"terrain_height": "mixed_1_2"},
}

SCENARIOS = tuple(EVAL_SCENARIOS.keys())
CUBE_SHAPING_MODES = ("off", "low_power", "intercept")
LOCOMOTION_SHAPING_MODES = ("off", "power_efficiency")


def apply_scenario_defaults(
    scenario: Optional[str],
    *,
    cube_spawn_preset: Optional[str] = None,
    power_start_fraction: Optional[float] = None,
    terrain_height: Optional[str] = None,
    cube_shaping: Optional[str] = None,
    forced_cube_distance: Optional[float] = None,
    forced_cube_bearing_deg: Optional[float] = None,
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
    if forced_cube_distance is not None:
        cfg["forced_cube_distance"] = forced_cube_distance
        cfg.pop("forced_cube_distance_range", None)
    if forced_cube_bearing_deg is not None:
        cfg["forced_cube_bearing_deg"] = forced_cube_bearing_deg
        cfg.pop("forced_cube_bearing_range", None)
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


class MissionSupervisorWrapper(gym.Wrapper):
    """Apply the shared Rust mission supervisor once per policy decision."""

    def __init__(
        self,
        env: gym.Env,
        *,
        power_capacity_wh: float,
        low_power_enter_fraction: float = 0.35,
        low_power_exit_fraction: float = 0.50,
        path_safety_factor: float = 1.10,
        reserve_distance_m: float = 2.0,
        tilt_enter_deg: float = 20.0,
        tilt_exit_deg: float = 18.0,
        tilt_guard_min_speed_mps: float = 1.0,
        target_loss_grace_decisions: int = 0,
        beacon_guard_enabled: bool = False,
        beacon_first_distance_m: float = 100.0,
        beacon_spacing_m: float = 75.0,
        beacon_auto_deploy: bool = True,
        beacon_surface_score_threshold: float = 150.0,
    ) -> None:
        super().__init__(env)
        self.power_capacity_wh = float(power_capacity_wh)
        self._supervisor = MissionSupervisorCore(
            low_power_enter_fraction=low_power_enter_fraction,
            low_power_exit_fraction=low_power_exit_fraction,
            path_safety_factor=path_safety_factor,
            reserve_distance_m=reserve_distance_m,
            tilt_enter_deg=tilt_enter_deg,
            tilt_exit_deg=tilt_exit_deg,
            tilt_guard_min_speed_mps=tilt_guard_min_speed_mps,
            target_loss_grace_decisions=target_loss_grace_decisions,
            beacon_guard_enabled=beacon_guard_enabled,
            beacon_first_distance_m=beacon_first_distance_m,
            beacon_spacing_m=beacon_spacing_m,
            beacon_auto_deploy=beacon_auto_deploy,
            beacon_surface_score_threshold=beacon_surface_score_threshold,
        )
        self._last_obs: Any = None
        self._last_distance_m = 0.0
        self._decisions = 0
        self._overrides = 0
        self._mode_counts: dict[str, int] = {}

    def _annotate(
        self,
        info: dict[str, Any],
        *,
        proposed_action: int = 4,
        selected_action: int = 4,
        mode: str = "explore",
        overrode: bool = False,
        target_visible: bool = False,
        target_viable: bool = False,
        target_range_m: Optional[float] = None,
        available_range_m: float = 0.0,
    ) -> None:
        info["mission_supervisor"] = True
        info["supervisor_policy_action"] = proposed_action
        info["supervisor_action"] = selected_action
        info["supervisor_mode"] = mode
        info["supervisor_overrode"] = overrode
        info["supervisor_target_visible"] = target_visible
        info["supervisor_target_viable"] = target_viable
        info["supervisor_target_range_m"] = target_range_m
        info["supervisor_available_range_m"] = available_range_m
        info["supervisor_decisions"] = self._decisions
        info["supervisor_overrides"] = self._overrides
        info["supervisor_override_rate"] = (
            self._overrides / self._decisions if self._decisions else 0.0
        )
        for name in (
            "explore",
            "intercept",
            "commit",
            "preserve",
            "stabilize",
            "beacon_deploy",
            "beacon_hold",
        ):
            info[f"supervisor_{name}_steps"] = self._mode_counts.get(name, 0)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[Any, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        self._supervisor.reset()
        self._last_obs = obs
        self._last_distance_m = float(info.get("reward_distance", 0.0))
        self._decisions = 0
        self._overrides = 0
        self._mode_counts = {}
        self._annotate(info)
        return obs, info

    def step(self, action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        if self._last_obs is None:
            raise RuntimeError("MissionSupervisorWrapper.step() called before reset()")
        proposed_action = int(action)
        decision = self._supervisor.decide(
            [float(value) for value in self._last_obs],
            proposed_action,
            self.power_capacity_wh,
            self._last_distance_m,
        )
        (
            selected_action,
            mode,
            overrode,
            target_visible,
            target_viable,
            target_range_m,
            available_range_m,
        ) = decision
        obs, reward, terminated, truncated, info = self.env.step(selected_action)
        self._last_obs = obs
        self._last_distance_m = float(
            info.get("reward_distance", self._last_distance_m)
        )
        self._decisions += 1
        self._overrides += int(overrode)
        self._mode_counts[mode] = self._mode_counts.get(mode, 0) + 1
        self._annotate(
            info,
            proposed_action=proposed_action,
            selected_action=selected_action,
            mode=mode,
            overrode=overrode,
            target_visible=target_visible,
            target_viable=target_viable,
            target_range_m=target_range_m,
            available_range_m=available_range_m,
        )
        return obs, reward, terminated, truncated, info


class StagedRewardWrapper(gym.Wrapper):
    """Recompute reward from cumulative info components, weighted by stage.

    Adds an optional one-off ``flip_penalty`` when the episode terminates
    with the rover flipped, plus optional dense excessive-tilt shaping so
    the policy receives a warning signal before a rollover becomes terminal.
    """

    def __init__(
        self,
        env: gym.Env,
        stage: str,
        flip_penalty: float = 50.0,
        tilt_penalty: float = 0.0,
        tilt_threshold_deg: float = 45.0,
        scenario: Optional[str] = None,
        cube_spawn_preset: Optional[str] = None,
        locomotion_shaping: str = "off",
        cube_shaping: str = "off",
        low_power_threshold: float = 0.45,
        cube_approach_reward: float = 0.25,
        cube_heading_reward: float = 0.05,
        ignored_cube_penalty: float = 25.0,
        loss_of_sight_penalty: float = 5.0,
        intercept_failure_penalty: float = 50.0,
        coast_distance_bonus: float = 0.35,
        power_draw_penalty: float = 40.0,
        power_recovery_reward: float = 20.0,
        out_of_power_penalty: float = 75.0,
        low_power_no_target_throttle_penalty: float = 0.25,
        low_power_no_target_coast_reward: float = 0.02,
        low_power_visible_stall_throttle_penalty: float = 0.0,
        cube_progress_range_epsilon: float = 0.1,
        cube_progress_bearing_epsilon_deg: float = 1.0,
        rejected_beacon_penalty: float = 5.0,
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
        self.tilt_penalty = tilt_penalty
        self.tilt_threshold_deg = tilt_threshold_deg
        self.scenario = scenario
        self.cube_spawn_preset = cube_spawn_preset
        self.locomotion_shaping = locomotion_shaping
        self.cube_shaping = cube_shaping
        self.low_power_threshold = low_power_threshold
        self.cube_approach_reward = cube_approach_reward
        self.cube_heading_reward = cube_heading_reward
        self.ignored_cube_penalty = ignored_cube_penalty
        self.loss_of_sight_penalty = loss_of_sight_penalty
        self.intercept_failure_penalty = intercept_failure_penalty
        self.coast_distance_bonus = coast_distance_bonus
        self.power_draw_penalty = power_draw_penalty
        self.power_recovery_reward = power_recovery_reward
        self.out_of_power_penalty = out_of_power_penalty
        self.low_power_no_target_throttle_penalty = low_power_no_target_throttle_penalty
        self.low_power_no_target_coast_reward = low_power_no_target_coast_reward
        self.low_power_visible_stall_throttle_penalty = (
            low_power_visible_stall_throttle_penalty
        )
        self.cube_progress_range_epsilon = cube_progress_range_epsilon
        self.cube_progress_bearing_epsilon_deg = cube_progress_bearing_epsilon_deg
        self.rejected_beacon_penalty = rejected_beacon_penalty
        # Cumulative component totals from the previous step, so we can
        # take deltas. Seeded in reset().
        self._prev = {"distance": 0.0, "cube": 0.0, "mineral": 0.0, "beacon": 0.0}
        self._prev_power_frac = 1.0
        self._prev_visible_range: Optional[float] = None
        self._low_power_visible_steps = 0
        self._low_power_no_target_steps = 0
        self._low_power_visible_stall_steps = 0
        self._low_power_visible_stall_penalty_total = 0.0
        self._prev_guard_visible_range: Optional[float] = None
        self._prev_guard_visible_bearing: Optional[float] = None
        self._tilt_penalty_total = 0.0
        self._rejected_beacon_penalty_total = 0.0

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

    @staticmethod
    def _is_motor_action(action: int) -> bool:
        return int(action) in (0, 1, 2, 6, 7, 8)

    @staticmethod
    def _has_visible_cube(info: dict[str, Any]) -> bool:
        return int(info.get("visible_cube_count", 0) or 0) > 0

    def _locomotion_power_shaping(
        self,
        action: int,
        info: dict[str, Any],
        terminated: bool,
        distance_delta: float,
    ) -> float:
        if (
            self.stage
            not in (
                "locomotion",
                "power_idle",
                "power_cubes",
                "minerals",
            )
            or self.locomotion_shaping == "off"
        ):
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

    def _low_power_no_target_shaping(
        self,
        action: int,
        info: dict[str, Any],
    ) -> float:
        if (
            self.stage not in ("power_idle", "power_cubes", "minerals")
            or self.locomotion_shaping == "off"
        ):
            return 0.0

        power_frac = float(info.get("power_frac", self._prev_power_frac))
        if power_frac > self.low_power_threshold or self._has_visible_cube(info):
            return 0.0

        self._low_power_no_target_steps += 1
        if self._is_motor_action(action):
            return -self.low_power_no_target_throttle_penalty
        return self.low_power_no_target_coast_reward

    def _low_power_visible_progress_shaping(
        self,
        action: int,
        info: dict[str, Any],
    ) -> float:
        """Discourage draining power while a visible cube is not improving."""
        if (
            self.stage not in ("power_cubes", "minerals")
            or self.locomotion_shaping == "off"
            or self.low_power_visible_stall_throttle_penalty <= 0.0
        ):
            return 0.0

        power_frac = float(info.get("power_frac", self._prev_power_frac))
        nearest_raw = info.get("nearest_visible_cube_range")
        bearing_raw = info.get("nearest_visible_cube_bearing")
        if (
            power_frac > self.low_power_threshold
            or not self._has_visible_cube(info)
            or nearest_raw is None
            or bearing_raw is None
        ):
            self._prev_guard_visible_range = None
            self._prev_guard_visible_bearing = None
            return 0.0

        nearest = float(nearest_raw)
        bearing = abs(float(bearing_raw))
        prev_range = self._prev_guard_visible_range
        prev_bearing = self._prev_guard_visible_bearing
        self._prev_guard_visible_range = nearest
        self._prev_guard_visible_bearing = bearing

        # The first visible decision establishes a baseline. Thereafter a
        # motor action is allowed if it either closes range or turns the cube
        # closer to boresight; this leaves deliberate intercept turns intact.
        if (
            prev_range is None
            or prev_bearing is None
            or not self._is_motor_action(action)
        ):
            return 0.0
        range_improved = nearest <= prev_range - self.cube_progress_range_epsilon
        bearing_improved = (
            bearing <= prev_bearing - self.cube_progress_bearing_epsilon_deg
        )
        if range_improved or bearing_improved:
            return 0.0

        self._low_power_visible_stall_steps += 1
        penalty = self.low_power_visible_stall_throttle_penalty
        self._low_power_visible_stall_penalty_total += penalty
        return -penalty

    def _tilt_shaping(self, obs: Any) -> float:
        """Penalize only large pitch/roll excursions before they become flips."""
        if self.tilt_penalty <= 0.0:
            return 0.0

        # Observation layout starts with speed, heading, pitch, roll.
        max_tilt = max(abs(float(obs[2])), abs(float(obs[3])))
        excess = max(0.0, max_tilt - self.tilt_threshold_deg)
        if excess <= 0.0:
            return 0.0

        # The terminal flip threshold is 100 degrees. Squaring the normalized
        # excess leaves ordinary rough-terrain motion alone while making the
        # signal rise sharply as a rollover develops.
        span = max(1.0, 100.0 - self.tilt_threshold_deg)
        normalized_excess = min(1.0, excess / span)
        penalty = self.tilt_penalty * normalized_excess**2
        self._tilt_penalty_total += penalty
        return -penalty

    def _cube_approach_shaping(
        self,
        info: dict[str, Any],
        terminated: bool,
        truncated: bool,
        cube_delta: float,
    ) -> float:
        if (
            self.stage not in ("cube_intercept", "power_cubes", "minerals")
            or self.cube_shaping == "off"
        ):
            return 0.0
        power_frac = float(info.get("power_frac", 1.0))
        visible = self._has_visible_cube(info)
        nearest_raw = info.get("nearest_visible_cube_range")
        nearest = float(nearest_raw) if nearest_raw is not None else None
        active = (
            visible
            and nearest is not None
            and (
                self.cube_shaping == "intercept"
                or power_frac <= self.low_power_threshold
            )
        )
        if not active:
            shaped = 0.0
            if (
                self.stage == "cube_intercept"
                and self._prev_visible_range is not None
                and cube_delta <= 0.0
                and not visible
            ):
                shaped -= self.loss_of_sight_penalty
            if (
                self.stage == "cube_intercept"
                and (terminated or truncated)
                and cube_delta <= 0.0
            ):
                shaped -= self.intercept_failure_penalty
            self._prev_visible_range = nearest if visible else None
            return shaped

        self._low_power_visible_steps += 1
        shaped = 0.0
        if self._prev_visible_range is not None:
            # Positive when the policy moves toward the visible cube,
            # negative when it moves away while power is binding.
            shaped += (self._prev_visible_range - nearest) * self.cube_approach_reward
        self._prev_visible_range = nearest

        if self.cube_shaping == "intercept":
            bearing_raw = info.get("nearest_visible_cube_bearing")
            if bearing_raw is not None:
                bearing = abs(float(bearing_raw))
                # Sensor visibility is already cone-limited; this simply
                # rewards keeping the nearest visible cube near boresight.
                alignment = max(0.0, 1.0 - min(bearing, 60.0) / 60.0)
                shaped += alignment * self.cube_heading_reward

        if terminated and info.get("game_over") == "out_of_power":
            shaped -= self.ignored_cube_penalty
        if (
            self.stage == "cube_intercept"
            and (terminated or truncated)
            and cube_delta <= 0.0
        ):
            shaped -= self.intercept_failure_penalty
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
        self._low_power_no_target_steps = 0
        self._low_power_visible_stall_steps = 0
        self._low_power_visible_stall_penalty_total = 0.0
        self._prev_guard_visible_range = None
        self._prev_guard_visible_bearing = None
        self._tilt_penalty_total = 0.0
        self._rejected_beacon_penalty_total = 0.0
        self._annotate_info(info)
        info["tilt_penalty_total"] = self._tilt_penalty_total
        info["rejected_beacon_penalty_total"] = self._rejected_beacon_penalty_total
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
        shaped += self._low_power_no_target_shaping(int(action), info)
        shaped += self._low_power_visible_progress_shaping(int(action), info)
        shaped += self._tilt_shaping(obs)
        shaped += self._cube_approach_shaping(
            info, terminated, truncated, deltas["cube"]
        )
        if info.get("supervisor_mode") == "beacon_hold":
            shaped -= self.rejected_beacon_penalty
            self._rejected_beacon_penalty_total += self.rejected_beacon_penalty
        self._prev = cur
        self._prev_power_frac = float(info.get("power_frac", self._prev_power_frac))

        if terminated and info.get("game_over") == "flipped":
            shaped -= self.flip_penalty

        if (
            self.stage == "cube_intercept"
            and float(info.get("episode_cube_pickups", 0.0)) > 0.0
        ):
            terminated = True
            info["game_over"] = "cube_picked_up"

        self._annotate_info(info)
        info["low_power_visible_steps"] = self._low_power_visible_steps
        info["low_power_no_target_steps"] = self._low_power_no_target_steps
        info["low_power_visible_stall_steps"] = self._low_power_visible_stall_steps
        info["low_power_visible_stall_penalty_total"] = (
            self._low_power_visible_stall_penalty_total
        )
        info["tilt_penalty_total"] = self._tilt_penalty_total
        info["rejected_beacon_penalty_total"] = self._rejected_beacon_penalty_total
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
    tilt_penalty: float = 0.0,
    tilt_threshold_deg: float = 45.0,
    mission_supervisor: bool = False,
    supervisor_low_power_enter_fraction: float = 0.35,
    supervisor_low_power_exit_fraction: float = 0.50,
    supervisor_path_safety_factor: float = 1.10,
    supervisor_reserve_distance_m: float = 2.0,
    supervisor_tilt_enter_deg: float = 20.0,
    supervisor_tilt_exit_deg: float = 18.0,
    supervisor_tilt_guard_min_speed_mps: float = 1.0,
    supervisor_target_loss_grace_decisions: int = 0,
    supervisor_beacon_first_distance_m: float = 100.0,
    supervisor_beacon_spacing_m: float = 75.0,
    supervisor_beacon_auto_deploy: bool = True,
    supervisor_beacon_surface_score_threshold: float = 150.0,
    frame_skip: int = 1,
    power_capacity: Optional[float] = None,
    power_start_fraction: Optional[float] = None,
    cube_spawn_preset: Optional[str] = None,
    cube_spawn_lambda: Optional[float] = None,
    cube_spawn_extent: Optional[float] = None,
    cube_spawn_seed: Optional[int] = None,
    terrain_height_scale: Optional[float] = None,
    terrain_height_scale_range: Optional[tuple[float, float]] = None,
    forced_cube_distance: Optional[float] = None,
    forced_cube_distance_range: Optional[tuple[float, float]] = None,
    forced_cube_bearing_deg: Optional[float] = None,
    forced_cube_bearing_range: Optional[tuple[float, float]] = None,
    scenario: Optional[str] = None,
    locomotion_shaping: str = "off",
    locomotion_coast_bonus: float = 0.35,
    locomotion_power_draw_penalty: float = 40.0,
    locomotion_power_recovery_reward: float = 20.0,
    locomotion_out_of_power_penalty: float = 75.0,
    low_power_no_target_throttle_penalty: float = 0.25,
    low_power_no_target_coast_reward: float = 0.02,
    low_power_visible_stall_throttle_penalty: float = 0.0,
    cube_progress_range_epsilon: float = 0.1,
    cube_progress_bearing_epsilon_deg: float = 1.0,
    cube_shaping: Optional[str] = None,
    low_power_threshold: float = 0.45,
    cube_approach_reward: float = 0.25,
    cube_heading_reward: float = 0.05,
    ignored_cube_penalty: float = 25.0,
    loss_of_sight_penalty: float = 5.0,
    intercept_failure_penalty: float = 50.0,
    rejected_beacon_penalty: float = 5.0,
) -> gym.Env:
    """Construct a `RoverEnv` configured for `stage` and wrap its reward.

    Beacons are enabled only in the ``full`` stage; in ``locomotion`` and
    ``minerals`` action index 9 is an inert no-op so the action space
    stays `Discrete(10)` across every stage.

    `frame_skip` > 1 holds each action for that many physics ticks
    (`ActionRepeat`). The same value must be used at training, eval, and
    in-game-autopilot time so the policy acts at the cadence it learned.
    Wrapping order is RoverEnv → ActionRepeat → optional MissionSupervisor
    → StagedRewardWrapper. The supervisor therefore runs once per policy
    decision and its selected action is held across all skipped ticks.

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
        power_start_fraction = 0.35 if stage == "power_idle" else 1.0
    if cube_shaping is None:
        cube_shaping = "intercept" if stage == "cube_intercept" else "off"
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
    elif stage in ("cube_intercept", "power_idle"):
        cube_spawn_preset = "none"
        cube_spawn_lambda = 0.0 if cube_spawn_lambda is None else cube_spawn_lambda
        if cube_spawn_extent is None:
            cube_spawn_extent = DEFAULT_CUBE_SPAWN_EXTENT
    elif stage != "locomotion":
        cube_spawn_preset = "dense_training"
        if cube_spawn_lambda is None:
            cube_spawn_lambda = DEFAULT_CUBE_SPAWN_LAMBDA
        if cube_spawn_extent is None:
            cube_spawn_extent = DEFAULT_CUBE_SPAWN_EXTENT
    else:
        cube_spawn_preset = "sparse_game"
    if stage == "cube_intercept" and scenario != "no_cube_control":
        if forced_cube_distance is None and forced_cube_distance_range is None:
            forced_cube_distance_range = (15.0, 60.0)
        if forced_cube_bearing_deg is None and forced_cube_bearing_range is None:
            forced_cube_bearing_range = (-35.0, 35.0)
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
        forced_cube_distance=forced_cube_distance,
        forced_cube_distance_range=forced_cube_distance_range,
        forced_cube_bearing_deg=forced_cube_bearing_deg,
        forced_cube_bearing_range=forced_cube_bearing_range,
    )
    if frame_skip > 1:
        env = ActionRepeat(env, frame_skip)
    if mission_supervisor:
        env = MissionSupervisorWrapper(
            env,
            power_capacity_wh=power_capacity,
            low_power_enter_fraction=supervisor_low_power_enter_fraction,
            low_power_exit_fraction=supervisor_low_power_exit_fraction,
            path_safety_factor=supervisor_path_safety_factor,
            reserve_distance_m=supervisor_reserve_distance_m,
            tilt_enter_deg=supervisor_tilt_enter_deg,
            tilt_exit_deg=supervisor_tilt_exit_deg,
            tilt_guard_min_speed_mps=supervisor_tilt_guard_min_speed_mps,
            target_loss_grace_decisions=supervisor_target_loss_grace_decisions,
            beacon_guard_enabled=(stage == "full"),
            beacon_first_distance_m=supervisor_beacon_first_distance_m,
            beacon_spacing_m=supervisor_beacon_spacing_m,
            beacon_auto_deploy=supervisor_beacon_auto_deploy,
            beacon_surface_score_threshold=(supervisor_beacon_surface_score_threshold),
        )
    return StagedRewardWrapper(
        env,
        stage=stage,
        flip_penalty=flip_penalty,
        tilt_penalty=tilt_penalty,
        tilt_threshold_deg=tilt_threshold_deg,
        scenario=scenario,
        cube_spawn_preset=cube_spawn_preset,
        locomotion_shaping=locomotion_shaping,
        coast_distance_bonus=locomotion_coast_bonus,
        power_draw_penalty=locomotion_power_draw_penalty,
        power_recovery_reward=locomotion_power_recovery_reward,
        out_of_power_penalty=locomotion_out_of_power_penalty,
        low_power_no_target_throttle_penalty=low_power_no_target_throttle_penalty,
        low_power_no_target_coast_reward=low_power_no_target_coast_reward,
        low_power_visible_stall_throttle_penalty=(
            low_power_visible_stall_throttle_penalty
        ),
        cube_progress_range_epsilon=cube_progress_range_epsilon,
        cube_progress_bearing_epsilon_deg=cube_progress_bearing_epsilon_deg,
        cube_shaping=cube_shaping,
        low_power_threshold=low_power_threshold,
        cube_approach_reward=cube_approach_reward,
        cube_heading_reward=cube_heading_reward,
        ignored_cube_penalty=ignored_cube_penalty,
        loss_of_sight_penalty=loss_of_sight_penalty,
        intercept_failure_penalty=intercept_failure_penalty,
        rejected_beacon_penalty=rejected_beacon_penalty,
    )
