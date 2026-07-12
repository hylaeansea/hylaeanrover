"""Gymnasium-compatible wrapper around the Rust `RoverEnv` extension.

Usage:
    >>> from hylaeanrover import RoverEnv
    >>> env = RoverEnv(seed=42)
    >>> obs, info = env.reset()
    >>> obs, reward, terminated, truncated, info = env.step(action=0)

The heavy lifting (Bevy simulation, physics, sensors) lives in the
Rust extension module `hylaeanrover_py`. This thin wrapper just adds
the gymnasium contract on top: `observation_space`, `action_space`,
JSON-deserialized info dicts, and numpy arrays for SB3.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from hylaeanrover._native import RoverEnv as _RustRoverEnv
from hylaeanrover._native import MissionSupervisor as MissionSupervisorCore


class RoverEnv(gym.Env):
    """Gymnasium env. Discrete actions, fixed-length float32 observations.

    Action layout (10 discrete):
        Indices 0..8 are a 3x3 grid:
            throttle ∈ {−1, 0, +1}  (rows)
            steering ∈ {−1, 0, +1}  (cols)
            index = throttle_idx * 3 + steer_idx
            where each idx ∈ {0=neg, 1=zero, 2=pos}
        Index 9 is "drop beacon, no movement".

    Observation: see `hylaeanrover_py.RoverEnv.obs_dim()` for size.
    Layout matches the in-game JSON telemetry, flattened to float32.
    """

    metadata = {"render_modes": []}  # render_mode='human' is a follow-up

    def __init__(
        self,
        seed: int = 42,
        max_steps: int = 2000,
        render_mode: Optional[str] = None,
        beacons_enabled: bool = True,
        power_capacity: Optional[float] = None,
        power_start_fraction: float = 1.0,
        cube_spawn_lambda: Optional[float] = None,
        cube_spawn_extent: Optional[float] = None,
        cube_spawn_seed: Optional[int] = None,
        terrain_height_scale: Optional[float] = None,
        terrain_height_scale_range: Optional[tuple[float, float]] = None,
        forced_cube_distance: Optional[float] = None,
        forced_cube_distance_range: Optional[tuple[float, float]] = None,
        forced_cube_bearing_deg: Optional[float] = None,
        forced_cube_bearing_range: Optional[tuple[float, float]] = None,
    ) -> None:
        super().__init__()
        if render_mode is not None and render_mode != "rgb_array":
            # winit-on-Python's-main-thread is the blocker for 'human'.
            # 'rgb_array' is feasible later via an offscreen render
            # pass; not implemented yet.
            raise ValueError(
                f"render_mode={render_mode!r} not supported. "
                "Headless training is the only mode this branch ships."
            )
        self.render_mode = render_mode
        # When False, action index 9 (drop beacon) is an inert no-op and
        # the `beacons_deployed` game-over never fires — used by the RL
        # curriculum's locomotion / mineral stages so the action space
        # stays Discrete(10) across all stages (clean weight transfer).
        self.beacons_enabled = beacons_enabled
        # Battery capacity in Wh; None keeps the game's 1 kWh default.
        # Training passes a small value so the power budget binds within
        # one episode. The battery refills on every reset().
        self.power_capacity = power_capacity
        self.power_start_fraction = power_start_fraction
        # Power-cube Poisson spawn rate (cubes/sec) / spawn region
        # half-width (m); None keeps the game's defaults. The power_cubes
        # curriculum stage raises the rate and shrinks the region so a
        # short episode has enough reachable cubes to learn seek behavior.
        self.cube_spawn_lambda = cube_spawn_lambda
        self.cube_spawn_extent = cube_spawn_extent
        self.cube_spawn_seed = cube_spawn_seed
        self.terrain_height_scale = terrain_height_scale
        self.terrain_height_scale_range = terrain_height_scale_range
        self.forced_cube_distance = forced_cube_distance
        self.forced_cube_distance_range = forced_cube_distance_range
        self.forced_cube_bearing_deg = forced_cube_bearing_deg
        self.forced_cube_bearing_range = forced_cube_bearing_range

        self._env = _RustRoverEnv(
            seed=seed,
            max_steps=max_steps,
            beacons_enabled=beacons_enabled,
            power_capacity=power_capacity,
            power_start_fraction=power_start_fraction,
            cube_spawn_lambda=cube_spawn_lambda,
            cube_spawn_extent=cube_spawn_extent,
            cube_spawn_seed=cube_spawn_seed,
            terrain_height_scale=terrain_height_scale,
        )
        self._obs_dim = _RustRoverEnv.obs_dim()
        self._action_count = _RustRoverEnv.action_count()

        # No tight bounds on the observations — angles, distances,
        # rewards span wide ranges. SB3 will normalize via VecNormalize
        # if needed. -inf/+inf is the standard "any float" idiom.
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self._action_count)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        options = options or {}
        # Seed the gym RNG (deterministic iff `seed` is given), then draw a
        # fresh terrain seed from it for *every* episode. SB3's autoreset
        # calls reset() with seed=None; without this, np_random would not
        # advance and every episode would reuse the construction-time
        # terrain — the policy would overfit one map and any baseline-vs-
        # trained comparison would be on a single terrain. Drawing from the
        # RNG keeps it reproducible when a seed is supplied and varied (true
        # domain randomization) when it isn't.
        super().reset(seed=seed)
        terrain_seed = int(self.np_random.integers(0, 2**31 - 1))
        terrain_height_scale = options.get("terrain_height_scale")
        if terrain_height_scale is None and self.terrain_height_scale_range is not None:
            lo, hi = self.terrain_height_scale_range
            terrain_height_scale = float(self.np_random.uniform(lo, hi))
        if terrain_height_scale is None:
            terrain_height_scale = self.terrain_height_scale
        power_start_fraction = options.get(
            "power_start_fraction", self.power_start_fraction
        )
        forced_cube_distance = options.get(
            "forced_cube_distance", self.forced_cube_distance
        )
        if forced_cube_distance is None and self.forced_cube_distance_range is not None:
            lo, hi = self.forced_cube_distance_range
            forced_cube_distance = float(self.np_random.uniform(lo, hi))
        forced_cube_bearing_deg = options.get(
            "forced_cube_bearing_deg", self.forced_cube_bearing_deg
        )
        if (
            forced_cube_bearing_deg is None
            and self.forced_cube_bearing_range is not None
        ):
            lo, hi = self.forced_cube_bearing_range
            forced_cube_bearing_deg = float(self.np_random.uniform(lo, hi))
        obs_list, info_json = self._env.reset(
            terrain_seed,
            terrain_height_scale,
            power_start_fraction,
            forced_cube_distance,
            forced_cube_bearing_deg,
        )
        obs = np.asarray(obs_list, dtype=np.float32)
        info = json.loads(info_json) if info_json else {}
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        # SB3 + gymnasium hand us numpy scalars sometimes — coerce.
        action_int = int(action)
        obs_list, reward, terminated, truncated, info_json = self._env.step(action_int)
        obs = np.asarray(obs_list, dtype=np.float32)
        info = json.loads(info_json) if info_json else {}
        return obs, float(reward), bool(terminated), bool(truncated), info

    def close(self) -> None:
        # Rust env cleans up on drop; nothing to do here.
        pass


# Useful for downstream tooling that wants to introspect schema without
# instantiating the env.
OBS_DIM: int = _RustRoverEnv.obs_dim()
ACTION_COUNT: int = _RustRoverEnv.action_count()
FIXED_DT: float = _RustRoverEnv.fixed_dt()
BEACON_BUDGET: int = _RustRoverEnv.beacon_budget()

__all__ = [
    "RoverEnv",
    "OBS_DIM",
    "ACTION_COUNT",
    "FIXED_DT",
    "BEACON_BUDGET",
    "MissionSupervisorCore",
]
