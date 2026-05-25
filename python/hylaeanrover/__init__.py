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

        self._env = _RustRoverEnv(seed=seed, max_steps=max_steps)
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
        del options  # not used (kept for the gymnasium signature)
        obs_list, info_json = self._env.reset(seed)
        obs = np.asarray(obs_list, dtype=np.float32)
        info = json.loads(info_json) if info_json else {}
        return obs, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        # SB3 + gymnasium hand us numpy scalars sometimes — coerce.
        action_int = int(action)
        obs_list, reward, terminated, truncated, info_json = self._env.step(
            action_int
        )
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
]
