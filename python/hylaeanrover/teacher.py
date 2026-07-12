"""Small teacher policies for curriculum bootstrapping.

These are not runtime autopilots. They generate supervised labels for
isolated curriculum stages before PPO fine-tuning takes over.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CUBE_OBS_START = 15
CUBE_OBS_WIDTH = 3
MAX_VISIBLE_CUBES = 6
POWER_OBS_INDEX = CUBE_OBS_START + MAX_VISIBLE_CUBES * CUBE_OBS_WIDTH


@dataclass
class CubeInterceptTeacher:
    """Reactive teacher for the single-visible-cube intercept stage."""

    bearing_deadband_deg: float = 6.0
    pickup_hold_range_m: float = 3.0
    last_bearing_deg: float | None = None

    def reset(self) -> None:
        self.last_bearing_deg = None

    def action(self, obs: np.ndarray) -> int:
        cube = nearest_visible_cube(obs)
        if cube is None:
            if self.last_bearing_deg is None:
                return 4
            # Once the target drops out of the sensor cone, keep committing
            # forward. Continuing the last hard turn burns low-power episodes
            # and often curves away from the intercept.
            return 7

        bearing, distance = cube
        self.last_bearing_deg = bearing
        if distance <= self.pickup_hold_range_m:
            return 4
        if abs(bearing) <= self.bearing_deadband_deg:
            return 7
        return 8 if bearing < 0.0 else 6


@dataclass
class PowerIdleTeacher:
    """Teacher for low-power no-target discipline.

    The policy should coast when the actionable cube slots are empty and
    power is low. If a cube is visible, keep the previously learned
    intercept behavior in the supervised labels so this bootstrap does not
    deliberately erase cube seeking.
    """

    low_power_threshold: float = 0.45
    intercept_teacher: CubeInterceptTeacher | None = None

    def __post_init__(self) -> None:
        if self.intercept_teacher is None:
            self.intercept_teacher = CubeInterceptTeacher()

    def reset(self) -> None:
        assert self.intercept_teacher is not None
        self.intercept_teacher.reset()

    def action(self, obs: np.ndarray) -> int:
        assert self.intercept_teacher is not None
        if nearest_visible_cube(obs) is not None:
            return self.intercept_teacher.action(obs)
        power_frac = float(obs[POWER_OBS_INDEX])
        if power_frac <= self.low_power_threshold:
            return 4
        return 4


@dataclass
class MineralExploreTeacher:
    """Teacher for Stage 2 exploration before PPO fine-tuning.

    The mineral observation is local concentration under the rover, not a
    directional gradient. This teacher therefore does not pretend to know
    where deposits are. It provides the missing motor prior: cover ground
    while power is healthy, then recover when power gets low. During
    low-power recovery, visible cubes stay survival targets rather than
    becoming a paid objective.
    """

    low_power_threshold: float = 0.45
    resume_power_threshold: float = 0.65
    intercept_teacher: CubeInterceptTeacher | None = None
    recovering: bool = False

    def __post_init__(self) -> None:
        if self.intercept_teacher is None:
            self.intercept_teacher = CubeInterceptTeacher()

    def reset(self) -> None:
        assert self.intercept_teacher is not None
        self.intercept_teacher.reset()
        self.recovering = False

    def action(self, obs: np.ndarray) -> int:
        assert self.intercept_teacher is not None
        power_frac = float(obs[POWER_OBS_INDEX])
        if power_frac <= self.low_power_threshold:
            self.recovering = True
        elif power_frac >= self.resume_power_threshold:
            self.recovering = False
            self.intercept_teacher.reset()
        if self.recovering and nearest_visible_cube(obs) is not None:
            return self.intercept_teacher.action(obs)
        return 4 if self.recovering else 7


def nearest_visible_cube(obs: np.ndarray) -> tuple[float, float] | None:
    """Return ``(bearing_deg, range_m)`` for the nearest valid cube slot."""
    best: tuple[float, float] | None = None
    for i in range(MAX_VISIBLE_CUBES):
        j = CUBE_OBS_START + i * CUBE_OBS_WIDTH
        valid = float(obs[j + 2]) > 0.5
        if not valid:
            continue
        bearing = float(obs[j])
        distance = float(obs[j + 1])
        if best is None or distance < best[1]:
            best = (bearing, distance)
    return best


def cube_intercept_teacher_action(obs: np.ndarray) -> int:
    """Stateless convenience wrapper used by simple scripts/tests."""
    return CubeInterceptTeacher().action(obs)
