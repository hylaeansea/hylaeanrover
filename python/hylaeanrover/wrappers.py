"""Staged reward shaping for the RL curriculum.

The training plan (see `docs/rl_training_plan.md`) trains the rover in
stages that share one fixed observation/action space and differ *only*
in the reward, so each stage's policy weights initialize the next:

    locomotion  →  minerals  →  full

`StagedRewardWrapper` recomputes the per-step reward from the cumulative
reward components the Rust env already exposes in its `info` dict
(`reward_distance`, `reward_mineral_integral`, `reward_beacon_bonus`),
weighting them per stage. Keeping the shaping in Python means we can
retune it without rebuilding the Rust extension.

`make_staged_env` is the convenience constructor: it picks the right
`beacons_enabled` for the stage (off for locomotion/minerals so action
index 9 is an inert no-op) and wraps the env.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import gymnasium as gym

from hylaeanrover import RoverEnv

# Per-stage weights on the three reward components. The component values
# themselves come from the Rust reward (distance in meters, the
# scarcity-weighted mineral line-integral, and the 50x beacon bonus).
STAGE_WEIGHTS: dict[str, dict[str, float]] = {
    # Drive far, stay upright, manage power. Densest signal.
    "locomotion": {"distance": 1.0, "mineral": 0.0, "beacon": 0.0},
    # Also reward crossing scarce-mineral ground.
    "minerals": {"distance": 1.0, "mineral": 1.0, "beacon": 0.0},
    # Full mission, including strategic beacon placement.
    "full": {"distance": 1.0, "mineral": 1.0, "beacon": 1.0},
}

STAGES = tuple(STAGE_WEIGHTS.keys())


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
    ) -> None:
        if stage not in STAGE_WEIGHTS:
            raise ValueError(f"unknown stage {stage!r}; choose from {STAGES}")
        super().__init__(env)
        self.stage = stage
        self._w = STAGE_WEIGHTS[stage]
        self.flip_penalty = flip_penalty
        # Cumulative component totals from the previous step, so we can
        # take deltas. Seeded in reset().
        self._prev = {"distance": 0.0, "mineral": 0.0, "beacon": 0.0}

    @staticmethod
    def _components(info: dict[str, Any]) -> dict[str, float]:
        return {
            "distance": float(info.get("reward_distance", 0.0)),
            "mineral": float(info.get("reward_mineral_integral", 0.0)),
            "beacon": float(info.get("reward_beacon_bonus", 0.0)),
        }

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[Any, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        self._prev = self._components(info)
        return obs, info

    def step(self, action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        obs, _reward, terminated, truncated, info = self.env.step(action)

        cur = self._components(info)
        shaped = (
            self._w["distance"] * (cur["distance"] - self._prev["distance"])
            + self._w["mineral"] * (cur["mineral"] - self._prev["mineral"])
            + self._w["beacon"] * (cur["beacon"] - self._prev["beacon"])
        )
        self._prev = cur

        if terminated and info.get("game_over") == "flipped":
            shaped -= self.flip_penalty

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
    """
    if stage not in STAGE_WEIGHTS:
        raise ValueError(f"unknown stage {stage!r}; choose from {STAGES}")
    env: gym.Env = RoverEnv(
        seed=seed,
        max_steps=max_steps,
        beacons_enabled=(stage == "full"),
    )
    if frame_skip > 1:
        env = ActionRepeat(env, frame_skip)
    return StagedRewardWrapper(env, stage=stage, flip_penalty=flip_penalty)
