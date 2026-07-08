"""Evaluate a trained policy (or a random baseline) on a curriculum stage.

Reports the metrics the go/no-go gates in `docs/rl_training_plan.md` use:
mean episode return / length, terminal-reason breakdown, and the final
per-component reward (distance, cube bonus, mineral integral, beacon
bonus) plus cubes picked up and beacons used.

Examples
--------
Random baseline on the locomotion stage::

    python examples/evaluate.py --stage locomotion --random --episodes 20

A trained stage-0 policy::

    python examples/evaluate.py --stage locomotion --episodes 20 \
        --load runs/stage0/model.zip --vecnorm runs/stage0/vecnorm.pkl
"""

from __future__ import annotations

import argparse
from collections import Counter
from typing import Any

import numpy as np

from hylaeanrover import BEACON_BUDGET
from hylaeanrover.wrappers import (
    CUBE_SHAPING_MODES,
    CUBE_SPAWN_PRESETS,
    EVAL_SCENARIOS,
    HORIZON_STEPS,
    STAGES,
    apply_scenario_defaults,
    make_staged_env,
    parse_terrain_height,
    resolve_vecnorm,
)


def _env_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    cfg = apply_scenario_defaults(
        args.scenario,
        cube_spawn_preset=args.cube_spawn_preset,
        power_start_fraction=args.power_start_fraction,
        terrain_height=args.terrain_height,
        cube_shaping=args.cube_shaping,
        forced_cube_distance=args.forced_cube_distance,
        forced_cube_bearing_deg=args.forced_cube_bearing_deg,
    )
    terrain_scale, terrain_range = parse_terrain_height(cfg.pop("terrain_height", None))
    cfg["terrain_height_scale"] = terrain_scale
    cfg["terrain_height_scale_range"] = terrain_range
    cfg["cube_spawn_seed"] = args.cube_spawn_seed
    if "scenario" not in cfg:
        cfg["scenario"] = args.scenario
    return cfg


class EpisodeTracker:
    def __init__(self, low_power_threshold: float) -> None:
        self.low_power_threshold = low_power_threshold
        self.first_low_power_distance: float | None = None
        self.visible_low_power_steps = 0
        self.visible_approach_steps = 0
        self._prev_visible_range: float | None = None

    def observe(self, info: dict[str, Any]) -> None:
        distance = float(info.get("reward_distance", 0.0))
        power_frac = float(info.get("power_frac", 1.0))
        low_power = power_frac <= self.low_power_threshold
        if low_power and self.first_low_power_distance is None:
            self.first_low_power_distance = distance

        visible = int(info.get("visible_cube_count", 0) or 0) > 0
        nearest_raw = info.get("nearest_visible_cube_range")
        nearest = float(nearest_raw) if nearest_raw is not None else None
        if low_power and visible and nearest is not None:
            self.visible_low_power_steps += 1
            if (
                self._prev_visible_range is not None
                and nearest < self._prev_visible_range
            ):
                self.visible_approach_steps += 1
            self._prev_visible_range = nearest
        elif not visible:
            self._prev_visible_range = None

    def finish(self, info: dict[str, Any]) -> dict[str, float]:
        distance = float(info.get("reward_distance", 0.0))
        after_low = (
            0.0
            if self.first_low_power_distance is None
            else distance - self.first_low_power_distance
        )
        approach_rate = (
            0.0
            if self.visible_low_power_steps == 0
            else self.visible_approach_steps / self.visible_low_power_steps
        )
        return {
            "end_power_frac": float(info.get("power_frac", 0.0)),
            "distance_after_low_power": after_low,
            "visible_low_power_steps": float(self.visible_low_power_steps),
            "visible_approach_rate": approach_rate,
        }


def _episode_metrics(
    info: dict[str, Any],
    ep_return: float,
    ep_len: int,
    truncated: bool,
    cube_pickups: float = 0.0,
    tracker_metrics: dict[str, float] | None = None,
) -> dict[str, Any]:
    reason = info.get("game_over") or ("truncated" if truncated else "unknown")
    metrics = {
        "return": ep_return,
        "length": ep_len,
        "reason": reason,
        "distance": float(info.get("reward_distance", 0.0)),
        "cube_bonus": float(info.get("reward_cube_bonus", 0.0)),
        # Per-episode pickup count, explicitly diffed by the caller against
        # the cumulative `cube_pickups` info key (which is process-lifetime
        # cumulative, not per-episode — see hylaeanrover_py's make_info()).
        "cube_pickups": cube_pickups,
        "mineral": float(info.get("reward_mineral_integral", 0.0)),
        "beacon_bonus": float(info.get("reward_beacon_bonus", 0.0)),
        "beacons_used": BEACON_BUDGET
        - int(info.get("beacons_remaining", BEACON_BUDGET)),
        "end_power_frac": float(info.get("power_frac", 0.0)),
        "distance_after_low_power": 0.0,
        "visible_low_power_steps": 0.0,
        "visible_approach_rate": 0.0,
        "terrain_height_scale": float(info.get("terrain_height_scale", 1.0)),
    }
    if tracker_metrics:
        metrics.update(tracker_metrics)
    return metrics


def run_random(
    stage: str,
    episodes: int,
    seed: int,
    max_steps: int,
    frame_skip: int = 1,
    power_capacity: float | None = None,
    env_kwargs: dict[str, Any] | None = None,
    low_power_threshold: float = 0.45,
) -> list[dict[str, Any]]:
    env_kwargs = env_kwargs or {}
    env = make_staged_env(
        stage,
        seed=seed,
        max_steps=max_steps,
        frame_skip=frame_skip,
        power_capacity=power_capacity,
        **env_kwargs,
    )
    rng = np.random.default_rng(seed)
    out = []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        # cube_pickups is cumulative across the env's whole lifetime, so
        # the value right after reset() is this episode's baseline.
        prev_pickups = float(info.get("cube_pickups", 0.0))
        tracker = EpisodeTracker(low_power_threshold)
        tracker.observe(info)
        ep_return, ep_len, truncated = 0.0, 0, False
        while True:
            action = int(rng.integers(env.action_space.n))
            obs, reward, terminated, truncated, info = env.step(action)
            tracker.observe(info)
            ep_return += reward
            ep_len += 1
            if terminated or truncated:
                break
        cube_pickups = float(info.get("episode_cube_pickups", -1.0))
        if cube_pickups < 0.0:
            cube_pickups = float(info.get("cube_pickups", 0.0)) - prev_pickups
        out.append(
            _episode_metrics(
                info, ep_return, ep_len, truncated, cube_pickups, tracker.finish(info)
            )
        )
    env.close()
    return out


def run_model(
    stage: str,
    episodes: int,
    seed: int,
    max_steps: int,
    model_path: str,
    vecnorm_path: str | None,
    frame_skip: int = 1,
    power_capacity: float | None = None,
    env_kwargs: dict[str, Any] | None = None,
    low_power_threshold: float = 0.45,
) -> list[dict[str, Any]]:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    env_kwargs = env_kwargs or {}
    env = make_staged_env(
        stage,
        seed=seed,
        max_steps=max_steps,
        frame_skip=frame_skip,
        power_capacity=power_capacity,
        **env_kwargs,
    )
    normalizer = None
    if vecnorm_path:
        normalizer = VecNormalize.load(vecnorm_path, DummyVecEnv([lambda: env]))
        normalizer.training = False
        normalizer.norm_reward = False
    model = PPO.load(model_path, device="cpu")

    out = []
    try:
        for ep in range(episodes):
            obs, info = env.reset(seed=seed + ep)
            # cube_pickups is cumulative across the env's whole lifetime, so
            # the value right after reset() is this episode's baseline.
            prev_pickups = float(info.get("cube_pickups", 0.0))
            tracker = EpisodeTracker(low_power_threshold)
            tracker.observe(info)
            ep_return, ep_len, truncated = 0.0, 0, False
            while True:
                model_obs = np.asarray([obs], dtype=np.float32)
                if normalizer is not None:
                    model_obs = normalizer.normalize_obs(model_obs)
                action, _ = model.predict(model_obs, deterministic=True)
                step_action = int(np.asarray(action).reshape(-1)[0])
                obs, reward, terminated, truncated, info = env.step(step_action)
                tracker.observe(info)
                ep_return += float(reward)
                ep_len += 1
                if terminated or truncated:
                    break
            total_pickups = float(info.get("cube_pickups", 0.0))
            cube_pickups = float(info.get("episode_cube_pickups", -1.0))
            if cube_pickups < 0.0:
                cube_pickups = total_pickups - prev_pickups
            out.append(
                _episode_metrics(
                    info,
                    ep_return,
                    ep_len,
                    truncated,
                    cube_pickups,
                    tracker.finish(info),
                )
            )
    finally:
        if normalizer is not None:
            normalizer.close()
        else:
            env.close()
    return out


def summarize(label: str, eps: list[dict[str, Any]]) -> None:
    arr = lambda k: np.array([e[k] for e in eps], dtype=float)
    reasons = Counter(e["reason"] for e in eps)
    n = len(eps)
    print(f"\n=== {label} ({n} episodes) ===")
    print(
        f"  return        mean={arr('return').mean():10.2f}  std={arr('return').std():8.2f}"
    )
    print(
        f"  length        mean={arr('length').mean():10.1f}  std={arr('length').std():8.1f}"
    )
    print(f"  distance      mean={arr('distance').mean():10.2f}")
    print(f"  cube_bonus    mean={arr('cube_bonus').mean():10.2f}")
    print(f"  cube_pickups  mean={arr('cube_pickups').mean():10.2f}")
    print(f"  end_power     mean={arr('end_power_frac').mean():10.2f}")
    print(f"  dist_low_pwr  mean={arr('distance_after_low_power').mean():10.2f}")
    print(f"  visible_low   mean={arr('visible_low_power_steps').mean():10.2f}")
    print(f"  approach_rate mean={arr('visible_approach_rate').mean():10.2%}")
    print(f"  terrain_scale mean={arr('terrain_height_scale').mean():10.2f}")
    print(f"  mineral       mean={arr('mineral').mean():10.2f}")
    print(f"  beacon_bonus  mean={arr('beacon_bonus').mean():10.2f}")
    print(f"  beacons_used  mean={arr('beacons_used').mean():10.2f}")
    print(f"  flip_rate     {reasons.get('flipped', 0) / n:.2%}")
    print(f"  out_of_power  {reasons.get('out_of_power', 0) / n:.2%}")
    print(f"  terminal reasons: {dict(reasons)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=STAGES, required=True)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument(
        "--horizon",
        choices=HORIZON_STEPS.keys(),
        default=None,
        help="named episode horizon; overrides --max-steps",
    )
    p.add_argument("--random", action="store_true", help="evaluate a random policy")
    p.add_argument("--load", type=str, default=None, help="model.zip to evaluate")
    p.add_argument("--vecnorm", type=str, default=None, help="vecnorm.pkl to apply")
    p.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="must match the value used during training",
    )
    p.add_argument(
        "--power-capacity",
        type=float,
        default=None,
        help="battery capacity in Wh (default: the training default from "
        "hylaeanrover.wrappers); must match the value used during training",
    )
    p.add_argument("--scenario", choices=EVAL_SCENARIOS.keys(), default=None)
    p.add_argument(
        "--cube-spawn-preset", choices=CUBE_SPAWN_PRESETS.keys(), default=None
    )
    p.add_argument("--cube-spawn-seed", type=int, default=None)
    p.add_argument("--power-start-fraction", type=float, default=None)
    p.add_argument(
        "--forced-cube-distance",
        type=float,
        default=None,
        help=(
            "place one cube at reset this many meters from the rover; "
            "must be paired with --forced-cube-bearing-deg"
        ),
    )
    p.add_argument(
        "--forced-cube-bearing-deg",
        type=float,
        default=None,
        help=(
            "bearing for a reset-time forced cube, within the cube sensor "
            "cone; must be paired with --forced-cube-distance"
        ),
    )
    p.add_argument(
        "--terrain-height",
        type=str,
        default=None,
        help="terrain height preset/name, fixed value, or min:max range",
    )
    p.add_argument(
        "--cube-shaping",
        choices=CUBE_SHAPING_MODES,
        default="off",
        help="usually off for acceptance evals; train can enable shaping",
    )
    p.add_argument("--low-power-threshold", type=float, default=0.45)
    args = p.parse_args()
    if args.horizon is not None:
        args.max_steps = HORIZON_STEPS[args.horizon]
    env_kwargs = _env_kwargs_from_args(args)

    if args.random:
        summarize(
            "random baseline",
            run_random(
                args.stage,
                args.episodes,
                args.seed,
                args.max_steps,
                args.frame_skip,
                args.power_capacity,
                env_kwargs,
                args.low_power_threshold,
            ),
        )
    elif args.load:
        vecnorm = resolve_vecnorm(args.load, args.vecnorm)
        summarize(
            f"trained ({args.load})",
            run_model(
                args.stage,
                args.episodes,
                args.seed,
                args.max_steps,
                args.load,
                vecnorm,
                args.frame_skip,
                args.power_capacity,
                env_kwargs,
                args.low_power_threshold,
            ),
        )
    else:
        p.error("pass --random or --load <model.zip>")


if __name__ == "__main__":
    main()
