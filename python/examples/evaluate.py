"""Evaluate a trained policy (or a random baseline) on a curriculum stage.

Reports the metrics the go/no-go gates in `docs/rl_training_plan.md` use:
mean episode return / length, terminal-reason breakdown, and the final
per-component reward (distance, mineral integral, beacon bonus) plus
beacons used.

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
from hylaeanrover.wrappers import STAGES, make_staged_env, resolve_vecnorm


def _episode_metrics(info: dict[str, Any], ep_return: float, ep_len: int, truncated: bool) -> dict[str, Any]:
    reason = info.get("game_over") or ("truncated" if truncated else "unknown")
    return {
        "return": ep_return,
        "length": ep_len,
        "reason": reason,
        "distance": float(info.get("reward_distance", 0.0)),
        "mineral": float(info.get("reward_mineral_integral", 0.0)),
        "beacon_bonus": float(info.get("reward_beacon_bonus", 0.0)),
        "beacons_used": BEACON_BUDGET - int(info.get("beacons_remaining", BEACON_BUDGET)),
    }


def run_random(
    stage: str,
    episodes: int,
    seed: int,
    max_steps: int,
    frame_skip: int = 1,
    power_capacity: float | None = None,
) -> list[dict[str, Any]]:
    env = make_staged_env(
        stage, seed=seed, max_steps=max_steps, frame_skip=frame_skip,
        power_capacity=power_capacity,
    )
    rng = np.random.default_rng(seed)
    out = []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        ep_return, ep_len, truncated = 0.0, 0, False
        while True:
            action = int(rng.integers(env.action_space.n))
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            ep_len += 1
            if terminated or truncated:
                break
        out.append(_episode_metrics(info, ep_return, ep_len, truncated))
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
) -> list[dict[str, Any]]:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    venv = DummyVecEnv(
        [
            lambda: make_staged_env(
                stage, seed=seed, max_steps=max_steps, frame_skip=frame_skip,
                power_capacity=power_capacity,
            )
        ]
    )
    if vecnorm_path:
        venv = VecNormalize.load(vecnorm_path, venv)
        venv.training = False
        venv.norm_reward = False
    model = PPO.load(model_path, device="cpu")

    out = []
    obs = venv.reset()
    ep_return, ep_len = 0.0, 0
    while len(out) < episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = venv.step(action)
        ep_return += float(rewards[0])
        ep_len += 1
        if dones[0]:
            # DummyVecEnv has already autoreset; the pre-reset info is in
            # infos[0] (with the env's final telemetry keys preserved).
            truncated = bool(infos[0].get("TimeLimit.truncated", False))
            out.append(_episode_metrics(infos[0], ep_return, ep_len, truncated))
            ep_return, ep_len = 0.0, 0
    venv.close()
    return out


def summarize(label: str, eps: list[dict[str, Any]]) -> None:
    arr = lambda k: np.array([e[k] for e in eps], dtype=float)
    reasons = Counter(e["reason"] for e in eps)
    n = len(eps)
    print(f"\n=== {label} ({n} episodes) ===")
    print(f"  return        mean={arr('return').mean():10.2f}  std={arr('return').std():8.2f}")
    print(f"  length        mean={arr('length').mean():10.1f}  std={arr('length').std():8.1f}")
    print(f"  distance      mean={arr('distance').mean():10.2f}")
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
    p.add_argument("--random", action="store_true", help="evaluate a random policy")
    p.add_argument("--load", type=str, default=None, help="model.zip to evaluate")
    p.add_argument("--vecnorm", type=str, default=None, help="vecnorm.pkl to apply")
    p.add_argument("--frame-skip", type=int, default=1,
                   help="must match the value used during training")
    p.add_argument("--power-capacity", type=float, default=None,
                   help="battery capacity in Wh (default: the training default from "
                        "hylaeanrover.wrappers); must match the value used during training")
    args = p.parse_args()

    if args.random:
        summarize(
            "random baseline",
            run_random(args.stage, args.episodes, args.seed, args.max_steps, args.frame_skip,
                       args.power_capacity),
        )
    elif args.load:
        vecnorm = resolve_vecnorm(args.load, args.vecnorm)
        summarize(
            f"trained ({args.load})",
            run_model(args.stage, args.episodes, args.seed, args.max_steps, args.load, vecnorm,
                      args.frame_skip, args.power_capacity),
        )
    else:
        p.error("pass --random or --load <model.zip>")


if __name__ == "__main__":
    main()
