"""End-to-end smoke test: train PPO on RoverEnv for a few hundred steps.

This isn't meant to converge — it's a "does the plumbing work?" check.
Run with:
    cd python && maturin develop --release && python examples/train_ppo.py

The script prints sample observations and a tiny training summary. If
PPO updates without crashing, the env/SB3 contract is satisfied.
"""

from __future__ import annotations

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from hylaeanrover import RoverEnv, OBS_DIM, ACTION_COUNT, FIXED_DT


def main() -> None:
    print(f"RoverEnv schema: obs_dim={OBS_DIM}, action_count={ACTION_COUNT}, dt={FIXED_DT:.4f}s")

    env = RoverEnv(seed=42, max_steps=600)

    # `check_env` walks the gymnasium contract — catches dtype, shape,
    # info-dict mismatches early so we don't waste a training run.
    print("Running gymnasium.check_env() …")
    check_env(env, warn=True, skip_render_check=True)
    print("  ✓ env contract OK")

    # Inspect a fresh observation.
    obs, info = env.reset(seed=42)
    print(f"\nFresh obs shape={obs.shape}, dtype={obs.dtype}")
    print(f"  imu (first 7):      {np.round(obs[:7], 3)}")
    print(f"  lidar (next 8):     {np.round(obs[7:15], 1)}")
    print(f"  minerals (35:41):   {np.round(obs[35:41], 3)}")
    print(f"  beacons_remaining:  {obs[41]}")
    print(f"  info: {info}")

    # Random rollout sanity check — confirms step() returns sensible
    # types and the env terminates within max_steps.
    print("\nRandom rollout:")
    obs, _ = env.reset(seed=42)
    total = 0.0
    for t in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        if terminated or truncated:
            print(f"  episode end at step {t}: total_reward={total:.2f}, info={info}")
            break
    else:
        print(f"  200-step partial: total_reward={total:.2f}")

    # Tiny training run — verifies SB3 can call learn() without crashing.
    # n_steps small + batch tiny + epochs=1 keeps it under a minute.
    print("\nPPO smoke test (1024 timesteps, may take ~30s on CPU)…")
    env = RoverEnv(seed=42, max_steps=600)
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=256,
        batch_size=64,
        n_epochs=1,
        verbose=0,
    )
    model.learn(total_timesteps=1024)
    print("  ✓ PPO.learn() completed without errors")


if __name__ == "__main__":
    main()
