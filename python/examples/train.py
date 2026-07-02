"""Staged PPO training for the Hylaean Rover.

Trains one curriculum stage (locomotion → minerals → full), optionally
warm-starting from the previous stage's weights + observation
normalization. See `docs/rl_training_plan.md` for the design.

Examples
--------
Stage 0 (locomotion) from scratch::

    python examples/train.py --stage locomotion --timesteps 1000000 \
        --save runs/stage0

Stage 1 (minerals), warm-started from stage 0::

    python examples/train.py --stage minerals --timesteps 1000000 \
        --load runs/stage0/model.zip --vecnorm runs/stage0/vecnorm.pkl \
        --save runs/stage1

Watch progress with ``tensorboard --logdir runs/``.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

import hylaeanrover
from hylaeanrover.wrappers import (
    DEFAULT_POWER_CAPACITY_WH,
    STAGES,
    make_staged_env,
    resolve_vecnorm,
)


class SaveBestVecNormalize(BaseCallback):
    """Save the training VecNormalize stats whenever EvalCallback finds a
    new best model. Pass as EvalCallback(callback_on_new_best=...).

    Without this, `best/best_model.zip` would have no matching
    `vecnorm.pkl`, and loading it later (to resume or to run the
    autopilot) would silently feed un-normalized observations.
    """

    def __init__(self, save_path: str) -> None:
        super().__init__()
        self.save_path = save_path

    def _on_step(self) -> bool:
        vn = self.model.get_vec_normalize_env()
        if vn is not None:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            vn.save(self.save_path)
        return True


class EpisodeOutcomeCallback(BaseCallback):
    """Log how episodes end, per rollout, to TensorBoard.

    Environment-driven failures look exactly like RL instability in the
    default SB3 metrics (ep_rew_mean tanks, ep_len_mean shrinks) — the
    battery-never-reset bug ran for weeks disguised as "PPO collapse".
    Terminal-reason fractions make the difference visible immediately:

      episodes/frac_time_limit    ended by truncation (ran the clock out)
      episodes/frac_out_of_power  battery died, rover came to rest
      episodes/frac_flipped       tipped past the flip angle at rest
      episodes/end_power_frac     mean battery fraction left at episode
                                  end — how hard the power budget binds
    """

    _REASONS = ("time_limit", "out_of_power", "flipped", "beacons_deployed", "other")

    def __init__(self) -> None:
        super().__init__()
        self._counts: Counter[str] = Counter()
        self._end_power: list[float] = []

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if not done:
                continue
            reason = info.get("game_over")
            if reason is None:
                reason = (
                    "time_limit" if info.get("TimeLimit.truncated", False) else "other"
                )
            self._counts[reason if reason in self._REASONS else "other"] += 1
            if "power_frac" in info:
                self._end_power.append(float(info["power_frac"]))
        return True

    def _on_rollout_end(self) -> None:
        total = sum(self._counts.values())
        if total == 0:
            return
        for reason in self._REASONS:
            self.logger.record(f"episodes/frac_{reason}", self._counts[reason] / total)
        if self._end_power:
            self.logger.record(
                "episodes/end_power_frac", float(np.mean(self._end_power))
            )
        self._counts.clear()
        self._end_power.clear()


def _env_thunk(stage: str, seed: int, max_steps: int, frame_skip: int, power_capacity: float):
    """Picklable factory (no late-binding closure) for one Monitor-wrapped env."""

    def _make():
        return Monitor(
            make_staged_env(
                stage,
                seed=seed,
                max_steps=max_steps,
                frame_skip=frame_skip,
                power_capacity=power_capacity,
            )
        )

    return _make


def _make_vec_env(
    stage: str,
    base_seed: int,
    max_steps: int,
    frame_skip: int,
    n_envs: int,
    power_capacity: float,
):
    """Vectorized env for the given stage.

    `n_envs == 1` uses `DummyVecEnv` (no process overhead). `n_envs > 1`
    uses `SubprocVecEnv` — separate processes, because the Bevy `App` is
    `!Send` so thread-based parallelism (DummyVecEnv) can't run sims
    concurrently. Each worker gets a distinct seed so they explore
    different terrain. `start_method="spawn"` is required: forking after
    Bevy has spawned its task-pool threads is unsafe.
    """
    fns = [
        _env_thunk(stage, base_seed + i, max_steps, frame_skip, power_capacity)
        for i in range(n_envs)
    ]
    if n_envs == 1:
        return DummyVecEnv(fns)
    return SubprocVecEnv(fns, start_method="spawn")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=STAGES, required=True)
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--n-envs", type=int, default=1,
                   help="parallel sim processes (SubprocVecEnv when >1); ~physical cores")
    p.add_argument("--frame-skip", type=int, default=1,
                   help="hold each action for N physics ticks (fewer decisions/sec)")
    p.add_argument("--save", type=str, required=True, help="output dir")
    p.add_argument("--load", type=str, default=None, help="prev stage model.zip")
    p.add_argument("--vecnorm", type=str, default=None, help="prev stage vecnorm.pkl")
    p.add_argument("--eval-freq", type=int, default=25_000)
    p.add_argument("--n-eval-episodes", type=int, default=10)
    p.add_argument("--checkpoint-freq", type=int, default=100_000)
    # PPO stability knobs. Defaults reproduce SB3's stock behavior, so the
    # run is unchanged unless these are passed. They exist to fight the
    # "climbs then collapses" failure mode: a rollout with an oversized
    # policy update (approx_kl spike) knocks the policy into a degenerate
    # region it can't escape.
    p.add_argument("--n-epochs", type=int, default=10,
                   help="PPO epochs per rollout; lower = smaller, more conservative updates")
    p.add_argument("--ent-coef", type=float, default=0.0,
                   help="entropy bonus; >0 (e.g. 0.01) keeps exploration alive to resist collapse")
    p.add_argument("--target-kl", type=float, default=None,
                   help="early-stop a rollout's epochs once approx_kl exceeds this (e.g. 0.02)")
    p.add_argument("--power-capacity", type=float, default=DEFAULT_POWER_CAPACITY_WH,
                   help="battery capacity in Wh per episode (refilled on reset). The "
                        "default binds within an episode so power management is part "
                        "of the learned behavior; keep it consistent across curriculum "
                        "stages, like --frame-skip. The game battery is 1000.")
    args = p.parse_args()

    os.makedirs(args.save, exist_ok=True)

    # Warm-starting a policy without its observation normalization stats
    # would feed it un-normalized obs (wrong scale → broken transfer), so
    # whenever --load is given we require the matching vecnorm (falling
    # back to the sibling vecnorm.pkl train.py saves next to the model).
    vecnorm_path = resolve_vecnorm(args.load, args.vecnorm) if args.load else args.vecnorm

    # --- Training env, normalized -----------------------------------------
    base = _make_vec_env(
        args.stage, args.seed, args.max_steps, args.frame_skip, args.n_envs,
        args.power_capacity,
    )
    if vecnorm_path:
        # Carry the observation normalization stats forward (obs
        # distribution is stable across stages) but reset the reward
        # stats — reward scale jumps between stages.
        venv = VecNormalize.load(vecnorm_path, base)
        venv.training = True
        venv.norm_reward = True
        venv.ret_rms = RunningMeanStd(shape=())
        venv.returns = np.zeros(venv.num_envs)
        print(f"Loaded obs-normalization stats from {vecnorm_path} (reward stats reset)")
    else:
        venv = VecNormalize(base, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # --- Eval env (separate, single process; EvalCallback syncs obs stats) -
    eval_venv = VecNormalize(
        _make_vec_env(
            args.stage, args.seed + 1000, args.max_steps, args.frame_skip, 1,
            args.power_capacity,
        ),
        norm_obs=True,
        norm_reward=False,
        training=False,
    )

    # --- Model: warm-start or fresh ---------------------------------------
    tb_dir = os.path.join(args.save, "tb")
    if args.load:
        model = PPO.load(args.load, env=venv, tensorboard_log=tb_dir, device="cpu")
        # CLI flags are authoritative on a resume: apply the stability knobs
        # rather than silently inheriting whatever the saved model used.
        model.n_epochs = args.n_epochs
        model.ent_coef = args.ent_coef
        model.target_kl = args.target_kl
        print(f"Warm-started policy from {args.load}")
    else:
        model = PPO(
            "MlpPolicy",
            venv,
            n_steps=2048,
            batch_size=64,
            n_epochs=args.n_epochs,
            gamma=0.99,
            ent_coef=args.ent_coef,
            target_kl=args.target_kl,
            tensorboard_log=tb_dir,
            verbose=1,
            device="cpu",
        )
    print(
        f"PPO: n_epochs={args.n_epochs} ent_coef={args.ent_coef} "
        f"target_kl={args.target_kl}"
    )
    print(f"env: power_capacity={args.power_capacity} Wh, frame_skip={args.frame_skip}")
    print(f"env binary: {hylaeanrover._native.__file__}")

    callbacks = [
        EpisodeOutcomeCallback(),
        CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=os.path.join(args.save, "checkpoints"),
            name_prefix=f"ppo_{args.stage}",
        ),
        EvalCallback(
            eval_venv,
            best_model_save_path=os.path.join(args.save, "best"),
            log_path=os.path.join(args.save, "eval"),
            eval_freq=args.eval_freq,
            n_eval_episodes=args.n_eval_episodes,
            deterministic=True,
            # Save matching obs-normalization stats next to best_model.zip
            # so the best checkpoint is a complete, loadable bundle.
            callback_on_new_best=SaveBestVecNormalize(
                os.path.join(args.save, "best", "vecnorm.pkl")
            ),
        ),
    ]

    model.learn(total_timesteps=args.timesteps, callback=callbacks)

    # --- Persist for the next stage + for evaluate.py ---------------------
    model.save(os.path.join(args.save, "model.zip"))
    venv.save(os.path.join(args.save, "vecnorm.pkl"))
    print(f"\nSaved model + vecnorm to {args.save}/")
    print(f"Next stage: --load {args.save}/model.zip --vecnorm {args.save}/vecnorm.pkl")


if __name__ == "__main__":
    main()
