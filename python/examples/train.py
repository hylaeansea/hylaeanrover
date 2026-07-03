"""Staged PPO training for the Hylaean Rover.

Trains one curriculum stage (locomotion → power_cubes → minerals → full), optionally
warm-starting from the previous stage's weights + observation
normalization. See `docs/rl_training_plan.md` for the design.

Examples
--------
Stage 0 (locomotion) from scratch::

    python examples/train.py --stage locomotion --timesteps 1000000 \
        --save runs/stage0

Stage 1 (power cubes), warm-started from stage 0::

    python examples/train.py --stage power_cubes --timesteps 1000000 \
        --load runs/stage0/model.zip --vecnorm runs/stage0/vecnorm.pkl \
        --save runs/stage1

Watch progress with ``tensorboard --logdir runs/``.
"""

from __future__ import annotations

import argparse
import os
import re
from collections import Counter

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.utils import FloatSchedule
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

import hylaeanrover
from hylaeanrover.wrappers import (
    CUBE_SHAPING_MODES,
    CUBE_SPAWN_PRESETS,
    DEFAULT_POWER_CAPACITY_WH,
    EVAL_SCENARIOS,
    HORIZON_STEPS,
    LOCOMOTION_SHAPING_MODES,
    STAGES,
    apply_scenario_defaults,
    make_staged_env,
    parse_terrain_height,
    resolve_vecnorm,
)

_STAGE_HINT_PATTERNS = {
    "locomotion": (r"(^|[/_.-])(stage0|stage_0|locomotion)($|[/_.-])",),
    "power_cubes": (
        r"(^|[/_.-])(stage1|stage_1|power_cubes|power-cubes|powercubes)($|[/_.-])",
    ),
    "minerals": (r"(^|[/_.-])(stage2|stage_2|minerals)($|[/_.-])",),
    "full": (r"(^|[/_.-])(stage3|stage_3|full)($|[/_.-])",),
}


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
      episodes/mean_cube_bonus    mean per-episode cube pickup bonus
                                  (reward_cube_bonus resets every episode,
                                  so this is already episode-scoped;
                                  divide by CUBE_PICKUP_BONUS in
                                  reward.rs for mean pickups/episode)
    """

    _REASONS = ("time_limit", "out_of_power", "flipped", "beacons_deployed", "other")

    def __init__(self) -> None:
        super().__init__()
        self._counts: Counter[str] = Counter()
        self._end_power: list[float] = []
        self._cube_bonus: list[float] = []

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
            if "reward_cube_bonus" in info:
                self._cube_bonus.append(float(info["reward_cube_bonus"]))
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
        if self._cube_bonus:
            self.logger.record(
                "episodes/mean_cube_bonus", float(np.mean(self._cube_bonus))
            )
        self._counts.clear()
        self._end_power.clear()
        self._cube_bonus.clear()


def _env_thunk(
    stage: str,
    seed: int,
    max_steps: int,
    frame_skip: int,
    power_capacity: float,
    env_kwargs: dict,
):
    """Picklable factory (no late-binding closure) for one Monitor-wrapped env."""

    def _make():
        return Monitor(
            make_staged_env(
                stage,
                seed=seed,
                max_steps=max_steps,
                frame_skip=frame_skip,
                power_capacity=power_capacity,
                **env_kwargs,
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
    env_kwargs: dict,
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
        _env_thunk(
            stage, base_seed + i, max_steps, frame_skip, power_capacity, env_kwargs
        )
        for i in range(n_envs)
    ]
    if n_envs == 1:
        return DummyVecEnv(fns)
    return SubprocVecEnv(fns, start_method="spawn")


def _env_kwargs_from_args(
    args: argparse.Namespace, scenario: str | None = None
) -> dict:
    terrain_height = args.terrain_height
    if terrain_height is None and args.stage in ("locomotion", "power_cubes"):
        terrain_height = "mixed_1_2"
    cube_shaping = args.cube_shaping
    if cube_shaping == "auto":
        cube_shaping = "low_power" if args.stage == "power_cubes" else "off"
    locomotion_shaping = args.locomotion_shaping
    if locomotion_shaping == "auto":
        locomotion_shaping = "power_efficiency" if args.stage == "locomotion" else "off"
    cfg = apply_scenario_defaults(
        scenario if scenario is not None else args.scenario,
        cube_spawn_preset=args.cube_spawn_preset,
        power_start_fraction=args.power_start_fraction,
        terrain_height=terrain_height,
        cube_shaping=cube_shaping,
    )
    terrain_scale, terrain_range = parse_terrain_height(cfg.pop("terrain_height", None))
    cfg["terrain_height_scale"] = terrain_scale
    cfg["terrain_height_scale_range"] = terrain_range
    cfg["cube_spawn_seed"] = args.cube_spawn_seed
    cfg["locomotion_shaping"] = locomotion_shaping
    cfg["locomotion_coast_bonus"] = args.locomotion_coast_bonus
    cfg["locomotion_power_draw_penalty"] = args.locomotion_power_draw_penalty
    cfg["locomotion_power_recovery_reward"] = args.locomotion_power_recovery_reward
    cfg["locomotion_out_of_power_penalty"] = args.locomotion_out_of_power_penalty
    if "scenario" not in cfg:
        cfg["scenario"] = scenario if scenario is not None else args.scenario
    return cfg


def _stage_hint_from_paths(*paths: str | None) -> str | None:
    hints: set[str] = set()
    for path in paths:
        if not path:
            continue
        normalized = path.lower().replace("\\", "/")
        for stage, patterns in _STAGE_HINT_PATTERNS.items():
            if any(re.search(pattern, normalized) for pattern in patterns):
                hints.add(stage)
    if len(hints) == 1:
        return next(iter(hints))
    return None


def _reward_stats_reset_mode(
    args: argparse.Namespace, vecnorm_path: str | None
) -> tuple[bool, str]:
    if args.reset_reward_stats:
        return True, "forced by --reset-reward-stats"
    if args.preserve_reward_stats:
        return False, "forced by --preserve-reward-stats"

    loaded_stage = _stage_hint_from_paths(args.load, vecnorm_path)
    if loaded_stage == args.stage:
        return False, f"loaded path looks like same stage ({loaded_stage})"
    if loaded_stage is not None:
        return True, f"stage transition {loaded_stage} -> {args.stage}"
    return True, "loaded stage is unknown"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=STAGES, required=True)
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument(
        "--horizon",
        choices=HORIZON_STEPS.keys(),
        default=None,
        help="named episode horizon; overrides --max-steps",
    )
    p.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="parallel sim processes (SubprocVecEnv when >1); ~physical cores",
    )
    p.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="hold each action for N physics ticks (fewer decisions/sec)",
    )
    p.add_argument("--save", type=str, required=True, help="output dir")
    p.add_argument("--load", type=str, default=None, help="prev stage model.zip")
    p.add_argument("--vecnorm", type=str, default=None, help="prev stage vecnorm.pkl")
    reward_stats_group = p.add_mutually_exclusive_group()
    reward_stats_group.add_argument(
        "--reset-reward-stats",
        action="store_true",
        help="reset VecNormalize reward RMS after loading stats; use for stage transitions",
    )
    reward_stats_group.add_argument(
        "--preserve-reward-stats",
        action="store_true",
        help="keep loaded VecNormalize reward RMS; use for same-stage continuation",
    )
    p.add_argument("--eval-freq", type=int, default=25_000)
    p.add_argument("--n-eval-episodes", type=int, default=10)
    p.add_argument("--checkpoint-freq", type=int, default=100_000)
    p.add_argument(
        "--scenario",
        choices=EVAL_SCENARIOS.keys(),
        default=None,
        help="named Stage 0/1 scenario preset",
    )
    p.add_argument(
        "--extra-eval-scenarios",
        type=str,
        default="",
        help="comma-separated scenario names to evaluate periodically",
    )
    p.add_argument(
        "--cube-spawn-preset", choices=CUBE_SPAWN_PRESETS.keys(), default=None
    )
    p.add_argument("--cube-spawn-seed", type=int, default=None)
    p.add_argument("--power-start-fraction", type=float, default=None)
    p.add_argument(
        "--terrain-height",
        type=str,
        default=None,
        help="terrain height preset/name, fixed value, or min:max range",
    )
    p.add_argument(
        "--cube-shaping",
        choices=("auto",) + CUBE_SHAPING_MODES,
        default="auto",
        help="Stage 1 low-power cube approach shaping; auto enables it for power_cubes",
    )
    p.add_argument(
        "--locomotion-shaping",
        choices=("auto",) + LOCOMOTION_SHAPING_MODES,
        default="auto",
        help="Stage 0 pacing shaping; auto enables power_efficiency for locomotion",
    )
    p.add_argument(
        "--locomotion-coast-bonus",
        type=float,
        default=0.35,
        help="extra reward multiplier for zero-throttle distance in Stage 0 shaping",
    )
    p.add_argument(
        "--locomotion-power-draw-penalty",
        type=float,
        default=40.0,
        help="penalty per battery-fraction spent in Stage 0 shaping",
    )
    p.add_argument(
        "--locomotion-power-recovery-reward",
        type=float,
        default=20.0,
        help="reward per battery-fraction recovered in Stage 0 shaping",
    )
    p.add_argument(
        "--locomotion-out-of-power-penalty",
        type=float,
        default=75.0,
        help="terminal penalty for out-of-power in Stage 0 shaping",
    )
    # PPO stability knobs. Defaults reproduce SB3's stock behavior, so the
    # run is unchanged unless these are passed. They exist to fight the
    # "climbs then collapses" failure mode: a rollout with an oversized
    # policy update (approx_kl spike) knocks the policy into a degenerate
    # region it can't escape.
    p.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="PPO learning rate; lower this when KL early-stopping is frequent",
    )
    p.add_argument(
        "--n-epochs",
        type=int,
        default=10,
        help="PPO epochs per rollout; lower = smaller, more conservative updates",
    )
    p.add_argument(
        "--ent-coef",
        type=float,
        default=0.0,
        help="entropy bonus; >0 (e.g. 0.01) keeps exploration alive to resist collapse",
    )
    p.add_argument(
        "--target-kl",
        type=float,
        default=None,
        help="early-stop a rollout's epochs once approx_kl exceeds this (e.g. 0.02)",
    )
    p.add_argument(
        "--power-capacity",
        type=float,
        default=DEFAULT_POWER_CAPACITY_WH,
        help="battery capacity in Wh per episode (refilled on reset). The "
        "default binds within an episode so power management is part "
        "of the learned behavior; keep it consistent across curriculum "
        "stages, like --frame-skip. The game battery is 1000.",
    )
    args = p.parse_args()
    if args.horizon is not None:
        args.max_steps = HORIZON_STEPS[args.horizon]

    os.makedirs(args.save, exist_ok=True)
    env_kwargs = _env_kwargs_from_args(args)

    # Warm-starting a policy without its observation normalization stats
    # would feed it un-normalized obs (wrong scale → broken transfer), so
    # whenever --load is given we require the matching vecnorm (falling
    # back to the sibling vecnorm.pkl train.py saves next to the model).
    vecnorm_path = (
        resolve_vecnorm(args.load, args.vecnorm) if args.load else args.vecnorm
    )

    # --- Training env, normalized -----------------------------------------
    base = _make_vec_env(
        args.stage,
        args.seed,
        args.max_steps,
        args.frame_skip,
        args.n_envs,
        args.power_capacity,
        env_kwargs,
    )
    if vecnorm_path:
        # Carry observation normalization stats forward. Reward stats reset by
        # default for stage transitions, but same-stage continuations should
        # preserve them so reward scaling does not jump mid-training.
        reset_reward_stats, reward_stats_reason = _reward_stats_reset_mode(
            args, vecnorm_path
        )
        venv = VecNormalize.load(vecnorm_path, base)
        venv.training = True
        venv.norm_reward = True
        if reset_reward_stats:
            venv.ret_rms = RunningMeanStd(shape=())
        venv.returns = np.zeros(venv.num_envs)
        reward_status = "reset" if reset_reward_stats else "preserved"
        print(
            f"Loaded normalization stats from {vecnorm_path} "
            f"(reward stats {reward_status}: {reward_stats_reason})"
        )
    else:
        venv = VecNormalize(base, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # --- Eval env (separate, single process; EvalCallback syncs obs stats) -
    eval_venv = VecNormalize(
        _make_vec_env(
            args.stage,
            args.seed + 1000,
            args.max_steps,
            args.frame_skip,
            1,
            args.power_capacity,
            env_kwargs,
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
        model.learning_rate = args.learning_rate
        model.lr_schedule = FloatSchedule(args.learning_rate)
        print(f"Warm-started policy from {args.load}")
    else:
        model = PPO(
            "MlpPolicy",
            venv,
            n_steps=2048,
            batch_size=64,
            n_epochs=args.n_epochs,
            learning_rate=args.learning_rate,
            gamma=0.99,
            ent_coef=args.ent_coef,
            target_kl=args.target_kl,
            tensorboard_log=tb_dir,
            verbose=1,
            device="cpu",
        )
    print(
        f"PPO: learning_rate={args.learning_rate} n_epochs={args.n_epochs} "
        f"ent_coef={args.ent_coef} target_kl={args.target_kl}"
    )
    print(f"env: power_capacity={args.power_capacity} Wh, frame_skip={args.frame_skip}")
    print(f"env kwargs: {env_kwargs}")
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

    for scenario in [
        s.strip() for s in args.extra_eval_scenarios.split(",") if s.strip()
    ]:
        if scenario not in EVAL_SCENARIOS:
            raise SystemExit(f"unknown extra eval scenario {scenario!r}")
        extra_kwargs = _env_kwargs_from_args(args, scenario=scenario)
        extra_eval = VecNormalize(
            _make_vec_env(
                args.stage,
                args.seed + 2000 + len(callbacks),
                args.max_steps,
                args.frame_skip,
                1,
                args.power_capacity,
                extra_kwargs,
            ),
            norm_obs=True,
            norm_reward=False,
            training=False,
        )
        callbacks.append(
            EvalCallback(
                extra_eval,
                best_model_save_path=None,
                log_path=os.path.join(args.save, "eval", scenario),
                eval_freq=args.eval_freq,
                n_eval_episodes=args.n_eval_episodes,
                deterministic=True,
            )
        )

    model.learn(total_timesteps=args.timesteps, callback=callbacks)

    # --- Persist for the next stage + for evaluate.py ---------------------
    model.save(os.path.join(args.save, "model.zip"))
    venv.save(os.path.join(args.save, "vecnorm.pkl"))
    print(f"\nSaved model + vecnorm to {args.save}/")
    print(f"Next stage: --load {args.save}/model.zip --vecnorm {args.save}/vecnorm.pkl")


if __name__ == "__main__":
    main()
