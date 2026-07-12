"""Staged PPO training for the Hylaean Rover.

Trains one curriculum stage (locomotion → cube_intercept → power_idle →
power_cubes → minerals → full), optionally warm-starting from the previous stage's
weights + observation
normalization. See `docs/rl_training_plan.md` for the design.

Examples
--------
Stage 0 (locomotion) from scratch::

    python examples/train.py --stage locomotion --timesteps 1000000 \
        --save runs/stage0

Stage 1A (cube intercept), warm-started from stage 0::

    python examples/train.py --stage cube_intercept --timesteps 0 \
        --load runs/stage0/model.zip --vecnorm runs/stage0/vecnorm.pkl \
        --scenario cube_intercept --teacher-pretrain-samples 20000 \
        --teacher-pretrain-only --save runs/stage1_cube_intercept_bc

Watch progress with ``tensorboard --logdir runs/``.
"""

from __future__ import annotations

import argparse
import os
import re
from collections import Counter

import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecNormalize,
    sync_envs_normalization,
)

import hylaeanrover
from hylaeanrover.teacher import (
    CubeInterceptTeacher,
    MineralExploreTeacher,
    PowerIdleTeacher,
)
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
    "cube_intercept": (
        r"(^|[/_.-])(cube_intercept|cube-intercept|intercept)($|[/_.-])",
    ),
    "power_idle": (
        r"(^|[/_.-])(power_idle|power-idle|idle|no_cube|no-cube)($|[/_.-])",
    ),
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


class SafetyEvalCallback(EvalCallback):
    """Select checkpoints with a robust reward statistic and failure cost.

    Mineral returns are heavy-tailed: a single rare hotspot can dominate a
    small evaluation mean and save an unsafe policy. This callback keeps the
    standard SB3 evaluation logs, but optionally ranks checkpoints by median
    return minus a cost for out-of-power and flipped episodes.
    """

    _FAILURE_REASONS = ("out_of_power", "flipped")

    def __init__(
        self,
        eval_env,
        *,
        best_model_save_path: str,
        best_vecnorm_path: str,
        selection_stat: str = "mean",
        failure_penalty: float = 0.0,
        primary_selection_name: str = "primary",
        additional_selection_envs: list[tuple[str, object]] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            eval_env,
            best_model_save_path=None,
            callback_on_new_best=None,
            **kwargs,
        )
        self.selection_model_path = best_model_save_path
        self.selection_vecnorm_path = best_vecnorm_path
        self.selection_stat = selection_stat
        self.failure_penalty = failure_penalty
        self.primary_selection_name = primary_selection_name
        self.additional_selection_envs = additional_selection_envs or []
        self.best_selection_score = -np.inf
        self._evaluation_reasons: list[str] = []

    @staticmethod
    def selection_score(
        rewards: list[float],
        failure_count: int,
        selection_stat: str,
        failure_penalty: float,
    ) -> tuple[float, float, float]:
        values = np.asarray(rewards, dtype=np.float64)
        reward_stat = float(
            np.median(values) if selection_stat == "median" else np.mean(values)
        )
        failure_rate = failure_count / max(1, len(values))
        return reward_stat - failure_penalty * failure_rate, reward_stat, failure_rate

    @staticmethod
    def composite_selection_score(scores: dict[str, float]) -> float:
        """Require every selected scenario to clear the same safety bar."""
        if not scores:
            raise ValueError("checkpoint selection needs at least one scenario")
        return min(scores.values())

    def _log_success_callback(self, locals_: dict, globals_: dict) -> None:
        super()._log_success_callback(locals_, globals_)
        if locals_.get("done"):
            info = locals_.get("info", {})
            reason = info.get("game_over")
            if reason is None and info.get("TimeLimit.truncated", False):
                reason = "time_limit"
            self._evaluation_reasons.append(reason or "other")

    def _on_step(self) -> bool:
        evaluating = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0
        if evaluating:
            self._evaluation_reasons = []

        continue_training = super()._on_step()
        if not evaluating:
            return continue_training

        primary_rewards = [float(value) for value in self.evaluations_results[-1]]
        primary_reasons = self._evaluation_reasons.copy()
        failure_count = sum(
            reason in self._FAILURE_REASONS for reason in self._evaluation_reasons
        )
        score, reward_stat, failure_rate = self.selection_score(
            primary_rewards,
            failure_count,
            self.selection_stat,
            self.failure_penalty,
        )
        scenario_scores = {self.primary_selection_name: score}
        scenario_metrics = {
            self.primary_selection_name: (reward_stat, failure_rate, score)
        }

        for name, env in self.additional_selection_envs:
            if self.model.get_vec_normalize_env() is not None:
                sync_envs_normalization(self.training_env, env)
            self._evaluation_reasons = []
            rewards, _lengths = evaluate_policy(
                self.model,
                env,
                n_eval_episodes=self.n_eval_episodes,
                render=False,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )
            failures = sum(
                reason in self._FAILURE_REASONS for reason in self._evaluation_reasons
            )
            scenario_score, scenario_stat, scenario_failure_rate = self.selection_score(
                [float(value) for value in rewards],
                failures,
                self.selection_stat,
                self.failure_penalty,
            )
            scenario_scores[name] = scenario_score
            scenario_metrics[name] = (
                scenario_stat,
                scenario_failure_rate,
                scenario_score,
            )

        self._evaluation_reasons = primary_reasons
        score = self.composite_selection_score(scenario_scores)
        self.logger.record("eval/selection_score", score)
        for name, (
            scenario_stat,
            scenario_failure_rate,
            scenario_score,
        ) in scenario_metrics.items():
            self.logger.record(f"eval_selection/{name}_reward_stat", scenario_stat)
            self.logger.record(
                f"eval_selection/{name}_failure_rate", scenario_failure_rate
            )
            self.logger.record(f"eval_selection/{name}_score", scenario_score)
        print(
            "Checkpoint selection: "
            + ", ".join(
                f"{name} {self.selection_stat}={stat:.2f} "
                f"failure_rate={rate:.2%} score={scenario_score:.2f}"
                for name, (stat, rate, scenario_score) in scenario_metrics.items()
            )
            + f"; composite={score:.2f}"
        )

        if score > self.best_selection_score:
            os.makedirs(self.selection_model_path, exist_ok=True)
            self.model.save(os.path.join(self.selection_model_path, "best_model"))
            vn = self.model.get_vec_normalize_env()
            if vn is not None:
                os.makedirs(os.path.dirname(self.selection_vecnorm_path), exist_ok=True)
                vn.save(self.selection_vecnorm_path)
            self.best_selection_score = score
            print("New best safety-adjusted checkpoint!")
        return continue_training


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
    env_kwargs: dict | list[dict],
):
    """Vectorized env for the given stage.

    `n_envs == 1` uses `DummyVecEnv` (no process overhead). `n_envs > 1`
    uses `SubprocVecEnv` — separate processes, because the Bevy `App` is
    `!Send` so thread-based parallelism (DummyVecEnv) can't run sims
    concurrently. Each worker gets a distinct seed so they explore
    different terrain. `start_method="spawn"` is required: forking after
    Bevy has spawned its task-pool threads is unsafe.
    """
    env_kwargs_by_worker = (
        env_kwargs
        if isinstance(env_kwargs, list)
        else [env_kwargs for _ in range(max(1, n_envs))]
    )
    fns = []
    for i in range(n_envs):
        worker_kwargs = dict(env_kwargs_by_worker[i % len(env_kwargs_by_worker)])
        fns.append(
            _env_thunk(
                stage,
                base_seed + i,
                max_steps,
                frame_skip,
                power_capacity,
                worker_kwargs,
            )
        )
    if n_envs == 1:
        return DummyVecEnv(fns)
    return SubprocVecEnv(fns, start_method="spawn")


def _env_kwargs_from_args(
    args: argparse.Namespace, scenario: str | None = None
) -> dict:
    terrain_height = args.terrain_height
    if terrain_height is None and args.stage in (
        "locomotion",
        "cube_intercept",
        "power_idle",
        "power_cubes",
        "minerals",
    ):
        terrain_height = "mixed_1_2"
    cube_shaping_override = None if args.cube_shaping == "auto" else args.cube_shaping
    locomotion_shaping = args.locomotion_shaping
    if locomotion_shaping == "auto":
        locomotion_shaping = (
            "power_efficiency"
            if args.stage in ("locomotion", "power_idle", "power_cubes", "minerals")
            else "off"
        )
    cfg = apply_scenario_defaults(
        scenario if scenario is not None else args.scenario,
        cube_spawn_preset=args.cube_spawn_preset,
        power_start_fraction=args.power_start_fraction,
        terrain_height=terrain_height,
        cube_shaping=cube_shaping_override,
        forced_cube_distance=args.forced_cube_distance,
        forced_cube_bearing_deg=args.forced_cube_bearing_deg,
    )
    if "cube_shaping" not in cfg:
        if args.stage == "cube_intercept":
            cfg["cube_shaping"] = "intercept"
        elif args.stage == "power_cubes":
            cfg["cube_shaping"] = "low_power"
        else:
            cfg["cube_shaping"] = "off"
    terrain_scale, terrain_range = parse_terrain_height(cfg.pop("terrain_height", None))
    cfg["terrain_height_scale"] = terrain_scale
    cfg["terrain_height_scale_range"] = terrain_range
    cfg["cube_spawn_seed"] = args.cube_spawn_seed
    cfg["flip_penalty"] = args.flip_penalty
    cfg["tilt_penalty"] = args.tilt_penalty
    cfg["tilt_threshold_deg"] = args.tilt_threshold_deg
    cfg["mission_supervisor"] = args.mission_supervisor
    cfg["supervisor_low_power_enter_fraction"] = (
        args.supervisor_low_power_enter_fraction
    )
    cfg["supervisor_low_power_exit_fraction"] = args.supervisor_low_power_exit_fraction
    cfg["supervisor_path_safety_factor"] = args.supervisor_path_safety_factor
    cfg["supervisor_reserve_distance_m"] = args.supervisor_reserve_distance_m
    cfg["supervisor_tilt_enter_deg"] = args.supervisor_tilt_enter_deg
    cfg["supervisor_tilt_exit_deg"] = args.supervisor_tilt_exit_deg
    cfg["supervisor_tilt_guard_min_speed_mps"] = (
        args.supervisor_tilt_guard_min_speed_mps
    )
    cfg["supervisor_target_loss_grace_decisions"] = (
        args.supervisor_target_loss_grace_decisions
    )
    cfg["supervisor_beacon_first_distance_m"] = args.supervisor_beacon_first_distance_m
    cfg["supervisor_beacon_spacing_m"] = args.supervisor_beacon_spacing_m
    cfg["supervisor_beacon_auto_deploy"] = args.supervisor_beacon_auto_deploy
    cfg["supervisor_beacon_surface_score_threshold"] = (
        args.supervisor_beacon_surface_score_threshold
    )
    cfg["rejected_beacon_penalty"] = args.rejected_beacon_penalty
    cfg["locomotion_shaping"] = locomotion_shaping
    cfg["locomotion_coast_bonus"] = args.locomotion_coast_bonus
    cfg["locomotion_power_draw_penalty"] = args.locomotion_power_draw_penalty
    cfg["locomotion_power_recovery_reward"] = args.locomotion_power_recovery_reward
    cfg["locomotion_out_of_power_penalty"] = args.locomotion_out_of_power_penalty
    cfg["low_power_no_target_throttle_penalty"] = (
        args.low_power_no_target_throttle_penalty
    )
    cfg["low_power_no_target_coast_reward"] = args.low_power_no_target_coast_reward
    cfg["low_power_visible_stall_throttle_penalty"] = (
        args.low_power_visible_stall_throttle_penalty
    )
    cfg["cube_progress_range_epsilon"] = args.cube_progress_range_epsilon
    cfg["cube_progress_bearing_epsilon_deg"] = args.cube_progress_bearing_epsilon_deg
    cfg["low_power_threshold"] = args.low_power_threshold
    cfg["cube_approach_reward"] = args.cube_approach_reward
    cfg["cube_heading_reward"] = args.cube_heading_reward
    cfg["ignored_cube_penalty"] = args.ignored_cube_penalty
    cfg["loss_of_sight_penalty"] = args.loss_of_sight_penalty
    cfg["intercept_failure_penalty"] = args.intercept_failure_penalty
    if "scenario" not in cfg:
        cfg["scenario"] = scenario if scenario is not None else args.scenario
    return cfg


def _parse_scenario_list(value: str) -> list[str]:
    scenarios = [s.strip() for s in value.split(",") if s.strip()]
    for scenario in scenarios:
        if scenario not in EVAL_SCENARIOS:
            raise SystemExit(f"unknown train scenario {scenario!r}")
    return scenarios


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


def _collect_teacher_dataset(
    args: argparse.Namespace,
    env_kwargs: dict,
    sample_count: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    env = make_staged_env(
        args.stage,
        seed=args.seed + 50_000,
        max_steps=args.max_steps,
        frame_skip=args.frame_skip,
        power_capacity=args.power_capacity,
        **env_kwargs,
    )
    if args.stage == "cube_intercept":
        teacher = CubeInterceptTeacher()
    elif args.stage == "power_idle":
        teacher = PowerIdleTeacher(low_power_threshold=args.low_power_threshold)
    elif args.stage == "minerals":
        teacher = MineralExploreTeacher(low_power_threshold=args.low_power_threshold)
    else:
        raise SystemExit(
            "--teacher-pretrain-samples is currently supported only for "
            "cube_intercept, power_idle, and minerals"
        )
    obs_buf: list[np.ndarray] = []
    action_buf: list[int] = []
    pickups = 0
    episodes = 0
    seed = args.seed + 60_000
    obs, info = env.reset(seed=seed)
    teacher.reset()
    while len(obs_buf) < sample_count:
        action = teacher.action(np.asarray(obs, dtype=np.float32))
        obs_buf.append(np.asarray(obs, dtype=np.float32).copy())
        action_buf.append(action)
        obs, _reward, terminated, truncated, info = env.step(action)
        picked_up = float(info.get("episode_cube_pickups", 0.0)) > 0.0
        if picked_up:
            pickups += 1
        if terminated or truncated or picked_up:
            episodes += 1
            seed += 1
            obs, info = env.reset(seed=seed)
            teacher.reset()
    env.close()
    stats = {
        "samples": float(len(obs_buf)),
        "episodes": float(max(episodes, 1)),
        "pickups": float(pickups),
        "pickup_rate": pickups / max(episodes, 1),
    }
    return np.asarray(obs_buf, dtype=np.float32), np.asarray(action_buf), stats


def _teacher_scenarios(args: argparse.Namespace) -> list[str]:
    if args.teacher_scenarios:
        return _parse_scenario_list(args.teacher_scenarios)
    if args.teacher_scenario is not None:
        return [args.teacher_scenario]
    if args.stage == "power_idle":
        return ["power_idle"]
    if args.stage == "minerals":
        return ["minerals_explore"]
    return ["cube_intercept"]


def _collect_teacher_dataset_for_scenarios(
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    scenarios = _teacher_scenarios(args)
    remaining = args.teacher_pretrain_samples
    samples_per_scenario = int(np.ceil(remaining / len(scenarios)))
    obs_chunks = []
    action_chunks = []
    total_episodes = 0.0
    total_pickups = 0.0
    for scenario in scenarios:
        sample_count = min(samples_per_scenario, remaining)
        if sample_count <= 0:
            break
        teacher_kwargs = _env_kwargs_from_args(args, scenario=scenario)
        obs, actions, stats = _collect_teacher_dataset(
            args, teacher_kwargs, sample_count
        )
        obs_chunks.append(obs)
        action_chunks.append(actions)
        total_episodes += stats["episodes"]
        total_pickups += stats["pickups"]
        remaining -= len(actions)
        print(
            f"Teacher scenario {scenario}: "
            f"samples={len(actions)} episodes={int(stats['episodes'])} "
            f"pickup_rate={stats['pickup_rate']:.2%}"
        )
    obs_all = np.concatenate(obs_chunks, axis=0)
    actions_all = np.concatenate(action_chunks, axis=0)
    if len(actions_all) > args.teacher_pretrain_samples:
        obs_all = obs_all[: args.teacher_pretrain_samples]
        actions_all = actions_all[: args.teacher_pretrain_samples]
    stats = {
        "samples": float(len(actions_all)),
        "episodes": float(max(total_episodes, 1.0)),
        "pickups": float(total_pickups),
        "pickup_rate": total_pickups / max(total_episodes, 1.0),
    }
    return obs_all, actions_all, stats


def _policy_accuracy(
    policy,
    obs_tensor: th.Tensor,
    action_tensor: th.Tensor,
    batch_size: int,
) -> float:
    correct = 0
    with th.no_grad():
        for start in range(0, len(action_tensor), batch_size):
            end = start + batch_size
            dist = policy.get_distribution(obs_tensor[start:end])
            pred = dist.distribution.logits.argmax(dim=1)
            correct += int((pred == action_tensor[start:end]).sum().item())
    return correct / max(1, len(action_tensor))


def _teacher_pretrain(model: PPO, args: argparse.Namespace) -> None:
    if args.teacher_pretrain_samples <= 0:
        return
    if args.stage not in ("cube_intercept", "power_idle", "minerals"):
        raise SystemExit(
            "--teacher-pretrain-samples is currently only for cube_intercept "
            "power_idle, and minerals"
        )
    obs, actions, stats = _collect_teacher_dataset_for_scenarios(args)
    if (
        args.stage == "cube_intercept"
        and stats["pickup_rate"] < args.teacher_min_pickup_rate
    ):
        raise SystemExit(
            "teacher pickup rate below required threshold: "
            f"{stats['pickup_rate']:.2%} < {args.teacher_min_pickup_rate:.2%}"
        )
    vn = model.get_vec_normalize_env()
    if vn is not None:
        vn.obs_rms.update(obs)
        obs = vn.normalize_obs(obs)
    print(
        "Teacher dataset: "
        f"samples={len(obs)} episodes={int(stats['episodes'])} "
        f"pickup_rate={stats['pickup_rate']:.2%}"
    )
    action_counts = Counter(int(action) for action in actions.tolist())
    print(f"Teacher action counts: {dict(sorted(action_counts.items()))}")

    rng = np.random.default_rng(args.seed)
    policy = model.policy
    policy.set_training_mode(True)
    obs_tensor = th.as_tensor(obs, dtype=th.float32, device=policy.device)
    action_tensor = th.as_tensor(actions, dtype=th.long, device=policy.device)
    n = len(actions)
    old_lrs = [group["lr"] for group in policy.optimizer.param_groups]
    for group in policy.optimizer.param_groups:
        group["lr"] = args.teacher_pretrain_learning_rate
    for epoch in range(args.teacher_pretrain_epochs):
        order = rng.permutation(n)
        losses = []
        for start in range(0, n, args.teacher_pretrain_batch_size):
            idx = order[start : start + args.teacher_pretrain_batch_size]
            dist = policy.get_distribution(obs_tensor[idx])
            logits = dist.distribution.logits
            loss = F.cross_entropy(logits, action_tensor[idx])
            policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            policy.optimizer.step()
            losses.append(float(loss.detach().cpu()))
        accuracy = _policy_accuracy(
            policy, obs_tensor, action_tensor, args.teacher_pretrain_batch_size
        )
        print(
            f"teacher_pretrain epoch={epoch + 1}/{args.teacher_pretrain_epochs} "
            f"loss={float(np.mean(losses)):.4f} accuracy={accuracy:.2%}"
        )
    for group, lr in zip(policy.optimizer.param_groups, old_lrs):
        group["lr"] = lr


def _save_model_bundle(
    model: PPO,
    venv: VecNormalize,
    save_dir: str,
    *,
    save_best_copy: bool = False,
) -> None:
    model.save(os.path.join(save_dir, "model.zip"))
    venv.save(os.path.join(save_dir, "vecnorm.pkl"))
    if not save_best_copy:
        return
    best_dir = os.path.join(save_dir, "best")
    os.makedirs(best_dir, exist_ok=True)
    model.save(os.path.join(best_dir, "best_model.zip"))
    venv.save(os.path.join(best_dir, "vecnorm.pkl"))


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
    p.add_argument(
        "--eval-freq",
        type=int,
        default=25_000,
        help=(
            "EvalCallback frequency in vector-env step calls; total "
            "timesteps between evals are roughly eval_freq * n_envs"
        ),
    )
    p.add_argument("--n-eval-episodes", type=int, default=10)
    p.add_argument(
        "--checkpoint-freq",
        type=int,
        default=100_000,
        help=(
            "Checkpoint frequency in vector-env step calls; total timesteps "
            "between checkpoints are roughly checkpoint_freq * n_envs"
        ),
    )
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
        "--train-scenarios",
        type=str,
        default="",
        help=(
            "comma-separated scenario names to rotate across vectorized "
            "training workers; defaults to --scenario"
        ),
    )
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
        choices=("auto",) + CUBE_SHAPING_MODES,
        default="auto",
        help=("Cube approach shaping; auto enables low_power for power_cubes"),
    )
    p.add_argument(
        "--low-power-threshold",
        type=float,
        default=0.45,
        help="battery fraction where low_power cube shaping activates",
    )
    p.add_argument(
        "--cube-approach-reward",
        type=float,
        default=0.25,
        help="reward per meter of visible-cube range reduction during cube shaping",
    )
    p.add_argument(
        "--cube-heading-reward",
        type=float,
        default=0.05,
        help="intercept-mode reward for keeping a visible cube near sensor boresight",
    )
    p.add_argument(
        "--ignored-cube-penalty",
        type=float,
        default=25.0,
        help="out-of-power penalty when a visible cube was available under cube shaping",
    )
    p.add_argument(
        "--loss-of-sight-penalty",
        type=float,
        default=5.0,
        help="cube_intercept penalty when a visible forced cube is lost before pickup",
    )
    p.add_argument(
        "--intercept-failure-penalty",
        type=float,
        default=50.0,
        help="cube_intercept terminal/truncation penalty when no cube was picked up",
    )
    p.add_argument(
        "--locomotion-shaping",
        choices=("auto",) + LOCOMOTION_SHAPING_MODES,
        default="auto",
        help=(
            "Power pacing shaping; auto enables power_efficiency for "
            "locomotion, power_idle, power_cubes, and minerals"
        ),
    )
    p.add_argument(
        "--flip-penalty",
        type=float,
        default=50.0,
        help="terminal penalty when the rover flips",
    )
    p.add_argument(
        "--tilt-penalty",
        type=float,
        default=0.0,
        help=(
            "dense maximum penalty per decision as pitch/roll approaches "
            "the 100-degree flip threshold"
        ),
    )
    p.add_argument(
        "--tilt-threshold-deg",
        type=float,
        default=45.0,
        help="pitch/roll magnitude where dense tilt shaping begins",
    )
    p.add_argument(
        "--mission-supervisor",
        action="store_true",
        help="apply the shared reachability/intercept/tilt supervisor",
    )
    p.add_argument("--supervisor-low-power-enter-fraction", type=float, default=0.35)
    p.add_argument("--supervisor-low-power-exit-fraction", type=float, default=0.50)
    p.add_argument("--supervisor-path-safety-factor", type=float, default=1.10)
    p.add_argument("--supervisor-reserve-distance-m", type=float, default=2.0)
    p.add_argument("--supervisor-tilt-enter-deg", type=float, default=20.0)
    p.add_argument("--supervisor-tilt-exit-deg", type=float, default=18.0)
    p.add_argument("--supervisor-tilt-guard-min-speed-mps", type=float, default=1.0)
    p.add_argument("--supervisor-target-loss-grace-decisions", type=int, default=0)
    p.add_argument("--supervisor-beacon-first-distance-m", type=float, default=100.0)
    p.add_argument("--supervisor-beacon-spacing-m", type=float, default=75.0)
    p.add_argument(
        "--supervisor-beacon-auto-deploy",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--supervisor-beacon-surface-score-threshold", type=float, default=150.0
    )
    p.add_argument("--rejected-beacon-penalty", type=float, default=5.0)
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
    p.add_argument(
        "--low-power-no-target-throttle-penalty",
        type=float,
        default=0.25,
        help=(
            "extra per-decision penalty for motor actions when power is low "
            "and no actionable cube is visible"
        ),
    )
    p.add_argument(
        "--low-power-no-target-coast-reward",
        type=float,
        default=0.02,
        help=(
            "small per-decision reward for coast/no-op actions when power is "
            "low and no actionable cube is visible"
        ),
    )
    p.add_argument(
        "--low-power-visible-stall-throttle-penalty",
        type=float,
        default=0.0,
        help=(
            "extra penalty for a low-power motor action when the nearest "
            "visible cube neither closes nor becomes better aligned"
        ),
    )
    p.add_argument(
        "--cube-progress-range-epsilon",
        type=float,
        default=0.1,
        help="minimum visible-cube range reduction that counts as progress",
    )
    p.add_argument(
        "--cube-progress-bearing-epsilon-deg",
        type=float,
        default=1.0,
        help="minimum visible-cube bearing reduction that counts as progress",
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
        "--n-steps",
        type=int,
        default=2048,
        help="PPO rollout steps per worker before each update",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="PPO minibatch size; should divide n_steps * n_envs when possible",
    )
    p.add_argument(
        "--n-epochs",
        type=int,
        default=10,
        help="PPO epochs per rollout; lower = smaller, more conservative updates",
    )
    p.add_argument(
        "--clip-range",
        type=float,
        default=0.2,
        help="PPO policy clipping range; lower values preserve a warm-start policy more strongly",
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
        "--eval-selection-stat",
        choices=("mean", "median"),
        default="mean",
        help="episode-return statistic used to select the primary best checkpoint",
    )
    p.add_argument(
        "--eval-failure-penalty",
        type=float,
        default=0.0,
        help=(
            "checkpoint-selection cost multiplied by the primary eval's "
            "out-of-power plus flip rate"
        ),
    )
    p.add_argument(
        "--selection-extra-scenarios",
        type=str,
        default="",
        help=(
            "comma-separated scenarios that must also pass the primary "
            "checkpoint-selection score"
        ),
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
    p.add_argument(
        "--teacher-pretrain-samples",
        type=int,
        default=0,
        help=(
            "collect this many teacher observations before PPO "
            "(cube_intercept, power_idle, or minerals)"
        ),
    )
    p.add_argument(
        "--teacher-pretrain-epochs",
        type=int,
        default=20,
        help="supervised epochs over the teacher dataset before PPO",
    )
    p.add_argument(
        "--teacher-pretrain-batch-size",
        type=int,
        default=256,
        help="batch size for teacher cross-entropy pretraining",
    )
    p.add_argument(
        "--teacher-scenario",
        choices=EVAL_SCENARIOS.keys(),
        default=None,
        help="scenario used to collect teacher labels; defaults by stage",
    )
    p.add_argument(
        "--teacher-scenarios",
        type=str,
        default="",
        help=(
            "comma-separated teacher scenarios; overrides --teacher-scenario "
            "when set"
        ),
    )
    p.add_argument(
        "--teacher-pretrain-learning-rate",
        type=float,
        default=1e-3,
        help="optimizer learning rate used only during teacher cross-entropy pretraining",
    )
    p.add_argument(
        "--teacher-min-pickup-rate",
        type=float,
        default=0.80,
        help=(
            "abort cube_intercept pretraining if the empirical teacher "
            "pickup rate is below this"
        ),
    )
    p.add_argument(
        "--teacher-pretrain-only",
        action="store_true",
        help="save the behavior-cloned checkpoint and exit before PPO learn()",
    )
    args = p.parse_args()
    if args.horizon is not None:
        args.max_steps = HORIZON_STEPS[args.horizon]
    if args.teacher_pretrain_only and args.teacher_pretrain_samples <= 0:
        raise SystemExit(
            "--teacher-pretrain-only requires --teacher-pretrain-samples > 0"
        )

    os.makedirs(args.save, exist_ok=True)
    env_kwargs = _env_kwargs_from_args(args)
    train_scenarios = _parse_scenario_list(args.train_scenarios)
    if train_scenarios and args.n_envs < len(train_scenarios):
        raise SystemExit(
            "--n-envs must be at least the number of --train-scenarios; "
            f"got n_envs={args.n_envs} for {len(train_scenarios)} scenarios. "
            "Scenarios are assigned per vector-env worker."
        )
    train_env_kwargs = (
        [_env_kwargs_from_args(args, scenario=scenario) for scenario in train_scenarios]
        if train_scenarios
        else env_kwargs
    )

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
        train_env_kwargs,
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
    selection_extra_scenarios = _parse_scenario_list(args.selection_extra_scenarios)
    if args.scenario in selection_extra_scenarios:
        raise SystemExit("--selection-extra-scenarios must not repeat --scenario")
    selection_extra_envs = []
    for index, scenario in enumerate(selection_extra_scenarios):
        extra_kwargs = _env_kwargs_from_args(args, scenario=scenario)
        selection_extra_envs.append(
            (
                scenario,
                VecNormalize(
                    _make_vec_env(
                        args.stage,
                        args.seed + 1_500 + index,
                        args.max_steps,
                        args.frame_skip,
                        1,
                        args.power_capacity,
                        extra_kwargs,
                    ),
                    norm_obs=True,
                    norm_reward=False,
                    training=False,
                ),
            )
        )

    # --- Model: warm-start or fresh ---------------------------------------
    tb_dir = os.path.join(args.save, "tb")
    if args.load:
        # CLI flags are authoritative on a resume. Pass the knobs through
        # PPO.load() kwargs so SB3 rebuilds the rollout buffer with the new
        # n_steps/batch_size instead of leaving stale saved-buffer sizing.
        model = PPO.load(
            args.load,
            env=venv,
            tensorboard_log=tb_dir,
            device="cpu",
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            target_kl=args.target_kl,
        )
        print(f"Warm-started policy from {args.load}")
    else:
        model = PPO(
            "MlpPolicy",
            venv,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            learning_rate=args.learning_rate,
            clip_range=args.clip_range,
            gamma=0.99,
            ent_coef=args.ent_coef,
            target_kl=args.target_kl,
            tensorboard_log=tb_dir,
            verbose=1,
            device="cpu",
        )
    print(
        f"PPO: learning_rate={args.learning_rate} n_steps={args.n_steps} "
        f"batch_size={args.batch_size} n_epochs={args.n_epochs} "
        f"clip_range={args.clip_range} ent_coef={args.ent_coef} "
        f"target_kl={args.target_kl}"
    )
    print(f"env: power_capacity={args.power_capacity} Wh, frame_skip={args.frame_skip}")
    print(f"env kwargs: {env_kwargs}")
    if train_scenarios:
        print(f"train scenarios: {train_scenarios}")
    print(f"env binary: {hylaeanrover._native.__file__}")

    callbacks = [
        EpisodeOutcomeCallback(),
        CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=os.path.join(args.save, "checkpoints"),
            name_prefix=f"ppo_{args.stage}",
        ),
        SafetyEvalCallback(
            eval_venv,
            best_model_save_path=os.path.join(args.save, "best"),
            best_vecnorm_path=os.path.join(args.save, "best", "vecnorm.pkl"),
            log_path=os.path.join(args.save, "eval"),
            eval_freq=args.eval_freq,
            n_eval_episodes=args.n_eval_episodes,
            deterministic=True,
            selection_stat=args.eval_selection_stat,
            failure_penalty=args.eval_failure_penalty,
            primary_selection_name=args.scenario or "primary",
            additional_selection_envs=selection_extra_envs,
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

    _teacher_pretrain(model, args)
    if args.teacher_pretrain_only:
        _save_model_bundle(model, venv, args.save, save_best_copy=True)
        print(f"\nSaved behavior-cloned model + vecnorm to {args.save}/")
        print("Stopped before PPO because --teacher-pretrain-only was set.")
        return

    model.learn(total_timesteps=args.timesteps, callback=callbacks)

    # --- Persist for the next stage + for evaluate.py ---------------------
    _save_model_bundle(model, venv, args.save)
    print(f"\nSaved model + vecnorm to {args.save}/")
    print(f"Next stage: --load {args.save}/model.zip --vecnorm {args.save}/vecnorm.pkl")


if __name__ == "__main__":
    main()
