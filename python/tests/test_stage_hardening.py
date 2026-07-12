import unittest

import gymnasium as gym
import numpy as np

from examples.train import SafetyEvalCallback

from hylaeanrover import OBS_DIM, RoverEnv
from hylaeanrover.teacher import (
    CUBE_OBS_START,
    POWER_OBS_INDEX,
    CubeInterceptTeacher,
    MineralExploreTeacher,
    PowerIdleTeacher,
)
from hylaeanrover.wrappers import (
    CUBE_SHAPING_MODES,
    CUBE_SPAWN_PRESETS,
    LOCOMOTION_SHAPING_MODES,
    MissionSupervisorWrapper,
    STAGE_WEIGHTS,
    StagedRewardWrapper,
    apply_scenario_defaults,
    make_staged_env,
    parse_terrain_height,
)


class StageHardeningPresetTests(unittest.TestCase):
    def test_safety_eval_selection_resists_reward_outliers_and_failures(self) -> None:
        score, reward_stat, failure_rate = SafetyEvalCallback.selection_score(
            [100.0, 110.0, 120.0, 80_000.0],
            failure_count=1,
            selection_stat="median",
            failure_penalty=10_000.0,
        )
        self.assertEqual(reward_stat, 115.0)
        self.assertEqual(failure_rate, 0.25)
        self.assertEqual(score, -2385.0)

    def test_safety_eval_selection_uses_the_worst_transition_scenario(self) -> None:
        score = SafetyEvalCallback.composite_selection_score(
            {"minerals_transition": 3_000.0, "transition": 1_500.0}
        )
        self.assertEqual(score, 1_500.0)

    def test_named_presets_cover_dense_sparse_and_no_cube_controls(self) -> None:
        self.assertGreater(CUBE_SPAWN_PRESETS["dense_training"]["lambda"], 1.0)
        self.assertEqual(CUBE_SPAWN_PRESETS["bridge_training"]["lambda"], 0.75)
        self.assertEqual(CUBE_SPAWN_PRESETS["transition"]["lambda"], 0.30)
        self.assertEqual(CUBE_SPAWN_PRESETS["sparse_game"]["lambda"], 0.05)
        self.assertEqual(CUBE_SPAWN_PRESETS["none"]["lambda"], 0.0)
        self.assertIn("intercept", CUBE_SHAPING_MODES)
        self.assertIn("power_efficiency", LOCOMOTION_SHAPING_MODES)

    def test_power_cubes_distance_weight_stays_below_cube_weight(self) -> None:
        self.assertLess(
            STAGE_WEIGHTS["power_cubes"]["distance"],
            STAGE_WEIGHTS["power_cubes"]["cube"],
        )
        self.assertEqual(STAGE_WEIGHTS["power_idle"]["distance"], 0.0)
        self.assertEqual(STAGE_WEIGHTS["power_idle"]["cube"], 0.0)

    def test_minerals_stage_does_not_directly_reward_cube_pickups(self) -> None:
        self.assertEqual(STAGE_WEIGHTS["minerals"]["cube"], 0.0)
        self.assertGreater(STAGE_WEIGHTS["minerals"]["distance"], 0.0)
        self.assertGreater(STAGE_WEIGHTS["minerals"]["mineral"], 0.0)

    def test_coverage_scenario_enables_versioned_observation(self) -> None:
        cfg = apply_scenario_defaults("minerals_coverage")
        self.assertTrue(cfg["coverage_observation"])
        self.assertEqual(cfg["cube_spawn_preset"], "sparse_game")

    def test_scenario_defaults_can_be_overridden(self) -> None:
        cfg = apply_scenario_defaults(
            "low_power_start",
            cube_spawn_preset="sparse_game",
            power_start_fraction=0.25,
        )
        self.assertEqual(cfg["cube_spawn_preset"], "sparse_game")
        self.assertEqual(cfg["power_start_fraction"], 0.25)

    def test_bridge_low_power_scenario_uses_bridge_spawn_density(self) -> None:
        cfg = apply_scenario_defaults("bridge_low_power")
        self.assertEqual(cfg["cube_spawn_preset"], "bridge_training")
        self.assertEqual(cfg["power_start_fraction"], 0.35)

    def test_sparse_low_power_scenario_uses_sparse_spawn_density(self) -> None:
        cfg = apply_scenario_defaults("sparse_low_power")
        self.assertEqual(cfg["cube_spawn_preset"], "sparse_game")
        self.assertEqual(cfg["power_start_fraction"], 0.35)

    def test_sparse_visible_scenarios_force_visible_reset_cube(self) -> None:
        cfg = apply_scenario_defaults("sparse_visible_reset")
        self.assertEqual(cfg["cube_spawn_preset"], "sparse_game")
        self.assertEqual(cfg["forced_cube_distance_range"], (30.0, 60.0))
        self.assertEqual(cfg["forced_cube_bearing_range"], (-35.0, 35.0))

        low_power_cfg = apply_scenario_defaults("sparse_visible_low_power")
        self.assertEqual(low_power_cfg["cube_spawn_preset"], "sparse_game")
        self.assertEqual(low_power_cfg["power_start_fraction"], 0.35)
        self.assertEqual(low_power_cfg["forced_cube_distance_range"], (30.0, 60.0))

    def test_cube_intercept_stage_defaults_to_forced_cube_only(self) -> None:
        cfg = apply_scenario_defaults("cube_intercept")
        self.assertEqual(cfg["cube_spawn_preset"], "none")
        self.assertEqual(cfg["forced_cube_distance_range"], (15.0, 60.0))
        self.assertEqual(cfg["forced_cube_bearing_range"], (-35.0, 35.0))
        self.assertEqual(cfg["cube_shaping"], "intercept")

        close_cfg = apply_scenario_defaults("cube_intercept_close")
        self.assertEqual(close_cfg["cube_spawn_preset"], "none")
        self.assertEqual(close_cfg["forced_cube_distance_range"], (15.0, 20.0))
        self.assertEqual(close_cfg["forced_cube_bearing_range"], (0.0, 5.0))

    def test_minerals_explore_scenario_has_no_power_cube_spawns(self) -> None:
        cfg = apply_scenario_defaults("minerals_explore")
        self.assertEqual(cfg["cube_spawn_preset"], "none")
        self.assertEqual(cfg["terrain_height"], "mixed_1_2")
        self.assertEqual(cfg["cube_shaping"], "off")

    def test_minerals_sparse_scenario_uses_sparse_survival_cubes(self) -> None:
        cfg = apply_scenario_defaults("minerals_sparse")
        self.assertEqual(cfg["cube_spawn_preset"], "sparse_game")
        self.assertEqual(cfg["terrain_height"], "mixed_1_2")
        self.assertEqual(cfg["cube_shaping"], "low_power")

    def test_power_idle_scenario_is_low_power_no_cube_control(self) -> None:
        cfg = apply_scenario_defaults("power_idle")
        self.assertEqual(cfg["cube_spawn_preset"], "none")
        self.assertEqual(cfg["power_start_fraction"], 0.35)
        self.assertEqual(cfg["cube_shaping"], "off")

    def test_terrain_height_parser_accepts_presets_ranges_and_fixed_values(
        self,
    ) -> None:
        self.assertEqual(parse_terrain_height("fixed_1_5"), (1.5, None))
        self.assertEqual(parse_terrain_height("1.0:2.0"), (None, (1.0, 2.0)))
        self.assertEqual(parse_terrain_height("2.0"), (2.0, None))


class NativeResetInstrumentationTests(unittest.TestCase):
    def test_reset_applies_power_and_terrain_options_without_changing_obs_shape(
        self,
    ) -> None:
        env = RoverEnv(
            seed=7,
            max_steps=20,
            power_capacity=100.0,
            power_start_fraction=1.0,
            cube_spawn_lambda=0.0,
            terrain_height_scale=1.0,
        )
        try:
            obs, info = env.reset(
                seed=11,
                options={"power_start_fraction": 0.30, "terrain_height_scale": 1.50},
            )
            self.assertEqual(obs.shape, (OBS_DIM,))
            self.assertAlmostEqual(info["power_frac"], 0.30, places=3)
            self.assertAlmostEqual(info["power_start_fraction"], 0.30, places=3)
            self.assertAlmostEqual(info["terrain_height_scale"], 1.50, places=3)
            self.assertEqual(info["cube_spawn_lambda"], 0.0)
            self.assertIn("visible_cube_count", info)
            self.assertIn("nearest_visible_cube_range", info)
        finally:
            env.close()

    def test_reset_forced_cube_is_visible_without_changing_obs_shape(self) -> None:
        env = RoverEnv(
            seed=7,
            max_steps=20,
            power_capacity=100.0,
            power_start_fraction=0.35,
            cube_spawn_lambda=0.0,
            terrain_height_scale=1.0,
        )
        try:
            obs, info = env.reset(
                seed=11,
                options={
                    "forced_cube_distance": 35.0,
                    "forced_cube_bearing_deg": 0.0,
                },
            )
            self.assertEqual(obs.shape, (OBS_DIM,))
            self.assertGreaterEqual(info["visible_cube_count"], 1)
            self.assertAlmostEqual(info["forced_cube_distance"], 35.0, places=2)
            self.assertAlmostEqual(info["forced_cube_bearing_deg"], 0.0, places=2)
            self.assertTrue(info["nearest_cube_actionable"])
            self.assertIsNotNone(info["nearest_cube_height_above_ground_m"])
            self.assertLess(info["nearest_cube_height_above_ground_m"], 2.5)
            self.assertIsNotNone(info["nearest_visible_cube_range"])
            self.assertLess(abs(float(info["nearest_visible_cube_bearing"])), 5.0)
        finally:
            env.close()

    def test_teacher_picks_up_forced_cubes_with_training_frame_skip(self) -> None:
        cases = [
            (42, 15.0, -35.0),
            (43, 15.0, 35.0),
            (44, 30.0, -20.0),
            (45, 30.0, 20.0),
            (46, 60.0, -35.0),
            (47, 60.0, 35.0),
            (48, 60.0, 0.0),
            (49, 45.0, -10.0),
            (50, 45.0, 10.0),
        ]
        for seed, distance, bearing in cases:
            with self.subTest(seed=seed, distance=distance, bearing=bearing):
                env = make_staged_env(
                    "cube_intercept",
                    seed=seed,
                    max_steps=2400,
                    frame_skip=4,
                    power_capacity=100.0,
                    cube_spawn_preset="none",
                    forced_cube_distance=distance,
                    forced_cube_bearing_deg=bearing,
                    terrain_height_scale=1.0,
                )
                teacher = CubeInterceptTeacher()
                try:
                    obs, info = env.reset(seed=seed)
                    teacher.reset()
                    self.assertTrue(info["nearest_cube_actionable"])
                    self.assertGreaterEqual(info["visible_cube_count"], 1)
                    for _ in range(600):
                        obs, _reward, terminated, truncated, info = env.step(
                            teacher.action(obs)
                        )
                        if info["episode_cube_pickups"] > 0:
                            break
                        if terminated or truncated:
                            break
                    self.assertGreater(info["episode_cube_pickups"], 0)
                finally:
                    env.close()

    def test_teacher_commits_forward_after_losing_visible_cube(self) -> None:
        teacher = CubeInterceptTeacher()
        visible = np.zeros(OBS_DIM, dtype=np.float32)
        visible[CUBE_OBS_START] = 0.0
        visible[CUBE_OBS_START + 1] = 30.0
        visible[CUBE_OBS_START + 2] = 1.0

        self.assertEqual(teacher.action(visible), 7)
        self.assertEqual(teacher.action(np.zeros(OBS_DIM, dtype=np.float32)), 7)
        self.assertEqual(
            CubeInterceptTeacher().action(np.zeros(OBS_DIM, dtype=np.float32)), 4
        )

    def test_power_idle_teacher_coasts_without_visible_cube(self) -> None:
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[POWER_OBS_INDEX] = 0.30
        self.assertEqual(PowerIdleTeacher().action(obs), 4)

    def test_power_idle_teacher_preserves_visible_cube_intercept(self) -> None:
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[POWER_OBS_INDEX] = 0.30
        obs[CUBE_OBS_START] = 0.0
        obs[CUBE_OBS_START + 1] = 30.0
        obs[CUBE_OBS_START + 2] = 1.0
        self.assertEqual(PowerIdleTeacher().action(obs), 7)

    def test_mineral_explore_teacher_drives_when_power_is_healthy(self) -> None:
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[POWER_OBS_INDEX] = 0.90
        self.assertEqual(MineralExploreTeacher().action(obs), 7)

    def test_mineral_explore_teacher_ignores_cubes_while_power_is_healthy(
        self,
    ) -> None:
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[POWER_OBS_INDEX] = 0.90
        obs[CUBE_OBS_START] = 25.0
        obs[CUBE_OBS_START + 1] = 30.0
        obs[CUBE_OBS_START + 2] = 1.0
        self.assertEqual(MineralExploreTeacher().action(obs), 7)

    def test_mineral_explore_teacher_intercepts_visible_cube_while_recovering(
        self,
    ) -> None:
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[POWER_OBS_INDEX] = 0.30
        obs[CUBE_OBS_START] = 25.0
        obs[CUBE_OBS_START + 1] = 30.0
        obs[CUBE_OBS_START + 2] = 1.0
        self.assertEqual(MineralExploreTeacher().action(obs), 6)

    def test_mineral_explore_teacher_recovers_until_resume_threshold(self) -> None:
        teacher = MineralExploreTeacher(
            low_power_threshold=0.45,
            resume_power_threshold=0.65,
        )
        obs = np.zeros(OBS_DIM, dtype=np.float32)

        obs[POWER_OBS_INDEX] = 0.40
        self.assertEqual(teacher.action(obs), 4)

        obs[POWER_OBS_INDEX] = 0.60
        self.assertEqual(teacher.action(obs), 4)

        obs[POWER_OBS_INDEX] = 0.70
        self.assertEqual(teacher.action(obs), 7)

    def test_mineral_explore_teacher_reset_clears_recovery_state(self) -> None:
        teacher = MineralExploreTeacher()
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[POWER_OBS_INDEX] = 0.30
        self.assertEqual(teacher.action(obs), 4)

        teacher.reset()
        obs[POWER_OBS_INDEX] = 0.50
        self.assertEqual(teacher.action(obs), 7)

    def test_teacher_handles_previous_sparse_visible_low_power_failure(self) -> None:
        cfg = apply_scenario_defaults(
            "sparse_visible_low_power",
            terrain_height="mixed_1_2",
            cube_shaping="off",
        )
        terrain_scale, terrain_range = parse_terrain_height(cfg.pop("terrain_height"))
        cfg["terrain_height_scale"] = terrain_scale
        cfg["terrain_height_scale_range"] = terrain_range
        cfg["cube_spawn_seed"] = None
        env = make_staged_env(
            "cube_intercept",
            seed=142,
            max_steps=7200,
            frame_skip=4,
            power_capacity=100.0,
            **cfg,
        )
        teacher = CubeInterceptTeacher()
        try:
            obs, info = env.reset(seed=142)
            teacher.reset()
            self.assertGreaterEqual(info["visible_cube_count"], 1)
            for _ in range(600):
                obs, _reward, terminated, truncated, info = env.step(
                    teacher.action(obs)
                )
                if info["episode_cube_pickups"] > 0:
                    break
                if terminated or truncated:
                    break
            self.assertGreater(info["episode_cube_pickups"], 0)
            self.assertEqual(info["game_over"], "cube_picked_up")
        finally:
            env.close()

    def test_cube_intercept_no_cube_control_does_not_force_cube(self) -> None:
        env = make_staged_env(
            "cube_intercept",
            seed=77,
            max_steps=20,
            frame_skip=4,
            power_capacity=100.0,
            scenario="no_cube_control",
            cube_spawn_preset="none",
            terrain_height_scale=1.0,
            cube_shaping="off",
        )
        try:
            _obs, info = env.reset(seed=77)
            self.assertEqual(info["visible_cube_count"], 0)
            self.assertIsNone(info["forced_cube_distance"])
            self.assertIsNone(info["forced_cube_bearing_deg"])
        finally:
            env.close()

    def test_power_idle_stage_defaults_to_low_power_no_cube(self) -> None:
        env = make_staged_env(
            "power_idle",
            seed=78,
            max_steps=20,
            frame_skip=4,
            power_capacity=100.0,
            terrain_height_scale=1.0,
        )
        try:
            _obs, info = env.reset(seed=78)
            self.assertEqual(info["cube_spawn_preset"], "none")
            self.assertEqual(info["cube_spawn_lambda"], 0.0)
            self.assertAlmostEqual(info["power_start_fraction"], 0.35, places=3)
            self.assertAlmostEqual(info["power_frac"], 0.35, places=3)
            self.assertEqual(info["visible_cube_count"], 0)
        finally:
            env.close()


class _OneStepEnv(gym.Env):
    action_space = gym.spaces.Discrete(10)
    observation_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
    )

    def __init__(
        self,
        *,
        distance: float,
        power_frac: float,
        cube_bonus: float = 0.0,
        episode_cube_pickups: float = 0.0,
        game_over: str | None = None,
        reset_power_frac: float = 1.0,
        novel_distance: float = 0.0,
        novel_mineral: float = 0.0,
        supervisor_mode: str | None = None,
    ) -> None:
        self.distance = distance
        self.power_frac = power_frac
        self.cube_bonus = cube_bonus
        self.episode_cube_pickups = episode_cube_pickups
        self.game_over = game_over
        self.reset_power_frac = reset_power_frac
        self.novel_distance = novel_distance
        self.novel_mineral = novel_mineral
        self.supervisor_mode = supervisor_mode

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(OBS_DIM, dtype=np.float32), {
            "reward_distance": 0.0,
            "reward_cube_bonus": 0.0,
            "reward_mineral_integral": 0.0,
            "reward_beacon_bonus": 0.0,
            "reward_novel_distance": 0.0,
            "reward_novel_mineral_integral": 0.0,
            "power_frac": self.reset_power_frac,
        }

    def step(self, action):
        return (
            np.zeros(OBS_DIM, dtype=np.float32),
            0.0,
            self.game_over is not None,
            False,
            {
                "reward_distance": self.distance,
                "reward_cube_bonus": self.cube_bonus,
                "reward_mineral_integral": 0.0,
                "reward_beacon_bonus": 0.0,
                "reward_novel_distance": self.novel_distance,
                "reward_novel_mineral_integral": self.novel_mineral,
                "power_frac": self.power_frac,
                "episode_cube_pickups": self.episode_cube_pickups,
                "game_over": self.game_over,
                "supervisor_mode": self.supervisor_mode,
            },
        )


class _VisibleCubeEnv(_OneStepEnv):
    def __init__(
        self, *, power_frac: float, bearing: float, reset_power_frac: float = 1.0
    ) -> None:
        super().__init__(
            distance=0.0,
            power_frac=power_frac,
            reset_power_frac=reset_power_frac,
        )
        self.bearing = bearing

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info.update(
            {
                "visible_cube_count": 1,
                "nearest_visible_cube_range": 20.0,
                "nearest_visible_cube_bearing": self.bearing,
            }
        )
        return obs, reward, terminated, truncated, info


class _SupervisorEnv(gym.Env):
    action_space = gym.spaces.Discrete(10)
    observation_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
    )

    def __init__(
        self,
        *,
        power_frac: float,
        cube: tuple[float, float] | None = None,
        distance_m: float = 0.0,
        beacons_remaining: float = 5.0,
    ) -> None:
        self.power_frac = power_frac
        self.cube = cube
        self.distance_m = distance_m
        self.beacons_remaining = beacons_remaining
        self.last_action: int | None = None

    def _obs(self) -> np.ndarray:
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[POWER_OBS_INDEX] = self.power_frac
        obs[-1] = self.beacons_remaining
        if self.cube is not None:
            bearing, distance = self.cube
            obs[CUBE_OBS_START] = bearing
            obs[CUBE_OBS_START + 1] = distance
            obs[CUBE_OBS_START + 2] = 1.0
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.last_action = None
        return self._obs(), {
            "reward_distance": self.distance_m,
            "coverage_features": [index / 18.0 for index in range(18)],
        }

    def step(self, action):
        self.last_action = int(action)
        return (
            self._obs(),
            0.0,
            False,
            False,
            {
                "reward_distance": self.distance_m,
                "coverage_features": [index / 18.0 for index in range(18)],
            },
        )


class MissionSupervisorWrapperTests(unittest.TestCase):
    def test_low_power_without_cube_overrides_policy_to_coast(self) -> None:
        base = _SupervisorEnv(power_frac=0.30)
        env = MissionSupervisorWrapper(base, power_capacity_wh=100.0)
        env.reset()
        _, _, _, _, info = env.step(7)
        self.assertEqual(base.last_action, 4)
        self.assertTrue(info["supervisor_overrode"])
        self.assertEqual(info["supervisor_mode"], "preserve")

    def test_healthy_power_preserves_policy_action(self) -> None:
        base = _SupervisorEnv(power_frac=0.90)
        env = MissionSupervisorWrapper(base, power_capacity_wh=100.0)
        env.reset()
        _, _, _, _, info = env.step(2)
        self.assertEqual(base.last_action, 2)
        self.assertFalse(info["supervisor_overrode"])
        self.assertEqual(info["supervisor_mode"], "explore")

    def test_policy_never_sees_cubes_owned_by_supervisor(self) -> None:
        base = _SupervisorEnv(power_frac=0.99, cube=(25.0, 40.0))
        env = MissionSupervisorWrapper(base, power_capacity_wh=1000.0)
        obs, _ = env.reset()
        self.assertTrue(np.all(obs[CUBE_OBS_START:POWER_OBS_INDEX] == 0.0))
        self.assertEqual(obs[POWER_OBS_INDEX], np.float32(1.0))

        obs, _, _, _, _ = env.step(7)
        self.assertTrue(np.all(obs[CUBE_OBS_START:POWER_OBS_INDEX] == 0.0))

    def test_coverage_policy_sees_frontier_features_while_supervisor_keeps_cube(
        self,
    ) -> None:
        base = _SupervisorEnv(power_frac=0.30, cube=(15.0, 30.0))
        env = MissionSupervisorWrapper(
            base,
            power_capacity_wh=100.0,
            coverage_observation=True,
        )
        obs, _ = env.reset()
        np.testing.assert_allclose(
            obs[CUBE_OBS_START:POWER_OBS_INDEX],
            np.asarray([index / 18.0 for index in range(18)], dtype=np.float32),
        )
        _, _, _, _, info = env.step(4)
        self.assertEqual(info["supervisor_mode"], "intercept")
        self.assertEqual(base.last_action, 6)
        self.assertEqual(obs[POWER_OBS_INDEX], np.float32(1.0))

    def test_mid_power_remains_in_exploration_mode(self) -> None:
        base = _SupervisorEnv(power_frac=0.40)
        env = MissionSupervisorWrapper(base, power_capacity_wh=100.0)
        env.reset()
        _, _, _, _, info = env.step(7)
        self.assertEqual(base.last_action, 7)
        self.assertFalse(info["supervisor_overrode"])
        self.assertEqual(info["supervisor_mode"], "explore")

    def test_reachable_cube_uses_intercept_controller(self) -> None:
        base = _SupervisorEnv(power_frac=0.30, cube=(15.0, 30.0))
        env = MissionSupervisorWrapper(base, power_capacity_wh=100.0)
        obs, _ = env.reset()
        self.assertTrue(np.all(obs[CUBE_OBS_START:POWER_OBS_INDEX] == 0.0))
        _, _, _, _, info = env.step(4)
        self.assertEqual(base.last_action, 6)
        self.assertEqual(info["supervisor_mode"], "intercept")
        self.assertTrue(info["supervisor_target_viable"])

    def test_beacon_guard_requires_exploration_distance(self) -> None:
        early = _SupervisorEnv(power_frac=0.90, distance_m=50.0)
        early_env = MissionSupervisorWrapper(
            early,
            power_capacity_wh=100.0,
            beacon_guard_enabled=True,
        )
        early_env.reset()
        _, _, _, _, early_info = early_env.step(9)
        self.assertEqual(early.last_action, 4)
        self.assertEqual(early_info["supervisor_mode"], "beacon_hold")

        ready = _SupervisorEnv(power_frac=0.90, distance_m=100.0)
        ready_env = MissionSupervisorWrapper(
            ready,
            power_capacity_wh=100.0,
            beacon_guard_enabled=True,
        )
        ready_env.reset()
        _, _, _, _, ready_info = ready_env.step(9)
        self.assertEqual(ready.last_action, 9)
        self.assertEqual(ready_info["supervisor_mode"], "beacon_deploy")


class _VisibleCubeSequenceEnv(_OneStepEnv):
    def __init__(self, observations: list[tuple[float, float]]) -> None:
        super().__init__(distance=0.0, power_frac=0.30, reset_power_frac=0.30)
        self.observations = observations
        self.index = 0

    def reset(self, *, seed=None, options=None):
        self.index = 0
        return super().reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        nearest, bearing = self.observations[
            min(self.index, len(self.observations) - 1)
        ]
        self.index += 1
        info.update(
            {
                "visible_cube_count": 1,
                "nearest_visible_cube_range": nearest,
                "nearest_visible_cube_bearing": bearing,
            }
        )
        return obs, reward, terminated, truncated, info


class _TiltEnv(_OneStepEnv):
    def __init__(self, *, pitch: float = 0.0, roll: float = 0.0) -> None:
        super().__init__(distance=0.0, power_frac=1.0)
        self.pitch = pitch
        self.roll = roll

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        obs[2] = self.pitch
        obs[3] = self.roll
        return obs, reward, terminated, truncated, info


class LocomotionPowerShapingTests(unittest.TestCase):
    def test_coverage_reward_pays_new_ground_once_without_repeat_penalty(self) -> None:
        env = StagedRewardWrapper(
            _OneStepEnv(
                distance=10.0,
                power_frac=1.0,
                novel_distance=10.0,
                novel_mineral=20.0,
            ),
            stage="minerals",
            coverage_observation=True,
        )
        env.reset()
        _, first_reward, _, _, _ = env.step(7)
        _, repeat_reward, _, _, _ = env.step(7)
        self.assertAlmostEqual(first_reward, 30.0)
        self.assertEqual(repeat_reward, 0.0)

    def test_coverage_reward_is_not_attributed_to_supervisor_motion(self) -> None:
        env = StagedRewardWrapper(
            _OneStepEnv(
                distance=10.0,
                power_frac=0.30,
                novel_distance=10.0,
                novel_mineral=20.0,
                supervisor_mode="intercept",
            ),
            stage="minerals",
            coverage_observation=True,
        )
        env.reset()
        _, reward, _, _, _ = env.step(7)
        self.assertAlmostEqual(reward, 1.0)

    def test_tilt_shaping_ignores_normal_terrain_attitude(self) -> None:
        env = StagedRewardWrapper(
            _TiltEnv(pitch=30.0, roll=-40.0),
            stage="minerals",
            locomotion_shaping="off",
            tilt_penalty=5.0,
            tilt_threshold_deg=45.0,
        )
        env.reset()
        _, reward, _, _, info = env.step(7)
        self.assertEqual(reward, 0.0)
        self.assertEqual(info["tilt_penalty_total"], 0.0)

    def test_tilt_shaping_penalizes_preterminal_rollover_risk(self) -> None:
        env = StagedRewardWrapper(
            _TiltEnv(roll=100.0),
            stage="minerals",
            locomotion_shaping="off",
            tilt_penalty=5.0,
            tilt_threshold_deg=45.0,
        )
        env.reset()
        _, reward, _, _, info = env.step(7)
        self.assertAlmostEqual(reward, -5.0)
        self.assertAlmostEqual(info["tilt_penalty_total"], 5.0)

    def test_power_efficiency_rewards_coasting_distance(self) -> None:
        env = StagedRewardWrapper(
            _OneStepEnv(distance=10.0, power_frac=1.0),
            stage="locomotion",
            locomotion_shaping="power_efficiency",
        )
        env.reset()
        _, reward, _, _, info = env.step(4)
        self.assertGreater(reward, 10.0)
        self.assertEqual(info["locomotion_shaping"], "power_efficiency")

    def test_power_efficiency_penalizes_sprint_to_empty(self) -> None:
        env = StagedRewardWrapper(
            _OneStepEnv(distance=10.0, power_frac=0.0, game_over="out_of_power"),
            stage="locomotion",
            locomotion_shaping="power_efficiency",
        )
        env.reset()
        _, reward, terminated, _, _ = env.step(7)
        self.assertTrue(terminated)
        self.assertLess(reward, 0.0)

    def test_power_cubes_power_efficiency_penalizes_sprint_to_empty(self) -> None:
        env = StagedRewardWrapper(
            _OneStepEnv(distance=10.0, power_frac=0.0, game_over="out_of_power"),
            stage="power_cubes",
            locomotion_shaping="power_efficiency",
            cube_shaping="off",
        )
        env.reset()
        _, reward, terminated, _, info = env.step(7)
        self.assertTrue(terminated)
        self.assertLess(reward, 0.0)
        self.assertEqual(info["locomotion_shaping"], "power_efficiency")

    def test_minerals_power_efficiency_penalizes_sprint_to_empty(self) -> None:
        env = StagedRewardWrapper(
            _OneStepEnv(distance=10.0, power_frac=0.0, game_over="out_of_power"),
            stage="minerals",
            locomotion_shaping="power_efficiency",
            cube_shaping="off",
        )
        env.reset()
        _, reward, terminated, _, info = env.step(7)
        self.assertTrue(terminated)
        self.assertLess(reward, 0.0)
        self.assertEqual(info["locomotion_shaping"], "power_efficiency")

    def test_minerals_reward_ignores_raw_cube_bonus(self) -> None:
        env = StagedRewardWrapper(
            _OneStepEnv(distance=0.0, power_frac=1.0, cube_bonus=100.0),
            stage="minerals",
            locomotion_shaping="off",
            cube_shaping="off",
        )
        env.reset()
        _, reward, _, _, _ = env.step(7)
        self.assertEqual(reward, 0.0)

    def test_power_idle_penalizes_low_power_no_target_motor_action(self) -> None:
        env = StagedRewardWrapper(
            _OneStepEnv(distance=0.0, power_frac=0.30, reset_power_frac=0.30),
            stage="power_idle",
            locomotion_shaping="power_efficiency",
            low_power_no_target_throttle_penalty=0.5,
            low_power_no_target_coast_reward=0.1,
        )
        env.reset()
        _, reward, _, _, info = env.step(7)
        self.assertLess(reward, 0.0)
        self.assertEqual(info["low_power_no_target_steps"], 1)

    def test_power_idle_rewards_low_power_no_target_coast_action(self) -> None:
        env = StagedRewardWrapper(
            _OneStepEnv(distance=0.0, power_frac=0.30, reset_power_frac=0.30),
            stage="power_idle",
            locomotion_shaping="power_efficiency",
            low_power_no_target_throttle_penalty=0.5,
            low_power_no_target_coast_reward=0.1,
        )
        env.reset()
        _, reward, _, _, info = env.step(4)
        self.assertGreater(reward, 0.0)
        self.assertEqual(info["low_power_no_target_steps"], 1)

    def test_minerals_penalizes_low_power_no_target_motor_action(self) -> None:
        env = StagedRewardWrapper(
            _OneStepEnv(distance=0.0, power_frac=0.30, reset_power_frac=0.30),
            stage="minerals",
            locomotion_shaping="power_efficiency",
            low_power_no_target_throttle_penalty=0.5,
            low_power_no_target_coast_reward=0.1,
        )
        env.reset()
        _, reward, _, _, info = env.step(7)
        self.assertLess(reward, 0.0)
        self.assertEqual(info["low_power_no_target_steps"], 1)

    def test_power_cubes_no_target_penalty_ignores_visible_cube(self) -> None:
        env = StagedRewardWrapper(
            _VisibleCubeEnv(power_frac=0.30, bearing=0.0, reset_power_frac=0.30),
            stage="power_cubes",
            locomotion_shaping="power_efficiency",
            cube_shaping="off",
            low_power_no_target_throttle_penalty=0.5,
        )
        env.reset()
        _, reward, _, _, info = env.step(7)
        self.assertEqual(reward, 0.0)
        self.assertEqual(info["low_power_no_target_steps"], 0)

    def test_minerals_no_target_penalty_ignores_visible_cube(self) -> None:
        env = StagedRewardWrapper(
            _VisibleCubeEnv(power_frac=0.30, bearing=0.0, reset_power_frac=0.30),
            stage="minerals",
            locomotion_shaping="power_efficiency",
            cube_shaping="off",
            low_power_no_target_throttle_penalty=0.5,
        )
        env.reset()
        _, reward, _, _, info = env.step(7)
        self.assertEqual(reward, 0.0)
        self.assertEqual(info["low_power_no_target_steps"], 0)

    def test_low_power_visible_stall_penalizes_unproductive_motor_action(self) -> None:
        env = StagedRewardWrapper(
            _VisibleCubeSequenceEnv([(20.0, 15.0), (20.0, 15.0)]),
            stage="minerals",
            locomotion_shaping="power_efficiency",
            cube_shaping="off",
            low_power_visible_stall_throttle_penalty=0.5,
        )
        env.reset()
        _, first_reward, _, _, _ = env.step(7)
        _, second_reward, _, _, info = env.step(7)
        self.assertEqual(first_reward, 0.0)
        self.assertEqual(second_reward, -0.5)
        self.assertEqual(info["low_power_visible_stall_steps"], 1)
        self.assertEqual(info["low_power_visible_stall_penalty_total"], 0.5)

    def test_low_power_visible_progress_preserves_committed_intercept(self) -> None:
        env = StagedRewardWrapper(
            _VisibleCubeSequenceEnv([(20.0, 20.0), (20.0, 15.0), (19.5, 15.0)]),
            stage="minerals",
            locomotion_shaping="power_efficiency",
            cube_shaping="off",
            low_power_visible_stall_throttle_penalty=0.5,
        )
        env.reset()
        env.step(7)
        _, turn_reward, _, _, turn_info = env.step(7)
        _, approach_reward, _, _, approach_info = env.step(7)
        self.assertEqual(turn_reward, 0.0)
        self.assertEqual(approach_reward, 0.0)
        self.assertEqual(turn_info["low_power_visible_stall_steps"], 0)
        self.assertEqual(approach_info["low_power_visible_stall_steps"], 0)


class CubeShapingTests(unittest.TestCase):
    def test_low_power_cube_shaping_ignores_high_power_visible_cube(self) -> None:
        env = StagedRewardWrapper(
            _VisibleCubeEnv(power_frac=1.0, bearing=0.0),
            stage="power_cubes",
            cube_shaping="low_power",
            cube_heading_reward=1.0,
        )
        env.reset()
        _, reward, _, _, _ = env.step(0)
        self.assertEqual(reward, 0.0)

    def test_intercept_cube_shaping_rewards_high_power_alignment(self) -> None:
        env = StagedRewardWrapper(
            _VisibleCubeEnv(power_frac=1.0, bearing=0.0),
            stage="power_cubes",
            cube_shaping="intercept",
            cube_heading_reward=1.0,
        )
        env.reset()
        _, reward, _, _, info = env.step(0)
        self.assertGreater(reward, 0.0)
        self.assertEqual(info["cube_shaping"], "intercept")

    def test_minerals_intercept_cube_shaping_rewards_high_power_alignment(
        self,
    ) -> None:
        env = StagedRewardWrapper(
            _VisibleCubeEnv(power_frac=1.0, bearing=0.0),
            stage="minerals",
            cube_shaping="intercept",
            cube_heading_reward=1.0,
        )
        env.reset()
        _, reward, _, _, info = env.step(0)
        self.assertGreater(reward, 0.0)
        self.assertEqual(info["cube_shaping"], "intercept")


class CubeInterceptTerminationTests(unittest.TestCase):
    def test_cube_intercept_terminates_on_pickup(self) -> None:
        env = StagedRewardWrapper(
            _OneStepEnv(
                distance=0.0,
                power_frac=1.0,
                cube_bonus=100.0,
                episode_cube_pickups=1.0,
            ),
            stage="cube_intercept",
            cube_shaping="off",
        )
        env.reset()
        _, reward, terminated, truncated, info = env.step(7)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["game_over"], "cube_picked_up")
        self.assertEqual(reward, 100.0)


if __name__ == "__main__":
    unittest.main()
