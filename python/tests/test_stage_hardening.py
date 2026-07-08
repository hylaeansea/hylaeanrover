import unittest

import gymnasium as gym
import numpy as np

from hylaeanrover import OBS_DIM, RoverEnv
from hylaeanrover.teacher import (
    CUBE_OBS_START,
    POWER_OBS_INDEX,
    CubeInterceptTeacher,
    PowerIdleTeacher,
)
from hylaeanrover.wrappers import (
    CUBE_SHAPING_MODES,
    CUBE_SPAWN_PRESETS,
    LOCOMOTION_SHAPING_MODES,
    STAGE_WEIGHTS,
    StagedRewardWrapper,
    apply_scenario_defaults,
    make_staged_env,
    parse_terrain_height,
)


class StageHardeningPresetTests(unittest.TestCase):
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
    ) -> None:
        self.distance = distance
        self.power_frac = power_frac
        self.cube_bonus = cube_bonus
        self.episode_cube_pickups = episode_cube_pickups
        self.game_over = game_over
        self.reset_power_frac = reset_power_frac

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(OBS_DIM, dtype=np.float32), {
            "reward_distance": 0.0,
            "reward_cube_bonus": 0.0,
            "reward_mineral_integral": 0.0,
            "reward_beacon_bonus": 0.0,
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
                "power_frac": self.power_frac,
                "episode_cube_pickups": self.episode_cube_pickups,
                "game_over": self.game_over,
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


class LocomotionPowerShapingTests(unittest.TestCase):
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
