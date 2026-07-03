import unittest

import gymnasium as gym
import numpy as np

from hylaeanrover import OBS_DIM, RoverEnv
from hylaeanrover.wrappers import (
    CUBE_SPAWN_PRESETS,
    LOCOMOTION_SHAPING_MODES,
    StagedRewardWrapper,
    apply_scenario_defaults,
    parse_terrain_height,
)


class StageHardeningPresetTests(unittest.TestCase):
    def test_named_presets_cover_dense_sparse_and_no_cube_controls(self) -> None:
        self.assertGreater(CUBE_SPAWN_PRESETS["dense_training"]["lambda"], 1.0)
        self.assertEqual(CUBE_SPAWN_PRESETS["transition"]["lambda"], 0.30)
        self.assertEqual(CUBE_SPAWN_PRESETS["sparse_game"]["lambda"], 0.05)
        self.assertEqual(CUBE_SPAWN_PRESETS["none"]["lambda"], 0.0)
        self.assertIn("power_efficiency", LOCOMOTION_SHAPING_MODES)

    def test_scenario_defaults_can_be_overridden(self) -> None:
        cfg = apply_scenario_defaults(
            "low_power_start",
            cube_spawn_preset="sparse_game",
            power_start_fraction=0.25,
        )
        self.assertEqual(cfg["cube_spawn_preset"], "sparse_game")
        self.assertEqual(cfg["power_start_fraction"], 0.25)

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
        game_over: str | None = None,
    ) -> None:
        self.distance = distance
        self.power_frac = power_frac
        self.game_over = game_over

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(OBS_DIM, dtype=np.float32), {
            "reward_distance": 0.0,
            "reward_cube_bonus": 0.0,
            "reward_mineral_integral": 0.0,
            "reward_beacon_bonus": 0.0,
            "power_frac": 1.0,
        }

    def step(self, action):
        return (
            np.zeros(OBS_DIM, dtype=np.float32),
            0.0,
            self.game_over is not None,
            False,
            {
                "reward_distance": self.distance,
                "reward_cube_bonus": 0.0,
                "reward_mineral_integral": 0.0,
                "reward_beacon_bonus": 0.0,
                "power_frac": self.power_frac,
                "game_over": self.game_over,
            },
        )


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


if __name__ == "__main__":
    unittest.main()
