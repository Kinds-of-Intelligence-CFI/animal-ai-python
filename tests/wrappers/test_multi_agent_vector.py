import unittest
import warnings

import gymnasium
import numpy as np
from gymnasium.vector.utils import batch_space

from animalai.wrappers.animalai_gymnasium import AnimalAIGymnasiumWrapper
from animalai.wrappers.multi_agent import AnimalAIAgentStackEnv
from animalai.wrappers.unity_gymnasium import UnityGymnasiumException

from .multi_agent_fakes import CAMERA_SHAPE, ScriptedAAIEnv, deciding, ending

NOOPS = np.zeros((2, 2), dtype=np.int64)


class TestStackEnvSpaces(unittest.TestCase):
    def test_spaces_are_the_single_agent_spaces_with_a_leading_agent_dimension(self):
        single = AnimalAIGymnasiumWrapper(ScriptedAAIEnv([deciding([0]), deciding([0])]))
        env = AnimalAIAgentStackEnv(ScriptedAAIEnv([deciding([2, 0, 1])]))

        self.assertEqual(env.observation_space, batch_space(single.observation_space, 3))
        self.assertEqual(env.action_space, batch_space(single.action_space, 3))
        self.assertEqual(env.observation_space["camera"].shape, (3, *CAMERA_SHAPE))
        self.assertEqual(env.observation_space["health"].shape, (3,))
        self.assertEqual(env.action_space.shape, (3, 2))

    def test_keyword_arguments_select_the_streams(self):
        env = AnimalAIAgentStackEnv(
            ScriptedAAIEnv([deciding([0, 1])]),
            include_rays=False,
            include_health=False,
            include_velocity=False,
            include_position=False,
            uint8_visual=True,
        )

        self.assertEqual(env.observation_space.shape, (2, *CAMERA_SHAPE))
        self.assertEqual(env.observation_space.dtype, np.uint8)


class TestStackEnvReset(unittest.TestCase):
    def test_returns_observations_in_agent_order(self):
        env = AnimalAIAgentStackEnv(ScriptedAAIEnv([deciding([1, 0])]))

        obs, info = env.reset()

        np.testing.assert_allclose(obs["health"], [100.0, 99.0])
        np.testing.assert_allclose(obs["position"][:, 0], [0.0, 10.0])
        self.assertEqual(info, {})

    def test_observations_are_in_the_observation_space(self):
        env = AnimalAIAgentStackEnv(ScriptedAAIEnv([deciding([1, 0])]))

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            obs, _ = env.reset()
            self.assertTrue(env.observation_space.contains(obs))

    def test_seed_seeds_the_action_space(self):
        env = AnimalAIAgentStackEnv(ScriptedAAIEnv([deciding([0, 1])]))

        env.reset(seed=3)
        first = [env.action_space.sample() for _ in range(5)]
        env.reset(seed=3)
        second = [env.action_space.sample() for _ in range(5)]

        np.testing.assert_array_equal(first, second)

    def test_arena_config_is_sent_to_unity(self):
        unity_env = ScriptedAAIEnv([deciding([0, 1]), deciding([0, 1])])
        env = AnimalAIAgentStackEnv(unity_env)

        env.reset(options={"arena_config": "other.yml"})

        unity_env.reset.assert_called_once_with(arenas_configurations="other.yml")

    def test_reset_after_the_final_step_makes_no_unity_reset(self):
        unity_env = ScriptedAAIEnv([deciding([0, 1]), ending([0, 1], next_agents=[1, 0])])
        env = AnimalAIAgentStackEnv(unity_env)
        env.step(NOOPS)

        obs, _ = env.reset()

        unity_env.reset.assert_not_called()
        np.testing.assert_allclose(obs["health"], [100.0, 99.0])


class TestStackEnvStep(unittest.TestCase):
    def test_reward_is_the_sum_and_info_has_the_reward_of_each_agent(self):
        env = AnimalAIAgentStackEnv(ScriptedAAIEnv([deciding([0, 1]), deciding([1, 0], rewards=[0.5, 0.25])]))

        obs, reward, terminated, truncated, info = env.step(NOOPS)

        self.assertAlmostEqual(reward, 0.75)
        self.assertIs(type(reward), float)
        np.testing.assert_allclose(info["agent_rewards"], [0.25, 0.5])
        self.assertEqual(info["agent_rewards"].dtype, np.float32)
        self.assertIs(terminated, False)
        self.assertIs(truncated, False)
        self.assertTrue(env.observation_space.contains(obs))

    def test_final_step_terminates(self):
        env = AnimalAIAgentStackEnv(ScriptedAAIEnv([deciding([0, 1]), ending([1, 0], rewards=[1.0, 2.0])]))

        _, reward, terminated, truncated, info = env.step(NOOPS)

        self.assertAlmostEqual(reward, 3.0)
        np.testing.assert_allclose(info["agent_rewards"], [2.0, 1.0])
        self.assertIs(terminated, True)
        self.assertIs(truncated, False)

    def test_interrupted_final_step_truncates(self):
        env = AnimalAIAgentStackEnv(ScriptedAAIEnv([deciding([0, 1]), ending([0, 1], interrupted=True)]))

        _, _, terminated, truncated, _ = env.step(NOOPS)

        self.assertIs(terminated, False)
        self.assertIs(truncated, True)

    def test_each_action_row_reaches_its_agent(self):
        unity_env = ScriptedAAIEnv([deciding([1, 0]), deciding([0, 1])])
        env = AnimalAIAgentStackEnv(unity_env)

        env.step(np.array([[1, 0], [2, 0]]))

        _, action_tuple = unity_env.set_actions.call_args.args
        np.testing.assert_array_equal(action_tuple.discrete, [[2, 0], [1, 0]])

    def test_sampled_actions_are_accepted(self):
        env = AnimalAIAgentStackEnv(ScriptedAAIEnv([deciding([0, 1]), deciding([0, 1])]))
        env.reset(seed=0)

        env.step(env.action_space.sample())

    def test_step_after_the_final_step_raises(self):
        env = AnimalAIAgentStackEnv(ScriptedAAIEnv([deciding([0, 1]), ending([0, 1]), deciding([0, 1])]))
        env.step(NOOPS)

        with self.assertRaises(UnityGymnasiumException):
            env.step(NOOPS)


class TestStackEnvRenderAndClose(unittest.TestCase):
    def test_render_returns_the_latest_camera_frame_of_agent_0(self):
        env = AnimalAIAgentStackEnv(ScriptedAAIEnv([deciding([1, 0])]))
        obs, _ = env.reset()

        np.testing.assert_array_equal(env.render(), obs["camera"][0])

    def test_render_returns_none_when_the_camera_is_not_selected(self):
        env = AnimalAIAgentStackEnv(ScriptedAAIEnv([deciding([0, 1])]), include_camera=False)
        env.reset()

        self.assertIsNone(env.render())

    def test_close_closes_the_unity_env(self):
        unity_env = ScriptedAAIEnv([deciding([0, 1])])
        env = AnimalAIAgentStackEnv(unity_env)

        env.close()

        self.assertTrue(unity_env.closed)


if __name__ == "__main__":
    unittest.main()
