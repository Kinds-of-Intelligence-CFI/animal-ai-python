import unittest
from unittest.mock import MagicMock

import numpy as np
from gymnasium import spaces

from animalai.wrappers.multi_agent import MultiAgentCore
from animalai.wrappers.unity_gymnasium import UnityGymnasiumException

from .multi_agent_fakes import (
    CAMERA_SHAPE,
    RAY_LENGTH,
    ScriptedAAIEnv,
    deciding,
    silent,
    unity_id,
)


class TestConstruction(unittest.TestCase):
    def test_agent_names_follow_sorted_agent_ids(self):
        core = MultiAgentCore(ScriptedAAIEnv([deciding([2, 0, 1])]))

        self.assertEqual(core.num_agents, 3)
        self.assertEqual(core.agent_ids, (unity_id(0), unity_id(1), unity_id(2)))
        self.assertEqual(core.agent_names, ["agent_0", "agent_1", "agent_2"])

    def test_accepts_one_agent(self):
        core = MultiAgentCore(ScriptedAAIEnv([deciding([0])]))

        self.assertEqual(core.agent_names, ["agent_0"])

    def test_does_not_reset_or_step_unity(self):
        env = ScriptedAAIEnv([deciding([0, 1])])

        MultiAgentCore(env)

        env.reset.assert_not_called()
        env.step.assert_not_called()

    def test_raises_when_no_agent_waits_for_an_action(self):
        env = ScriptedAAIEnv([silent()])

        with self.assertRaises(UnityGymnasiumException) as context:
            MultiAgentCore(env)

        self.assertIn("No agent", str(context.exception))
        env.step.assert_not_called()

    def test_two_behaviors_raise_and_the_message_names_them(self):
        env = ScriptedAAIEnv([deciding([0, 1])], behaviors=("AnimalAI?team=0", "AnimalAI?team=1"))

        with self.assertRaises(UnityGymnasiumException) as context:
            MultiAgentCore(env)

        self.assertIn("AnimalAI?team=0", str(context.exception))
        self.assertIn("AnimalAI?team=1", str(context.exception))
        self.assertIn("teams", str(context.exception))

    def test_rejects_an_env_that_is_not_animal_ai(self):
        with self.assertRaises(TypeError):
            MultiAgentCore(MagicMock())


class TestSingleAgentSpaces(unittest.TestCase):
    def test_action_space_has_the_two_animal_ai_branches(self):
        core = MultiAgentCore(ScriptedAAIEnv([deciding([0, 1])]))

        self.assertEqual(core.single_action_space, spaces.MultiDiscrete([3, 3]))

    def test_observation_space_is_a_dict_of_the_streams_of_one_agent(self):
        core = MultiAgentCore(ScriptedAAIEnv([deciding([0, 1])]))

        space = core.single_observation_space
        self.assertIsInstance(space, spaces.Dict)
        self.assertEqual(set(space.spaces.keys()), {"camera", "rays", "health", "velocity", "position"})
        self.assertEqual(space["camera"].shape, CAMERA_SHAPE)
        self.assertEqual(space["rays"].shape, (RAY_LENGTH,))
        self.assertEqual(space["health"].shape, ())
        self.assertEqual(space["velocity"].shape, (3,))
        self.assertEqual(space["position"].shape, (3,))

    def test_include_flags_select_the_streams(self):
        core = MultiAgentCore(
            ScriptedAAIEnv([deciding([0, 1])]),
            include_camera=False,
            include_velocity=False,
        )

        self.assertEqual(set(core.single_observation_space.spaces.keys()), {"rays", "health", "position"})

    def test_one_stream_gives_a_bare_box(self):
        core = MultiAgentCore(
            ScriptedAAIEnv([deciding([0, 1])]),
            include_rays=False,
            include_health=False,
            include_velocity=False,
            include_position=False,
        )

        self.assertIsInstance(core.single_observation_space, spaces.Box)
        self.assertEqual(core.single_observation_space.shape, CAMERA_SHAPE)

    def test_stream_that_the_env_does_not_produce_raises(self):
        env = ScriptedAAIEnv([deciding([0, 1], useCamera=False)], useCamera=False)

        with self.assertRaises(ValueError):
            MultiAgentCore(env, include_camera=True)


class TestObservationOrder(unittest.TestCase):
    def test_observations_follow_the_agent_order_when_unity_permutes_rows(self):
        core = MultiAgentCore(ScriptedAAIEnv([deciding([2, 0, 1])]))

        obs, _ = core.reset()

        for agent in range(3):
            np.testing.assert_allclose(obs["camera"][agent], np.full(CAMERA_SHAPE, agent / 10.0))
            np.testing.assert_allclose(obs["rays"][agent], np.full(RAY_LENGTH, float(agent)))
            self.assertAlmostEqual(float(obs["health"][agent]), 100.0 - agent)
            np.testing.assert_allclose(obs["velocity"][agent], [agent + 0.1, agent + 0.2, agent + 0.3], rtol=1e-6)
            np.testing.assert_allclose(obs["position"][agent], [10.0 * agent, 1.0, 20.0])

    def test_each_agent_row_is_in_the_single_observation_space(self):
        core = MultiAgentCore(ScriptedAAIEnv([deciding([1, 0])]))

        obs, _ = core.reset()

        self.assertEqual(obs["camera"].shape, (2, *CAMERA_SHAPE))
        self.assertEqual(obs["health"].shape, (2,))
        for agent in range(2):
            row = {key: value[agent] for key, value in obs.items()}
            self.assertTrue(core.single_observation_space.contains(row))

    def test_uint8_visual_converts_the_camera(self):
        core = MultiAgentCore(ScriptedAAIEnv([deciding([1, 0])]), uint8_visual=True)

        obs, _ = core.reset()

        self.assertEqual(obs["camera"].dtype, np.uint8)
        self.assertTrue((obs["camera"][0] == 0).all())
        self.assertTrue((obs["camera"][1] == 25).all())

    def test_one_stream_returns_a_bare_array(self):
        core = MultiAgentCore(
            ScriptedAAIEnv([deciding([1, 0])]),
            include_camera=False,
            include_rays=False,
            include_velocity=False,
            include_position=False,
        )

        obs, _ = core.reset()

        np.testing.assert_allclose(obs, [100.0, 99.0])

    def test_first_reset_does_not_reset_unity(self):
        env = ScriptedAAIEnv([deciding([0, 1])])
        core = MultiAgentCore(env)

        core.reset()

        env.reset.assert_not_called()
        env.step.assert_not_called()


if __name__ == "__main__":
    unittest.main()
