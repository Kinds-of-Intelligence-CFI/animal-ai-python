import unittest
from unittest.mock import MagicMock

import numpy as np
from gymnasium import spaces

from animalai.wrappers.multi_agent import MultiAgentCore
from animalai.wrappers.unity_gymnasium import UnityGymnasiumException

from .multi_agent_fakes import (
    BEHAVIOR,
    CAMERA_SHAPE,
    RAY_LENGTH,
    ScriptedAAIEnv,
    deciding,
    decision_steps,
    ending,
    silent,
    terminal_steps,
    unity_id,
)

NOOP = [0, 0]


def _sent_actions(env: ScriptedAAIEnv, call: int = -1):
    """The behavior name and the ActionTuple of one ``set_actions`` call."""
    behavior, action_tuple = env.set_actions.call_args_list[call].args
    return behavior, action_tuple


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
            row = {key: np.asarray(value[agent]) for key, value in obs.items()}
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


class TestActionRouting(unittest.TestCase):
    def test_each_action_reaches_the_unity_row_of_its_agent(self):
        env = ScriptedAAIEnv([deciding([2, 0, 1]), deciding([0, 1, 2])])
        core = MultiAgentCore(env)

        core.step(np.array([[0, 0], [1, 1], [2, 2]]))

        behavior, action_tuple = _sent_actions(env)
        self.assertEqual(behavior, BEHAVIOR)
        np.testing.assert_array_equal(action_tuple.discrete, [[2, 2], [0, 0], [1, 1]])

    def test_routing_uses_the_row_order_of_the_latest_decision_steps(self):
        env = ScriptedAAIEnv([deciding([0, 1]), deciding([1, 0]), deciding([0, 1])])
        core = MultiAgentCore(env)
        actions = np.array([[1, 0], [2, 0]])

        core.step(actions)
        core.step(actions)

        np.testing.assert_array_equal(_sent_actions(env, 0)[1].discrete, [[1, 0], [2, 0]])
        np.testing.assert_array_equal(_sent_actions(env, 1)[1].discrete, [[2, 0], [1, 0]])

    def test_actions_are_sent_before_unity_steps(self):
        env = ScriptedAAIEnv([deciding([0, 1]), deciding([0, 1])])
        core = MultiAgentCore(env)
        calls = MagicMock()
        calls.attach_mock(env.set_actions, "set_actions")
        calls.attach_mock(env.step, "step")

        core.step([NOOP, NOOP])

        self.assertEqual([call[0] for call in calls.mock_calls], ["set_actions", "step"])

    def test_action_tuple_has_the_ml_agents_format(self):
        env = ScriptedAAIEnv([deciding([0, 1]), deciding([0, 1])])
        core = MultiAgentCore(env)

        core.step([[1, 2], [2, 1]])

        action_tuple = _sent_actions(env)[1]
        self.assertEqual(action_tuple.discrete.dtype, np.int32)
        self.assertEqual(action_tuple.discrete.shape, (2, 2))
        self.assertEqual(action_tuple.continuous.shape, (2, 0))


class TestStepResults(unittest.TestCase):
    def test_observations_and_rewards_follow_the_agent_order(self):
        env = ScriptedAAIEnv([deciding([0, 1, 2]), deciding([1, 2, 0], rewards=[0.1, 0.2, 0.0])])
        core = MultiAgentCore(env)

        obs, rewards, _, _, _ = core.step([NOOP] * 3)

        np.testing.assert_allclose(rewards, [0.0, 0.1, 0.2])
        self.assertEqual(rewards.dtype, np.float32)
        np.testing.assert_allclose(obs["health"], [100.0, 99.0, 98.0])
        np.testing.assert_allclose(obs["position"][:, 0], [0.0, 10.0, 20.0])

    def test_flags_are_false_while_the_episode_runs(self):
        env = ScriptedAAIEnv([deciding([0, 1]), deciding([0, 1])])
        core = MultiAgentCore(env)

        _, _, terminated, truncated, _ = core.step([NOOP, NOOP])

        np.testing.assert_array_equal(terminated, [False, False])
        np.testing.assert_array_equal(truncated, [False, False])
        self.assertEqual(terminated.dtype, np.bool_)
        self.assertEqual(truncated.dtype, np.bool_)

    def test_episode_end_terminates_all_agents(self):
        env = ScriptedAAIEnv([deciding([0, 1]), ending([1, 0], rewards=[1.0, 2.0])])
        core = MultiAgentCore(env)

        obs, rewards, terminated, truncated, _ = core.step([NOOP, NOOP])

        np.testing.assert_array_equal(terminated, [True, True])
        np.testing.assert_array_equal(truncated, [False, False])
        np.testing.assert_allclose(rewards, [2.0, 1.0])
        np.testing.assert_allclose(obs["health"], [100.0, 99.0])

    def test_interrupted_episode_end_truncates_all_agents(self):
        env = ScriptedAAIEnv([deciding([0, 1]), ending([0, 1], interrupted=True)])
        core = MultiAgentCore(env)

        _, _, terminated, truncated, _ = core.step([NOOP, NOOP])

        np.testing.assert_array_equal(terminated, [False, False])
        np.testing.assert_array_equal(truncated, [True, True])

    def test_episode_end_uses_the_terminal_steps_when_the_next_episode_is_in_the_same_exchange(self):
        env = ScriptedAAIEnv([deciding([0, 1]), ending([0, 1], rewards=[1.0, 1.0], next_agents=[0, 1])])
        core = MultiAgentCore(env)

        _, rewards, terminated, _, _ = core.step([NOOP, NOOP])

        np.testing.assert_allclose(rewards, [1.0, 1.0])
        np.testing.assert_array_equal(terminated, [True, True])


class TestAgentSetChecks(unittest.TestCase):
    def _assert_step_raises(self, second_exchange, number_of_agents=2):
        env = ScriptedAAIEnv([deciding(range(number_of_agents)), second_exchange])
        core = MultiAgentCore(env)
        with self.assertRaises(UnityGymnasiumException) as context:
            core.step([NOOP] * number_of_agents)
        return str(context.exception)

    def test_missing_agent_raises(self):
        message = self._assert_step_raises(deciding([0, 2]), number_of_agents=3)

        self.assertIn(str([unity_id(0), unity_id(1), unity_id(2)]), message)
        self.assertIn(str([unity_id(0), unity_id(2)]), message)
        self.assertIn("Agent items", message)

    def test_extra_agent_raises(self):
        self._assert_step_raises(deciding([0, 1, 2]))

    def test_replaced_agent_id_raises(self):
        self._assert_step_raises((decision_steps([0, 1], ids=[unity_id(0), 99]), terminal_steps()))

    def test_duplicate_agent_id_raises(self):
        self._assert_step_raises((decision_steps([0, 1], ids=[unity_id(0), unity_id(0)]), terminal_steps()))

    def test_exchange_with_no_agents_raises(self):
        self._assert_step_raises(silent())

    def test_episode_end_for_some_agents_only_raises(self):
        message = self._assert_step_raises(ending([0], next_agents=[1]))

        self.assertIn("TerminalSteps", message)


class TestActionChecks(unittest.TestCase):
    def _assert_actions_rejected(self, actions):
        env = ScriptedAAIEnv([deciding([0, 1]), deciding([0, 1])])
        core = MultiAgentCore(env)
        with self.assertRaises(ValueError) as context:
            core.step(actions)
        env.set_actions.assert_not_called()
        env.step.assert_not_called()
        return str(context.exception)

    def test_one_action_for_two_agents_raises(self):
        message = self._assert_actions_rejected([NOOP])

        self.assertIn("(2, 2)", message)
        self.assertIn("(1, 2)", message)

    def test_flat_action_raises(self):
        self._assert_actions_rejected([0, 0])

    def test_wrong_number_of_branches_raises(self):
        self._assert_actions_rejected([[0, 0, 0], [0, 0, 0]])

    def test_action_above_the_range_raises_and_names_the_agent(self):
        message = self._assert_actions_rejected([NOOP, [3, 0]])

        self.assertIn("agent_1", message)

    def test_negative_action_raises(self):
        self._assert_actions_rejected([[0, -1], NOOP])

    def test_action_that_is_not_an_integer_raises(self):
        self._assert_actions_rejected([[0.5, 0.0], [1.0, 1.0]])


if __name__ == "__main__":
    unittest.main()
