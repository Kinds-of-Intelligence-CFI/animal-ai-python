import contextlib
import io
import unittest
import warnings

import numpy as np
from pettingzoo import ParallelEnv
from pettingzoo.test import parallel_api_test

from animalai.wrappers.animalai_gymnasium import AnimalAIGymnasiumWrapper
from animalai.wrappers.animalai_pettingzoo import AnimalAIParallelEnv
from animalai.wrappers.unity_gymnasium import UnityGymnasiumException

from .multi_agent_fakes import CAMERA_SHAPE, ScriptedAAIEnv, deciding, ending

NOOP = [0, 0]


def _noops(env: AnimalAIParallelEnv) -> dict:
    return {agent: NOOP for agent in env.agents}


class TestAgents(unittest.TestCase):
    def test_possible_agents_are_named_in_agent_order(self):
        env = AnimalAIParallelEnv(ScriptedAAIEnv([deciding([2, 0, 1])]))

        self.assertEqual(env.possible_agents, ["agent_0", "agent_1", "agent_2"])
        self.assertEqual(env.agents, ["agent_0", "agent_1", "agent_2"])
        self.assertEqual(env.max_num_agents, 3)

    def test_agents_is_empty_after_the_final_step_and_full_after_reset(self):
        env = AnimalAIParallelEnv(
            ScriptedAAIEnv([deciding([0, 1]), deciding([0, 1]), ending([0, 1], next_agents=[0, 1])])
        )
        env.reset()

        env.step(_noops(env))
        agents_during_the_episode = list(env.agents)
        env.step(_noops(env))
        agents_after_the_final_step = list(env.agents)
        env.reset()

        self.assertEqual(agents_during_the_episode, ["agent_0", "agent_1"])
        self.assertEqual(agents_after_the_final_step, [])
        self.assertEqual(env.agents, ["agent_0", "agent_1"])
        self.assertEqual(env.possible_agents, ["agent_0", "agent_1"])


class TestSpaces(unittest.TestCase):
    def _assert_same_spaces_as_the_single_agent_wrapper(self, **wrapper_kwargs):
        single = AnimalAIGymnasiumWrapper(ScriptedAAIEnv([deciding([0]), deciding([0])]), **wrapper_kwargs)
        multi = AnimalAIParallelEnv(ScriptedAAIEnv([deciding([0, 1])]), **wrapper_kwargs)

        for agent in multi.possible_agents:
            self.assertEqual(multi.observation_space(agent), single.observation_space)
            self.assertEqual(multi.action_space(agent), single.action_space)

    def test_default_spaces_equal_the_single_agent_wrapper_spaces(self):
        self._assert_same_spaces_as_the_single_agent_wrapper()

    def test_selected_stream_spaces_equal_the_single_agent_wrapper_spaces(self):
        self._assert_same_spaces_as_the_single_agent_wrapper(
            include_rays=False, include_position=False, uint8_visual=True
        )

    def test_one_stream_spaces_equal_the_single_agent_wrapper_spaces(self):
        self._assert_same_spaces_as_the_single_agent_wrapper(
            include_rays=False, include_health=False, include_velocity=False, include_position=False
        )

    def test_space_methods_return_the_same_object_each_time(self):
        env = AnimalAIParallelEnv(ScriptedAAIEnv([deciding([0, 1])]))

        self.assertIs(env.observation_space("agent_0"), env.observation_space("agent_0"))
        self.assertIs(env.action_space("agent_1"), env.action_space("agent_1"))

    def test_unknown_agent_raises(self):
        env = AnimalAIParallelEnv(ScriptedAAIEnv([deciding([0, 1])]))

        with self.assertRaises(KeyError):
            env.observation_space("agent_2")
        with self.assertRaises(KeyError):
            env.action_space("agent_2")


class TestReset(unittest.TestCase):
    def test_returns_the_observation_of_each_agent(self):
        env = AnimalAIParallelEnv(ScriptedAAIEnv([deciding([2, 0, 1])]))

        obs, infos = env.reset()

        self.assertEqual(set(obs.keys()), {"agent_0", "agent_1", "agent_2"})
        self.assertEqual(set(infos.keys()), {"agent_0", "agent_1", "agent_2"})
        for number, agent in enumerate(env.possible_agents):
            self.assertAlmostEqual(float(obs[agent]["health"]), 100.0 - number)
            np.testing.assert_allclose(obs[agent]["position"], [10.0 * number, 1.0, 20.0])
            self.assertEqual(obs[agent]["camera"].shape, CAMERA_SHAPE)
            self.assertEqual(infos[agent], {})

    def test_each_observation_is_in_the_observation_space(self):
        env = AnimalAIParallelEnv(ScriptedAAIEnv([deciding([1, 0])]))

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            obs, _ = env.reset()
            for agent in env.agents:
                self.assertTrue(env.observation_space(agent).contains(obs[agent]))

    def test_one_stream_gives_a_bare_array_for_each_agent(self):
        env = AnimalAIParallelEnv(
            ScriptedAAIEnv([deciding([1, 0])]),
            include_rays=False,
            include_health=False,
            include_velocity=False,
            include_position=False,
        )

        obs, _ = env.reset()

        np.testing.assert_allclose(obs["agent_1"], np.full(CAMERA_SHAPE, 0.1))

    def test_seed_seeds_the_action_space(self):
        env = AnimalAIParallelEnv(ScriptedAAIEnv([deciding([0, 1])]))

        env.reset(seed=7)
        first = [env.action_space("agent_0").sample() for _ in range(5)]
        env.reset(seed=7)
        second = [env.action_space("agent_0").sample() for _ in range(5)]

        np.testing.assert_array_equal(first, second)

    def test_arena_config_is_sent_to_unity(self):
        unity_env = ScriptedAAIEnv([deciding([0, 1]), deciding([0, 1])])
        env = AnimalAIParallelEnv(unity_env)

        env.reset(options={"arena_config": "other.yml"})

        unity_env.reset.assert_called_once_with(arenas_configurations="other.yml")


class TestStep(unittest.TestCase):
    def test_returns_five_dicts_with_the_values_of_each_agent(self):
        env = AnimalAIParallelEnv(ScriptedAAIEnv([deciding([0, 1]), deciding([1, 0], rewards=[0.5, 0.25])]))
        env.reset()

        obs, rewards, terminations, truncations, infos = env.step(_noops(env))

        self.assertEqual(rewards, {"agent_0": 0.25, "agent_1": 0.5})
        self.assertEqual(terminations, {"agent_0": False, "agent_1": False})
        self.assertEqual(truncations, {"agent_0": False, "agent_1": False})
        self.assertEqual(infos, {"agent_0": {}, "agent_1": {}})
        self.assertAlmostEqual(float(obs["agent_0"]["health"]), 100.0)
        self.assertAlmostEqual(float(obs["agent_1"]["health"]), 99.0)
        self.assertIs(type(rewards["agent_0"]), float)
        self.assertIs(type(terminations["agent_0"]), bool)
        self.assertIs(type(truncations["agent_0"]), bool)

    def test_final_step_gives_the_flags_and_rewards_of_each_agent(self):
        env = AnimalAIParallelEnv(ScriptedAAIEnv([deciding([0, 1]), ending([1, 0], rewards=[1.0, 2.0])]))
        env.reset()

        _, rewards, terminations, truncations, _ = env.step(_noops(env))

        self.assertEqual(rewards, {"agent_0": 2.0, "agent_1": 1.0})
        self.assertEqual(terminations, {"agent_0": True, "agent_1": True})
        self.assertEqual(truncations, {"agent_0": False, "agent_1": False})

    def test_each_action_reaches_its_agent(self):
        unity_env = ScriptedAAIEnv([deciding([1, 0]), deciding([0, 1])])
        env = AnimalAIParallelEnv(unity_env)
        env.reset()

        env.step({"agent_1": np.array([2, 0]), "agent_0": np.array([1, 0])})

        _, action_tuple = unity_env.set_actions.call_args.args
        np.testing.assert_array_equal(action_tuple.discrete, [[2, 0], [1, 0]])

    def test_step_after_the_final_step_raises(self):
        env = AnimalAIParallelEnv(ScriptedAAIEnv([deciding([0, 1]), ending([0, 1]), deciding([0, 1])]))
        env.reset()
        env.step(_noops(env))

        with self.assertRaises(UnityGymnasiumException) as context:
            env.step({})

        self.assertIn("reset", str(context.exception))


class TestActionKeyChecks(unittest.TestCase):
    def _assert_actions_rejected(self, actions):
        unity_env = ScriptedAAIEnv([deciding([0, 1]), deciding([0, 1])])
        env = AnimalAIParallelEnv(unity_env)
        env.reset()
        with self.assertRaises(ValueError) as context:
            env.step(actions)
        unity_env.set_actions.assert_not_called()
        unity_env.step.assert_not_called()
        return str(context.exception)

    def test_missing_agent_raises_and_names_it(self):
        message = self._assert_actions_rejected({"agent_0": NOOP})

        self.assertIn("agent_1", message)

    def test_extra_agent_raises_and_names_it(self):
        message = self._assert_actions_rejected({"agent_0": NOOP, "agent_1": NOOP, "agent_2": NOOP})

        self.assertIn("agent_2", message)

    def test_unknown_agent_name_raises(self):
        self._assert_actions_rejected({"agent_0": NOOP, "bob": NOOP})

    def test_action_out_of_range_raises_and_names_the_agent(self):
        message = self._assert_actions_rejected({"agent_0": NOOP, "agent_1": [0, 3]})

        self.assertIn("agent_1", message)

    def test_action_with_the_wrong_shape_raises(self):
        self._assert_actions_rejected({"agent_0": NOOP, "agent_1": 4})


class TestRenderAndClose(unittest.TestCase):
    def test_render_returns_the_latest_camera_frame_of_agent_0(self):
        env = AnimalAIParallelEnv(ScriptedAAIEnv([deciding([1, 0]), deciding([1, 0])]))
        obs, _ = env.reset()

        frame = env.render()

        self.assertEqual(frame.shape, CAMERA_SHAPE)
        np.testing.assert_array_equal(frame, obs["agent_0"]["camera"])

    def test_render_returns_none_when_the_camera_is_not_selected(self):
        env = AnimalAIParallelEnv(ScriptedAAIEnv([deciding([0, 1])]), include_camera=False)
        env.reset()

        self.assertIsNone(env.render())

    def test_close_closes_the_unity_env(self):
        unity_env = ScriptedAAIEnv([deciding([0, 1])])
        env = AnimalAIParallelEnv(unity_env)

        env.close()

        self.assertTrue(unity_env.closed)


class TestPettingZooApi(unittest.TestCase):
    def test_parallel_api_test_passes_with_no_warnings(self):
        episode = [deciding([0, 1]), deciding([1, 0]), ending([0, 1], next_agents=[0, 1])]
        env = AnimalAIParallelEnv(ScriptedAAIEnv([deciding([0, 1])] + episode + episode))

        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("error")
            parallel_api_test(env, num_cycles=3)


class TestLazyExport(unittest.TestCase):
    def test_wrappers_package_exports_the_parallel_env(self):
        from animalai import wrappers

        self.assertIs(wrappers.AnimalAIParallelEnv, AnimalAIParallelEnv)
        self.assertIn("AnimalAIParallelEnv", wrappers.__all__)


if __name__ == "__main__":
    unittest.main()
