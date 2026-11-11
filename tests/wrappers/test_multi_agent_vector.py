import unittest
import warnings

import gymnasium
import numpy as np
from gymnasium.vector.utils import batch_space

from animalai.wrappers.animalai_gymnasium import AnimalAIGymnasiumWrapper
from animalai.wrappers.multi_agent import AnimalAIAgentStackEnv
from animalai.wrappers.unity_gymnasium import UnityGymnasiumException

from animalai.wrappers import vector

from .multi_agent_fakes import CAMERA_SHAPE, FactoryScriptedAAIEnv, ScriptedAAIEnv, deciding, ending

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
        self.assertEqual(set(info), {"agent_rewards"})
        np.testing.assert_array_equal(info["agent_rewards"], [0.0, 0.0])
        self.assertEqual(info["agent_rewards"].dtype, np.float32)

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


class _FactoryTestCase(unittest.TestCase):
    def setUp(self):
        FactoryScriptedAAIEnv.instances = []
        # The factory constructs AnimalAIEnvironment by name; swap in the fake.
        self._real_env = vector.AnimalAIEnvironment
        vector.AnimalAIEnvironment = FactoryScriptedAAIEnv

    def tearDown(self):
        vector.AnimalAIEnvironment = self._real_env


class TestEnvFnsWrapperClass(_FactoryTestCase):
    def test_default_wrapper_is_the_single_agent_wrapper(self):
        env_fns = vector.make_animalai_env_fns(1, env_kwargs={"number_of_agents": 1}, serialize_launch=False)

        self.assertIsInstance(env_fns[0](), AnimalAIGymnasiumWrapper)

    def test_wrapper_cls_selects_the_wrapper(self):
        env_fns = vector.make_animalai_env_fns(
            2,
            wrapper_cls=AnimalAIAgentStackEnv,
            wrapper_kwargs={"include_camera": False},
            serialize_launch=False,
        )

        envs = [env_fn() for env_fn in env_fns]

        for env in envs:
            self.assertIsInstance(env, AnimalAIAgentStackEnv)
            self.assertNotIn("camera", env.observation_space.spaces)
        self.assertEqual([instance.worker_id for instance in FactoryScriptedAAIEnv.instances], [0, 1])


class TestMultiAgentVecEnv(_FactoryTestCase):
    def _make(self, num_envs=2, **kwargs):
        kwargs.setdefault("vectorization_mode", "sync")
        return vector.make_animalai_multi_agent_vec_env(num_envs, **kwargs)

    def test_sync_mode_gives_a_sync_vector_env_of_stack_envs(self):
        venv = self._make(2)

        self.assertIsInstance(venv, gymnasium.vector.SyncVectorEnv)
        self.assertEqual(venv.num_envs, 2)
        for env in venv.envs:
            self.assertIsInstance(env, AnimalAIAgentStackEnv)
        venv.close()

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            self._make(2, vectorization_mode="not-a-mode")

    def test_wrapper_cls_is_not_an_argument(self):
        with self.assertRaises(TypeError):
            self._make(2, wrapper_cls=AnimalAIGymnasiumWrapper)

    def test_each_instance_gets_its_own_worker_id_and_seed(self):
        venv = self._make(3, worker_id_start=4, seed=10, base_port=6000)

        instances = FactoryScriptedAAIEnv.instances
        self.assertEqual([instance.worker_id for instance in instances], [4, 5, 6])
        self.assertEqual([instance.seed_value for instance in instances], [10, 11, 12])
        self.assertEqual({instance.base_port for instance in instances}, {6000})
        venv.close()

    def test_spaces_have_an_instance_dimension_then_an_agent_dimension(self):
        venv = self._make(2, env_kwargs={"number_of_agents": 3})

        self.assertEqual(venv.single_action_space.shape, (3, 2))
        self.assertEqual(venv.action_space.shape, (2, 3, 2))
        self.assertEqual(venv.observation_space["camera"].shape, (2, 3, *CAMERA_SHAPE))
        self.assertEqual(venv.observation_space["health"].shape, (2, 3))
        venv.close()

    def test_reset_returns_observations_for_each_instance_and_agent(self):
        venv = self._make(2, env_kwargs={"number_of_agents": 3})

        obs, _ = venv.reset()

        self.assertEqual(obs["camera"].shape, (2, 3, *CAMERA_SHAPE))
        np.testing.assert_allclose(obs["health"], [[100.0, 99.0, 98.0]] * 2)
        venv.close()

    def test_step_returns_the_reward_of_each_instance_and_each_agent(self):
        venv = self._make(2)
        venv.reset()

        obs, rewards, terminations, truncations, infos = venv.step(np.zeros((2, 2, 2), dtype=np.int64))

        np.testing.assert_allclose(infos["agent_rewards"], [[0.0, 0.1], [1.0, 1.1]], rtol=1e-6)
        np.testing.assert_allclose(rewards, [0.1, 2.1], rtol=1e-6)
        np.testing.assert_array_equal(terminations, [False, False])
        np.testing.assert_array_equal(truncations, [False, False])
        np.testing.assert_allclose(obs["health"], [[100.0, 99.0]] * 2)
        venv.close()

    def test_each_action_reaches_its_instance_and_agent(self):
        venv = self._make(2)
        venv.reset()
        venv.step(np.zeros((2, 2, 2), dtype=np.int64))
        actions = np.array([[[1, 0], [2, 0]], [[0, 1], [0, 2]]])

        venv.step(actions)

        # In the second exchange Unity sends the agents in the order [1, 0]
        first, second = FactoryScriptedAAIEnv.instances
        np.testing.assert_array_equal(first.set_actions.call_args.args[1].discrete, [[2, 0], [1, 0]])
        np.testing.assert_array_equal(second.set_actions.call_args.args[1].discrete, [[0, 2], [0, 1]])
        venv.close()

    def test_autoreset_starts_the_next_episode_with_no_unity_reset(self):
        venv = self._make(2)
        noops = np.zeros((2, 2, 2), dtype=np.int64)
        venv.reset()
        venv.step(noops)

        _, _, final_terminations, _, _ = venv.step(noops)
        obs, rewards, terminations, _, infos = venv.step(noops)
        _, next_rewards, _, _, _ = venv.step(noops)

        np.testing.assert_array_equal(final_terminations, [True, True])
        np.testing.assert_array_equal(terminations, [False, False])
        np.testing.assert_array_equal(rewards, [0.0, 0.0])
        np.testing.assert_array_equal(infos["agent_rewards"], np.zeros((2, 2)))
        np.testing.assert_allclose(obs["health"], [[100.0, 99.0]] * 2)
        np.testing.assert_allclose(next_rewards, [0.1, 2.1], rtol=1e-6)
        for instance in FactoryScriptedAAIEnv.instances:
            instance.reset.assert_not_called()
            self.assertEqual(instance.set_actions.call_count, 3)
        venv.close()

    def test_close_closes_each_unity_env(self):
        venv = self._make(2)

        venv.close()

        self.assertTrue(all(instance.closed for instance in FactoryScriptedAAIEnv.instances))

    def test_async_mode_is_the_default_and_steps_in_subprocesses(self):
        venv = vector.make_animalai_multi_agent_vec_env(2)
        try:
            self.assertIsInstance(venv, gymnasium.vector.AsyncVectorEnv)
            obs, _ = venv.reset()
            _, rewards, _, _, infos = venv.step(np.zeros((2, 2, 2), dtype=np.int64))
            self.assertEqual(obs["camera"].shape, (2, 2, *CAMERA_SHAPE))
            np.testing.assert_allclose(infos["agent_rewards"], [[0.0, 0.1], [1.0, 1.1]], rtol=1e-6)
            np.testing.assert_allclose(rewards, [0.1, 2.1], rtol=1e-6)
        finally:
            venv.close()

    def test_wrappers_package_exports_the_factory(self):
        from animalai import wrappers

        self.assertIs(wrappers.make_animalai_multi_agent_vec_env, vector.make_animalai_multi_agent_vec_env)
        self.assertIn("make_animalai_multi_agent_vec_env", wrappers.__all__)


if __name__ == "__main__":
    unittest.main()
