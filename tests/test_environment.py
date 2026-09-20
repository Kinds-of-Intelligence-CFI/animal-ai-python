import unittest

import numpy as np

from animalai.environment import AnimalAIEnvironment


def _make_env(useCamera: bool, useRayCasts: bool) -> AnimalAIEnvironment:
    """An AnimalAIEnvironment with just the state get_obs_dict needs (no Unity launch)."""
    env = AnimalAIEnvironment.__new__(AnimalAIEnvironment)
    env.useCamera = useCamera
    env.useRayCasts = useRayCasts
    env.obsdict = {"camera": [], "rays": [], "health": [], "velocity": [], "position": []}
    return env


def _vector(agent: int) -> list:
    """Intrinsic observation: health, velocity (3), position (3), distinct per agent."""
    return [100.0 - agent, agent + 0.1, agent + 0.2, agent + 0.3, 10.0 * agent, 1.0, 20.0]


def _obs(number_of_agents: int, useCamera: bool, useRayCasts: bool) -> list:
    """Observations as ML-Agents returns them: one array per sensor, one row per agent."""
    obs = []
    if useCamera:
        obs.append(np.stack([np.full((4, 4, 3), a / 10.0, dtype=np.float32) for a in range(number_of_agents)]))
    if useRayCasts:
        obs.append(np.stack([np.full(40, float(a), dtype=np.float32) for a in range(number_of_agents)]))
    obs.append(np.array([_vector(a) for a in range(number_of_agents)], dtype=np.float32))
    return obs


class TestGetObsDict(unittest.TestCase):
    def test_defaults_to_the_first_agent(self):
        env = _make_env(useCamera=True, useRayCasts=True)
        obs = _obs(2, useCamera=True, useRayCasts=True)

        obs_dict = env.get_obs_dict(obs)

        np.testing.assert_array_equal(obs_dict["camera"], obs[0][0])
        np.testing.assert_array_equal(obs_dict["rays"], obs[1][0])
        self.assertAlmostEqual(obs_dict["health"], 100.0)
        np.testing.assert_allclose(obs_dict["velocity"], [0.1, 0.2, 0.3])
        np.testing.assert_allclose(obs_dict["position"], [0.0, 1.0, 20.0])

    def test_reads_the_requested_agent(self):
        env = _make_env(useCamera=True, useRayCasts=True)
        obs = _obs(2, useCamera=True, useRayCasts=True)

        obs_dict = env.get_obs_dict(obs, agent_index=1)

        np.testing.assert_array_equal(obs_dict["camera"], obs[0][1])
        np.testing.assert_array_equal(obs_dict["rays"], obs[1][1])
        self.assertAlmostEqual(obs_dict["health"], 99.0)
        np.testing.assert_allclose(obs_dict["position"], [10.0, 1.0, 20.0])

    def test_keeps_updating_the_env_obsdict(self):
        env = _make_env(useCamera=False, useRayCasts=True)
        obsdict = env.obsdict

        returned = env.get_obs_dict(_obs(1, useCamera=False, useRayCasts=True))

        self.assertIs(returned, obsdict)
        self.assertIs(env.obsdict, obsdict)

    def test_rays_only_and_vector_only(self):
        rays_env = _make_env(useCamera=False, useRayCasts=True)
        rays_obs = _obs(2, useCamera=False, useRayCasts=True)
        np.testing.assert_array_equal(rays_env.get_obs_dict(rays_obs, 1)["rays"], rays_obs[0][1])

        vector_env = _make_env(useCamera=False, useRayCasts=False)
        vector_obs = _obs(2, useCamera=False, useRayCasts=False)
        self.assertAlmostEqual(vector_env.get_obs_dict(vector_obs, 1)["health"], 99.0)

    def test_agent_index_out_of_range_raises(self):
        env = _make_env(useCamera=False, useRayCasts=False)
        with self.assertRaises(IndexError):
            env.get_obs_dict(_obs(2, useCamera=False, useRayCasts=False), agent_index=2)


class TestGetObsDicts(unittest.TestCase):
    def test_returns_one_dict_per_agent_in_order(self):
        env = _make_env(useCamera=True, useRayCasts=True)
        obs = _obs(3, useCamera=True, useRayCasts=True)

        obs_dicts = env.get_obs_dicts(obs)

        self.assertEqual(len(obs_dicts), 3)
        for agent, obs_dict in enumerate(obs_dicts):
            np.testing.assert_array_equal(obs_dict["camera"], obs[0][agent])
            np.testing.assert_array_equal(obs_dict["rays"], obs[1][agent])
            self.assertAlmostEqual(obs_dict["health"], 100.0 - agent)

    def test_dicts_are_independent(self):
        env = _make_env(useCamera=False, useRayCasts=False)

        obs_dicts = env.get_obs_dicts(_obs(2, useCamera=False, useRayCasts=False))

        self.assertIsNot(obs_dicts[0], obs_dicts[1])
        self.assertIsNot(obs_dicts[0], env.obsdict)


if __name__ == "__main__":
    unittest.main()
