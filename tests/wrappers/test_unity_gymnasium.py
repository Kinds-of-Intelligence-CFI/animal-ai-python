import unittest
from unittest.mock import MagicMock

import numpy as np
from gymnasium import spaces

from animalai.wrappers.unity_gymnasium import (
    ActionFlattener,
    UnityGymnasiumException,
    UnityToGymnasiumWrapper,
)


# ---------------------------------------------------------------------------
# Fakes for the mlagents_envs BaseEnv API (no Unity binary involved)
# ---------------------------------------------------------------------------

class _ObsSpec:
    def __init__(self, shape):
        self.shape = shape


class _ActionSpec:
    def __init__(self, discrete_branches=(3, 3), continuous_size=0):
        self.discrete_branches = discrete_branches
        self.continuous_size = continuous_size

    @property
    def discrete_size(self):
        return len(self.discrete_branches)

    def is_discrete(self):
        return self.continuous_size == 0

    def is_continuous(self):
        return self.continuous_size > 0


class _BehaviorSpec:
    def __init__(self, observation_specs, action_spec):
        self.observation_specs = observation_specs
        self.action_spec = action_spec


class _Steps:
    def __init__(self, obs, reward, n=1, interrupted=None):
        self.obs = obs
        self.reward = reward
        self._n = n
        if interrupted is not None:
            self.interrupted = interrupted

    def __len__(self):
        return self._n


class _FakeEnv:
    def __init__(self, behavior_spec, decision_steps, terminal_steps, name="Brain?team=0"):
        self.behavior_specs = {name: behavior_spec}
        self._decision = decision_steps
        self._terminal = terminal_steps
        self.set_actions = MagicMock()
        self.step = MagicMock()
        self.reset = MagicMock()
        self.closed = False

    def get_steps(self, name):
        return self._decision, self._terminal

    def close(self):
        self.closed = True


def _camera_obs(h=4, w=4, c=3, val=0.5):
    return np.full((1, h, w, c), val, dtype=np.float32)


def _vector_obs(values):
    return np.array([values], dtype=np.float32)


def _empty_terminal():
    return _Steps(obs=[], reward=[], n=0)


def _make_vector_env(n_decision=1):
    spec = _BehaviorSpec([_ObsSpec((7,))], _ActionSpec())
    obs = [_vector_obs([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])]
    decision = _Steps(obs=obs, reward=[0.0], n=n_decision)
    return _FakeEnv(spec, decision, _empty_terminal())


def _make_visual_env(uint8_val=0.5):
    spec = _BehaviorSpec([_ObsSpec((4, 4, 3))], _ActionSpec())
    decision = _Steps(obs=[_camera_obs(val=uint8_val)], reward=[0.0], n=1)
    return _FakeEnv(spec, decision, _empty_terminal())


def _make_multi_env():
    spec = _BehaviorSpec([_ObsSpec((4, 4, 3)), _ObsSpec((7,))], _ActionSpec())
    obs = [_camera_obs(), _vector_obs([0.0] * 7)]
    decision = _Steps(obs=obs, reward=[0.0], n=1)
    return _FakeEnv(spec, decision, _empty_terminal())


# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------

class TestActionSpace(unittest.TestCase):
    def test_default_is_multidiscrete(self):
        wrapper = UnityToGymnasiumWrapper(_make_vector_env())
        self.assertIsInstance(wrapper.action_space, spaces.MultiDiscrete)
        np.testing.assert_array_equal(wrapper.action_space.nvec, [3, 3])

    def test_flatten_branched_is_discrete(self):
        wrapper = UnityToGymnasiumWrapper(_make_vector_env(), flatten_branched=True)
        self.assertIsInstance(wrapper.action_space, spaces.Discrete)
        self.assertEqual(wrapper.action_space.n, 9)

    def test_action_space_seeded(self):
        wrapper = UnityToGymnasiumWrapper(_make_vector_env(), action_space_seed=123)
        first = wrapper.action_space.sample()
        wrapper.action_space.seed(123)
        np.testing.assert_array_equal(wrapper.action_space.sample(), first)


class TestActionFlattener(unittest.TestCase):
    def test_lookup_maps_scalar_to_branches(self):
        flattener = ActionFlattener([3, 3])
        self.assertEqual(flattener.action_space.n, 9)
        self.assertEqual(flattener.lookup_action(0), [0, 0])
        self.assertEqual(flattener.lookup_action(4), [1, 1])
        self.assertEqual(flattener.lookup_action(8), [2, 2])


# ---------------------------------------------------------------------------
# Observation space
# ---------------------------------------------------------------------------

class TestObservationSpace(unittest.TestCase):
    def test_vector_only(self):
        wrapper = UnityToGymnasiumWrapper(_make_vector_env())
        self.assertIsInstance(wrapper.observation_space, spaces.Box)
        self.assertEqual(wrapper.observation_space.shape, (7,))

    def test_visual_only_float(self):
        wrapper = UnityToGymnasiumWrapper(_make_visual_env())
        self.assertEqual(wrapper.observation_space.shape, (4, 4, 3))
        self.assertEqual(wrapper.observation_space.dtype, np.float32)

    def test_visual_only_uint8(self):
        wrapper = UnityToGymnasiumWrapper(_make_visual_env(), uint8_visual=True)
        self.assertEqual(wrapper.observation_space.dtype, np.uint8)
        self.assertEqual(wrapper.observation_space.high.max(), 255)

    def test_multi_default_returns_first_visual(self):
        wrapper = UnityToGymnasiumWrapper(_make_multi_env())
        self.assertIsInstance(wrapper.observation_space, spaces.Box)
        self.assertEqual(wrapper.observation_space.shape, (4, 4, 3))

    def test_multi_allow_multiple_returns_tuple(self):
        wrapper = UnityToGymnasiumWrapper(_make_multi_env(), allow_multiple_obs=True)
        self.assertIsInstance(wrapper.observation_space, spaces.Tuple)
        self.assertEqual(len(wrapper.observation_space.spaces), 2)


# ---------------------------------------------------------------------------
# reset / step
# ---------------------------------------------------------------------------

class TestResetStep(unittest.TestCase):
    def test_reset_returns_obs_and_info(self):
        wrapper = UnityToGymnasiumWrapper(_make_vector_env())
        obs, info = wrapper.reset()
        self.assertEqual(obs.shape, (7,))
        self.assertIn("step", info)

    def test_reset_seeds_action_space(self):
        wrapper = UnityToGymnasiumWrapper(_make_vector_env())
        obs, _ = wrapper.reset(seed=42)
        first = wrapper.action_space.sample()
        wrapper.reset(seed=42)
        np.testing.assert_array_equal(wrapper.action_space.sample(), first)

    def test_step_decision_returns_5_tuple(self):
        wrapper = UnityToGymnasiumWrapper(_make_vector_env())
        result = wrapper.step([1, 0])
        self.assertEqual(len(result), 5)
        obs, reward, terminated, truncated, info = result
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(reward, 0.0)

    def test_step_terminated_when_not_interrupted(self):
        env = _make_vector_env()
        env._terminal = _Steps(
            obs=[_vector_obs([1.0] * 7)], reward=[2.5], n=1, interrupted=[False]
        )
        wrapper = UnityToGymnasiumWrapper(env)
        _, reward, terminated, truncated, _ = wrapper.step([1, 0])
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(reward, 2.5)

    def test_step_truncated_when_interrupted(self):
        env = _make_vector_env()
        env._terminal = _Steps(
            obs=[_vector_obs([1.0] * 7)], reward=[0.0], n=1, interrupted=[True]
        )
        wrapper = UnityToGymnasiumWrapper(env)
        _, _, terminated, truncated, _ = wrapper.step([1, 0])
        self.assertFalse(terminated)
        self.assertTrue(truncated)

    def test_step_sends_action_tuple(self):
        env = _make_vector_env()
        wrapper = UnityToGymnasiumWrapper(env)
        wrapper.step([2, 1])
        env.set_actions.assert_called()
        _, action_tuple = env.set_actions.call_args.args
        np.testing.assert_array_equal(action_tuple.discrete, [[2, 1]])


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

class TestMisc(unittest.TestCase):
    def test_no_observations_raises(self):
        spec = _BehaviorSpec([], _ActionSpec())
        env = _FakeEnv(spec, _Steps(obs=[], reward=[0.0], n=1), _empty_terminal())
        with self.assertRaises(UnityGymnasiumException):
            UnityToGymnasiumWrapper(env)

    def test_close_delegates(self):
        env = _make_vector_env()
        wrapper = UnityToGymnasiumWrapper(env)
        wrapper.close()
        self.assertTrue(env.closed)

    def test_render_returns_latest_visual(self):
        wrapper = UnityToGymnasiumWrapper(_make_visual_env())
        wrapper.reset()
        frame = wrapper.render()
        self.assertEqual(frame.shape, (4, 4, 3))


if __name__ == "__main__":
    unittest.main()
