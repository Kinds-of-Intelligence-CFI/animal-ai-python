import unittest
from unittest.mock import MagicMock

import numpy as np
from gymnasium import spaces

from animalai.environment import AnimalAIEnvironment
from animalai.wrappers.animalai_gymnasium import AnimalAIGymnasiumWrapper


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _ObsSpec:
    def __init__(self, shape):
        self.shape = shape


class _ActionSpec:
    discrete_branches = (3, 3)
    continuous_size = 0

    @property
    def discrete_size(self):
        return len(self.discrete_branches)

    def is_discrete(self):
        return True

    def is_continuous(self):
        return False


class _BehaviorSpec:
    def __init__(self, observation_specs):
        self.observation_specs = observation_specs
        self.action_spec = _ActionSpec()


class _Steps:
    def __init__(self, obs, reward, n=1, interrupted=None):
        self.obs = obs
        self.reward = reward
        self._n = n
        if interrupted is not None:
            self.interrupted = interrupted

    def __len__(self):
        return self._n


class _FakeAAIEnv(AnimalAIEnvironment):
    """An AnimalAIEnvironment that skips the real Unity launch.

    Subclassing the real class satisfies the wrapper's isinstance check, while
    the inherited get_obs_dict() runs against the fake step observations.
    """

    def __init__(self, observation_specs, decision, terminal, useCamera, useRayCasts):
        self.useCamera = useCamera
        self.useRayCasts = useRayCasts
        self.obsdict = {
            "camera": [],
            "rays": [],
            "health": [],
            "velocity": [],
            "position": [],
        }
        self._specs = {"Brain?team=0": _BehaviorSpec(observation_specs)}
        self._decision = decision
        self._terminal = terminal
        self.set_actions = MagicMock()
        self.step = MagicMock()
        self.reset = MagicMock()
        self.closed = False

    @property
    def behavior_specs(self):
        return self._specs

    def get_steps(self, name):
        return self._decision, self._terminal

    def close(self):
        self.closed = True


def _camera_batch(h=4, w=4, c=3, val=0.5):
    return np.full((1, h, w, c), val, dtype=np.float32)


def _vector_batch(health=5.0, velocity=(1.0, 2.0, 3.0), position=(7.0, 8.0, 9.0)):
    return np.array([[health, *velocity, *position]], dtype=np.float32)


def _rays_batch(length=6, val=0.3):
    return np.full((1, length), val, dtype=np.float32)


def _empty_terminal():
    return _Steps(obs=[], reward=[], n=0)


def _make_env(useCamera=True, useRayCasts=False, camera_val=0.5):
    specs = []
    obs = []
    if useCamera:
        specs.append(_ObsSpec((4, 4, 3)))
        obs.append(_camera_batch(val=camera_val))
    if useRayCasts:
        specs.append(_ObsSpec((6,)))
        obs.append(_rays_batch())
    specs.append(_ObsSpec((7,)))
    obs.append(_vector_batch())
    decision = _Steps(obs=obs, reward=[0.0], n=1)
    return _FakeAAIEnv(specs, decision, _empty_terminal(), useCamera, useRayCasts)


# ---------------------------------------------------------------------------
# Observation space
# ---------------------------------------------------------------------------

class TestObservationSpace(unittest.TestCase):
    def test_camera_only_keys(self):
        wrapper = AnimalAIGymnasiumWrapper(_make_env(useCamera=True, useRayCasts=False))
        self.assertIsInstance(wrapper.observation_space, spaces.Dict)
        self.assertEqual(
            set(wrapper.observation_space.spaces),
            {"camera", "health", "velocity", "position"},
        )

    def test_camera_and_rays_keys(self):
        wrapper = AnimalAIGymnasiumWrapper(_make_env(useCamera=True, useRayCasts=True))
        self.assertEqual(
            set(wrapper.observation_space.spaces),
            {"camera", "rays", "health", "velocity", "position"},
        )
        self.assertEqual(wrapper.observation_space["rays"].shape, (6,))

    def test_rays_only_keys(self):
        wrapper = AnimalAIGymnasiumWrapper(_make_env(useCamera=False, useRayCasts=True))
        self.assertEqual(
            set(wrapper.observation_space.spaces),
            {"rays", "health", "velocity", "position"},
        )

    def test_camera_space_shape_and_dtype(self):
        wrapper = AnimalAIGymnasiumWrapper(_make_env())
        cam = wrapper.observation_space["camera"]
        self.assertEqual(cam.shape, (4, 4, 3))
        self.assertEqual(cam.dtype, np.float32)

    def test_camera_uint8(self):
        wrapper = AnimalAIGymnasiumWrapper(_make_env(), uint8_visual=True)
        self.assertEqual(wrapper.observation_space["camera"].dtype, np.uint8)

    def test_vector_subspaces(self):
        wrapper = AnimalAIGymnasiumWrapper(_make_env())
        self.assertEqual(wrapper.observation_space["health"].shape, ())
        self.assertEqual(wrapper.observation_space["velocity"].shape, (3,))
        self.assertEqual(wrapper.observation_space["position"].shape, (3,))


# ---------------------------------------------------------------------------
# Observations from reset / step
# ---------------------------------------------------------------------------

class TestObservations(unittest.TestCase):
    def test_reset_returns_dict_obs(self):
        wrapper = AnimalAIGymnasiumWrapper(_make_env(useCamera=True, useRayCasts=True))
        obs, info = wrapper.reset()
        self.assertEqual(
            set(obs), {"camera", "rays", "health", "velocity", "position"}
        )
        self.assertTrue(wrapper.observation_space.contains(obs))

    def test_obs_values_parsed(self):
        wrapper = AnimalAIGymnasiumWrapper(_make_env())
        obs, _ = wrapper.reset()
        self.assertEqual(obs["health"].shape, ())
        np.testing.assert_array_equal(obs["health"], np.float32(5.0))
        np.testing.assert_array_equal(obs["velocity"], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(obs["position"], [7.0, 8.0, 9.0])

    def test_camera_uint8_conversion(self):
        wrapper = AnimalAIGymnasiumWrapper(_make_env(camera_val=0.5), uint8_visual=True)
        obs, _ = wrapper.reset()
        self.assertEqual(obs["camera"].dtype, np.uint8)
        self.assertEqual(int(obs["camera"].flat[0]), 127)

    def test_step_returns_dict_obs_in_5_tuple(self):
        wrapper = AnimalAIGymnasiumWrapper(_make_env())
        result = wrapper.step([1, 0])
        self.assertEqual(len(result), 5)
        obs, reward, terminated, truncated, info = result
        self.assertIn("camera", obs)
        self.assertFalse(terminated)
        self.assertFalse(truncated)


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

class TestGuards(unittest.TestCase):
    def test_rejects_non_aai_env(self):
        with self.assertRaises(TypeError):
            AnimalAIGymnasiumWrapper(MagicMock())


if __name__ == "__main__":
    unittest.main()
