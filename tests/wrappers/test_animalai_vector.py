import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import gymnasium
from gymnasium import spaces

from animalai.environment import AnimalAIEnvironment
from animalai.wrappers import vector


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


def _camera_batch(h=4, w=4, c=3, val=0.5):
    return np.full((1, h, w, c), val, dtype=np.float32)


def _vector_batch():
    return np.array([[5.0, 1.0, 2.0, 3.0, 7.0, 8.0, 9.0]], dtype=np.float32)


def _rays_batch(length=6, val=0.3):
    return np.full((1, length), val, dtype=np.float32)


class _FakeAAIEnv(AnimalAIEnvironment):
    """An AnimalAIEnvironment that records its construction kwargs and skips the
    real Unity launch.

    Its __init__ mirrors the keyword arguments the factory passes through
    (worker_id / base_port / seed / arenas_configurations plus any env_kwargs),
    so tests can assert how the factory assigned them. Subclassing the real
    class satisfies the wrapper's isinstance check.
    """

    # Records every instance built during a test so assertions can inspect them.
    instances: list = []

    def __init__(
        self,
        *,
        worker_id=0,
        base_port=5005,
        seed=0,
        arenas_configurations="",
        useCamera=True,
        useRayCasts=False,
        **extra,
    ):
        self.worker_id = worker_id
        self.base_port = base_port
        self.seed_value = seed
        self.arenas_configurations = arenas_configurations
        self.extra = extra
        self.useCamera = useCamera
        self.useRayCasts = useRayCasts
        self.obsdict = {
            "camera": [],
            "rays": [],
            "health": [],
            "velocity": [],
            "position": [],
        }

        specs = []
        obs = []
        if useCamera:
            specs.append(_ObsSpec((4, 4, 3)))
            obs.append(_camera_batch())
        if useRayCasts:
            specs.append(_ObsSpec((6,)))
            obs.append(_rays_batch())
        specs.append(_ObsSpec((7,)))
        obs.append(_vector_batch())
        self._specs = {"Brain?team=0": _BehaviorSpec(specs)}
        self._decision = _Steps(obs=obs, reward=[0.0], n=1)
        self._terminal = _Steps(obs=[], reward=[], n=0)

        self.set_actions = MagicMock()
        self.step = MagicMock()
        self.reset = MagicMock()
        self.closed = False
        _FakeAAIEnv.instances.append(self)

    @property
    def behavior_specs(self):
        return self._specs

    def get_steps(self, name):
        return self._decision, self._terminal

    def close(self):
        self.closed = True


class _VectorTestCase(unittest.TestCase):
    def setUp(self):
        _FakeAAIEnv.instances = []
        # The factory constructs AnimalAIEnvironment by name; swap in the fake.
        self._real_env = vector.AnimalAIEnvironment
        vector.AnimalAIEnvironment = _FakeAAIEnv

    def tearDown(self):
        vector.AnimalAIEnvironment = self._real_env


# ---------------------------------------------------------------------------
# Thunk construction: make_animalai_env_fns
# ---------------------------------------------------------------------------

class TestEnvFns(_VectorTestCase):
    def test_returns_num_envs_callables(self):
        fns = vector.make_animalai_env_fns(3)
        self.assertEqual(len(fns), 3)
        self.assertTrue(all(callable(fn) for fn in fns))
        # Thunks are lazy: nothing built until called.
        self.assertEqual(_FakeAAIEnv.instances, [])

    def test_unique_worker_ids_and_seeds(self):
        fns = vector.make_animalai_env_fns(3, worker_id_start=10, seed=100)
        wrappers = [fn() for fn in fns]
        self.assertEqual(len(wrappers), 3)
        worker_ids = [w._env.worker_id for w in wrappers]
        seeds = [w._env.seed_value for w in wrappers]
        self.assertEqual(worker_ids, [10, 11, 12])
        self.assertEqual(seeds, [100, 101, 102])

    def test_base_port_shared(self):
        fns = vector.make_animalai_env_fns(2, base_port=6000)
        envs = [fn()._env for fn in fns]
        self.assertEqual([e.base_port for e in envs], [6000, 6000])

    def test_single_arena_config_shared(self):
        fns = vector.make_animalai_env_fns(2, arenas_configurations="a.yml")
        envs = [fn()._env for fn in fns]
        self.assertEqual([e.arenas_configurations for e in envs], ["a.yml", "a.yml"])

    def test_per_env_arena_config_list(self):
        fns = vector.make_animalai_env_fns(
            2, arenas_configurations=["a.yml", "b.yml"]
        )
        envs = [fn()._env for fn in fns]
        self.assertEqual([e.arenas_configurations for e in envs], ["a.yml", "b.yml"])

    def test_wrong_length_arena_list_raises(self):
        with self.assertRaises(ValueError):
            vector.make_animalai_env_fns(3, arenas_configurations=["a.yml", "b.yml"])

    def test_env_kwargs_forwarded(self):
        fns = vector.make_animalai_env_fns(
            1, env_kwargs={"useRayCasts": True, "no_graphics": True}
        )
        env = fns[0]()._env
        self.assertTrue(env.useRayCasts)
        self.assertEqual(env.extra.get("no_graphics"), True)

    def test_wrapper_kwargs_forwarded(self):
        fns = vector.make_animalai_env_fns(
            1,
            env_kwargs={"useRayCasts": True},
            wrapper_kwargs={
                "include_rays": True,
                "include_camera": False,
                "include_health": False,
                "include_velocity": False,
                "include_position": False,
            },
        )
        wrapper = fns[0]()
        # A single selected stream collapses to a bare Box.
        self.assertIsInstance(wrapper.observation_space, spaces.Box)
        self.assertEqual(wrapper.observation_space.shape, (6,))

    def test_timeout_wait_forwarded(self):
        fns = vector.make_animalai_env_fns(1, env_kwargs={"timeout_wait": 120})
        env = fns[0]()._env
        self.assertEqual(env.extra.get("timeout_wait"), 120)

    def test_reserved_keys_in_env_kwargs_raise(self):
        for key in ("worker_id", "base_port", "seed", "arenas_configurations"):
            with self.assertRaises(ValueError):
                vector.make_animalai_env_fns(1, env_kwargs={key: 1})

    def test_invalid_num_envs_raises(self):
        with self.assertRaises(ValueError):
            vector.make_animalai_env_fns(0)


# ---------------------------------------------------------------------------
# Vector env assembly: make_animalai_vec_env
# ---------------------------------------------------------------------------

class TestVecEnv(_VectorTestCase):
    def _make(self, num_envs=2, **kwargs):
        kwargs.setdefault("env_kwargs", {"useCamera": False})
        return vector.make_animalai_vec_env(num_envs, **kwargs)

    def test_defaults_to_async_vector_env(self):
        venv = self._make(2)
        self.assertIsInstance(venv, gymnasium.vector.AsyncVectorEnv)
        self.assertEqual(venv.num_envs, 2)
        venv.close()

    def test_sync_mode_selects_sync_env(self):
        venv = self._make(2, vectorization_mode="sync")
        self.assertIsInstance(venv, gymnasium.vector.SyncVectorEnv)
        self.assertEqual(venv.num_envs, 2)
        venv.close()

    def test_async_mode_selects_async_env(self):
        # Only assert the mode is validated/dispatched; do not spin up
        # subprocesses in the unit test.
        with self.assertRaises(ValueError):
            self._make(2, vectorization_mode="not-a-mode")

    def test_reset_returns_batched_obs(self):
        venv = self._make(2)
        obs, info = venv.reset()
        self.assertEqual(set(obs), {"health", "velocity", "position"})
        self.assertEqual(obs["velocity"].shape, (2, 3))
        self.assertEqual(obs["position"].shape, (2, 3))
        venv.close()

    def test_close_propagates(self):
        venv = self._make(2)
        envs = list(_FakeAAIEnv.instances)
        venv.close()
        self.assertTrue(all(e.closed for e in envs))


# ---------------------------------------------------------------------------
# Launch serialization: cross-process file lock around env construction
# ---------------------------------------------------------------------------

class _SpyLock:
    """Stand-in for filelock.FileLock that records its lifecycle.

    events interleaves lock enter/exit with env construction so a test can
    assert the env is built *while the lock is held*. calls records the
    (path, timeout) each FileLock was constructed with.
    """

    events: list = []
    calls: list = []

    def __init__(self, path, timeout=-1):
        _SpyLock.calls.append((str(path), timeout))
        self._path = str(path)

    def __enter__(self):
        _SpyLock.events.append(("enter", self._path))
        return self

    def __exit__(self, *exc):
        _SpyLock.events.append(("exit", self._path))
        return False


class _LoggingFakeEnv(_FakeAAIEnv):
    """A fake env that marks the moment of its construction on _SpyLock.events."""

    def __init__(self, **kwargs):
        _SpyLock.events.append(("construct", None))
        super().__init__(**kwargs)


class TestLaunchLock(_VectorTestCase):
    def setUp(self):
        super().setUp()
        _SpyLock.events = []
        _SpyLock.calls = []

    def test_lock_held_around_construction(self):
        vector.AnimalAIEnvironment = _LoggingFakeEnv
        with patch.object(vector, "FileLock", _SpyLock):
            fn = vector.make_animalai_env_fns(1, env_kwargs={"useCamera": False})[0]
            fn()
        self.assertEqual(
            [kind for kind, _ in _SpyLock.events],
            ["enter", "construct", "exit"],
        )

    def test_lock_path_keyed_on_base_port(self):
        with patch.object(vector, "FileLock", _SpyLock):
            vector.make_animalai_env_fns(1, base_port=6123)[0]()
        self.assertEqual(len(_SpyLock.calls), 1)
        path, _ = _SpyLock.calls[0]
        self.assertIn("animalai_launch_6123", path)

    def test_lock_timeout_forwarded(self):
        with patch.object(vector, "FileLock", _SpyLock):
            vector.make_animalai_env_fns(1, launch_lock_timeout=42.0)[0]()
        self.assertEqual(_SpyLock.calls[0][1], 42.0)

    def test_shared_path_across_thunks(self):
        with patch.object(vector, "FileLock", _SpyLock):
            for fn in vector.make_animalai_env_fns(3, base_port=7000):
                fn()
        paths = {path for path, _ in _SpyLock.calls}
        self.assertEqual(len(paths), 1)

    def test_custom_lock_path(self):
        with patch.object(vector, "FileLock", _SpyLock):
            vector.make_animalai_env_fns(
                1, launch_lock_path="/tmp/my.lock"
            )[0]()
        self.assertEqual(_SpyLock.calls[0][0], "/tmp/my.lock")

    def test_opt_out_skips_lock(self):
        with patch.object(vector, "FileLock", _SpyLock):
            vector.make_animalai_env_fns(1, serialize_launch=False)[0]()
        self.assertEqual(_SpyLock.calls, [])
        # Env is still built without the lock.
        self.assertEqual(len(_FakeAAIEnv.instances), 1)


if __name__ == "__main__":
    unittest.main()
