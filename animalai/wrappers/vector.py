"""Factory helpers for running several Animal-AI environments in parallel.

Animal-AI already supports many simultaneous Unity instances: each
``AnimalAIEnvironment`` forwards ``worker_id`` to the underlying
``UnityEnvironment`` and binds to port ``base_port + worker_id`` as its own OS
process. The gymnasium wrapper, however, is strictly single-environment. These
helpers bridge the gap by building N ``AnimalAIEnvironment`` + wrapper pairs --
each with a unique ``worker_id`` (and ``seed``) so ports do not collide -- and
composing them with Gymnasium's built-in vector envs.
"""

import os
import tempfile
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

try:
    import gymnasium
    from filelock import FileLock
except ImportError as exc:
    raise ImportError(
        "The gymnasium wrappers require the optional 'gymnasium' extra. "
        "Install it with: pip install animalai[gymnasium]"
    ) from exc

from animalai.environment import AnimalAIEnvironment
from animalai.wrappers.animalai_gymnasium import AnimalAIGymnasiumWrapper

# Owned by the factory; supplying them via env_kwargs would clash with the
# per-environment values the factory assigns.
_RESERVED_ENV_KWARGS = ("worker_id", "base_port", "seed", "arenas_configurations")

ArenaConfig = Union[str, Sequence[str]]


def make_animalai_env_fns(
    num_envs: int,
    *,
    worker_id_start: int = 0,
    base_port: int = 5005,
    seed: int = 0,
    arenas_configurations: ArenaConfig = "",
    env_kwargs: Optional[Dict[str, Any]] = None,
    wrapper_kwargs: Optional[Dict[str, Any]] = None,
    serialize_launch: bool = True,
    launch_lock_timeout: float = 300.0,
    launch_lock_path: Optional[str] = None,
) -> List[Callable[[], AnimalAIGymnasiumWrapper]]:
    """Build ``num_envs`` zero-argument env factories ("thunks").

    Each thunk constructs an ``AnimalAIEnvironment`` with a unique
    ``worker_id`` (``worker_id_start + i``) and ``seed`` (``seed + i``), then
    wraps it in an :class:`AnimalAIGymnasiumWrapper`. The thunks are lazy:
    nothing launches Unity until a thunk is called, which is what lets a vector
    env decide when (and, for async, in which process) to construct each env.

    Parameters
    ----------
    num_envs : int
        Number of environments / thunks to create. Must be >= 1.
    worker_id_start : int
        ``worker_id`` of the first environment; subsequent envs increment from
        it. Set this to avoid colliding with other Animal-AI instances already
        running against the same ``base_port``.
    base_port : int
        Base port shared by all environments. Each env's actual port is
        ``base_port + worker_id``.
    seed : int
        Seed of the first environment; subsequent envs increment from it so
        rollouts differ across the batch.
    arenas_configurations : str | Sequence[str]
        Either a single arena YAML path shared by every environment, or a
        sequence of length ``num_envs`` giving one config per environment.
    env_kwargs : dict, optional
        Extra keyword arguments forwarded to every ``AnimalAIEnvironment``
        (e.g. ``file_name``, ``useCamera``, ``useRayCasts``, ``no_graphics``,
        ``timescale``). Must not contain any of ``worker_id``, ``base_port``,
        ``seed`` or ``arenas_configurations`` -- those are owned by the factory.
    wrapper_kwargs : dict, optional
        Extra keyword arguments forwarded to every ``AnimalAIGymnasiumWrapper``
        (e.g. the ``include_*`` observation-stream flags).
    serialize_launch : bool
        When ``True`` (default), each thunk acquires a cross-process file lock
        while it constructs its environment, so only one Unity instance goes
        through the launch-and-handshake at a time. This prevents concurrent
        launches (notably under ``async`` on Windows) from contending and
        exceeding the handshake timeout. The lock is released once the
        environment has connected, so stepping still runs fully in parallel.
        Set to ``False`` to launch all environments simultaneously.
    launch_lock_timeout : float
        Seconds to wait to acquire the launch lock before raising, guarding
        against a hung launch blocking the rest of the batch indefinitely.
    launch_lock_path : str, optional
        Path of the lock file shared by this batch's thunks. Defaults to
        ``<tempdir>/animalai_launch_<base_port>.lock``; keying on ``base_port``
        lets independent vector envs launch without blocking each other.
    """
    if num_envs < 1:
        raise ValueError(f"num_envs must be >= 1, got {num_envs}.")

    env_kwargs = dict(env_kwargs or {})
    wrapper_kwargs = dict(wrapper_kwargs or {})

    reserved = [key for key in _RESERVED_ENV_KWARGS if key in env_kwargs]
    if reserved:
        raise ValueError(
            "env_kwargs must not contain factory-owned keys "
            f"{reserved}; pass them as arguments to make_animalai_env_fns instead."
        )

    arena_configs = _resolve_arena_configs(arenas_configurations, num_envs)

    if serialize_launch and launch_lock_path is None:
        launch_lock_path = os.path.join(
            tempfile.gettempdir(), f"animalai_launch_{base_port}.lock"
        )

    def _make_thunk(index: int) -> Callable[[], AnimalAIGymnasiumWrapper]:
        def _thunk() -> AnimalAIGymnasiumWrapper:
            launch_guard = (
                FileLock(launch_lock_path, timeout=launch_lock_timeout)
                if serialize_launch
                else nullcontext()
            )
            with launch_guard:
                env = AnimalAIEnvironment(
                    worker_id=worker_id_start + index,
                    base_port=base_port,
                    seed=seed + index,
                    arenas_configurations=arena_configs[index],
                    **env_kwargs,
                )
                return AnimalAIGymnasiumWrapper(env, **wrapper_kwargs)

        return _thunk

    return [_make_thunk(i) for i in range(num_envs)]


def make_animalai_vec_env(
    num_envs: int,
    *,
    vectorization_mode: str = "async",
    **kwargs: Any,
) -> "gymnasium.vector.VectorEnv":
    """Build a Gymnasium vector env wrapping ``num_envs`` Animal-AI envs.

    A thin convenience over :func:`make_animalai_env_fns`: it builds the thunks
    and hands them to Gymnasium's vector machinery.

    Parameters
    ----------
    num_envs : int
        Number of environments to run in parallel.
    vectorization_mode : {"sync", "async"}
        ``"sync"`` steps the environments sequentially in this process
        via ``SyncVectorEnv``. ``"async"`` (default) runs each environment in its own
        subprocess via ``AsyncVectorEnv`` for step-time parallelism.
    **kwargs
        Forwarded to :func:`make_animalai_env_fns` (``worker_id_start``,
        ``base_port``, ``seed``, ``arenas_configurations``, ``env_kwargs``,
        ``wrapper_kwargs``, ``serialize_launch``, ``launch_lock_timeout``,
        ``launch_lock_path``).
    """
    if vectorization_mode not in ("sync", "async"):
        raise ValueError(
            f"vectorization_mode must be 'sync' or 'async', got {vectorization_mode!r}."
        )

    env_fns = make_animalai_env_fns(num_envs, **kwargs)

    if vectorization_mode == "async":
        return gymnasium.vector.AsyncVectorEnv(env_fns)
    return gymnasium.vector.SyncVectorEnv(env_fns)


def _resolve_arena_configs(arenas_configurations: ArenaConfig, num_envs: int) -> List[str]:
    if isinstance(arenas_configurations, str):
        return [arenas_configurations] * num_envs

    configs = list(arenas_configurations)
    if len(configs) != num_envs:
        raise ValueError(
            f"arenas_configurations has {len(configs)} entries but num_envs is "
            f"{num_envs}; pass a single string or one config per environment."
        )
    return configs
