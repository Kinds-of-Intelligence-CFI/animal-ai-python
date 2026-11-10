from typing import Dict, List, Optional, Sequence, Union

import numpy as np

try:
    from gymnasium import spaces
except ImportError as exc:
    raise ImportError(
        "The gymnasium wrappers require the optional 'gymnasium' extra. "
        "Install it with: pip install animalai[gymnasium]"
    ) from exc

from mlagents_envs.base_env import DecisionSteps, TerminalSteps

from animalai.environment import AnimalAIEnvironment
from animalai.wrappers.unity_gymnasium import UnityToGymnasiumWrapper


def select_obs_keys(
    unity_env: AnimalAIEnvironment,
    *,
    include_camera: Optional[bool] = None,
    include_rays: Optional[bool] = None,
    include_health: bool = True,
    include_velocity: bool = True,
    include_position: bool = True,
) -> List[str]:
    """Return the observation streams to expose, in their fixed order.

    ``include_camera`` / ``include_rays`` set to ``None`` mean "include if the
    environment produces it". ``True`` for a stream that the environment does
    not produce raises ``ValueError``, as does a selection with no streams.
    """
    obs_keys: List[str] = []
    if _resolve_stream(include_camera, unity_env.useCamera, "include_camera", "useCamera"):
        obs_keys.append("camera")
    if _resolve_stream(include_rays, unity_env.useRayCasts, "include_rays", "useRayCasts"):
        obs_keys.append("rays")
    if include_health:
        obs_keys.append("health")
    if include_velocity:
        obs_keys.append("velocity")
    if include_position:
        obs_keys.append("position")
    if not obs_keys:
        raise ValueError(
            "At least one observation must be selected, but all include_* "
            "flags resolved to False."
        )
    return obs_keys


def build_obs_spaces(
    observation_specs: Sequence,
    obs_keys: Sequence[str],
    *,
    env_camera: bool,
    env_rays: bool,
    uint8_visual: bool,
) -> Dict[str, spaces.Box]:
    """Build one ``Box`` per selected stream for a single agent.

    ``env_camera`` / ``env_rays`` say which streams the environment produces.
    They set the position of each stream in ``observation_specs``, which is
    independent of the streams in ``obs_keys``.
    """
    idx = 0
    cam_shape = ray_shape = None
    if env_camera:
        cam_shape = observation_specs[idx].shape
        idx += 1
    if env_rays:
        ray_shape = observation_specs[idx].shape
        idx += 1

    builders = {
        "camera": lambda: (
            spaces.Box(0, 255, dtype=np.uint8, shape=cam_shape)
            if uint8_visual
            else spaces.Box(0, 1, dtype=np.float32, shape=cam_shape)
        ),
        "rays": lambda: spaces.Box(0, np.inf, dtype=np.float32, shape=ray_shape),
        "health": lambda: spaces.Box(0, np.inf, dtype=np.float32, shape=()),
        "velocity": lambda: spaces.Box(-np.inf, np.inf, dtype=np.float32, shape=(3,)),
        "position": lambda: spaces.Box(-np.inf, np.inf, dtype=np.float32, shape=(3,)),
    }
    return {key: builders[key]() for key in obs_keys}


def _resolve_stream(flag, produced, flag_name, env_flag_name) -> bool:
    if flag is None:
        return produced
    if flag and not produced:
        raise ValueError(
            f"{flag_name}=True but the environment was created with "
            f"{env_flag_name}=False, so it produces no such observation."
        )
    return bool(flag)


class AnimalAIGymnasiumWrapper(UnityToGymnasiumWrapper):
    """Animal AI specific Gymnasium wrapper.

    Unlike the generic ``UnityToGymnasiumWrapper``, which returns a single
    anonymous array, this wrapper lets you pick which observation streams to
    expose via the ``include_*`` flags. The available streams are ``camera``
    and ``rays`` (only if the underlying ``AnimalAIEnvironment`` was created
    with ``useCamera`` / ``useRayCasts``) plus the intrinsic vector, split into
    the always-available ``health``, ``velocity`` and ``position``.

    When more than one stream is selected the observation is a ``spaces.Dict``
    keyed by stream name. When exactly one stream is selected the observation
    collapses to that stream's bare ``Box`` (and ``reset`` / ``step`` return the
    raw array), which is convenient for single-input policies such as
    Stable-Baselines3's ``CnnPolicy``. Note that selecting ``health`` on its own
    yields a scalar ``Box`` of shape ``()``.

    The ``camera`` / ``rays`` flags default to ``None``, meaning "include if the
    environment produces it"; setting one to ``True`` when the environment does
    not produce that stream raises ``ValueError``. ``health`` / ``velocity`` /
    ``position`` default to ``True``. The default construction therefore matches
    the previous behaviour (full dict of every available stream).

    This is the recommended wrapper for Animal-AI. The wrapped ``unity_env``
    must be an ``AnimalAIEnvironment``.
    """

    def __init__(
        self,
        unity_env: AnimalAIEnvironment,
        *,
        include_camera: Optional[bool] = None,
        include_rays: Optional[bool] = None,
        include_health: bool = True,
        include_velocity: bool = True,
        include_position: bool = True,
        **kwargs,
    ):
        if not isinstance(unity_env, AnimalAIEnvironment):
            raise TypeError(
                "AnimalAIGymnasiumWrapper requires an AnimalAIEnvironment. "
                "Use UnityToGymnasiumWrapper for other Unity environments."
            )

        self._env : AnimalAIEnvironment

        # Whether the env actually produces these streams (drives spec order).
        self._env_camera = unity_env.useCamera
        self._env_rays = unity_env.useRayCasts

        self._obs_keys = select_obs_keys(
            unity_env,
            include_camera=include_camera,
            include_rays=include_rays,
            include_health=include_health,
            include_velocity=include_velocity,
            include_position=include_position,
        )
        self.useCamera = "camera" in self._obs_keys
        self.useRayCasts = "rays" in self._obs_keys
        self._single_obs = len(self._obs_keys) == 1

        super().__init__(unity_env, **kwargs)

    def _build_observation_space(self):
        obs_spaces = build_obs_spaces(
            self.group_spec.observation_specs,
            self._obs_keys,
            env_camera=self._env_camera,
            env_rays=self._env_rays,
            uint8_visual=self.uint8_visual,
        )
        if self._single_obs:
            return obs_spaces[self._obs_keys[0]]
        return spaces.Dict(obs_spaces)

    def _get_obs(
        self, info: Union[DecisionSteps, TerminalSteps]
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        obs_dict = self._env.get_obs_dict(info.obs)
        out: dict = {}
        for key in self._obs_keys:
            if key == "camera":
                out["camera"] = self._preprocess_single(obs_dict["camera"])
                self.visual_obs = out["camera"]
            else:
                out[key] = np.asarray(obs_dict[key], dtype=np.float32)
        if self._single_obs:
            return out[self._obs_keys[0]]
        return out

    def _reset_unity_env(self, options: Optional[dict]) -> None:
        arena = options.get("arena_config") if options else None
        if arena is not None:
            self._env.reset(arenas_configurations=arena)
        else:
            self._env.reset()
