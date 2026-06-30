from typing import Dict, Optional, Union

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

        self.useCamera = self._resolve_stream(
            include_camera, self._env_camera, "include_camera", "useCamera"
        )
        self.useRayCasts = self._resolve_stream(
            include_rays, self._env_rays, "include_rays", "useRayCasts"
        )

        self._obs_keys: list = []
        if self.useCamera:
            self._obs_keys.append("camera")
        if self.useRayCasts:
            self._obs_keys.append("rays")
        if include_health:
            self._obs_keys.append("health")
        if include_velocity:
            self._obs_keys.append("velocity")
        if include_position:
            self._obs_keys.append("position")
        if not self._obs_keys:
            raise ValueError(
                "At least one observation must be selected, but all include_* "
                "flags resolved to False."
            )
        self._single_obs = len(self._obs_keys) == 1

        super().__init__(unity_env, **kwargs)

    @staticmethod
    def _resolve_stream(flag, produced, flag_name, env_flag_name) -> bool:
        if flag is None:
            return produced
        if flag and not produced:
            raise ValueError(
                f"{flag_name}=True but the environment was created with "
                f"{env_flag_name}=False, so it produces no such observation."
            )
        return bool(flag)

    def _build_observation_space(self):
        specs = self.group_spec.observation_specs
        # Spec order follows what the env produces, not what we expose.
        idx = 0
        cam_shape = ray_shape = None
        if self._env_camera:
            cam_shape = specs[idx].shape
            idx += 1
        if self._env_rays:
            ray_shape = specs[idx].shape
            idx += 1

        builders = {
            "camera": lambda: (
                spaces.Box(0, 255, dtype=np.uint8, shape=cam_shape)
                if self.uint8_visual
                else spaces.Box(0, 1, dtype=np.float32, shape=cam_shape)
            ),
            "rays": lambda: spaces.Box(0, np.inf, dtype=np.float32, shape=ray_shape),
            "health": lambda: spaces.Box(0, np.inf, dtype=np.float32, shape=()),
            "velocity": lambda: spaces.Box(-np.inf, np.inf, dtype=np.float32, shape=(3,)),
            "position": lambda: spaces.Box(-np.inf, np.inf, dtype=np.float32, shape=(3,)),
        }
        obs_spaces = {key: builders[key]() for key in self._obs_keys}
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
