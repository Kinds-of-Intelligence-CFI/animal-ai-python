from typing import Any, Optional, Union

import numpy as np

try:
    from gymnasium import spaces
except ImportError as exc:
    raise ImportError(
        "The gymnasium wrappers require the optional 'gym' extra. "
        "Install it with: pip install animalai[gym]"
    ) from exc

from mlagents_envs.base_env import DecisionSteps, TerminalSteps

from animalai.environment import AnimalAIEnvironment
from animalai.wrappers.unity_gymnasium import UnityToGymnasiumWrapper


class AnimalAIGymnasiumWrapper(UnityToGymnasiumWrapper):
    """Animal AI specific Gymnasium wrapper.

    Unlike the generic ``UnityToGymnasiumWrapper``, which returns a single
    anonymous array, this wrapper returns a dictionary observation matching
    ``AnimalAIEnvironment.get_obs_dict``. The keys present depend on the
    environment's ``useCamera`` / ``useRayCasts`` flags; the intrinsic vector
    (health, velocity, position) is always present.

    This is the recommended wrapper for Animal-AI. The wrapped ``unity_env``
    must be an ``AnimalAIEnvironment``.
    """

    def __init__(self, unity_env: AnimalAIEnvironment, **kwargs):
        if not isinstance(unity_env, AnimalAIEnvironment):
            raise TypeError(
                "AnimalAIGymnasiumWrapper requires an AnimalAIEnvironment. "
                "Use UnityToGymnasiumWrapper for other Unity environments."
            )
        self.useCamera = unity_env.useCamera
        self.useRayCasts = unity_env.useRayCasts
        super().__init__(unity_env, **kwargs)

    def _build_observation_space(self) -> spaces.Dict:
        specs = self.group_spec.observation_specs
        obs_spaces: dict = {}
        idx = 0
        if self.useCamera:
            cam_shape = specs[idx].shape
            if self.uint8_visual:
                obs_spaces["camera"] = spaces.Box(
                    0, 255, dtype=np.uint8, shape=cam_shape
                )
            else:
                obs_spaces["camera"] = spaces.Box(
                    0, 1, dtype=np.float32, shape=cam_shape
                )
            idx += 1
        if self.useRayCasts:
            ray_shape = specs[idx].shape
            obs_spaces["rays"] = spaces.Box(
                0, np.inf, dtype=np.float32, shape=ray_shape
            )
            idx += 1
        # The intrinsic vector observation (size 7): [health, velocity(3), position(3)].
        obs_spaces["health"] = spaces.Box(0, np.inf, dtype=np.float32, shape=())
        obs_spaces["velocity"] = spaces.Box(
            -np.inf, np.inf, dtype=np.float32, shape=(3,)
        )
        obs_spaces["position"] = spaces.Box(
            -np.inf, np.inf, dtype=np.float32, shape=(3,)
        )
        return spaces.Dict(obs_spaces)

    def _get_obs(self, info: Union[DecisionSteps, TerminalSteps]) -> dict:
        obs_dict = self._env.get_obs_dict(info.obs)
        out: dict = {}
        if self.useCamera:
            out["camera"] = self._preprocess_single(obs_dict["camera"])
            self.visual_obs = out["camera"]
        if self.useRayCasts:
            out["rays"] = np.asarray(obs_dict["rays"], dtype=np.float32)
        out["health"] = np.asarray(obs_dict["health"], dtype=np.float32)
        out["velocity"] = np.asarray(obs_dict["velocity"], dtype=np.float32)
        out["position"] = np.asarray(obs_dict["position"], dtype=np.float32)
        return out
