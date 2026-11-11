"""Shared core of the multi-agent wrappers.

An arena can contain several ``Agent`` items. ``MultiAgentCore`` reads one such
``AnimalAIEnvironment`` and returns arrays with a leading agent dimension, in an
agent order that does not change.
"""

from typing import Dict, List, Optional, Tuple, Union

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
from animalai.wrappers.animalai_gymnasium import build_obs_spaces, select_obs_keys
from animalai.wrappers.unity_gymnasium import UnityGymnasiumException

BatchedObservation = Union[np.ndarray, Dict[str, np.ndarray]]


class MultiAgentCore:
    """Tracks the agents of one multi-agent ``AnimalAIEnvironment``.

    Agent ``k`` (``agent_k``) is the agent with the k-th smallest Unity agent id.
    Unity creates the agents in the order of the ``Agent`` items of the arena, so
    agent ``k`` is the k-th ``Agent`` item. All arrays that this class returns
    have one row per agent in this order, whatever row order Unity uses.

    The observation streams and the ``include_*`` / ``uint8_visual`` arguments
    are the same as for ``AnimalAIGymnasiumWrapper``.
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
        uint8_visual: bool = False,
    ):
        if not isinstance(unity_env, AnimalAIEnvironment):
            raise TypeError("The multi-agent wrappers require an AnimalAIEnvironment.")
        self._env = unity_env
        self.uint8_visual = uint8_visual

        behaviors = list(unity_env.behavior_specs.keys())
        if len(behaviors) != 1:
            raise UnityGymnasiumException(
                "The multi-agent wrappers support one behavior only, but found "
                f"{behaviors}. In Animal-AI, agents on different `teams` have "
                "different behaviors. Put all agents on one team, or use the "
                "environment directly."
            )
        self.behavior_name = behaviors[0]
        behavior_spec = unity_env.behavior_specs[self.behavior_name]

        self._obs_keys = select_obs_keys(
            unity_env,
            include_camera=include_camera,
            include_rays=include_rays,
            include_health=include_health,
            include_velocity=include_velocity,
            include_position=include_position,
        )
        self._single_obs = len(self._obs_keys) == 1
        obs_spaces = build_obs_spaces(
            behavior_spec.observation_specs,
            self._obs_keys,
            env_camera=unity_env.useCamera,
            env_rays=unity_env.useRayCasts,
            uint8_visual=uint8_visual,
        )
        self.single_observation_space: spaces.Space = (
            obs_spaces[self._obs_keys[0]] if self._single_obs else spaces.Dict(obs_spaces)
        )

        if not behavior_spec.action_spec.is_discrete():
            raise UnityGymnasiumException(
                "The multi-agent wrappers support discrete actions only."
            )
        self.single_action_space = spaces.MultiDiscrete(
            behavior_spec.action_spec.discrete_branches
        )

        decision_steps, _ = unity_env.get_steps(self.behavior_name)
        if len(decision_steps) == 0:
            raise UnityGymnasiumException(
                "No agent waiting for actions."
            )
        self._decision_steps = decision_steps
        self.agent_ids: Tuple[int, ...] = tuple(
            sorted(int(agent_id) for agent_id in self._decision_steps.agent_id)
        )
        self.num_agents = len(self.agent_ids)
        self.agent_names: List[str] = [f"agent_{k}" for k in range(self.num_agents)]

    def reset(self) -> Tuple[BatchedObservation, dict]:
        """Return the observations of all agents at the start of the episode."""
        return self._observations(self._decision_steps), {}

    def _row_order(self, steps: Union[DecisionSteps, TerminalSteps]) -> List[int]:
        """The Unity row of each agent, in agent order."""
        row_of = steps.agent_id_to_index
        return [row_of[agent_id] for agent_id in self.agent_ids]

    def _observations(self, steps: Union[DecisionSteps, TerminalSteps]) -> BatchedObservation:
        rows = self._env.get_obs_dicts(steps.obs)
        order = self._row_order(steps)
        batch: Dict[str, np.ndarray] = {}
        for key in self._obs_keys:
            stacked = np.stack([rows[row][key] for row in order])
            if key == "camera" and self.uint8_visual:
                batch[key] = (255.0 * stacked).astype(np.uint8)
            else:
                batch[key] = stacked.astype(np.float32, copy=False)
        if self._single_obs:
            return batch[self._obs_keys[0]]
        return batch
