from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from gymnasium import spaces
    from pettingzoo import ParallelEnv
except ImportError as exc:
    raise ImportError(
        "The PettingZoo wrapper requires the optional 'gymnasium' extra. "
        "Install it with: pip install animalai[gymnasium]"
    ) from exc

from animalai.environment import AnimalAIEnvironment
from animalai.wrappers.multi_agent import BatchedObservation, MultiAgentCore


class AnimalAIParallelEnv(ParallelEnv):
    """PettingZoo ``ParallelEnv`` for an arena with several agents.

    The agents are named ``agent_0``, ``agent_1``, ... in the order of the
    ``Agent`` items of the arena. Each agent has the observation space and the
    action space that ``AnimalAIGymnasiumWrapper`` gives to a single agent, and
    the keyword arguments (``include_*``, ``uint8_visual``) are the same.

    All agents must be on one team, and the number of agents must not change
    while the wrapper is in use. All agents finish an episode in the same step.
    After that step ``agents`` is empty until the next ``reset``.

    Unity builds the next arena itself when an episode ends, so ``reset`` after
    an episode end does not reset Unity and no arena is skipped. Give
    ``options={"arena_config": path}`` to ``reset`` to load a new configuration
    file.
    """

    metadata = {
        "name": "animalai_multi_agent_v0",
        "render_modes": ["rgb_array"],
        "is_parallelizable": True,
    }

    def __init__(self, unity_env: AnimalAIEnvironment, **core_kwargs: Any):
        self._core = MultiAgentCore(unity_env, **core_kwargs)
        self.possible_agents: List[str] = list(self._core.agent_names)
        self.agents: List[str] = list(self.possible_agents)
        self.render_mode = "rgb_array"

    def observation_space(self, agent: str) -> spaces.Space:
        self._check_agent_name(agent)
        return self._core.single_observation_space

    def action_space(self, agent: str) -> spaces.Space:
        self._check_agent_name(agent)
        return self._core.single_action_space

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Dict[str, Any], Dict[str, dict]]:
        """Start an episode and return the observation and info of each agent.

        ``seed`` seeds the action space only. The seed of the Unity environment
        is set when the ``AnimalAIEnvironment`` is created.
        """
        if seed is not None:
            self._core.single_action_space.seed(seed)
        observations, _ = self._core.reset(options=options)
        self.agents = list(self.possible_agents)
        return self._by_agent(observations), {agent: {} for agent in self.agents}

    def step(self, actions: Dict[str, Any]) -> Tuple[
        Dict[str, Any], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, dict]
    ]:
        """Send one action for each agent in ``agents`` and return five dicts keyed by agent."""
        missing = [agent for agent in self.agents if agent not in actions]
        unexpected = [agent for agent in actions if agent not in self.agents]
        if missing or unexpected:
            raise ValueError(
                "step needs exactly one action for each agent in `agents`. "
                f"Missing: {missing}. Unexpected: {unexpected}."
            )

        agents = self.agents
        observations, rewards, terminated, truncated, _ = self._core.step(
            [actions[agent] for agent in agents]
        )
        if (terminated | truncated).all():
            self.agents = []
        return (
            self._by_agent(observations),
            {agent: float(rewards[k]) for k, agent in enumerate(agents)},
            {agent: bool(terminated[k]) for k, agent in enumerate(agents)},
            {agent: bool(truncated[k]) for k, agent in enumerate(agents)},
            {agent: {} for agent in agents},
        )

    def render(self) -> Optional[np.ndarray]:
        """Return the latest camera frame of ``agent_0``, or None if the camera is not selected.

        This does not render a new frame of the environment.
        """
        if self._core.camera_frames is None:
            return None
        return self._core.camera_frames[0]

    def close(self) -> None:
        """Close the Unity environment."""
        self._core.close()

    def _check_agent_name(self, agent: str) -> None:
        if agent not in self.possible_agents:
            raise KeyError(f"Unknown agent {agent!r}. The agents are {self.possible_agents}.")

    def _by_agent(self, observations: BatchedObservation) -> Dict[str, Any]:
        if isinstance(observations, dict):
            return {
                agent: {key: np.asarray(rows[k]) for key, rows in observations.items()}
                for k, agent in enumerate(self.possible_agents)
            }
        return {agent: np.asarray(observations[k]) for k, agent in enumerate(self.possible_agents)}
