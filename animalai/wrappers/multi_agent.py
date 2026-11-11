"""Shared core of the multi-agent wrappers.

An arena can contain several ``Agent`` items. ``MultiAgentCore`` reads one such
``AnimalAIEnvironment`` and returns arrays with a leading agent dimension, in an
agent order that does not change.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import gymnasium
    from gymnasium import spaces
    from gymnasium.vector.utils import batch_space
except ImportError as exc:
    raise ImportError(
        "The gymnasium wrappers require the optional 'gymnasium' extra. "
        "Install it with: pip install animalai[gymnasium]"
    ) from exc

from mlagents_envs.base_env import ActionTuple, DecisionSteps, TerminalSteps

from animalai.environment import AnimalAIEnvironment
from animalai.wrappers.animalai_gymnasium import build_obs_spaces, select_obs_keys
from animalai.wrappers.unity_gymnasium import UnityGymnasiumException

BatchedObservation = Union[np.ndarray, Dict[str, np.ndarray]]
# (observations, rewards, terminated, truncated, info)
StepResult = Tuple[BatchedObservation, np.ndarray, np.ndarray, np.ndarray, dict]


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
        self._episode_started = False
        self._episode_over = False
        # The latest camera frame of each agent, or None if the camera is not selected.
        self.camera_frames: Optional[np.ndarray] = None

    def reset(self, *, options: Optional[dict] = None) -> Tuple[BatchedObservation, dict]:
        """Return the observations of all agents at the start of an episode.

        Unity moves on to the next arena each time it resets, and it resets
        itself when an episode ends. ``reset`` thus resets Unity only if the
        caller stops an episode before its end, or gives a new configuration
        file in ``options={"arena_config": path}``.
        """
        arena_config = options.get("arena_config") if options else None
        if arena_config is not None:
            self._env.reset(arenas_configurations=arena_config)
            self._read_decision_steps()
        elif self._episode_over:
            if len(self._decision_steps) == 0:
                self._env.step()
                self._read_decision_steps()
        elif self._episode_started:
            self._env.reset()
            self._read_decision_steps()

        order = self._row_order(self._decision_steps)
        self._episode_over = False
        self._episode_started = False
        return self._observations(self._decision_steps, order), {}

    def step(self, actions) -> StepResult:
        """Send one action per agent and return the result for each agent.

        ``actions`` has shape ``(num_agents, 2)``, one row per agent in agent
        order. The result is ``(observations, rewards, terminated, truncated,
        info)``. The rewards and the two flags have shape ``(num_agents,)``.
        All agents of an arena finish in the same step.
        """
        if self._episode_over:
            raise UnityGymnasiumException(
                "The episode is finished. Call reset() before the next step."
            )
        self._send_actions(actions)
        self._env.step()
        decision_steps, terminal_steps = self._env.get_steps(self.behavior_name)

        if len(terminal_steps) > 0:
            order = self._row_order(terminal_steps)
            # These decision steps are the start of the next episode, if Unity
            # sent it in the same exchange. reset() reads them.
            self._decision_steps = decision_steps
            self._episode_over = True
            truncated = terminal_steps.interrupted[order].astype(np.bool_)
            return self._result(terminal_steps, order, ~truncated, truncated)

        order = self._row_order(decision_steps)
        self._decision_steps = decision_steps
        self._episode_started = True
        not_done = np.zeros(self.num_agents, dtype=np.bool_)
        return self._result(decision_steps, order, not_done, not_done.copy())

    def close(self) -> None:
        """Close the Unity environment."""
        self._env.close()

    def _read_decision_steps(self) -> None:
        self._decision_steps, _ = self._env.get_steps(self.behavior_name)

    def _send_actions(self, actions) -> None:
        actions = np.asarray(actions)
        expected_shape = (self.num_agents, *self.single_action_space.shape)
        if actions.shape != expected_shape:
            raise ValueError(
                f"Expected actions with shape {expected_shape} (one row per "
                f"agent), but got shape {actions.shape}."
            )
        if not np.issubdtype(actions.dtype, np.integer):
            raise ValueError(f"Actions must be integers, but got dtype {actions.dtype}.")
        for name, action in zip(self.agent_names, actions):
            if not self.single_action_space.contains(action):
                raise ValueError(
                    f"The action {action.tolist()} of {name} is not in "
                    f"{self.single_action_space}."
                )

        unity_rows = np.empty(expected_shape, dtype=np.int32)
        unity_rows[self._row_order(self._decision_steps)] = actions
        self._env.set_actions(self.behavior_name, ActionTuple(discrete=unity_rows))

    def _result(
        self,
        steps: Union[DecisionSteps, TerminalSteps],
        order: List[int],
        terminated: np.ndarray,
        truncated: np.ndarray,
    ) -> StepResult:
        rewards = steps.reward[order].astype(np.float32)
        return self._observations(steps, order), rewards, terminated, truncated, {}

    def _row_order(self, steps: Union[DecisionSteps, TerminalSteps]) -> List[int]:
        """The Unity row of each agent, in agent order.

        Raises if the agents in ``steps`` are not exactly the agents that the
        wrapper found at construction.
        """
        received = sorted(int(agent_id) for agent_id in steps.agent_id)
        if received != list(self.agent_ids):
            raise UnityGymnasiumException(
                f"The agents changed. The wrapper expects the agent ids "
                f"{list(self.agent_ids)}, but the {type(steps).__name__} from Unity "
                f"has the agent ids {received}. The wrapper needs the same agents "
                "in each step, and all agents must finish an episode together. A "
                "probable cause is a configuration file in which the arenas have "
                "different numbers of Agent items."
            )
        row_of = steps.agent_id_to_index
        return [row_of[agent_id] for agent_id in self.agent_ids]

    def _observations(
        self, steps: Union[DecisionSteps, TerminalSteps], order: List[int]
    ) -> BatchedObservation:
        rows = self._env.get_obs_dicts(steps.obs)
        batch: Dict[str, np.ndarray] = {}
        for key in self._obs_keys:
            stacked = np.stack([rows[row][key] for row in order])
            if key == "camera" and self.uint8_visual:
                batch[key] = (255.0 * stacked).astype(np.uint8)
            else:
                batch[key] = stacked.astype(np.float32, copy=False)
        self.camera_frames = batch.get("camera")
        if self._single_obs:
            return batch[self._obs_keys[0]]
        return batch


class AnimalAIAgentStackEnv(gymnasium.Env):
    """A ``gymnasium.Env`` in which each array has a leading agent dimension.

    It is the building block of ``make_animalai_multi_agent_vec_env``. For one
    Unity instance, ``AnimalAIParallelEnv`` is the recommended wrapper.

    The observation space and the action space are the spaces of one agent with
    ``num_agents`` rows, in the order of the ``Agent`` items of the arena. The
    keyword arguments (``include_*``, ``uint8_visual``) are the same as for
    ``AnimalAIGymnasiumWrapper``.

    A ``gymnasium.Env`` has one reward and one pair of flags for each step:
    - The reward is the sum of the agent rewards. ``info["agent_rewards"]`` has
      the reward of each agent.
    - ``terminated`` and ``truncated`` are single values, because all agents of
      an arena finish in the same step.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, unity_env: AnimalAIEnvironment, **core_kwargs: Any):
        self._core = MultiAgentCore(unity_env, **core_kwargs)
        self.num_agents = self._core.num_agents
        self.observation_space = batch_space(self._core.single_observation_space, self.num_agents)
        self.action_space = batch_space(self._core.single_action_space, self.num_agents)
        self.render_mode = "rgb_array"

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[BatchedObservation, dict]:
        """Start an episode. ``seed`` seeds the action space only."""
        super().reset(seed=seed)
        if seed is not None:
            self.action_space.seed(seed)
        observations, _ = self._core.reset(options=options)
        return observations, {}

    def step(self, actions) -> Tuple[BatchedObservation, float, bool, bool, dict]:
        """Send one action row per agent. ``actions`` has shape ``(num_agents, 2)``."""
        observations, rewards, terminated, truncated, _ = self._core.step(actions)
        info = {"agent_rewards": rewards}
        return observations, float(rewards.sum()), bool(terminated.any()), bool(truncated.any()), info

    def render(self) -> Optional[np.ndarray]:
        """Return the latest camera frame of the first agent, or None if the camera is not selected."""
        if self._core.camera_frames is None:
            return None
        return self._core.camera_frames[0]

    def close(self) -> None:
        """Close the Unity environment."""
        self._core.close()
