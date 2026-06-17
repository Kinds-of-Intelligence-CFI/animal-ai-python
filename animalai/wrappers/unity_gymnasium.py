import itertools
from typing import Any, List, Optional, Tuple, Union

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import error, spaces
except ImportError as exc:
    raise ImportError(
        "The gymnasium wrappers require the optional 'gym' extra. "
        "Install it with: pip install animalai[gym]"
    ) from exc

from mlagents_envs.base_env import ActionTuple, BaseEnv
from mlagents_envs.base_env import DecisionSteps, TerminalSteps
from mlagents_envs import logging_util


class UnityGymnasiumException(error.Error):
    """Any error related to the gymnasium wrapper of ml-agents."""

    pass


logger = logging_util.get_logger(__name__)

# (observation, reward, terminated, truncated, info)
GymStepResult = Tuple[Any, float, bool, bool, dict]


class UnityToGymnasiumWrapper(gym.Env):
    """Wrapper that converts a Unity BaseEnv into a gymnasium.Env.

    Based off the UnityToGymWrapper in the ml-agents repo, but updated to work with gymnasium so we aren't version locked.

    This is the generic wrapper that should work for all Unity environments.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        unity_env: BaseEnv,
        uint8_visual: bool = False,
        flatten_branched: bool = False,
        allow_multiple_obs: bool = False,
        action_space_seed: Optional[int] = None,
        render_mode: Optional[str] = "rgb_array",
    ):
        """
        Environment initialization.

        Args:
          unity_env: The Unity BaseEnv to wrap. It is closed when this wrapper closes.
          uint8_visual: Return visual observations as uint8 (0-255) instead of float (0.0-1.0).
          flatten_branched: If True, turn a branched discrete action space into a single
            Discrete space rather than a MultiDiscrete space.
          allow_multiple_obs: If True, return a list of np.ndarrays as observations, with the
            leading elements containing the visual observations and the final element containing
            the vector observations. If False, return a single np.ndarray containing either a
            single visual observation or the vector observation.
          action_space_seed: If non-None, used to seed the created action space.
          render_mode: The render mode. Only "rgb_array" is supported.
        """
        self._env = unity_env

        # Take a single step so that the brain information will be sent over.
        if not self._env.behavior_specs:
            self._env.step()

        self.render_mode = render_mode
        self.visual_obs = None
        self.uint8_visual = uint8_visual

        self._previous_decision_step: Optional[DecisionSteps] = None
        self._flattener = None
        self._allow_multiple_obs = allow_multiple_obs

        if len(self._env.behavior_specs) != 1:
            raise UnityGymnasiumException(
                "There can only be one behavior in a UnityEnvironment "
                "if it is wrapped in a gymnasium env."
            )

        self.name = list(self._env.behavior_specs.keys())[0]
        self.group_spec = self._env.behavior_specs[self.name]

        if self._get_n_vis_obs() == 0 and self._get_vec_obs_size() == 0:
            raise UnityGymnasiumException(
                "There are no observations provided by the environment."
            )

        if not self._get_n_vis_obs() >= 1 and uint8_visual:
            logger.warning(
                "uint8_visual was set to true, but visual observations are not in use. "
                "This setting will not have any effect."
            )

        # Check for number of agents in scene.
        self._env.reset()
        decision_steps, _ = self._env.get_steps(self.name)
        self._check_agents(len(decision_steps))
        self._previous_decision_step = decision_steps

        self.action_space = self._build_action_space(flatten_branched)
        if action_space_seed is not None:
            self.action_space.seed(action_space_seed)

        self.observation_space = self._build_observation_space()

    def _build_action_space(self, flatten_branched: bool) -> gym.Space:
        if self.group_spec.action_spec.is_discrete():
            self.action_size = self.group_spec.action_spec.discrete_size
            branches = self.group_spec.action_spec.discrete_branches
            if self.group_spec.action_spec.discrete_size == 1:
                return spaces.Discrete(branches[0])
            if flatten_branched:
                self._flattener = ActionFlattener(branches)
                return self._flattener.action_space
            return spaces.MultiDiscrete(branches)

        if self.group_spec.action_spec.is_continuous():
            if flatten_branched:
                logger.warning(
                    "The environment has a non-discrete action space. It will "
                    "not be flattened."
                )
            self.action_size = self.group_spec.action_spec.continuous_size
            high = np.ones(self.group_spec.action_spec.continuous_size, dtype=np.float32)
            return spaces.Box(-high, high, dtype=np.float32)

        raise UnityGymnasiumException(
            "The gymnasium wrapper does not provide explicit support for both "
            "discrete and continuous actions."
        )

    def _build_observation_space(self) -> gym.Space:
        list_spaces: List[gym.Space] = []
        shapes = self._get_vis_obs_shape()
        for shape in shapes:
            if self.uint8_visual:
                list_spaces.append(spaces.Box(0, 255, dtype=np.uint8, shape=shape))
            else:
                list_spaces.append(spaces.Box(0, 1, dtype=np.float32, shape=shape))
        if self._get_vec_obs_size() > 0:
            # Vector observation is last.
            high = np.full(self._get_vec_obs_size(), np.inf, dtype=np.float32)
            list_spaces.append(spaces.Box(-high, high, dtype=np.float32))
        if self._allow_multiple_obs:
            return spaces.Tuple(list_spaces)
        if len(list_spaces) >= 2:
            logger.warning(
                "The environment contains multiple observations. "
                "You must define allow_multiple_obs=True to receive them all. "
                "Otherwise, only the first visual observation (or the vector "
                "observation if there are no visual observations) will be provided."
            )
        return list_spaces[0]  # only return the first one

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Any, dict]:
        """Reset the environment and return the initial observation and info dict."""
        super().reset(seed=seed)
        if seed is not None:
            self.action_space.seed(seed)

        self._env.reset()
        decision_step, _ = self._env.get_steps(self.name)
        n_agents = len(decision_step)
        self._check_agents(n_agents)

        obs = self._get_obs(decision_step)
        return obs, {"step": decision_step}

    def step(self, action: List[Any]) -> GymStepResult:
        """Run one timestep and return (obs, reward, terminated, truncated, info)."""
        if self._flattener is not None:
            # Translate the discrete scalar action into a branched action list.
            action = self._flattener.lookup_action(action)

        action = np.array(action).reshape((1, self.action_size))

        action_tuple = ActionTuple()
        if self.group_spec.action_spec.is_continuous():
            action_tuple.add_continuous(action)
        else:
            action_tuple.add_discrete(action)
        self._env.set_actions(self.name, action_tuple)

        self._env.step()
        decision_step, terminal_step = self._env.get_steps(self.name)
        self._check_agents(max(len(decision_step), len(terminal_step)))
        if len(terminal_step) != 0:
            return self._single_step(terminal_step, terminal=True)
        return self._single_step(decision_step, terminal=False)

    def _single_step(
        self, info: Union[DecisionSteps, TerminalSteps], terminal: bool
    ) -> GymStepResult:
        obs = self._get_obs(info)
        if terminal:
            # interrupted == hit a step limit (truncation); otherwise a real terminal state.
            interrupted = bool(info.interrupted[0])
            terminated = not interrupted
            truncated = interrupted
        else:
            terminated = False
            truncated = False
        return obs, float(info.reward[0]), terminated, truncated, {"step": info}

    def _get_obs(self, info: Union[DecisionSteps, TerminalSteps]) -> Any:
        """Build the observation for the current step.

        Subclasses can override this to change the observation format while
        reusing the reward / terminated / truncated logic in ``_single_step``.
        """
        if self._allow_multiple_obs:
            visual_obs = self._get_vis_obs_list(info)
            default_observation: Any = [
                self._preprocess_single(obs[0]) for obs in visual_obs
            ]
            if self._get_vec_obs_size() >= 1:
                default_observation.append(self._get_vector_obs(info)[0, :])
        else:
            if self._get_n_vis_obs() >= 1:
                visual_obs = self._get_vis_obs_list(info)
                default_observation = self._preprocess_single(visual_obs[0][0])
            else:
                default_observation = self._get_vector_obs(info)[0, :]

        if self._get_n_vis_obs() >= 1:
            visual_obs = self._get_vis_obs_list(info)
            self.visual_obs = self._preprocess_single(visual_obs[0][0])

        return default_observation

    def _preprocess_single(self, single_visual_obs: np.ndarray) -> np.ndarray:
        if self.uint8_visual:
            return (255.0 * single_visual_obs).astype(np.uint8)
        return single_visual_obs

    def _get_n_vis_obs(self) -> int:
        result = 0
        for obs_spec in self.group_spec.observation_specs:
            if len(obs_spec.shape) == 3:
                result += 1
        return result

    def _get_vis_obs_shape(self) -> List[Tuple]:
        result: List[Tuple] = []
        for obs_spec in self.group_spec.observation_specs:
            if len(obs_spec.shape) == 3:
                result.append(obs_spec.shape)
        return result

    def _get_vis_obs_list(
        self, step_result: Union[DecisionSteps, TerminalSteps]
    ) -> List[np.ndarray]:
        result: List[np.ndarray] = []
        for obs in step_result.obs:
            if len(obs.shape) == 4:
                result.append(obs)
        return result

    def _get_vector_obs(
        self, step_result: Union[DecisionSteps, TerminalSteps]
    ) -> np.ndarray:
        result: List[np.ndarray] = []
        for obs in step_result.obs:
            if len(obs.shape) == 2:
                result.append(obs)
        return np.concatenate(result, axis=1)

    def _get_vec_obs_size(self) -> int:
        result = 0
        for obs_spec in self.group_spec.observation_specs:
            if len(obs_spec.shape) == 1:
                result += obs_spec.shape[0]
        return result

    def render(self):
        """Return the latest visual observation.

        Note that this does not render a new frame of the environment.
        """
        return self.visual_obs

    def close(self) -> None:
        """Close the wrapped Unity environment."""
        self._env.close()

    @property
    def reward_range(self) -> Tuple[float, float]:
        return -float("inf"), float("inf")

    @staticmethod
    def _check_agents(n_agents: int) -> None:
        if n_agents > 1:
            raise UnityGymnasiumException(
                f"There can only be one Agent in the environment but {n_agents} were detected."
            )


class ActionFlattener:
    """Flattens branched discrete action spaces into single-branch discrete spaces."""

    def __init__(self, branched_action_space):
        """
        Args:
          branched_action_space: A list of the sizes of each branch of the action
            space, e.g. [2, 3, 3] for three branches of size 2, 3 and 3.
        """
        self._action_shape = branched_action_space
        self.action_lookup = self._create_lookup(self._action_shape)
        self.action_space = spaces.Discrete(len(self.action_lookup))

    @classmethod
    def _create_lookup(cls, branched_action_space):
        """Create a dict mapping discrete scalar actions to branched action lists."""
        possible_vals = [range(_num) for _num in branched_action_space]
        all_actions = [list(_action) for _action in itertools.product(*possible_vals)]
        action_lookup = {
            _scalar: _action for (_scalar, _action) in enumerate(all_actions)
        }
        return action_lookup

    def lookup_action(self, action):
        """Convert a scalar discrete action into its branched action list."""
        return self.action_lookup[action]
