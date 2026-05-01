from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np

from animalai.environment import AnimalAIEnvironment
from mlagents_envs.base_env import ActionTuple

ObsType = TypeVar("ObsType")


class EnvironmentScaffold(ABC, Generic[ObsType]):
    """Abstract base class for environment scaffolds which handle converting actions from an llm (whatever the harness) into actions in AAI and return some results."""
    def __init__(self, env: AnimalAIEnvironment):
        self.env = env

    @abstractmethod
    def step(self, action: str) -> tuple[ObsType, float, bool, dict]:
        """Takes an action and returns (obs, reward, done, info) following gym convention."""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> tuple[ObsType, dict]:
        """Resets the environment and returns the initial observation."""
        raise NotImplementedError
    
    @abstractmethod
    def is_finished(self) -> bool:
        raise NotImplementedError
    
    def close(self):
        """Closes the environment."""
        self.env.close()

    @property
    @abstractmethod
    def available_actions(self) -> list[str]:
        """Returns a list of available actions that the LLM can choose from."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_default_system_prompt() -> str:
        """Returns a default system prompt to prime the LLM with."""
        raise NotImplementedError

    @property
    @abstractmethod
    def total_reward(self) -> float:
        """Returns the total reward accumulated so far."""
        raise NotImplementedError
    


DEFAULT_LAST_FRAME: np.ndarray = np.array([])
DEFAULT_TOTAL_STEPS: int = 0
DEFAULT_TOTAL_REWARD: float = 0.0
DEFAULT_DONE: bool = False

ACTION_MAP = {
    "NOOP":          (0, 0),
    "FORWARD":       (1, 0),
    "BACKWARD":      (2, 0),
    "TURN_RIGHT":    (0, 1),
    "TURN_LEFT":     (0, 2),
    "FORWARD_RIGHT": (1, 1),
    "FORWARD_LEFT":  (1, 2),
}

AVAILABLE_ACTIONS = list(ACTION_MAP.keys())

START_PROMPT = """
You are a PLAYER in a game set in a square arena with a white fence. Your task is to collect all the rewards as quickly and efficiently as possible. The rewards are green and yellow balls.

To successfully collect a reward, you must fully pass through it.

The game ends when you have collected all the rewards and the arena closes. If you are still in the arena, the game is NOT finished and you have NOT collected all the rewards.

NOTE: When you collect a reward, your remaining health will INCREASE compared to the previous timestep. If it doesn't increase, the reward was not collected. Always compare your current health with the previous timestep to confirm this.\n\n

Be mindful of obstacles:

Red lava puddles and red balls: If you run into them, you will die.
Holes: Some may contain rewards, but if you fall into an empty hole, you will be trapped and unable to collect other rewards.
Blue paths: These are slightly raised paths. You can walk on them, but once you step off, you won't be able to get back onto them.
Purple ramps: You can climb them to get to the other side. Once you climb over the ramp, you cannot climb back over the same ramp.
Transparent walls: You can see through them, but you cannot walk through them.
Pushable grey blocks: These are cube-like structures, patterned with dark grey rectangles on each face. If viewed from one side, they will look like a rectangular structure. They can be pushed, but they are heavy! To move these blocks, you need to run into them.
Immovable objects: Walls and arches cannot be moved.\n

Available actions:
{actions}
"""

def _process_obs(obs: np.ndarray) -> np.ndarray:
    """Convert a raw Unity observation to uint8 (H, W, C)."""
    # Transpose (C, H, W) -> (H, W, C) if needed
    if obs.ndim == 3 and obs.shape[0] in (1, 3):
        obs = np.transpose(obs, (1, 2, 0))
    # Squeeze grayscale (H, W, 1) -> (H, W)
    if obs.ndim == 3 and obs.shape[-1] == 1:
        obs = obs.squeeze(-1)
    # float [0,1] -> uint8
    if obs.dtype != np.uint8:
        obs = (obs * 255).astype(np.uint8)
    return obs

class FrameByFrameScaffold(EnvironmentScaffold[np.ndarray]):
    """Scaffold that provides a frame-by-frame interface for llms to act in animal ai."""

    @staticmethod
    def get_default_system_prompt() -> str:
        return START_PROMPT.format(actions=AVAILABLE_ACTIONS)

    def __init__(self, env: AnimalAIEnvironment, skipframe: int = 1):
        super().__init__(env)
        assert skipframe >= 1, "skipframe must be at least 1"
        self.skipframe = skipframe
        self.last_frame: np.ndarray = DEFAULT_LAST_FRAME
        self._total_steps = DEFAULT_TOTAL_STEPS
        self._total_reward = DEFAULT_TOTAL_REWARD
        self._done = DEFAULT_DONE

        self.behavior_name = list(self.env.behavior_specs.keys())[0]
        self._collect_obs()

    def _collect_obs(self) -> np.ndarray:
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        if len(terminal_steps) > 0:
            obs = _process_obs(terminal_steps.obs[0][0])
            self._done = True
        else:
            obs = _process_obs(decision_steps.obs[0][0])
        self.last_frame = obs
        return obs

    @property
    def available_actions(self) -> list[str]:
        return AVAILABLE_ACTIONS

    @property
    def total_reward(self) -> float:
        return self._total_reward

    def step(self, action: str) -> tuple[np.ndarray, float, bool, dict]:
        branch0, branch1 = ACTION_MAP[action]
        reward = 0.0
        for _ in range(self.skipframe):
            if self._done:
                break
            decision_steps, _ = self.env.get_steps(self.behavior_name)
            n_agents = len(decision_steps)
            if n_agents == 0:
                self._done = True
                break
            discrete = np.array([[branch0, branch1]] * n_agents, dtype=np.int32)
            self.env.set_actions(self.behavior_name, ActionTuple(discrete=discrete))
            self.env.step()
    
            decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
            if len(terminal_steps) > 0:
                reward += terminal_steps.reward[0]
                obs = _process_obs(terminal_steps.obs[0][0])
                self.last_frame = obs
                self._done = True
            elif len(decision_steps) > 0:
                reward += decision_steps.reward[0]
                obs = _process_obs(decision_steps.obs[0][0])
                self.last_frame = obs
    
        self._total_reward += reward
        self._total_steps += 1
        return self.last_frame, reward, self._done, {}
    
    def reset(self) -> tuple[np.ndarray, dict]:
        self.last_frame = DEFAULT_LAST_FRAME
        self._total_steps = DEFAULT_TOTAL_STEPS
        self._total_reward = DEFAULT_TOTAL_REWARD
        self._done = DEFAULT_DONE
        self.env.reset()
        self._collect_obs()
        return self.last_frame, {}

    def is_finished(self) -> bool:
        return self._done

    def get_results(self) -> dict:
        return {"num_steps": self._total_steps, "total_reward": self._total_reward}



