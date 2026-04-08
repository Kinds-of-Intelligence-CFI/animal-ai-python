


from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import moviepy
from PIL import Image

from animalai.environment import AnimalAIEnvironment
from mlagents_envs.base_env import ActionTuple




class EnvironmentScaffold(ABC):
    """Abstract base class for environment scaffolds which handle turning actions from an llm (whatever the harness) into actions in AAI and return some results."""
    def __init__(self, env: AnimalAIEnvironment):
        self.env = env

    @abstractmethod
    def step(self, action: str) -> tuple[Any, float, bool, dict]:
        """Takes an action and returns (obs, reward, done, info) following gym convention."""
        raise NotImplementedError
    
    @abstractmethod
    def reset(self) -> tuple[Any, dict]:
        """Resets the environment and returns the initial observation."""
        raise NotImplementedError
    
    @abstractmethod
    def is_finished(self) -> bool:
        raise NotImplementedError
    
    def close(self):
        """Closes the environment."""
        self.env.close()

    @property
    def available_actions(self) -> list[str]:
        """Returns a list of available actions that the LLM can choose from."""
        raise NotImplementedError
    
    @property
    def default_system_prompt(self) -> str:
        """Returns a default system prompt to prime the LLM with."""
        raise NotImplementedError
    
    @property
    def total_reward(self) -> float:
        """Returns the total reward accumulated so far."""
        raise NotImplementedError
    


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
You will be controlling an agent in the Animal-AI environment.
Your goal is to navigate toward GREEN spheres (positive reward)
and avoid RED spheres (negative reward).

Available actions: {actions}

You can select exactly one action per turn.

Think step by step before making a decision.

The GREEN goal gives you a positive reward.  Move TOWARD it.
The RED goal gives you a negative reward.  AVOID it.

If the green sphere is on your left, TURN_LEFT first, then go FORWARD.
If the green sphere is on your right, TURN_RIGHT first, then go FORWARD.

Do not just keep turning in circles — go FORWARD when the goal
is roughly centred in your view.

Analyse the effects of each move.  After thinking, select a single
action and display its full name.
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

class FrameByFrameScaffold(EnvironmentScaffold):
    """Scaffold that provides a frame-by-frame interface for llms to act in animal ai."""
    def __init__(self, env: AnimalAIEnvironment):
        super().__init__(env)
        self.frames: list[np.ndarray] = []
        self.total_steps = 0
        self._total_reward = 0.0
        self._done = False

        self.behavior_name = list(self.env.behavior_specs.keys())[0]
        self._collect_obs()

    def _collect_obs(self) -> np.ndarray:
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        if len(terminal_steps) > 0:
            obs = _process_obs(terminal_steps.obs[0][0])
            self._done = True
        else:
            obs = _process_obs(decision_steps.obs[0][0])
        self.frames.append(obs)
        return obs

    @property
    def available_actions(self) -> list[str]:
        return AVAILABLE_ACTIONS
    
    @property
    def default_system_prompt(self) -> str:
        return START_PROMPT.format(actions=self.available_actions)

    def step(self, action: str, skipframe: int = 3) -> tuple[Any, float, bool, dict]:
        if isinstance(action, str):
            branch0, branch1 = ACTION_MAP[action]
        else:
            branch0, branch1 = action
    
        reward = 0.0
        for _ in range(skipframe):
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
                self.frames.append(obs)
                self._done = True
            elif len(decision_steps) > 0:
                reward += decision_steps.reward[0]
                obs = _process_obs(decision_steps.obs[0][0])
                self.frames.append(obs)
    
        self._total_reward += reward
        self.total_steps += 1
        return self.frames[-1], reward, self._done, {}
    
    def reset(self) -> tuple[Any, dict]:
        self.frames = []
        self.total_steps = 0
        self._total_reward = 0.0
        self._done = False
        self._collect_obs()
        return self.frames[-1], {}

    def is_finished(self) -> bool:
        return self._done

    def display(self) -> Image.Image:
        return Image.fromarray(self.frames[-1])

    def display_video(self):
        video = moviepy.ImageSequenceClip(self.frames, fps=10)
        return moviepy.display_in_notebook(
            video, verbose=False, rd_kwargs=dict(logger=None)
        )

    def get_results(self) -> dict:
        return {"num_steps": self.total_steps, "total_reward": self._total_reward}

    def repeat_last_frame(self, n: int):
        self.frames += [self.frames[-1]] * n


