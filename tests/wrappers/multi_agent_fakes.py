"""Scripted stand-ins for a multi-agent Animal-AI environment. No Unity binary is involved."""

from typing import List, Optional, Sequence, Tuple
from unittest.mock import MagicMock

import numpy as np
from mlagents_envs.base_env import ActionSpec, BehaviorSpec, DecisionSteps, TerminalSteps

from animalai.environment import AnimalAIEnvironment

BEHAVIOR = "AnimalAI?team=0"
CAMERA_SHAPE = (4, 4, 3)
RAY_LENGTH = 40

Exchange = Tuple[DecisionSteps, TerminalSteps]


class _ObsSpec:
    def __init__(self, shape):
        self.shape = shape


def unity_id(agent: int) -> int:
    """The Unity id of agent ``agent``. The ids are not 0..M-1, so a test cannot pass by accident."""
    return 3 + 4 * agent


def agent_obs(agents: Sequence[int], useCamera: bool = True, useRayCasts: bool = True) -> List[np.ndarray]:
    """Observations as ML-Agents returns them: one array per sensor, one row per entry of ``agents``.

    The values in a row identify its agent: camera ``agent / 10``, rays ``agent``,
    health ``100 - agent``, velocity ``agent + (0.1, 0.2, 0.3)``, position ``(10 * agent, 1, 20)``.
    """
    numbers = np.asarray(agents, dtype=np.float32)
    count = len(numbers)
    obs = []
    if useCamera:
        obs.append(np.broadcast_to((numbers / 10.0).reshape(-1, 1, 1, 1), (count, *CAMERA_SHAPE)).copy())
    if useRayCasts:
        obs.append(np.broadcast_to(numbers.reshape(-1, 1), (count, RAY_LENGTH)).copy())
    vector = np.zeros((count, 7), dtype=np.float32)
    vector[:, 0] = 100.0 - numbers
    vector[:, 1:4] = numbers.reshape(-1, 1) + np.array([0.1, 0.2, 0.3], dtype=np.float32)
    vector[:, 4] = 10.0 * numbers
    vector[:, 5] = 1.0
    vector[:, 6] = 20.0
    obs.append(vector)
    return obs


def _ids(agents: Sequence[int], ids: Optional[Sequence[int]]) -> np.ndarray:
    return np.array([unity_id(a) for a in agents] if ids is None else ids, dtype=np.int32)


def _rewards(agents: Sequence[int], rewards: Optional[Sequence[float]]) -> np.ndarray:
    return np.zeros(len(agents), dtype=np.float32) if rewards is None else np.array(rewards, dtype=np.float32)


def decision_steps(
    agents: Sequence[int] = (),
    rewards: Optional[Sequence[float]] = None,
    ids: Optional[Sequence[int]] = None,
    **obs_kwargs,
) -> DecisionSteps:
    """A DecisionSteps with one row per entry of ``agents``, in that row order."""
    return DecisionSteps(
        obs=agent_obs(agents, **obs_kwargs),
        reward=_rewards(agents, rewards),
        agent_id=_ids(agents, ids),
        action_mask=None,
        group_id=np.zeros(len(agents), dtype=np.int32),
        group_reward=np.zeros(len(agents), dtype=np.float32),
    )


def terminal_steps(
    agents: Sequence[int] = (),
    rewards: Optional[Sequence[float]] = None,
    interrupted: bool = False,
    ids: Optional[Sequence[int]] = None,
    **obs_kwargs,
) -> TerminalSteps:
    """A TerminalSteps with one row per entry of ``agents``, in that row order."""
    return TerminalSteps(
        obs=agent_obs(agents, **obs_kwargs),
        reward=_rewards(agents, rewards),
        interrupted=np.full(len(agents), interrupted, dtype=bool),
        agent_id=_ids(agents, ids),
        group_id=np.zeros(len(agents), dtype=np.int32),
        group_reward=np.zeros(len(agents), dtype=np.float32),
    )


def deciding(agents: Sequence[int], rewards: Optional[Sequence[float]] = None, **obs_kwargs) -> Exchange:
    """An exchange in which the agents in ``agents`` request a decision and no agent finishes."""
    return decision_steps(agents, rewards, **obs_kwargs), terminal_steps(**obs_kwargs)


def silent(**obs_kwargs) -> Exchange:
    """An exchange with no agents at all."""
    return decision_steps(**obs_kwargs), terminal_steps(**obs_kwargs)


class ScriptedAAIEnv(AnimalAIEnvironment):
    """An AnimalAIEnvironment that plays back a script and starts no Unity process.

    ``script`` holds one ``(DecisionSteps, TerminalSteps)`` pair per exchange with Unity. The
    first pair is the state after construction. Each ``step`` or ``reset`` call moves on to
    the next pair. ``step``, ``reset`` and ``set_actions`` are mocks, so tests can read their calls.
    """

    def __init__(
        self,
        script: Sequence[Exchange],
        useCamera: bool = True,
        useRayCasts: bool = True,
        behaviors: Sequence[str] = (BEHAVIOR,),
    ):
        self.useCamera = useCamera
        self.useRayCasts = useRayCasts
        observation_specs = []
        if useCamera:
            observation_specs.append(_ObsSpec(CAMERA_SHAPE))
        if useRayCasts:
            observation_specs.append(_ObsSpec((RAY_LENGTH,)))
        observation_specs.append(_ObsSpec((7,)))
        spec = BehaviorSpec(observation_specs, ActionSpec.create_discrete((3, 3)))
        self._specs = {name: spec for name in behaviors}
        self._script = list(script)
        self._current = self._script.pop(0)
        self.set_actions = MagicMock()
        self.step = MagicMock(side_effect=self._advance)
        self.reset = MagicMock(side_effect=self._advance)
        self.closed = False

    def _advance(self, *args, **kwargs) -> None:
        if not self._script:
            raise AssertionError("The wrapper asked Unity for more exchanges than the script holds.")
        self._current = self._script.pop(0)

    @property
    def behavior_specs(self):
        return self._specs

    def get_steps(self, behavior_name):
        return self._current

    def close(self):
        self.closed = True
