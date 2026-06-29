import base64
from typing import Optional

try:
    from inspect_ai.tool import (
        Tool,
        ToolCall,
        ToolCallContent,
        ToolCallView,
        ToolCallViewer,
        ToolResult,
        tool,
        computer,
    )
    from inspect_ai.scorer import Score, Scorer, Target, accuracy, mean, scorer, std
    from inspect_ai.solver import Generate, Solver, TaskState, basic_agent, solver
    from inspect_ai.util import StoreModel, message_limit, store_as
    from inspect_ai.model import ChatMessageAssistant, ChatMessageUser, Content, ContentImage, ContentText
    from inspect_ai.log import transcript
except ImportError as e:
    raise ImportError(
        "inspect_wrapper requires the 'inspect-ai' package. "
        "Install it with: pip install animalai[inspect-ai]"
    ) from e


import numpy as np

from animalai.LLM_scaffolds.environment_scaffolds import EnvironmentScaffold
from animalai.environment import AnimalAIEnvironment

# Termination reasons from the harness rather than the env (whose reasons arrive
# via the scaffold's `info["done_reason"]`).
TERMINATION_ERROR = "error"                  # an exception was raised during step
TERMINATION_INSPECT_LIMIT = "inspect_limit"  # Inspect stopped the sample (message/token/time limit)

import io 
import base64
import numpy as np
from PIL import Image

def encode_camera_obs(obs: np.ndarray) -> str:
    img = Image.fromarray(obs)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

class AAIStateModel(StoreModel):
    AAI : Optional[EnvironmentScaffold] = None
    rewards: list[float] = []
    termination_reason: Optional[str] = None

def parse_action(game_wrapper: EnvironmentScaffold, response: str) -> str:
    actions = game_wrapper.available_actions
    for word in response.split()[::-1]:
        cleaned = word.strip(".,!?;:'\"()[]{}")
        if cleaned in actions:
            return cleaned
    raise ValueError(f"Could not parse action from '{response.strip()}'")


def create_env(scaffold_type: type[EnvironmentScaffold], state: TaskState) -> EnvironmentScaffold:
    """Construct an environment scaffold from the sample's metadata."""
    return scaffold_type(AnimalAIEnvironment(
        file_name=state.metadata.get("file_name", None),
        arenas_configurations=state.metadata["arenas_configurations"],
        useCamera=state.metadata.get("useCamera", True),
        no_graphics=state.metadata.get("no_graphics", False),
        resolution=state.metadata.get("resolution", 84),
        base_port=state.metadata.get("base_port", 5005),
        worker_id=state.metadata.get("worker_id", 0),
        seed=state.metadata.get("seed", None),
        timescale=state.metadata.get("timescale", 5),
        targetFrameRate=state.metadata.get("targetFrameRate", -1),
    ))

@tool()
def act(scaffold_type: type[EnvironmentScaffold], state: TaskState, instance: str | None = None) -> Tool:
    """
    Tool used to take actions in the animalai environment.

    Returns:
      The execute method called to take the action in the environment.
    """
    async def execute(action: str) -> ToolResult:
        """
        Use this function to move and take actions in AnimalAI.

        Args:
          action (str): The action to take in the environment must be from the set of available actions.

        Returns:
          The observations from the environment.
        """
        # get the state of the current environment (start_animalai will usually
        # have created it already; fall back to lazy construction if not)
        AAI_state = store_as(AAIStateModel, instance=instance)
        if AAI_state.AAI is None:
            AAI_state.AAI = create_env(scaffold_type, state)

        clean_action = parse_action(AAI_state.AAI, action)
        try:
            obs, reward, done, info = AAI_state.AAI.step(clean_action)
        except Exception:
            AAI_state.termination_reason = TERMINATION_ERROR
            AAI_state.AAI.close()
            raise
        AAI_state.rewards.append(reward)
        if done:
            AAI_state.termination_reason = info.get("done_reason")
            state.completed = True
            AAI_state.AAI.close()

        if isinstance(obs, str):
            return obs
        elif isinstance(obs, np.ndarray):
            encoded_image = encode_camera_obs(obs)
            return ContentImage(image=f"data:image/png;base64,{encoded_image}")
        else:
            raise ValueError(f"Unexpected observation type: {type(obs)}")

    return execute

@solver
def start_animalai(scaffold_type: type[EnvironmentScaffold], instance: str | None = None) -> Solver:
    """Boot the environment and show the agent the initial frame before it acts.

    Without this the environment is only created on the first `act` call, so the
    model's first action is taken with no observation. This solver constructs the
    environment up front and appends its initial frame as a user message.
    """
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        AAI_state = store_as(AAIStateModel, instance=instance)
        if AAI_state.AAI is None:
            AAI_state.AAI = create_env(scaffold_type, state)
        encoded_image = encode_camera_obs(AAI_state.AAI.last_frame)
        state.messages.append(
            ChatMessageUser(content=[ContentImage(image=f"data:image/png;base64,{encoded_image}")])
        )
        return state

    return solve

@solver
def add_act_tool(scaffold_type: type[EnvironmentScaffold]) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.tools.append(act(scaffold_type=scaffold_type, state=state))
        return state

    return solve

def _termination_reason(AAI_state: AAIStateModel) -> str:
    """The episode's end reason. No env-side reason means the harness stopped it
    (e.g. a message limit), which the `act` tool never gets to observe."""
    return AAI_state.termination_reason or TERMINATION_INSPECT_LIMIT


@scorer(metrics=[mean(), std()])
def total_reward_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        # Read from the store, which is populated every step, so the reward
        # survives even when an Inspect limit cuts the episode short.
        AAI_state = store_as(AAIStateModel)
        return Score(
            value=sum(AAI_state.rewards),
            metadata={
                "termination_reason": _termination_reason(AAI_state),
                "num_steps": len(AAI_state.rewards),
            },
        )
    return score


async def close_environment(state: TaskState, instance: str | None = None):
    """Task cleanup: record the episode outcome, then close the env.

    Runs however the sample ended, so it is the reliable place to record why.
    No env-side reason means the harness stopped the sample (e.g. a message
    limit), which Inspect already logs as its own `sample_limit` event.
    """
    AAI_state = store_as(AAIStateModel, instance=instance)
    if AAI_state.AAI is None:
        return

    outcome = {
        "termination_reason": _termination_reason(AAI_state),
        "total_reward": sum(AAI_state.rewards),
        "num_steps": len(AAI_state.rewards),
    }
    state.metadata["episode_outcome"] = outcome
    # Surface the outcome in the viewer's Transcript tab, next to Inspect's events.
    transcript().info(outcome, source="animalai")
    AAI_state.AAI.close()
