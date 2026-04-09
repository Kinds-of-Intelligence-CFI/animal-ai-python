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
    from inspect_ai.model import ChatMessageAssistant, Content, ContentImage, ContentText
except ImportError as e:
    raise ImportError(
        "inspect_wrapper requires the 'inspect-ai' package. "
        "Install it with: pip install animalai[inspect-ai]"
    ) from e


import numpy as np

from animalai.LLM_scaffolds.environment_scaffolds import EnvironmentScaffold
from animalai.environment import AnimalAIEnvironment

import io 
import base64
import numpy as np
from PIL import Image

def encode_camera_obs(obs: np.ndarray) -> str:
    # ML-Agents visual obs: float32 in [0, 1], shape (H, W, C)
    arr = (obs * 255).clip(0, 255).astype(np.uint8)
    # Grayscale comes through as (H, W, 1) — PIL wants (H, W) for mode "L"
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr.squeeze(-1)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

class AAIStateModel(StoreModel):
    AAI : Optional[EnvironmentScaffold] = None
    rewards: list[float] = []

def parse_action(game_wrapper: EnvironmentScaffold, response: str) -> str:
    actions = game_wrapper.available_actions
    for word in response.split()[::-1]:
        cleaned = word.strip(".,!?;:'\"()[]{}")
        if cleaned in actions:
            return cleaned
    raise ValueError(f"Could not parse action from '{response.strip()}'")

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
        # get the state of the current environment 
        AAI_state = store_as(AAIStateModel, instance=instance)
        if AAI_state.AAI is None:
            # create the environment based off the information in the task state
            AAI_state.AAI = scaffold_type(AnimalAIEnvironment(
                arenas_configurations=state.metadata["arenas_configurations"],
                useCamera=state.metadata.get("useCamera", True),
                no_graphics=state.metadata.get("no_graphics", True),
                resolution=state.metadata.get("resolution", 84),
                base_port=state.metadata.get("base_port", 5005),
                worker_id=state.metadata.get("worker_id", 0),
                seed=state.metadata.get("seed", None),
                timescale=state.metadata.get("timescale", 5),
                targetFrameRate=state.metadata.get("targetFrameRate", -1),
            ))

        clean_action = parse_action(AAI_state.AAI, action)
        try:
            obs, reward, done, info = AAI_state.AAI.step(clean_action)
        except Exception:
            AAI_state.AAI.close()
            raise
        AAI_state.rewards.append(reward)
        if done:
            state.metadata["rewards"] = AAI_state.rewards
            state.completed = True
            AAI_state.AAI.close()

        if isinstance(obs, str):
            return obs
        elif isinstance(obs, np.ndarray):
            # encode the image as a base64 string
            encoded_image = encode_camera_obs(obs)
            return ContentImage(image=f"data:image/png;base64,{encoded_image}")
        else:
            raise ValueError(f"Unexpected observation type: {type(obs)}")

    return execute

@solver
def add_act_tool(scaffold_type: type[EnvironmentScaffold]) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.tools.append(act(scaffold_type=scaffold_type, state=state))
        return state
    
    return solve

@scorer(metrics=[mean(), std()])
def total_reward_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        return Score(value=sum(state.metadata.get("rewards", [])))
    return score