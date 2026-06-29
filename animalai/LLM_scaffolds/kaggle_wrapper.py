
import numpy as np

from animalai.LLM_scaffolds.environment_scaffolds import EnvironmentScaffold

try:
    from kaggle_benchmarks import (
        actors,
        assertions,
        chats,
        content_types,
        llm,
        messages,
        task,
    )
except ImportError as e:
    raise ImportError(
        "KaggleWrapper requires the 'kaggle-benchmarks' package. "
        "Install it with: pip install animalai[kaggle]"
    ) from e

class KaggleWrapper(actors.Actor):
    """Actor that drives the Animal-AI environment."""

    def __init__(self, environment_scaffold: EnvironmentScaffold[np.ndarray]):
        super().__init__(name="AnimalAI", role="user", avatar="🐾")
        self.game_wrapper = environment_scaffold

    def parse_action(self, response: str) -> str:
        actions = self.game_wrapper.available_actions
        for word in response.split()[::-1]:
            cleaned = word.strip(".,!?;:'\"()[]{}")
            if cleaned in actions:
                return cleaned
        raise ValueError(f"Could not parse action from '{response.strip()}'")

    def initialize(self, start_prompt: str | None = None):
        if start_prompt is None:
            start_prompt = self.game_wrapper.get_default_system_prompt()
        self._send(start_prompt, is_visible_to_llm=True)
        obs, _ = self.game_wrapper.reset()
        self._send(content_types.images.from_array(obs), is_visible_to_llm=True)

    def respond(self):
        chat = chats.get_current_chat()
        return self.send(chat.messages[-1])

    def send(self, message: str | messages.Message) -> messages.Message[str]:
        if isinstance(message, messages.Message):
            message = message.text

        action = self.parse_action(message)
        obs, reward, done, info = self.game_wrapper.step(action)
        self._send(content_types.images.from_array(obs), is_visible_to_llm=True)

        if self.game_wrapper.is_finished():
            return self._send("EPISODE FINISHED!", is_visible_to_llm=True)

        return self._send(f"LAST MOVE REWARD: {reward}", is_visible_to_llm=True)
    
    def close(self):
        self.game_wrapper.close()

    def is_finished(self)-> bool:
        return self.game_wrapper.is_finished()