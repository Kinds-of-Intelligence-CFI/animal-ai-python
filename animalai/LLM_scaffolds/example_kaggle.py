from kaggle_benchmarks.actors.llms import LLMChat, LLMResponse

from animalai.LLM_scaffolds.environment_scaffolds import FrameByFrameScaffold
from animalai.LLM_scaffolds.framework_wrappers import KaggleWrapper
from animalai.environment import AnimalAIEnvironment


from kaggle_benchmarks import (
    assertions,
    chats,
    llm,
    task,
)


@task(name="Can it play Animal-AI?")
def can_it_play_aai(llm):
    global _last_aai
    aai = KaggleWrapper(FrameByFrameScaffold(AnimalAIEnvironment(
        arenas_configurations="animalai/arenas/GoodGoal_Fixed.yml",
        useCamera=True,
        no_graphics=False,
        resolution=150,
        base_port=5005,
        worker_id=0,
        )))
    _last_aai = aai

    try:
        with chats.new("AnimalAI"):
            aai.initialize()

            for turn in range(100):
                print(f"TURN {turn}")
                llm.respond()
                aai.respond()

                if aai.game_wrapper.is_finished():
                    break

    finally:
        aai.game_wrapper.close()

    assertions.assert_true(
        aai.game_wrapper.total_reward > 0,
        f"LLM should finish with positive reward. "
        f"LLM scored {aai.game_wrapper.total_reward}",
    )


class MockLLM(LLMChat):
    def __init__(self, response="FORWARD"):
        super().__init__(name="mock-llm")
        self.response = response

    def invoke(self, messages, system=None, **kwargs):
        return LLMResponse(content=self.response)



if __name__ == "__main__":
    llm = MockLLM() 

    can_it_play_aai.run(llm)