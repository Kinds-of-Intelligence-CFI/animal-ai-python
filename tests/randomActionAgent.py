import random
import numpy as np
from pathlib import Path

from animalai import AnimalAIEnvironment, arenas
from animalai.actions import AAIActions
from animalai.executable import find_executable
from animalai.agents import RandomActionAgent


def main():
    """
    Manual test to see if random action agent works.
    """

    env = AnimalAIEnvironment(
        file_name=str(find_executable(Path("test/executable/"))),
        arenas_configurations=arenas.GOOD_GOAL_RANDOM_POS,
        base_port=5005 + random.randint(0, 1000),
        useRayCasts=True,
        rayMaxDegrees=30,
        raysPerSide=3,
        play=False,
        inference=True,
    )

    randomAgent = RandomActionAgent(step_length_distribution=lambda: np.random.normal(
        5, 1))  # a Rayleigh walker (sampling from normal distribution)

    # by default should be AnimalAI?team=0
    behavior = list(env.behavior_specs.keys())[0]

    # define actions
    actions = AAIActions()

    # Run two episodes
    for _episode in range(2):
        env.step()
        done = False
        episodeReward = 0

        previous_action = actions.NOOP
        new_action = randomAgent.get_new_action(prev_step=previous_action)

        while not done:
            step_list = randomAgent.get_num_steps(prev_step=new_action)

            for action in step_list:
                print(action)
                env.set_actions(behavior, action.action_tuple)
                env.step()
                previous_action = action

                dec, term = env.get_steps(behavior)

                if len(term) > 0:
                    episodeReward += term.reward
                    print(f"Episode Reward: {episodeReward}")
                    done = True
                    break

            new_action = randomAgent.get_new_action(prev_step=previous_action)

    print("Closing environment")
    env.close()
    print("Environment Closed")


if __name__ == "__main__":
    main()
