import random
from pathlib import Path

from animalai import AnimalAIEnvironment, arenas
from animalai.executable import find_executable
from animalai.agents import Braitenberg


def main():
    """
    Manual test to see if brainenberg agent works.
    """
    n_rays = 9

    env = AnimalAIEnvironment(
        file_name=str(find_executable(Path("test/executable/"))),
        arenas_configurations=arenas.GOOD_GOAL_RANDOM_POS,
        base_port=5005 + random.randint(0, 1000),
        useRayCasts=True,
        rayMaxDegrees=30,
        raysPerSide=(n_rays - 1) // 2,
        play=False,
    )

    # A simple BraitenBerg Agent that heads towards food items.
    braitenbergAgent = Braitenberg(no_rays=n_rays)
    # by default should be AnimalAI?team=0
    behavior = list(env.behavior_specs.keys())[0]

    # Run two episodes
    for _episode in range(2):
        episodeReward = 0
        while True:
            # Step the environment
            env.step()
            step, term = env.get_steps(behavior)

            # Episode is over
            if len(term) > 0:
                episodeReward += term.reward
                print(f"Episode Reward: {episodeReward}")
                break  # Go to next episode

            # Get observations
            raycasts = env.get_obs_dict(step.obs)["rays"]
            braitenbergAgent.prettyPrint(raycasts)
            episodeReward += step.reward if len(step.reward) > 0 else 0

            # Get agent action
            action = braitenbergAgent.get_action(raycasts)
            print(action)
            env.set_actions(behavior, action.action_tuple)

    print("Closing environment")
    env.close()
    print("Environment Closed")


if __name__ == "__main__":
    main()
