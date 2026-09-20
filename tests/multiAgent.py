import random
from pathlib import Path

from animalai import AnimalAIEnvironment
from animalai.actions import AAIActions, stack_actions
from animalai.executable import find_executable
from animalai.raycastparser import RayCastObjects, RayCastParser


def main():
    """
    Manual test for multi-agent arenas: two agents take independent random actions.
    """
    rays_per_side = 2
    env = AnimalAIEnvironment(
        file_name=str(find_executable(Path("test/executable/"))),
        arenas_configurations=str(Path(__file__).parent / "test-Configs" / "multiAgentArena.yml"),
        base_port=5005 + random.randint(0, 1000),
        useRayCasts=True,
        raysPerSide=rays_per_side,
        play=False,
        inference=True,
    )

    # All agents share one behavior unless the config puts them on different teams
    behavior = list(env.behavior_specs.keys())[0]
    print(f"Behaviors: {list(env.behavior_specs.keys())}")

    actions = AAIActions()
    parser = RayCastParser([RayCastObjects.AGENT, RayCastObjects.GOODGOAL], 2 * rays_per_side + 1)

    # Run two episodes
    for _episode in range(2):
        episode_rewards = {}
        while True:
            env.step()
            decision_steps, terminal_steps = env.get_steps(behavior)

            for agent_id, reward in zip(decision_steps.agent_id, decision_steps.reward):
                episode_rewards[agent_id] = episode_rewards.get(agent_id, 0) + reward

            # Every agent in the arena finishes in the same step
            if len(terminal_steps) > 0:
                for agent_id, reward in zip(terminal_steps.agent_id, terminal_steps.reward):
                    episode_rewards[agent_id] = episode_rewards.get(agent_id, 0) + reward
                print(f"Episode rewards by agent id: {episode_rewards}")
                break

            if len(decision_steps) == 0:
                continue

            for agent_id, obs in zip(decision_steps.agent_id, env.get_obs_dicts(decision_steps.obs)):
                print(f"agent {agent_id}: health {obs['health']:.1f}")
                parser.prettyPrint(obs["rays"])

            # One action per agent, in the order of decision_steps.agent_id
            env.set_actions(behavior, stack_actions([actions.random() for _ in decision_steps.agent_id]))

    print("Closing environment")
    env.close()
    print("Environment Closed")


if __name__ == "__main__":
    main()
