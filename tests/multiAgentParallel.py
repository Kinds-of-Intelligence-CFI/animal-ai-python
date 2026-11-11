import random
import sys
from pathlib import Path

import numpy as np
from pettingzoo.test import parallel_api_test

from animalai import AnimalAIEnvironment
from animalai.wrappers import AnimalAIParallelEnv

ARENA = Path(__file__).parent / "test-Configs" / "multiAgentArena.yml"
# The x position of each Agent item in multiAgentArena.yml
SPAWN_X = {"agent_0": 10.0, "agent_1": 30.0}
FORWARDS = [1, 0]
NOOP = [0, 0]


def horizontal_position(obs: dict, agent: str) -> np.ndarray:
    return np.array([obs[agent]["position"][0], obs[agent]["position"][2]])


def check_spawn_order(env: AnimalAIParallelEnv) -> None:
    """agent_k must be the k-th Agent item of the arena."""
    obs, _ = env.reset()
    for agent, spawn_x in SPAWN_X.items():
        x = float(obs[agent]["position"][0])
        print(f"  {agent}: x = {x:.2f} (Agent item x = {spawn_x})")
        assert abs(x - spawn_x) < 1.0, f"{agent} is not at the position of its Agent item"


def check_action_routing(env: AnimalAIParallelEnv, moving_agent: str) -> None:
    """Only the agent that gets FORWARDS must move."""
    first_obs, _ = env.reset()
    env.step({agent: NOOP for agent in env.agents})
    for _ in range(10):
        actions = {agent: FORWARDS if agent == moving_agent else NOOP for agent in env.agents}
        obs, _, _, _, _ = env.step(actions)
    for agent in env.possible_agents:
        distance = float(np.linalg.norm(horizontal_position(obs, agent) - horizontal_position(first_obs, agent)))
        print(f"  {agent}: moved {distance:.2f}")
        if agent == moving_agent:
            assert distance > 0.5, f"{agent} got FORWARDS but did not move"
        else:
            assert distance < 0.1, f"{agent} got NOOP but moved"


def check_episodes(env: AnimalAIParallelEnv, number_of_episodes: int) -> None:
    """Agents must finish together, and the same agents must start each episode."""
    env.reset()
    env.step({agent: NOOP for agent in env.agents})
    for episode in range(number_of_episodes):
        # The previous episode is either stopped early (first loop) or finished
        obs, _ = env.reset()
        for agent, spawn_x in SPAWN_X.items():
            assert abs(float(obs[agent]["position"][0]) - spawn_x) < 1.0, f"{agent} did not start at its Agent item"
        episode_rewards = dict.fromkeys(env.agents, 0.0)
        steps = 0
        while env.agents:
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            _, rewards, terminations, truncations, _ = env.step(actions)
            steps += 1
            for agent, reward in rewards.items():
                episode_rewards[agent] += reward
        assert all(terminations[agent] or truncations[agent] for agent in env.possible_agents)
        print(f"  episode {episode}: {steps} steps, rewards {episode_rewards}")


def main():
    """
    Manual test of AnimalAIParallelEnv with a real build that supports several agents.
    Give the path of the executable as the first argument. With no argument, the
    environment finds or downloads the executable.
    """
    env = AnimalAIParallelEnv(
        AnimalAIEnvironment(
            file_name=sys.argv[1] if len(sys.argv) > 1 else None,
            arenas_configurations=str(ARENA),
            base_port=5005 + random.randint(0, 1000),
            useCamera=True,
            resolution=32,
            useRayCasts=True,
            raysPerSide=2,
        )
    )
    try:
        print(f"Agents: {env.possible_agents}")
        assert env.possible_agents == list(SPAWN_X), "multiAgentArena.yml has two Agent items"

        print("Check 1: agent order follows the Agent items")
        check_spawn_order(env)

        print("Check 2: each action reaches its agent")
        check_action_routing(env, moving_agent="agent_0")
        check_action_routing(env, moving_agent="agent_1")

        print("Check 3: three episodes with the same agents")
        check_episodes(env, number_of_episodes=3)

        print("Check 4: PettingZoo parallel API test")
        parallel_api_test(env, num_cycles=50)

        print("All checks passed")
    finally:
        env.close()
        print("Environment Closed")


if __name__ == "__main__":
    main()
