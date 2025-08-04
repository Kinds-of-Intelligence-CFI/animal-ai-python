from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock, Mock
import sys
from animalai import arenas
from animalai.environment import AnimalAIEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper

from animalai.executable import find_executable


def test_vec_env():
    test_env = UnityToGymWrapper(AnimalAIEnvironment(
        file_name=str(find_executable(Path("test/executable/"))),
        arenas_configurations=arenas.GOOD_GOAL_RANDOM_POS,))
    test_actions = test_env.action_space.sample()
    for i in range(200):
        print(f"step {i}")
        _ = test_env.step(action=test_actions)

    test_env.close()


    env = AnimalAIEnvironment.make_vec_env(
            4, 
            file_name=str(find_executable(Path("test/executable/"))),
            arenas_configurations=arenas.GOOD_GOAL_RANDOM_POS,
            )
    actions = [env.action_space.sample() for i in range(4)]

    for i in range(200):
        print(f"step {i}")
        _ = env.step(actions=actions)

    env.close()

if __name__ == "__main__":
    test_vec_env()