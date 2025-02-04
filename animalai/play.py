import argparse
import random

from pathlib import Path

from animalai import AnimalAIEnvironment, arenas
from animalai.executable import find_executable

def play(configuration_file: str = None, env_path: str = None, log_path: str = None) -> None:
    """
    Load a config file and play
    :param configuration_file: str path to a yaml configuration. Plays animalai.arenas.GoodGoal_Random.yml by default
    :param env_path: str path to AAI environment executable. Looks for it in root by default.
    :return: None
    """

    print("Initializing AAI environment")
    environment = AnimalAIEnvironment(
        file_name=env_path if env_path is not None else str(find_executable(Path(""))),
        base_port=5005 + random.randint(0, 1000),
        arenas_configurations=configuration_file if configuration_file is not None else arenas.GOOD_GOAL_RANDOM_POS,
        play=True,
        log_folder=log_path,
    )

    print("Press Q in the Unity window then Ctrl+C in the command line to close the environment effectively.")

    # Run the environment until signal is lost
    try:
        while environment._process:  # type: ignore
            continue
    except KeyboardInterrupt:
        pass
    finally:
        environment.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This file allows the user to open and play an AnimalAI environment.')
    parser.add_argument("--AAI_path", default=None, type=str, help="The path to the AnimalAI binary to use.")
    parser.add_argument("--config_path", default=None, type=str, help="The path to the config file of the environment you want ot use.")
    parser.add_argument("--log_path", default="./logs", type=str, help="The folder AnimalAI should output its logs to.")

    # parse the args
    args = parser.parse_args()

    # play in the ennvironment
    play(configuration_file=args.config_path, env_path=args.AAI_path, log_path=args.log_path)
