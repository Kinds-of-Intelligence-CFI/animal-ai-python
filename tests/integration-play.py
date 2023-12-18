import random
import sys
from pathlib import Path

from animalai import AnimalAIEnvironment, arenas
from animalai.executable import find_executable


def main():
    """
    Manual test to see if play mode works.
    """
    env = AnimalAIEnvironment(
        file_name=str(find_executable(Path("tests/executable/"))),
        arenas_configurations=arenas.GoodGoal_Random,
        base_port=5005 + random.randint(0, 1000),
        play=True,
    )

    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("Closing environment")
        env.close()
        sys.exit(0)


if __name__ == "__main__":
    main()
