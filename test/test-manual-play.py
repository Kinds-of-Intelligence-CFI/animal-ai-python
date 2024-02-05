import unittest
import random
import sys

from animalai import AnimalAIEnvironment


class TestManualPlay(unittest.TestCase):
    
    def test_environment_initialization(self):
        """ Test if the AnimalAI environment initializes in manual play mode. """

        # Initialize the AnimalAI environment from environment.py script
        env = AnimalAIEnvironment(
            file_name='test/executable', 
            base_port=5005 + random.randint(0, 1000),
            play=True, # Here, we check if play mode is launched correctly
        )
        
        try:
            env.reset()  # Checking the main loop of the env code
            env.close()
        except Exception as e:
            self.fail(f"Environment initialization failed with an exception: {e}")

if __name__ == '__main__':
    unittest.main()
