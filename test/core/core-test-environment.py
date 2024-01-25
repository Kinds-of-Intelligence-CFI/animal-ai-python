import unittest
from unittest.mock import patch
from animalai.environment import AnimalAIEnvironment

""" This file contains tests for the environment class. Please keep it this way. """

class TestEnvironmentInitialization(unittest.TestCase):
    @patch("animalai.environment.UnityEnvironment")
    def test_initialization_with_default_parameters(self, mock_env):
        """Test initialization with default parameters."""
        env = AnimalAIEnvironment()
        mock_env.assert_called_once()
        self.assertIsNotNone(env)

    @patch("animalai.environment.UnityEnvironment")
    def test_initialization_with_custom_parameters(self, mock_env):
        """Test initialization with custom parameters."""
        env = AnimalAIEnvironment(file_name="custom_file", worker_id=2)
        mock_env.assert_called_once()
        self.assertIsNotNone(env)




if __name__ == "__main__":
    unittest.main()
