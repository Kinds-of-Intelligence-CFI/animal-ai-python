import unittest
from unittest.mock import patch
from animalai.environment import AnimalAIEnvironment
from mlagents_envs.side_channel.raw_bytes_channel import RawBytesChannel
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)

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
        env = AnimalAIEnvironment(
            file_name="custom_file.yml", worker_id=2
        )  # For custom file path, use full path.
        mock_env.assert_called_once()
        self.assertIsNotNone(env)


class TestSideChannelsConfiguration(unittest.TestCase):
    def test_side_channels_configuration(self):
        """Test if side channels are correctly configured."""
        try:
            env = AnimalAIEnvironment()

            # Check for specific types of side channels
            has_engine_config_channel = any(
                isinstance(channel, EngineConfigurationChannel)
                for channel in env.side_channels
            )
            self.assertTrue(
                has_engine_config_channel, "EngineConfigurationChannel not found"
            )

        except Exception as e:
            self.fail(f"Side channels configuration test failed: {e}")


if __name__ == "__main__":
    unittest.main()
