import random
import unittest
from unittest.mock import patch
from animalai.environment import AnimalAIEnvironment
from mlagents_envs.side_channel.raw_bytes_channel import RawBytesChannel
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)

""" This file contains tests for the environment class. Please keep it this way. """

# This should be changed from a hard coded value but there is no way to do that at the moment.
AAI_filepath = "C:/AnimalAI/4.2.0/Animal-AI.exe"

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

    def test_timeout_default_train(self):
        """Test if the default timeout is set correctly in train mode."""
        port = 5005 + random.randint(0, 1000)
        env = AnimalAIEnvironment(file_name=AAI_filepath, base_port=port, play=False)
        self.assertEqual(env.timeout, 60)
        try:
            env.close()
        except Exception as e:
            self.fail(
                f"Environment initialization failed with an exception: {e}")

    def test_timeout_default_play(self):
        """Test if the default timeout is set correctly in play mode."""
        port = 5005 + random.randint(0, 1000)
        env = AnimalAIEnvironment(file_name=AAI_filepath, base_port=port, play=True)
        self.assertEqual(env.timeout, 10)
        try:
            env.close()
        except Exception as e:
            self.fail(
                f"Environment initialization failed with an exception: {e}")
            
    def test_timeout_custom(self):
        """Test if the custom timeout is set correctly when a custom value is provided."""
        port = 5005 + random.randint(0, 1000)
        env = AnimalAIEnvironment(file_name=AAI_filepath, base_port=port, timeout=45)
        self.assertEqual(env.timeout, 45)
        try:
            env.close()
        except Exception as e:
            self.fail(
                f"Environment initialization failed with an exception: {e}")

    def test_timeout_invalid(self):
        """Test if the invalid timeout raises a ValueError."""
        with self.assertRaises(AssertionError):
            AnimalAIEnvironment(file_name=AAI_filepath, timeout=-1)


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
