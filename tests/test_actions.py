import unittest
from animalai.actions import AAIActions

""" This file contains tests for the AAIActions class. Please keep it this way. """


class TestAAIActions(unittest.TestCase):
    def test_action_initialization(self):
        """Test if AAIActions initializes correctly"""
        actions = AAIActions()
        # Basic tests for actions.
        self.assertIsNotNone(actions.NOOP)
        self.assertIsNotNone(actions.LEFT)
        self.assertIsNotNone(actions.RIGHT)

    def test_random_action(self):
        """Test if random action selection works"""
        actions = AAIActions()
        random_action = actions.random()
        self.assertIn(random_action, actions.allActions)


if __name__ == "__main__":
    unittest.main()
