import unittest

import numpy as np

from animalai.actions import AAIActions, stack_actions

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

    def test_actions_have_one_row_per_agent(self):
        """Each action carries one discrete and one continuous row per agent"""
        actions = AAIActions(no_agents=3)
        for action in actions.allActions:
            self.assertEqual(action.action_tuple.discrete.shape, (3, 2))
            self.assertEqual(action.action_tuple.continuous.shape, (3, 0))
        np.testing.assert_array_equal(actions.FORWARDSLEFT.action_tuple.discrete, [[1, 2]] * 3)

    def test_stack_actions_gives_each_agent_its_own_action(self):
        actions = AAIActions()
        stacked = stack_actions([actions.FORWARDS, actions.LEFT, actions.NOOP])
        np.testing.assert_array_equal(stacked.discrete, [[1, 0], [0, 2], [0, 0]])
        self.assertEqual(stacked.continuous.shape, (3, 0))

    def test_stack_actions_needs_an_action(self):
        with self.assertRaises(ValueError):
            stack_actions([])


if __name__ == "__main__":
    unittest.main()
