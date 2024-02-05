import numpy as np
from mlagents_envs.base_env import ActionTuple


## Python Class to help access the discrete action space of AAI environment.
class AAIActions:
    def __init__(self, no_agents=1):
        if not isinstance(no_agents, int) or no_agents < 1:
            raise ValueError("no_agents must be a positive integer.")

        self.NOOP = AAIAction(
            "noop",
            ActionTuple(
                continuous=np.zeros((no_agents, 0)),
                discrete=np.array([[0, 0]], dtype=np.int32),
            ),
        )
        self.LEFT = AAIAction(
            "left",
            ActionTuple(
                continuous=np.zeros((no_agents, 0)),
                discrete=np.array([[0, 2]], dtype=np.int32),
            ),
        )
        self.RIGHT = AAIAction(
            "right",
            ActionTuple(
                continuous=np.zeros((no_agents, 0)),
                discrete=np.array([[0, 1]], dtype=np.int32),
            ),
        )
        self.FORWARDS = AAIAction(
            "forwards",
            ActionTuple(
                continuous=np.zeros((no_agents, 0)),
                discrete=np.array([[1, 0]], dtype=np.int32),
            ),
        )
        self.FORWARDSLEFT = AAIAction(
            "forwards&left",
            ActionTuple(
                continuous=np.zeros((no_agents, 0)),
                discrete=np.array([[1, 2]], dtype=np.int32),
            ),
        )
        self.FORWARDSRIGHT = AAIAction(
            "forwards&right",
            ActionTuple(
                continuous=np.zeros((no_agents, 0)),
                discrete=np.array([[1, 1]], dtype=np.int32),
            ),
        )
        self.BACKWARDS = AAIAction(
            "backwards",
            ActionTuple(
                continuous=np.zeros((no_agents, 0)),
                discrete=np.array([[2, 0]], dtype=np.int32),
            ),
        )
        self.BACKWARDSLEFT = AAIAction(
            "backwards&left",
            ActionTuple(
                continuous=np.zeros((no_agents, 0)),
                discrete=np.array([[2, 2]], dtype=np.int32),
            ),
        )
        self.BACKWARDSRIGHT = AAIAction(
            "backwards&right",
            ActionTuple(
                continuous=np.zeros((no_agents, 0)),
                discrete=np.array([[2, 1]], dtype=np.int32),
            ),
        )
        self.allActions: list = [
            self.NOOP,
            self.LEFT,
            self.RIGHT,
            self.FORWARDS,
            self.FORWARDSLEFT,
            self.FORWARDSRIGHT,
            self.BACKWARDS,
            self.BACKWARDSLEFT,
            self.BACKWARDSRIGHT,
        ]

    def random(self):
        return np.random.choice(self.allActions)


class AAIAction:
    """Actions have a name and an associated ActionTuple"""

    def __init__(self, name, action_tuple):
        if not isinstance(name, str):
            raise TypeError("name must be a string.")
        if not isinstance(action_tuple, ActionTuple):
            raise TypeError("action_tuple must be an instance of ActionTuple.")
        self.name = name
        self.action_tuple = action_tuple

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"AAIAction(name={self.name}, action_tuple={self.action_tuple})"
