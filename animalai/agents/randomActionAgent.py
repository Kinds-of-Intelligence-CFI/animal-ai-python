import numpy as np
from collections import deque
from typing import Deque
from animalai.actions import AAIActions, AAIAction


class RandomActionAgent:
    """
    Implements a random action agent. Chooses one of 9 actions, and then repeats it for a number of steps before picking a new action.
    Parameters for distributions for choosing step lengths, biases towards certain actions, and correlations with previous biases.
    
    Parameters
    ----------
    max_step_length : int
        The maximum number of steps the agent should take of one action before picking another. Ignored unless `step_length_distribution` = None.
    step_length_distribution : np.random.* lambda function 
        Distribution from which to sample number of steps before picking a new action. Absolute value is taken from these distributions, rounded to the nearest integer. 
        Passing `lambda: np.random.standard_cauchy()` simulates a Levy walker, drawing step lengths from the cauchy distribution. `lambda: np.random.normal(5, 1)` simulates a Rayleigh walker, drawing step lengths from the normal distribution. 
        Default is None (fixed length walker)
    action_biases : list
        A list of biases for each of the action, ordered as [NOOP,LEFT,RIGHT,FORWARDS,BACKWARDS,FORWARDSLEFT,FORWARDSRIGHT,BACKWARDSLEFT,BACKWARDSRIGHT]. 
        These values are softmaxed, so relative magnitude is important. Default is [1,1,1,1,1,1,1,1,1]
    prev_step_bias : float
        Probability that the next action to be selected is the previous action. Must be between 0 and 1 inclusive. Default is 0.
    remove_prev_step : bool
        Exclude the previous step when selecting next action, to ensure diversity of actions. Default is False. 
    """

    def __init__(self, max_step_length=10, step_length_distribution=None, action_biases=None, prev_step_bias=0, remove_prev_step=False):
        self.max_step_length = max_step_length
        self.step_length_distribution = step_length_distribution if step_length_distribution is not None else lambda: max_step_length
        self.action_biases = action_biases if action_biases is not None else [1] * 9
        self.prev_step_bias = prev_step_bias
        self.remove_prev_step = remove_prev_step
        self.actions = [AAIActions().NOOP, 
                        AAIActions().LEFT, 
                        AAIActions().RIGHT, 
                        AAIActions().FORWARDS, 
                        AAIActions().BACKWARDS, 
                        AAIActions().FORWARDSLEFT, 
                        AAIActions().FORWARDSRIGHT, 
                        AAIActions().BACKWARDSLEFT, 
                        AAIActions().BACKWARDSRIGHT,
                        ]

    def get_num_steps(self, prev_step: AAIAction) -> Deque[AAIAction]:
        num_steps = abs(int(self.step_length_distribution()))

        step_list = deque([prev_step] * num_steps)
        return step_list

    def get_new_action(self, prev_step: AAIAction) -> AAIAction:
        """
        Provide a vector of 9 real values, one for each action, which is then softmaxed to provide the probability of selecting that action. Relative differences between the values is what is important. 

        Provide an initial probability of selecting the previous step again. If that action is not selected, then the next step is picked according to the softmaxed action biases. The previous action can be removed
        from the softmaxed biases (by continually sampling until an action is picked that is not the previous action), by changing `remove_prev_step` to `True`.
        """
        assert len(self.action_biases) == 9, "You must provide biases for all nine (9) actions. A uniform distribution is [1,1,1,1,1,1,1,1,1]"
        assert 0 <= self.prev_step_bias <= 1, "The bias towards the previous action must be a scalar value between 0 and 1."

        action_is_prev_step = np.random.choice(a=[False, True], size=1, p=[(1 - self.prev_step_bias), self.prev_step_bias])

        if action_is_prev_step:
            return prev_step
        else:
            action_biases_softmax = (np.exp(self.action_biases) / np.sum(np.exp(self.action_biases))) 
            if self.remove_prev_step:
                action = prev_step
                while action == prev_step:
                    action = np.random.choice(a=self.actions, size=1, p=action_biases_softmax)
            else:
                action = np.random.choice(a=self.actions, size=1, p=action_biases_softmax)

            return action[0]