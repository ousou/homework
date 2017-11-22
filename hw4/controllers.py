import numpy as np
from cost_functions import trajectory_cost_fn
import time

class Controller():
    def __init__(self):
        pass

    # Get the appropriate action(s) for this state(s)
    def get_action(self, state):
        pass


class RandomController(Controller):
    def __init__(self, env):
        self.env = env

    def get_action(self, state):
        """ YOUR CODE HERE """
        return self.env.action_space.sample()

class MPCcontroller(Controller):
    """ Controller built using the MPC method outlined in https://arxiv.org/abs/1708.02596 """
    def __init__(self,
                 env,
                 dyn_model,
                 horizon=5,
                 cost_fn=None,
                 num_simulated_paths=10,
                 ):
        self.env = env
        self.dyn_model = dyn_model
        self.horizon = horizon
        self.cost_fn = cost_fn
        self.num_simulated_paths = num_simulated_paths
        self.rnd_control = RandomController(env)

    def get_action(self, state):
        trajectories = np.empty((3,self.horizon,self.num_simulated_paths))
        states = np.repeat(state, self.num_simulated_paths)
        for h in range(self.horizon):
            actions = []
            for i in range(self.num_simulated_paths):
                actions.append(self.rnd_control.get_action(states[i]))
            actions = np.array(actions)
            next_states = self.dyn_model.predict(states, actions)
            trajectories[0, h] = states
            trajectories[1, h] = actions
            trajectories[2, h] = next_states
            states = next_states
        costs = []
        for i in range(self.num_simulated_paths):
            trajectory = trajectories[:,:,i]
            costs.append(trajectory_cost_fn(
                self.cost_fn,trajectory[0], trajectory[1], trajectory[2]))
        costs = np.array(costs)
        lowest_cost_index = np.argmax(costs)
        first_action = trajectories[0,0,lowest_cost_index]
        return first_action




