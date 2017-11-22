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
        timesteps = []
        states = np.tile(state, (self.num_simulated_paths,1))
        for h in range(self.horizon):
            actions = []
            for i in range(self.num_simulated_paths):
                actions.append(self.rnd_control.get_action(states[i]))
            actions = np.array(actions)
            next_states = self.dyn_model.predict(states, actions)
            timestep = {
                'states': states,
                'actions': actions,
                'next_states': next_states
            }
            timesteps.append(timestep)

        trajectories = []

        for _ in range(self.num_simulated_paths):
            trajectories.append({
                'states': [],
                'actions': [],
                'next_states': []
            })

        for t in range(len(timesteps)):
            timestep = timesteps[t]
            for tr in range(len(trajectories)):
                trajectory = trajectories[tr]
                trajectory['states'].append(timestep['states'][tr])
                trajectory['actions'].append(timestep['actions'][tr])
                trajectory['next_states'].append(timestep['next_states'][tr])

        costs = []
        for trajectory in trajectories:
            costs.append(trajectory_cost_fn(
                self.cost_fn,
                np.array(trajectory['states']),
                np.array(trajectory['actions']),
                np.array(trajectory['next_states'])))
        costs = np.array(costs)
        lowest_cost_index = np.argmax(costs)
        first_action = trajectories[lowest_cost_index]['actions'][0]
        return first_action




