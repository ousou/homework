import numpy as np
import tensorflow as tf
import gym
from dynamics import NNDynamicsModel
from controllers import MPCcontroller, RandomController
from cost_functions import cheetah_cost_fn, trajectory_cost_fn
import time
import logz
import os
import copy
import matplotlib.pyplot as plt
# from cheetah_env import HalfCheetahEnvNew
import roboschool

def sample(env, 
           controller, 
           num_paths=10, 
           horizon=1000, 
           render=False,
           verbose=False):
    """
        Write a sampler function which takes in an environment, a controller (either random or the MPC controller), 
        and returns rollouts by running on the env. 
        Each path can have elements for observations, next_observations, rewards, returns, actions, etc.
    """
    paths = []
    obs = env.reset()
    for itr in range(num_paths):
        obs = env.reset()
        steps = 0
        observations = []
        next_observations = []
        actions = []
        rewards = []
        while True:
            if render:
                env.render()
                time.sleep(0.05)
            observations.append(obs)
            action = controller.get_action(obs)
            actions.append(action)
            next_obs, reward, done, info = env.step(action)
            if verbose:
                print('info:', info)
            next_observations.append(next_obs)
            rewards.append(reward)
            steps += 1
            obs = next_obs
            if (done or steps >= horizon):
                break
        current_path = {
            'states': np.array(observations),
            'next_states': np.array(next_observations),
            'actions': np.array(actions),
            'rewards': np.array(rewards)
        }
        paths.append(current_path)

    return paths

# Utility to compute cost a path for a given cost function
def path_cost(cost_fn, path):
    return trajectory_cost_fn(cost_fn, path['states'], path['actions'], path['next_states'])

def extract_from_path_list(pathlist):
    states = []
    actions = []
    next_states = []
    deltas = []

    for path in pathlist:
        states.extend(path['states'])
        actions.extend(path['actions'])
        next_states.extend(path['next_states'])
        deltas.extend(path['next_states'] - path['states'])

    return {
        'states': np.array(states),
        'actions': np.array(actions),
        'next_states': np.array(next_states),
        'deltas': np.array(deltas)
    }

def calculate_costs_and_rewards_from_path_list(pathlist, cost_fn):
    costs = []
    rewards = []
    for path in pathlist:
        rewards.extend(path['rewards'])
        costs.append(path_cost(cost_fn, path))
    return np.array(costs), np.array(rewards)

def compute_normalization(data):
    states = data['states']
    actions = data['actions']
    deltas = data['deltas']

    mean_state = np.mean(states,axis=0)
    std_state = np.std(states,axis=0)
    mean_deltas = np.mean(deltas,axis=0)
    std_deltas = np.std(deltas,axis=0)
    mean_action = np.mean(actions,axis=0)
    std_action = np.std(actions,axis=0)
    return mean_state, std_state, mean_deltas, std_deltas, mean_action, std_action


def plot_comparison(env, dyn_model):
    """
    Write a function to generate plots comparing the behavior of the model predictions for each element of the state to the actual ground truth, using randomly sampled actions. 
    """
    """ YOUR CODE HERE """
    pass

def train(env, 
         cost_fn,
         logdir=None,
         render=False,
         learning_rate=1e-3,
         onpol_iters=10,
         dynamics_iters=60,
         batch_size=512,
         num_paths_random=10, 
         num_paths_onpol=10, 
         num_simulated_paths=10000,
         env_horizon=1000, 
         mpc_horizon=15,
         n_layers=2,
         size=500,
         activation=tf.nn.relu,
         output_activation=None
         ):

    """

    Arguments:

    onpol_iters                 Number of iterations of onpolicy aggregation for the loop to run. 

    dynamics_iters              Number of iterations of training for the dynamics model
    |_                          which happen per iteration of the aggregation loop.

    batch_size                  Batch size for dynamics training.

    num_paths_random            Number of paths/trajectories/rollouts generated 
    |                           by a random agent. We use these to train our 
    |_                          initial dynamics model.
    
    num_paths_onpol             Number of paths to collect at each iteration of
    |_                          aggregation, using the Model Predictive Control policy.

    num_simulated_paths         How many fictitious rollouts the MPC policy
    |                           should generate each time it is asked for an
    |_                          action.

    env_horizon                 Number of timesteps in each path.

    mpc_horizon                 The MPC policy generates actions by imagining 
    |                           fictitious rollouts, and picking the first action
    |                           of the best fictitious rollout. This argument is
    |                           how many timesteps should be in each fictitious
    |_                          rollout.

    n_layers/size/activations   Neural network architecture arguments. 

    """

    logz.configure_output_dir(logdir)

    #========================================================
    # 
    # First, we need a lot of data generated by a random
    # agent, with which we'll begin to train our dynamics
    # model.

    random_controller = RandomController(env)
    pathlist = sample(env, random_controller, num_paths_random, env_horizon)
    data = extract_from_path_list(pathlist)
    print('random data size: ',len(data['states']))


    #========================================================
    # 
    # The random data will be used to get statistics (mean
    # and std) for the observations, actions, and deltas
    # (where deltas are o_{t+1} - o_t). These will be used
    # for normalizing inputs and denormalizing outputs
    # from the dynamics network. 
    # 
    normalization = compute_normalization(data)
    # print('normalization', normalization)
    #
    # if True:
    #     return

    #========================================================
    # 
    # Build dynamics model and MPC controllers.
    # 
    sess = tf.Session()

    dyn_model = NNDynamicsModel(env=env, 
                                n_layers=n_layers, 
                                size=size, 
                                activation=activation, 
                                output_activation=output_activation, 
                                normalization=normalization,
                                batch_size=batch_size,
                                iterations=dynamics_iters,
                                learning_rate=learning_rate,
                                sess=sess)

    mpc_controller = MPCcontroller(env=env, 
                                   dyn_model=dyn_model, 
                                   horizon=mpc_horizon, 
                                   cost_fn=cost_fn, 
                                   num_simulated_paths=num_simulated_paths)


    #========================================================
    # 
    # Tensorflow session building.
    # 
    sess.__enter__()
    tf.global_variables_initializer().run()

    #========================================================
    # 
    # Take multiple iterations of onpolicy aggregation at each iteration refitting the dynamics model to current dataset and then taking onpolicy samples and aggregating to the dataset. 
    # Note: You don't need to use a mixing ratio in this assignment for new and old data as described in https://arxiv.org/abs/1708.02596
    #
    print('onpol_iters', onpol_iters)
    for itr in range(onpol_iters):
        dyn_model.fit(data)
        print('data fit done for itr', itr)
        new_pathlist = sample(env, mpc_controller, num_paths_onpol, env_horizon)
        pathlist.extend(new_pathlist)
        costs, returns = calculate_costs_and_rewards_from_path_list(new_pathlist, cost_fn)
        data = extract_from_path_list(pathlist)
        print('data size: ', len(data['states']))
        # print('action', action)
        """ YOUR CODE HERE """



        # LOGGING
        # Statistics for performance of MPC policy using
        # our learned dynamics model
        logz.log_tabular('Iteration', itr)
        # In terms of cost function which your MPC controller uses to plan
        logz.log_tabular('AverageCost', np.mean(costs))
        logz.log_tabular('StdCost', np.std(costs))
        logz.log_tabular('MinimumCost', np.min(costs))
        logz.log_tabular('MaximumCost', np.max(costs))
        # In terms of true environment reward of your rolled out trajectory using the MPC controller
        logz.log_tabular('AverageReturn', np.mean(returns))
        logz.log_tabular('StdReturn', np.std(returns))
        logz.log_tabular('MinimumReturn', np.min(returns))
        logz.log_tabular('MaximumReturn', np.max(returns))

        logz.dump_tabular()

def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='RoboschoolHalfCheetah-v1')
    # Experiment meta-params
    parser.add_argument('--exp_name', type=str, default='mb_mpc')
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--render', action='store_true')
    # Training args
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--onpol_iters', '-n', type=int, default=10)
    parser.add_argument('--dyn_iters', '-nd', type=int, default=60)
    parser.add_argument('--batch_size', '-b', type=int, default=512)
    # Data collection
    parser.add_argument('--random_paths', '-r', type=int, default=100)
    parser.add_argument('--onpol_paths', '-d', type=int, default=10)
    parser.add_argument('--simulated_paths', '-sp', type=int, default=1000)
    parser.add_argument('--ep_len', '-ep', type=int, default=1000)
    # Neural network architecture args
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=500)
    # MPC Controller
    parser.add_argument('--mpc_horizon', '-m', type=int, default=15)
    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    # Make data directory if it does not already exist
    if not(os.path.exists('data')):
        os.makedirs('data')
    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data', logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    # Make env
    # if args.env_name is "HalfCheetah-v1":
    #     env = HalfCheetahEnvNew()
    #     cost_fn = cheetah_cost_fn
    env = gym.make(args.env_name)
    cost_fn = cheetah_cost_fn
    env.reset()
    print('ob dim',env.observation_space.shape[0])
    print('ac dim',env.action_space.shape[0])
    # env.render()
    # done = False
    # while not done:
    #     action = env.action_space.sample()
    #     last_obs, reward, done, info = env.step(action)

    train(env=env,
                 cost_fn=cost_fn,
                 logdir=logdir,
                 render=args.render,
                 learning_rate=args.learning_rate,
                 onpol_iters=args.onpol_iters,
                 dynamics_iters=args.dyn_iters,
                 batch_size=args.batch_size,
                 num_paths_random=args.random_paths,
                 num_paths_onpol=args.onpol_paths,
                 num_simulated_paths=args.simulated_paths,
                 env_horizon=args.ep_len,
                 mpc_horizon=args.mpc_horizon,
                 n_layers = args.n_layers,
                 size=args.size,
                 activation=tf.nn.relu,
                 output_activation=None,
                 )

if __name__ == "__main__":
    main()
