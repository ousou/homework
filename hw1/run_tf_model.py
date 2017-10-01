#!/usr/bin/env python

"""
Code to load a tf policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_tf_model.py model/hopper/hopper.meta Hopper-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of roll outs')
    parser.add_argument('--dump_to_file', type=str, default='')
    args = parser.parse_args()

    with tf.Session() as sess:
        tf_util.initialize()
        print('loading policy from file')
        saver = tf.train.import_meta_graph(args.model_policy_file + '.meta')
        saver.restore(sess, args.model_policy_file)
        print('loaded')
        graph = tf.get_default_graph()
        policy = graph.get_tensor_by_name("neural_net:0")
        x = graph.get_tensor_by_name("input:0")

        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = sess.run(policy, feed_dict={x: [obs]})
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}
        if args.dump_to_file:
            print('dumping result to file', args.dump_to_file)
            with open(args.dump_to_file, 'wb') as f:
                pickle.dump(expert_data, f)


if __name__ == '__main__':
    main()
