import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--write_policy_to_file', type=str, default='')
    args = parser.parse_args()

    expert_policy = load_policy.load_policy(args.expert_policy_file)

    import gym
    env = gym.make(args.envname)
    max_steps = env.spec.timestep_limit

    print('Running expert policy')
    expert_obs, expert_actions = run_expert_policy(expert_policy, env, max_steps, 3)

    expert_data = {
        'observations': np.array(expert_obs),
        'actions': np.array(expert_actions)
    }

    print('number of data points in data set:', len(expert_obs))

    training_set, test_set = split_data(expert_data, 0.8)
    print('training_set size:', training_set['observations'].shape[0])
    print('test_set size:', test_set['observations'].shape[0])

    training_set_size = training_set['observations'].shape[0]

    input_vector_size = training_set['observations'].shape[1]
    output_vector_size = training_set['actions'].shape[1]
    print('input', input_vector_size)
    print('output', output_vector_size)
    n_hidden_1 = 50
    n_hidden_2 = 50
    batch_size = 500
    training_steps = 30000
    collect_data_step = 500
    display_step = 1000
    learning_rate = 0.01

    weights = {
        'h1': tf.Variable(tf.random_normal([input_vector_size, n_hidden_1]), name='h1'),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name='h2'),
        'out': tf.Variable(tf.random_normal([n_hidden_2, output_vector_size]), name='out')
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1]), name='b1'),
        'b2': tf.Variable(tf.random_normal([n_hidden_2]), name='b2'),
        'out': tf.Variable(tf.random_normal([output_vector_size]), name='out')
    }

    x = tf.placeholder(tf.float32, [None, input_vector_size], name='input')

    y = neural_net(x, weights, biases)
    y_ = tf.placeholder(tf.float32, [None, output_vector_size], name='actual_output')

    mse = tf.reduce_sum(tf.pow(y-y_, 2))/(2*training_set_size)
    # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(mse)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    saver = tf.train.Saver()
    observations = training_set['observations']
    actions = training_set['actions']

    print('observations size', observations.shape[0])
    print('actions size', actions.shape[0])
    for step in range(training_steps):
        indices = np.random.choice(len(observations), batch_size, replace=False)
        batch_xs = observations[indices]
        batch_ys = actions[indices]
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        if (step + 1) % collect_data_step == 0:
            new_observations, new_actions = run_policy(sess, x, y, expert_policy, env, max_steps, 1)
            # amount = len(new_observations)
            # observations = observations[amount:]
            # actions = actions[amount:]
            observations = np.append(observations, np.array(new_observations), axis=0)
            actions = np.append(actions, np.array(new_actions), axis=0)
        if (step + 1) % display_step == 0:
            cost = sess.run(mse, feed_dict={x: test_set['observations'], y_: test_set['actions']})
            print("Step:", '%04d' % (step + 1), "cost=", "{:.9f}".format(cost))


    print("Optimization Finished!")
    cost = sess.run(mse, feed_dict={x: test_set['observations'], y_: test_set['actions']})

 #   print(sess.run(y, feed_dict={x: test_set['observations'][0:1]}))
 #   print("h1", sess.run(weights['h1']))
    print("cost=", "{:.9f}".format(cost))
    if args.write_policy_to_file:
        print('writing policy file', args.write_policy_to_file)
        saver.save(sess, args.write_policy_to_file)

def run_expert_policy(expert_policy, env, max_steps, iterations):
    with tf.Session():
        tf_util.initialize()
        observations = []
        actions = []
        for i in range(iterations):
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = expert_policy(obs[None,:])
                observations.append(obs)
                actions.append(action.flatten())
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                #if steps % 100 == 0: print("%i/%i" % (steps, max_steps))
                if steps >= max_steps:
                    break
        return observations, actions


def run_policy(sess, x, trained_policy, expert_policy, env, max_steps, iterations):
    observations = []
    actions = []
    for i in range(iterations):
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = sess.run(trained_policy, feed_dict={x: [obs]})
            observations.append(obs)
            expert_action = expert_policy(obs[None,:])
            actions.append(expert_action.flatten())
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            #if steps % 100 == 0: print("%i/%i" % (steps, max_steps))
            if steps >= max_steps:
                break
        print("Steps: %i/%i" % (steps, max_steps))
    return observations, actions

def neural_net(x, weights, biases):
    # Hidden fully connected layer
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.elu(layer_1)
    # Hidden fully connected layer
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.elu(layer_2)
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    out_layer = tf.identity(out_layer, name="neural_net")
    return out_layer


def split_data(data_set, test_data_perc):
    test_data_end_index = round(test_data_perc * data_set['observations'].shape[0])
    test_set = {
        'observations': data_set['observations'][:test_data_end_index],
        'actions': data_set['actions'][:test_data_end_index]
    }
    validation_set = {
        'observations': data_set['observations'][test_data_end_index:],
        'actions': data_set['actions'][test_data_end_index:]
    }
    return test_set, validation_set


def read_expert_data(filename):
    with open(filename, 'rb') as f:
        expert_data = pickle.loads(f.read())
    expert_data['actions'] = get_clean_actions(expert_data)
    return expert_data

def get_clean_actions(expert_data):
    actions_raw = expert_data['actions']
    actions = np.empty((actions_raw.shape[0], actions_raw.shape[2]))
    for i in range (0, actions_raw.shape[0]):
        actions[i] = actions_raw[i].flatten()
    return actions

if __name__ == '__main__':
    main()