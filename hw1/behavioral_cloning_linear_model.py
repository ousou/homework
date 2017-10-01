import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy

def main():
    expert_data = read_expert_data('expert_data/hopper_expert_data.pkl')
    all_obs = expert_data['observations']
    all_acts = expert_data['actions']
    print('number of data points in data set:', all_obs.shape[0])

    training_set, test_set = split_data(expert_data, 0.8)
    print('training_set size:', training_set['observations'].shape[0])
    print('test_set size:', test_set['observations'].shape[0])

    training_set_size = training_set['observations'].shape[0]

    input_vector_size = all_obs.shape[1]
    output_vector_size = all_acts.shape[1]
    print('input', input_vector_size)
    print('output', output_vector_size)
    batch_size = 500
    training_steps = 100000
    display_step = 1000
    learning_rate = 0.5

    x = tf.placeholder(tf.float32, [None, input_vector_size], name='input')
    W = tf.Variable(tf.zeros([input_vector_size, output_vector_size]))
    b = tf.Variable(tf.zeros([output_vector_size]))

    y = tf.nn.softmax(tf.matmul(x, W) + b)
    y_ = tf.placeholder(tf.float32, [None, output_vector_size], name='actual_output')

    mse = tf.reduce_sum(tf.pow(y - y_, 2)) / (2 * training_set_size)
    # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(mse)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    for epoch in range(training_steps):
        indices = np.random.choice(len(training_set['observations']), batch_size, replace=False)
        batch_xs = training_set['observations'][indices]
        batch_ys = training_set['actions'][indices]
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        if (epoch + 1) % display_step == 0:
            cost = sess.run(mse, feed_dict={x: test_set['observations'], y_: test_set['actions']})
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(cost))

    print("Optimization Finished!")
    cost = sess.run(mse, feed_dict={x: test_set['observations'], y_: test_set['actions']})
    print("cost=", "{:.9f}".format(cost), \
          "W=", sess.run(W), "b=", sess.run(b))


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