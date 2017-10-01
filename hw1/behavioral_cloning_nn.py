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
    n_hidden_1 = 50
    n_hidden_2 = 50
    batch_size = 500
    training_steps = 200000
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

    for step in range(training_steps):
        indices = np.random.choice(len(training_set['observations']), batch_size, replace=False)
        batch_xs = training_set['observations'][indices]
        batch_ys = training_set['actions'][indices]
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        if (step + 1) % display_step == 0:
            cost = sess.run(mse, feed_dict={x: test_set['observations'], y_: test_set['actions']})
            print("Step:", '%04d' % (step + 1), "cost=", "{:.9f}".format(cost))

    print("Optimization Finished!")
    cost = sess.run(mse, feed_dict={x: test_set['observations'], y_: test_set['actions']})

 #   print(sess.run(y, feed_dict={x: test_set['observations'][0:1]}))
 #   print("h1", sess.run(weights['h1']))
    print("cost=", "{:.9f}".format(cost))

    saver.save(sess, 'saved_model_data/hopper/hopper_try2')

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