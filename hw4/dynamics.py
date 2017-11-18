import tensorflow as tf
import numpy as np
import gym

# Predefined function to build a feedforward neural network
def build_mlp(input_placeholder, 
              output_size,
              scope, 
              n_layers=2, 
              size=500, 
              activation=tf.tanh,
              output_activation=None
              ):
    out = input_placeholder
    with tf.variable_scope(scope):
        for _ in range(n_layers):
            out = tf.layers.dense(out, size, activation=activation)
        out = tf.layers.dense(out, output_size, activation=output_activation)
    return out

class NNDynamicsModel():
    def __init__(self, 
                 env, 
                 n_layers,
                 size, 
                 activation, 
                 output_activation, 
                 normalization,
                 batch_size,
                 iterations,
                 learning_rate,
                 sess
                 ):
        self.env = env
        self.state_mean = normalization[0]
        self.state_std = normalization[1]
        self.delta_mean = normalization[2]
        self.delta_std = normalization[3]
        self.act_mean = normalization[4]
        self.act_std = normalization[5]
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.sess = sess
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.n if isinstance(env.action_space, gym.spaces.Discrete) else env.action_space.shape[0]
        self.input_ph = tf.placeholder(shape=[None, ob_dim + ac_dim], name="input", dtype=tf.float32)
        self.model = build_mlp(self.input_ph, ob_dim, "dyn_model", n_layers, size, activation, output_activation)
        self.eps = 0.000001

    def fit(self, data):
        """
        Write a function to take in a dataset of (unnormalized)states, (unnormalized)actions, (unnormalized)next_states and fit the dynamics model going from normalized states, normalized actions to normalized state differences (s_t+1 - s_t)
        """

        """YOUR CODE HERE """

    def predict(self, states, actions):
        output_states = []
        for i in range(len(states)):
            state_input = self._normalize_state(states[i])
            action_input = self._normalize_action(actions[i])
            # state_input = (states[i] - self.state_mean) / (self.state_std + self.eps)
            # action_input = (actions[i] - self.act_mean) / (self.act_std + self.eps)
            input = np.concatenate((state_input, action_input), axis=0)
            state_delta = self.sess.run(self.model, feed_dict={self.input_ph: [input]})
            state_delta = self._denormalize_state_diff(state_delta)
            output_states.append(np.sum(state_delta, states[i]))
        return np.array(output_states)

    def _denormalize_state_diff(self, state):
        state = np.array(state)
        return self.delta_mean + (self.delta_std * state)

    def _normalize_state(self, state):
        return (state - self.state_mean) / (self.state_std + self.eps)

    def _normalize_action(self, action):
        return (action - self.act_mean) / (self.act_std + self.eps)
