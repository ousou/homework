import tensorflow as tf

def main():
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    # L has shape [2, 5, 2]
    L = tf.constant([[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0]])
    print(sess.run(tf.nn.softmax(L)))
    print(sess.run(tf.nn.log_softmax(L)))
    print(sess.run(tf.argmax(L,1)))
    # print(sess.run(L))
    # dims = L.get_shape().as_list()
    # print(dims)
    # N = dims[-1]  # here N = 2
    # print(N)
    #
    # logits = tf.reshape(L, [-1, N])  # shape [10, 2]
    # print(sess.run(logits))

    samples = tf.multinomial(L, 5)
    print(samples)
    # We reshape to match the initial shape minus the last dimension
    # res = tf.reshape(samples, 4)

    print(sess.run(samples))

if __name__ == "__main__":
    main()