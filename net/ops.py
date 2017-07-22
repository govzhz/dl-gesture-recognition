import numpy as np
import tensorflow as tf

from net.cell import ConvLSTMCell


def conv_lstm_layer(name, x, shape, filters, kernel=None):
    """

    :param x: [batch_size, timesteps] + shape + [channels]
    :param shape: (width, height)
    :param filters: n_hidden
    :param kernel: filter size
    :return:
    """
    with tf.variable_scope(name):
        if kernel is None:
            kernel = [3, 3]
        cell = ConvLSTMCell(shape, filters, kernel)
        outputs, state = tf.nn.dynamic_rnn(cell, x, dtype=x.dtype)  # state: (c, h)
    return outputs, state


def conv3d_layer(name, x, w, b):
    return tf.nn.bias_add(
        tf.nn.conv3d(x, w, strides=[1, 1, 1, 1, 1], padding='SAME', name=name),
        b)


def fc_with_dropout_layer(name, x, w, b, dropout):
    with tf.variable_scope(name):
        fc = tf.add(tf.matmul(x, w), b)
        fc = tf.nn.relu(fc)
    return tf.nn.dropout(fc, dropout)


def max_pool_layer(name, x, k):
    return tf.nn.max_pool3d(x, ksize=[1, k, 2, 2, 1], strides=[1, k, 2, 2, 1], padding='SAME', name=name)


def spp_layer(name, input_, levels):
    shape = input_.get_shape().as_list()
    with tf.variable_scope(name):
        pyramid = []
        for n in levels:
            # https://github.com/tensorflow/tensorflow/issues/6011
            # https://stackoverflow.com/questions/40913794/how-to-implement-the-fixed-length-spatial-pyramid-pooling-layer
            # compare np.floor or np.ceil
            stride_1 = np.floor(float(shape[1] / n)).astype(np.int32)
            stride_2 = np.floor(float(shape[2] / n)).astype(np.int32)
            ksize_1 = stride_1 + (shape[1] % n)
            ksize_2 = stride_2 + (shape[2] % n)
            pool = tf.nn.max_pool(input_,
                                  ksize=[1, ksize_1, ksize_2, 1],
                                  strides=[1, stride_1, stride_2, 1],
                                  padding='VALID')

            print(" Pool Level {}: shape {}".format(n, pool.get_shape().as_list()))
            pyramid.append(tf.reshape(pool, [shape[0], -1]))
        spp_pool = tf.concat(pyramid, 1)
    return spp_pool


def batch_norm_layer(name, x, is_training, epsilon=0.001, decay=0.99, reuse=None):
    """
    Performs a batch normalization layer

    Args:
        x: input tensor
        scope: scope name
        is_training: python boolean value
        epsilon: the variance epsilon - a small float number to avoid dividing by 0
        decay: the moving average decay

    Returns:
        The ops of a batch normalization layer
    """
    with tf.variable_scope(name, reuse=reuse):
        shape = x.get_shape().as_list()
        # gamma: a trainable scale factor
        gamma = tf.get_variable("gamma", shape[-1], initializer=tf.constant_initializer(1.0), trainable=True)
        # beta: a trainable shift value
        beta = tf.get_variable("beta", shape[-1], initializer=tf.constant_initializer(0.0), trainable=True)
        moving_avg = tf.get_variable("moving_avg", shape[-1], initializer=tf.constant_initializer(0.0), trainable=False)
        moving_var = tf.get_variable("moving_var", shape[-1], initializer=tf.constant_initializer(1.0), trainable=False)
        if is_training:
            # tf.nn.moments == Calculate the mean and the variance of the tensor x
            avg, var = tf.nn.moments(x, range(len(shape)-1))
            update_moving_avg = moving_averages.assign_moving_average(moving_avg, avg, decay)
            update_moving_var = moving_averages.assign_moving_average(moving_var, var, decay)
            control_inputs = [update_moving_avg, update_moving_var]
        else:
            avg = moving_avg
            var = moving_var
            control_inputs = []
        with tf.control_dependencies(control_inputs):
            output = tf.nn.batch_normalization(x, avg, var, offset=beta, scale=gamma, variance_epsilon=epsilon)

    return output


def weight_variable(name, shape, stddev=5e-2, reg=None, reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        initializer = tf.truncated_normal_initializer(stddev=stddev)
        weight = tf.get_variable('weights', shape, initializer=initializer)

        if reg is not None:
            # output = sum(t ** 2) / 2, t is Weights here
            weight_l2_loss = tf.multiply(tf.nn.l2_loss(weight), reg, name='l2_loss')
            tf.add_to_collection('losses', weight_l2_loss)
    return weight


def bias_variable(name, shape, initvalue=0.0, reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        initializer = tf.constant_initializer(initvalue)
        var = tf.get_variable('biases', shape, initializer=initializer)
    return var