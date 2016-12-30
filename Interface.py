import tensorflow as tf


# 卷积
def conv2d(x, W, b, strides=1):
    #   padding=SAME: to ensure that the same images size
    #   padding_width = (kernel_width - 1) / 2
    #   padding_height = (kernel_height - 1) / 2
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


# 池化
def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# 权重
def weight_variable(name, shape, stddev=5e-2, reg=None, reuse=None):
    """ create an initialized variable Weight

    :param shape: convolution kernel (list of ints)
    :param stddev: weight scales
    :param add L2Loss weight decay if it's not null
    :return: Variable Tensor Weight
    """
    with tf.variable_scope(name, reuse=reuse):
        initializer = tf.truncated_normal_initializer(stddev=stddev)
        weight = tf.get_variable('weights', shape, initializer=initializer)

        if reg is not None:
            # output = sum(t ** 2) / 2, t is Weights here
            weight_l2_loss = tf.mul(tf.nn.l2_loss(weight), reg, name='l2_loss')
            tf.add_to_collection('losses', weight_l2_loss)
    return weight


# 偏置
def bias_variable(name, shape, initvalue=0.0, reuse=None):
    """ create an initialized variable Bias

    :param shape: list of ints
    :return: Variable Tensor Bias
    """
    with tf.variable_scope(name, reuse=reuse):
        initializer = tf.constant_initializer(initvalue)
        var = tf.get_variable('biases', shape, initializer=initializer)
    return var