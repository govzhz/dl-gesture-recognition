from Interface import *


# 构建CNN模型
def create_cnn_model(x, height_image, width_image, channel, num_classes, dropout, reuse=False):
    weights = {
        # Conv(卷积层)：每个卷积核大小 5 * 5 * 3
        'wc1': weight_variable('wc1', shape=[5, 5, 3, 32], reuse=reuse),
        # Conv(卷积层)：每个卷积核大小 5 * 5 * 32
        'wc2': weight_variable('wc2', shape=[5, 5, 32, 64], reuse=reuse),
        # FC(全连接层), 15*15*64 inputs, 1024 outputs
        'wd1': weight_variable('wd1', shape=[15*15*64, 1024], reuse=reuse),
        # Out(输出层)：1024 inputs, 10 outputs (class prediction)
        'out': weight_variable('out', shape=[1024, num_classes], reuse=reuse)
    }

    biases = {
        'bc1': bias_variable('bc1', shape=[32], reuse=reuse),
        'bc2': bias_variable('bc2', shape=[64], reuse=reuse),
        'bd1': bias_variable('bd1', shape=[1024], reuse=reuse),
        'out': bias_variable('out', shape=[num_classes], reuse=reuse),
    }

    # Reshape input image
    x = tf.reshape(x, shape=[-1, height_image, width_image, channel])

    # Convolution Layer1
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1)

    # Convolution Layer2
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2)

    # Fully connected layer1
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output layer
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


# 计算损失
def compute_loss(out, y):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(out, y))
    return loss