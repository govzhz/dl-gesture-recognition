from net.ops import *

weights = {
    'wc3d1': weight_variable('wc3d1', shape=[3, 3, 3, 3, 64]),
    'wc3d2': weight_variable('wc3d2', shape=[3, 3, 3, 64, 128]),
    'wc3d3': weight_variable('wc3d3', shape=[3, 3, 3, 128, 256]),
    'wc3d4': weight_variable('wc3d4', shape=[3, 3, 3, 256, 256]),
    'wfc': weight_variable('wfc', shape=[26880, 249])
}

biases = {
    'bc3d1': bias_variable('bc3d1', shape=[64]),
    'bc3d2': bias_variable('bc3d2', shape=[128]),
    'bc3d3': bias_variable('bc3d3', shape=[256]),
    'bc3d4': bias_variable('bc3d4', shape=[256]),
    'bfc': bias_variable('bfc', shape=[249])
}

convLstm = {
    'n_hidden1': 256,
    'n_hidden2': 384
}


def c3d_convlstm(x, is_training=None):
    assert is_training is not None

    # Conv3dlayer 1
    conv3d_1 = conv3d_layer('Conv3d_1', x, weights['wc3d1'], biases['bc3d1'])
    batch_norm_1 = batch_norm_layer('BatchNorm_1', conv3d_1, is_training)
    pool_1 = max_pool_layer('Pool3d_1', batch_norm_1, k=1)
    print(" Conv3dlayer Level {}: shape {}".format(1, pool_1.get_shape().as_list()))

    # Conv3dlayer 2
    conv3d_2 = conv3d_layer('Conv3d_2', pool_1, weights['wc3d2'], biases['bc3d2'])
    batch_norm_2 = batch_norm_layer('BatchNorm_2', conv3d_2, is_training)
    pool_2 = max_pool_layer('Pool3d_2', batch_norm_2, k=2)
    print(" Conv3dlayer Level {}: shape {}".format(2, pool_2.get_shape().as_list()))

    # Conv3dlayer 3
    conv3d_3 = conv3d_layer('Conv3d_3', pool_2, weights['wc3d3'], biases['bc3d3'])
    print(" Conv3dlayer Level {}: shape {}".format(3, conv3d_3.get_shape().as_list()))

    # Conv3dlayer 4
    conv3d_4 = conv3d_layer('Conv3d_4', conv3d_3, weights['wc3d4'], biases['bc3d4'])
    batch_norm_4 = batch_norm_layer('BatchNorm_4', conv3d_4, is_training)
    print(" Conv3dlayer Level {}: shape {}".format(4, batch_norm_4.get_shape().as_list()))

    # ConvLstmlayer 1
    shape3d = batch_norm_4.get_shape().as_list()
    convlstm1, _ = conv_lstm_layer('ConvLstm_1', batch_norm_4, shape=[shape3d[2], shape3d[3]], filters=convLstm['n_hidden1'])
    print(" ConvLstmlayer Level {}: shape {}".format(1, convlstm1.get_shape().as_list()))

    # ConvLstmlayer 2
    convlstm2, _ = conv_lstm_layer('ConvLstm_2', convlstm1, shape=[shape3d[2], shape3d[3]], filters=convLstm['n_hidden2'])
    convlstm2 = convlstm2[:, -1, :]
    print(" ConvLstmlayer Level {}: shape {}".format(2, convlstm2.get_shape().as_list()))

    # Spplayer
    spp = spp_layer('SPP_layer', convlstm2, levels=(1, 2, 4, 7))
    print(" Spplayer Level: shape {}".format(spp.get_shape().as_list()))

    # FC
    fc = fc_with_dropout_layer('FC', spp, weights['wfc'], biases['bfc'], dropout=0.5)
    print(" FC Level: shape {}".format(fc.get_shape().as_list()))
    return fc