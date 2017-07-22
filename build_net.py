import tensorflow as tf

from net.C3D_ConvLstm import c3d_convlstm

x = tf.placeholder(tf.float32, [3, 32, 112, 112, 3], name='x')
y = tf.placeholder(tf.int32, shape=[3, ], name='y')

sess = tf.InteractiveSession()

print('Build C3D-ConvLstm Model...')
net = c3d_convlstm(x, is_training=False)

sess.run(tf.global_variables_initializer())


