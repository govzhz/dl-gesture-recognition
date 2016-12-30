from CNNModel import *
from Input_data import *
import time
from datetime import datetime
import math

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('dropout', 0.8, 'Keep probability for training dropout')
flags.DEFINE_integer('max_steps', 400, 'Number of steps to run trainer')
flags.DEFINE_integer('width_image', 60, 'Width of images')
flags.DEFINE_integer('height_image', 60, 'Height of images')
flags.DEFINE_integer('channel', 3, 'Channel of images')
flags.DEFINE_integer('num_classes', 4, 'The number of image classes')
flags.DEFINE_integer('batch_size', 128, 'Number of images to process in a batch')
flags.DEFINE_float('learning_rate', 9e-4, 'Initial learning rate')
flags.DEFINE_integer('steps_per_print_loss', 50, 'Print information per steps')
flags.DEFINE_integer('steps_per_print_accuracy', 100, 'Print information per steps')
flags.DEFINE_string('model_dir', './model', 'Directory where to save model')


def train():
    # 读取数据
    znyp = read_data_sets("ZNYP_data")

    # 创建占位符
    x = tf.placeholder(tf.float32, [None, FLAGS.height_image, FLAGS.width_image, FLAGS.channel])
    y = tf.placeholder(tf.float32, [None, FLAGS.num_classes])
    keep_prob = tf.placeholder(tf.float32)

    # 构建CNN模型
    out = create_cnn_model(x, FLAGS.height_image, FLAGS.width_image, FLAGS.channel, FLAGS.num_classes, FLAGS.dropout)

    # 计算损失
    total_loss = compute_loss(out, y)

    # adam梯度下降最小化loss
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(total_loss)

    # 训练
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        for step in range(FLAGS.max_steps):
            start_time = time.time()
            # 每次训练一个batch_size的数据
            batch_x, _, batch_y_one_hot = znyp.train.next_batch(FLAGS.batch_size)
            _, loss = sess.run([optimizer, total_loss], feed_dict={x: batch_x, y: batch_y_one_hot, keep_prob: FLAGS.dropout})
            duration = time.time() - start_time

            # 打印训练集验证集准确率
            if step % FLAGS.steps_per_print_accuracy == 0:
                x_val, y_val_one_hot = znyp.validation.images, znyp.validation.labels_one_hot
                acc_train = compute_accuracy(sess, batch_x, batch_y_one_hot)
                acc_val = compute_accuracy(sess, x_val, y_val_one_hot)
                print('%s: %s precision @ 1 = %.3f \n%s: %s precision @ 1 = %.3f ' %
                      (datetime.now(), 'train', acc_train, datetime.now(), 'validation', acc_val))

            # 打印损失信息
            if step % FLAGS.steps_per_print_loss == 0:
                num_examples_per_step = FLAGS.batch_size
                example_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
                print('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)' %
                      (datetime.now(), step, loss, example_per_sec, sec_per_batch))

            # 保存模型
            if step % 50 == 0 or (step + 1) == FLAGS.max_steps:
                if not os.path.exists(FLAGS.model_dir):
                    os.mkdir(FLAGS.model_dir)
                checkpoint_path = os.path.join(FLAGS.model_dir, 'model')
                saver.save(sess, checkpoint_path)

        # 打印测试集准确率
        x_test, y_test_one_hot = znyp.test.images, znyp.test.labels_one_hot
        acc_test = compute_accuracy(sess, x_test, y_test_one_hot)
        print('%s: %s precision @ 1 = %.3f' %
              (datetime.now(), 'test', acc_test))


# 计算准确率
def compute_accuracy(sess, x, y):
    '''
    # 这种方式当测试集过大会导致内存不够
    out = create_cnn_model(x, FLAGS.height_image, FLAGS.width_image, FLAGS.channel,
                           FLAGS.num_classes, 1.0, reuse=True)
    correct_pred = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    precision = sess.run(accuracy)
    '''
    num_images = x.shape[0]
    num_iter = int(math.ceil(num_images / FLAGS.batch_size))

    true_count = 0
    for step in range(num_iter):
        image_batchsize = x[int(step * FLAGS.batch_size):int((step + 1) * FLAGS.batch_size)]
        label_batchsize = y[int(step * FLAGS.batch_size):int((step + 1) * FLAGS.batch_size)]
        out = create_cnn_model(image_batchsize, FLAGS.height_image, FLAGS.width_image, FLAGS.channel,
                               FLAGS.num_classes, 1.0, reuse=True)
        correct_pred = tf.equal(tf.argmax(out, 1), tf.argmax(label_batchsize, 1))
        true_count += np.sum(sess.run(correct_pred))
    precision = true_count / num_images
    return precision

if __name__ == '__main__':
    train()