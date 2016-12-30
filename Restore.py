from PIL import Image
import os
import numpy as np
from CNNModel import *

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('width_image', 60, 'Width of images')
flags.DEFINE_integer('height_image', 60, 'Height of images')
flags.DEFINE_integer('channel', 3, 'Channel of images')
flags.DEFINE_integer('num_classes', 4, 'The number of image classes')


def restore(model_dir, predict_images_dir):
    # 读取预测数据
    predict_data = read_predict_images_set(predict_images_dir)

    # 构建模型前向传播
    x = tf.placeholder(tf.float32, [None, FLAGS.height_image, FLAGS.width_image, FLAGS.channel])
    out = create_cnn_model(x, FLAGS.height_image, FLAGS.width_image, FLAGS.channel,
                           FLAGS.num_classes, 1.0)
    prediction = tf.argmax(out, 1)

    # 加载保存参数
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, model_dir)
        # 输出预测值
        return sess.run(prediction, feed_dict={x: predict_data})


def reset():
    tf.reset_default_graph()


def read_predict_images_set(predict_images_dir):
    # get the predict images location and sort it
    Filelist = []
    for dirname in os.listdir(predict_images_dir):
        path = os.path.join(predict_images_dir, dirname)
        if len(Filelist) == 0:
            Filelist.append(path)
        else:
            int_name = int(dirname.split('.')[0])
            for index, existdirname in enumerate(Filelist):
                int_exist_name = int(os.path.basename(existdirname).split('.')[0])
                if int_name < int_exist_name:
                    Filelist.insert(index, path)
                    break
            else:
                Filelist.append(path)
    # print(Filelist)

    # create dataset
    data = []
    num_images = 0
    for idx, filename in enumerate(Filelist):
        im = Image.open(filename)
        im = (np.array(im))
        H, W = im.shape[0], im.shape[1]
        if H != FLAGS.width_image or W != FLAGS.height_image:
            raise Exception('Please make sure you use the corrent image(width/height)')

        if FLAGS.channel == 1:
            r = im.flatten()
            g = []
            b = []
        elif FLAGS.channel == 3:
            # get pixel from red channel, then green then blue
            r = im[:, :, 0].flatten()
            g = im[:, :, 1].flatten()
            b = im[:, :, 2].flatten()
        else:
            raise Exception('The channel for the image should be 1 or 3')

        num_images += 1
        # append the pixel
        data += (list(r) + list(g) + list(b))

    # convert the list to numpy
    data = np.array(data, np.uint8)
    # reshape
    predict_data = data.reshape(num_images, FLAGS.channel, FLAGS.height_image, FLAGS.width_image).transpose(0, 2, 3, 1).astype("float32")
    return predict_data

if __name__ == '__main__':
    prediction = restore(model_dir="./model/model", predict_images_dir='./predict_image')
    print('Prediction: %s' % prediction)