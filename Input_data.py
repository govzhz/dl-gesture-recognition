import numpy as np
import pickle
import os
import collections


class DataSet(object):
    def __init__(self, images, labels, labels_ont_hot):
        self._images = images
        self._labels = labels
        self._labels_one_hot = labels_ont_hot
        self._index_in_epoch = 0
        self._num_images = images.shape[0]

    @property
    def images(self):
        """
            self._images: (N, H, W, C)
        """
        return self._images

    @property
    def labels(self):
        """
            self._labels: (N, )
        """
        return self._labels

    @property
    def labels_one_hot(self):
        """
            self._labels_ont_hot: (N, )
        """
        return self._labels_one_hot

    def next_batch(self, batch_size):
        """Return the next batch_size examples from the data set"""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_images:
            # shuffle the data
            perm = np.arange(self._num_images)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            self._labels_one_hot = self._labels_one_hot[perm]
            # reset
            start = 0
            self._index_in_epoch = batch_size

        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end], self._labels_one_hot[start:end]


def _unpickle_dataset(filename):
    """ Unpickle Dataset

    input:
        filename: the save location of "pickled" object
        category: dataset category

    return:
        X: images (N, H, W, C)
        Y: labels (N, )
    """
    with open(filename, 'rb') as f:
        datadict = pickle.load(f)
        X = datadict['data']
        Y = datadict['labels']
        H = datadict['height']
        W = datadict['width']
        C = datadict['channel']
        num_images = datadict['num_images']
        X = X.reshape(num_images, C, H, W).transpose(0, 2, 3, 1).astype("float32")
        Y = np.array(Y)
        Y_one_hot = dense_to_one_hot(Y, 4)
        return X, Y, Y_one_hot


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def read_data_sets(dataset_root):
    """load dataset

    input:
        dataset_root: the root location of dataset
    """
    TRAIN_DATASET = 'train_data'
    TEST_DATASET = 'test_data'
    VALIDATION_SIZE = 50

    train_file = os.path.join(dataset_root, TRAIN_DATASET)
    x_train, y_train, y_train_one_hot = _unpickle_dataset(train_file)

    test_file = os.path.join(dataset_root, TEST_DATASET)
    x_test, y_test, y_test_one_hot = _unpickle_dataset(test_file)

    x_validation, y_validation, y_validation_ont_hot = x_train[:VALIDATION_SIZE], y_train[:VALIDATION_SIZE], y_train_one_hot[:VALIDATION_SIZE]
    x_train, y_train, y_train_one_hot = x_train[VALIDATION_SIZE:], y_train[VALIDATION_SIZE:], y_train_one_hot[VALIDATION_SIZE:]

    train = DataSet(x_train, y_train, y_train_one_hot)
    validation = DataSet(x_validation, y_validation, y_validation_ont_hot)
    test = DataSet(x_test, y_test, y_test_one_hot)

    print('Train set: %s' % train.labels.shape)
    print('validation set: %s' % validation.labels.shape)
    print('test set: %s' % test.labels.shape)

    Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])
    return Datasets(train=train, validation=validation, test=test)

if __name__ == '__main__':
    znyp = read_data_sets('./ZNYP_data')
    # print(znyp.train.images.dtype)
    # print(znyp.test.labels.shape)
    # print(znyp.validation.labels.shape)