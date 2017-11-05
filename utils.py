import os
import scipy
import numpy as np
import tensorflow as tf

def load_mnist(config):
    fd = open(os.path.join(config.dataset, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

    fd = open(os.path.join(config.dataset, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.float)

    fd = open(os.path.join(config.dataset, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

    fd = open(os.path.join(config.dataset, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.float)

    # normalization and convert to a tensor [60000, 28, 28, 1]
    #trX = tf.convert_to_tensor(trX / 255., tf.float32)
    #teX = tf.convert_to_tensor(teX / 255., tf.float32)
    trX = trX / 255.
    teX = teX / 255.
    # => [num_samples, 10]
    #trY = tf.one_hot(trY, depth=10, axis=1, dtype=tf.float32)
    #teY = tf.one_hot(teY, depth=10, axis=1, dtype=tf.float32)
    if config.is_training:
        return trX, trY
    else:
        return teX, teY

def get_batch_data(config):
    trX, trY = load_mnist(config)

    data_queues = tf.train.slice_input_producer([trX, trY])
    X, Y = tf.train.shuffle_batch(data_queues, num_threads=config.num_threads,
                                  batch_size=config.batchsize,
                                  capacity=config.batchsize * 64,
                                  min_after_dequeue=config.batchsize * 32,
                                  allow_smaller_final_batch=False)

    return X, Y

def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))

# load data from datasets
def get_image(batch_file, is_grayscale=False):
    if is_grayscale:
        return scipy.misc.imread(batch_file, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(batch_file).astype(np.float)
        
def save_images(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    return img
