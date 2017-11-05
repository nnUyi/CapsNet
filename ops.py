import tensorflow as tf
import numpy as np
import scipy.misc
import math

# return required weights
def weights_variable(shape, name):
    # get_variable or Variable
    with tf.variable_scope(name):
        w = tf.get_variable('weights', shape, tf.float32, initializer=tf.random_normal_initializer(stddev=0.02))
    #w = tf.Variable(tf.random_normal(shape))
    return w
    
# return required bias
def bias_variable(shape, name):
    with tf.variable_scope(name):
        b = tf.get_variable('bias', shape, tf.float32, initializer=tf.constant_initializer(0.0))
    #b = tf.Variable(tf.random_normal(shape))
    return b

# conv2d
def conv2d(input_data_x, weights, conv_type='SAME'):
    # parameters of tf.nn.conv2d(input, filter, strides, padding,...)
    # input:[batch, height, width, channels]
    # filter:[height, width, in_channels, out_channels]
    if conv_type == 'SAME':
        return tf.nn.conv2d(input_data_x, weights, strides=[1,2,2,1], padding=conv_type)
    else:
        return tf.nn.conv2d(input_data_x, weights, strides=[1,2,2,1], padding=conv_type)

# pooling
def max_pool(input_data_x, filter_shape=[1,2,2,1], pooling_type='SAME'):
    if pooling_type == 'SAME':
        return tf.nn.max_pool(input_data_x, ksize=filter_shape, strides=[1,2,2,1], padding=pooling_type)
    else:
        return tf.nn.max_pool(input_data_x, ksize=filter_shape, strides=[1,2,2,1], padding=pooling_type)

# deconv2d
def deconv2d(input_data_x, filter_, output_shape, deconv_type='SAME'):
    try:
        return tf.nn.conv2d_transpose(input_data_x, filter_, output_shape, strides = [1,2,2,1])
    except AttributeError:
        return tf.nn.deconv2d(input_data_x, filter_, output_shape, strides=[1,2,2,1])

#batch_norm
class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon  = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum, 
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name)

#batch_normalization
def batch_normalization(x, epsilon=1e-5, momentum=0.9, train = True, name='batch_name', reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        batch_norm = tf.contrib.layers.batch_norm(x,
                                              decay=momentum,
                                              updates_collections=None,
                                              epsilon=epsilon,
                                              scale=True,
                                              is_training=train,
                                              scope=name)
        return batch_norm

# leakly relu
def leakly_relu(x, leakly=0.2, name='leakly_relu'):
    return tf.maximum(x, leakly*x)

# linear fully connected
def linear(input_, output_size, scope='linear', stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()
    with tf.variable_scope(scope):
        matrix = tf.get_variable("matrix_", [shape[1], output_size], tf.float32, tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias_", [output_size],tf.float32, tf.constant_initializer(0.0))
    #return tf.matmul(input_, matrix) + bias, matrix, bias
    if with_w:
    	return tf.matmul(input_, matrix) + bias, matrix, bias
    else:
    	return tf.matmul(input_, matrix) + bias

# conv out size
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
