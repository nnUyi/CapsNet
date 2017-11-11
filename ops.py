import tensorflow as tf

# convolution
def conv2d(input_x, kernel_size, stride=[1,2,2,1], scope_name='conv2d', conv_type='SAME'):
    output_len = kernel_size[3]
    with tf.variable_scope(scope_name):
        weights = tf.get_variable('weights', kernel_size, tf.float32, initializer=tf.random_normal_initializer(stddev=0.02))
        bias = tf.get_variable('bias', output_len, tf.float32, initializer=tf.constant_initializer(0))
        conv = tf.nn.bias_add(tf.nn.conv2d(input_x, weights, strides=stride, padding=conv_type), bias)
        return(conv)

# deconvolution
def deconv2d(input_x, kernel_size, output_shape, stride=[1,2,2,1], scope_name='deconv2d', deconv_type='SAME'):
    output_len = kernel_size[2]
    with tf.variable_scope(scope_name):
        weights = tf.get_variable('weights', kernel_size, tf.float32, initializer=tf.random_normal_initializer(stddev=0.02))
        bias = tf.get_variable('bias', output_len, tf.float32, initializer=tf.constant_initializer(0))
        try:
            deconv = tf.nn.bias_add(tf.nn.conv2d_transpose(input_x, weights, output_shape, strides=stride, padding=deconv_type), bias)
        except:
            deconv = tf.nn.bias_add(tf.nn.deconv2d(input_x, weights, output_shape, strides=stride, padding=deconv_type), bias)
        return deconv

# batch normalization
def batch_normalization(input_x, epsilon=1e-5, momentum=0.9, is_training = True, name='batch_name'):
    with tf.variable_scope(name) as scope:
        batch_norm = tf.contrib.layers.batch_norm(input_x,
                                              decay=momentum,
                                              updates_collections=None,
                                              epsilon=epsilon,
                                              scale=True,
                                              is_training=is_training,
                                              scope=name)
        return batch_norm
        
# fully connected
def linear(input_x, output_size, scope_name='linear'):
    shape = input_x.get_shape()
    input_size = shape[1]
    with tf.variable_scope(scope_name):
        weights = tf.get_variable('weights', [input_size, output_size], tf.float32, initializer=tf.random_normal_initializer(stddev=0.02))
        bias = tf.get_variable('bias', output_size, tf.float32, initializer=tf.constant_initializer(0))        
        output = tf.matmul(input_x, weights) + bias
        return output

# leaky_relu
def leaky_relu(input_x, leaky=0.2):
    return tf.maximum(leaky*input_x, input_x)
