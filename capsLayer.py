import tensorflow as tf
import numpy as np
from ops import *

class CapsLayer():
    def __init__(self, layer_type, config, output_num = 10, output_len=16, caps_len=8):
        self.layer_type = layer_type
        self.caps_len = caps_len
        self.config = config
        self.output_num = output_num
        self.output_len = output_len

    def capsLayer(self, input_x, kernel_size, stride=[1,2,2,1], scope_name='capsLayer', conv_type='SAME'):
        if self.layer_type == 'conv':
            self.input_shape = input_x.get_shape()
            capsules = []
            with tf.variable_scope(scope_name):
                for i in range(self.caps_len):
                    with tf.variable_scope(scope_name+ '_unit_' + str(i)):
                        caps_i = conv2d(input_x, kernel_size=kernel_size, stride=[1,2,2,1], conv_type=conv_type)
                        caps_i_shape = caps_i.get_shape()
                        caps_i = tf.expand_dims(caps_i, axis=3)
                        capsules.append(caps_i)

                #print(capsules)                
                assert capsules[0].get_shape() == [self.config.batchsize, caps_i_shape[1], caps_i_shape[2], 1, caps_i_shape[3]]
                # shape = [self.config.batchsize, caps_i_shape[1], caps_i_shape[2], self.caps_len, caps_i_shape[3]]
                capsules = tf.concat(capsules, 3)
                capsules_squash = self.squashing(capsules)
                return(capsules_squash)
        elif self.layer_type == 'fc':
            input_x = tf.reshape(input_x, shape=[self.config.batchsize, -1, 1, self.caps_len, 1])
            with tf.variable_scope(scope_name):
                with tf.variable_scope(scope_name+'_routing'):
                    # b_ij: [1, num_caps, num_outputs, 1, 1]
                    b_ij = tf.constant(np.zeros([1, input_x.shape[1].value, self.output_num, 1, 1], dtype=np.float32))
                    capsules = self.routing(input_x, b_ij)
                    # squeeze dim1
                    capsules = tf.squeeze(capsules, axis=1)
            return capsules
    
    def squashing(self, tensor):
        tensor_norm = tf.reduce_sum(tf.square(tensor), -2, keep_dims=True)
        squash_factor = (tensor_norm/(1+tensor_norm))/tf.sqrt(tensor_norm + self.config.epsilon)
        tensor_squash = tensor*squash_factor
        return tensor_squash
    
    def routing(self, input_x, b_ij):
        # sharing weights
        input_shape = input_x.get_shape()
        weights = tf.get_variable('weights', shape=(1, input_shape[1], self.output_num, self.caps_len, self.output_len), dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.02))
        # input_x shape=[batchsize, input_caps_num, self.output_num, self.caps_len, 1]
        # weights shape = [batchsize, input_caps_num, output_num, self.caps_len, self.output_len]
        input_x = tf.tile(input_x, [1, 1, self.output_num, 1, 1])
        weights = tf.tile(weights, [self.config.batchsize, 1, 1, 1, 1])
        
        # only working on the last 2 dims
        u_hat = tf.matmul(weights, input_x, transpose_a = True)
        assert u_hat.get_shape() == [self.config.batchsize, input_shape[1], self.output_num, self.output_len, 1]
        
        for r_ite in range(self.config.r_ite):
            with tf.variable_scope('iteration_' + str(r_ite)):
                c_ij = tf.nn.softmax(b_ij, dim=2)#???????
                c_ij = tf.tile(c_ij, [self.config.batchsize,1,1,1,1])
                s_j = tf.multiply(c_ij, u_hat)
                
                s_j = tf.reduce_sum(s_j, axis=1, keep_dims=True)
                assert s_j.get_shape() == [self.config.batchsize, 1, self.output_num, self.output_len, 1]
                # squashing
                v_j = self.squashing(s_j)
                assert v_j.get_shape() == [self.config.batchsize, 1, self.output_num, self.output_len, 1]
                
                v_j_tile = tf.tile(v_j, [1,input_shape[1],1,1,1])
                u_v_product = tf.matmul(u_hat, v_j_tile, transpose_a=True)
                b_ij = b_ij + tf.reduce_sum(u_v_product, axis=0, keep_dims=True)
        return(v_j)
