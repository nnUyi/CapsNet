import tensorflow as tf
import numpy as np
import os
from utils import *
from ops import *
from capsLayer import *

class CapsNet:
    def __init__(self, config, input_height=28, input_width=28, input_channel=1, output_num=10, output_len=16, caps_len=8, model_name='mnist',dataset_name='mnist', sess=None):
        self.config = config
        self.input_height = input_height
        self.input_width = input_width
        self.input_channel = input_channel
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.output_num = output_num
        self.output_len = output_len
        self.caps_len = caps_len
        self.batchsize = config.batchsize
        self.sess = sess

    def build_model(self):
        self.input_x = tf.placeholder(tf.float32, shape=(self.batchsize, self.input_height, self.input_width, 1))
        self.label_y = tf.placeholder(tf.int32, shape=(self.batchsize))
        self.input_y = tf.one_hot(self.label_y, depth=10, axis=1, dtype=tf.float32)
        
        with tf.variable_scope('convLayer'):
            kernel_size = [9,9,self.input_channel, 256]
            self.conv1 = tf.nn.relu(conv2d(self.input_x, kernel_size, stride=[1,1,1,1], conv_type='VALID'))
            print(self.conv1)
            
        with tf.variable_scope('primaryCaps'):
            primaryCaps = CapsLayer(layer_type='conv', config=self.config, output_num = self.output_num, output_len=self.output_len, caps_len=self.caps_len)
            self.caps1 = primaryCaps.capsLayer(self.conv1, kernel_size=[9,9,256,32], stride=[1,2,2,1], conv_type='VALID')
            #layer_type, config, output_num = 10, output_len=16, caps_len=8
            print(self.caps1)
            
        with tf.variable_scope('digitCaps'):
            digitCaps = CapsLayer(layer_type='fc', config=self.config, output_num = self.output_num, output_len=self.output_len, caps_len=self.caps_len)
            self.caps2 = digitCaps.capsLayer(self.caps1, kernel_size=[])
            print(self.caps2)
        
        with tf.variable_scope('masking'):
            if self.config.mask_with_y:
                self.mask_v = tf.matmul(tf.squeeze(self.caps2), tf.reshape(self.input_y, shape=[-1, self.output_num, 1]), transpose_a=True)
                self.v_len = tf.sqrt(tf.reduce_sum(tf.square(self.caps2), axis=2, keep_dims=True)+self.config.epsilon)
            else:
                self.v_len = tf.sqrt(tf.reduce_sum(tf.square(self.caps2), axis=2, keep_dims=True)+self.config.epsilon)
                print('v_len',self.v_len)
                self.softmax_v = tf.nn.softmax(self.v_len, dim=1)
                print('softmax_v',self.softmax_v)
                argmax_idx = tf.to_int32(tf.argmax(self.softmax_v, axis=1))
                argmax_idx = tf.reshape(argmax_idx, [self.batchsize, -1])
                
                self.argmax_idx = argmax_idx
                
                mask_v = []
                
                for index in range(self.batchsize):
                    v = self.caps2[index][argmax_idx[index][0]]
                    mask_v.append(v)
                self.mask_v = tf.concat(mask_v, axis=0)
                
        with tf.variable_scope('decoder'):
            v = tf.reshape(self.mask_v, [self.batchsize, -1])
            fc1_output_size = 512
            fc1 = tf.nn.relu(linear(v, fc1_output_size, scope_name='fc1'))
            fc2_output_size = 1024
            fc2 = tf.nn.relu(linear(fc1, fc2_output_size, scope_name='fc2'))
            fc3_output_size = self.input_height*self.input_width*self.input_channel
            if self.input_channel == 1:
                fc3 = tf.nn.sigmoid(linear(fc2, fc3_output_size, scope_name='fc3'))
            else:
                fc3 = tf.nn.tanh(linear(fc2, fc3_output_size, scope_name='fc3'))
            self.decoder = fc3
            
        
        # loss
        Tc = self.input_y
        max_positive = tf.square(tf.maximum(0.0, self.config.m_plus-self.v_len))
        max_negative = tf.square(tf.maximum(0.0, self.v_len-self.config.m_minus))
        
        max_positive = tf.reshape(max_positive, shape=[self.batchsize, -1])
        max_negative = tf.reshape(max_negative, shape=[self.batchsize, -1])
        
        Lc = Tc*max_positive + self.config.lamb*(1-Tc)*max_negative
                
        input_x = self.input_x
        
        decoder_x = self.decoder
        decoder_x = tf.reshape(decoder_x, [self.config.batchsize, self.input_height, self.input_width, self.input_channel])
        self.sample = decoder_x
        
        self.margin_loss = tf.reduce_mean(Lc)
        self.decoder_loss = tf.reduce_mean(tf.square(input_x-decoder_x))
        self.total_loss = self.margin_loss + self.config.gamma*self.decoder_loss
        
        self.optimization = tf.train.AdamOptimizer().minimize(self.total_loss)
        self.saver = tf.train.Saver()
        
    def train(self):
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()
        batch_num = int(60000/self.batchsize)
        input_x, label_y = load_mnist(self.config)
        print(label_y.shape)
        check_bool, counter = self.load_model(self.config.checkpoint_dir)
        if check_bool:
            counter = counter + 1
            print("[***] load model successfully")
        else:
            counter = 1
            print("[***] fail to load model")
        for ite in range(self.config.iteration):
            for step in range(batch_num):
                batch_x = input_x[step*self.batchsize:(step+1)*self.batchsize]
                batch_y = label_y[step*self.batchsize:(step+1)*self.batchsize]
                _, total_loss, samples = self.sess.run([self.optimization,self.total_loss, self.sample], feed_dict={self.input_x: batch_x,
                                                            self.label_y: batch_y})
                                                            
                print("epoch %d:[%d/%d]" %(ite,step,batch_num),'total_loss: ', total_loss)
                counter = counter+1
                if np.mod(step, 100)==0:
                    save_images(samples, [8, 8], './{}/train_{:02d}_{:04d}.png'.format(self.config.sample, ite, step))
                    pass
                if np.mod(counter, 500)==0:
                    self.save_model(self.config.checkpoint_dir, counter)
    
    def test(self):
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()
        batch_num = int(10000/self.batchsize)
        input_x, label_y = load_mnist(self.config)
        print(label_y.shape)
        check_bool, counter = self.load_model(self.config.checkpoint_dir)
        if check_bool:
            print("[***] load model successfully")
        else:
            print("[***] fail to load model")
        
        accuracy = 0
        for step in range(batch_num):
            batch_x = input_x[step*self.batchsize:(step+1)*self.batchsize]
            batch_y = label_y[step*self.batchsize:(step+1)*self.batchsize]
            
            np_batch_y = np.array(batch_y).astype(np.float).reshape((self.batchsize,1))

            argmax_idx = self.sess.run(self.argmax_idx, feed_dict={self.input_x: batch_x})
            batch_acc = np.sum(np.array(np.equal(np_batch_y, argmax_idx)).astype(np.float))
            
            accuracy = accuracy + batch_acc
            print("batch:[%d]" %(step), "batch_acc", batch_acc)
        print("accuracy:", accuracy/(self.batchsize*(batch_num-1)))
        
    @property
    def model_dir(self):
        return "{}_{}_{}".format(
            self.model_name, self.dataset_name,
            self.batchsize)

    def save_model(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

    def load_model(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
