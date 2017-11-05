import tensorflow as tf
import os

from capsNet import *

flags = tf.app.flags
flags.DEFINE_integer("batchsize", 64, "batchsize for training")
flags.DEFINE_integer("r_ite", 3, "iteration for routing")
flags.DEFINE_integer("num_threads", 4, "num threads of shuffling")
flags.DEFINE_integer("iteration", 20, "num of epoch")
flags.DEFINE_float("epsilon", 1e-9, "epsilon to avoid zero under dividing")
flags.DEFINE_float("m_minus", 0.1, "m minus")
flags.DEFINE_float("m_plus", 0.9, "m plus")
flags.DEFINE_float("gamma", 0.0005, "gamma for decoder_loss")
flags.DEFINE_float("lamb", 0.5, "lambda for margin_loss")
flags.DEFINE_float("learning_rate", 0.0002, "learning rate")
flags.DEFINE_string("dataset", "./data/mnist", "dataset directory")
flags.DEFINE_bool("is_training", False, "training or testing")
flags.DEFINE_string("checkpoint_dir", "./checkpoint", "checkpoint directory")
flags.DEFINE_string("sample", "./sample", "checkpoint directory")
flags.DEFINE_bool("mask_with_y", False, "mask with y label")
FLAGS = flags.FLAGS

def check_dir():
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.mkdir(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample):
        os.mkdir(FLAGS.sample)

def main(_):
    check_dir()        
    with tf.Session() as sess:
        Caps = CapsNet(config=FLAGS, sess=sess)
        Caps.build_model()
        if FLAGS.is_training:
            Caps.train()
        else:
            Caps.test()

if __name__=='__main__':
    tf.app.run()
