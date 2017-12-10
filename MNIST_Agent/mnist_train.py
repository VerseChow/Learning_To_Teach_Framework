import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from model import *

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
feature_state = tf.placeholder(tf.float32, [None, 25])
train_agent = MNIST_Model()

# train_agent.train(mnist, x, y)

with tf.Session() as sess:
    train_agent.train_one_step_setup(x, y, feature_state, sess)
    for i in range(140000):
        train_agent.train_one_step(mnist.train.next_batch(50), x, y, feature_state, sess, i)
