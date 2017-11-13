import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from model import *

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

train_agent = MNIST_Model()

train_agent.train(mnist, x, y)