import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from model import *
from RL_model import *

num_iteration = 20000
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# train_agent = MNIST_Model(num_iterations = num_iteration)

train_agent  = L2T_model()

train_agent.MNIST_train_run(x, y, mnist.train.next_batch(50))