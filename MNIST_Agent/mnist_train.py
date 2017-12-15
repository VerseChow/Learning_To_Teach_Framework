import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from model import *
import shutil

if not os.path.exists('./teacherlog'):
    os.makedirs('teacherlog')
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
feature_state = tf.placeholder(tf.float32, [None, 25])
train_agent = MNIST_Model()

with tf.Session() as sess:
    train_agent.train_one_step_setup(x, y, feature_state, sess)
    # clear log
    shutil.rmtree('./teacherlog')
    writer_teacher = tf.summary.FileWriter('./teacherlog', sess.graph)
    with open('reward_count.txt', 'wb') as txtWriter:
      for i in range(2000000):
          train_agent.train_one_step(mnist.train.next_batch(25), x, y, feature_state, sess,
                                    txtWriter, writer_teacher)
