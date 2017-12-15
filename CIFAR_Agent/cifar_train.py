import tensorflow as tf
from model import *
import shutil

if not os.path.exists('./teacherlog'):
    os.makedirs('teacherlog')

maybe_download_and_extract()

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
feature_state = tf.placeholder(tf.float32, [None, 25])
# Initialize the Train object
train_agent = CIFAR_Model(teacher_training = False)
# Start the training session
#train_agent.train()
print("this is cifar")

with tf.Session() as sess:
    train_agent.train_one_step_setup(x, y, feature_state, sess)
    shutil.rmtree('./teacherlog')
    writer_teacher = tf.summary.FileWriter('./teacherlog', sess.graph)
    with open('accuracy.txt', 'wb') as txtWriter:
        for i in range(1000000):
            accuracy = train_agent.train_one_step(x, y, feature_state, sess,
                                        txtWriter, writer_teacher)
            if accuracy > 0.9 and train_agent.teacher_training == False:
              break

