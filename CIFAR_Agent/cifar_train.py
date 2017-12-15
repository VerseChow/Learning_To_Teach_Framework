import tensorflow as tf
from model import *
import shutil
import argparse

if not os.path.exists('./teacherlog'):
  os.makedirs('teacherlog')

if not os.path.exists('./pretrained_weight_for_teacher'):
  os.makedirs('./pretrained_weight_for_teacher')

maybe_download_and_extract()

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='CIFAR model')
  parser.add_argument('--teacher_flg', dest='teacher_flg',
                      type=str2bool, default=False,
                      help='whether to use learning to teach framework')

  args = parser.parse_args()
  x = tf.placeholder(tf.float32, [None, 784])
  y = tf.placeholder(tf.float32, [None, 10])
  feature_state = tf.placeholder(tf.float32, [None, 25])
  # Initialize the Train object
  train_agent = CIFAR_Model(teacher_training = args.teacher_flg)
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

