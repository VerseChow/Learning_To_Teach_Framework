import __init__

from scipy.stats import rankdata
import tensorflow as tf
import numpy as np
import os
import sys
import Teacher_Agent.model as t
import math
from tensorflow.examples.tutorials.mnist import input_data
class MNIST_Model():

    def __init__(self,
                 training=True,
                 batch_size=20,
                 init_learning_rate=1e-4,
                 num_iterations = 20000,
                 discount_factor = 1,
                 reuse=None):
        self.training = training
        self.reuse = reuse
        self.batch_size = batch_size
        self.init_learning_rate = init_learning_rate
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name='lr')
        self.pretrained_weight_path = './pretrained_weight'
        self.test_weight = './test_weight'
        self.num_iterations = num_iterations
        self.average_loss = 0.0
        self.best_loss = 100.0
        self.student_trajectory = []
        self.reward = []
        self.T_max = 10000.0
        self.discount_factor = discount_factor
        self.new_batch_data = [];
        self.new_batch_label = [];
        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        self.mnist_l = int(self.mnist.train.images.shape[0])
        self.mnist_indexes = list(range(self.mnist_l))
        np.random.shuffle(self.mnist_indexes)
        self.half_l = int(self.mnist.train.images.shape[0] / 2)
        self.train_teach_indexes = self.mnist_indexes[:self.half_l]
        self.train_student_indexes = self.mnist_indexes[self.half_l:]
        '''
        #this shows how to read minist data
        self.mnist_train_teach_img = mnist.train.images[:half_l,:]
        self.mnist_train_teach_lbl = mnist.train.labels[:half_l,:]
        self.mnist_train_stud_img = mnist.train.images[half_l:, :]
        self.mnist_train_stud_lbl = mnist.train.labels[half_l:, :]
        '''
        self.D_dev_l = int(self.half_l * 0.05)
        temp_train_teach_indexes = self.train_teach_indexes[:]  # deep copy
        np.random.shuffle(temp_train_teach_indexes)
        self.D_dev_indexes = temp_train_teach_indexes[:self.D_dev_l]
        self.D_dev_img = self.mnist.train.images[self.D_dev_indexes,:]
        self.D_dev_lbl = self.mnist.train.labels[self.D_dev_indexes]

        self.iter_index = 0

    def chkpoint_restore(self, sess):
        saver = tf.train.Saver(max_to_keep=2)
        if self.training:
            ckpt = tf.train.get_checkpoint_state(self.pretrained_weight_path)
        else:
            ckpt = tf.train.get_checkpoint_state(self.test_weight)

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            if self.training:
                saver.restore(sess, os.path.join(self.pretrained_weight_path, ckpt_name))
            else:
                saver.restore(sess, os.path.join(self.test_weight, ckpt_name))
            print('[*] Success to read {}'.format(ckpt_name))
        else:
            if self.training:
                print('[*] Failed to find a checkpoint. Start training from scratch ...')
            else:
                raise ValueError('[*] Failed to find a checkpoint.')
        return saver

    def conv_pool(self, x, num_filters, ksize=5, name='conv_pool'):
        with tf.variable_scope(name):
            x = tf.layers.conv2d(x, num_filters, ksize, 1,
                                 padding='same', use_bias=True, reuse=self.reuse,
                                 name='conv2d', trainable=self.training)
            x = tf.layers.max_pooling2d(x, 2, 2, name='pool')
            return tf.nn.relu(x, name='relu')

    def fc_relu(self, x, num, num_filters, name='fc_relu'):
        with tf.variable_scope(name):
            w_fc = tf.truncated_normal(shape=[num, num_filters], stddev=0.1)
            b_fc = tf.constant(0.1, shape=[num_filters])
            w_fc = tf.Variable(w_fc)
            b_fc = tf.Variable(b_fc)
        return tf.nn.relu(tf.matmul(x, w_fc)+b_fc, name='relu')

    def dropout(self, x, name='dropout'):
        with tf.variable_scope(name):
            prob = tf.placeholder(tf.float32)
            x = tf.nn.dropout(x, prob)
        return x, prob

    def build_model(self, x, y):
        with tf.variable_scope('student_model'):

            with tf.name_scope('reshape'):
                x = tf.reshape(x, [-1, 28, 28, 1])

            cov1 = self.conv_pool(x, 32, name='conv_pool1')

            cov2 = self.conv_pool(cov1, 64, name='conv_pool2')

            with tf.name_scope('flatten'):
                flatten = tf.layers.flatten(cov2)

            fc1 = self.fc_relu(flatten, 7*7*64, 1024, name='fc_relu1')

            dpout, prob = self.dropout(fc1, name='dropout')

            fc2 = self.fc_relu(dpout, 1024, 10, name='fc_relu1')

            logits = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=fc2)

            loss = tf.reduce_mean(logits)

        label_pred = tf.nn.softmax(fc2, dim=-1)

        return label_pred, logits, loss, prob

    def train(self, data, x, y):
        # build model
        logits, loss, prob = self.build_model(x, y)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(logits, -1), tf.argmax(y, -1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)

        accuracy = tf.reduce_mean(correct_prediction)

        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('learning_rate', self.learning_rate)

        num_param = 0
        vars_trainable = tf.trainable_variables(scope = 'student_model')

        for var in vars_trainable:
            num_param += np.prod(var.get_shape()).value
            tf.summary.histogram(var.name, var)
        sum_all = tf.summary.merge_all()
        # set up trainner
        with tf.name_scope('adam_optimizer'):
            train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss, var_list=vars_trainable)

        print('\nTotal nummber of parameters = %d' % num_param)

        lr = self.init_learning_rate

        # training
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter("./logs", sess.graph)
            saver = self.chkpoint_restore(sess)

            for i in range(self.num_iterations):
                batch = data.train.next_batch(self.batch_size)

                if i % 100 == 0:
                    train_accuracy = accuracy.eval(feed_dict={
                        x: batch[0], y: batch[1], prob: 1.0})
                    print('step %d, training accuracy %g' % (i, train_accuracy))
                if i % 1000 == 0:
                    print('Saving checkpoint ...')
                    saver.save(sess, self.test_weight + '/MNIST.ckpt')

                train_step.run(feed_dict={x: batch[0], y: batch[1], self.learning_rate: lr, prob: 0.5})
                writer.add_summary(sess.run(sum_all, feed_dict={x: batch[0], y: batch[1], self.learning_rate: lr, prob: 0.5}), i)

    def feature_state(self, label, label_pred, logits, loss):
        self.average_loss = self.average_loss+(loss-self.average_loss)/(1.0+float(self.iter_index))
        print(self.average_loss)
        self.iter_index += 1
        #self.best_loss = np.minimum(np.amin(logits), self.best_loss)
        print(self.best_loss)
        margin_value = np.array([])
        # Calculate margin value
        for i in range(self.batch_size):
            l = label[i]
            l_p = label_pred[i]
            indx = np.argmax(l_p)
            P = l_p[indx]
            l_p[indx] = 0.0
            P = P-l_p[np.argmax(l_p)]
            margin_value = np.append(margin_value, P)
        # Rank the data
        margin_rank = rankdata(margin_value, 'min')
        loss_rank = rankdata(logits, 'min')
        # predefine feature state
        feature = np.zeros([self.batch_size, 25])
        # constuct feature state
        indx = float((self.iter_index-1)*self.batch_size)
        for i in range(self.batch_size):
            # Date features
            feature[i, 0:10] = label[i]
            # Model fearures
            feature[i, 10] = float(indx+i)/self.T_max
            feature[i, 11] = self.average_loss/self.T_max
            feature[i, 12] = self.best_loss/self.T_max
            # Combined features
            feature[i, 13:23] = label_pred[i]
            # normalized rank
            m_r = float(margin_rank[i]-1)/(self.batch_size-1)
            l_r = float(loss_rank[i]-1)/(self.batch_size-1)
            feature[i, 23] = l_r
            feature[i, 24] = m_r
        feature.astype(float)
        return feature

    def train_one_step_setup(self, x, y, feature_state, sess):

        self.label_pred, self.logits, self.loss, self.prob = self.build_model(x, y)
        self.vars_trainable = tf.trainable_variables(scope='student_model')
        # Build Teacher Agent
        self.teacher = t.TeacherAgent()
        action_prob, prob = self.teacher.build_model(feature_state)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(self.label_pred, -1), tf.argmax(y, -1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)

        self.accuracy = tf.reduce_mean(correct_prediction)

        with tf.name_scope('adam_optimizer'):
            self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, var_list=self.vars_trainable)

        sess.run(tf.global_variables_initializer())


    def train_one_step(self, batch, x, y, feature_state, sess, txtWriter, writer_teacher):
        # get one batch from train_teach
        temp_train_teach_indexes = self.train_teach_indexes[:]
        np.random.shuffle(temp_train_teach_indexes)
        batch_indexes = temp_train_teach_indexes[:self.batch_size]
        batch_img = self.mnist.train.images[batch_indexes, :]
        batch_lbl = self.mnist.train.labels[batch_indexes, :]
        [label_pred, logits, loss] = sess.run([self.label_pred, self.logits, self.loss],
                                    feed_dict={x: batch_img, y: batch_lbl, self.prob: 1.0})
        [loss_val, train_accuracy] = sess.run([self.loss, self.accuracy],
                                    feed_dict={x: self.D_dev_img, y: self.D_dev_lbl, self.prob: 1.0})
        self.best_loss = np.minimum(loss_val, self.best_loss)
        # feed to teacher agent
        features = self.feature_state(batch_lbl, label_pred, logits, loss)
        action_prob = self.teacher.estimate(sess, features, feature_state)
        # construct new batch of data based on the action_prob
        for i, prob in enumerate(action_prob[0]):
            # sample action
            action = np.random.choice(2, 1, p=[1.0-prob[0], prob[0]])
            if action[0] == 1 and len(self.new_batch_data) != self.batch_size:
                self.new_batch_data.append(batch_img[i,:])
                self.new_batch_label.append(batch_lbl[i,:])
                # append reward and features
                self.student_trajectory.append(features[i])
                self.reward.append(0.0)
        # terminate trajectory episode and calculate rewards
        print(' training accuracy %g' % (train_accuracy))
        if train_accuracy >= 0.9:
            print(' length of reward %g' % len(self.reward))
            if len(self.reward) > 0:
                self.reward[-1] = -math.log(float(len(self.reward))/self.T_max)
                reward = self.reward[-1]
                trajectory = np.asarray(self.student_trajectory, dtype=np.float32)
                #for traj in trajectory:
                #    traj = traj.reshape((-1, 25))
                self.teacher.update(sess, reward, trajectory, feature_state, writer_teacher)
                txtWriter.write(bytes(str(len(self.reward))+'\n', 'UTF-8'))
            re_initialize_para = tf.variables_initializer(self.vars_trainable)
            # reset
            sess.run(re_initialize_para)
            self.student_trajectory.clear()
            self.reward.clear()
            self.iter_index = 0
            self.average_loss = 0.0
            self.best_loss = 100.0

        if len(self.new_batch_data) == self.batch_size:
            self.train_step.run(
                feed_dict={x: self.new_batch_data, y: self.new_batch_label, self.learning_rate: self.init_learning_rate,
                self.prob: 0.5})
            self.new_batch_data = []
            self.new_batch_label = []

