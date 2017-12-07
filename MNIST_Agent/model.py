import __init__

from scipy.stats import rankdata
import tensorflow as tf
import numpy as np
import os
import sys
import Teacher_Agent.model as t

class MNIST_Model():

    def __init__(self,
                 training=True,
                 batch_size=50,
                 init_learning_rate=1e-4,
                 num_iterations = 20000,
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
        self.start_train_num = 100
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
        vars_trainable = tf.trainable_variables()

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

    def feature_state(self, label, label_pred, logits, loss, iter_index):
        self.average_loss = self.average_loss+(loss-self.average_loss)/(1+iter_index)
        self.best_loss = np.minimum(np.amin(logits), self.best_loss)
        margin_value = np.array([])
        # Calculate margin value
        for i in range(self.batch_size):
            l = label[i]
            l_p = label_pred[i]
            indx = np.argmax(l)
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
        for i in range(self.batch_size):
            # Date features
            feature[i, 0:10] = label[i]
            # Model fearures
            feature[i, 10] = iter_index
            feature[i, 11] = self.average_loss
            feature[i, 12] = self.best_loss
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
        # Build Teacher Agent
        self.teacher = t.TeacherAgent()
        self.action_space, self.action, self.action_prob = self.teacher.build_model(feature_state)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(self.label_pred, -1), tf.argmax(y, -1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)

        self.accuracy = tf.reduce_mean(correct_prediction)
        vars_trainable = tf.trainable_variables()

        with tf.name_scope('adam_optimizer'):
            self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, var_list=vars_trainable)

        sess.run(tf.global_variables_initializer())


    def train_one_step(self, batch, x, y, feature_state, sess, iteration_index):
        new_batch_data = [];
        new_batch_label = [];
        [label_pred, logits, loss, train_accuracy] = sess.run([self.label_pred, self.logits, self.loss, self.accuracy],
            feed_dict={x: batch[0], y: batch[1], self.prob: 1.0})
        print(' training accuracy %g' % (train_accuracy))
        # feed to teacher agent        
        features = self.feature_state(batch[1], label_pred, logits, loss, iteration_index)
        action_space, action = self.teacher.estimate(sess,self.action_space, self.action,self.action_prob,features,feature_state)
        # action_space, action = sess.run([self.action_space, self.action], feed_dict={feature_state: features, self.action_prob: 1.0})

        for i,j in enumerate(action):
            if j == 1:
                new_batch_data.append(batch[0][i])
                new_batch_label.append(batch[1][i])

        print(len(new_batch_data))


        if len(new_batch_data) != 0 and self.start_train_num <=0:
            self.train_step.run(
                feed_dict={x: new_batch_data, y: new_batch_label, self.learning_rate: self.init_learning_rate,
                self.prob: 0.5})
        elif self.start_train_num >0:
            self.train_step.run(
                feed_dict={x: batch[0], y: batch[1], self.learning_rate: self.init_learning_rate,
                           self.prob: 0.5})
            self.start_train_num = self.start_train_num - 1