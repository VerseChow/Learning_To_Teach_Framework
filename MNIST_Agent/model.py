import tensorflow as tf
import numpy as np
import os

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

        return tf.nn.softmax(fc2, dim=-1), loss, prob

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


    def train_one_step_setup(self, x, y, sess):

        logits, loss, self.prob = self.build_model(x, y)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(logits, -1), tf.argmax(y, -1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)

        self.accuracy = tf.reduce_mean(correct_prediction)
        vars_trainable = tf.trainable_variables()


        with tf.name_scope('adam_optimizer'):
            self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss, var_list=vars_trainable)

        # with (sess or tf.get_default_session()):
        sess.run(tf.global_variables_initializer())


    def train_one_step(self, batch, x, y):
        train_accuracy = self.accuracy.eval(feed_dict={
            x: batch[0], y: batch[1], self.prob: 1.0})
        print(' training accuracy %g' % (train_accuracy))
        self.train_step.run(
            feed_dict={x: batch[0], y: batch[1], self.learning_rate: self.init_learning_rate, self.prob: 0.5})