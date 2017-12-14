import __init__

from scipy.stats import rankdata
import tensorflow as tf
import numpy as np
import os
import sys
import Teacher_Agent.model as t
from resnet import *
from datetime import datetime
import time
from cifar10_input import *
import pandas as pd
import math
class CIFAR_Model():
    def __init__(self):
        # Set up all the placeholders
        self.batch_size = FLAGS.train_batch_size
        self.iter_index = 0
        self.init_learning_rate = 0.001
        self.average_loss = 0.0
        self.best_loss = 100.0
        self.student_trajectory = []
        self.reward = []
        self.T_max = 60000.0
        self.discount_factor = 0
        self.new_batch_data = [];
        self.new_batch_label = [];
        self.latest_reward = 0
        self.latest_episode_length = 0
        self.train_tao = 0.4
        self.all_data, self.all_labels = prepare_train_data(padding_size=FLAGS.padding_size)
        self.cifar_l = EPOCH_SIZE
        self.cifar_indexes = list(range(self.cifar_l))
        np.random.shuffle(self.cifar_indexes)
        self.half_l = int(EPOCH_SIZE / 2)
        self.train_teach_indexes = self.cifar_indexes[0:self.half_l]
        self.train_student_indexes = self.cifar_indexes[self.half_l:]
        self.D_dev_l = 300#int(self.half_l * 0.05)
        print("Dev Num",self.D_dev_l)
        temp_train_teach_indexes = self.train_teach_indexes[:]  # deep copy
        np.random.shuffle(temp_train_teach_indexes)
        self.D_dev_indexes = temp_train_teach_indexes[0:self.D_dev_l]
        self.D_dev_img = self.all_data[self.D_dev_indexes, 1:33,1:33,:]
        self.D_dev_lbl = self.all_labels[self.D_dev_indexes]
        self.train_teach_data = self.all_data[self.train_teach_indexes,...]
        self.train_teach_label = self.all_labels[self.train_teach_indexes,...]
        self.placeholders()
    def placeholders(self):
        '''
        There are five placeholders in total.
        image_placeholder and label_placeholder are for train images and labels
        vali_image_placeholder and vali_label_placeholder are for validation imgaes and labels
        lr_placeholder is for learning rate. Feed in learning rate each time of training
        implements learning rate decay easily
        '''
        self.image_placeholder = tf.placeholder(dtype=tf.float32,
                                                shape=[FLAGS.train_batch_size, IMG_HEIGHT,
                                                       IMG_WIDTH, IMG_DEPTH])
        self.label_placeholder = tf.placeholder(dtype=tf.int32, shape=[FLAGS.train_batch_size])

        self.vali_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[self.D_dev_l,
                                                                              IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
        self.vali_label_placeholder = tf.placeholder(dtype=tf.int32, shape=[self.D_dev_l])

        self.lr_placeholder = tf.placeholder(dtype=tf.float32, shape=[])
        self.feature_state = tf.placeholder(tf.float32, [None, 25])

    def build_train_validation_graph(self):
        '''
        This function builds the train graph and validation graph at the same time.

        '''
        global_step = tf.Variable(0, trainable=False)
        validation_step = tf.Variable(0, trainable=False)

        # Logits of training data and valiation data come from the same graph. The inference of
        # validation data share all the weights with train data. This is implemented by passing
        # reuse=True to the variable scopes of train graph
        logits = inference(self.image_placeholder, FLAGS.num_residual_blocks, reuse=False)
        vali_logits = inference(self.vali_image_placeholder, FLAGS.num_residual_blocks, reuse=True)

        # The following codes calculate the train loss, which is consist of the
        # softmax cross entropy and the relularization loss
        regu_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = self.loss(logits, self.label_placeholder)
        self.full_loss = tf.add_n([loss] + regu_losses)

        predictions = tf.nn.softmax(logits)
        self.train_top1_error = self.top_k_error(predictions, self.label_placeholder, 1)

        # Validation loss
        self.vali_loss = self.loss(vali_logits, self.vali_label_placeholder)
        vali_predictions = tf.nn.softmax(vali_logits)
        self.vali_top1_error = self.top_k_error(vali_predictions, self.vali_label_placeholder, 1)

        self.train_op, self.train_ema_op = self.train_operation(global_step, self.full_loss,
                                                                self.train_top1_error)
        self.val_op = self.validation_op(validation_step, self.vali_top1_error, self.vali_loss)


    def train(self):
        '''
        This is the main function for training
        '''

        # For the first step, we are loading all training images and validation images into the
        # memory
        all_data, all_labels = prepare_train_data(padding_size=FLAGS.padding_size)
        vali_data, vali_labels = read_validation_data()

        # Build the graph for train and validation
        self.build_train_validation_graph()

        # Initialize a saver to save checkpoints. Merge all summaries, so we can run all
        # summarizing operations by running summary_op. Initialize a new session
        saver = tf.train.Saver(tf.global_variables())
        summary_op = tf.summary.merge_all()
        init = tf.initialize_all_variables()
        sess = tf.Session()

        # If you want to load from a checkpoint
        if FLAGS.is_use_ckpt is True:
            saver.restore(sess, FLAGS.ckpt_path)
            print('Restored from checkpoint...')
        else:
            sess.run(init)

        # This summary writer object helps write summaries on tensorboard
        summary_writer = tf.summary.FileWriter(train_dir, sess.graph)

        # These lists are used to save a csv file at last
        step_list = []
        train_error_list = []
        val_error_list = []

        print('Start training...')
        print('----------------------------')

        for step in range(FLAGS.train_steps):

            train_batch_data, train_batch_labels = self.generate_augment_train_batch(all_data, all_labels,
                                                                                     FLAGS.train_batch_size)

            validation_batch_data, validation_batch_labels = self.generate_vali_batch(vali_data,
                                                                                      vali_labels,
                                                                                      FLAGS.validation_batch_size)

            # Want to validate once before training. You may check the theoretical validation
            # loss first
            if step % FLAGS.report_freq == 0:

                if FLAGS.is_full_validation is True:
                    validation_loss_value, validation_error_value = self.full_validation(loss=self.vali_loss,
                                                                                         top1_error=self.vali_top1_error,
                                                                                         vali_data=vali_data,
                                                                                         vali_labels=vali_labels,
                                                                                         session=sess,
                                                                                         batch_data=train_batch_data,
                                                                                         batch_label=train_batch_labels)

                    vali_summ = tf.Summary()
                    vali_summ.value.add(tag='full_validation_error',
                                        simple_value=validation_error_value.astype(np.float))
                    summary_writer.add_summary(vali_summ, step)
                    summary_writer.flush()

                else:
                    _, validation_error_value, validation_loss_value = sess.run([self.val_op,
                                                                                 self.vali_top1_error,
                                                                                 self.vali_loss],
                                                                                {
                                                                                    self.image_placeholder: train_batch_data,
                                                                                    self.label_placeholder: train_batch_labels,
                                                                                    self.vali_image_placeholder: validation_batch_data,
                                                                                    self.vali_label_placeholder: validation_batch_labels,
                                                                                    self.lr_placeholder: FLAGS.init_lr})

                val_error_list.append(validation_error_value)

            start_time = time.time()

            _, _, train_loss_value, train_error_value = sess.run([self.train_op, self.train_ema_op,
                                                                  self.full_loss, self.train_top1_error],
                                                                 {self.image_placeholder: train_batch_data,
                                                                  self.label_placeholder: train_batch_labels,
                                                                  self.vali_image_placeholder: validation_batch_data,
                                                                  self.vali_label_placeholder: validation_batch_labels,
                                                                  self.lr_placeholder: FLAGS.init_lr})
            duration = time.time() - start_time

            if step % FLAGS.report_freq == 0:
                summary_str = sess.run(summary_op, {self.image_placeholder: train_batch_data,
                                                    self.label_placeholder: train_batch_labels,
                                                    self.vali_image_placeholder: validation_batch_data,
                                                    self.vali_label_placeholder: validation_batch_labels,
                                                    self.lr_placeholder: FLAGS.init_lr})
                summary_writer.add_summary(summary_str, step)

                num_examples_per_step = FLAGS.train_batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.4f (%.1f examples/sec; %.3f ' 'sec/batch)')
                print(format_str % (datetime.now(), step, train_loss_value, examples_per_sec,
                                    sec_per_batch))
                print('Train top1 error = ', train_error_value)
                print('Validation top1 error = %.4f' % validation_error_value)
                print('Validation loss = ', validation_loss_value)
                print('----------------------------')

                step_list.append(step)
                train_error_list.append(train_error_value)

            if step == FLAGS.decay_step0 or step == FLAGS.decay_step1:
                FLAGS.init_lr = 0.1 * FLAGS.init_lr
                print('Learning rate decayed to ', FLAGS.init_lr)

            # Save checkpoints every 10000 steps
            if step % 10000 == 0 or (step + 1) == FLAGS.train_steps:
                checkpoint_path = os.path.join(train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

                df = pd.DataFrame(data={'step': step_list, 'train_error': train_error_list,
                                        'validation_error': val_error_list})
                df.to_csv(train_dir + FLAGS.version + '_error.csv')

    def test(self, test_image_array):
        '''
        This function is used to evaluate the test data. Please finish pre-precessing in advance

        :param test_image_array: 4D numpy array with shape [num_test_images, img_height, img_width,
        img_depth]
        :return: the softmax probability with shape [num_test_images, num_labels]
        '''
        num_test_images = len(test_image_array)
        num_batches = num_test_images // FLAGS.test_batch_size
        remain_images = num_test_images % FLAGS.test_batch_size
        print('%i test batches in total...' % num_batches)

        # Create the test image and labels placeholders
        self.test_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[FLAGS.test_batch_size,
                                                                              IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])

        # Build the test graph
        logits = inference(self.test_image_placeholder, FLAGS.num_residual_blocks, reuse=False)
        predictions = tf.nn.softmax(logits)

        # Initialize a new session and restore a checkpoint
        saver = tf.train.Saver(tf.all_variables())
        sess = tf.Session()

        saver.restore(sess, FLAGS.test_ckpt_path)
        print('Model restored from ', FLAGS.test_ckpt_path)

        prediction_array = np.array([]).reshape(-1, NUM_CLASS)
        # Test by batches
        for step in range(num_batches):
            if step % 10 == 0:
                print('%i batches finished!' % step)
            offset = step * FLAGS.test_batch_size
            test_image_batch = test_image_array[offset:offset + FLAGS.test_batch_size, ...]

            batch_prediction_array = sess.run(predictions,
                                              feed_dict={self.test_image_placeholder: test_image_batch})

            prediction_array = np.concatenate((prediction_array, batch_prediction_array))

        # If test_batch_size is not a divisor of num_test_images
        if remain_images != 0:
            self.test_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[remain_images,
                                                                                  IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
            # Build the test graph
            logits = inference(self.test_image_placeholder, FLAGS.num_residual_blocks, reuse=True)
            predictions = tf.nn.softmax(logits)

            test_image_batch = test_image_array[-remain_images:, ...]

            batch_prediction_array = sess.run(predictions, feed_dict={
                self.test_image_placeholder: test_image_batch})

            prediction_array = np.concatenate((prediction_array, batch_prediction_array))

        return prediction_array

    ## Helper functions
    def loss(self, logits, labels):
        '''
        Calculate the cross entropy loss given logits and true labels
        :param logits: 2D tensor with shape [batch_size, num_labels]
        :param labels: 1D tensor with shape [batch_size]
        :return: loss tensor with shape [1]
        '''
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                       labels=labels, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        return cross_entropy_mean

    def top_k_error(self, predictions, labels, k):
        '''
        Calculate the top-k error
        :param predictions: 2D tensor with shape [batch_size, num_labels]
        :param labels: 1D tensor with shape [batch_size, 1]
        :param k: int
        :return: tensor with shape [1]
        '''
        batch_size = predictions.get_shape().as_list()[0]
        in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k=1))
        num_correct = tf.reduce_sum(in_top1)
        return (batch_size - num_correct) / float(batch_size)

    def generate_vali_batch(self, vali_data, vali_label, vali_batch_size):
        '''
        If you want to use a random batch of validation data to validate instead of using the
        whole validation data, this function helps you generate that batch
        :param vali_data: 4D numpy array
        :param vali_label: 1D numpy array
        :param vali_batch_size: int
        :return: 4D numpy array and 1D numpy array
        '''
        offset = np.random.choice(10000 - vali_batch_size, 1)[0]
        vali_data_batch = vali_data[offset:offset + vali_batch_size, ...]
        vali_label_batch = vali_label[offset:offset + vali_batch_size]
        return vali_data_batch, vali_label_batch

    def generate_augment_train_batch_fit(self, train_data, train_labels, train_batch_size):
        '''
        This function helps generate a batch of train data, and random crop, horizontally flip
        and whiten them at the same time
        :param train_data: 4D numpy array
        :param train_labels: 1D numpy array
        :param train_batch_size: int
        :return: augmented train batch data and labels. 4D numpy array and 1D numpy array
        '''
        offset = np.random.choice(int(EPOCH_SIZE/2) - train_batch_size, 1)[0]
        batch_data = train_data[offset:offset + train_batch_size, ...]
        batch_data = random_crop_and_flip(batch_data, padding_size=FLAGS.padding_size)

        batch_data = whitening_image(batch_data)
        batch_label = train_labels[offset:offset + FLAGS.train_batch_size]

        return batch_data, batch_label
    def generate_augment_train_batch(self, train_data, train_labels, train_batch_size):
        '''
        This function helps generate a batch of train data, and random crop, horizontally flip
        and whiten them at the same time
        :param train_data: 4D numpy array
        :param train_labels: 1D numpy array
        :param train_batch_size: int
        :return: augmented train batch data and labels. 4D numpy array and 1D numpy array
        '''
        #len_data =
        offset = np.random.choice(EPOCH_SIZE - train_batch_size, 1)[0]
        batch_data = train_data[offset:offset + train_batch_size, ...]
        batch_data = random_crop_and_flip(batch_data, padding_size=FLAGS.padding_size)

        batch_data = whitening_image(batch_data)
        batch_label = train_labels[offset:offset + FLAGS.train_batch_size]

        return batch_data, batch_label
    def train_operation(self, global_step, total_loss, top1_error):
        '''
        Defines train operations
        :param global_step: tensor variable with shape [1]
        :param total_loss: tensor with shape [1]
        :param top1_error: tensor with shape [1]
        :return: two operations. Running train_op will do optimization once. Running train_ema_op
        will generate the moving average of train error and train loss for tensorboard
        '''
        # Add train_loss, current learning rate and train error into the tensorboard summary ops
        tf.summary.scalar('learning_rate', self.lr_placeholder)
        tf.summary.scalar('train_loss', total_loss)
        tf.summary.scalar('train_top1_error', top1_error)

        # The ema object help calculate the moving average of train loss and train error
        ema = tf.train.ExponentialMovingAverage(FLAGS.train_ema_decay, global_step)
        train_ema_op = ema.apply([total_loss, top1_error])
        tf.summary.scalar('train_top1_error_avg', ema.average(top1_error))
        tf.summary.scalar('train_loss_avg', ema.average(total_loss))

        opt = tf.train.MomentumOptimizer(learning_rate=self.lr_placeholder, momentum=0.9)
        train_op = opt.minimize(total_loss, global_step=global_step)
        return train_op, train_ema_op

    def validation_op(self, validation_step, top1_error, loss):
        '''
        Defines validation operations
        :param validation_step: tensor with shape [1]
        :param top1_error: tensor with shape [1]
        :param loss: tensor with shape [1]
        :return: validation operation
        '''

        # This ema object help calculate the moving average of validation loss and error

        # ema with decay = 0.0 won't average things at all. This returns the original error
        ema = tf.train.ExponentialMovingAverage(0.0, validation_step)
        ema2 = tf.train.ExponentialMovingAverage(0.95, validation_step)

        val_op = tf.group(validation_step.assign_add(1), ema.apply([top1_error, loss]),
                          ema2.apply([top1_error, loss]))
        top1_error_val = ema.average(top1_error)
        top1_error_avg = ema2.average(top1_error)
        loss_val = ema.average(loss)
        loss_val_avg = ema2.average(loss)

        # Summarize these values on tensorboard
        tf.summary.scalar('val_top1_error', top1_error_val)
        tf.summary.scalar('val_top1_error_avg', top1_error_avg)
        tf.summary.scalar('val_loss', loss_val)
        tf.summary.scalar('val_loss_avg', loss_val_avg)
        return val_op

    def full_validation(self, loss, top1_error, session, vali_data, vali_labels, batch_data,
                        batch_label):
        '''
        Runs validation on all the 10000 valdiation images
        :param loss: tensor with shape [1]
        :param top1_error: tensor with shape [1]
        :param session: the current tensorflow session
        :param vali_data: 4D numpy array
        :param vali_labels: 1D numpy array
        :param batch_data: 4D numpy array. training batch to feed dict and fetch the weights
        :param batch_label: 1D numpy array. training labels to feed the dict
        :return: float, float
        '''
        num_batches = 10000 // FLAGS.validation_batch_size
        order = np.random.choice(10000, num_batches * FLAGS.validation_batch_size)
        vali_data_subset = vali_data[order, ...]
        vali_labels_subset = vali_labels[order]

        loss_list = []
        error_list = []

        for step in range(num_batches):
            offset = step * FLAGS.validation_batch_size
            feed_dict = {self.image_placeholder: batch_data, self.label_placeholder: batch_label,
                         self.vali_image_placeholder: vali_data_subset[offset:offset + FLAGS.validation_batch_size,
                                                      ...],
                         self.vali_label_placeholder: vali_labels_subset[offset:offset + FLAGS.validation_batch_size],
                         self.lr_placeholder: FLAGS.init_lr}
            loss_value, top1_error_value = session.run([loss, top1_error], feed_dict=feed_dict)
            loss_list.append(loss_value)
            error_list.append(top1_error_value)

        return np.mean(loss_list), np.mean(error_list)
    def feature_state_f(self, label, label_pred, logits, loss):
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
    def build_model(self):#self.label_pred, self.logits, self.loss
        global_step = tf.Variable(0, trainable=False)

        self.logits = inference(self.image_placeholder, FLAGS.num_residual_blocks, reuse=False)
        self.vali_logits = inference(self.vali_image_placeholder, FLAGS.num_residual_blocks, reuse=True)
        regu_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss_v = self.loss(self.logits, self.label_placeholder)
        self.full_loss = tf.add_n([self.loss_v] + regu_losses)
        self.label_pred = tf.nn.softmax(self.logits)
        #for vali
        self.vali_loss = self.loss(self.vali_logits, self.vali_label_placeholder)
        self.train_top1_error = self.top_k_error(self.label_pred, self.label_placeholder, 1)
        self.train_op, self.train_ema_op = self.train_operation(global_step, self.full_loss,
                                                                self.train_top1_error)
        self.vali_label_pred = tf.nn.softmax(self.vali_logits)
    def train_one_step_setup(self, x, y, feature_state, sess):
        #student model
        self.build_model()
        self.vars_trainable = tf.trainable_variables(scope='resnet')
        self.teacher = t.TeacherAgent()
        action_prob, prob = self.teacher.build_model(feature_state)
        y = tf.one_hot(self.label_placeholder,10)
        #for training
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(self.label_pred, -1), tf.argmax(y, -1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)

        self.accuracy = tf.reduce_mean(correct_prediction)
        vali_y = tf.one_hot(self.vali_label_placeholder, 10)
        #for vali
        with tf.name_scope('accuracy'):
            vali_correct_prediction = tf.equal(tf.argmax(self.vali_label_pred, -1), tf.argmax(vali_y, -1))
            vali_correct_prediction = tf.cast(vali_correct_prediction, tf.float32)

        self.vali_accuracy = tf.reduce_mean(vali_correct_prediction)

        with tf.name_scope('momentum_optimizer'):
            self.train_step = tf.train.MomentumOptimizer(learning_rate=self.lr_placeholder, momentum=0.9).minimize(self.full_loss)#,var_list=self.vars_trainable)

        sess.run(tf.global_variables_initializer())

    def train_one_step(self, x, y, feature_state, sess, txtWriter, writer_teacher):




        # get one batch from train_teachs
        batch_img, batch_lbl = self.generate_augment_train_batch_fit(self.train_teach_data,
                                                                                     self.train_teach_label,
                                                                                     FLAGS.train_batch_size)
        [label_pred, logits, loss] = sess.run([self.label_pred, self.logits, self.loss_v],
                                    feed_dict={self.image_placeholder: batch_img, self.label_placeholder: batch_lbl, self.lr_placeholder: FLAGS.init_lr})

        [loss_val, train_accuracy] = sess.run([self.vali_loss, self.vali_accuracy],
                                    feed_dict={self.vali_image_placeholder: self.D_dev_img, self.vali_label_placeholder: self.D_dev_lbl, self.lr_placeholder: FLAGS.init_lr})
        self.best_loss = np.minimum(loss_val, self.best_loss)
        # feed to teacher agent
        features = self.feature_state_f(batch_lbl, label_pred, logits, loss)
        action_prob = self.teacher.estimate(sess, features, feature_state)
        # construct new batch of data based on the action_prob
        for i, prob in enumerate(action_prob[0]):
            # sample action
            action = np.random.choice(2, 1, p=[1.0-prob[0], prob[0]])
            if action[0] == 1 :#and len(self.new_batch_data) != self.batch_size
                self.new_batch_data.append(batch_img[i,:])
                self.new_batch_label.append(batch_lbl[i])
                # append reward and features
                self.student_trajectory.append(features[i])
                self.reward.append(0.0)
        # terminate trajectory episode and calculate rewards
        print(' training accuracy %g' % (train_accuracy),"ite",self.iter_index,"last_reward",self.latest_reward,"last_epi_length",self.latest_episode_length)
        if train_accuracy >= self.train_tao:
            print(' length of reward %g' % len(self.reward))
            if len(self.reward) > 0:
                self.reward[-1] = -math.log(float(len(self.reward))/self.T_max)
                reward = self.reward[-1]
                self.latest_reward = reward
                self.latest_episode_length = len(self.reward)
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
            FLAGS.init_lr = 0.1
        if len(self.new_batch_data) >= FLAGS.train_batch_size:
            if self.iter_index == FLAGS.decay_step0 or self.iter_index == FLAGS.decay_step1:
                FLAGS.init_lr = 0.1 * FLAGS.init_lr
                print('Learning rate decayed to ', FLAGS.init_lr)
            sess.run([self.train_op, self.train_ema_op],
                     {self.image_placeholder: self.new_batch_data[0:FLAGS.train_batch_size],
                      self.label_placeholder: self.new_batch_label[0:FLAGS.train_batch_size],
                      self.lr_placeholder: FLAGS.init_lr})
            self.new_batch_data = self.new_batch_data[FLAGS.train_batch_size:]
            self.new_batch_label = self.new_batch_label[FLAGS.train_batch_size:]

