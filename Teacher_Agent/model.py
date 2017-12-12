import tensorflow as tf

class TeacherAgent():

    def __init__(self,
                training=True,
                reuse=None,
                learning_rate=0.001):
            self.training = training
            self.reuse = reuse
            self.target = tf.placeholder(dtype=tf.float32, name="target")
            self.average_reward_tf = tf.placeholder(dtype=tf.float32, name="average_reward")
            self.learning_rate = learning_rate
            self.episode_count = 0.0
            self.average_reward = 0.0

    def fc(self, x, num, num_filters, bias=0.0, name='fc'):
        with tf.variable_scope(name):
            w_fc = tf.random_uniform(shape=[num, num_filters], minval=-0.01 ,maxval=0.01)
            b_fc = tf.constant(bias, shape=[num_filters])
            w_fc = tf.Variable(w_fc)
            b_fc = tf.Variable(b_fc)
        return tf.matmul(x, w_fc)+b_fc

    def conv1d_relu(self, x, num_filters, kernel_size, strides,
                    name = 'conv1d_relu', reuse = None):
        with tf.variable_scope(name):
            y = tf.layers.conv1d(x, num_filters, kernel_size, strides=strides,
                padding='same', name='conv',
                    reuse=None)
        return tf.nn.relu(y, name='relu')

    def dropout(self, x, name='dropout'):
        with tf.variable_scope(name):
            prob = tf.placeholder(tf.float32)
            x = tf.nn.dropout(x, prob)
        return x, prob

    def build_model(self, x):
        with tf.variable_scope('teacher_model'):
            fc1 = self.fc(x, 25, 12, name='fc1')
            dpout, self.prob = self.dropout(fc1, name='dropout')
            tanh1 = tf.nn.tanh(dpout, name='tanh')
            fc2 = self.fc(tanh1, 12, 1, bias=0.0, name='fc2')
            self.action = tf.nn.sigmoid(fc2, name='sigmoid')
            logits = -tf.log(self.action) * (self.target-self.average_reward_tf)
            self.loss = tf.reduce_sum(logits)

        var_list = tf.trainable_variables(scope='teacher_model')
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss, var_list=var_list)
        # set up tensorboard
        action_prob_tensor = tf.summary.histogram('action_prob', self.action)
        teacher_loss_tensor = tf.summary.scalar('teacher_loss', self.loss)
        self.sum_all = tf.summary.merge([action_prob_tensor, teacher_loss_tensor])
        self.action = tf.round(self.action)
        return self.action, self.prob

    def estimate(self, sess, features, feature_state):
        action = sess.run([self.action], feed_dict={feature_state: features, self.prob: 1.0})
        print(action)
        return action

    def update(self, sess, target, features, feature_state, writer_teacher,if_write_teacher = True):
        print ('teacher update')
        self.average_reward = self.average_reward+(target-self.average_reward)/(1.0+self.episode_count)
        self.episode_count += 1.0
        feed_dict = {feature_state: features, self.prob: 0.5, self.target: target,
                    self.average_reward_tf: self.average_reward}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)

        # save log
        if if_write_teacher:
            writer_teacher.add_summary(sess.run(self.sum_all,
                            feed_dict={feature_state: features, self.prob: 0.5, self.target: target,
                            self.average_reward_tf: self.average_reward}), int(self.episode_count))
        print(loss)
        return loss