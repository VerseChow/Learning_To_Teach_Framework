import tensorflow as tf

class TeacherAgent():

    def __init__(self,
                training=True,
                batch_size=50,
                state_size=25,
                reuse=None,
                learning_rate=0.001):
            self.training = training
            self.reuse = reuse
            self.batch_size = batch_size
            self.state_size = state_size
            self.target = tf.placeholder(dtype=tf.float32, name="target")

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

    def build_model(self, x):

        fc1 = self.fc_relu(x, 25, 1024, name='fc_relu1')

        dpout, self.prob = self.dropout(fc1, name='dropout')

        fc2 = self.fc_relu(dpout, 1024, 1, name='fc_relu1')

        self.action_space = tf.nn.sigmoid(fc2, name='sigmoid')
        self.action = tf.round(self.action_space, name='round')
        self.loss = -tf.log(self.action_space) * self.target


        return self.action_space, self.action, self.prob

    def estimate(self,sess, features, feature_state):

        action_space, action = sess.run([self.action_space, self.action],
                                        feed_dict={feature_state: features, self.prob: 1.0})
        return action_space,action


   