import tensorflow as tf

class TeacherAgent():

    def __init__(self,
                training=True,
                batch_size=50,
                state_size=25,
                reuse=None):
            self.training = training
            self.reuse = reuse
            self.batch_size = batch_size
            self.state_size = state_size
    
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

        dpout, prob = self.dropout(fc1, name='dropout')

        fc2 = self.fc_relu(dpout, 1024, 1, name='fc_relu1')

        action_space = tf.nn.sigmoid(fc2, name='sigmoid')
        action = tf.round(action_space, name='round')

        return action_space, action, prob