import tensorflow as tf

class TeacherAgent():

    def __init__(self,
                training=True,
                batch_size=50,
                state_size=25,
                reuse=None,
                learning_rate=0.0001):
            self.training = training
            self.reuse = reuse
            self.batch_size = batch_size
            self.state_size = state_size
            self.target = tf.placeholder(dtype=tf.float32, name="target")
            self.learning_rate = learning_rate

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
            #dpout = tf.reshape(dpout, shape=[-1, 1024, 1])
            #conv1 = self.conv1d_relu(dpout, 1, 3, 1, name='conv1d_relu1')
            #conv1 = tf.reshape(conv1, shape=[-1, 1024])
            fc2 = self.fc(tanh1, 12, 1, bias=2.0, name='fc2')
            self.action = tf.nn.sigmoid(fc2, name='sigmoid')
            logits = -tf.log(self.action) * self.target
            self.loss = tf.reduce_mean(logits)
        var_list = tf.trainable_variables(scope='teacher_model')
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss, var_list=var_list)

        return self.action, self.prob

    def estimate(self,sess, features, feature_state):
        action = sess.run([self.action], feed_dict={feature_state: features, self.prob: 1.0})
        print(action)
        return action


    def update(self,sess, target, features, feature_state):
        print ('update')
        feed_dict = {feature_state: features, self.prob: 0.5, self.target: target}
        _,loss = sess.run([self.train_op, self.loss], feed_dict)
        print(loss)
        return loss