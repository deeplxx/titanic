import tensorflow as tf
import numpy as np
# from sklearn import preprocessing


class Dnn:
    def __init__(self, data_train, label_train, data_test, label_test):
        self.data_train = data_train.values.tolist()
        self.label_train = label_train.values.tolist()
        self.data_test = data_test.values.tolist()
        self.label_test = label_test.values.tolist()
        self.nums_col = len(data_train.columns)
        self.nums_h_unit = self.nums_col // 2
        self.paramters = {
            'w1': tf.Variable(tf.truncated_normal((self.nums_col, self.nums_h_unit)), name='w1'),
            'b1': tf.Variable(tf.zeros([self.nums_h_unit]), name='b1'),
            'w2': tf.Variable(tf.zeros((self.nums_h_unit, self.nums_h_unit // 2)), name='w2'),
            'b2': tf.Variable(tf.zeros([self.nums_h_unit // 2]), name='b2'),
            'w3': tf.Variable(tf.zeros((self.nums_h_unit // 2, 1)), name='w3'),
            'b3': tf.Variable(tf.zeros([1]), name='b3'),
        }

    def inference(self):
        x = tf.placeholder(tf.float32, (None, self.nums_col), name='input')
        y_ = tf.placeholder(tf.float32, (None, ), name='output')
        keep_prob = tf.placeholder(tf.float32)

        h1 = tf.nn.relu(tf.matmul(x, self.paramters['w1']) + self.paramters['b1'], name='h1')
        h2 = tf.nn.relu(tf.matmul(h1, self.paramters['w2']) + self.paramters['b2'], name='h2')
        h2_drop = tf.nn.dropout(h2, keep_prob)
        y = tf.nn.softmax(tf.matmul(h2_drop, self.paramters['w3']) + self.paramters['b3'])

        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=1))
        train_op = tf.train.AdagradOptimizer(1e-3).minimize(cross_entropy)

        # y.eval(feed_dict={})
        # correct_predic = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        # accuracy_ = tf.reduce_mean(tf.cast(correct_predic, tf.float32))

        batch_size = 89
        epoch = 10000
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(epoch):
                random_index = np.random.randint(len(self.data_train) // batch_size)
                x_batch = self.data_train[random_index*batch_size: (random_index+1)*batch_size]
                y_batch = self.label_train[random_index*batch_size: (random_index+1)*batch_size]
                feed_d = {x: x_batch, y_: y_batch, keep_prob: 0.5}
                sess.run(train_op, feed_dict=feed_d)
                # if i % 500 == 0:
                #     train_accuracy = accuracy_.eval(feed_dict=feed_d)
                #     print('step {0}, training accuracy is {1}'.format(i, train_accuracy))

            # for i in range(len(self.data_test) // 100):
            #     feed_d = {x: self.data_test.ix[i*100: (i+1)*100],
            #               y: self.label_test.ix[i*100: (i+1)*100], keep_prob: 1}
            #     print('test accuracy: {0}'.format(accuracy_.eval(feed_dict=feed_d)))


if __name__ == '__main__':
    import src.main
    x_train, x_test, y_train, y_test = src.main.preprocessing()

    model = Dnn(x_train, y_train, x_test, y_test)
    model.inference()
