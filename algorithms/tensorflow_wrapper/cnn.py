import tensorflow.python.platform

import tensorflow as tf
import numpy as np

class CNNclassifier:
    def __init__(self, FLAGS):
        # Get the number of epochs for training.
        self.num_epochs = FLAGS.num_epochs

    def form_input_graph(self, num_features, num_labels):
        # Feed training sample and labels to the graph
        self.num_features = num_features
        self.num_labels = num_labels
        self.x = tf.placeholder("float", shape=[None, num_features])
        self.y_ = tf.placeholder("float", shape=[None,num_labels])

    def load_model(self):
        self.__add_convolutional_layers()
        self.__add_fully_connected_layers()
        self.__add_optimizer()
        self.__add_metrics()


    def __add_convolutional_layers(self):
        ###### 1st layer CNN
        # initialize weight and bias of CNN
        self.W_conv1 = self.weight_variable([5, 5, 1, 32])
        self.b_conv1 = self.bias_variable([32])
        # reshape x to a 4d tensor
        self.x_image = tf.reshape(self.x, [-1,28,self.num_features/28,1])
        # add cnn layer with RELU
        self.h_conv1 = tf.nn.relu(self.conv2d(self.x_image, self.W_conv1) + self.b_conv1)
        # add max pooling
        self.h_pool1 = self.max_pool_2x2(self.h_conv1)

        ###### 2nd Layer CNN
        # initialize weight and bias of CNN
        self.W_conv2 = self.weight_variable([5, 5, 32, 64])
        self.b_conv2 = self.bias_variable([64])
        # add cnn layer with RELU
        self.h_conv2 = tf.nn.relu(self.conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)
        # add max pooling
        self.h_pool2 = self.max_pool_2x2(self.h_conv2)

    def weight_variable(self,shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

    def bias_variable(self,shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)

    def conv2d(self, x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')


    def __add_fully_connected_layers(self):
        ###### Densely Connected layer - 1 (relu)
        self.W_fc1 = self.weight_variable([7 * 7 * 64, 1024])
        self.b_fc1 = self.bias_variable([1024])
        self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 7*7*64])
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1)

        ###### Dropout layer
        self.keep_prob = tf.placeholder(tf.float32)
        self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)

        ###### Densely Conected layer -2 (softmax)
        self.W_fc2 = self.weight_variable([1024, self.num_labels])
        self.b_fc2 = self.bias_variable([self.num_labels])
        self.y_conv=tf.nn.softmax(tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2)


    def __add_optimizer(self):
        # Loss function
        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y_conv),\
                                                reduction_indices=[1]))
        # Optimizer
        self.learning_rate = 5e-4
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cross_entropy)

    def __add_metrics(self):
        self.correct_prediction = tf.equal(tf.argmax(self.y_conv,1), \
                                           tf.argmax(self.y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        prediction=tf.argmax(self.y_conv,1)

    def initialize_session(self):
        ## Create and initialize the interactive session
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.initialize_all_variables())


    def fit(self, train_X, train_y, val_X, val_y, batch_size=50):
        train_size = train_X.shape[0]
        # Iterate and train.
        for step in xrange(self.num_epochs * train_size // batch_size):
            offset = (step * batch_size) % train_size
            batch_data = train_X[offset:(offset + batch_size), :]
            batch_labels = train_y[offset:(offset + batch_size)]
            # Train
            self.sess.run(self.optimizer,feed_dict={\
                                        self.x: batch_data, \
                                        self.y_: batch_labels, \
                                        self.keep_prob: 0.5})

            if(step % 10 == 0):
                print 'Step Count:', step
                # Get a validation accuracy
                print 'Validation Acc: ',self.sess.run(self.accuracy, \
                                            feed_dict={self.x: val_X, \
                                                        self.y_: val_y, \
                                                        self.keep_prob: 1.0})

    def predict(self, test_X):
        self.test_prediction = self.sess.run(self.prediction,
                        feed_dict={self.x: test_X, self.keep_prob: 1.0})

        return self.test_prediction
