'''
    Task: Digit Recognition using Tensorflow based CNN implementation
    Purpose: Knowledge Acquisition
    Author: Harshal Priyadarshi
    Competition: https://www.kaggle.com/c/digit-recognizer
    Affiliation: University of Texas at Austin - MS Computer Science Student
    Type: Script to pool all the libraries for the final implementation
    Run Instructions: python tensorflow_script.py --train data/train.csv --test data/test.csv --num_epochs 10
    GPU : Enabled
    Data: in /data directory
    Results: in /results directory
    Algorithms: in /algorithms directory


    Design:
        1. Data handler class - loads data, cleans and normalizes the data,
                                parses data into features and labels
                                and then stores the final results
        2. CNN class - does all the meat of CNN.

'''
import numpy as np
import tensorflow as tf
from algorithms.tensorflow_wrapper.dataloader import Handler
from algorithms.tensorflow_wrapper.cnn import CNNclassifier


# Global params
NUM_LABELS = 10
BATCH_SIZE = 50

# Add the optional arguments
tf.app.flags.DEFINE_string('train', None,
                            'file containing the training data (labels & features).')

tf.app.flags.DEFINE_string('test', None,
                            'file containing just the test data.')

tf.app.flags.DEFINE_integer('num_epochs', 1,
                            'Number of examples to separate from the training '
                            'data for the validation set.')

tf.app.flags.DEFINE_boolean('verbose', False,'Produce verbose output.')

FLAGS = tf.app.flags.FLAGS


def main(argv=None):
    handler = Handler(FLAGS)
    handler.extract_train_and_validation_data(NUM_LABELS)
    handler.extract_test_data()


    # Get the training, validation and testing data
    train_X, train_y = handler.get_train_data()
    val_X, val_y = handler.get_validation_data()
    test_X = handler.get_test_data()


    # Get the shape of the training data.
    train_size = handler.get_train_num_samples()
    num_features = handler.get_num_features()

    # Load model and initialize tensorflow session
    model = CNNclassifier(FLAGS)
    model.form_input_graph(num_features, NUM_LABELS)
    model.load_model()
    model.initialize_session()

    # Fit model
    model.fit(train_X,train_y, val_X, val_y,batch_size=50)

    # Make prediction on test data
    test_prediction = model.predict(test_X)

    # Store results
    outfile = './results/cnn_2d_2layer_full_connected_2layer_tensorflow.csv'
    handler.save_results(test_prediction, outfile)

if __name__ == '__main__':
    tf.app.run()
