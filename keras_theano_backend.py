'''
    Task: Digit Recognition using Keras based CNN implementation
    Purpose: Knowledge Acquisition
    Author: Harshal Priyadarshi
    Competition: https://www.kaggle.com/c/digit-recognizer
    Affiliation: University of Texas at Austin - MS Computer Science Student
    Type: Script to pool all the libraries for the final implementation
    Run Instructions: python keras_theano_backend.py
    Backend: Theano (trying to use this for GPU implementation as Tensorflow doesn't support GPU yet)
    Data: in /data directory
    Results: in /results directory
    Algorithms: in /lib directory


    Design:
        1. Data handler class - loads data, cleans and normalizes the data,
                                parses data into features and labels
                                and then stores the final results
        2. CNN class - does all the meat of CNN.

'''
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
os.environ["KERAS_BACKEND"]="theano"
import numpy as np
import pandas as pd
from algorithms.keras_wrapper.cnn import CNNclassifier
from algorithms.keras_wrapper.dataloader import Handler


def main(argv=None):
    train_path = 'data/train.csv'
    test_path = 'data/test.csv'
    out_path = 'results/cnn2d_3layer_fullconnected_3layer.csv'
    # Get the features
    data_handler = Handler(train_path, test_path)
    data_handler.process_data()
    train_x, train_y, test_x = data_handler.get_features_and_labels()

    # Load the model
    model = CNNclassifier()
    model.load_model()

    # Train the model
    model.fit(train_x, train_y)

    # Test the model
    test_y = model.predict(test_x)

    # Store the result
    data_handler.store_prediction(test_y, out_path)


if __name__=="__main__":
    main()
