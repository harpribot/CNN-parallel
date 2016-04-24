import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing


class Handler:
    def __init__(self, FLAGS):
        # Be verbose?
        self.verbose = FLAGS.verbose
        # Get the data.
        self.train_data_filename = FLAGS.train
        self.test_data_filename = FLAGS.test


    def extract_train_and_validation_data(self,num_labels):
        data = pd.read_csv(self.train_data_filename, header=0).values
        # convert to Numpy array forms
        feature_vec = data[0::,1::]
        labels = data[0::,0]

        # mean normalize features
        min_max_scaler = preprocessing.MinMaxScaler()
        feature_vec = min_max_scaler.fit_transform(feature_vec.T).T

        # convert to one hot form for labels
        labels_onehot = (np.arange(num_labels) == labels[:, None]).astype(np.float32)

        # divide data into train and validation data
        self.train_X, self.val_X, self.train_y, self.val_y = train_test_split(\
                                            feature_vec, labels_onehot,
                                            test_size=0.2, random_state=42)


    def extract_test_data(self):
        feature_vec = pd.read_csv(self.test_data_filename, header=0).values

        # mean normalize features
        min_max_scaler = preprocessing.MinMaxScaler()
        feature_vec = min_max_scaler.fit_transform(feature_vec.T).T

        self.test_X = feature_vec

    def get_train_data(self):
        return self.train_X, self.train_y

    def get_validation_data(self):
        return self.val_X, self.val_y

    def get_test_data(self):
        return self.test_X

    def get_train_num_samples(self):
        return self.train_X.shape[0]

    def get_num_features(self):
        return self.train_X.shape[1]

    def store_results(self, result, outfile):
        # Predict result for test data
        df = pd.DataFrame()
        df['ImageId'] = np.arange(1, self.test_X.shape[0] + 1)
        df['Label'] = result
        df.to_csv(outfile, index=False)
