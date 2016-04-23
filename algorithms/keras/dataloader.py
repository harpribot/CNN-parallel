import numpy as np
import pandas as pd
from keras.utils import np_utils

class Handler:
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        self.__load_data()

    def __load_data(self):
        self.df_train = pd.read_csv(self.train_path)
        self.df_test = pd.read_csv(self.test_path)

    def process_data(self):
        ## TRAIN FEAT, LABELS
        self.train_x = (self.df_train.ix[:,1:].values).astype('float32')
        self.train_y = (self.df_train.ix[:,0].values).astype('int32')
        # converts to one hot form
        self.train_y = np_utils.to_categorical(self.train_y)
        ## TEST FEAT
        self.test_x = (self.df_test.ix[:,0:].values).astype('float32')
        ## Normalize training and testing data
        self.__normalize_data()
        ## Reshape the training and testing data
        self.__reshape_data()


    def __normalize_data(self):
        # Scale
        max_value = np.max(self.train_x)
        self.train_x /= max_value
        self.test_x /= max_value
        # Make mean = 0
        mean_value = np.std(self.train_x)
        self.train_x -= mean_value
        self.test_x -= mean_value

    def __reshape_data(self):
        num_train = self.train_x.shape[0]
        num_test = self.test_x.shape[0]
        self.train_x = self.train_x.reshape(num_train, 1, 28, 28)
        self.test_x = self.test_x.reshape(num_test, 1, 28, 28)

    def get_features_and_labels(self):
        return self.train_x, self.train_y, self.test_x

    def store_prediction(self, prediction,outpath):
        pd.DataFrame({"ImageId": list(range(1,len(prediction)+1)),\
                    "Label": prediction}).to_csv(outpath, index=False, header=True)
