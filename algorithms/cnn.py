from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD


class CNNclassifier:
    def __init__(self):
        pass

    def load_model(self):
        self.model = Sequential()
        self.__add_convolutional_layers()
        self.__do_flattening()
        self.__add_fully_connected_layers()
        self.__add_optimizer()


    def __add_convolutional_layers(self):
        # first convolutional layer
        self.model.add(ZeroPadding2D((1,1),input_shape=(1,28,28)))
        self.model.add(Convolution2D(32,3,3, activation='relu'))

        # second convolutional layer
        self.model.add(ZeroPadding2D((1,1)))
        self.model.add(Convolution2D(48,3,3, activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))

        # third convolutional layer
        self.model.add(ZeroPadding2D((1,1)))
        self.model.add(Convolution2D(32,3,3, activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))

    def __do_flattening(self):
        # convert convolutional filters to flatt so they can be feed to
        # fully connected layers
        self.model.add(Flatten())

    def __add_fully_connected_layers(self):
        # first fully connected layer
        self.model.add(Dense(128, init='lecun_uniform'))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.25))

        # second fully connected layer
        self.model.add(Dense(128, init='lecun_uniform'))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.25))

        # last fully connected layer which output classes
        self.model.add(Dense(10, init='lecun_uniform'))
        self.model.add(Activation('softmax'))

    def __add_optimizer(self):
        sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd)

    def fit(self, train_x, train_y):
        self.model.fit(train_x, train_y, \
                        nb_epoch=10, batch_size=1000, \
                            validation_split=0.2, show_accuracy=True)

    def predict(self, test_x):
        return self.model.predict_classes(test_x, verbose=0)
