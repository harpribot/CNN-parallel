# Parallel Implementation of CNN on all major platforms
This task is in progress, right now these are the platform implementations of CNN. All implementations force GPU presence. So make sure that all platforms are GPU enabled

1. Tensorflow
python tensorflow_script.py --train data/train.csv --test data/test.csv --num_epochs 10
2. Keras - Tensorflow backend
python keras_tensorflow_backend.py
3. Keras - Theano backend
python keras_theano_backend.py
4. Torch - Coming Next
5. Lasagne - Coming Next
6. Caffe - Coming Next

# Dataset Used
The data used is traditional MNIST data of numeric digits. The data is included in data directory, but can also be downloaded from the Kaggle competition,

https://www.kaggle.com/c/digit-recognizer

# Purpose
There is a lot of initial resistance, when people start building their first implementations, in different framework. Most tutorials, specially in deep learning, are not complete in nature, starting from getting data, processing, model construction, fitting, predicting and saving result. This project is aimed at giving refuge to students who need a wholistic solution.

Also most implementations are script oriented. I have tried and made libraries for easy modular use, rather than writing scripts for each different implementation.

# Collaboration
Collaboration is most welcome. I am trying to build a one-place-for-all project, that will have all the deep learning implementation on all possible major frameworks.
