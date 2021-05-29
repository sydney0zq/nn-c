#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
mnist_loader
------------

A library to load the MNIST image data. For details of the data
structures that are returned, see the doc comments for `load_data`
and `load_data_wrapper`. In practice, `load_data_wrapper` is the 
function usually called by our neural network code.
"""

#Standard library
#import cPickle
import pickle
import gzip

#Third-party libraries
import numpy as np


def load_data():
    """
    Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The `training data` is returned as a tuple with two entries. The 
    first entry contains the actual training images. This is a numpy
    ndarray with 50000 entries. Each entry is, in turn, a numpy ndarray
    with 784 values, representing the 28*28 = 784 pixels in a single
    MNIST image.
    
    The second entry in the `training data` tuple is a numpy ndarray
    containing 50000 entries. Those entries are just the digit value
    (0,...,9) for the corrsponding images contained in the first entry
    of the tuple.

    The `validation_data` and the `test_data` are similar, except each 
    containing only 10000 images.

    This si a nice data format, but for use in NN it is helpful to 
    modify the format of `training data` a little. That is done in the 
    wrapper function `load_data_wrapper()`, see below.
    """

    f = gzip.open("../dataset/mnist.pkl.gz", "rb")
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()
    return (training_data, validation_data, test_data)


def load_data_wrapper():
    """
    Return a tuple containing `(training_data, validation_data, test_data)`.
    Based on `load_data`, but the format is more convenient for use in our
    implementation of NN.

    In particular, `training_data` is a list containing 50000 2-tuples `(x,y)`.
    `x` is 784 dimensional numpy.ndarray containing the input image. `y` is a 
    10-dimensional numpy.ndarray representing the unit vector corresponding to
    the correct digit for `x`.

    `validation_data` and `test_data` are lists containing 10000 2-tuples `(x,y)`.
    In each case, `x` is a 784 dimensional numpy.ndarray containing the input image,
    and `y` is the corresponding classification, i.e., the digit values(integers)
    corresponding to `x`.

    Obviously, this means we are using slightly different formats for the training
    data and the validation/test data. These formats turn out to be the most convenient
    for use in our neural network code.
    """ 

    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    
    with open("dataset_train.txt", "w") as f:
        for item in training_inputs:
            line = ""
            for i in range(784):
                line += str(float(item[i]))+" "
            f.write(line + "\n")
    with open("dataset_train_label.txt", "w") as f:
        for item in training_results:
            line = ""
            for i in range(10):
                line += str(int(item[i]))+" "
            f.write(line + "\n")

    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_results = [vectorized_result(y) for y in va_d[1]]
    with open("dataset_vali.txt", "w") as f:
        for item in validation_inputs:
            line = ""
            for i in range(784):
                line += str(float(item[i]))+" "
            f.write(line + "\n")

    with open("dataset_vali_label.txt", "w") as f:
        for item in validation_results:
            line = ""
            for i in range(10):
                line += str(int(item[i]))+" "
            f.write(line + "\n")

    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_results= [vectorized_result(y) for y in te_d[1]]
    with open("dataset_test.txt", "w") as f:
        for item in test_inputs:
            line = ""
            for i in range(784):
                line += str(float(item[i]))+" "
            f.write(line + "\n")
    with open("dataset_test_label.txt", "w") as f:
        for item in test_results:
            line = ""
            for i in range(10):
                line += str(int(item[i]))+" "
            f.write(line + "\n")


def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

load_data_wrapper()
