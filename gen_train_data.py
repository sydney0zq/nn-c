#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017-07-16 Sydney <theodoruszq@gmail.com>

"""
Generate dataset.
"""

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model

# Generate a dataset and plot it
np.random.seed(0)
# The dataset we generated has two classes, plotted as red and blue points.
# You can think of the blue dots as male patients 
# and the red dots as female patients, with the x- and y- axis being medical measurements.
X, y = sklearn.datasets.make_moons(200, noise=0.20)
# Drop the square brackets (equivalent to removing the wrapping list() call above) will instead pass a temporary generator to file.writelines()
with open("dataset.txt", "w") as f:
    i = 0
    for item in X:
        f.write(str(item[0]) + " " + str(item[1]) + " " + str(y[i]) + "\n")
        i += 1

print (X, y)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
plt.savefig("./dataset_distribution.png")

