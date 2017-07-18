#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017-07-18 Sydney <theodoruszq@gmail.com>

"""
"""
import numpy as np

W1 = np.array([[-0.328729, 0.148182, 0.098424], [0.098424,0.119484,-1.067337]])
W2 = np.array([[-0.817034, 0.156175], [-0.112129, -0.893283],[0.074016,0.384173]])
X = np.array([[0.743461176196, 0.464656328377], [1.65755661779,-0.632031569098]])
z1 = X.dot(W1)
print ("z1:", z1)
a1 = np.tanh(z1)
z2 = a1.dot(W2)
print ("a1:", np.tanh(z1))
exp_scores = np.exp(z2)
print ("exp_scores:", exp_scores)
probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
print ("probs:", probs)
delta3 = probs
delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
print ("delta2", delta2)
dW1 = np.dot(X.T, delta2)




