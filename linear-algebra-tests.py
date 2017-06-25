# -*- coding: utf-8 -*-
"""
Title: linear-algebra-tests
Tests for linear algebra in Python and Numpy
Created on Wed May  3 10:49:21 2017

@author: esoroush
"""
import numpy as np
a = np.array([0.1, 1, 2])
b = np.array([1, 2, 3])
print("f(a) + f(b) = {} f(a+b) = {} ".format(
        np.linalg.norm(a) + np.linalg.norm(b), np.linalg.norm(a+b)))
# L-Norm in numpy
for l in range(-1, 3):
    print("L{} norm of a: {}".format(l, np.linalg.norm(a,ord=l)))
print("L-inf norm of a: {}".format(np.linalg.norm(a, ord=-np.inf)))

# compute eigen values and vectors and confirm some functionality of them
a = np.arange(1, 10).reshape(3,3)
evalue, evector = np.linalg.eig(a)
print("Av ==? lambda*v: {}".format(np.dot(a, evector[:, 0]) / evector[:, 0] - evalue[0]))
# eigenvalue decomposition and reconstruction:
print("Reconstructed  with eigenvalue decomposition A: {}".format(np.dot(np.dot(evector, np.diag(evalue)), 
                                          np.linalg.inv(evector))))
# reconstruct matrix correspond to high eigenvalues (first two eigen values) (PCA)
selected_evector = evector[:,:2]
a_coded = np.dot(selected_evector.T, a)
a_decoded = np.dot(selected_evector, a_coded)
print("Error reconstruction: {}".format(np.linalg.norm(a_decoded - a)/np.linalg.norm(a)))

# compute signular value decompisition:
u, d, v = np.linalg.svd(a)
print("Reconstructed with SVD A: {}".format(np.dot(np.dot(u, np.diag(d)), v)))
