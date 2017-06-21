#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 17:27:13 2017
least square for solving 1/2||Ax-b||**2

@author: esoroush
"""

import numpy as np

A = np.random.rand(5, 5)
b = np.random.rand(5, 1)
x = np.dot(np.linalg.inv(A), b)
delta = 1e-3
epsilon = 1e-1

def constrained_least_square():
    l = np.random.rand(1)
    x = np.dot(np.linalg.inv(np.dot(A.T, A) + 2*l*np.eye(A.shape[0])), np.dot(A.T, b))
    grad_l = np.dot(x.T, x) - 1
    epoch = 0
    max_epoch = 500
    while np.linalg.norm(grad_l) > delta:
        l += epsilon*grad_l.squeeze() # gradient ascent
        x = np.dot(np.linalg.inv(np.dot(A.T, A) + 2*l*np.eye(A.shape[0])), np.dot(A.T, b))
        grad_l = np.dot(x.T, x) - 1
        epoch += 1
        if epoch > max_epoch:
            print("Couldn't converge in %d epochs"%max_epoch)
    print("x_hat: {} and norm(x) is: {}".format(x, np.linalg.norm(x)))
    print("error is: %.4f"%np.linalg.norm(np.dot(A, x)-b) )
    
    
def least_square():
    x_hat = np.random.rand(5, 1)
    grad = np.dot(np.dot(A.T, A), x_hat) - np.dot(A.T, b)
    while np.linalg.norm(grad) > delta:
        x_hat -= epsilon*grad # gradient descent
        grad = np.dot(np.dot(A.T, A), x_hat) - np.dot(A.T, b)
    print("x_hat is {}".format(x_hat))
    print("error is: %.4f"%(np.linalg.norm(np.dot(A, x_hat)-b)))


#least_square()    

constrained_least_square()