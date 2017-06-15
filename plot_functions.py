#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 12:59:49 2017

@author: esoroush
"""

import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.stats import logistic
from scipy.special import expit



# its faster in runtime, but it only supports single value
def my_sigmoid(x):
    return 1/(1+math.exp(-x))

def my_arraysigmoid(x):
    return 1/(1+np.exp(-x))


def plot_sigmoid(x):
#    plt.plot(x, my_arraysigmoid(x),'-r', label='sigmoid')
#    plt.plot(x, logistic.cdf(x),'-r', label='sigmoid')
    plt.plot(x, expit(x),'-r', label='sigmoid')
    
def plot_tanh(x):
    plt.plot(x, np.tanh(x),'-b', label='tanh')
    
def main():
    x = np.linspace(-10, 10, 1000)
    plot_tanh(x)    
    plot_sigmoid(x)
    plt.legend(['tanh', 'sigmoid'])

if __name__ == '__main__':
    main()