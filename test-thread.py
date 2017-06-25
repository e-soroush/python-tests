#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 21:24:44 2017

@author: esoroush
"""
import numpy as np
import threading

def myfcn(x):
    y = np.power(x, 10)
    print("y = {}".format(x))
    return y


for i in range(4):
    t = threading.Thread(target=myfcn, args=(2,))
    t.start()