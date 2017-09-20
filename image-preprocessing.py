#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 15:59:37 2017

This script reproduces some preprocessing techinques described in http://cs231n.github.io/neural-networks-2/
Given some images it will calculate eigen values of images and reproduces images using top 36 eigenvalues of images.
@author: esoroush
"""

import os, pickle
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

dataset_path = '/home/esoroush/Datasets/pedestrian/pedestrians128x64'
list_images = os.listdir(dataset_path)
rawimages_filepath = dataset_path + '/raw.p'
width = 64
height = 128
m = width * height # dimension of each image in one row
n = 100 # number of images
fig_x_width = 4
fig_y_height = 5
num_eigenvector = 36
force = True
if os.path.exists(rawimages_filepath) and force == False:
    with open(rawimages_filepath, 'r') as f:
        X = pickle.load(f)
else:
    X = np.zeros((m, n), dtype='uint8')
    for i, image_name in enumerate(list_images):
        X[:, i] = np.array(Image.open('%s/%s' % (dataset_path, image_name)).convert('L')).reshape(-1,)
        if i == n-1:
            break
    with open(rawimages_filepath, 'w') as f:
        pickle.dump(X, f)

U, s, V = np.linalg.svd(X)
U_selected = U[:, :num_eigenvector]

fig, axis = plt.subplots(int(np.sqrt(num_eigenvector)), 
                         int(np.sqrt(num_eigenvector)), 
                         figsize=(fig_x_width*np.sqrt(num_eigenvector), 
                                  fig_y_height*np.sqrt(num_eigenvector)))

# save and show eigen images
for i, ax in enumerate(axis.ravel()):
    ax.imshow((255*U_selected[i,:].reshape(height, width)).astype('uint8'), cmap='gray')
plt.show()
fig.savefig('eigenImages.png')

# save and show reduced images
X_rot = np.dot(U_selected, X)
X_reduced = np.dot(U_selected.T, X_rot)
fig, axis = plt.subplots(int(np.sqrt(n)),
                         int(np.sqrt(n)),
                         figsize=(fig_x_width*np.sqrt(n),
                                  fig_y_height*np.sqrt(n)))
for i, ax in enumerate(axis.ravel()):
    ax.imshow(X_reduced[:, i].astype('uint8').reshape(height, width), cmap='gray')
plt.show()
fig.savefig('reducedImage.png')

