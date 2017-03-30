from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle


#load the data from the pickled file
with open('notMNIST.pickle') as f:
    notMNIST = pickle.load(f)

train_dataset = notMNIST['train_dataset']
train_labels = notMNIST['train_labels']
test_dataset = notMNIST['test_dataset']
test_labels = notMNIST['test_labels']
valid_dataset = notMNIST['valid_dataset']
valid_labels = notMNIST['valid_labels']

print('train_dataset shape: ', train_dataset.shape)
print('train_labels shape: ', train_labels.shape)
print('test_dataset shape: ', test_dataset.shape)
print('test_labels shape: ', test_labels.shape)
print('valid_dataset shape: ', valid_dataset.shape)
print('valid_labels shape: ', valid_labels.shape)

#Problem 5: determine the amount of overlap between the training, test and validation datasets
print('\nProblem5\n')

def multidim_intersect(arr1, arr2):
    arr1_view = arr1.view([('',arr1.dtype)]*arr1.shape[1])
    arr2_view = arr2.view([('',arr2.dtype)]*arr2.shape[1])
    intersected = np.intersect1d(arr1_view, arr2_view)
    return intersected.view(arr1.dtype).reshape(-1, arr1.shape[1])

overlapTrainTest = multidim_intersect(np.reshape(test_dataset, (test_dataset.shape[0], 28*28)), np.reshape(train_dataset, (train_dataset.shape[0], 28*28)))
print('No of overlapping images in train and test datsets: ',overlapTrainTest.shape[0])
overlapTrainValid = multidim_intersect(np.reshape(train_dataset, (train_dataset.shape[0], 28*28)), np.reshape(valid_dataset, (valid_dataset.shape[0], 28*28)))
print('No of overlapping images in train and validation datsets: ',overlapTrainValid.shape[0])
overlapTestValid = multidim_intersect(np.reshape(test_dataset, (test_dataset.shape[0], 28*28)), np.reshape(valid_dataset, (valid_dataset.shape[0], 28*28)))
print('No of overlapping images in test and validation datsets: ',overlapTestValid.shape[0])

"""
#creating sanitized datasets
def sanitize(arr1, arr2):
    ratio = arr1.shape[0]/(arr1.shape[0] + arr2.shape[0])
    a = np.append(arr1, arr2)
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    unique_a = unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))
    return unique_a[:int(ratio*unique_a.shape[0]),:], unique_a[int(ratio*unique_a.shape[0]):,:]

test = test_dataset[np.in1d(test_dataset.view(dtype='i,i').reshape(test_dataset.shape[0]),valid_dataset.view(dtype='i,i').reshape(valid_dataset.shape[0]))]
print(test)
"""
