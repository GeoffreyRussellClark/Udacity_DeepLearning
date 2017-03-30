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

#Train the data using LogisticRegression
print('\nProblem6\n')

train_dataset_rs = train_dataset.reshape(train_dataset.shape[0], 28*28)
valid_dataset_rs = valid_dataset.reshape(valid_dataset.shape[0], 28*28)

"""
def accuracy(predictions, labels):
    return 1 - float(np.count_nonzero(predictions - labels)) / predictions.shape[0]"""

def Train_LogisticReg(batch):
    model = LogisticRegression()
    model.fit(train_dataset_rs[:batch,], train_labels[:batch,])
    #predictions = model.predict(valid_dataset_rs)
    #return accuracy(predictions, valid_labels)
    return model.score(valid_dataset_rs, valid_labels)

print("LogisticRegression accuracy trainng size of 50: ", Train_LogisticReg(50))
print("LogisticRegression accuracy trainng size of 100: ", Train_LogisticReg(100))
#print("LogisticRegression accuracy trainng size of 1000: ", Train_LogisticReg(1000))
#print("LogisticRegression accuracy trainng size of 5000: ", Train_LogisticReg(5000))
#print("LogisticRegression accuracy trainng size of 10000: ", Train_LogisticReg(10000))
#This one gives as accuracy of 89% on the test set
#print("LogisticRegression accuracy trainng size of 200000: ", Train_LogisticReg(200000))

"""
#logistic regression from scratch
def sigmoid(z):
    return 1.0/(1.0+e**(-1.0*z))

def compute_cost(theta, X, y):
    m = y.shape[0]
    theta = reshape(theta,(len(theta),1))
    J = (1.0/m) * (-transpose(y).dot(log(sigmoid(X.dot(theta)))) - (transpose(1-y)).dot(log(1-sigmoid(X.dot(theta)))))
    #cost = -y * log(sigmoid(theta * X)) - (1-y) * log(1 - sigmoid(theta * X))
    #return 1.0/y.shape[0] * sum(cost)
    return J[0][0]"""
