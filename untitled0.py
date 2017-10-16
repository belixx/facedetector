

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn import svm
from sklearn.naive_bayes import *
from sklearn.linear_model import Perceptron
import numpy as np
from datetime import timedelta
from sklearn.decomposition import RandomizedPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from numpy import genfromtxt 
from sklearn.neighbors import KNeighborsClassifier

from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC


train_data = genfromtxt('mydata70.csv',delimiter=',') 
train_target = genfromtxt('Q7.csv',delimiter=',') 

test_data = genfromtxt('mydata30.csv',delimiter=',')
test_target = genfromtxt('Q3.csv',delimiter=',')

n_components = 300
pca = RandomizedPCA(n_components=n_components, whiten=True).fit(train_data)

X_train_pca = pca.transform(train_data)
X_test_pca = pca.transform(test_data)

print("Fitting the classifier to the training set")

param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='sigmoid', class_weight='auto'), param_grid)
clf = clf.fit(X_train_pca, train_target)
Y_pred3 =  clf.predict(X_test_pca)
acc_rbf = metrics.accuracy_score(test_target, Y_pred3)
print("Accuracy of rbf_svc : %.3f" % acc_rbf)


#def svm_classify():    
#     
#     neigh = KNeighborsClassifier(n_neighbors=10)
#     
#     linear_svc = svm.SVC(kernel='linear')
#     polynomial_svc = svm.SVC(kernel='poly')
#     rbf_svc = svm.SVC(kernel='rbf')
#     sigmoid_svc = svm.SVC(kernel='sigmoid')
#      
#     neigh.fit(X_train_pca, train_target)
#     linear_svc.fit(X_train_pca, train_target) 
#
#     polynomial_svc.fit(X_train_pca, train_target) 
#
#     rbf_svc.fit(X_train_pca, train_target)
#     
#     sigmoid_svc.fit(X_train_pca, train_target)
#     
#     
#     Y_pred1 =  linear_svc.predict(X_test_pca)
#     Y_pred2 =  polynomial_svc.predict(X_test_pca)
#     Y_pred3 =  rbf_svc.predict(X_test_pca)
#     Y_pred4 =  sigmoid_svc.predict(X_test_pca)
#     Y_pred5 =  neigh.predict(X_test_pca)
#     
#     acc_linear = metrics.accuracy_score(test_target, Y_pred1)
#     acc_polynomial = metrics.accuracy_score(test_target, Y_pred2)
#     acc_rbf = metrics.accuracy_score(test_target, Y_pred3)
#     acc_sigmoid = metrics.accuracy_score(test_target, Y_pred4)
#     acc_neigh = metrics.accuracy_score(test_target, Y_pred5)
#     
#     print("=============SVM================")
#     print("Accuracy of linear_svc : %.3f" % acc_linear)
#
#     print("\n")
#     print("Accuracy of polynomial_svc : %.3f" % acc_polynomial)
#
#     print("\n")
#     print("Accuracy of rbf_svc : %.3f" % acc_rbf)
#
#     print("\n")
#     print("Accuracy of sigmoid_svc : %.3f" % acc_sigmoid)
#     
#     print("\n")
#     print("Accuracy of neighbors : %.3f" % acc_neigh)
   
     


#svm_classify()

# Train a SVM classification model

