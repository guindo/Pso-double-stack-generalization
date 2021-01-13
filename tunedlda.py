# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 17:00:40 2020

@author: Mahamed


"""


import numpy as np
from sklearn.datasets import load_linnerud
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import pandas as pd

#import the dataset
dataset=pd.read_csv('NSOM.csv').values
X = dataset[:,:-3]
y = dataset[:,2151:2152]
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV

clf = LinearDiscriminantAnalysis()


parameter_space = {
    'solver': ['svd', 'lsqr', 'eigen'],
    'shrinkage': ['auto', 'float'],
    
}

grid_search = GridSearchCV(clf, parameter_space, n_jobs=-1, cv=5, scoring = 'accuracy')
grid_search.fit(X, y) # X is train samples and y is the corresponding labels
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)