# Pso-double-stack-generalization
This is a new framework combining feature selection and Regression modeling for NIR data
1)First step creating the multiple stack generalization for classification purpose(FSGC)
  a)Find all the best parameters for different models through computing Hyperparameters tunning (TunedKnn,tunedSVC,tunedRf,tunedRidge.....all files started with tuned)
  b) choose all the best models and create the First multiple Stacked Generalization for Classification purpose(FSGC)
  
 
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier



import numpy as np
from sklearn.datasets import load_linnerud
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import pandas as pd


    final_layer = StackingClassifier(
    estimators=[('ridge', RidgeClassifier(alpha= 0.1)),
                ('knn', KNeighborsClassifier(metric= 'euclidean', n_neighbors= 1, weights= 'uniform'))],
    final_estimator=SVC(C= 1, gamma= 0.01, kernel= 'rbf')
    )

    multi_layer_classificator = StackingClassifier(
    estimators=[('SVC',SVC(C= 1, gamma= 0.01, kernel= 'rbf')),
                ('Lr', LogisticRegression(C= 100, penalty= 'l2', solver= 'newton-cg'))],
    final_estimator=final_layer
    )
 #
 #
 #
 #
 #

 2)Computing PSO and call the FSGC as model and find the best parameter for particle, iteration and alpha
#
#
#
#
#
#

  3)  Creating the Second multiple Stack Generalization Regressor we call (SSGR)
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor


final_layer = StackingRegressor(
    estimators=[('ridge', Ridge(alpha= 0.1)),
                ('knn', KNeighborsRegressor(metric= 'euclidean', n_neighbors= 1, weights= 'uniform'))],
    final_estimator=SVR(C= 1, gamma= 0.01, kernel= 'rbf')
    )

multi_layer_regressor = StackingRegressor(
    estimators=[('SVR',SVR(C= 1, gamma= 0.01, kernel= 'rbf')),
                ('mlp', MLPRegressor(activation= 'tanh', alpha= 0.0001, hidden_layer_sizes= (10, 30, 10), learning_rate= 'constant', solver= 'adam'))],
    final_estimator=final_layer
)
multi_layer_regressor.fit(X_train, y_train)
Prediction
y_pred = multi_layer_regressor.predict(X_test)
y_predtrain = multi_layer_regressor.predict(X_train)
