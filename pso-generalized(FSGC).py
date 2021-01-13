# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 16:05:48 2020

@author: Mahamed
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 18:27:53 2020

@author: Mahamed
"""
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

#import the dataset
dataset=pd.read_csv('mydata.csv')
X = dataset.iloc[:,:-3].values
#extract the label
y = dataset.iloc[:,2151:2152].values
#scaling 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
#encoding label
from sklearn.preprocessing import LabelEncoder
y=LabelEncoder().fit_transform(y)


'''
when SVM is classificator instead of generalized same parameter used inside generalized
#multi_layer_classificator=SVC(C= 1, gamma= 0.01, kernel= 'rbf')

'''
'''
Creating the multiple Stacking classifier with ridge,knn,svc,lr because based of our data they obtained the best accuracy and faster computing
when using gridsearch for everymodel with tuning (X data and y the label cv=5)
'''
#multi_layer_classificator.fit(X, y)
    final_layer = StackingClassifier(
    estimators=[('ridge', RidgeClassifier(alpha= 0.1)),
                ('knn', KNeighborsClassifier(metric= 'euclidean', n_neighbors= 1, weights= 'uniform'))],
    final_estimator=SVC(C= 1, gamma= 0.01, kernel= 'rbf')
    )

    multi_layer_classificator = StackingClassifier(
    estimators=[('SVC',SVC(C= 1, gamma= 0.01, kernel= 'rbf')),
#                ('mlp', MLPClassifier(activation= 'tanh', alpha= 0.0001, hidden_layer_sizes= (10, 30, 10), learning_rate= 'constant', solver= 'adam')),
                ('Lr', LogisticRegression(C= 100, penalty= 'l2', solver= 'newton-cg'))],
    final_estimator=final_layer
    )


'''
Starting the pSO COMPUTING
'''
from scipy import optimize
def f_per_particle(m, alpha):
    """Computes for the objective function per particle

    Inputs
    ------
    m : numpy.ndarray
        Binary mask that can be obtained from BinaryPSO, will
        be used to mask features.
    alpha: float (default is 0.5)
        Constant weight for trading-off classifier performance
        and number of features

    Returns
    -------
    numpy.ndarray
        Computed objective function
    """
    total_features = 2151
    X_subset=X
    # Get the subset of the features from the binary mask
    
    # Perform classification and store performance in P
    multi_layer_classificator.fit(X_subset, y)
    P = (multi_layer_classificator.predict(X_subset) == y).mean()
    # Compute for the objective function
    j = (alpha * (1.0 - P)
        + (1.0 - alpha) * (1 - (X_subset.shape[1] / total_features)))

    return j

def f(x, alpha=0.5):
    """Higher-level method to do classification in the
    whole swarm.

    Inputs
    ------
    x: numpy.ndarray of shape (n_particles, dimensions)
        The swarm that will perform the search

    Returns
    -------
    numpy.ndarray of shape (n_particles, )
        The computed loss for each particle
    """
    n_particles = x.shape[0]
    j = [f_per_particle(x[i], alpha) for i in range(n_particles)]
    return np.array(j)

# Initialize swarm, arbitrary
options = {'c1': 0.5, 'c2': 0.5, 'w':0.9, 'k': 30, 'p':2}
import pyswarms as ps

# Call instance of PSO
dimensions = 2151 # dimensions should be the number of features
optimizer.reset()

optimizer = ps.discrete.BinaryPSO(n_particles=50, dimensions=dimensions, options=options)

# Perform optimization
cost, pos = optimizer.optimize(f, iters=10, verbose=10)

#get the new subset
X_selected = X[:,pos==1]  # subset






'''
Creatinon of the second stacked generalization for prediction purpose important to replace LR to MLP because LR is only for binary
by the way as the model is composed of MLP u can fit many times 
'''


from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor

#extract the true variables
y1 = dataset.iloc[:,2152:2153].values
#important to scale the true variables
sc_y = StandardScaler()
y1 = sc_y.fit_transform(y1)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_selected, y1, test_size = 0.25, random_state = 1)

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

y_pred = multi_layer_regressor.predict(X_test)
y_predtrain = multi_layer_regressor.predict(X_train)


#Verify how good is the model 

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
#coef determination
Rtrain=r2_score(y_train,y_predtrain)
Rtest=r2_score(y_test,y_pred)
#mean squarre error
Mse_train = mean_squared_error(y_train,y_predtrain)
Msetest=mean_squared_error(y_test,y_pred)