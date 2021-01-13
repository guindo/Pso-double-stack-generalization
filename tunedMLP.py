# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 12:22:14 2020

@author: Mahamed
"""

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

mlp_gs = MLPClassifier(max_iter=100)


parameter_space = {
    'hidden_layer_sizes': [(10,30,10),(20,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}
grid_search = GridSearchCV(mlp_gs, parameter_space, n_jobs=-1, cv=5, scoring = 'accuracy')
grid_search.fit(X, y) # X is train samples and y is the corresponding labels
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)