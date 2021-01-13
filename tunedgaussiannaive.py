# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 12:48:12 2020

@author: Mahamed
"""

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV


gnb = GaussianNB()



param_grid = {
    
}
grid_search = GridSearchCV(estimator = gnb, param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 2, scoring = 'accuracy')

grid_search.fit(X,y)


best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)
