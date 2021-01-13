# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 11:17:06 2020

@author: Mahamed
"""

'''
Support vector machine with classification
parameter set by cross validation
'''
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
tuned_params = [
        {'kernel':['rbf'], 'gamma':[1e-1, 1e-2,1e-3,1e-4], 'C':[1,10,100,1000]},
        {'kernel':['linear'], 'C':[1,10,20,100]},
        {'kernel':['sigmoid'], 'C':[1,10,100,1000]}
        ]



grid_search = GridSearchCV(estimator = SVC(),
                           param_grid = tuned_params,
                           scoring = 'accuracy',
                           cv = 5,
                           n_jobs = -1)
grid_search.fit(X, y)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)


#scores = ['precision' , 'recall', average:micro]
#
#for score in scores:
#    print("#Tuning hyper-parameters for %s" % score)
#    print()
#    
#    clfsvc = GridSearchCV(SVC(), tuned_params, cv=5,
#                          scoring =score)
#    clfsvc.fit(X,y)
#    
#    print("Best parameters set found :")
#    print()
#    print(clfsvc.best_params_)
#    print()
#    print()