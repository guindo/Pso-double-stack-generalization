# Pso-double-stack-generalization
This is a new framework combining feature selection and Regression modeling for NIR data
1)First step creating the multiple stack generalization for classification purpose(FSGC)

  a)Find all the best parameters for different models through computing Hyperparameters tunning (TunedKnn,tunedSVC,tunedRf,tunedRidge.....all files started with tuned)
  b) choose all the best models and create the First multiple Stacked Generalization for Classification purpose(FSGC)(check pso-generalized file)
    
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


2)Computing PSO and call the FSGC as model and find the best parameter for particle, iteration and alpha(check pso-generalized file)
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

3)  Creating the Second multiple Stack Generalization Regressor we call (SSGR)(check pso-generalized file)
       
       
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


