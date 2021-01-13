# Pso-double-stack-generalization
This is a new framework combining feature selection and Regression modeling for NIR data
1)First step creating the multiple stack generalization for classification purpose(FSGC)

  a)Find all the best parameters for different models through computing Hyperparameters tunning (TunedKnn,tunedSVC,tunedRf,tunedRidge.....all files started with tuned)
  b) choose all the best models and create the First multiple Stacked Generalization for Classification purpose(FSGC)(check pso-generalized file)

2)Computing PSO and call the FSGC as model and find the best parameter for particle, iteration and alpha(check pso-generalized file)


3)  Creating the Second multiple Stack Generalization Regressor we call (SSGR)(check pso-generalized file)
   
   

