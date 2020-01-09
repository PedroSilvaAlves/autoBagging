autoBagging Python
------------------------------
ML (Machine Learning) & Mtl (MetaLearning)

Um Sistema autoML que é capaz de recomendar modelos de Bagging num certo tipo de dataset.

autoBaggingClassifier: Cria um modelo ensemble Bagging recomendado a um certo tipo de problema de Classificação

autoBaggingRegression: Cria um modelo ensemble Bagging recomendado a um certo tipo de problema de Regressão

Example:

.. code-block:: bash
  
  
  post_processing_steps = [Mean(),
                         StandardDeviation(),
                         Skew(),
                         Kurtosis()]


  meta_functions = [Entropy(),
                  MutualInformation(),
                  SpearmanCorrelation(),
                  basic_meta_functions.Mean(),
                  basic_meta_functions.StandardDeviation(),
                  basic_meta_functions.Skew(),
                  basic_meta_functions.Kurtosis()]
                  
                  
  autoBagging = autoBaggingRegressor(meta_functions,post_processing_steps)
  autoBagging.fit(Datasets, TargetNames)
  RecommendedBagging = autoBagging.predict(Dataset,TargetName)
