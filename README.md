# autoBagging Python

## ML (Machine Learning) & Mtl (MetaLearning)
Um autoML System que seja capaz de avaliar modelos de Bagging num certo tipo de dataset.

autoBaggingClassifier: Cria um modelo ensemble Bagging recomendado a um certo tipo de problema de Classificação

autoBaggingRegression: Cria um modelo ensemble Bagging recomendado a um certo tipo de problema de Regressão

### Example:
autoBagging = autoBaggingRegressor(meta_functions,post_processing_steps)
autoBagging.fit(FileNameDatasets, TargetNames)
RecommendedBagging = autoBagging.predict(Dataset,TargetName)
