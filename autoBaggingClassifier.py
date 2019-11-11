import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from metafeatures.core.object_analyzer import analyze_pd_dataframe
from metafeatures.meta_functions.entropy import Entropy
from metafeatures.meta_functions import basic as basic_meta_functions
from metafeatures.post_processing_functions.basic import Mean, StandardDeviation, Skew, Kurtosis
from metafeatures.meta_functions.pearson_correlation import PearsonCorrelation
from metafeatures.meta_functions.mutual_information import MutualInformation
from metafeatures.meta_functions.spearman_correlation import SpearmanCorrelation
from metafeatures.post_processing_functions.basic import NonAggregated
from metafeatures.core.engine import metafeature_generator


class autoBaggingClassifier:

    base_estimators = {    'Decision Tree (max_depth=1)': DecisionTreeClassifier(max_depth=1),
                           'Decision Tree (max_depth=2)': DecisionTreeClassifier(max_depth=2),
                           'Decision Tree (max_depth=3)': DecisionTreeClassifier(max_depth=3),
                           'Naive Bayes': GaussianNB(),
                           'Majority Class': DummyClassifier()}
    grid = ParameterGrid({"n_estimators" : [50,100,200],
                           "bootstrap" : [True, False]
                         })
                          
    def __init__(self,state):
        if state == 'create' or state  == 'predict':
            self.type = state
        else:
            self.type = 'error'

    def fit(self,
            file_name_datasets,
            target_names,
            meta_functions_categorical,
            meta_functions_numerical,
            post_processing_steps):
        # Por cada file abrir o csv e tirar para um array de DataFrames
        self.datasets = []
        self.bagging_workflows = []
        if self.type == 'create':
            meta_features = []
            for file_name, target in zip(file_name_datasets, target_names):
                print(file_name)
                dataset = pd.read_csv(file_name)
                self.datasets.append(dataset)
                if self._validateDataset(dataset,target):
                    # MetaFeatures
                    meta_features_estematic = self._metafeatures(dataset,[target],meta_functions_categorical,meta_functions_numerical,post_processing_steps)
                    meta_features_estematic['Number of Examples'] = dataset.shape[0]
                    meta_features_estematic['Number of Features'] = dataset.shape[1]
                    meta_features_estematic['Number of Classes'] = dataset[target].unique().shape[0]
                    # Falta as previções
                    # Então
                    # É necessário dividir os exemplos e os targets
                    X = SimpleImputer().fit_transform(dataset.drop(target,axis=1))
                    y = dataset[target]
                    scoring = 'accuracy'
                    # Criar os modelos base
                    for params in self.grid: # Combinações de Parametros
                        # Criar uma Copia das metafeatures
                        meta_features_final = meta_features_estematic.copy()
                        for base_estimator in self.base_estimators: # em cada combinação dos parametros adiciona uma feature por algoritmo base
                            # Criar modelo
                            print(base_estimator)
                            #print(self.base_estimators[base_estimator])
                            bagging_workflow = BaggingClassifier(base_estimator=self.base_estimators[base_estimator],
                                              random_state=0,
                                              **params)
                            #Avaliar Algoritmos
                            kfold = KFold(n_splits=5, random_state=0)
                            cv_results = cross_val_score(bagging_workflow, X, y, cv=kfold, scoring=scoring)

                            # Adicionar a lista de Workflows
                            self.bagging_workflows.append(bagging_workflow)
                            # adicionar ao array, uma previção do algoritmo atual
                            print("Score: %0.2f (+/-) %0.2f)" % (cv_results.mean(), cv_results.std() * 2))                        
                            meta_features_final['Algorithm: ' + base_estimator]  = cv_results.mean() # = bagging_workflow.score() ou cross validation
                        
                        meta_features.append(meta_features_final)   # Este array contem as metafeatures criadas e as previções dos algoritmos base


                    #Fit & Validate
                    
                    

                    
            self.meta_data = pd.DataFrame(meta_features)
            self.meta_data.to_csv('meta_data.csv')
            return self

    def _metafeatures(self, dataset,target,meta_functions_categorical,meta_functions_numerical,post_processing_steps):
            
        #Instantiate metafunctions and post-processing functions
        entropy = Entropy()
        correlation = PearsonCorrelation()
        mutual_information =  MutualInformation()
        scorrelation = SpearmanCorrelation()
        _mean = Mean()
        _sd = StandardDeviation()
        _nagg = NonAggregated()
        #Run experiments
        metafeatures_values, metafeatures_names = metafeature_generator(
        dataset, # Pandas Dataframe
        target, # Name of the target variable
        [mutual_information, entropy, correlation, scorrelation], # Metafunctions
        [_mean, _sd, _nagg] # Post-processing functions
        )
        
        #print(metafeatures_names)
        
        metafeatures_values = np.array(metafeatures_values)
        #print(metafeatures_values)
        metafeatures_names = np.array(metafeatures_names)
        #meta_features = pd.DataFrame(columns = metafeatures_names)
        meta_featureExample = dict(zip(metafeatures_names,metafeatures_values))
        return meta_featureExample
        
        

    def predict(self, dataset, targetname):
        if self.type == 'predict':
            if self._validateDataset(dataset,targetname):
                print("Erro, error não é um problema de Classifier")

    def _validateDataset(self,dataset,targetname):
        if dataset[targetname].dtype != 'object':
            if sorted(dataset[targetname].unique()) != [0,1]:
                print("False, não é válido")
                return False
        #print("True, é valido")
        return True

TargetNames = []
FileNameDataset = []

FileNameDataset.append('./datasets/heart.csv')
TargetNames.append('target')
FileNameDataset.append('./datasets/titanic.csv')
TargetNames.append('Survived')
#FileNameDataset.append('./datasets/categoricalfeatureencoding.csv')
#TargetNames.append('target')
#FileNameDataset.append('./datasets/sanfranciscocrime_split.csv')
#TargetNames.append('Category')

# Metafeatures
post_processing_steps = {'mean': Mean(),
                         'std': StandardDeviation(),
                         'skew': Skew(),
                         'kurtosis': Kurtosis()}
meta_functions_categorical = {'entropy': Entropy()}
meta_functions_numerical = {'mean': basic_meta_functions.Mean(),
                            'std': basic_meta_functions.StandardDeviation(),
                            'skew': basic_meta_functions.Skew(),
                            'kurtosis': basic_meta_functions.Kurtosis()}
#



model = autoBaggingClassifier('create')
model.fit(FileNameDataset,TargetNames, meta_functions_categorical, meta_functions_numerical, post_processing_steps)

