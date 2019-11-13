import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.metrics import cohen_kappa_score
from metafeatures.core.object_analyzer import analyze_pd_dataframe
from metafeatures.meta_functions.entropy import Entropy
from metafeatures.meta_functions import basic as basic_meta_functions
from metafeatures.meta_functions.pearson_correlation import PearsonCorrelation
from metafeatures.meta_functions.mutual_information import MutualInformation
from metafeatures.meta_functions.spearman_correlation import SpearmanCorrelation
from metafeatures.post_processing_functions.basic import Mean, StandardDeviation, Skew, Kurtosis
from metafeatures.post_processing_functions.basic import NonAggregated
from metafeatures.core.engine import metafeature_generator

class autoBaggingClassifier:
        
    def __init__(self,state):
        if state == 'create' or state  == 'predict':
            self.type = state
        else:
            self.type = 'error'
        self.base_estimators = {    'Decision Tree (max_depth=1)': DecisionTreeClassifier(max_depth=1,random_state=0),
                           'Decision Tree (max_depth=2)': DecisionTreeClassifier(max_depth=2,random_state=0),
                           'Decision Tree (max_depth=3)': DecisionTreeClassifier(max_depth=3,random_state=0),
                           'Decision Tree (max_depth=4)': DecisionTreeClassifier(max_depth=4,random_state=0),
                           'Naive Bayes': GaussianNB(),
                           'Majority Class': DummyClassifier(random_state=0)}
        self.grid = ParameterGrid({"n_estimators" : [50,100,200],
                           "bootstrap" : [True, False]
                         })

    def fit(self,
            file_name_datasets,     # Nome de todos os ficheiros .CSV
            target_names,           # Nome dos targets de todas os datasets
            meta_functions,         # Meta - Functions
            post_processing_steps): 
        # Por cada file abrir o csv e tirar para um array de DataFrames
        self.datasets = []          # Vai conter todos os Datasets
        self.bagging_workflows = [] # Vai conter todos os Bagging workflows
        if self.type == 'create':
            x_meta = []      # Vai conter todas as Metafeatures, uma linha um exemplo de um algoritmo com um certo tipo de parametros
            y_meta = []
            for file_name, target in zip(file_name_datasets, target_names):
                print(file_name)
                dataset = pd.read_csv(file_name)
                self.datasets.append(dataset)
                if self._validateDataset(dataset,target):
                    # MetaFeatures
                    meta_features_estematic = self._metafeatures(dataset,target,meta_functions,post_processing_steps)
                    
                    # É necessário dividir o dataset em exemplos e os targets
                    X = SimpleImputer().fit_transform(dataset.drop(target,axis=1))
                    y = dataset[target]
                    scoring = 'accuracy'
                    # Criar base-models
                    for params in self.grid: # Combinações de Parametros
                        meta_features = meta_features_estematic.copy()
                        Rank = {}
                        for base_estimator in self.base_estimators: # Em cada combinação dos parametros adiciona uma feature por algoritmo base
                            # Criar modelo
                            bagging_workflow = BaggingClassifier(base_estimator=self.base_estimators[base_estimator],
                                              random_state=0,
                                              **params)

                            # Avaliar Algoritmos
                            #kfold = KFold(n_splits=4, random_state=0)
                            #cv_results = cross_val_score(bagging_workflow, X, y, cv=kfold, scoring=scoring)
                            #print(base_estimator," --> Score: %0.2f (+/-) %0.2f)" % (cv_results.mean(), cv_results.std() * 2))
                            #Rank[base_estimator] = cv_results.mean()
                            
                            
                            # Treinar o modelo
                            bagging_workflow.fit(X,y)
                            # Adicionar ao array de metafeatures, landmark do algoritmo atual
                            Rank[base_estimator] = cohen_kappa_score(y,bagging_workflow.predict(X))
                            print(base_estimator, " --> Score: %0.3f)" % Rank[base_estimator])
                            # Adicionar a lista de Workflows
                            self.bagging_workflows.append(bagging_workflow)

                            # Adicionar ao array de metafeatures, as caracteriticas dos baggings workflows
                            meta_features['bootstrap'] = np.multiply(params['bootstrap'],1)
                            meta_features['n_estimators'] = params['n_estimators']

                        #print(sorted(Rank, key=Rank.__getitem__ ,reverse=True))
                        i = 1
                        for base_estimator in sorted(Rank, key=Rank.__getitem__ ,reverse=True):
                            meta_features['Algorithm:' + base_estimator] = i
                            Rank[base_estimator] = i
                            i=i+1
                            array_rank = []
                        for value in Rank.values():
                            array_rank.append(value)
                        #print(array_rank)
                        y_meta.append(array_rank)
                        x_meta.append(meta_features)   # Este array a adiconar contem as metafeatures do dataset e o scores do algoritmo base a testar

            # Meta Data é a junção de todas as metafeatures com os scores dos respeticos algoritmos base
            self.meta_data = pd.DataFrame(x_meta)
            self.meta_data.to_csv('meta_data.csv') # Guardar Meta Data num ficheiro .CSV
            

            # Criar o target para o Meta Data
            print('-------X--------')
            print('----->Exel<-----')
            print('-------Y--------')
            y_meta = np.array(y_meta)
            print(y_meta)

            # Criar o Meta Model XGBOOST
            meta_model = xgb.XGBRegressor(  colsample_bytree = 0.3,
                                            learning_rate = 0.1,
                                            max_depth = 5,
                                            alpha = 10,
                                            n_estimators = 100)
            self.meta_model = MultiOutputRegressor(meta_model)

            # Aplicar Learning algorithm
            
            #meta_model.fit(self.meta_data,y_meta)

            # Avaliar meta_model

            # Get dataset de teste
            #dataset = pd.read_csv('./datasets/test/weatherAUS.csv')
            #dataset.drop('RISK_MM', axis=1)
            #target_test = 'RainTomorrow'
            #X_test = self._metafeatures(dataset.drop(target_test,axis=1),target,meta_functions,post_processing_steps)
            #preds = meta_model.predict(X_test)

            #print(preds)

            return self
    def predict(self, dataset, targetname):
        if self.type == 'predict':
            if self._validateDataset(dataset,targetname):
                print("Erro, error não é um problema de Classifier")

    def _metafeatures(self, dataset,target,meta_functions,post_processing_steps):

        metafeatures_values, metafeatures_names = metafeature_generator(
        dataset, # Pandas Dataframe
        [target], # Name of the target variable
        meta_functions, # Metafunctions
        post_processing_steps # Post-processing functions
        )
        
        metafeatures_values = np.array(metafeatures_values)
        metafeatures_names = np.array(metafeatures_names)
        meta_features = dict(zip(metafeatures_names,metafeatures_values))

        meta_features['Number of Examples'] = dataset.shape[0]
        meta_features['Number of Features'] = dataset.shape[1]
        meta_features['Number of Classes'] = dataset[target].unique().shape[0]
        return meta_features
    
    def _bb(self,target, # Target name ?
                preds, # Predicts na training data
                data, # training data
                cutPoint): # ratio of the total n umber of models to cut off
        targets = data[target]
        print(targets)
        #
        # TO-DO
        #
        
        
    def _mdsq(self,target, # Target name ?
                preds, # Predicts na training data
                data, # training data
                cutPoint): # ratio of the total n umber of models to cut off
        targets = data[target]
        print(targets)
        #prunedN = min( data[target].shape[1] - (data[target].shape[1] * cutPoint))
        #
        # TO-DO
        #    
    
    def _validateDataset(self,dataset,targetname):
        if dataset[targetname].dtype != 'object':
            if sorted(dataset[targetname].unique()) != [0,1]:
                print("Não é válido o Dataset")
                return False
        #print("True, é valido")
        return True



# MAIN FUNCTION 
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

post_processing_steps = [Mean(),
                         StandardDeviation(),
                         Skew(),
                         Kurtosis()]


meta_functions = [Entropy(),
                  PearsonCorrelation(),
                  MutualInformation(),
                  SpearmanCorrelation(),
                  basic_meta_functions.Mean(),
                  basic_meta_functions.StandardDeviation(),
                  basic_meta_functions.Skew(),
                  basic_meta_functions.Kurtosis()]


model = autoBaggingClassifier('create')
model.fit(FileNameDataset,TargetNames, meta_functions, post_processing_steps)
