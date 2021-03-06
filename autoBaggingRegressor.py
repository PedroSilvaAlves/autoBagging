import numpy as np
import pandas as pd
import xgboost as xgb
import math
import joblib
import warnings
import time
import sys
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from metafeatures.core.object_analyzer import analyze_pd_dataframe
from metafeatures.meta_functions.entropy import Entropy
from metafeatures.meta_functions import basic as basic_meta_functions
from metafeatures.meta_functions.pearson_correlation import PearsonCorrelation
from metafeatures.meta_functions.mutual_information import MutualInformation
from metafeatures.meta_functions.spearman_correlation import SpearmanCorrelation
from metafeatures.post_processing_functions.basic import Mean, StandardDeviation, Skew, Kurtosis
from metafeatures.post_processing_functions.basic import NonAggregated
from metafeatures.core.engine import metafeature_generator
from statistics import mean
from deslibautobagging.des.knora_e import KNORAE
from deslibautobagging.dcs.ola import OLA

#from DESIP.DESIP import DESIP

class autoBaggingRegressor(BaseEstimator):

    def __init__(self, silence=True):
        super().__init__()
        self.silence=silence
        self.meta_functions = [Entropy(),
                  MutualInformation(),
                  SpearmanCorrelation(),
                  basic_meta_functions.Mean(),
                  basic_meta_functions.StandardDeviation(),
                  basic_meta_functions.Skew(),
                  basic_meta_functions.Kurtosis()]
        self.post_processing_steps = [Mean(),
                         StandardDeviation(),
                         Skew(),
                         Kurtosis()]              
        self.base_estimators = {'Decision Tree (max_depth=1)': DecisionTreeRegressor(max_depth=1, random_state=0),
                                #'Decision Tree (max_depth=2)': DecisionTreeRegressor(max_depth=2, random_state=0),
                                #'Decision Tree (max_depth=3)': DecisionTreeRegressor(max_depth=3, random_state=0),
                                'Decision Tree (max_depth=None)': DecisionTreeRegressor(random_state=0),
                                }
        self.estimators_switcher = {'Decision Tree (max_depth=1)': 1,
                                    #'Decision Tree (max_depth=2)': 2,
                                    #'Decision Tree (max_depth=3)': 3,
                                    'Decision Tree (max_depth=None)': 2}
        self.estimators_switcher2 = {1 : 'Decision Tree (max_depth=1)',
                                     2 : 'Decision Tree (max_depth=None)'}
        self.bagging_grid = ParameterGrid({"n_estimators": [50, 100,200]})
        self.pruning = ParameterGrid({'pruning_method' : [0,1],
                                      'pruning_cp': [0.25,0.5,0.75]})
        self.DStechique = ParameterGrid({ 'ds' : [-1,0,1]})
    
    def fit(self,
            datasets,                # Lista com datasets
            target_names):           # Nome dos targets de todas os datasets
            
        # Por cada file abrir o csv e tirar para um array de DataFrames
        x_meta = []     # Vai conter todas as Meta-features, uma linha um exemplo de um algoritmo com um certo tipo de parametros
        y_meta = []     # Vai conter o Meta-Target, em cada linha têm a avaliação de 1-n de cada algoritmo
                        # + parametros do bagging workflow
        ndataset = 0
        t = time.time()
        for dataset, target in zip(datasets, target_names):  # Percorre todos os datasets para treino do meta-model
           if self._validateDataset(dataset, target):
                ndataset= ndataset + 1
                if not self.silence:
                    print("________________________________________________________________________")
                    print("Dataset nº ", ndataset)
                    print("Shape: {}(examples, features)".format(np.shape(dataset)))
                # Number of Bagging Workflows
                indexBagging = 1
                indexMaxBagging = 0
                for params in self.bagging_grid:                                # Combinações de Parametros
                    for DS in self.DStechique:                                  # Combinações do Dynamic Selection
                        for pruning in self.pruning:                            # Combinações dos Pruning Methods
                            for base_estimator in self.base_estimators:         # Combinação dos algoritmos base
                                # Skip Useless Combinations
                                if(self._skipCombination(pruning)):
                                    continue
                                indexMaxBagging = indexMaxBagging + 1
                # Time
                t = time.time()
                # Drop Categorial features, DecisionTree do sklearn não aceitam
                for f in dataset.columns:
                    if dataset[f].dtype == 'object':
                        if type(dataset[f]) != pd.core.series.Series:
                            dataset = dataset.drop(columns=f, axis=1)
                        else:
                            dataset[f] = pd.to_numeric(dataset[f], errors='coerce')
                
                # MetaFeatures
                meta_features_estematic = self._metafeatures(
                    dataset, target, self.meta_functions, self.post_processing_steps)
                
                # Convert +/- Inf to NaN
                dataset.replace(np.inf, np.nan)
                dataset.replace(-np.inf, np.nan)
                # Drop Columns with all NaN values
                dataset = dataset.dropna(axis=1,how ='all')
                # Drop examples with some Nan Values
                dataset = dataset.dropna(axis = 0,how = 'any') 
                dataset = dataset.reset_index(drop=True)
                
                # Dividir o dataset em exemplos e os targets
                simpleImputer = SimpleImputer()
                X = simpleImputer.fit_transform(dataset.drop(target, axis=1))
                y = dataset[target]
                
                # Criar base-models
                for params in self.bagging_grid:  # Combinações de Parametros
                    for DS in self.DStechique:
                        for pruning in self.pruning:
                            for base_estimator in self.base_estimators:  # Combinação dos algoritmos base
                                # Skip Useless Combinations
                                if(self._skipCombination(pruning)):
                                    continue
                                sys.stdout.write('\r'+ "Creating Baggings Workflows... [{}/{}]".format(indexBagging,indexMaxBagging))
                                meta_features = meta_features_estematic.copy() # Meta-features do dataset só é criado uma vez
                                
                                # Cross Validation 4 Folds
                                Ranks = []
                                kf = KFold(n_splits=4)
                                for train_index, test_index in kf.split(X):
                                    # Separar set do Fold atual
                                    X_train, X_test = X[train_index], X[test_index]
                                    y_train, y_test = y[train_index], y[test_index]
                                    y_train = y_train.reset_index(drop=True)
                                    y_test = y_test.reset_index(drop=True)
                                    
                                    # Criar modelo
                                    bagging_workflow = BaggingRegressor(base_estimator=self.base_estimators[base_estimator],
                                                                            random_state=0, n_jobs= -1,
                                                                            **params)
                                    # Treinar o modelo
                                    bagging_workflow.fit(X_train, y_train)

                                    predictions = []
                                    # PRUNING METHODS
                                    if pruning['pruning_method'] == 1 and pruning['pruning_cp'] != 0:
                                        # Criar predicts para todos os base-model
                                        for estimator, features in zip(bagging_workflow.estimators_,bagging_workflow.estimators_features_):
                                            predictions.append(estimator.predict(X_train[:, features]))
                                        re_index= self._re(y_train, predictions, X_train, pruning['pruning_cp'])
                                        # Pruning the bagging_workflow
                                        estimators = []
                                        for i in re_index.values():
                                            estimators.append(bagging_workflow.estimators_[i])
                                        bagging_workflow.estimators_ = estimators
                                    else:
                                        pruning['pruning_cp'] = 0
                                    # Dynamic Select
                                    if DS['ds'] == -1:
                                        bagging_workflow = KNORAE(bagging_workflow, random_state=0)
                                        bagging_workflow.fit(X_train,y_train)
                                        Rank_fold = mean_squared_error(bagging_workflow.predict(X_test),y_test)

                                    else:
                                        if DS['ds'] == 1:
                                            bagging_workflow = OLA(bagging_workflow,random_state=0)
                                            bagging_workflow.fit(X_train,y_train)
                                            Rank_fold = mean_squared_error(bagging_workflow.predict(X_test),y_test)
                                        else:
                                            # Criar landmark do baggingworkflow atual
                                            Rank_fold = mean_squared_error(bagging_workflow.predict(X_test),y_test)
                                    Ranks.append(Rank_fold)
                                
                                # Adicionar ao array de metafeatures, as caracteriticas dos baggings workflows
                                meta_features['n_estimators'] = params['n_estimators']
                                meta_features['pruning_method'] = pruning['pruning_method']
                                meta_features['pruning_cp'] = pruning['pruning_cp']
                                meta_features['ds'] = DS['ds']
                                meta_features['Algorithm'] = self.estimators_switcher[base_estimator]
                                # Este array é o meta target do score do algoritmo
                                y_meta.append(float(mean(Ranks)))
                                
                                # Este array contem as várias metafeatures do dataset e o scores do algoritmo base/parametros a testar
                                x_meta.append(meta_features)
                                indexBagging = indexBagging + 1
                sys.stdout.write('\r'+ "                                                ")
                sys.stdout.write('\r'+ "Elapsed: %.2f seconds\n"  % (time.time() - t))
                # Backup Data
                pd.DataFrame(x_meta).to_csv("./metadata/Meta_Data_Regressor_backup.csv")
                pd.DataFrame(y_meta).to_csv("./metadata/Meta_Target_Regressor_backup.csv")
        if not self.silence:
            print("________________________________________________________________________")
        # Meta Data é a junção de todas as metafeatures com os scores dos respeticos algoritmos base
        self.meta_data = pd.DataFrame(x_meta)
        self.meta_target = np.array(y_meta)
        # Guardar Meta Data num ficheiro .CSV
        self.meta_data.to_csv('./metadata/Meta_Data_Regressor.csv')
        pd.DataFrame(self.meta_target).to_csv('./metadata/Meta_Target_Regressor.csv')
        if not self.silence:
            print("Meta-Data Created and Saved.")
        # Tratar dos dados para entrar no XGBOOST
        for f in self.meta_data.columns:
            if self.meta_data[f].dtype == 'object':
                lbl = LabelEncoder()
                lbl.fit(list(self.meta_data[f].values))
                self.meta_data[f] = lbl.transform(
                    list(self.meta_data[f].values))

        self.meta_data.fillna((-999), inplace=True)
        self.meta_data = np.array(self.meta_data)
        self.meta_data = self.meta_data.astype(float)
        if not self.silence:
            print("Constructing Meta-Model:")
        # Criar o Meta Model XGBOOST
        self.meta_model = xgb.XGBRegressor(objective="reg:squarederror",
                                        colsample_bytree=0.3,
                                        learning_rate=0.1,
                                        max_depth=6,
                                        alpha=1,
                                        n_estimators=100,
                                        n_jobs=-1)

        # Aplicar Learning algorithm
        self.meta_model.fit(self.meta_data, self.meta_target)
        self.is_fitted = True
        return self
    
    def load_fit(self, meta_data, meta_target):
        # Meta Data é a junção de todas as metafeatures com os scores dos respeticos algoritmos base
        self.meta_data= meta_data.drop(meta_data.columns[0], axis=1)
        self.meta_data = self.meta_data[[
        'Features.Entropy.Mean',
        'Features.Entropy.StandardDeviation',
        'Features.Entropy.Skew',
        'Features.Entropy.Kurtosis',
        'Features.MutualInformation.Mean',
        'Features.MutualInformation.StandardDeviation',
        'Features.MutualInformation.Skew',
        'Features.MutualInformation.Kurtosis',
        'Features.SpearmanCorrelation.Mean',
        'Features.SpearmanCorrelation.StandardDeviation',
        'Features.SpearmanCorrelation.Skew',
        'Features.SpearmanCorrelation.Kurtosis',
        'FeaturesLabels.SpearmanCorrelation.Mean',
        'FeaturesLabels.SpearmanCorrelation.StandardDeviation',
        'FeaturesLabels.SpearmanCorrelation.Skew',
        'FeaturesLabels.SpearmanCorrelation.Kurtosis',
        'Features.Mean.Mean',
        'Features.Mean.StandardDeviation',
        'Features.Mean.Skew',
        'Features.Mean.Kurtosis',
        'Features.StandardDeviation.Mean',
        'Features.StandardDeviation.StandardDeviation',
        'Features.StandardDeviation.Skew',
        'Features.StandardDeviation.Kurtosis',
        'Features.Skew.Mean',
        'Features.Skew.StandardDeviation',
        'Features.Skew.Skew',
        'Features.Skew.Kurtosis',
        'Features.Kurtosis.Mean',
        'Features.Kurtosis.StandardDeviation',
        'Features.Kurtosis.Skew',
        'Features.Kurtosis.Kurtosis',
        'FeaturesLabels.MutualInformation.Mean',
        'FeaturesLabels.MutualInformation.StandardDeviation',
        'FeaturesLabels.MutualInformation.Skew',
        'FeaturesLabels.MutualInformation.Kurtosis',
        'Number of Examples',
        'Number of Features',
        'n_estimators',
        'pruning_method',
        'pruning_cp',
        'ds',
        'Algorithm'
        ]]
        self.meta_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.meta_data.fillna((-999), inplace=True)
        
        meta_target.fillna((-0.8), inplace=True)
        meta_target = np.array(meta_target)
        self.meta_target = np.squeeze(meta_target[:,1])
        y_meta = []
        for x in np.nditer(self.meta_target):
             y_meta.append(float(x))
        self.meta_target = np.array(y_meta)
        if not self.silence:
            print("Meta-Data Loaded.")
            print("Constructing Meta-Model:")

         # Create Meta Model of XGBOOST
        self.meta_model = xgb.XGBRegressor(objective="reg:squarederror",
                                       colsample_bytree=0.7,
                                       learning_rate=0.1,
                                       max_depth=5,
                                       alpha=7,
                                       eval_metric='auc',
                                       n_estimators=100,
                                       n_jobs=-1,
                                       min_child_weight=3)
        self.meta_model = self.meta_model.fit(np.array(self.meta_data), self.meta_target)
        self.is_fitted = True
        return self

    def predict(self, dataset, target):
        if self._validateDataset(dataset, target):
            for f in dataset.columns:
                if dataset[f].dtype == 'object':
                    dataset = dataset.drop(columns=f, axis=1)

            meta_features_estematic = self._metafeatures(
                dataset, target, self.meta_functions, self.post_processing_steps)
            simpleImputer = SimpleImputer()
            X = simpleImputer.fit_transform(dataset.drop(target, axis=1))
            y = dataset[target]
            BestScore = -1
            score = 0
            RecommendedBagging = {}
            bagging_combination = []
            for params in self.bagging_grid:  # Combinações de Parametros
                    for DS in self.DStechique:
                        for pruning in self.pruning:
                            for base_estimator in self.base_estimators:  # Combinação dos algoritmos base
                                meta_features = meta_features_estematic.copy()
                                meta_features['n_estimators'] = params['n_estimators']
                                meta_features['pruning_method'] = pruning['pruning_method']
                                meta_features['pruning_cp'] = pruning['pruning_cp']
                                meta_features['ds'] = DS['ds']
                                meta_features['Algorithm'] = self.estimators_switcher[base_estimator]
                                meta_features_dic = meta_features
                                features = []
                                features.append(meta_features)
                                meta_features = pd.DataFrame(features)
                                meta_features = meta_features[[
                                    'Features.Entropy.Mean',
                                    'Features.Entropy.StandardDeviation',
                                    'Features.Entropy.Skew',
                                    'Features.Entropy.Kurtosis',
                                    'Features.MutualInformation.Mean',
                                    'Features.MutualInformation.StandardDeviation',
                                    'Features.MutualInformation.Skew',
                                    'Features.MutualInformation.Kurtosis',
                                    'Features.SpearmanCorrelation.Mean',
                                    'Features.SpearmanCorrelation.StandardDeviation',
                                    'Features.SpearmanCorrelation.Skew',
                                    'Features.SpearmanCorrelation.Kurtosis',
                                    'FeaturesLabels.SpearmanCorrelation.Mean',
                                    'FeaturesLabels.SpearmanCorrelation.StandardDeviation',
                                    'FeaturesLabels.SpearmanCorrelation.Skew',
                                    'FeaturesLabels.SpearmanCorrelation.Kurtosis',
                                    'Features.Mean.Mean',
                                    'Features.Mean.StandardDeviation',
                                    'Features.Mean.Skew',
                                    'Features.Mean.Kurtosis',
                                    'Features.StandardDeviation.Mean',
                                    'Features.StandardDeviation.StandardDeviation',
                                    'Features.StandardDeviation.Skew',
                                    'Features.StandardDeviation.Kurtosis',
                                    'Features.Skew.Mean',
                                    'Features.Skew.StandardDeviation',
                                    'Features.Skew.Skew',
                                    'Features.Skew.Kurtosis',
                                    'Features.Kurtosis.Mean',
                                    'Features.Kurtosis.StandardDeviation',
                                    'Features.Kurtosis.Skew',
                                    'Features.Kurtosis.Kurtosis',
                                    'FeaturesLabels.MutualInformation.Mean',
                                    'FeaturesLabels.MutualInformation.StandardDeviation',
                                    'FeaturesLabels.MutualInformation.Skew',
                                    'FeaturesLabels.MutualInformation.Kurtosis',
                                    'Number of Examples',
                                    'Number of Features',
                                    'n_estimators',
                                    'pruning_method',
                                    'pruning_cp',
                                    'ds',
                                    'Algorithm'
                                    ]]
                                meta_features.replace([np.inf, -np.inf], np.nan, inplace=True)
                                meta_features.fillna((-999), inplace=True)
                                bagging_combination.append(np.array(meta_features.copy()))
            bagging_combination = np.squeeze(np.array(bagging_combination))
            scores = self.meta_model.predict(bagging_combination)
            BestScore = (np.abs(scores)).argmin()
            RecommendedBagging = bagging_combination[BestScore]
            # Prints e construção do Bagging previsto
            npSize= RecommendedBagging.size
            n_estimators = int(RecommendedBagging[npSize-5])
            pruning_method = int(RecommendedBagging[npSize-4])
            pruning_cp = float(RecommendedBagging[npSize-3]/100)
            ds = int(RecommendedBagging[npSize-2])
            base_estimator = int(RecommendedBagging[npSize-1])

            # String para visualização
            if pruning_method == 1:
                pruning_method_str = 'RE'
            else:
                pruning_method_str = 'None'
            if ds > 0.5:
                    ds_str = 'KNORAE'
            else:
                if ds < -0.5:
                    ds_str = 'OLA'
                else:
                    ds_str = 'None'
            if not self.silence:    
                print("Recommended Bagging workflow: ")
                print("\tNumber of models: ", n_estimators)
                if pruning_method != 0:
                    print("\tPruning Method: ", pruning_method_str)
                    print("\tPruning CutPoint: ", pruning_cp*100)
                else:
                    print("\tPruning: ",pruning_method_str)
                print("\tDynamic Selection: ", ds_str)
                print("\tAlgorithm: ", base_estimator)

            # BaggingWorkflow
            bagging_workflow = BaggingRegressor(
                    base_estimator= self.base_estimators[self.estimators_switcher2[base_estimator]],
                    n_estimators=n_estimators,
                    random_state=0,
                    n_jobs=-1
                    )

            
            # Dividir o dataset em exemplos e os targets
            X = SimpleImputer().fit_transform(dataset.drop(target, axis=1))
            y = dataset[target]
            X_train = X
            y_train = y
            # Treinar o modelo
            bagging_workflow.fit(X_train, y_train)
            predictions = []
            if pruning_method == 1 and pruning_cp != 0:
                if not self.silence:
                    print("Waiting for RE")
                for estimator, features in zip(bagging_workflow.estimators_,bagging_workflow.estimators_features_):
                    predictions.append(estimator.predict(X_train[:, features]))
                re_index= self._re(y_train, predictions, X_train, pruning_cp)
                # Pruning the bagging_workflow
                estimators = []
                for i in re_index.values():
                    estimators.append(bagging_workflow.estimators_[i])
                bagging_workflow.estimators_ = estimators
            # Dynamic Select        
            if ds == -1:
                bagging_workflow = KNORAE(bagging_workflow, random_state=0)
                bagging_workflow.fit(X_train,y_train)
            else:
                if ds == 1:
                    bagging_workflow = OLA(bagging_workflow,random_state=0)
                    bagging_workflow.fit(X_train,y_train)
             
            return bagging_workflow
        else:
            print("Erro, não é um problema de Regressão")

    def bagging_test(self, dataset, target):
        bagging_workflow_ranks = []
        if self._validateDataset(dataset, target):
            for f in dataset.columns:
                if dataset[f].dtype == 'object':
                    dataset = dataset.drop(columns=f, axis=1)

            meta_features_estematic = self._metafeatures(
                dataset, target, self.meta_functions, self.post_processing_steps)
            simpleImputer = SimpleImputer()
            X = simpleImputer.fit_transform(dataset.drop(target, axis=1))
            y = dataset[target]

            for params in self.bagging_grid:  # Combinações de Parametros
                    for DS in self.DStechique:
                        for pruning in self.pruning:
                            for base_estimator in self.base_estimators:  # Combinação dos algoritmos base
                                # Skip Useless Combinations
                                if(self._skipCombination(pruning)):
                                    continue

                                t = time.time()
                                # Cross Validation 4 Folds
                                Ranks = []
                                kf = KFold(n_splits=4)
                                for train_index, test_index in kf.split(X):
                                    # Slit the dataset of the current Fold
                                    X_train, X_test = X[train_index], X[test_index]
                                    y_train, y_test = y[train_index], y[test_index]
                                    y_train = y_train.reset_index(drop=True)
                                    y_test = y_test.reset_index(drop=True)
                                    
                                    # Create BaggingClassifier Model
                                    bagging_workflow = BaggingRegressor(base_estimator=self.base_estimators[base_estimator],
                                                                            random_state=0, n_jobs= -1,
                                                                            **params)
                                    # Apply Learning Algorithm
                                    bagging_workflow.fit(X_train,y_train)
                                    
                                    predictions = []
                                    # PRUNING METHODS
                                    if pruning['pruning_method'] == 1 and pruning['pruning_cp'] != 0:
                                        # RE Method
                                        # Create predicts for each base-estimator
                                        for estimator, features in zip(bagging_workflow.estimators_,bagging_workflow.estimators_features_):
                                            predictions.append(estimator.predict(X_train[:, features]))
                                        # Calculate the index of the base-estimators that will remain in the bagging
                                        re_index = self._re(y_train, predictions, X_train, pruning['pruning_cp'])
                                        # The actual prunning of the bagging
                                        estimators = []
                                        for i in re_index.values():
                                            estimators.append(bagging_workflow.estimators_[i])
                                        bagging_workflow.estimators_ = estimators
                                    else:
                                        # No pruning method
                                        pruning['pruning_cp'] = 0
                                    
                                    # Dynamic Select
                                    if DS['ds'] == -1:
                                        # KNORAE
                                        bagging_workflow = KNORAE(bagging_workflow, random_state=0)
                                        bagging_workflow.fit(X_train,y_train)
                                    else:
                                        if DS['ds'] == 1:
                                            # OLA
                                            bagging_workflow = OLA(bagging_workflow,random_state=0)
                                            bagging_workflow.fit(X_train,y_train)
                                        
                                    Rank_fold = mean_squared_error(bagging_workflow.predict(X_test),y_test)
                                    # Save the current bagging rank
                                    Ranks.append(Rank_fold)
                                    
                                # y_meta is the Meta Target Array that contains the score of the bagging workflow
                                sys.stdout.write('\r'+ "Elapsed: %.2f seconds\n"  % (time.time() - t))
                                bagging_workflow_ranks.append(float(mean(Ranks)))
        return bagging_workflow_ranks

    def metamodel_test(self,dataset,target):
        bagging_workflow_ranks = []
        if self._validateDataset(dataset, target):
            for f in dataset.columns:
                if dataset[f].dtype == 'object':
                    dataset = dataset.drop(columns=f, axis=1)

            meta_features_estematic = self._metafeatures(
                dataset, target, self.meta_functions, self.post_processing_steps)
            simpleImputer = SimpleImputer()
            X = simpleImputer.fit_transform(dataset.drop(target, axis=1))
            y = dataset[target]

            bagging_combination = []
            for params in self.bagging_grid:  # Combinações de Parametros
                    for DS in self.DStechique:
                        for pruning in self.pruning:
                            for base_estimator in self.base_estimators:  # Combinação dos algoritmos base
                                # Skip Useless Combinations
                                if(self._skipCombination(pruning)):
                                    continue

                                #t = time.time()
                                meta_features = meta_features_estematic.copy()
                                meta_features['n_estimators'] = params['n_estimators']
                                meta_features['pruning_method'] = pruning['pruning_method']
                                meta_features['pruning_cp'] = pruning['pruning_cp']
                                meta_features['ds'] = DS['ds']
                                meta_features['Algorithm'] = self.estimators_switcher[base_estimator]
                                meta_features_dic = meta_features
                                features = []
                                features.append(meta_features)
                                meta_features = pd.DataFrame(features)
                                meta_features = meta_features[[
                                    'Features.Entropy.Mean',
                                    'Features.Entropy.StandardDeviation',
                                    'Features.Entropy.Skew',
                                    'Features.Entropy.Kurtosis',
                                    'Features.MutualInformation.Mean',
                                    'Features.MutualInformation.StandardDeviation',
                                    'Features.MutualInformation.Skew',
                                    'Features.MutualInformation.Kurtosis',
                                    'Features.SpearmanCorrelation.Mean',
                                    'Features.SpearmanCorrelation.StandardDeviation',
                                    'Features.SpearmanCorrelation.Skew',
                                    'Features.SpearmanCorrelation.Kurtosis',
                                    'FeaturesLabels.SpearmanCorrelation.Mean',
                                    'FeaturesLabels.SpearmanCorrelation.StandardDeviation',
                                    'FeaturesLabels.SpearmanCorrelation.Skew',
                                    'FeaturesLabels.SpearmanCorrelation.Kurtosis',
                                    'Features.Mean.Mean',
                                    'Features.Mean.StandardDeviation',
                                    'Features.Mean.Skew',
                                    'Features.Mean.Kurtosis',
                                    'Features.StandardDeviation.Mean',
                                    'Features.StandardDeviation.StandardDeviation',
                                    'Features.StandardDeviation.Skew',
                                    'Features.StandardDeviation.Kurtosis',
                                    'Features.Skew.Mean',
                                    'Features.Skew.StandardDeviation',
                                    'Features.Skew.Skew',
                                    'Features.Skew.Kurtosis',
                                    'Features.Kurtosis.Mean',
                                    'Features.Kurtosis.StandardDeviation',
                                    'Features.Kurtosis.Skew',
                                    'Features.Kurtosis.Kurtosis',
                                    'FeaturesLabels.MutualInformation.Mean',
                                    'FeaturesLabels.MutualInformation.StandardDeviation',
                                    'FeaturesLabels.MutualInformation.Skew',
                                    'FeaturesLabels.MutualInformation.Kurtosis',
                                    'Number of Examples',
                                    'Number of Features',
                                    'n_estimators',
                                    'pruning_method',
                                    'pruning_cp',
                                    'ds',
                                    'Algorithm'
                                    ]]
                                meta_features.replace([np.inf, -np.inf], np.nan, inplace=True)
                                meta_features.fillna((-999), inplace=True)
                                bagging_combination.append(np.array(meta_features.copy()))
                                #sys.stdout.write('\r'+ "Elapsed: %.2f seconds\n"  % (time.time() - t))
            bagging_combination = np.squeeze(np.array(bagging_combination))
            scores = self.meta_model.predict(bagging_combination)
        return scores

    def _metafeatures(self, dataset, target, meta_functions, post_processing_steps):

        metafeatures_values, metafeatures_names = metafeature_generator(
            dataset,  # Pandas Dataframe
            [target],  # Name of the target variable
            meta_functions,  # Metafunctions
            post_processing_steps  # Post-processing functions
        )
        metafeatures_values = np.array(metafeatures_values)
        metafeatures_names = np.array(metafeatures_names)
        meta_features = dict(zip(metafeatures_names, metafeatures_values))
        
        # Inicializa as metafeatures
        meta_features['Number of Examples'] = dataset.shape[0]
        meta_features['Number of Features'] = dataset.shape[1]
        meta_features_allnames = [
        'Features.SpearmanCorrelation.Mean',
        'Features.SpearmanCorrelation.StandardDeviation',
        'Features.SpearmanCorrelation.Skew',
        'Features.SpearmanCorrelation.Kurtosis',
        'FeaturesLabels.SpearmanCorrelation.Mean',
        'FeaturesLabels.SpearmanCorrelation.StandardDeviation',
        'FeaturesLabels.SpearmanCorrelation.Skew',
        'FeaturesLabels.SpearmanCorrelation.Kurtosis',
        'Features.Mean.Mean',
        'Features.Mean.StandardDeviation',
        'Features.Mean.Skew',
        'Features.Mean.Kurtosis',
        'Features.StandardDeviation.Mean',
        'Features.StandardDeviation.StandardDeviation',
        'Features.StandardDeviation.Skew',
        'Features.StandardDeviation.Kurtosis',
        'Features.Skew.Mean',
        'Features.Skew.StandardDeviation',
        'Features.Skew.Skew',
        'Features.Skew.Kurtosis',
        'Features.Kurtosis.Mean',
        'Features.Kurtosis.StandardDeviation',
        'Features.Kurtosis.Skew',
        'Features.Kurtosis.Kurtosis',
        'Features.Entropy.Mean',
        'Features.Entropy.StandardDeviation',
        'Features.Entropy.Skew',
        'Features.Entropy.Kurtosis',
        'Features.MutualInformation.Mean',
        'Features.MutualInformation.StandardDeviation',
        'Features.MutualInformation.Skew',
        'Features.MutualInformation.Kurtosis',
        'FeaturesLabels.MutualInformation.Mean',
        'FeaturesLabels.MutualInformation.StandardDeviation',
        'FeaturesLabels.MutualInformation.Skew',
        'FeaturesLabels.MutualInformation.Kurtosis',
        'Number of Examples',
        'Number of Features',
        'n_estimators',
        'pruning_method',
        'pruning_cp',
        'ds',
        'Algorithm'
        ]
        for feature_name in meta_features_allnames:
            if not (feature_name) in meta_features:
                meta_features[feature_name] = np.nan
        return meta_features

    def _validateDataset(self, dataset, target):
        dtype = dataset[target].dtype
        if dtype in (np.object,):
            return True
        elif dtype in (np.int, np.int32, np.int64, np.float, np.float32,
                       np.float64, int, float):
            return True
        else:
            print("Não é válido o Dataset")
            return False
    
    def _skipCombination(self,pruning):
        if pruning['pruning_method'] == 0:
            if pruning['pruning_cp'] == 0.5 or pruning['pruning_cp'] == 0.75:
                return True
            else:
                return False

    # Prunning: Reduced-Error
    def _re(self,target, # Target names
                preds, # Predicts na training data
                data, # training data
                cutPoint): # ratio of the total number of models to cut off
        prunedN = math.ceil((len(preds) - (len(preds) * cutPoint)))

        ordem = {}
        N = data.shape[0]
        M = len(preds)

        C = []
        for m in range(M):
            c = 0
            for n in range(N):
                c = c + preds[m][n] - target[n]
            C.append(c)

        for i in range(prunedN):
            # (1)
            S = []
            for w in range(len(preds)):
                s = 0
                for j in range(len(preds)):
                    if j not in ordem:
                        s = s + C[w]
                s = s + C[j]
                s = pow(i,-1 * s)
                S.append(s)

            valor  = max(S) * 2
            for w in ordem.values():
                S[w] = valor
            ordem[i] = np.argmin(S)
        return ordem
    # Prunning: Orderer Aggregation
    def _oa(self,target, # Target names
            preds, # Predicts na training data
            data, # training data
            cutPoint): # ratio of the total number of models to cut off
        prunedN = math.ceil((len(preds) - (len(preds) * cutPoint)))
        print(" TO - DO ")
