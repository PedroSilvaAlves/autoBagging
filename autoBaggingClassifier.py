import numpy as np
import pandas as pd
import xgboost as xgb
import math as m
import joblib
import warnings
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.metrics import cohen_kappa_score
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
from deslib.des.knora_e import KNORAE
from deslib.dcs.ola import OLA


class autoBaggingClassifier(BaseEstimator):

    def __init__(self, meta_functions,post_processing_steps):
        self.meta_functions = meta_functions
        self.post_processing_steps = post_processing_steps
        self.base_estimators = {'Decision Tree (max_depth=1)': DecisionTreeClassifier(max_depth=1, random_state=0),
                                'Decision Tree (max_depth=2)': DecisionTreeClassifier(max_depth=2, random_state=0),
                                'Decision Tree (max_depth=3)': DecisionTreeClassifier(max_depth=3, random_state=0),
                                'Decision Tree (max_depth=4)': DecisionTreeClassifier(max_depth=4, random_state=0),
                                'Naive Bayes': GaussianNB(),
                                'Majority Class': DummyClassifier(random_state=0)}
        self.grid = ParameterGrid({"n_estimators" : [50, 100, 200],
                                   "bootstrap" : [True],
                                   "bootstrap_features" : [False],
                                   "max_samples" : [1.0],
                                   "max_features": [1.0]})
        self.pruning = ParameterGrid({'pruning_method' : [0],
                                      'pruning_cp': [0.1]})
        self.DStechique = ParameterGrid({ 'ds' : [0,1]})

    def fit(self,
            datasets,                # Lista com datasets
            target_names):           # Nome dos targets de todas os datasets
            
        # Por cada file abrir o csv e tirar para um array de DataFrames
        x_meta = []     # Vai conter todas as Meta-features, uma linha um exemplo de um algoritmo com um certo tipo de parametros
        y_meta = []     # Vai conter o Meta-Target, em cada linha têm a avaliação de 1-n de cada algoritmo
                        # + parametros do bagging workflow
        ndataset = 0
        for dataset, target in zip(datasets, target_names):  # Percorre todos os datasets para criar Meta Data
            if self._validateDataset(dataset, target):
                ndataset= ndataset+1
                print("Dataset nº ",ndataset)
                # Tratar do Dataset
                # Drop Categorial features sklearn não aceita
                for f in dataset.columns:
                    if dataset[f].dtype == 'object':
                        dataset = dataset.drop(columns=f, axis=1)
                
                # MetaFeatures
                meta_features_estematic = self._metafeatures(
                    dataset, target, self.meta_functions, self.post_processing_steps)
                
                # Dividir o dataset em exemplos e os targets
                X = SimpleImputer().fit_transform(dataset.drop(target, axis=1))
                y = dataset[target]
                X, X_test, y, y_test = train_test_split(X, y, test_size=0.33,
                                                    random_state=0)
                # Split the data into training and DSEL for DS techniques
                X, X_dsel, y, y_dsel = train_test_split(X, y,
                                                    test_size=0.5,
                                                    random_state=0)
                # Criar base-models
                for params in self.grid:  # Combinações de Parametros
                    for DS in self.DStechique:
                        for pruning in self.pruning:
                            meta_features = meta_features_estematic.copy() # Meta-features do dataset só é criado uma vez
                            Rank = {}
                            for base_estimator in self.base_estimators:  # Combinação dos algoritmos base
                                # Criar modelo
                                bagging_workflow = BaggingClassifier(base_estimator=self.base_estimators[base_estimator],
                                                                        random_state=0,
                                                                        **params)
                                # Treinar o modelo
                                bagging_workflow.fit(X, y)
                                predictions = bagging_workflow.predict(X_test)
                                # Criar landmark do baggingworkflow atual
                                predictions = []
                                
                                if pruning['pruning_method'] == 1:
                                    for estimator, features in zip(bagging_workflow.estimators_,bagging_workflow.estimators_features_):
                                        predictions.append(estimator.predict(X[:, features]))
                                    bb_index= self._bb(y, predictions, X, pruning['pruning_cp'])
                                    print("BB_index = ", bb_index)
                                    estimators = []
                                    for i in bb_index.values():
                                        estimators.append(bagging_workflow.estimators_[i])
                                    bagging_workflow.estimators_ = estimators
                                else:
                                    if pruning['pruning_method'] == -1:
                                        print("MDSQ")
                                
                                if DS['ds'] == -1:
                                    #print("KNORAE BEFORE-> ", cohen_kappa_score(y_test, bagging_workflow.predict(X_test)))
                                    bagging_workflow = KNORAE(bagging_workflow)
                                    bagging_workflow.fit(X_dsel,y_dsel)
                                    #print("KNORAE AFTER -> ", cohen_kappa_score(y_test, bagging_workflow.predict(X_test)))
                                    #print("----------------------------")

                                if DS['ds'] == 1:
                                    #print("OLA BEFORE-> ", cohen_kappa_score(y_test, bagging_workflow.predict(X_test)))
                                    bagging_workflow = OLA(bagging_workflow)
                                    bagging_workflow.fit(X_dsel,y_dsel)
                                    #print("OLA AFTER -> ", cohen_kappa_score(y_test, bagging_workflow.predict(X_test)))
                                    #print("----------------------------")
                                    
                                predictions = bagging_workflow.predict(X_test)
                                Rank[base_estimator] = cohen_kappa_score(y_test, predictions)
                                #print("Rank -> ", Rank[base_estimator])
                                # Adicionar ao array de metafeatures, as caracteriticas dos baggings workflows
                                meta_features['bootstrap'] = np.multiply(params['bootstrap'], 1)
                                meta_features['bootstrap_features'] = np.multiply(params['bootstrap_features'], 1)
                                meta_features['n_estimators'] = params['n_estimators']
                                meta_features['max_samples'] = params['max_samples']
                                meta_features['max_features'] = params['max_features']
                                meta_features['pruning_method'] = pruning['pruning_method']
                                meta_features['pruning_cp'] = pruning['pruning_cp']
                                meta_features['ds'] = DS['ds']
                            # Fim combinação dos algoritmos base, ainda dentro da Combinação de Parametros
                            i = 1
                            for base_estimator in sorted(Rank, key=Rank.__getitem__, reverse=True):
                                # Ordena os algoritmos atravez do landmark e atribui um rank (1 é o melhor)
                                meta_features['Algorithm: ' + base_estimator] = i
                                Rank[base_estimator] = i
                                i = i+1
                            array_rank = [] # Este array vai contem o target deste algoritmo
                            for value in Rank.values():
                                array_rank.append(value)  # Adicina os ranks dos algoritmos
                            # Adiciona os vários parametros
                            array_rank.append(params['n_estimators'])
                            array_rank.append(np.multiply(params['bootstrap'], 1))
                            array_rank.append(np.multiply(params['bootstrap_features'], 1))
                            array_rank.append(params['max_samples'])
                            array_rank.append(params['max_features'])
                            array_rank.append(pruning['pruning_method'])
                            array_rank.append(pruning['pruning_cp'])
                            array_rank.append(DS['ds'])

                            y_meta.append(array_rank)
                            # Este array contem as várias metafeatures do dataset e o scores do algoritmo base/parametros a testar
                            x_meta.append(meta_features)

        # Meta Data é a junção de todas as metafeatures com os scores dos respeticos algoritmos base
        self.meta_data = pd.DataFrame(x_meta)
        self.meta_target = np.array(y_meta)
        # Guardar Meta Data num ficheiro .CSV
        self.meta_data.to_csv('./metadata/Meta_Data_Classifier.csv')
        print("Meta-Data Created.")
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

        print("Constructing Meta-Model:")
        # Criar o Meta Model XGBOOST
        meta_model = xgb.XGBRegressor(objective="reg:squarederror",
                                        colsample_bytree=0.3,
                                        learning_rate=0.1,
                                        max_depth=5,
                                        alpha=10,
                                        n_estimators=100)
        self.meta_model = MultiOutputRegressor(meta_model)

        # Aplicar Learning algorithm
        self.meta_model.fit(self.meta_data, self.meta_target)
        return self

    def predict(self, dataset, targetname):
        if self._validateDataset(dataset, targetname):

            meta_features = self._metafeatures(
                dataset, targetname, self.meta_functions, self.post_processing_steps)
            
            # Trata dataset para a predict
            X_test = pd.DataFrame(meta_features, index=[0])
            # for f in X_test.columns:
            #     if X_test[f].dtype == 'object':
            #         lbl = LabelEncoder()
            #         lbl.fit(list(X_test[f].values))
            #         X_test[f] = lbl.transform(list(X_test[f].values))
            X_test.fillna((-999), inplace=True)
            # X_test=np.array(X_test)
            X_test = X_test.astype(float)

            # Predict the best algorithm
            preds = self.meta_model.predict(X_test)

            # Prints e construção do Bagging previsto
            # Trocar para classificação?
            n_estimators = int(preds[0][6])
            if preds[0][7] > 0.5:
                bootstrap = True
            else:
                bootstrap = False
            if preds[0][8] > 0.5:
                bootstrap_features = True
            else:
                bootstrap_features = False
            if preds[0][9] > 0.75:
                max_samples = 1.0
            else:
                max_samples = 0.5
            if preds[0][10] < 1.5:
                max_features = 1.0
            else:
                if preds[0][10] < 3:
                    max_features = 2.0
                else:
                    max_features = 4.0
            if preds[0][11] > 0.5:
                pruning_method = 'BB'
            else:
                if preds[0][11] < -0.5:
                    pruning_method = 'MDSQ'
                else:
                    pruning_method = 'None'
            pruning_cp = int(preds[0][12])

            if preds[0][13] > 0.5:
                ds = 'KNORAE'
            else:
                if preds[0][13] < -0.5:
                    ds = 'OLA'
                else:
                    ds = 'None'

            algorithm_index = 0
            algorithm_score = preds[0][0]
            for i in range(0, 6):
                if preds[0][i] < algorithm_score:
                    algorithm_score = preds[0][i]
                    algorithm_index = i
            switcher = {
               -1: "Error",
                0: "Decision Tree (max_depth=1)",
                1: "Decision Tree (max_depth=2)",
                2: "Decision Tree (max_depth=3)",
                3: "Decision Tree (max_depth=4)",
                4: "Naive Bayes",
                5: "Majority Class"}

            print("Recommended Bagging workflow: ")
            print("\tNumber of models: ", n_estimators)
            print("\tBootstrap: ", bootstrap)
            print("\tBootstrap_features: ",bootstrap_features)
            print("\tMax_samples: ", max_samples)
            print("\tMax_features: ", max_features)
            print("\tPruning Method: ", pruning_method)
            print("\tPruning CutPoint: ", pruning_cp)
            print("\tDynamic Selection: ", ds)
            print("\tAlgorithm: ", switcher.get(algorithm_index))

            # BaggingWorkflow
            return BaggingClassifier(
                    base_estimator= self.base_estimators[switcher.get(algorithm_index)],
                    n_estimators=n_estimators,
                    bootstrap=bootstrap,
                    bootstrap_features=bootstrap_features,
                    max_samples=max_samples,
                    max_features=max_features,
                    random_state=0,
                    )
        else:
            print("Erro, error não é um problema de Classifier")

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
        meta_features['Number of Classes'] = dataset[target].unique().shape[0]
        meta_features_allnames = ['Features.SpearmanCorrelation.Mean',
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
                                  'bootstrap',
                                  'bootstrap_features',
                                  'n_estimators',
                                  'max_samples',
                                  'max_features',
                                  'pruning_method',
                                  'pruning_cp',
                                  'ds',
                                  'Algorithm: Decision Tree (max_depth=1)',
                                  'Algorithm: Decision Tree (max_depth=2)',
                                  'Algorithm: Decision Tree (max_depth=3)',
                                  'Algorithm: Decision Tree (max_depth=4)',
                                  'Algorithm: Naive Bayes',
                                  'Algorithm: Majority Class']
        for feature_name in meta_features_allnames:
            if not (feature_name) in meta_features:
                meta_features[feature_name] = np.nan
        return meta_features

    def _validateDataset(self, dataset, targetname):
        # Classificação Binária
        if dataset[targetname].dtype != 'object':
            if sorted(dataset[targetname].unique()) != [0, 1]:
                print("Não é válido o Dataset")
                return False
        return True

    # Prunning: Boosting-based pruning of models
    def _bb(self,target, # Target names
                preds, # vetor de predicts de cada estimator no training data
                data, # training data
                cutPoint): # ratio of the total n umber of models to cut off
        prunedN = m.ceil((len(preds) - (len(preds) * cutPoint)))
        print("NumModelos = ", len(preds),"\nPrunedN = ", prunedN)
        weights = []
        for i in range(data.shape[0]):
            weights.append(1/data.shape[0])

        ordem = {}
        for i in range(prunedN):
            errors = []
            for w in range(len(preds)):
                erro = 0
                for x in range(len(weights)):
                    erro = erro + ((not((preds[w][x] == target[x]))* -1) * weights[x])
                errors.append(erro)
            valor  = max(errors) *2
            for w in ordem.values():
                errors[w] = valor
            ordem[i] = np.argmin(errors)
            errorU = min(errors)
            predU = []
            for x in range(len(weights)):
                predU.append(preds[ordem[i]][x] == target[x])

            if errorU > 0.5:
                weights = []
                for i in range(data.shape[0]):
                    weights.append(1/data.shape[0])
            else:
                for w in range(len(weights)):
                    if predU[w] == True:
                        try:
                            weights[w] = weights[w] / (2*errorU)
                            break
                        except ZeroDivisionError:
                            weights[w] = 10.000e+300
                    else:
                        try:
                            weights[w] = weights[w] / (2*(1 - errorU))
                            break
                        except ZeroDivisionError:
                            weights[w] = 10.000e+300
        return ordem

    # Prunning: Margin Distance Minimization   
    def _mdsq(self,target, # Target names
                preds, # Predicts na training data
                data, # training data
                cutPoint): # ratio of the total number of models to cut off
        
        prunedN = m.ceil((len(preds) - (len(preds) * cutPoint)))
        print("NumModelos = ", len(preds),"\nPrunedN = ", prunedN)
        pred = [] # 1 ou -1 se acertar ou não
        ens = []
        o = []


        for i in range(target.length):
            ens.append(0)
            o.append(0.075)
        for i in range(len(preds)):
            for x, tg in zip(preds[i],target):
                if x == tg:
                    pred.append(1)
                else:
                    pred.append(-1)

        pred = pd.Dataframe(pred)

        ordem = []
        for i in range(1,prunedN):
            dist = []
            for x, y, z in zip(pred, ens, o):
                dist.append(m.sqrt(sum((((x + y) / i) - z)^2)))
            ens = ens + pred[min(dist)]
            pred = pd.Dataframe(pred[list(set(pred) - set(min(dist)))]) # Buscar o name em vez do valor, está mal.
            ordem[i] = min(dist) # Como inteiro, pelo nome
        ordem = {}
        return ordem   