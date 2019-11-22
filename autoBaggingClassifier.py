import numpy as np
import pandas as pd
import xgboost as xgb
import math as m
import joblib

from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
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
import warnings


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
        self.grid = ParameterGrid({"n_estimators": [50, 100, 200],
                                   "bootstrap": [True, False],
                                   "bootstrap_features" : [True, False],
                                   "max_samples": [0.5, 1.0],
                                   "max_features": [1, 2, 4]})

    def fit(self,
            file_name_datasets,     # Nome de todos os ficheiros .CSV
            target_names):           # Nome dos targets de todas os datasets
            
        # Por cada file abrir o csv e tirar para um array de DataFrames
        x_meta = []     # Vai conter todas as Meta-features, uma linha um exemplo de um algoritmo com um certo tipo de parametros
        y_meta = []     # Vai conter o Meta-Target, em cada linha têm a avaliação de 1-n de cada algoritmo
                        # + parametros do bagging workflow
        
        for file_name, target in zip(file_name_datasets, target_names):  # Percorre todos os datasets para treino do meta-model
            print("Creating Meta-features for: ", file_name)
            try:
                dataset = pd.read_csv(file_name)
            except FileNotFoundError:
                print(
                    "Path do dataset está errado, deve conter uma pasta 'dataset' no path do ficheiro autoBagging")
                quit()

            if self._validateDataset(dataset, target):
                # MetaFeatures
                meta_features_estematic = self._metafeatures(
                    dataset, target, self.meta_functions, self.post_processing_steps)
                # Label Encoder para melhores resultados
                for f in dataset.columns:
                    if dataset[f].dtype == 'object':
                        lbl = LabelEncoder()
                        lbl.fit(list(dataset[f].values))
                        dataset[f] = lbl.transform(list(dataset[f].values))
                # Dividir o dataset em exemplos e os targets
                X = SimpleImputer().fit_transform(dataset.drop(target, axis=1))
                y = dataset[target]
                # Criar base-models
                for params in self.grid:  # Combinações de Parametros
                    meta_features = meta_features_estematic.copy() # Meta-features do dataset só é criado uma vez
                    Rank = {}
                    for base_estimator in self.base_estimators:  # Combinação dos algoritmos base
                        # Criar modelo
                        bagging_workflow = BaggingClassifier(base_estimator=self.base_estimators[base_estimator],
                                                                random_state=0,
                                                                **params)
                        # Treinar o modelo
                        bagging_workflow.fit(X, y)
                        # Criar landmark do baggingworkflow atual
                        predictions = bagging_workflow.predict(X)
                        Rank[base_estimator] = cohen_kappa_score(
                            y, predictions)
                        # Adicionar ao array de metafeatures, as caracteriticas dos baggings workflows
                        meta_features['bootstrap'] = np.multiply(params['bootstrap'], 1)
                        meta_features['bootstrap_features'] = np.multiply(params['bootstrap_features'], 1)
                        meta_features['n_estimators'] = params['n_estimators']
                        meta_features['max_samples'] = params['max_samples']
                        meta_features['max_features'] = params['max_features']
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
            for f in X_test.columns:
                if X_test[f].dtype == 'object':
                    lbl = LabelEncoder()
                    lbl.fit(list(X_test[f].values))
                    X_test[f] = lbl.transform(list(X_test[f].values))
            X_test.fillna((-999), inplace=True)
            X_test=np.array(X_test)
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
                max_features = 1
            else:
                if preds[0][10] < 3:
                    max_features = 2
                else:
                    max_features = 4
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
            print("\tAlgorithm: ", switcher.get(algorithm_index))
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


#######################################################
################### MAIN FUNCTION #####################
#######################################################

warnings.simplefilter(action='ignore', category=FutureWarning)
TargetNames = []
FileNameDataset = []


FileNameDataset.append('./datasets_classifier/titanic.csv')
TargetNames.append('Survived')
FileNameDataset.append('./datasets_classifier/heart.csv')
TargetNames.append('target')

post_processing_steps = [Mean(),
                         StandardDeviation(),
                         Skew(),
                         Kurtosis()]


meta_functions = [Entropy(),
                 #PearsonCorrelation(),
                  MutualInformation(),
                  SpearmanCorrelation(),
                  basic_meta_functions.Mean(),
                  basic_meta_functions.StandardDeviation(),
                  basic_meta_functions.Skew(),
                  basic_meta_functions.Kurtosis()]


#######################################################
################ AutoBagging Classifier################
#######################################################
print("\n\n\n***************** AutoBagging Classifier *****************")
model = autoBaggingClassifier(meta_functions,post_processing_steps)
model = model.fit(FileNameDataset, TargetNames)
joblib.dump(model, "./models/autoBaggingClassifierModel.sav")


#######################################################
################ Loading Test Dataset #################
#######################################################
dataset = pd.read_csv('./datasets_classifier/test/weatherAUS.csv')
dataset = dataset.drop('RISK_MM', axis=1)
targetname = 'RainTomorrow'

dataset.fillna((-999), inplace=True)
for f in dataset.columns:
    if dataset[f].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(dataset[f].values))
        dataset[f] = lbl.transform(list(dataset[f].values))
X = SimpleImputer().fit_transform(dataset.drop(targetname, axis=1))
y = dataset[targetname]

# Getting recommended Bagging model of the dataset
bestBagging = model.predict(dataset,targetname)

# Getting Default Bagging
DefaultBagging = BaggingClassifier(random_state=0)

print("Verify Bagging algorithm score:")
#######################################################
################## Testing Bagging ####################
#######################################################
kfold = KFold(n_splits=10, random_state=0)
cv_results = cross_val_score(bestBagging, X, y, cv=kfold, scoring='accuracy')
print("Recommended Bagging --> Score: %0.2f (+/-) %0.2f)" % (cv_results.mean(), cv_results.std() * 2))

kfold = KFold(n_splits=10, random_state=0)
cv_results = cross_val_score(DefaultBagging, X, y, cv=kfold, scoring='accuracy')
print("Default Bagging --> Score: %0.2f (+/-) %0.2f)" % (cv_results.mean(), cv_results.std() * 2))
