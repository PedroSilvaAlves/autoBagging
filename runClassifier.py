import numpy as np
import pandas as pd
import xgboost as xgb
import math as m
import joblib
import warnings
import openml
from autoBaggingClassifier import autoBaggingClassifier
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
from sklearn.utils.multiclass import type_of_target

#######################################################
################### MAIN FUNCTION #####################
#######################################################

warnings.simplefilter(action='ignore', category=FutureWarning)
openml.config.apikey = '2754bfd67b4aa8a5854f00d3fc4bdd89'
TargetNames = []
Datasets = []
### LOCAL DATASETS ###
# try:
    
#     #Datasets.append(pd.read_csv('./datasets_classifier/titanic.csv'))
#     #TargetNames.append('Survived')
#     Datasets.append(pd.read_csv('./datasets_classifier/heart.csv'))
#     TargetNames.append('target')
#     Datasets.append(pd.read_csv('./datasets_classifier/titanic.csv'))#     TargetNames.append('Survived')
# except FileNotFoundError:
#     print(
#         "Path do dataset está errado, deve conter uma pasta 'dataset' no path do ficheiro autoBagging")
#     quit()
######################

####### OPENML #######
#Valid datasets 201, 204, 205, 207, 213
#index = [41021,197,200,201,218,287,294,298,301,482,494,513,516,524,533,536,537,549,41514,555,556,557,41518,41515,41516,562,41517,41524,41525,41519,41523,573,574,41539,688,703,41943,41968,41969,1027,1028,1029,1030,1035,42092,42110,42111,42112,42113,42165,42166,42176,42178,42183]
index = [41021,197,200,201,287,294,298,301,482,494,513,516,524,533,536,537,549,41514,555,556,557,41518,41515,41516,562,41517,41524,41525,41519,41523]
GoodDatasets = []
for i in index:
    try:
        dataset = openml.datasets.get_dataset(i)
        X, y, categorical_indicator, attribute_names = dataset.get_data(dataset_format='dataframe')
        target = dataset.default_target_attribute
        dtype = X[target].dtype
        y_type = type_of_target(X[target])
        
        if y_type in ['binary', 'multiclass', 'multiclass-multioutput',
                      'multilabel-indicator', 'multilabel-sequences']:
            if dtype in (np.object,):
                print("Dataset Válido = ",y_type)
                Datasets.append(X)
                TargetNames.append(target)
                GoodDatasets.append(i)
            elif dtype in (np.int, np.int32, np.int64, np.float, np.float32,
                        np.float64, int, float):
                print("Dataset Válido = ",y_type)
                Datasets.append(X)
                TargetNames.append(target)
                GoodDatasets.append(i)
        else:     
            print("Invalid!")      
        
    except Exception:
        print("Error ", i)
        #print("Valid= ")
        #print(*GoodDatasets)

print("Valid Datasets:", *GoodDatasets)
with open("ValidDatasets.txt", "w") as txt_file:
    for id in GoodDatasets:
        txt_file.write(str(id) + ",")
#####################

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
model = autoBaggingClassifier(meta_functions=meta_functions,
                              post_processing_steps=post_processing_steps)
model = model.fit(Datasets, TargetNames)
joblib.dump(model, "./models/autoBaggingClassifierModel.sav")


#######################################################
################ Loading Test Dataset #################
#######################################################
dataset = pd.read_csv('./datasets_classifier/test/weatherAUS.csv')
dataset = dataset.drop('RISK_MM', axis=1)
targetname = 'RainTomorrow'
dataset[targetname] = dataset[targetname].map({'No': 0, 'Yes' : 1}).astype(int)
dataset.fillna((-999), inplace=True)
for f in dataset.columns:
    if dataset[f].dtype == 'object':
        dataset = dataset.drop(columns=f, axis=1)

dataset_train, dataset_test = train_test_split(dataset,test_size=0.33,
                                                    random_state=0,shuffle=True)
X_train = SimpleImputer().fit_transform(dataset_train.drop(targetname, axis=1))
y_train = dataset_train[targetname]
X_test = SimpleImputer().fit_transform(dataset_test.drop(targetname, axis=1))
y_test = dataset_test[targetname]

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
#                                                    random_state=0,shuffle=False)
# Getting recommended Bagging model of the dataset

bestBagging = model.predict(dataset_train,targetname)

# Getting Default Bagging
DefaultBagging = BaggingClassifier(random_state=0)
print("Verify Bagging algorithm score:")
#######################################################
################## Testing Bagging ####################
#######################################################

score = bestBagging.score(X_test,y_test)
print("Recommended  Bagging --> Score: %0.2f" % score)
DefaultBagging.fit(X_train,y_train)
score = DefaultBagging.score(X_test,y_test)
print("Default Bagging --> Score: %0.2f" % score)
#kfold = KFold(n_splits=10, random_state=0)
#cv_results = cross_val_score(bestBagging, X, y, cv=kfold, scoring='accuracy')
#print("Recommended Bagging --> Score: %0.2f (+/-) %0.2f)" % (cv_results.mean(), cv_results.std() * 2))
#kfold = KFold(n_splits=10, random_state=0)
#cv_results = cross_val_score(DefaultBagging, X, y, cv=kfold, scoring='accuracy')
#print("Default Bagging --> Score: %0.2f (+/-) %0.2f)" % (cv_results.mean(), cv_results.std() * 2))