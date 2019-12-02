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
from metafeatures.core.object_analyzer import analyze_pd_dataframe
from metafeatures.meta_functions.entropy import Entropy
from metafeatures.meta_functions import basic as basic_meta_functions
from metafeatures.meta_functions.pearson_correlation import PearsonCorrelation
from metafeatures.meta_functions.mutual_information import MutualInformation
from metafeatures.meta_functions.spearman_correlation import SpearmanCorrelation
from metafeatures.post_processing_functions.basic import Mean, StandardDeviation, Skew, Kurtosis
from metafeatures.post_processing_functions.basic import NonAggregated
from metafeatures.core.engine import metafeature_generator

#######################################################
################### MAIN FUNCTION #####################
#######################################################

warnings.simplefilter(action='ignore', category=FutureWarning)
TargetNames = []
Datasets = []
# ### LOCAL DATASETS ###
try:
    Datasets.append(pd.read_csv('./datasets_classifier/titanic.csv'))
    TargetNames.append('Survived')
    Datasets.append(pd.read_csv('./datasets_classifier/heart.csv'))
    TargetNames.append('target')
except FileNotFoundError:
    print(
        "Path do dataset está errado, deve conter uma pasta 'dataset' no path do ficheiro autoBagging")
    quit()
######################

####### OPENML #######

# index = [1,21,6,31]
# for i in index:
#     dataset = openml.datasets.get_dataset(i)
#     X, y, categorical_indicator, attribute_names = dataset.get_data(dataset_format='dataframe')
#     target = dataset.default_target_attribute
#     Datasets.append(X)
#     TargetNames.append(target)
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
