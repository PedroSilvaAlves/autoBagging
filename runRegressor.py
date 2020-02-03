import numpy as np
import pandas as pd
import xgboost as xgb
import math as m
import joblib
import warnings
import openml
from autoBaggingRegressor import autoBaggingRegressor
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

#######################################################
################### MAIN FUNCTION #####################
#######################################################

#warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
openml.config.apikey = '2754bfd67b4aa8a5854f00d3fc4bdd89'
TargetNames = []
Datasets = []
### LOCAL DATASETS ###
try:
    Datasets.append(pd.read_csv('./datasets_regressor/analcatdata_negotiation.csv'))
    TargetNames.append('Future_business')
    Datasets.append(pd.read_csv('./datasets_regressor/baseball.csv'))
    TargetNames.append('RS')
    Datasets.append(pd.read_csv('./datasets_regressor/phpRULnTn.csv'))
    TargetNames.append('oz26')
    Datasets.append(pd.read_csv('./datasets_regressor/dataset_2193_autoPrice.csv'))
    TargetNames.append('class')
    Datasets.append(pd.read_csv('./datasets_regressor/dataset_8_liver-disorders.csv'))
    TargetNames.append('drinks')
    Datasets.append(pd.read_csv('./datasets_regressor/cpu_small.csv'))
    TargetNames.append('usr')
except FileNotFoundError:
    print(
        "Path do dataset está errado, deve conter uma pasta 'dataset' no path do ficheiro autoBagging")
    quit()
######################

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


#######################################################
################ AutoBagging Regressor#################
#######################################################
print("\n\n\n***************** AutoBagging Regressor *****************")
model = autoBaggingRegressor(meta_functions=meta_functions,
                             post_processing_steps=post_processing_steps)

'''
IN CASE THE PROGRAM FAIL'S WE CAN USE BACK UP DATA

meta_data = pd.read_csv("./metadata/MetaData_backup.csv")
meta_target = pd.read_csv("./metadata/MetaTarget_backup.csv")
meta_target = np.array(meta_target)
model = model.load_fit(meta_data,meta_target)
'''
print("\nCreate Meta-Data from {} Datasets then Fit Meta-Model".format(6))

model = model.fit(Datasets, TargetNames)
joblib.dump(model, "./models/autoBaggingRegressorModel.sav")



#######################################################
################## Loading Dataset ####################
#######################################################
dataset = pd.read_csv('./datasets_regressor/test/dataset_2190_cholesterol.csv')
targetname = 'chol'

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

# Getting recommended Bagging model of the dataset
bestBagging = model.predict(dataset_train,targetname)

# Getting Default Bagging
DefaultBagging = BaggingRegressor(random_state=0)
DefaultBagging.fit(X_train,y_train)

#######################################################
################## Testing Bagging ####################
#######################################################

print("Verify Bagging algorithm score:")
score = bestBagging.score(X_test,y_test)
print("Recommended  Bagging --> Score: %0.2f" % score)
score = DefaultBagging.score(X_test,y_test)
print("Default Bagging --> Score: %0.2f" % score)