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
# Open ML Valid Datasets
index = [551,556,557,561,674,659,661,663,665,703,710,712,684,687,688,690,
        692,697,1035,1027,1028,1029,1030,301,1089,1097,1098,1099,1072,1070,
        1091,1093,544,549,1228,456,482,483,485,506,509,510,521,523,524,526,527,
        530,533,535,536,539,540,511,513,515,516,518,519,520,491,492,494,497,498,
        191,194,195,199,200,203,204,205,207,224,230,231,211,213,427,
        294,299,42176,42111,42112,42113,42110,42165,42166,41943,40505,
        41514,41515,41516,41517,41518,41519,41021,41968,41969]
GoodDatasets = []
print("Get Datasets({})".format(len(index)))
# Load and Validate Datasets
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
                Datasets.append(X)
                TargetNames.append(target)
                GoodDatasets.append(i)
                print("OpenML Dataset[{}]: {} - {} (examples, features)".format(i,y_type,np.shape(X)))
            elif dtype in (np.int, np.int32, np.int64, np.float, np.float32,
                        np.float64, int, float):
                Datasets.append(X)
                TargetNames.append(target)
                GoodDatasets.append(i)
                print("OpenML Dataset[{}]: {} - {} (examples, features)".format(i,y_type,np.shape(X)))
        else:     
            print("Invalid!")      
        
    except Exception:
        print("Error ", i)

with open("ValidDatasets.txt", "w") as txt_file:
    for id in GoodDatasets:
        txt_file.write(str(id) + ",")

print("Total amount of Datasets:",len(index))
#####################

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
################ AutoBagging Classifier################
#######################################################

print("\n\n\n***************** AutoBagging Classifier *****************")
model = autoBaggingClassifier(meta_functions=meta_functions,
                              post_processing_steps=post_processing_steps)
'''
IN CASE THE PROGRAM FAIL'S WE CAN USE BACK UP DATA

meta_data = pd.read_csv("./metadata/MetaData_backup.csv")
meta_target = pd.read_csv("./metadata/MetaTarget_backup.csv")
meta_target = np.array(meta_target)
model = model.load_fit(meta_data,meta_target)
'''

print("\nCreate Meta-Data from {} Datasets then Fit Meta-Model".format(len(index)))
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


# Getting recommended Bagging model of the dataset
bestBagging = model.predict(dataset_train,targetname)
# Getting Default Bagging
DefaultBagging = BaggingClassifier(random_state=0)
DefaultBagging.fit(X_train,y_train)

#######################################################
############## Single Testing Bagging #################
#######################################################
print("Verify Bagging algorithm score:")
score = bestBagging.score(X_test,y_test)
print("Recommended  Bagging --> Score: %0.2f" % score)
score = DefaultBagging.score(X_test,y_test)
print("Default Bagging --> Score: %0.2f" % score)