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
warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
warnings.filterwarnings("ignore", message="divide by zero encountered") 
warnings.filterwarnings("ignore", message="invalid value encountered")
openml.config.apikey = '819d1d52e9314798feeb0d96fbd45b7f'
TargetNames = []
Datasets = []
# Open ML Valid Datasets
working = [1,3,5,6]
index = [1,3,5,9,10,11,12,14,15,16,18,20,21,22,23,24,28,29,30,31,32,
36,37,38,40,42,43,44,46,48,50,53,54,55,56,59,60,61,174,181,182,183,188,
300,307,312,313,333,334,335,377,444,448,451,458,461,464,469,470,478,
714,726,736,747,748,754,782,783,784,811,829,867,875,885,890,895,902,916,
921,955,969,974,1013,1038,1043,1063,1116,1462,1464,1466,1467,1468,1475,
1479,1480,1485,1487,1489,1491,1492,1493,1494,1497,1501,1504,1510,1515,
1570,4134,4538,4550,6332,23381,40496,40499,40509,40536,40670,40701,
40900,40910,40966,40971,40975,40978,40979,40981,40982,40983,40984,40994]

GoodDatasets = []
print("Get Datasets({})".format(len(index)))
# Load and Validate Datasets
n_dataset=1
for i in index:
    try:
        dataset = openml.datasets.get_dataset(i)
        X, y, categorical_indicator, attribute_names = dataset.get_data(dataset_format='dataframe')
        target = dataset.default_target_attribute
        dtype = X[target].dtype
        y_type = type_of_target(X[target])
        
        if y_type in ['binary', 'multiclass', 'multiclass-multioutput',
                      'multilabel-indicator', 'multilabel-sequences']:
            if(dtype == 'category'):
                Datasets.append(X)
                TargetNames.append(target)
                print("OpenML Dataset[{}][{}]: {} - {} (examples, features)".format(n_dataset,i,y_type,np.shape(X)))
                n_dataset=n_dataset+1
            elif dtype in (np.object,):
                Datasets.append(X)
                TargetNames.append(target)
                print("OpenML Dataset[{}][{}]: {} - {} (examples, features)".format(n_dataset,i,y_type,np.shape(X)))
                n_dataset=n_dataset+1
            elif dtype in (np.int, np.int32, np.int64, np.float, np.float32,
                np.float64, int, float):
                Datasets.append(X)
                TargetNames.append(target)
                print("OpenML Dataset[{}][{}]: {} - {} (examples, features)".format(n_dataset,i,y_type,np.shape(X)))
                n_dataset=n_dataset+1
        else:     
            print("Invalid!")      
        
    except Exception as e:
        print("Error ", i, e)

with open("ValidDatasets.txt", "w") as txt_file:
    for id in GoodDatasets:
        txt_file.write(str(id) + ",")

print("Total amount of Datasets:",len(GoodDatasets))
#####################
#######################################################
################ AutoBagging Classifier################
#######################################################

print("\n\n\n***************** AutoBagging Classifier *****************")
model = autoBaggingClassifier()
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