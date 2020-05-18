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
from sklearn.utils.multiclass import type_of_target

#######################################################
################### MAIN FUNCTION #####################
#######################################################

#warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
openml.config.apikey = '819d1d52e9314798feeb0d96fbd45b7f'
TargetNames = []
Datasets = []


###### OPEN ML #######
index = [8,189,190,191,193,194,195,199,200,203,204,206,210,211,213,217,222,223,227,228,232,491,506,507,511,524,531,541,562,566,578,639,673,703,707,1096,41700,42224]
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
        if dtype in (np.int, np.int32, np.int64, np.float, np.float32,
                    np.float64, int, float):
            Datasets.append(X)
            TargetNames.append(target)
            GoodDatasets.append(i)
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

### LOCAL DATASETS ###
# try:
#     Datasets.append(pd.read_csv('./datasets_regressor/analcatdata_negotiation.csv'))
#     TargetNames.append('Future_business')
#     Datasets.append(pd.read_csv('./datasets_regressor/baseball.csv'))
#     TargetNames.append('RS')
#     Datasets.append(pd.read_csv('./datasets_regressor/phpRULnTn.csv'))
#     TargetNames.append('oz26')
#     Datasets.append(pd.read_csv('./datasets_regressor/dataset_2193_autoPrice.csv'))
#     TargetNames.append('class')
#     Datasets.append(pd.read_csv('./datasets_regressor/dataset_8_liver-disorders.csv'))
#     TargetNames.append('drinks')
#     Datasets.append(pd.read_csv('./datasets_regressor/cpu_small.csv'))
#     TargetNames.append('usr')
# except FileNotFoundError:
#     print(
#         "Path do dataset está errado, deve conter uma pasta 'dataset' no path do ficheiro autoBagging")
#     quit()
######################

#######################################################
################ AutoBagging Regressor#################
#######################################################
print("\n\n\n***************** AutoBagging Regressor *****************")
model = autoBaggingRegressor()

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
