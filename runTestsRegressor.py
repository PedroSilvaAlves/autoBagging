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
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingRegressor
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

warnings.simplefilter(action='ignore')
warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
warnings.filterwarnings("ignore", message="divide by zero encountered") 
warnings.filterwarnings("ignore", message="invalid value encountered")
openml.config.apikey = '819d1d52e9314798feeb0d96fbd45b7f'
TargetNames = []
Datasets = []

index = [189,191,506,507,194,200,531,199,562,639]
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

#######################################################
################ AutoBagging Classifier################
#######################################################

print("\n\n\n***************** AutoBagging Classifier *****************")
model = autoBaggingRegressor(silence=False)
meta_data = pd.read_csv("./metadata/Meta_Data_Regressor_final.csv")
meta_target = pd.read_csv("./metadata/Meta_Target_Regressor_final.csv")
model = model.load_fit(meta_data,meta_target)

#print("\nCreate Meta-Data from {} Datasets then Fit Meta-Model".format(len(index)))
#model = model.fit(Datasets, TargetNames)
#joblib.dump(model, "./models/autoBaggingClassifierModel.sav")


#######################################################
################ Loading Test Dataset #################
#######################################################

autoBagging_score = []
defaultBagging_score = []

i = 0
for dataset, target in zip(Datasets, TargetNames):
    i= i + 1
    print("________________________________________________________________________")
    print("Dataset nº ", i)
    print("Shape: {}(examples, features)".format(np.shape(dataset)))

    for f in dataset.columns:
        if dataset[f].dtype == 'object':
            dataset = dataset.drop(columns=f, axis=1)
    
    # Convert +/- Inf to NaN
    dataset.replace(np.inf, np.nan)
    dataset.replace(-np.inf, np.nan)
    # Drop Columns with all NaN values
    dataset = dataset.dropna(axis=1,how ='all')
    # Drop examples with some Nan Values
    dataset = dataset.dropna(axis=0,how = 'any') 
    dataset = dataset.reset_index(drop=True)
    
    labelencoder = LabelEncoder()
    dataset[target] = labelencoder.fit_transform(dataset[target])
    #print(dataset.head())
    dataset_train, dataset_test = train_test_split(dataset, test_size=0.3,
                                                    random_state=0, shuffle=True)
    dataset_train = dataset_train.reset_index(drop=True)
    dataset_test = dataset_test.reset_index(drop=True)
    
    # Getting recommended Bagging model of the dataset
    #autoBagging = model.predict(dataset_train.copy(), target)

    # Getting Default Bagging
    #X_train = SimpleImputer().fit_transform(dataset_train.drop(target, axis=1))
    #y_train = dataset_train[target]
    #X_test = SimpleImputer().fit_transform(dataset_test.drop(target, axis=1))
    #y_test = dataset_test[target]
    
    #DefaultBagging = BaggingRegressor(random_state=0, n_estimators=5)
    #DefaultBagging.fit(X_train.copy(),y_train.copy())

    #######################################################
    ################ Testing Bagging ######################
    #######################################################

    #score = autoBagging.score(X_test.copy(),y_test.copy())
    ranks = model.teste()
    #autoBagging_score.append(ranks)
    print("Ranks Bagging[",i,"]Scores:", ranks)
    #print("Recommended Bagging --> Score: %0.2f" % score)
    #score = DefaultBagging.score(X_test.copy(),y_test.copy())
    #defaultBagging_score.append(score)
    #print("Default Bagging     --> Score: %0.2f" % score)

with open("Results_123.txt", "w") as txt_file:
    #txt_file.write("AUTOBAGGING:\n")
    #for auto in autoBagging_score:
    #    txt_file.write(str(auto) + ",")
    #txt_file.write("DEFAULT:\n")
    #for default in defaultBagging_score:
    #    txt_file.write(str(default) + ",")
    txt_file.write("AUTOBAGGING_META_MODEL:\n")
    for auto in autoBagging_score:
        txt_file.write("[")
        for i in auto:
            txt_file.write(str(i) + ",")
        txt_file.write("],\n")
    
    #for auto,default in zip(autoBagging_score,defaultBagging_score):
    #    print(default, ",", auto)
    #    txt_file.write(str(default) + ","+ str(auto)+ "\n")

#print("Auto    Bagging ->",np.array(autoBagging_score).mean())
#print("Default Bagging ->",np.array(defaultBagging_score).mean())

#print("Percentile ->", np.array(autoBagging_score).mean()/np.array(defaultBagging_score).mean())
