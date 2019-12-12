import numpy as np
import pandas as pd
import xgboost as xgb
import math as m
import joblib
import warnings

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier

model = joblib.load("./models/autoBaggingClassifierModel.sav")


#######################################################
################ Loading Test Dataset #################
#######################################################
#dataset = pd.read_csv('./datasets_classifier/test/weatherAUS.csv')
#dataset = dataset.drop('RISK_MM', axis=1)
#targetname = 'RainTomorrow'
dataset = pd.read_csv('./datasets_classifier/test/titanic.csv')
targetname = 'Survived'
#dataset[targetname] = dataset[targetname].map({'No': 0, 'Yes' : 1}).astype(int)
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