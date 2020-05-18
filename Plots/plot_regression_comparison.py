import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification, make_regression
from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegressionCV, LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA

from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
#"Nearest Neighbors(k=1)","Nearest Neighbors(k=3)"
#"Decision Tree (Max Depth = 2)","Decision Tree"
#"Linear Regression"
names = ["Decision Tree (Max Depth = 5)","Decision Tree"]
datasets = [make_regression(n_samples=200,n_features=2,noise=0.1, random_state=1),
            make_regression(n_samples=200,n_features=5,noise=0.1, random_state=1),
            make_regression(n_samples=200,n_features=50,noise=0.1, random_state=1)
            #make_regression(n_samples=50,n_features=50,noise=0.1, random_state=1)
            ]
#ds = make_regression(n_samples=100,n_features=1,noise=0.1, random_state=0)
regressors = [  #LogisticRegressionCV,
                #KNeighborsRegressor(n_neighbors=1),
                #KNeighborsRegressor(n_neighbors=3),
                #LinearRegression(),
                #LinearRegression(normalize=False)
                DecisionTreeRegressor(max_depth=5),
                DecisionTreeRegressor()
                #LogisticRegressionCV()
]
figure = plt.figure(figsize=(27, 9))
h = .02
i=1
for ds_cnt, ds in enumerate(datasets):
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.4, random_state=42)

    ax = plt.subplot(len(datasets), len(regressors) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    train = np.array(y_train)
    ax.plot(train)
    i += 1

    for name, clf in zip(names, regressors):
        ax = plt.subplot(len(datasets), len(regressors)+1,i)
        clf.fit(X_train,y_train)
        score = clf.score(X_test,y_test)

        test = np.array(y_test)
        predict = np.array(clf.predict(X_test))
        ax.plot(test)
        ax.plot(predict)
        name = name + " | Score=" + ('%.2f' % score)
        ax.set_title(name)
        i+=1
#train = np.array(y_train)
#test = np.array(y_test)
#predict = np.array(model.predict(X_test))
#ax.plot(test)
#ax.plot(predict)
plt.tight_layout()
plt.show()