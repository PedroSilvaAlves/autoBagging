from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd
import math as m

class DESIP(BaseEstimator):
    def __init__(self, bagging):
        super().__init__()
        self.bagging = bagging
        self.base_estimators = bagging.estimators_

    def fit(self, X_train, y):
        
        preds = []
        for estimator, features in zip(self.bagging.estimators_,self.bagging.estimators_features_):
            preds.append(estimator.predict(X_train[:, features]))
        target = y
        data = X_train
        A = []
        N = data.shape[0]
        M = len(preds)
        for n in range(N):
            S = []
            C = []
            for m in range(M):
                C.append(preds[m][n] - target[n])
            for u in range(M):
                minimo = 10.000e10
                for k in range(M):
                    if k not in S:
                        z = 0
                        for i in range(u):
                            z = z + C[i]
                        z = z + C[k]
                        z = pow(u,-1 * z)
                        if z < minimo:
                            S.append(k)
                            minimo = z
            A.append(S)
        self.A = A
        self.X_train = X_train
        return self

    def predict(self, X_test):
        X_train = self.X_train
        N = len(self.A[0])
        G = []
        F = X_test.shape[1]
        for n in range(N):
            E = []
            for f in range(F):
                E.append(X_train[f] - X_test[f])
            G.append(sum(E))
        bestEnsemble = np.argmin(G)
        estimators = []
        for i in self.A[bestEnsemble]:
            estimators.append(self.base_estimators[i])
        self.bagging.estimators_ = estimators
        return self.bagging.predict(X_test)

    