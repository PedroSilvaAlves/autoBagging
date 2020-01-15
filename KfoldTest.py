import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score,cross_validate
from sklearn.model_selection import KFold
from sklearn.metrics import cohen_kappa_score, make_scorer
from deslib.des import METADES
from deslib.des import KNORAE


# Setting up the random state to have consistent results
rng = np.random.RandomState(42)

# Generate a classification dataset
X, y = make_classification(n_samples=1000, random_state=rng)
# split the data into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                                                    random_state=rng)

# Split the data into training and DSEL for DS techniques
X_train, X_dsel, y_train, y_dsel = train_test_split(X_train, y_train,
                                                    test_size=0.5,
                                                    random_state=rng)


bagging_workflow = BaggingClassifier(n_estimators=50,random_state=rng)
k_fold = KFold(n_splits=4, random_state=rng)
cross_vals = cross_validate(bagging_workflow,X_train,y_train,cv=k_fold, scoring=make_scorer(cohen_kappa_score), return_estimator=True)
print("Scores: ", cross_vals['test_score'])
bagging_workflow = cross_vals['estimator'][np.argmax(cross_vals['test_score'])]


kne = KNORAE(bagging_workflow,random_state=rng)


# CROSS VALIDATION
#k_fold = KFold(n_splits=5)
kne.fit(X_dsel,y_dsel)
scores = cross_val_score(kne.pool_classifiers, X_train,y_train,cv=k_fold, scoring=make_scorer(cohen_kappa_score))
print('Cross Val Scores: ', scores)
print('Cross Val Score mean: ', scores.mean())

# FIT DESLIB
#kne.fit(X_dsel,y_dsel)
#scores = kne.predict(X_test)
#print('Cohen Kappa Score: ', cohen_kappa_score(scores,y_test))
