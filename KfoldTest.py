import numpy as np
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

# Initialize the DS techniques. DS methods can be initialized without
# specifying a single input parameter. In this example, we just pass the random
# state in order to always have the same result.
kne = KNORAE(random_state=rng)
meta = METADES(random_state=rng)
k_fold = KFold(n_splits=5)
scores = cross_val_score(kne, X_dsel,y_dsel,cv=k_fold, scoring=make_scorer(cohen_kappa_score))

print('Cross Val Scores: ', scores)
print('Cross Val Score mean: ', scores.mean())