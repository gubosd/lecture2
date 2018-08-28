# ml_14.py
# cross-validation (KFold)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold

iris = load_iris()
model = SVC()
kfold=KFold(n_splits=5,shuffle=True,random_state=55)

scores = cross_val_score(model,iris.data,iris.target,cv=kfold)
print(scores)