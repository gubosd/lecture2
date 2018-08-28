# ml_13.py
# cross-validation

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

iris = load_iris()
model = SVC()

for cv in [2,3,4,5]:
	scores = cross_val_score(model,iris.data,iris.target,cv=cv)
	print('cv=%d :' % cv,scores)