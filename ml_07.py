# ml_07.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

iris = load_iris()
X_train,X_test,y_train,y_test = train_test_split(iris.data,iris.target)

for C in [0.01,0.1,1,10,100]:
	model=LinearSVC(C=C) # default value C=1.0
	model.fit(X_train,y_train)

	print('== %f ==' % C)
	print('Train Score: %f' % model.score(X_train,y_train))
	print('Test Score: %f' % model.score(X_test,y_test))