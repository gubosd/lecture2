# ml_10.py
# Kernel SVM (normalize)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

cancer = load_breast_cancer()
X_train,X_test,y_train,y_test = train_test_split(cancer.data,cancer.target)

plt.boxplot(X_train,manage_xticks=False)
plt.yscale('symlog')
plt.xlabel('features')
plt.ylabel('scale')
plt.show()

X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)
X_train_scaled = (X_train-X_mean)/X_std
X_test_scaled = (X_test-X_mean)/X_std

for C in [0.1,1,1000]:
	for gamma in [0.1,1,10]:
		model=SVC(C=C,gamma=gamma)
		model.fit(X_train_scaled,y_train)

		pred_y=model.predict(X_test_scaled)
		print('\n>>> C: %.1f, gamma: %.1f' % (C,gamma))
		print('Train Score: %f' % model.score(X_train_scaled,y_train))
		print('Test Score: %f' % model.score(X_test_scaled,y_test))