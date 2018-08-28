# ml_09.py
# Kernel SVM

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

cancer = load_breast_cancer()
X_train,X_test,y_train,y_test = train_test_split(cancer.data,cancer.target)

for C in [0.1,1,1000]:
	for gamma in [0.1,1,10]:
		model=SVC(C=C,gamma=gamma)
		model.fit(X_train,y_train)

		print('\n>>> C: %.1f, gamma: %.1f' % (C,gamma))
		print('Train Score: %f' % model.score(X_train,y_train))
		print('Test Score: %f' % model.score(X_test,y_test))

pred_y=model.predict(X_test)
print(np.where(pred_y!=y_test))