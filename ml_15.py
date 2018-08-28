# ml_15.py
# confusion matrix

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

cancer = load_breast_cancer()
X_train,X_test,y_train,y_test = train_test_split(cancer.data,cancer.target,random_state=1)

model = KNeighborsClassifier()
model.fit(X_train,y_train)

pred_y = model.predict(X_test)
print('Test score :',np.mean(pred_y == y_test))

print('confusion matrix :')
print(confusion_matrix(y_test,pred_y))