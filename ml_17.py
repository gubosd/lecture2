# ml_17.py
# pipeline

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

cancer = load_breast_cancer()
X_train,X_test,y_train,y_test = train_test_split(cancer.data,cancer.target)

scaler=MinMaxScaler().fit(X_train)
X_train_scaled=scaler.transform(X_train)
X_test_scaled=scaler.transform(X_test)

model=SVC()
model.fit(X_train_scaled,y_train)

print('Train Score: %f' % model.score(X_train_scaled,y_train))
print('Test Score: %f' % model.score(X_test_scaled,y_test))

### pipeline

pipe=Pipeline([ ('scaler',MinMaxScaler()), ('svm',SVC()) ])
pipe.fit(X_train,y_train)
print('Pipe Score :',pipe.score(X_test,y_test))