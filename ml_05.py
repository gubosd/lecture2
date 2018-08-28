# ml_05.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

iris = load_iris()
X_train,X_test,y_train,y_test = train_test_split(iris.data,iris.target,random_state=1)

model=LogisticRegression() # default value C=1.0
model.fit(X_train,y_train)

pred_y=model.predict(X_test)
print('Score: %f' % model.score(X_test,y_test))

print(pred_y)
print(y_test)
print('Score: %f (Error: %d)' % (np.mean(pred_y==y_test),np.sum(pred_y!=y_test)) )
print(np.where(pred_y!=y_test))
print(model.predict_proba(X_test[5:6]))

print(model.coef_)
print(model.intercept_)