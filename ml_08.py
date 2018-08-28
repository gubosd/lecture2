# ml_08.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X_train,X_test,y_train,y_test = train_test_split(iris.data,iris.target,random_state=1)

model=DecisionTreeClassifier(max_depth=2)
model.fit(X_train,y_train)

pred_y=model.predict(X_test)
print('Train Score: %f' % model.score(X_train,y_train))
print('Test Score: %f' % model.score(X_test,y_test))

print(model.feature_importances_)
print(model.tree_) # help(model.tree_)
