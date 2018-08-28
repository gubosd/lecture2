# ml_02.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

iris = load_iris()
X_train,X_test,y_train,y_test = train_test_split(iris.data,iris.target)

model=LinearRegression()
model.fit(X_train[:,2].reshape(-1,1),X_train[:,3])
w = model.coef_[0]
b = model.intercept_

plt.title('petal length vs petal width')
plt.scatter(X_train[:,2],X_train[:,3],c=y_train,s=60)
plt.xlabel(iris.feature_names[2])
plt.ylabel(iris.feature_names[3])
plt.colorbar()
plt.plot([0,8],[0*w+b,8*w+b],'g',lw=10,alpha=0.5)
plt.text(0,3,'coef: %f\nintercept: %f' % (w,b),
    va='top',fontsize=15,color='b')
plt.show()
