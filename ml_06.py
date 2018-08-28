# ml_06.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()

X1=iris.data[iris.target==0]     # (50,4) Setosa
X2=iris.data[iris.target==1]     # (50,4) Versicolor
X3=iris.data[iris.target==2]     # (50,4) Virginica

# LogisticRegression

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X=np.vstack([X1,X2])[:,:2]
y=np.array([0]*50+[1]*50)

model = LogisticRegression()
model.fit(X,y)
pred_y = model.predict(X)
acc=accuracy_score(y,pred_y)
print('accuracy :',acc)

w = model.coef_
b = model.intercept_

# plot results

fig=plt.figure()
fig.suptitle('Setosa - Versicolor')

plt.scatter(X1[:,0],X1[:,1],marker='o',color='r',label='Setosa')
plt.scatter(X2[:,0],X2[:,1],marker='s',color='b',label='Versicolor')
dots=np.array([4,7.5])
plt.plot(dots,-(w[0,0]*dots+b)/w[0,1],'g:',lw=2)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.legend()
plt.show()