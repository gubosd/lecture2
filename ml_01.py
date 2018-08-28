# ml_01.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X_train,X_test,y_train,y_test = train_test_split(iris['data'],iris['target'])
'''
plt.scatter(X_train[:,0],X_train[:,1],c=y_train,s=60)
plt.show()
'''
fig=plt.figure()
fig.suptitle('Iris (setosa:0, versicolor:1, virginica:2)')
count=0
for i in range(4):
	for j in range(i+1,4):
		count+=1
		plt.subplot(2,3,count)
		plt.scatter(X_train[:,i],X_train[:,j],c=y_train,s=60)
		plt.xlabel(iris.feature_names[i])
		plt.ylabel(iris.feature_names[j])

plt.colorbar()
plt.show()
