# ml_11.py
# k-means

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

iris = load_iris()
X = iris.data[:,2:]
y = iris.target

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

print(kmeans.labels_)

plt.title('k-means (for Iris)',fontsize=20)
plt.scatter(X[:,0],X[:,1],c=y,s=60)
cc=kmeans.cluster_centers_
plt.scatter(cc[:,0],cc[:,1],s=400,marker='*',c='r')
plt.xlabel(iris.feature_names[2])
plt.ylabel(iris.feature_names[3])
plt.show()
