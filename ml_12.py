# ml_12.py
# breast cancer data analysis

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

cancer = load_breast_cancer()
malignant = cancer.data[cancer.target==0]
benign = cancer.data[cancer.target==1]

fig=plt.figure()
fig.suptitle('Breast Cancer')
for col in range(cancer.feature_names.shape[0]): # 30 features
	plt.subplot(8,4,col+1)
	_,bins=np.histogram(cancer.data[:,col],bins=50)
	plt.hist(malignant[:,col],bins=bins,alpha=0.5,label='malignant')
	plt.hist(benign[:,col],bins=bins,alpha=0.5,label='benign')
	plt.title(cancer.feature_names[col])
	plt.xticks([])
	plt.yticks([])
	if col==0: plt.legend()
plt.show()