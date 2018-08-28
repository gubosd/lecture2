# ml_18.py
# Pearson correlation coefficient

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

cancer = load_breast_cancer()
mat=np.corrcoef(cancer.data.T)
print(mat)

plt.title('Breast Cancer - Correlation Coefficient',fontsize=15)
plt.imshow(mat,interpolation='none')
plt.colorbar()
plt.xticks(range(30),cancer.feature_names,rotation=60)
plt.show()