# Esteban Murillo
# ITP 449
# Project
# Question 1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)

# A:

wineDF = pd.read_csv('wineQualityReds.csv')

print(wineDF)

# B:

from sklearn.preprocessing import StandardScaler

# Assuming 'X' is your feature matrix

X = wineDF.iloc[:, :-1]  # Features
y = wineDF.iloc[:, -1]  # Target variable

# Standardize the features
scaler = StandardScaler()
scaler.fit(X)
X_scaled = pd.DataFrame(scaler.transform(X), columns = X.columns)

# C:

from sklearn.model_selection import train_test_split

# Partition the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=2024, stratify=y)

# Now, X_train and X_test contain the scaled features, and y_train and y_test contain the target variable.

X_trainA, X_trainB, y_trainA, y_trainB = train_test_split(X_train, y_train, test_size=0.25, random_state=2024, stratify=y_train)

# D:

from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

trainA_accuracy = []
trainB_accuracy = []

# Iterate over different values of K
for k in range(1, 31):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_trainA, y_trainA)
    y_predA = knn.predict(X_trainA)
    trainA_accuracy.append(metrics.accuracy_score(y_trainA, y_predA))
    y_predB = knn.predict(X_trainB)
    trainB_accuracy.append(metrics.accuracy_score(y_trainB, y_predB))



# Plot the accuracy scores
plt.plot(range(1, 31), trainA_accuracy, '--r', label='trainA_accuracy')
plt.plot(range(1, 31), trainB_accuracy, '--r', label='trainB_accuracy')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Accuracy')
plt.title('KNN Accuracy vs. K')
plt.show()

# 25 produced the best accuracy of .575

optimal_k = 20

# Train the final KNN model with the optimal K
knn_final = KNeighborsClassifier(n_neighbors = 15)
knn_final.fit(X_trainA, y_trainA)
y_pred_test = knn_final.predict(X_test)
print("Accuracy on Test Set: ", metrics.accuracy_score(y_test, y_pred_test))

metrics.confusion_matrix(y_test, y_pred_test)
metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test)
print(metrics.confusion_matrix(y_test, y_pred_test))
plt.show()

qualityDF = X_test.copy()
qualityDF['quality'] = y_test
qualityDF['Predicted Quality'] = y_pred_test
print(qualityDF)

print("The Model Accuracy: ", metrics.accuracy_score(y_test, y_pred_test))