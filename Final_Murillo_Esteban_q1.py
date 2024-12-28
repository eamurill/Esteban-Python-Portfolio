# Esteban Murillo
# ITP 449
# Final Exam
# Q1

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

wineDf = pd.read_csv('wineQualityReds.csv')

print(wineDf)
print(wineDf.info)
print(wineDf.shape)

# The dimensions are 1599 by 13

print(wineDf.isnull().sum())

wineDf = wineDf.dropna()

# There are no missing values

wineDf.drop(columns = ['Wine'], axis = 1, inplace = True)

X = wineDf.drop(columns='quality')
y = wineDf.quality

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=2024, stratify = y)

X_trainA, X_trainB, y_trainA, y_trainB = train_test_split(X_train, y_train, test_size=1/3, random_state=2024, stratify=y_train)

print()

print("Cases in X_train: ", X_train.shape)
print("Cases in Y_train: ", X_test.shape)

print()


from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

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

print(trainA_accuracy)
print(trainB_accuracy)

plt.plot(range(1, 31), trainA_accuracy, '--r', label='trainA_accuracy')
plt.plot(range(1, 31), trainB_accuracy, '--b', label='trainB_accuracy')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Accuracy')
plt.title('KNN Accuracy vs. K')
plt.show()

optimal_k = 27

from sklearn.metrics import accuracy_score

knn_final = KNeighborsClassifier(n_neighbors = optimal_k)
knn_final.fit(X_trainA, y_trainA)
y_pred_test = knn_final.predict(X_test)

y_pred_a = knn_final.predict(X_test)
y_pred_b = knn_final.predict(X_test)

accuracy_a = accuracy_score(y_test, y_pred_a)
accuracy_b = accuracy_score(y_test, y_pred_b)

print("Accuracy on Train A:", accuracy_a)
print("Accuracy on Train B:", accuracy_b)

metrics.confusion_matrix(y_test, y_pred_test)
metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test)
print(metrics.confusion_matrix(y_test, y_pred_test))
plt.show()

wine_data = pd.DataFrame([[8, .6, 0, 2.0, .067, 10, 30, .9978, 3.20, .5, 10.0]], columns = X.columns)
prediction = knn_final.predict(wine_data)
print(prediction)

# The wine Quality is 5

# Hints for question

# Drop 1

# Print .shape