# Esteban Murillo
# ITP 449
# Project
# Question 5

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)

diabetes_knn = pd.read_csv('diabetes_data.csv')

print(diabetes_knn)

num_rows, num_cols = diabetes_knn.shape

print("Number of rows:", num_rows)
print("Number of columns:", num_cols)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

X = diabetes_knn.iloc[:, :-1]  # Features (all columns except the last one)
y = diabetes_knn.iloc[:, -1]  # Target variable (last column)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into train A and train B sets
X_train_A, X_train_B, y_train_A, y_train_B = train_test_split(X_scaled, y, test_size=0.3, random_state=2024, stratify=y)

# KNN model evaluation with varying K values
knn_scores_train_A = []
knn_scores_train_B = []
for k in range(1, 10):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_A, y_train_A)
    y_pred_A = knn.predict(X_train_A)
    y_pred_B = knn.predict(X_train_B)
    knn_scores_train_A.append(accuracy_score(y_train_A, y_pred_A))
    knn_scores_train_B.append(accuracy_score(y_train_B, y_pred_B))

# Plot the KNN scores for train A and train B
plt.plot(range(1, 10), knn_scores_train_A, label='Train A')
plt.plot(range(1, 10), knn_scores_train_B, label='Train B')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('KNN Score (Accuracy)')
plt.title('KNN Score vs. K for Train A and Train B')
plt.legend()
plt.show()

# Choose the best K based on the plot (e.g., the K with highest Train B score)
best_k = 8 # Choose the best k value based on the plot

# Train the final KNN model with the best K on Train A
knn_final = KNeighborsClassifier(n_neighbors=best_k)
knn_final.fit(X_train_A, y_train_A)

# Make predictions on Train B data using the best KNN model
y_pred_A = knn_final.predict(X_train_A)
y_pred_B = knn_final.predict(X_train_B)

# Display the KNN score (accuracy) on Train B
accuracy_train_A = accuracy_score(y_train_A, y_pred_A)
accuracy_train_B = accuracy_score(y_train_B, y_pred_B)
print("KNN Score (Accuracy) on Train A with K={} : {:.2f}".format(best_k, accuracy_train_A))
print("KNN Score (Accuracy) on Train B with K={} : {:.2f}".format(best_k, accuracy_train_B))

# Confusion matrix for Train B with Train A as reference
cm = confusion_matrix(y_train_B, y_pred_B)
print("Confusion Matrix:")
print(cm)

# Plot the confusion matrix
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.title("Confusion Matrix (Train B)")
plt.show()

# New data for prediction
new_data = [[1, 150, 60, 12, 300, 28, 0.4, 45]]  # pregnancies, glucose, blood pressure, skin thickness, insulin, BMI, diabetes pedigree, age

# Predict outcome for the new data
prediction = knn_final.predict(new_data)

if prediction[0] == 0:
  print("Predicted Outcome: No Diabetes")
else:
  print("Predicted Outcome: Diabetes")



