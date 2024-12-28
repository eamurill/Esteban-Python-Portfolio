# Esteban Murillo
# ITP 449
# Final Exam
# Question 2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

bc = pd.read_csv('Breast_Cancer.csv')

print(bc.head())

print(bc.isnull().sum())

# There are no missing values

print(bc.columns)

print(bc['diagnosis'])

# Malignant and Benign

diagnosis = bc.diagnosis

bc.drop(['diagnosis'], axis=1, inplace=True)

print(bc)

sns.countplot(bc, x=diagnosis, hue=diagnosis)
plt.show()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(bc, diagnosis, test_size=0.25, random_state=0)

# Use iloc

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter = 10000) # letting python to try more
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn import metrics
from sklearn.metrics import confusion_matrix

print(metrics.classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import ConfusionMatrixDisplay

metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()