# Esteban Murillo
# ITP 449
# Project
# Question 4

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)

myDF = pd.read_csv('mushrooms_1_.csv')

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import plot_tree
from sklearn import metrics

X = myDF.drop(['class'], axis=1)
Xdummy = pd.get_dummies(X, drop_first=True)
y = myDF['class']
print(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(Xdummy, y, test_size=0.25, random_state=2024, stratify=y)

# Create a Decision Tree Classifier
clf = DecisionTreeClassifier(criterion='entropy', max_depth=6, random_state=2024)

# Fit the classifier to the training data
clf.fit(X_train, y_train)

# Make predictions on the test set


# Confusion Matrix
y_pred = clf.predict(X_test)
cf = metrics.confusion_matrix(y_test, y_pred)
print(cf)

# Visualize the confusion matrix

metrics.ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)
plt.show()

# B
# Accuracy on the training partition
y_pred_train = clf.predict(X_train)
print(accuracy_score(y_train, y_pred_train))
# 1

# C
# Accuracy on the test partition
print(accuracy_score(y_test, y_pred))

# Visualize the Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=Xdummy.columns, class_names=['e', 'p'])
plt.show()


'''
odor_n                        0.528111
bruises_t                     0.215596
stalk-surface-below-ring_y    0.078196
'''

toxic_names = ['Poisonous', 'Edible']
plot_tree(clf, feature_names=Xdummy.columns, class_names=toxic_names, filled=True)
plt.show()

feature_importances = pd.Series(clf.feature_importances_, index=Xdummy.columns).sort_values(ascending=False)
print(feature_importances.head(3))



# Classify the given mushroom
X2 = X.copy()
X2.loc[-1, :] = ['x', 's', 'n', 't', 'y', 'f', 'c', 'n', 'k', 'e', 'e', 's', 's', 'w', 'w', 'p', 'w', 'o', 'p', 'r', 's', 'u']
X2Dummies = pd.get_dummies(data=X2, drop_first=True)
sampleDummies = X2Dummies.tail(1)
print(clf.predict(sampleDummies))