# Esteban Murillo
# ITP 449
# Project
# Question 3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

pd.set_option('display.max_columns', None)

myDF = pd.read_csv('UniversalBank_1_.csv')

print(myDF)

# A. What is the target variable? Personal loan, trying to predict whether user accepted or rejected it

# B
myDF.drop(['Row', 'ZIP Code'], axis=1, inplace = True)

print(myDF)

personal_loan = myDF['Personal Loan']

X = myDF.loc[:, myDF.columns != 'Personal Loan']
y = myDF['Personal Loan']

# this is the loan column, which is the target

print(X)

myScaler = StandardScaler()
myScaler.fit(X)
x_scaled = pd.DataFrame(myScaler.transform(X), columns = X.columns)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=2024, stratify = y)

# D.

LogReg = LogisticRegression()
LogReg.fit(X_train, y_train)
print(LogReg.classes_)
y_pred = LogReg.predict(X_test)

from sklearn import metrics

cnfMatrix = metrics.confusion_matrix(y_test, y_pred)
print(cnfMatrix)

from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()


# Extract values from the confusion matrix

dt = DecisionTreeClassifier(criterion = 'entropy', max_depth = 3, random_state=2024)
dt.fit(X_train, y_train)

# E. Plot the classification tree Use entropy criterion. max_depth = 3, random_state = 2024
from sklearn.tree import DecisionTreeClassifier, plot_tree
dt = DecisionTreeClassifier(criterion='entropy', random_state=2024, max_depth=3)
dt.fit(X_train, y_train)
#class_names = list(map(str, dt.classes_.tolist()))
plot_tree(dt, feature_names=X.columns, class_names=list(map(str, dt.classes_.tolist())), filled=True)
plt.show()

'''cm = metrics.confusion_matrix(y_test, y_pred)
false_negatives = cm[1, 0]
false_positives = cm[0, 1]
print(false_negatives, false_positives)'''

Y_pred = dt.predict(X_test)
# F. On the testing partition, how many acceptors did the model classify as non-acceptors?
# calculate false positives
# It classified 19 acceptors as non-acceptors.

# G. On the testing partition, how many non-acceptors did the model classify as acceptors?
# calculate false negatives

# It classified 7 non-acceptors as acceptors

# H. What was the accuracy on the training partition?
trainAcc = accuracy_score(y_train, dt.predict(X_train))
print(trainAcc)
# The accuracy score was 97.84%

# I. What was the accuracy on the test partition
testAcc = accuracy_score(y_test, Y_pred)
print(testAcc)
# The accuracy score was 97.92%



