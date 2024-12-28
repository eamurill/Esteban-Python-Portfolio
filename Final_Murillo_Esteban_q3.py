# Esteban Murillo
# ITP 449
# Final Exam
# Question 3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# The target value is dpnm is the target column

# want to get top four features for the model to get the highest correlation

# Payments 1, 2, 3,4 for, features have highest correlation with my target

# those will be features for the model I have to make

# Use Decision Tree Classifier

# Create a Pd.Series, feature imoprtance borrow the things, sort it, and see which features

# indices of series will be columns

# The most important features will be shown with descending order

ccDefaults = pd.read_csv('ccDefaults.csv')

pd.set_option('display.max_columns', None)

print(ccDefaults)

print("Info: ", ccDefaults.info) # the number of non-null samples and feature datatypes is used by .info
print(ccDefaults.dtypes)

print(ccDefaults.head(5)) # the first 5 rows

print(ccDefaults.shape) # the dimensions of ccDefaults

ccDefaults.drop(columns = ['ID'], inplace = True)

print(ccDefaults)

ccDefaults.drop_duplicates(keep = 'first', inplace = True)

from sklearn.metrics import confusion_matrix

corr_matrix = ccDefaults.corr()
print(corr_matrix)

# Here is my feature matrix

X = ccDefaults[['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4']]
y = ccDefaults['dpnm']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 2024, stratify = y)

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(criterion = 'gini', max_depth = 3, random_state = 2024)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn import metrics

acc = accuracy_score(y_test, y_pred)
print("Accuracy score on the test set: ", acc)

cm = confusion_matrix(y_test, y_pred)
print(cm)
metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()

from sklearn import tree

class_names = list(map(str, dt.classes_.tolist()))
tree.plot_tree(dt, feature_names = X.columns, class_names = class_names, filled = True)
plt.show()

features = pd.Series(data = dt.feature_importances_, index = X.columns)

features.sort_values(ascending = False, inplace = True)

print("Here are the features: ", features)





