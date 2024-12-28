# Esteban Murillo
# ITP 449
# HW 7
# Q3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

myDF = pd.read_csv('../Titanic_simplified.csv')

print(myDF)

print(myDF.isnull().sum())

new_DF = myDF.drop('Passenger', axis = 1)

print(new_DF)

fig, ax = plt.subplots(2, 2, )

sns.countplot(x='Class', data=myDF, ax=ax[0, 0], hue = 'Class')
ax[0, 0].set_title('Distribution of Passenger Classes')

# Plot 2: Sex distribution
sns.countplot(x='Sex', data=myDF, ax=ax[0, 1], hue = 'Sex')
ax[0, 1].set_title('Distribution of Passengers by Sex')

# Plot 3: Age distribution (assuming 'Age' is categorical or binned)
sns.countplot(x='Age', data=myDF, ax=ax[1, 0], hue = 'Age')
ax[1, 0].set_title('Distribution of Passenger Ages')

# Plot 4: Survival distribution
sns.countplot(x='Survived', data=myDF, ax=ax[1, 1], hue = 'Survived')
ax[1, 1].set_title('Survival Distribution')

plt.tight_layout()
plt.show()


# Convert categorical variables to dummy variables
myDF = pd.get_dummies(myDF, columns=['Age', 'Class', 'Sex', 'Survived'], drop_first=True)
print(myDF)


X_dummy = myDF[['Age_Child', 'Class_2nd', 'Class_3rd', 'Class_Crew', 'Sex_Male']]
y_dummy = myDF['Survived_Yes']

print(X_dummy)
print(y_dummy)

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_dummy, y_dummy, test_size = 0.25, random_state = 2024)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

model = LogisticRegression()

# 8. Fit the model to the training data
model.fit(X_train, y_train)

# 9. Predict on the test set
y_pred = model.predict(X_test)

# 10. Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# 11. Display the confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No', 'Yes'])
disp.plot()

plt.show()

print(X_dummy)
# 12. Predict the survivability of an adult female passenger traveling 2nd class
print(X_dummy.columns)
my_sample_DF = pd.DataFrame([[False, True, False, False, False]], columns=X_dummy.columns)
print(my_sample_DF)

new_y_pred = model.predict(my_sample_DF)

print(new_y_pred)

# This answer is True, the passenger will survive



# model.predict(sample data)

