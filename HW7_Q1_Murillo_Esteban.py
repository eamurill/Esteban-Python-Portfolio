# Esteban Murillo
# ITP 449
# HW7
# Question 1

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split


insurance = pd.read_csv('../insurance.csv')

print(insurance)

sns.catplot(x = insurance['sex'], y = insurance['charges'], hue = insurance['sex'], col = insurance['smoker'],
            kind = 'bar', data = insurance)
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.scatter(x = 'age', y = 'charges', data = insurance)
ax1.set_ylabel('Charges')
ax1.set_xlabel('Age')
ax2.scatter(x = 'bmi', y = 'charges', data = insurance)
ax2.set_ylabel('Charges')
ax2.set_xlabel('BMI')
plt.tight_layout()
plt.show()

X = insurance[['age', 'bmi', 'smoker']]
y = insurance[['charges']]

X_dummy = pd.get_dummies(X, columns=['smoker'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X_dummy, y, test_size=0.3, random_state=2024)

from sklearn.linear_model import LinearRegression

model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# 7. Create a scatter plot
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Actual vs Predicted Medical Costs")
plt.show()


# 8. Calculate the score of the model for the test set
score = model.score(X_test, y_test)
print("Model Score:", score)

# 9. Predict the medical cost for a 51-year-old non-smoker with BMI 29.1
sample_data = np.array([[51, 29.1, 0]])
sample_df = pd.DataFrame(data=sample_data, columns=X_dummy.columns)
predicted_cost = model.predict(sample_df)
print("Predicted cost for a non-smoker:", predicted_cost)

# 10. Predict the medical cost for the same person as a smoker
sample_data_smoker = np.array([[51, 29.1, 1]])
sample_df_smoker = pd.DataFrame(data=sample_data_smoker, columns=X_dummy.columns)
predicted_cost_smoker = model.predict(sample_df_smoker)
print("Predicted cost for a smoker:", predicted_cost_smoker)