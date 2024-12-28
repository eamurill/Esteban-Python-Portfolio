# Esteban Murillo
# ITP 449
# HW 7
# Question 2

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

myDF = pd.read_csv('../auto_mpg.csv')
print(myDF.describe())


print(myDF['mpg'].mean())

# 23.514572864321607 is the mean

print(myDF['mpg'].median())

# 23.0 is the median

# The mean is greater


plt.hist(myDF['mpg'], bins=20, edgecolor='black')
plt.xlabel('MPG')
plt.ylabel('Frequency')
plt.title('Histogram of MPG')
plt.show()

# this plot is skewed to the right, meaning the mean is greater than the median

plotting_DF = myDF.select_dtypes(include='number').columns.tolist()
plotting_DF.remove('No')  # Remove 'No' column if it exists

corr_matrix = myDF[plotting_DF].corr()

sns.heatmap(corr_matrix, annot=True)

print(corr_matrix)
# use myDF.corr

sns.pairplot(myDF[plotting_DF])
plt.show()

# 5. the highest correlation is cylinders and horsepower

# 6. the lowest correlations is model year and acceleration

# please see my correlation heatmap which is where I got my results

# 7.

sns.scatterplot(x='displacement', y = 'mpg', data=myDF)
plt.show()

# 8.

from sklearn.linear_model import LinearRegression

X = myDF[['displacement']]
y = myDF['mpg']

# Create a linear regression model
model = LinearRegression()

# Fit the model to the data
model.fit(X, y)

from yellowbrick.regressor import ResidualsPlot

#y_predicted = model.predict(X)

# Calculate residuals
#residuals = y - y_predicted

# Create a residuals plot using Yellowbrick

residuals_plot = ResidualsPlot(model)
residuals_plot.fit(X, y)  # Fit the model to the data
#residuals_plot.plot(X, residuals)  # Plot the residuals
residuals_plot.show()

intercept = model.intercept_

# Coefficient (β1)
coefficient = model.coef_[0]  # Access the first coefficient

print("Intercept (β0):", intercept) # = 6.48
print("Coefficient (β1):", coefficient) # 1.1

print(f"Regression equation: mpg = {intercept:.2f} + {coefficient:.2f} * displacement")

print(35.17 + (-.06 * 2)) # = 35.05

print(35.17 + (-.06 * 10)) # = 34.57

# As displacement increases, our predicted miles per gallon will decrease

print(35.17 + (-.06 * 200))

# The predicted miles per gallon with a value of 200 for displacement would yield 23.17 mpg.

plt.scatter(X['displacement'], y)
plt.plot(X['displacement'], model.predict(X), color='red')
plt.xlabel('Displacement')
plt.ylabel('MPG')
plt.title('MPG vs Displacement')
plt.show()

# Plot the residuals
plt.scatter(model.predict(X), y - model.predict(X))
plt.xlabel('Predicted MPG')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()