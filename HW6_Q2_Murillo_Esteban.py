# Esteban Murillo
# ITP 449
# HW 6
# Q2
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



commute = pd.read_csv('../CommuteStLouis.csv')

print(commute)

print(commute.columns)

print(commute.describe())

plt.hist(commute['Age'], bins=10, edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Histogram of Age Distribution')
plt.show()

corr_matrix = commute[['Age', 'Distance', 'Time']].corr()
print(corr_matrix)

# Visualize the correlation matrix
sns.pairplot(commute[['Age', 'Distance', 'Time']])
plt.show()

sns.boxplot(x=commute['Sex'], y=commute['Distance'], data=commute, hue = commute['Sex'])
plt.title('Distance Commuted by Gender')
plt.show()



from sklearn.linear_model import LinearRegression
import numpy as np

model = LinearRegression()
X = commute[["Distance"]]
y = commute["Time"]
model.fit(X, y)

myFig = plt.figure(figsize=(12, 5))
xLine = np.linspace(0, 80, num=2).reshape(-1, 1)
yLine = model.predict(xLine)
ax1 = myFig.add_subplot(1, 2, 1)
ax1.scatter(X, y)
ax1.plot(xLine, yLine, "-")
ax1.set_xlabel("Distance")
ax1.set_ylabel("Time")
ax1.set_title("Scatter plot and Linear Regression of Time vs Distance")
plt.show()



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

print(f"Regression Equation: mpg = {intercept:.2f} + {coefficient:.2f} * displacement")
