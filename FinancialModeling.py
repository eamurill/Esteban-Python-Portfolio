# Esteban Murillo
# BUAD 281
# Financial Modeling

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

financial_dict = {'Units per quarter:': [7000, 10000, 12500, 15000,],
                  'Total Units': 44500,
                  'Sales Revenue': [560000, 800000, 1000000, 1200000],
                  'Total Sales Revenue': 3560000,
                  'Cost of One Unit': 32}

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.DataFrame({'Units': [7000, 10000, 12500, 15000]}, index=[1, 2, 3, 4])

# Prepare the data for training
X = df.index.values.reshape(-1, 1)  # Features (time index)
y = df['Units'].values

# Split the data into training and testing sets (optional for this simple model)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)


# Predict future units
future_years = np.array([[5], [6], [7]])  # For years 2025, 2026, 2027
predicted_units = model.predict(future_years)

print("Predicted units for 2025, 2026, and 2027:")
print(predicted_units)

current_year_units = [7000, 10000, 12500, 15000]

# Assuming a linear growth rate based on the current trend
growth_rate = (15000 - 7000) / 3  # Average quarterly growth

# Predict units for 2025, 2026, and 2027 (assuming 4 quarters per year)
predicted_units_2025 = [15000 + growth_rate * i for i in range(1, 5)]
predicted_units_2026 = [predicted_units_2025[-1] + growth_rate * i for i in range(1, 5)]
predicted_units_2027 = [predicted_units_2026[-1] + growth_rate * i for i in range(1, 5)]

# Combine all the data into a single list
all_units = current_year_units + predicted_units_2025 + predicted_units_2026 + predicted_units_2027

# Create a list of years for the x-axis
years = list(range(1, 17))

# Plot the data
plt.plot(years, all_units, marker='o')
plt.xlabel('Quarter')
plt.ylabel('Units')
plt.title('Units per Quarter for EcoStool Inc')
plt.grid(True)
plt.show()

plt.savefig('EcoStoolInc.png')

growth_rate = (15000 - 7000) / 3  # Average quarterly growth

print("Average quarterly growth rate:", growth_rate)

quarters_in_2025 = 4
quarters_in_2026 = 8

# Predicted units for 2025 and 2026
predicted_units_2025 = 15000 + growth_rate * quarters_in_2025
predicted_units_2026 = predicted_units_2025 + growth_rate * quarters_in_2026

# Predicted revenue for 2025 and 2026
predicted_revenue_2025 = predicted_units_2025 * 32  # Assuming the same unit price
predicted_revenue_2026 = predicted_units_2026 * 32

print("Predicted units for 2025:", predicted_units_2025)
print("Predicted revenue for 2025:", predicted_revenue_2025)
print("Predicted units for 2026:", predicted_units_2026)
print("Predicted revenue for 2026:", predicted_revenue_2026)