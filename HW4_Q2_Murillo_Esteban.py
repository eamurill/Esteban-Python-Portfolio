# Esteban Murillo
# ITP 449
# HW4
# Q2

import pandas as pd
import matplotlib.pyplot as plt

# Load the data from a CSV file (replace 'your_data.csv' with the actual file path)
data = pd.read_csv('../data.csv')

print(data)

# Extracting the Year and Value columns
years = data['Year']
temperatures = data['Value']

# Creating the plot using indexing plus laeling my plot
plt.plot(data['Year'], data["Value"], 'ro--')
plt.title("Global temperature")
plt.xlabel("Year")
plt.ylabel("Temperature Anomaly")
plt.show()



