# Esteban Murillo
# ITP 449
# HW6
# Q1

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)

avocado = pd.read_csv('../avocado.csv')

print(avocado.columns)

avocado = avocado[['Date', 'AveragePrice', 'Total Volume']].copy()
print(avocado)

avocado['Date'] = pd.to_datetime(avocado['Date'])

print(avocado)

# Part 2: Plotting

# Figure with 4 subplots

# Sort avocado by Date in ascending order
avocado.sort_values(by='Date',ascending=True, inplace=True)


myFig = plt.figure()
# Plot the average price of avocados over time in subplot 1
ax1 = myFig.add_subplot(2,2,1)
ax1.scatter(x=avocado['Date'], y = avocado['AveragePrice'])
ax1.set_ylabel('Average Price')

# Plot the total volume of avocados sold over time use scatter
ax2 = myFig.add_subplot(2,2,2)
ax2.scatter(x=avocado['Date'], y = avocado['Total Volume'])

# Create a new column in avocado called TotalRevenue
avocado['TotalRevenue'] = avocado['AveragePrice']*avocado['Total Volume']
avocado1 = avocado.groupby('Date').sum()
print(avocado1) # AveragePrice also got aggregated (which we don't want)

avocado1['AveragePrice'] = avocado1['TotalRevenue']/avocado1['Total Volume']

# Print again
print(avocado1)

# Plot the average price of avocado 1 over time in subplot 3. Using plot
ax3 = myFig.add_subplot(2,2,3)
ax3.plot(avocado1.index, avocado1['AveragePrice'])

# Plot the total volume of avocado1 sold over time in subplot 4. Using plot
ax4 = myFig.add_subplot(2,2,4)
ax4.plot(avocado1.index, avocado1['Total Volume'])
plt.show()

# Part 3: Plotting

# Create a figure with 2 subplots

# Smooth out the last two plots from question 2

myFig = plt.figure()
ax1 = myFig.add_subplot(1,2,1)
ax1.plot(avocado1.index, avocado1['AveragePrice'].rolling(20).mean())
ax1.set_xticklabels(avocado1.index, rotation = 90)
ax1.set_ylabel('Average Price')


ax2 = myFig.add_subplot(1,2,2)
ax2.plot(avocado1.index, avocado1['Total Volume'].rolling(20).mean())
ax2.set_xticklabels(avocado1.index, rotation = 90)
ax2.set_ylabel('Total Volume')

plt.title("Average Prices and Volumes Time Series")
plt.show()

