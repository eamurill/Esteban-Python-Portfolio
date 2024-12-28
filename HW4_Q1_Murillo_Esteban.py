# Esteban Murillo
# ITP 449
# HW4
# Q1

import numpy as np
import matplotlib.pyplot as plt

# Create 200 random integers between 1 and 200
X = np.random.randint(1, 201, size=200)
Y = np.random.randint(1, 201, size=200)

# generate random integers for our plot

# Create a scatter plot
plt.scatter(X, Y, color='red', alpha=0.6)

# creating a scatterplot with alpha value of .7 and the color red

# Set labels and title
plt.xlabel('Random Integer', color = 'blue')
plt.ylabel('Random Integer', color = 'blue')
plt.title('Scatter of random integers', color = 'green')

# labeling X axis, y axis, title

# Set the plot limits
plt.xlim(0, 200)
plt.ylim(0, 200)
# plot limits

# Show the plot
plt.show()