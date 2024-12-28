# Esteban Murillo
# ITP 449
# Homework 8
# Q1
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



wineDF = pd.read_csv('../wineQualityReds_1_.csv')

print(wineDF.head())

wineDF = wineDF.drop(columns=['Wine'])

print(wineDF)

quality = wineDF['quality']

wineDF = wineDF.drop(columns=['quality'])

print(wineDF)
print(quality)

from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.cluster import KMeans

scaler = MinMaxScaler()

df_scaled = scaler.fit_transform(wineDF)

print(df_scaled)

# Elbow Method to Find Optimal K
inertia_list = []
for k in range(1, 21):
    kmeans = KMeans(n_clusters=k, random_state=2024, n_init='auto')
    kmeans.fit(df_scaled)
    inertia_list.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.plot(range(1, 21), inertia_list, 'o-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# I would pick 20 clusters

kmeans = KMeans(n_clusters=6, random_state=2024, n_init='auto')
kmeans.fit(df_scaled)
wineDF['Cluster'] = kmeans.labels_

print(wineDF)

# Add quality back to the DataFrame

  # Assuming wineDF is your original DataFrame

wineDF['quality'] = quality

print()

# Create a crosstab of cluster number vs quality
crosstab = pd.crosstab(wineDF['quality'], wineDF['Cluster'])
print(crosstab)