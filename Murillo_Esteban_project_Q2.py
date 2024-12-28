# Esteban Murillo
# ITP 449
# Project
# Question 2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

stores = pd.read_csv('Stores.csv')

print(stores)

store = stores['Store']

stores.drop(columns=['Store'], axis = 1, inplace = True)

print(stores)
print(store)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(stores)

df_scaled = pd.DataFrame(scaler.transform(stores), columns = stores.columns)

print(df_scaled)

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

inertia_list = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=2024, n_init='auto')
    kmeans.fit(df_scaled)
    inertia_list.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia_list, 'o-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# D, the best K is 3 or 4 since the graph begins to flatten after this value
# It is 3.



# Add quality back to the DataFrame

  # Assuming wineDF is your original DataFrame

# E What cluster does this store belong to?

mod = KMeans(n_clusters = 3, n_init = 'auto', random_state=2024)
mod.fit(df_scaled)
df_scaled['cluster'] = mod.labels_
print(df_scaled)

seattle = np.array([[6.3, 3.5, 2.4, .5]])
seattleScaled = scaler.transform(seattle)
clusters = mod.predict(seattleScaled)
print("Individual cluster in which cluster?? ", clusters)

stores['Store'] = store
stores['cluster'] = df_scaled['cluster']
print(stores)

plt.hist(stores['cluster'], bins = 10)
plt.xlabel("# of Clusters")
plt.xticks([0, 1, 2], ["1", "2", "3"])
plt.ylabel("Stores")
plt.show()