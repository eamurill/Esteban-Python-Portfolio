# Esteban Murillo
# ITP 449
# HW3
# Question 1

import pandas as pd
import numpy as np

# Create a dictionary
data = {
    "attempts": [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
    "name": ["Anastasia", "Dima", "Katherine", "James", "Emily", "Michael", "Matthew", "Laura", "Kevin", "Jonas"],
    "qualify": ["yes", "no", "yes", "no", "no", "yes", "yes", "no", "no", "yes"],
    "score":[12.5, 9.0, 16.5, np.nan, 9.0, 20.0, 14.5, np.nan, 8.0, 19.0]}

# Define a DataFrame using the dictionary
df = pd.DataFrame(data)

print(df[df["qualify"] == "yes"][["name", "attempts"]])
print(df[df["qualify"] == "yes"][["name", "score"]])

df.loc[np.isnan(df["score"]), "score"] = 0

df = df.sort_values(["attempts", "score"], ascending=[True, False], inplace=False)
print(df)