# Esteban Murillo
# ITP 449
# HW 3
# Question 2
import pandas as pd
import os
import numpy as np
 # get current directory or change the current directory, redirect to the specific location


Trojans_data = pd.read_csv('../Trojans_roster.csv')

Trojans_df = pd.DataFrame(Trojans_data)

print(Trojans_df)

Trojans_df.set_index("#", inplace=True)
print(Trojans_df)

Trojans_df.drop(columns=['LAST SCHOOL', 'MAJOR'], inplace=True)
print(Trojans_df)

print(Trojans_df.iloc[:3])

print(Trojans_df.loc[Trojans_df['POS.'] == 'QB'])
print(Trojans_df.columns)

print(Trojans_df.loc[Trojans_df['HT.'].max(), ["NAME", "POS.", "HT.", "WT."]])

# da_index = Trojans_df['HOMETOWN'] == 'Los Angeles, Calif.'
#
# # Sum of the boolean index
# sum_of_true = da_index.sum()

print()

print((Trojans_df['HOMETOWN'] == 'Los Angeles, Calif.').sum())

print()

# ITP 259

# print(Trojans_df.loc[Trojans_df['HOMETOWN'] == 'Los Angeles, Calif.'].shape[0])

# sum of the Boolean indexes

print(Trojans_df.nlargest(3, "WT."))

Trojans_df["BMI"] = (703 * Trojans_df["WT."]) / (Trojans_df["HT."] ** 2)

print(Trojans_df)

print(Trojans_df[["HT.", "WT.", "BMI"]].mean())
print(Trojans_df[["HT.", "WT.", "BMI"]].median())
print(Trojans_df.groupby("POS.")[["HT.", "WT.", "BMI"]].agg(["mean", "median"]))

print(Trojans_df["POS."].value_counts())

print(Trojans_df[Trojans_df["BMI"] < Trojans_df["BMI"].mean()]["NAME"])
print(Trojans_df.index.unique())
