# Esteban Murillo
# ITP 449
# HW5
# Question 1
import pandas as pd
import numpy as np


frame = pd.read_csv('../CSV Files/mtcars.csv')

frame = pd.DataFrame(frame)

print(frame)

frame.set_index("Car Name", inplace=True)

print(frame)

df = pd.read_csv('../CSV Files/mtcars.csv')
df = pd.DataFrame(df)
df.set_index('Car Name', inplace = True)
df = frame[['cyl', 'gear', 'hp', 'mpg']]
df.columns = ['Cylinders', 'Gear', 'Horsepower', 'Miles Per Gallon']

print(df)

df['Powerful'] = df['Horsepower'] >= 110

print(df)