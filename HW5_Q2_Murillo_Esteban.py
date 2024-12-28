# Esteban Murillo
# eamurill@usc.edu
# ITP 449

import pandas as pd


df = pd.read_csv('../CSV Files/mtcars.csv')
df = pd.DataFrame(df)
df.set_index('Car Name', inplace = True)

selected_columns = df[['cyl', 'gear', 'hp', 'mpg']]

# Create a new DataFrame with only the selected columns
df = selected_columns
df.rename(columns={
    'cyl': 'Cylinders',
    'gear': 'Gear',
    'hp': 'Horsepower',
    'mpg': 'Miles Per Gallon'
}, inplace=True)
print(df)

df['Powerful'] = df['Horsepower'] >= 110

new_df = df.drop('Horsepower', axis=1)

# Print the new DataFrame
print(new_df)

# Filter and sort the original DataFrame (before dropping 'Horsepower')
print(df[df['Miles Per Gallon'] > 25].sort_values('Horsepower', ascending=False))

result = df[df['Powerful'] == True].sort_values('Miles Per Gallon', ascending=False).head(1)
print(result)


