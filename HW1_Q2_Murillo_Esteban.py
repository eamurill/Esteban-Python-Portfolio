# Esteban Murillo
# ITP 449
# HW1
# Question 2

name = input('What is your name? ')

name_without_space = name.replace(" ", "")

# I use this function to remove the spaces in between the first and last name.

length = len(name_without_space)

print(name, "your name has", length, "characters.")


