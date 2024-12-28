# Esteban Murillo
# ITP 449
# HW2
# Question 1

print("Change for $1")
for quarters in range(5):
  for dimes in range(11 - quarters * 2):
    for nickels in range(21 - quarters * 2 - dimes * 1):
      pennies = 100 - quarters * 25 - dimes * 10 - nickels * 5
      if pennies >= 0:
        print(f"{quarters} quarters, {dimes} dimes, {nickels} nickels, {pennies} pennies")