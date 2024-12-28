# Esteban Murillo
# ITP 449
# HW4
# Question 3

import matplotlib.pyplot as plt
import pandas as pd

# Get user input
loan_amount = float(input("Enter the loan amount: "))
annual_interest_rate = float(input("Enter the annual interest rate: "))
years = int(input("Enter the number of years: "))

# having the user input

# Calculate monthly interest rate and number of payments
monthly_interest_rate = annual_interest_rate / 12 / 100
num_payments = years * 12

# Calculate monthly payment
monthly_payment = (loan_amount * monthly_interest_rate * (1 + monthly_interest_rate) ** num_payments /
                   ((1 + monthly_interest_rate) ** num_payments - 1))

# Print monthly payment
print(f"Your monthly payment is: ${monthly_payment:.2f}")

# Calculate monthly interest and principal balance
monthly_interest = []
principal_balance = []
mv_bl = loan_amount

for month in range(1, num_payments + 1):
    interest_paid = mv_bl * monthly_interest_rate
    mv_bl = mv_bl + interest_paid - monthly_payment
    monthly_interest.append(interest_paid)
    principal_balance.append(mv_bl)

# Creating subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Plot monthly interest
axs[0].plot(range(1, num_payments + 1), monthly_interest, label='Monthly Interest', color='blue', marker='o',
            linewidth = .5)
axs[0].set_xlabel('Month')
axs[0].set_ylabel('Interest Paid')


# Plot principal balance
axs[1].plot(range(1, num_payments + 1), principal_balance, label='Principal Balance', color='blue', marker='o',
            linewidth = .5)
axs[1].set_xlabel('Month')
axs[1].set_ylabel('Loan Balance')


# Show the plots
plt.tight_layout()
plt.show()

year = [2016, 2017, 2018, 2019, 2020]
ITP115 = [180, 250, 390, 540, 720]
ITP449 = [70, 150, 130, 180, 220]

plt.bar(year, ITP115, label = 'ITP115')
plt.bar(year, ITP449, label = 'ITP449')
plt.xlabel('Year')
plt.ylabel('Enrollment')
plt.title('ITP Enrollment')
plt.legend()
plt.show()

plt.style.use('ggplot')
rainfall = [15, 19, 18, 13, 11, 7, 8, 7, 10, 12, 14, 15]
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
plt.bar(months, rainfall)
plt.xticks(rotation = 45)
plt.xlabel('Month')
plt.ylabel('Rainfall (cm)')
plt.title('Rainfall')
plt.savefig('rainfall.png', bbox_inches = 'tight')

plt.style.use('ggplot')
labels = ['Chrome', 'Firefox', 'Internet Explorer', 'Safari', 'Others']
marketShare = [61.64, 18.98, 11.02, 4.23, 4.13]
plt.pie(marketShare, labels=labels, autopct = '%.1f%')
plt.title('Browser Market Share')
plt.axis('equal')
plt.legend(labels, loc = 'lower left')
plt.show()


plt.plot([0, 1, 2, 3, 4], [0, 1, 8, 27, 64])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('X^3')
plt.show()

import seaborn as sns

sns.set_style('whitegrid')
puzzleCompletionTime = pd.read_csv('grade5ScienceFairProject.csv')
sns.swarmplot(x = 'Music', y = 'Seconds', data = puzzleCompletionTime, size = 10)
plt.show()

sns.boxplot(x = 'Music', y = 'Seconds', data = puzzleCompletionTime)
plt.show()

salaryData = pd.read_csv('salaries.csv')
sns.swarmplot(x = 'gender', y = 'salary', data = salaryData)
ax = plt.gca()
ax.set_title('Salary Distribution')
plt.show()

g = sns.catplot(x = 'gender')


