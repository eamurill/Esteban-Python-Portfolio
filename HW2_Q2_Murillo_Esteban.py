# Tommy Trojan
# ITP 499
# HW2
# Question 2

number1 = int(input('Please enter an integer: '))
number2 = int(input('Please enter another integer: '))
counter = 1

while(counter <= number2) and (number1 >= 0 and number2 <= 100):
    print(number1, "x", counter, "=", number1 * counter)
    counter += 1
