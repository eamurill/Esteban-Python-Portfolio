# Esteban Murillo
# ITP 449
# HW2
# Question 3


print("Please enter a password. Follow these requirements")
print("a. Must be at least 8 characters long")
print("b. Must contain both uppercase and lowercase letters")
print("c. Must contain at least one number between 0-9.")
print("d. Must contain one special character: !, @, #, $")
valid_password = False

while not valid_password:
  password = input("Password: ")

  if len(password) < 8:
    print("Invalid password: Password must be at least 8 characters long.")
  elif not any(char.isupper() for char in password):
    print("Invalid password: Password must contain at least one uppercase letter.")
  elif not any(char.islower() for char in password):
    print("Invalid password: Password must contain at least one lowercase letter.")
  elif not any(char.isdigit() for char in password):
    print("Invalid password: Password must contain at least one number.")
  elif not any(char in "!@#$" for char in password):
    print("Invalid password: Password must contain one special character.")
  else:
    valid_password = True

print("Access Granted!")
