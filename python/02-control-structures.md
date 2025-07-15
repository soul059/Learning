# Control Structures in Python

## Table of Contents
1. [Conditional Statements](#conditional-statements)
2. [Loops](#loops)
3. [Control Flow Statements](#control-flow-statements)
4. [Exception Handling](#exception-handling)

## Conditional Statements

### 1. if Statement
```python
# Basic if statement
age = 18
if age >= 18:
    print("You are an adult")

# if-else statement
score = 85
if score >= 90:
    print("Grade A")
else:
    print("Grade B or below")

# if-elif-else statement
marks = 78
if marks >= 90:
    grade = "A"
elif marks >= 80:
    grade = "B"
elif marks >= 70:
    grade = "C"
elif marks >= 60:
    grade = "D"
else:
    grade = "F"
print(f"Your grade is: {grade}")
```

### 2. Nested if Statements
```python
age = 25
has_license = True

if age >= 18:
    if has_license:
        print("You can drive")
    else:
        print("You need a license to drive")
else:
    print("You are too young to drive")
```

### 3. Conditional Expressions (Ternary Operator)
```python
# Syntax: value_if_true if condition else value_if_false
age = 20
status = "adult" if age >= 18 else "minor"
print(status)  # adult

# Multiple conditions
score = 85
result = "Pass" if score >= 60 else "Fail"

# Nested ternary
grade = "A" if score >= 90 else "B" if score >= 80 else "C"
```

### 4. Boolean Context in Conditions
```python
# Truthy and falsy values in conditions
name = ""
if name:
    print(f"Hello, {name}")
else:
    print("No name provided")

# Common patterns
numbers = [1, 2, 3, 4, 5]
if numbers:  # Check if list is not empty
    print(f"First number: {numbers[0]}")

# None checking
value = None
if value is not None:
    print(f"Value: {value}")
else:
    print("No value")
```

## Loops

### 1. for Loop

#### Basic for Loop
```python
# Iterating over a sequence
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(fruit)

# Iterating over a string
for char in "Python":
    print(char)

# Using range()
for i in range(5):      # 0 to 4
    print(i)

for i in range(1, 6):   # 1 to 5
    print(i)

for i in range(0, 10, 2):  # 0, 2, 4, 6, 8
    print(i)

for i in range(10, 0, -1):  # 10, 9, 8, ..., 1
    print(i)
```

#### Advanced for Loop Patterns
```python
# enumerate() - get index and value
fruits = ["apple", "banana", "cherry"]
for index, fruit in enumerate(fruits):
    print(f"{index}: {fruit}")

# Starting enumerate from different number
for index, fruit in enumerate(fruits, start=1):
    print(f"{index}: {fruit}")

# zip() - iterate over multiple sequences
names = ["Alice", "Bob", "Charlie"]
ages = [25, 30, 35]
for name, age in zip(names, ages):
    print(f"{name} is {age} years old")

# reversed() - iterate in reverse
for fruit in reversed(fruits):
    print(fruit)

# sorted() - iterate in sorted order
numbers = [3, 1, 4, 1, 5, 9]
for num in sorted(numbers):
    print(num)
```

#### Dictionary Iteration
```python
student = {"name": "John", "age": 20, "grade": "A"}

# Iterate over keys
for key in student:
    print(key)

# Iterate over values
for value in student.values():
    print(value)

# Iterate over key-value pairs
for key, value in student.items():
    print(f"{key}: {value}")
```

### 2. while Loop

#### Basic while Loop
```python
# Basic while loop
count = 0
while count < 5:
    print(count)
    count += 1

# Input validation loop
while True:
    user_input = input("Enter a positive number (or 'quit' to exit): ")
    if user_input.lower() == 'quit':
        break
    try:
        number = float(user_input)
        if number > 0:
            print(f"Square root: {number ** 0.5}")
            break
        else:
            print("Please enter a positive number")
    except ValueError:
        print("Please enter a valid number")
```

#### while with else
```python
# while-else: else block executes if loop completes normally
count = 0
while count < 3:
    print(count)
    count += 1
else:
    print("Loop completed normally")

# If break is used, else block is skipped
count = 0
while count < 10:
    if count == 3:
        break
    print(count)
    count += 1
else:
    print("This won't print")
```

### 3. Nested Loops
```python
# Multiplication table
for i in range(1, 4):
    for j in range(1, 4):
        print(f"{i} x {j} = {i * j}")
    print()  # Empty line after each table

# Pattern printing
for i in range(5):
    for j in range(i + 1):
        print("*", end="")
    print()

# Matrix traversal
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
for row in matrix:
    for element in row:
        print(element, end=" ")
    print()
```

## Control Flow Statements

### 1. break Statement
```python
# break in for loop
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for num in numbers:
    if num == 5:
        break
    print(num)  # Prints 1, 2, 3, 4

# break in while loop
count = 0
while True:
    if count == 3:
        break
    print(count)
    count += 1

# break in nested loops (only breaks inner loop)
for i in range(3):
    for j in range(3):
        if j == 1:
            break
        print(f"i={i}, j={j}")
```

### 2. continue Statement
```python
# continue in for loop
for i in range(10):
    if i % 2 == 0:  # Skip even numbers
        continue
    print(i)  # Prints 1, 3, 5, 7, 9

# continue in while loop
count = 0
while count < 10:
    count += 1
    if count % 2 == 0:
        continue
    print(count)

# Practical example: processing valid data
data = [1, 2, -1, 4, -2, 6, 7]
for value in data:
    if value < 0:
        continue  # Skip negative values
    print(f"Processing: {value}")
```

### 3. pass Statement
```python
# pass as placeholder
def future_function():
    pass  # TODO: implement this later

# pass in conditional
age = 15
if age >= 18:
    pass  # TODO: handle adult case
else:
    print("Minor")

# pass in loops
for i in range(5):
    if i == 2:
        pass  # Do nothing for i=2
    else:
        print(i)

# Empty class
class EmptyClass:
    pass
```

### 4. else Clause with Loops
```python
# for-else
numbers = [1, 3, 5, 7, 9]
target = 4

for num in numbers:
    if num == target:
        print(f"Found {target}")
        break
else:
    print(f"{target} not found in the list")

# while-else
password = "secret"
attempts = 0
max_attempts = 3

while attempts < max_attempts:
    user_input = input("Enter password: ")
    if user_input == password:
        print("Access granted")
        break
    attempts += 1
else:
    print("Access denied - too many attempts")
```

## Exception Handling

### 1. Basic try-except
```python
# Basic exception handling
try:
    number = int(input("Enter a number: "))
    result = 10 / number
    print(f"Result: {result}")
except ValueError:
    print("Invalid input! Please enter a number.")
except ZeroDivisionError:
    print("Cannot divide by zero!")
```

### 2. Multiple Exceptions
```python
# Handling multiple exceptions
try:
    data = input("Enter numbers separated by commas: ")
    numbers = [int(x.strip()) for x in data.split(",")]
    average = sum(numbers) / len(numbers)
    print(f"Average: {average}")
except ValueError:
    print("Invalid number format")
except ZeroDivisionError:
    print("No numbers provided")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
```

### 3. try-except-else-finally
```python
# Complete exception handling structure
filename = "data.txt"

try:
    file = open(filename, "r")
    content = file.read()
    number = int(content.strip())
except FileNotFoundError:
    print(f"File {filename} not found")
except ValueError:
    print("File content is not a valid number")
except Exception as e:
    print(f"Unexpected error: {e}")
else:
    # Executes if no exception occurred
    print(f"Successfully read number: {number}")
finally:
    # Always executes
    try:
        file.close()
        print("File closed")
    except:
        pass
```

### 4. Raising Exceptions
```python
# Raising custom exceptions
def divide(a, b):
    if b == 0:
        raise ValueError("Division by zero is not allowed")
    return a / b

try:
    result = divide(10, 0)
except ValueError as e:
    print(f"Error: {e}")

# Re-raising exceptions
def process_data(data):
    try:
        return int(data)
    except ValueError:
        print("Logging error...")
        raise  # Re-raise the same exception

try:
    result = process_data("abc")
except ValueError:
    print("Handled in outer try block")
```

### 5. Custom Exceptions
```python
# Creating custom exception classes
class CustomError(Exception):
    pass

class ValidationError(Exception):
    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code

def validate_age(age):
    if age < 0:
        raise ValidationError("Age cannot be negative", code="NEGATIVE_AGE")
    if age > 150:
        raise ValidationError("Age seems unrealistic", code="UNREALISTIC_AGE")

try:
    validate_age(-5)
except ValidationError as e:
    print(f"Validation error: {e}")
    print(f"Error code: {e.code}")
```

### 6. Context Managers (with statement)
```python
# Using with statement for file handling
with open("data.txt", "w") as file:
    file.write("Hello, World!")
# File is automatically closed

# Multiple context managers
with open("input.txt", "r") as infile, open("output.txt", "w") as outfile:
    content = infile.read()
    outfile.write(content.upper())

# Custom context manager
class DatabaseConnection:
    def __enter__(self):
        print("Connecting to database...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Closing database connection...")
        if exc_type:
            print(f"Exception occurred: {exc_val}")
        return False  # Don't suppress exceptions

with DatabaseConnection() as db:
    print("Working with database...")
```

## Best Practices

1. **Use meaningful condition names**:
```python
# Good
is_valid_email = "@" in email and "." in email
if is_valid_email:
    send_email(email)

# Better than
if "@" in email and "." in email:
    send_email(email)
```

2. **Avoid deep nesting**:
```python
# Instead of deep nesting
if user_logged_in:
    if has_permission:
        if data_valid:
            process_data()

# Use early returns
def process_user_data():
    if not user_logged_in:
        return "User not logged in"
    if not has_permission:
        return "No permission"
    if not data_valid:
        return "Invalid data"
    
    return process_data()
```

3. **Use exception handling appropriately**:
```python
# Don't use exceptions for control flow
# Bad
try:
    value = dictionary[key]
except KeyError:
    value = default_value

# Good
value = dictionary.get(key, default_value)
```

4. **Be specific with exceptions**:
```python
# Too broad
try:
    risky_operation()
except Exception:
    pass

# Better
try:
    risky_operation()
except (ValueError, TypeError) as e:
    handle_specific_error(e)
```
