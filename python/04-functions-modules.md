# Functions and Modules in Python

## Table of Contents
1. [Functions](#functions)
2. [Function Parameters](#function-parameters)
3. [Lambda Functions](#lambda-functions)
4. [Decorators](#decorators)
5. [Generators and Iterators](#generators-and-iterators)
6. [Closures and Nested Functions](#closures-and-nested-functions)
7. [Function Caching and Optimization](#function-caching-and-optimization)
8. [Modules and Packages](#modules-and-packages)
9. [Built-in Functions](#built-in-functions)
10. [Advanced Function Patterns](#advanced-function-patterns)

## Functions

Functions are reusable blocks of code that perform specific tasks.

### 1. Function Definition and Calling
```python
# Basic function definition
def greet():
    """A simple greeting function."""
    print("Hello, World!")

# Calling the function
greet()  # Output: Hello, World!

# Function with parameters
def greet_person(name):
    """Greet a specific person."""
    print(f"Hello, {name}!")

greet_person("Alice")  # Output: Hello, Alice!

# Function with return value
def add_numbers(a, b):
    """Add two numbers and return the result."""
    return a + b

result = add_numbers(5, 3)
print(result)  # Output: 8

# Function with multiple return values
def get_name_age():
    """Return name and age as a tuple."""
    return "John", 25

name, age = get_name_age()
print(f"Name: {name}, Age: {age}")
```

### 2. Function Documentation
```python
def calculate_area(length, width):
    """
    Calculate the area of a rectangle.
    
    Args:
        length (float): The length of the rectangle
        width (float): The width of the rectangle
    
    Returns:
        float: The area of the rectangle
    
    Raises:
        ValueError: If length or width is negative
    
    Examples:
        >>> calculate_area(5, 3)
        15.0
        >>> calculate_area(2.5, 4)
        10.0
    """
    if length < 0 or width < 0:
        raise ValueError("Length and width must be non-negative")
    
    return length * width

# Accessing documentation
print(calculate_area.__doc__)
help(calculate_area)
```

### 3. Function Scope and Variables
```python
# Global and local variables
global_var = "I'm global"

def scope_example():
    local_var = "I'm local"
    print(global_var)   # Can access global variable
    print(local_var)    # Can access local variable

def modify_global():
    global global_var
    global_var = "Modified global"

# Nonlocal variables
def outer_function():
    outer_var = "I'm in outer function"
    
    def inner_function():
        nonlocal outer_var
        outer_var = "Modified by inner function"
        print(outer_var)
    
    inner_function()
    print(outer_var)

# LEGB Rule: Local, Enclosing, Global, Built-in
x = "global"

def outer():
    x = "enclosing"
    
    def inner():
        x = "local"
        print(x)  # Prints "local"
    
    inner()
    print(x)  # Prints "enclosing"

outer()
print(x)  # Prints "global"
```

## Function Parameters

### 1. Parameter Types
```python
# Required parameters
def greet(name, age):
    print(f"Hello {name}, you are {age} years old")

greet("Alice", 25)

# Default parameters
def greet_with_default(name, age=18):
    print(f"Hello {name}, you are {age} years old")

greet_with_default("Bob")        # age defaults to 18
greet_with_default("Charlie", 30)  # age is 30

# Keyword arguments
def introduce(name, age, city):
    print(f"I'm {name}, {age} years old, from {city}")

introduce(city="Boston", name="David", age=28)  # Order doesn't matter

# Mixed positional and keyword arguments
introduce("Eve", age=22, city="Seattle")
```

### 2. *args and **kwargs
```python
# *args - Variable number of positional arguments
def sum_all(*args):
    """Sum all provided arguments."""
    total = 0
    for num in args:
        total += num
    return total

print(sum_all(1, 2, 3))        # 6
print(sum_all(1, 2, 3, 4, 5))  # 15

# **kwargs - Variable number of keyword arguments
def print_info(**kwargs):
    """Print all keyword arguments."""
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_info(name="Alice", age=25, city="Boston")

# Combining all parameter types
def complex_function(required, default_param=10, *args, **kwargs):
    print(f"Required: {required}")
    print(f"Default: {default_param}")
    print(f"Args: {args}")
    print(f"Kwargs: {kwargs}")

complex_function("hello", 20, 1, 2, 3, name="Bob", age=30)
```

### 3. Parameter Unpacking
```python
# Unpacking lists/tuples with *
def add_three(a, b, c):
    return a + b + c

numbers = [1, 2, 3]
result = add_three(*numbers)  # Equivalent to add_three(1, 2, 3)

# Unpacking dictionaries with **
def create_profile(name, age, city):
    return f"Name: {name}, Age: {age}, City: {city}"

person_data = {"name": "Alice", "age": 25, "city": "Boston"}
profile = create_profile(**person_data)

# Function with keyword-only arguments (Python 3+)
def advanced_function(pos_arg, *, keyword_only_arg):
    print(f"Positional: {pos_arg}")
    print(f"Keyword-only: {keyword_only_arg}")

advanced_function("hello", keyword_only_arg="world")
# advanced_function("hello", "world")  # This would raise TypeError
```

### 4. Type Hints (Python 3.5+)
```python
from typing import List, Dict, Optional, Union, Tuple

def process_numbers(numbers: List[int]) -> int:
    """Process a list of numbers and return their sum."""
    return sum(numbers)

def get_user_info(user_id: int) -> Optional[Dict[str, str]]:
    """Get user information by ID."""
    users = {1: {"name": "Alice", "email": "alice@example.com"}}
    return users.get(user_id)

def flexible_function(value: Union[int, str]) -> str:
    """Handle both int and str inputs."""
    return str(value)

def coordinate_function() -> Tuple[float, float]:
    """Return x, y coordinates."""
    return 10.5, 20.3

# Using type hints with complex types
def process_data(
    data: Dict[str, List[int]], 
    multiplier: float = 1.0
) -> Dict[str, float]:
    """Process data and return averages."""
    result = {}
    for key, values in data.items():
        result[key] = sum(values) * multiplier / len(values)
    return result
```

## Lambda Functions

Lambda functions are small, anonymous functions defined using the `lambda` keyword.

### 1. Basic Lambda Functions
```python
# Basic lambda
square = lambda x: x ** 2
print(square(5))  # 25

# Lambda with multiple arguments
add = lambda x, y: x + y
print(add(3, 4))  # 7

# Lambda with conditional expression
max_value = lambda x, y: x if x > y else y
print(max_value(10, 15))  # 15

# Immediately invoked lambda
result = (lambda x, y: x * y)(4, 5)  # 20
```

### 2. Lambda with Built-in Functions
```python
numbers = [1, 2, 3, 4, 5]

# Using lambda with map()
squared = list(map(lambda x: x ** 2, numbers))
print(squared)  # [1, 4, 9, 16, 25]

# Using lambda with filter()
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(evens)  # [2, 4]

# Using lambda with sorted()
students = [("Alice", 85), ("Bob", 90), ("Charlie", 78)]
sorted_by_grade = sorted(students, key=lambda student: student[1])
print(sorted_by_grade)  # [('Charlie', 78), ('Alice', 85), ('Bob', 90)]

# Using lambda with reduce()
from functools import reduce
product = reduce(lambda x, y: x * y, numbers)  # 1*2*3*4*5 = 120
```

### 3. Advanced Lambda Usage
```python
# Lambda in list comprehensions (though regular functions are often clearer)
operations = [lambda x: x + 1, lambda x: x * 2, lambda x: x ** 2]
results = [op(5) for op in operations]  # [6, 10, 25]

# Lambda for event handling or callbacks
def process_data(data, callback):
    result = sum(data)
    return callback(result)

data = [1, 2, 3, 4, 5]
formatted_result = process_data(data, lambda x: f"Total: {x}")

# Limitations of lambda (single expression only)
# This won't work:
# invalid_lambda = lambda x: 
#     if x > 0:
#         return x
#     else:
#         return -x

# Use regular function instead:
def abs_value(x):
    if x > 0:
        return x
    else:
        return -x
```

## Decorators

Decorators are functions that modify or enhance other functions.

### 1. Basic Decorators
```python
# Simple decorator
def my_decorator(func):
    def wrapper():
        print("Something before the function")
        func()
        print("Something after the function")
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

say_hello()
# Output:
# Something before the function
# Hello!
# Something after the function

# Equivalent to:
# say_hello = my_decorator(say_hello)
```

### 2. Decorators with Arguments
```python
# Decorator for functions with arguments
def argument_decorator(func):
    def wrapper(*args, **kwargs):
        print(f"Function {func.__name__} called with args: {args}, kwargs: {kwargs}")
        result = func(*args, **kwargs)
        print(f"Function {func.__name__} returned: {result}")
        return result
    return wrapper

@argument_decorator
def add_numbers(a, b):
    return a + b

result = add_numbers(3, 5)

# Preserving function metadata
from functools import wraps

def better_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        """Wrapper function."""
        print(f"Calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

@better_decorator
def example_function():
    """Example function docstring."""
    pass

print(example_function.__name__)  # example_function (not wrapper)
print(example_function.__doc__)   # Example function docstring.
```

### 3. Parameterized Decorators
```python
# Decorator that takes parameters
def repeat(times):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(3)
def greet(name):
    print(f"Hello, {name}!")

greet("Alice")  # Prints greeting 3 times

# Timer decorator
import time
from functools import wraps

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(1)
    return "Done"

result = slow_function()
```

### 4. Class-based Decorators
```python
class CountCalls:
    def __init__(self, func):
        self.func = func
        self.count = 0
    
    def __call__(self, *args, **kwargs):
        self.count += 1
        print(f"{self.func.__name__} has been called {self.count} times")
        return self.func(*args, **kwargs)

@CountCalls
def say_hello():
    print("Hello!")

say_hello()  # say_hello has been called 1 times
say_hello()  # say_hello has been called 2 times

# Property decorators
class Circle:
    def __init__(self, radius):
        self._radius = radius
    
    @property
    def radius(self):
        return self._radius
    
    @radius.setter
    def radius(self, value):
        if value < 0:
            raise ValueError("Radius must be non-negative")
        self._radius = value
    
    @property
    def area(self):
        return 3.14159 * self._radius ** 2

circle = Circle(5)
print(circle.area)  # 78.53975
circle.radius = 3
print(circle.area)  # 28.27431
```

## Modules and Packages

### 1. Creating and Using Modules
```python
# File: math_utils.py
"""
A module for mathematical utilities.
"""

PI = 3.14159

def add(a, b):
    """Add two numbers."""
    return a + b

def multiply(a, b):
    """Multiply two numbers."""
    return a * b

class Calculator:
    """A simple calculator class."""
    
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result

# Using the module
import math_utils

result = math_utils.add(5, 3)
print(math_utils.PI)

calc = math_utils.Calculator()
calc.add(10, 20)
```

### 2. Different Import Methods
```python
# Import entire module
import math
print(math.sqrt(16))

# Import specific functions
from math import sqrt, pi
print(sqrt(16))
print(pi)

# Import with alias
import math as m
print(m.sqrt(16))

from math import sqrt as square_root
print(square_root(16))

# Import all (generally not recommended)
from math import *
print(sqrt(16))

# Conditional imports
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

if HAS_NUMPY:
    array = np.array([1, 2, 3])
else:
    array = [1, 2, 3]
```

### 3. Module Search Path and __name__
```python
# File: example_module.py
print(f"Module name: {__name__}")

def main():
    print("This is the main function")

if __name__ == "__main__":
    # This code only runs when the script is executed directly
    print("Script is being run directly")
    main()
else:
    print("Script is being imported")

# Checking module search path
import sys
print(sys.path)

# Adding to module search path
sys.path.append("/path/to/my/modules")
```

### 4. Packages
```python
# Package structure:
# mypackage/
#     __init__.py
#     module1.py
#     module2.py
#     subpackage/
#         __init__.py
#         submodule.py

# File: mypackage/__init__.py
"""
My package initialization.
"""
from .module1 import function1
from .module2 import Class2

__version__ = "1.0.0"
__all__ = ["function1", "Class2"]

# File: mypackage/module1.py
def function1():
    return "Hello from module1"

def helper_function():
    return "This is a helper"

# Using the package
import mypackage
print(mypackage.function1())
print(mypackage.__version__)

# Relative imports within package
# File: mypackage/module2.py
from .module1 import helper_function  # Relative import
from . import module1                 # Import sibling module

class Class2:
    def method(self):
        return helper_function()
```

## Built-in Functions

### 1. Mathematical Functions
```python
# Basic math functions
print(abs(-5))          # 5
print(pow(2, 3))        # 8
print(round(3.14159, 2)) # 3.14
print(divmod(17, 5))    # (3, 2) - quotient and remainder

# Min/Max
print(min(5, 3, 8, 1))  # 1
print(max([1, 5, 3, 9])) # 9

numbers = [1, 2, 3, 4, 5]
print(sum(numbers))     # 15
print(sum(numbers, 10)) # 25 (with start value)
```

### 2. Type and Conversion Functions
```python
# Type checking
print(type(42))         # <class 'int'>
print(isinstance(42, int)) # True
print(isinstance("hello", (str, int))) # True (multiple types)

# Type conversion
print(int("42"))        # 42
print(float("3.14"))    # 3.14
print(str(42))          # "42"
print(bool(0))          # False
print(list("hello"))    # ['h', 'e', 'l', 'l', 'o']
print(tuple([1, 2, 3])) # (1, 2, 3)
print(set([1, 2, 2, 3])) # {1, 2, 3}

# ASCII and character functions
print(ord('A'))         # 65
print(chr(65))          # 'A'
```

### 3. Sequence Functions
```python
numbers = [1, 2, 3, 4, 5]

# Length and enumeration
print(len(numbers))     # 5
print(list(enumerate(numbers)))  # [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]

# Range
print(list(range(5)))        # [0, 1, 2, 3, 4]
print(list(range(2, 8, 2)))  # [2, 4, 6]

# Zip
names = ["Alice", "Bob", "Charlie"]
ages = [25, 30, 35]
print(list(zip(names, ages)))  # [('Alice', 25), ('Bob', 30), ('Charlie', 35)]

# Reversed and sorted
print(list(reversed(numbers)))  # [5, 4, 3, 2, 1]
print(sorted([3, 1, 4, 1, 5]))  # [1, 1, 3, 4, 5]
```

### 4. Functional Programming Functions
```python
# Map, filter, reduce
numbers = [1, 2, 3, 4, 5]

# Map - apply function to all elements
squared = list(map(lambda x: x**2, numbers))
print(squared)  # [1, 4, 9, 16, 25]

# Filter - filter elements based on condition
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(evens)    # [2, 4]

# Reduce - reduce sequence to single value
from functools import reduce
product = reduce(lambda x, y: x * y, numbers)
print(product)  # 120

# All and any
print(all([True, True, True]))   # True
print(all([True, False, True]))  # False
print(any([False, False, True])) # True
print(any([False, False, False])) # False

# All/any with conditions
numbers = [2, 4, 6, 8]
print(all(x % 2 == 0 for x in numbers))  # True (all even)
print(any(x > 5 for x in numbers))       # True (at least one > 5)
```

### 5. Object and Attribute Functions
```python
class Person:
    def __init__(self, name):
        self.name = name
    
    def greet(self):
        return f"Hello, I'm {self.name}"

person = Person("Alice")

# Object inspection
print(dir(person))      # List all attributes and methods
print(vars(person))     # Return __dict__ attribute
print(hasattr(person, 'name'))    # True
print(getattr(person, 'name'))    # 'Alice'
print(getattr(person, 'age', 0))  # 0 (default value)

setattr(person, 'age', 25)        # Set attribute
print(person.age)                 # 25

delattr(person, 'age')            # Delete attribute
# print(person.age)               # Would raise AttributeError

# Callable check
print(callable(person.greet))     # True
print(callable(person.name))      # False
```

## Best Practices

1. **Function Design Principles**:
```python
# Single Responsibility Principle
def calculate_tax(income, rate):
    """Calculate tax - does one thing well."""
    return income * rate

def format_currency(amount):
    """Format amount as currency - separate responsibility."""
    return f"${amount:.2f}"

# Instead of combining everything in one function
def process_income(income, tax_rate):
    tax = calculate_tax(income, tax_rate)
    formatted_tax = format_currency(tax)
    return formatted_tax
```

2. **Use type hints and documentation**:
```python
from typing import List, Optional

def process_scores(scores: List[float], threshold: float = 60.0) -> Optional[float]:
    """
    Calculate average of scores above threshold.
    
    Args:
        scores: List of numeric scores
        threshold: Minimum score to include in average
    
    Returns:
        Average of qualifying scores, or None if no scores qualify
    """
    qualifying_scores = [score for score in scores if score >= threshold]
    if not qualifying_scores:
        return None
    return sum(qualifying_scores) / len(qualifying_scores)
```

3. **Avoid global variables and use pure functions when possible**:
```python
# Bad - relies on global state
counter = 0

def increment():
    global counter
    counter += 1
    return counter

# Good - pure function
def increment_pure(value):
    return value + 1

# Good - encapsulated state
class Counter:
    def __init__(self):
        self._value = 0
    
    def increment(self):
        self._value += 1
        return self._value
```

4. **Use appropriate module organization**:
```python
# Good module structure
# utils/
#     __init__.py      # Package initialization
#     math_utils.py    # Mathematical utilities
#     string_utils.py  # String utilities
#     file_utils.py    # File operations
#     constants.py     # Application constants
```

## Generators and Iterators

### Generators

Generators are functions that return an iterator object, yielding values one at a time.

```python
# Basic generator function
def number_generator(n):
    """Generate numbers from 0 to n-1"""
    for i in range(n):
        yield i

# Using the generator
gen = number_generator(5)
print(type(gen))  # <class 'generator'>

# Iterating through generator
for num in gen:
    print(num)  # 0, 1, 2, 3, 4

# Generator expressions
squares = (x**2 for x in range(10))
print(list(squares))  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# Memory-efficient file processing
def read_large_file(file_path):
    """Read large file line by line without loading entire file"""
    with open(file_path, 'r') as file:
        for line in file:
            yield line.strip()

# Fibonacci generator
def fibonacci():
    """Generate Fibonacci sequence indefinitely"""
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

# Get first 10 Fibonacci numbers
fib_gen = fibonacci()
fib_numbers = [next(fib_gen) for _ in range(10)]
print(fib_numbers)  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```

## Function Caching and Optimization

### Memoization and Caching

```python
import functools
import time

# Using functools.lru_cache
@functools.lru_cache(maxsize=128)
def fibonacci_lru(n):
    """LRU cached Fibonacci calculation"""
    if n < 2:
        return n
    return fibonacci_lru(n-1) + fibonacci_lru(n-2)

# Performance comparison
def fibonacci_naive(n):
    """Naive Fibonacci calculation"""
    if n < 2:
        return n
    return fibonacci_naive(n-1) + fibonacci_naive(n-2)

# Time comparison
start = time.time()
result_naive = fibonacci_naive(30)
time_naive = time.time() - start

start = time.time()
result_cached = fibonacci_lru(30)
time_cached = time.time() - start

print(f"Naive: {time_naive:.4f}s")
print(f"Cached: {time_cached:.4f}s")
print(f"Cache info: {fibonacci_lru.cache_info()}")
```

## Advanced Function Patterns

### Function Factories and Higher-Order Functions

```python
# Function factory pattern
def create_validator(validator_type):
    """Factory function that creates validator functions"""
    
    if validator_type == 'email':
        def validate_email(email):
            import re
            pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            return bool(re.match(pattern, email))
        return validate_email
    
    elif validator_type == 'phone':
        def validate_phone(phone):
            import re
            pattern = r'^\+?1?\d{9,15}$'
            return bool(re.match(pattern, phone.replace('-', '').replace(' ', '')))
        return validate_phone
    
    else:
        raise ValueError(f"Unknown validator type: {validator_type}")

# Using function factory
email_validator = create_validator('email')
phone_validator = create_validator('phone')

print(email_validator('user@example.com'))  # True
print(phone_validator('123-456-7890'))      # True

# Function composition
def compose(*functions):
    """Compose multiple functions into one"""
    def composed_function(arg):
        result = arg
        for func in reversed(functions):
            result = func(result)
        return result
    return composed_function

# Example functions
def add_one(x):
    return x + 1

def multiply_by_two(x):
    return x * 2

def square(x):
    return x ** 2

# Compose functions
composed = compose(square, multiply_by_two, add_one)
result = composed(3)  # ((3 + 1) * 2) ** 2 = 64
print(result)

print("\nAdvanced function concepts completed!")
```
