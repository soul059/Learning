# Python Basics and Fundamentals

## Table of Contents
1. [Introduction to Python](#introduction-to-python)
2. [Python Installation and Setup](#python-installation-and-setup)
3. [Python IDEs and Development Environment](#python-ides-and-development-environment)
4. [Variables and Data Types](#variables-and-data-types)
5. [Operators](#operators)
6. [Input/Output](#inputoutput)
7. [String Formatting and Methods](#string-formatting-and-methods)
8. [Comments](#comments)
9. [Python Keywords](#python-keywords)
10. [Memory Management](#memory-management)
11. [Python Package Management](#python-package-management)

## Introduction to Python

Python is a high-level, interpreted, and general-purpose programming language created by Guido van Rossum in 1991.

### Key Features:
- **Easy to Learn**: Simple syntax resembling English
- **Interpreted**: No compilation needed
- **Cross-platform**: Runs on Windows, Mac, Linux
- **Object-Oriented**: Supports OOP concepts
- **Open Source**: Free to use and distribute
- **Large Standard Library**: "Batteries included"
- **Dynamic Typing**: Variables don't need explicit type declaration

### Python Zen (PEP 20)
```python
import this
# The Zen of Python principles:
# - Beautiful is better than ugly
# - Explicit is better than implicit
# - Simple is better than complex
# - Readability counts
```

## Python Installation and Setup

### Installing Python

#### Windows
```bash
# Download from python.org
# Or use package manager
winget install Python.Python.3.11

# Verify installation
python --version
python -m pip --version
```

#### macOS
```bash
# Using Homebrew
brew install python3

# Using MacPorts
sudo port install python311

# Verify installation
python3 --version
pip3 --version
```

#### Linux (Ubuntu/Debian)
```bash
# Update package list
sudo apt update

# Install Python
sudo apt install python3 python3-pip python3-venv

# Verify installation
python3 --version
pip3 --version
```

### Virtual Environments

Virtual environments isolate Python projects and their dependencies.

```python
# Create virtual environment
python -m venv myproject_env

# Activate virtual environment
# Windows
myproject_env\Scripts\activate

# macOS/Linux
source myproject_env/bin/activate

# Install packages in virtual environment
pip install requests numpy pandas

# Create requirements file
pip freeze > requirements.txt

# Install from requirements file
pip install -r requirements.txt

# Deactivate virtual environment
deactivate

# Remove virtual environment
rm -rf myproject_env  # macOS/Linux
rmdir /s myproject_env  # Windows
```

### Environment Variables
```python
import os

# Get environment variable
python_path = os.environ.get('PYTHONPATH', 'Not set')
print(f"PYTHONPATH: {python_path}")

# Set environment variable
os.environ['MY_VARIABLE'] = 'my_value'

# Common Python environment variables
print(f"Python executable: {os.sys.executable}")
print(f"Python version: {os.sys.version}")
print(f"Python path: {os.sys.path}")
```

## Python IDEs and Development Environment

### Popular Python IDEs and Editors

#### 1. PyCharm
- **Professional IDE** with advanced features
- Excellent debugging and testing tools
- Built-in version control integration
- Code completion and refactoring

#### 2. Visual Studio Code
- **Lightweight** and highly customizable
- Excellent Python extension
- Integrated terminal and debugging
- Git integration

#### 3. Jupyter Notebook/Lab
- **Interactive development** environment
- Great for data science and research
- Mix code, markdown, and visualizations
- Web-based interface

```python
# Jupyter magic commands
%timeit sum(range(100))          # Time execution
%matplotlib inline               # Display plots inline
%load_ext autoreload            # Auto-reload modules
%autoreload 2
```

#### 4. IDLE
- **Built-in** Python IDE
- Simple and lightweight
- Good for beginners
- Interactive shell

#### 5. Sublime Text / Atom / Vim
- **Text editors** with Python support
- Highly customizable
- Lightweight and fast

### Code Formatting and Linting

```python
# Install formatting tools
pip install black flake8 autopep8

# Format code with Black
black my_script.py

# Check code style with flake8
flake8 my_script.py

# Auto-format with autopep8
autopep8 --in-place --aggressive my_script.py
```

### Debugging Tools

```python
# Built-in debugger
import pdb

def divide_numbers(a, b):
    pdb.set_trace()  # Set breakpoint
    result = a / b
    return result

# Interactive debugging
result = divide_numbers(10, 2)

# Python debugger commands:
# n - next line
# s - step into function
# c - continue execution
# l - list current code
# p variable_name - print variable
# q - quit debugger
```

## Variables and Data Types

### Variable Declaration
```python
# Python uses dynamic typing
name = "John"           # String
age = 25               # Integer
height = 5.9           # Float
is_student = True      # Boolean

# Multiple assignment
x, y, z = 1, 2, 3
a = b = c = 100

# Variable naming rules
valid_name = "correct"
_private_var = "valid"
name2 = "valid"
# 2name = "invalid"  # Cannot start with number
# class = "invalid"  # Cannot use keywords
```

### Data Types

#### 1. Numeric Types
```python
# Integer
x = 10
y = -5
z = 0

# Float
pi = 3.14159
scientific = 1.5e-10  # Scientific notation

# Complex
complex_num = 3 + 4j
print(complex_num.real)  # 3.0
print(complex_num.imag)  # 4.0

# Type checking
print(type(x))  # <class 'int'>
print(isinstance(pi, float))  # True
```

#### 2. String Type
```python
# String creation
single_quote = 'Hello'
double_quote = "World"
triple_quote = """Multi-line
string example"""

# String methods
text = "Python Programming"
print(text.upper())        # PYTHON PROGRAMMING
print(text.lower())        # python programming
print(text.title())        # Python Programming
print(text.replace("Python", "Java"))  # Java Programming
print(text.split())        # ['Python', 'Programming']
print(len(text))          # 18

# String formatting
name = "Alice"
age = 30
# f-strings (Python 3.6+)
message = f"My name is {name} and I'm {age} years old"
# .format() method
message = "My name is {} and I'm {} years old".format(name, age)
# % formatting
message = "My name is %s and I'm %d years old" % (name, age)
```

#### 3. Boolean Type
```python
is_true = True
is_false = False

# Boolean operations
print(True and False)   # False
print(True or False)    # True
print(not True)         # False

# Truthy and Falsy values
print(bool(0))          # False
print(bool(""))         # False
print(bool([]))         # False
print(bool(None))       # False
print(bool("Hello"))    # True
print(bool([1, 2, 3]))  # True
```

#### 4. None Type
```python
value = None
print(type(value))  # <class 'NoneType'>

# Used for initialization or absence of value
def function_without_return():
    pass

result = function_without_return()
print(result)  # None
```

## Operators

### 1. Arithmetic Operators
```python
a, b = 10, 3

print(a + b)    # 13 (Addition)
print(a - b)    # 7  (Subtraction)
print(a * b)    # 30 (Multiplication)
print(a / b)    # 3.333... (Division)
print(a // b)   # 3  (Floor division)
print(a % b)    # 1  (Modulus)
print(a ** b)   # 1000 (Exponentiation)
```

### 2. Comparison Operators
```python
x, y = 5, 10

print(x == y)   # False (Equal to)
print(x != y)   # True  (Not equal to)
print(x < y)    # True  (Less than)
print(x > y)    # False (Greater than)
print(x <= y)   # True  (Less than or equal to)
print(x >= y)   # False (Greater than or equal to)
```

### 3. Logical Operators
```python
a, b = True, False

print(a and b)  # False
print(a or b)   # True
print(not a)    # False

# Short-circuit evaluation
print(False and print("This won't execute"))  # False
print(True or print("This won't execute"))    # True
```

### 4. Assignment Operators
```python
x = 10
x += 5    # x = x + 5, now x = 15
x -= 3    # x = x - 3, now x = 12
x *= 2    # x = x * 2, now x = 24
x /= 4    # x = x / 4, now x = 6.0
x //= 2   # x = x // 2, now x = 3.0
x %= 2    # x = x % 2, now x = 1.0
x **= 3   # x = x ** 3, now x = 1.0
```

### 5. Bitwise Operators
```python
a, b = 12, 25  # 12 = 1100, 25 = 11001

print(a & b)   # 8  (AND)
print(a | b)   # 29 (OR)
print(a ^ b)   # 21 (XOR)
print(~a)      # -13 (NOT)
print(a << 2)  # 48 (Left shift)
print(a >> 2)  # 3  (Right shift)
```

### 6. Identity and Membership Operators
```python
# Identity operators
a = [1, 2, 3]
b = [1, 2, 3]
c = a

print(a is b)     # False (different objects)
print(a is c)     # True (same object)
print(a is not b) # True

# Membership operators
numbers = [1, 2, 3, 4, 5]
print(3 in numbers)     # True
print(6 not in numbers) # True

text = "Hello World"
print("Hello" in text)  # True
print("Python" not in text)  # True
```

## Input/Output

### Input
```python
# Basic input (always returns string)
name = input("Enter your name: ")
print(f"Hello, {name}!")

# Converting input
age = int(input("Enter your age: "))
salary = float(input("Enter your salary: "))

# Multiple inputs
x, y = input("Enter two numbers: ").split()
x, y = int(x), int(y)

# Or in one line
x, y = map(int, input("Enter two numbers: ").split())
```

### Output
```python
# Basic print
print("Hello, World!")

# Multiple values
print("Name:", "John", "Age:", 25)

# Separator and end parameters
print("A", "B", "C", sep="-")        # A-B-C
print("Hello", end=" ")
print("World")                       # Hello World

# Formatted output
name, age = "Alice", 30
print("Name: %s, Age: %d" % (name, age))
print("Name: {}, Age: {}".format(name, age))
print(f"Name: {name}, Age: {age}")

# Formatting numbers
pi = 3.14159
print(f"Pi: {pi:.2f}")              # Pi: 3.14
print(f"Pi: {pi:.4f}")              # Pi: 3.1416

# File output
with open("output.txt", "w") as file:
    print("Hello, File!", file=file)
```

## String Formatting and Methods

### Advanced String Formatting

#### 1. F-strings (Python 3.6+) - Recommended
```python
name = "Alice"
age = 30
score = 95.67

# Basic f-string formatting
message = f"Hello {name}, you are {age} years old"
print(message)

# Number formatting
print(f"Score: {score:.2f}")           # Score: 95.67
print(f"Score: {score:.0f}")           # Score: 96
print(f"Age: {age:04d}")              # Age: 0030

# Alignment and padding
print(f"Name: {name:<10}|")           # Left align
print(f"Name: {name:>10}|")           # Right align  
print(f"Name: {name:^10}|")           # Center align

# Date formatting
from datetime import datetime
now = datetime.now()
print(f"Current time: {now:%Y-%m-%d %H:%M:%S}")

# Expression in f-strings
x, y = 10, 20
print(f"Sum of {x} and {y} is {x + y}")

# Dictionary and attribute access
person = {"name": "Bob", "age": 25}
print(f"Person: {person['name']} is {person['age']} years old")
```

#### 2. Format Method
```python
# Positional arguments
template = "Hello {}, you are {} years old"
message = template.format("Charlie", 35)
print(message)

# Named arguments
template = "Hello {name}, you are {age} years old"
message = template.format(name="David", age=40)
print(message)

# Index-based formatting
template = "Item: {0}, Price: ${1:.2f}, Quantity: {2}"
message = template.format("Apple", 1.99, 5)
print(message)

# Format specifications
number = 42
print("{:b}".format(number))          # Binary: 101010
print("{:o}".format(number))          # Octal: 52
print("{:x}".format(number))          # Hex: 2a
print("{:e}".format(1234.5))          # Scientific: 1.234500e+03
```

#### 3. Percent Formatting (Legacy)
```python
name = "Eve"
age = 28
salary = 50000.75

# Basic formatting
print("Name: %s, Age: %d" % (name, age))

# Float formatting
print("Salary: $%.2f" % salary)

# Dictionary formatting
data = {"name": "Frank", "age": 45}
print("Name: %(name)s, Age: %(age)d" % data)
```

### String Methods and Operations

```python
text = "  Python Programming Language  "

# Case methods
print(text.upper())                    # Uppercase
print(text.lower())                    # Lowercase
print(text.title())                    # Title Case
print(text.capitalize())               # Capitalize first letter
print(text.swapcase())                 # Swap case

# Whitespace methods
print(text.strip())                    # Remove leading/trailing whitespace
print(text.lstrip())                   # Remove leading whitespace
print(text.rstrip())                   # Remove trailing whitespace

# Search and replace
print(text.find("Python"))             # Find substring index
print(text.count("o"))                 # Count occurrences
print(text.replace("Python", "Java"))  # Replace substring

# Splitting and joining
words = text.strip().split()           # Split into list
print(words)
sentence = " ".join(words)             # Join list into string
print(sentence)

# String validation
email = "user@example.com"
print(email.startswith("user"))        # Check if starts with
print(email.endswith(".com"))          # Check if ends with
print("123".isdigit())                 # Check if all digits
print("abc".isalpha())                 # Check if all letters
print("abc123".isalnum())              # Check if alphanumeric

# String alignment and padding
text = "Python"
print(text.center(20, "-"))            # Center with padding
print(text.ljust(20, "."))             # Left justify with padding
print(text.rjust(20, "*"))             # Right justify with padding
print(text.zfill(10))                  # Zero padding

# Advanced string operations
import string

# Remove punctuation
text_with_punct = "Hello, World! How are you?"
translator = str.maketrans("", "", string.punctuation)
clean_text = text_with_punct.translate(translator)
print(clean_text)  # Hello World How are you

# String constants
print(string.ascii_letters)            # All ASCII letters
print(string.digits)                   # All digits
print(string.punctuation)              # All punctuation
print(string.whitespace)               # All whitespace characters
```

### Regular Expressions with Strings

```python
import re

text = "Contact: john.doe@email.com or call 123-456-7890"

# Find email
email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
email = re.search(email_pattern, text)
if email:
    print(f"Email found: {email.group()}")

# Find phone number
phone_pattern = r'\d{3}-\d{3}-\d{4}'
phone = re.search(phone_pattern, text)
if phone:
    print(f"Phone found: {phone.group()}")

# Replace patterns
cleaned_text = re.sub(r'\d+', 'XXX', text)  # Replace digits with XXX
print(f"Cleaned: {cleaned_text}")
```

## Comments

```python
# Single-line comment

"""
Multi-line comment
or docstring
"""

'''
Another way to write
multi-line comments
'''

def greet(name):
    """
    This is a docstring.
    It describes what the function does.
    
    Args:
        name (str): The name to greet
    
    Returns:
        str: A greeting message
    """
    return f"Hello, {name}!"

# Inline comment
x = 5  # This is an inline comment
```

## Python Keywords

```python
# Python 3.9+ keywords (35 total)
import keyword
print(keyword.kwlist)

# Common keywords and their usage:
# False, True, None - Built-in constants
# and, or, not - Logical operators
# if, elif, else - Conditional statements
# for, while, break, continue - Loops
# def, return, yield - Functions
# class - Object-oriented programming
# import, from, as - Module imports
# try, except, finally, raise - Exception handling
# with, as - Context managers
# lambda - Anonymous functions
# global, nonlocal - Variable scope
# is, in - Identity and membership
# assert - Debugging
# del - Deletion
# pass - Placeholder
# async, await - Asynchronous programming
```

## Best Practices

1. **Naming Conventions**:
   - Variables and functions: `snake_case`
   - Constants: `UPPER_CASE`
   - Classes: `PascalCase`
   - Private variables: `_private_var`

2. **Code Style (PEP 8)**:
   - Use 4 spaces for indentation
   - Limit lines to 79 characters
   - Use blank lines to separate functions and classes
   - Import statements at the top

3. **Documentation**:
   - Write clear docstrings
   - Use meaningful variable names
   - Comment complex logic

```python
# Good example
def calculate_circle_area(radius):
    """Calculate the area of a circle given its radius."""
    pi = 3.14159
    return pi * radius ** 2

# Usage
circle_radius = 5
area = calculate_circle_area(circle_radius)
print(f"Area: {area:.2f}")
```

## Memory Management

### Python Memory Model

Python manages memory automatically through reference counting and garbage collection.

```python
import sys
import gc

# Reference counting
x = [1, 2, 3, 4, 5]
print(f"Reference count for x: {sys.getrefcount(x)}")

y = x  # Creates another reference
print(f"Reference count after y = x: {sys.getrefcount(x)}")

del y  # Removes one reference
print(f"Reference count after del y: {sys.getrefcount(x)}")

# Memory usage
def get_memory_usage():
    """Get current memory usage"""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # MB

# Object size
numbers = list(range(1000))
print(f"Size of list with 1000 numbers: {sys.getsizeof(numbers)} bytes")

string_data = "Hello" * 1000
print(f"Size of repeated string: {sys.getsizeof(string_data)} bytes")

# Garbage collection
print(f"Garbage collection thresholds: {gc.get_threshold()}")
print(f"Garbage collection counts: {gc.get_count()}")

# Force garbage collection
collected = gc.collect()
print(f"Objects collected: {collected}")

# Memory optimization tips
# 1. Use generators for large datasets
def number_generator(n):
    for i in range(n):
        yield i * i

# Generator uses constant memory
gen = number_generator(1000000)
print(f"Generator object size: {sys.getsizeof(gen)} bytes")

# List would use much more memory
# large_list = [i * i for i in range(1000000)]  # Don't run - uses lots of memory

# 2. Use __slots__ to reduce memory overhead
class RegularClass:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class SlottedClass:
    __slots__ = ['x', 'y']
    
    def __init__(self, x, y):
        self.x = x
        self.y = y

regular_obj = RegularClass(1, 2)
slotted_obj = SlottedClass(1, 2)

print(f"Regular object size: {sys.getsizeof(regular_obj)} bytes")
print(f"Slotted object size: {sys.getsizeof(slotted_obj)} bytes")

# 3. Weak references to avoid circular references
import weakref

class Node:
    def __init__(self, value):
        self.value = value
        self.children = []
        self.parent = None
    
    def add_child(self, child):
        child.parent = weakref.ref(self)  # Weak reference to parent
        self.children.append(child)

# Memory profiling example
def memory_profiling_example():
    """Example of memory profiling techniques"""
    import tracemalloc
    
    # Start tracing
    tracemalloc.start()
    
    # Some memory-intensive operations
    data = []
    for i in range(10000):
        data.append([i] * 100)
    
    # Get current memory usage
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 1024 / 1024:.2f} MB")
    print(f"Peak memory usage: {peak / 1024 / 1024:.2f} MB")
    
    # Stop tracing
    tracemalloc.stop()

# Run memory profiling
memory_profiling_example()
```

### Memory Best Practices

```python
# 1. Delete large objects when done
large_data = list(range(1000000))
# Process data...
del large_data  # Free memory immediately

# 2. Use context managers for resources
with open('large_file.txt', 'r') as f:
    content = f.read()
# File automatically closed, memory freed

# 3. Avoid creating unnecessary copies
original_list = [1, 2, 3, 4, 5]

# Bad - creates copy
doubled_bad = [x * 2 for x in original_list]

# Better - use generator if you only iterate once
doubled_good = (x * 2 for x in original_list)

# 4. Use appropriate data structures
import array

# Use array for numeric data instead of list
numbers_list = [1, 2, 3, 4, 5] * 1000  # More memory
numbers_array = array.array('i', [1, 2, 3, 4, 5] * 1000)  # Less memory

print(f"List size: {sys.getsizeof(numbers_list)} bytes")
print(f"Array size: {sys.getsizeof(numbers_array)} bytes")
```

## Python Package Management

### pip - Python Package Installer

```python
# Basic pip commands (run in terminal/command prompt)

# Install a package
# pip install package_name

# Install specific version
# pip install package_name==1.2.3

# Install from requirements file
# pip install -r requirements.txt

# Upgrade a package
# pip install --upgrade package_name

# Uninstall a package
# pip uninstall package_name

# List installed packages
# pip list

# Show package information
# pip show package_name

# Generate requirements file
# pip freeze > requirements.txt

# Install in development mode
# pip install -e .  # For local package development
```

### Virtual Environments Management

```python
# Using venv (built-in)
import venv
import subprocess
import sys

def create_virtual_environment(env_name):
    """Create a virtual environment programmatically"""
    venv.create(env_name, with_pip=True)
    print(f"Virtual environment '{env_name}' created successfully")

# create_virtual_environment("my_project_env")

# Using virtualenv (third-party)
# pip install virtualenv
# virtualenv myenv
# source myenv/bin/activate  # On Unix
# myenv\Scripts\activate     # On Windows

# Using conda (for data science)
# conda create --name myenv python=3.9
# conda activate myenv
# conda install numpy pandas matplotlib
# conda list
# conda deactivate
```

### Package Creation and Distribution

```python
# setup.py example for creating your own package
setup_py_content = '''
from setuptools import setup, find_packages

setup(
    name="my_awesome_package",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A short description of your package",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/my_awesome_package",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "requests>=2.25.0",
        "numpy>=1.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
        ]
    },
    entry_points={
        "console_scripts": [
            "my-awesome-tool=my_awesome_package.cli:main",
        ],
    },
)
'''

# Build and distribute package
# python setup.py sdist bdist_wheel
# twine upload dist/*
```

### Package Management Best Practices

```python
# 1. Use requirements files for different environments

# requirements.txt (production)
requirements_prod = '''
Django==4.2.0
psycopg2-binary==2.9.5
gunicorn==20.1.0
'''

# requirements-dev.txt (development)
requirements_dev = '''
-r requirements.txt
pytest==7.2.0
black==23.1.0
flake8==6.0.0
coverage==7.1.0
'''

# 2. Pin exact versions for reproducible builds
# Good: Django==4.2.0
# Avoid: Django>=4.0  (unless you need flexibility)

# 3. Use virtual environments for isolation
print("Always use virtual environments!")
print("Never install packages globally unless absolutely necessary")

# 4. Keep requirements files updated
# pip-tools can help manage dependencies
# pip install pip-tools
# pip-compile requirements.in  # Generate requirements.txt

# 5. Security scanning
# pip install safety
# safety check  # Check for known security vulnerabilities

# 6. Dependency analysis
def analyze_dependencies():
    """Analyze installed packages and their dependencies"""
    import pkg_resources
    
    installed_packages = [d for d in pkg_resources.working_set]
    
    print("Installed packages:")
    for package in sorted(installed_packages, key=lambda x: x.project_name):
        print(f"{package.project_name}: {package.version}")
    
    # Check for unused packages
    print("\nTip: Use 'pip-autoremove' to find unused packages")
    print("pip install pip-autoremove")
    print("pip-autoremove package_name -y")

# analyze_dependencies()  # Uncomment to run

print("Package management setup completed!")
```
