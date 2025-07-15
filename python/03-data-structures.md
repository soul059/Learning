# Data Structures in Python

## Table of Contents
1. [Lists](#lists)
2. [Tuples](#tuples)
3. [Dictionaries](#dictionaries)
4. [Sets](#sets)
5. [Strings (Advanced)](#strings-advanced)
6. [Collections Module](#collections-module)
7. [Data Structure Performance](#data-structure-performance)
8. [Memory Efficiency](#memory-efficiency)
9. [Advanced Patterns](#advanced-patterns)
10. [Custom Data Structures](#custom-data-structures)

## Lists

Lists are ordered, mutable collections that can store elements of different data types.

### 1. List Creation and Basic Operations
```python
# Creating lists
empty_list = []
numbers = [1, 2, 3, 4, 5]
mixed_list = [1, "hello", 3.14, True]
nested_list = [[1, 2], [3, 4], [5, 6]]

# List from other iterables
list_from_string = list("Python")  # ['P', 'y', 't', 'h', 'o', 'n']
list_from_range = list(range(5))   # [0, 1, 2, 3, 4]

# Accessing elements
fruits = ["apple", "banana", "cherry", "date"]
print(fruits[0])    # apple (first element)
print(fruits[-1])   # date (last element)
print(fruits[1:3])  # ['banana', 'cherry'] (slicing)
print(fruits[:2])   # ['apple', 'banana'] (from start)
print(fruits[2:])   # ['cherry', 'date'] (to end)
print(fruits[::2])  # ['apple', 'cherry'] (step slicing)
```

### 2. List Methods
```python
numbers = [1, 2, 3]

# Adding elements
numbers.append(4)           # [1, 2, 3, 4]
numbers.insert(1, 1.5)      # [1, 1.5, 2, 3, 4]
numbers.extend([5, 6])      # [1, 1.5, 2, 3, 4, 5, 6]

# Removing elements
numbers.remove(1.5)         # Remove first occurrence
popped = numbers.pop()      # Remove and return last element
popped_index = numbers.pop(0)  # Remove and return element at index
numbers.clear()             # Remove all elements

# Finding elements
fruits = ["apple", "banana", "cherry", "banana"]
index = fruits.index("banana")      # 1 (first occurrence)
count = fruits.count("banana")      # 2

# Sorting and reversing
numbers = [3, 1, 4, 1, 5, 9]
numbers.sort()              # [1, 1, 3, 4, 5, 9] (modifies original)
numbers.reverse()           # [9, 5, 4, 3, 1, 1]

# Creating sorted copy
sorted_numbers = sorted([3, 1, 4, 1, 5, 9])  # Original unchanged
reversed_list = list(reversed([1, 2, 3]))    # [3, 2, 1]
```

### 3. List Comprehensions
```python
# Basic list comprehension
squares = [x**2 for x in range(10)]
# [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# With condition
even_squares = [x**2 for x in range(10) if x % 2 == 0]
# [0, 4, 16, 36, 64]

# Complex expressions
words = ["hello", "world", "python"]
lengths = [len(word) for word in words]
# [5, 5, 6]

# Nested list comprehension
matrix = [[i * j for j in range(1, 4)] for i in range(1, 4)]
# [[1, 2, 3], [2, 4, 6], [3, 6, 9]]

# Flattening nested lists
nested = [[1, 2], [3, 4], [5, 6]]
flattened = [item for sublist in nested for item in sublist]
# [1, 2, 3, 4, 5, 6]

# Conditional expression in comprehension
numbers = [1, 2, 3, 4, 5]
result = ["even" if x % 2 == 0 else "odd" for x in numbers]
# ['odd', 'even', 'odd', 'even', 'odd']
```

### 4. Advanced List Operations
```python
# Copying lists
original = [1, 2, 3]
shallow_copy = original.copy()      # or original[:]
import copy
deep_copy = copy.deepcopy(original)

# List concatenation
list1 = [1, 2, 3]
list2 = [4, 5, 6]
combined = list1 + list2            # [1, 2, 3, 4, 5, 6]
list1 += list2                      # Modifies list1

# List multiplication
repeated = [0] * 5                  # [0, 0, 0, 0, 0]
pattern = [1, 2] * 3                # [1, 2, 1, 2, 1, 2]

# Unpacking
numbers = [1, 2, 3, 4, 5]
first, *middle, last = numbers      # first=1, middle=[2,3,4], last=5
a, b, c = [1, 2, 3]                # a=1, b=2, c=3
```

## Tuples

Tuples are ordered, immutable collections.

### 1. Tuple Creation and Operations
```python
# Creating tuples
empty_tuple = ()
single_element = (42,)              # Comma needed for single element
coordinates = (10, 20)
mixed_tuple = (1, "hello", 3.14)
nested_tuple = ((1, 2), (3, 4))

# Tuple from other iterables
tuple_from_list = tuple([1, 2, 3])
tuple_from_string = tuple("Python")

# Accessing elements (same as lists)
point = (10, 20, 30)
x, y, z = point                     # Tuple unpacking
print(point[0])                     # 10
print(point[-1])                    # 30
print(point[1:])                    # (20, 30)
```

### 2. Tuple Methods and Operations
```python
numbers = (1, 2, 3, 2, 4, 2)

# Tuple methods
count_2 = numbers.count(2)          # 3
index_3 = numbers.index(3)          # 2

# Tuple operations
length = len(numbers)               # 6
maximum = max(numbers)              # 4
minimum = min(numbers)              # 1
total = sum(numbers)                # 14

# Membership testing
print(3 in numbers)                 # True
print(5 not in numbers)             # True

# Comparison
tuple1 = (1, 2, 3)
tuple2 = (1, 2, 4)
print(tuple1 < tuple2)              # True (lexicographic comparison)
```

### 3. Named Tuples
```python
from collections import namedtuple

# Creating named tuple class
Point = namedtuple('Point', ['x', 'y'])
Person = namedtuple('Person', ['name', 'age', 'email'])

# Creating instances
p1 = Point(10, 20)
p2 = Point(x=5, y=15)
person = Person("Alice", 30, "alice@email.com")

# Accessing elements
print(p1.x)                         # 10
print(p1[0])                        # 10 (also works)
print(person.name)                  # Alice

# Named tuple methods
print(p1._fields)                   # ('x', 'y')
point_dict = p1._asdict()           # {'x': 10, 'y': 20}
new_point = p1._replace(y=25)       # Point(x=10, y=25)
```

## Dictionaries

Dictionaries are unordered, mutable collections of key-value pairs.

### 1. Dictionary Creation and Basic Operations
```python
# Creating dictionaries
empty_dict = {}
student = {"name": "John", "age": 20, "grade": "A"}
mixed_dict = {1: "one", "two": 2, 3.0: "three"}

# Dictionary from other methods
dict_from_tuples = dict([("a", 1), ("b", 2)])
dict_from_kwargs = dict(name="Alice", age=25)
dict_comprehension = {x: x**2 for x in range(5)}

# Accessing elements
print(student["name"])              # John
print(student.get("age"))           # 20
print(student.get("city", "Unknown"))  # Unknown (default value)

# Modifying dictionaries
student["age"] = 21                 # Update existing key
student["city"] = "New York"        # Add new key
del student["grade"]                # Delete key
removed_value = student.pop("city") # Remove and return value
```

### 2. Dictionary Methods
```python
student = {"name": "John", "age": 20, "grade": "A"}

# Getting keys, values, items
keys = student.keys()               # dict_keys(['name', 'age', 'grade'])
values = student.values()           # dict_values(['John', 20, 'A'])
items = student.items()             # dict_items([('name', 'John'), ...])

# Converting to lists
key_list = list(student.keys())
value_list = list(student.values())

# Updating dictionaries
student.update({"city": "Boston", "age": 21})
student.update([("country", "USA")])

# Dictionary methods
copy_dict = student.copy()
student.clear()                     # Remove all items

# setdefault method
grades = {}
grades.setdefault("math", []).append(90)
grades.setdefault("math", []).append(85)
# grades = {"math": [90, 85]}
```

### 3. Dictionary Comprehensions
```python
# Basic dictionary comprehension
squares = {x: x**2 for x in range(5)}
# {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

# With condition
even_squares = {x: x**2 for x in range(10) if x % 2 == 0}
# {0: 0, 2: 4, 4: 16, 6: 36, 8: 64}

# From existing dictionary
prices = {"apple": 0.5, "banana": 0.3, "cherry": 1.0}
expensive = {k: v for k, v in prices.items() if v > 0.4}
# {"apple": 0.5, "cherry": 1.0}

# Swapping keys and values
original = {"a": 1, "b": 2, "c": 3}
swapped = {v: k for k, v in original.items()}
# {1: "a", 2: "b", 3: "c"}
```

### 4. Advanced Dictionary Operations
```python
# Merging dictionaries (Python 3.9+)
dict1 = {"a": 1, "b": 2}
dict2 = {"c": 3, "d": 4}
merged = dict1 | dict2              # {"a": 1, "b": 2, "c": 3, "d": 4}

# Pre-Python 3.9 merging
merged = {**dict1, **dict2}

# Nested dictionaries
students = {
    "student1": {"name": "Alice", "grades": [90, 85, 92]},
    "student2": {"name": "Bob", "grades": [78, 82, 88]}
}

# Accessing nested data
alice_grades = students["student1"]["grades"]
bob_name = students["student2"]["name"]

# Default dictionaries
from collections import defaultdict

# String grouping
dd = defaultdict(list)
words = ["apple", "banana", "apricot", "blueberry"]
for word in words:
    dd[word[0]].append(word)
# defaultdict(<class 'list'>, {'a': ['apple', 'apricot'], 'b': ['banana', 'blueberry']})
```

## Sets

Sets are unordered collections of unique elements.

### 1. Set Creation and Basic Operations
```python
# Creating sets
empty_set = set()                   # Note: {} creates empty dict
numbers = {1, 2, 3, 4, 5}
mixed_set = {1, "hello", 3.14}

# Set from other iterables
set_from_list = set([1, 2, 2, 3, 3])  # {1, 2, 3}
set_from_string = set("hello")         # {'h', 'e', 'l', 'o'}

# Adding and removing elements
fruits = {"apple", "banana"}
fruits.add("cherry")                # Add single element
fruits.update(["date", "elderberry"])  # Add multiple elements
fruits.remove("banana")             # Remove element (raises KeyError if not found)
fruits.discard("grape")             # Remove element (no error if not found)
popped = fruits.pop()               # Remove and return arbitrary element
```

### 2. Set Operations
```python
set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}

# Union (all elements from both sets)
union = set1 | set2                 # {1, 2, 3, 4, 5, 6}
union = set1.union(set2)

# Intersection (common elements)
intersection = set1 & set2          # {3, 4}
intersection = set1.intersection(set2)

# Difference (elements in set1 but not in set2)
difference = set1 - set2            # {1, 2}
difference = set1.difference(set2)

# Symmetric difference (elements in either set, but not both)
sym_diff = set1 ^ set2              # {1, 2, 5, 6}
sym_diff = set1.symmetric_difference(set2)

# Subset and superset checks
small_set = {1, 2}
print(small_set.issubset(set1))     # True
print(set1.issuperset(small_set))   # True
print(set1.isdisjoint(set2))        # False (they have common elements)
```

### 3. Set Comprehensions
```python
# Basic set comprehension
squares = {x**2 for x in range(10)}
# {0, 1, 4, 9, 16, 25, 36, 49, 64, 81}

# With condition
even_squares = {x**2 for x in range(10) if x % 2 == 0}
# {0, 4, 16, 36, 64}

# From string
vowels = {char.lower() for char in "Hello World" if char.lower() in "aeiou"}
# {'e', 'o'}
```

### 4. Frozen Sets
```python
# Immutable sets
frozen = frozenset([1, 2, 3, 4])
# frozen.add(5)  # This would raise AttributeError

# Frozen sets can be dictionary keys or set elements
dict_with_frozen_keys = {frozenset([1, 2]): "value1"}
set_of_frozen_sets = {frozenset([1, 2]), frozenset([3, 4])}
```

## Strings (Advanced)

### 1. String Methods
```python
text = "  Hello, World!  "

# Case methods
print(text.upper())         # "  HELLO, WORLD!  "
print(text.lower())         # "  hello, world!  "
print(text.title())         # "  Hello, World!  "
print(text.capitalize())    # "  hello, world!  "
print(text.swapcase())      # "  hELLO, wORLD!  "

# Whitespace methods
print(text.strip())         # "Hello, World!"
print(text.lstrip())        # "Hello, World!  "
print(text.rstrip())        # "  Hello, World!"

# Search methods
print(text.find("World"))   # 9 (index of first occurrence)
print(text.rfind("l"))      # 11 (index of last occurrence)
print(text.index("World"))  # 9 (like find, but raises ValueError if not found)
print(text.count("l"))      # 3

# Boolean methods
print("123".isdigit())      # True
print("abc".isalpha())      # True
print("abc123".isalnum())   # True
print("   ".isspace())      # True
print("Hello World".istitle())  # True
```

### 2. String Formatting
```python
name = "Alice"
age = 30
salary = 50000.75

# f-strings (Python 3.6+)
message = f"Name: {name}, Age: {age}, Salary: ${salary:.2f}"

# Format specifiers
pi = 3.14159
print(f"Pi: {pi:.2f}")          # Pi: 3.14
print(f"Pi: {pi:.4f}")          # Pi: 3.1416
print(f"Number: {42:05d}")      # Number: 00042
print(f"Percent: {0.85:.2%}")   # Percent: 85.00%

# .format() method
template = "Name: {}, Age: {}, Salary: ${:.2f}"
message = template.format(name, age, salary)

# Named placeholders
template = "Name: {n}, Age: {a}, Salary: ${s:.2f}"
message = template.format(n=name, a=age, s=salary)

# % formatting (older style)
message = "Name: %s, Age: %d, Salary: $%.2f" % (name, age, salary)
```

### 3. String Splitting and Joining
```python
# Splitting strings
data = "apple,banana,cherry"
fruits = data.split(",")            # ['apple', 'banana', 'cherry']

text = "Hello   World   Python"
words = text.split()                # ['Hello', 'World', 'Python']
words_limited = text.split(" ", 1)  # ['Hello', '  World   Python']

# Partition and rpartition
email = "user@domain.com"
user, sep, domain = email.partition("@")  # ('user', '@', 'domain.com')

# Joining strings
fruits = ["apple", "banana", "cherry"]
joined = ", ".join(fruits)          # "apple, banana, cherry"
path = "/".join(["home", "user", "documents"])  # "home/user/documents"

# Joining with different types
numbers = [1, 2, 3, 4]
joined_numbers = ", ".join(str(n) for n in numbers)  # "1, 2, 3, 4"
```

### 4. String Replacement and Translation
```python
text = "Hello, World! Hello, Python!"

# Replace method
new_text = text.replace("Hello", "Hi")              # "Hi, World! Hi, Python!"
new_text = text.replace("Hello", "Hi", 1)           # "Hi, World! Hello, Python!"

# Translation table
translation = str.maketrans("aeiou", "12345")
translated = "hello world".translate(translation)   # "h2ll4 w4rld"

# Remove characters
remove_vowels = str.maketrans("", "", "aeiou")
no_vowels = "hello world".translate(remove_vowels)  # "hll wrld"
```

## Collections Module

### 1. Counter
```python
from collections import Counter

# Counting elements
text = "hello world"
char_count = Counter(text)          # Counter({'l': 3, 'o': 2, 'h': 1, ...})
word_count = Counter(["apple", "banana", "apple", "cherry", "apple"])

# Most common elements
print(char_count.most_common(3))    # [('l', 3), ('o', 2), ('h', 1)]

# Counter operations
counter1 = Counter(['a', 'b', 'c', 'a'])
counter2 = Counter(['a', 'b', 'b'])
print(counter1 + counter2)          # Counter({'a': 3, 'b': 3, 'c': 1})
print(counter1 - counter2)          # Counter({'c': 1, 'a': 1})
```

### 2. defaultdict
```python
from collections import defaultdict

# Group words by first letter
word_groups = defaultdict(list)
words = ["apple", "banana", "apricot", "cherry", "avocado"]
for word in words:
    word_groups[word[0]].append(word)

# Count nested items
nested_counter = defaultdict(int)
data = [("a", 1), ("b", 2), ("a", 3), ("c", 1)]
for key, value in data:
    nested_counter[key] += value
```

### 3. deque (Double-ended queue)
```python
from collections import deque

# Creating deque
d = deque([1, 2, 3])

# Adding elements
d.append(4)         # Add to right: deque([1, 2, 3, 4])
d.appendleft(0)     # Add to left: deque([0, 1, 2, 3, 4])

# Removing elements
right = d.pop()     # Remove from right: 4
left = d.popleft()  # Remove from left: 0

# Rotating
d.rotate(1)         # Rotate right: deque([3, 1, 2])
d.rotate(-1)        # Rotate left: deque([1, 2, 3])

# Maximum length deque
limited_deque = deque(maxlen=3)
for i in range(5):
    limited_deque.append(i)  # Only keeps last 3 elements
print(limited_deque)        # deque([2, 3, 4], maxlen=3)
```

### 4. OrderedDict
```python
from collections import OrderedDict

# Maintains insertion order (note: regular dicts maintain order in Python 3.7+)
ordered = OrderedDict()
ordered['first'] = 1
ordered['second'] = 2
ordered['third'] = 3

# Move to end
ordered.move_to_end('first')    # Move 'first' to the end
ordered.move_to_end('second', last=False)  # Move 'second' to the beginning

# Pop items
last_item = ordered.popitem()           # Remove and return last item
first_item = ordered.popitem(last=False) # Remove and return first item
```

## Best Practices

1. **Choose the right data structure**:
   - Use lists for ordered, mutable sequences
   - Use tuples for ordered, immutable sequences
   - Use dictionaries for key-value mappings
   - Use sets for unique elements and set operations

2. **Memory efficiency**:
```python
# Use generators for large datasets
squares_gen = (x**2 for x in range(1000000))  # Generator
squares_list = [x**2 for x in range(1000000)] # List (uses more memory)

# Use appropriate data structures
# For lookups, use sets or dictionaries (O(1)) instead of lists (O(n))
valid_ids = {1, 2, 3, 4, 5}  # Fast lookup
if user_id in valid_ids:      # O(1) operation
    process_user()
```

3. **String operations**:
```python
# Efficient string concatenation
# Bad for many strings
result = ""
for item in items:
    result += str(item)

# Good
result = "".join(str(item) for item in items)
```

4. **Dictionary operations**:
```python
# Use get() method with defaults
value = dictionary.get(key, default_value)
# Instead of
# if key in dictionary:
#     value = dictionary[key]
# else:
#     value = default_value
```

## Data Structure Performance

### Time Complexity Comparison

```python
import time
import sys

def time_operation(func, *args, **kwargs):
    """Time a function execution"""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return result, end - start

# List vs Set lookup performance
def compare_lookup_performance():
    """Compare lookup performance between list and set"""
    
    # Create test data
    size = 10000
    data_list = list(range(size))
    data_set = set(range(size))
    search_value = size - 1  # Worst case for list
    
    # List lookup
    _, list_time = time_operation(lambda: search_value in data_list)
    
    # Set lookup
    _, set_time = time_operation(lambda: search_value in data_set)
    
    print(f"List lookup time: {list_time:.6f} seconds")
    print(f"Set lookup time: {set_time:.6f} seconds")
    print(f"Set is {list_time/set_time:.2f}x faster")

compare_lookup_performance()

# Dictionary vs List for key-value operations
def compare_dict_vs_list():
    """Compare dictionary vs list for key-value storage"""
    
    size = 10000
    
    # List of tuples approach
    data_list = [(i, f"value_{i}") for i in range(size)]
    
    # Dictionary approach
    data_dict = {i: f"value_{i}" for i in range(size)}
    
    search_key = size - 1
    
    # List search (linear)
    def find_in_list(key):
        for k, v in data_list:
            if k == key:
                return v
        return None
    
    # Dictionary search (hash table)
    def find_in_dict(key):
        return data_dict.get(key)
    
    _, list_time = time_operation(find_in_list, search_key)
    _, dict_time = time_operation(find_in_dict, search_key)
    
    print(f"\nKey-value lookup comparison:")
    print(f"List approach: {list_time:.6f} seconds")
    print(f"Dict approach: {dict_time:.6f} seconds")
    print(f"Dictionary is {list_time/dict_time:.2f}x faster")

compare_dict_vs_list()

# Time Complexity Table
print("\n" + "="*60)
print("TIME COMPLEXITY CHEAT SHEET")
print("="*60)

complexity_table = """
Data Structure | Access | Search | Insertion | Deletion
-------------- | ------ | ------ | --------- | --------
List           | O(1)   | O(n)   | O(n)      | O(n)
Tuple          | O(1)   | O(n)   | N/A       | N/A
Dict           | O(1)   | O(1)   | O(1)      | O(1)
Set            | N/A    | O(1)   | O(1)      | O(1)
Deque          | O(n)   | O(n)   | O(1)      | O(1)
"""

print(complexity_table)
```

### Space Complexity Analysis

```python
import sys

def analyze_space_complexity():
    """Analyze space usage of different data structures"""
    
    size = 1000
    
    # Lists
    int_list = list(range(size))
    str_list = [str(i) for i in range(size)]
    
    # Tuples
    int_tuple = tuple(range(size))
    
    # Sets
    int_set = set(range(size))
    
    # Dictionaries
    int_dict = {i: i for i in range(size)}
    
    print(f"Space usage for {size} integers:")
    print(f"List:  {sys.getsizeof(int_list):,} bytes")
    print(f"Tuple: {sys.getsizeof(int_tuple):,} bytes")
    print(f"Set:   {sys.getsizeof(int_set):,} bytes")
    print(f"Dict:  {sys.getsizeof(int_dict):,} bytes")
    
    # Memory overhead per element
    print(f"\nApproximate memory per element:")
    print(f"List:  {sys.getsizeof(int_list) / size:.1f} bytes/element")
    print(f"Tuple: {sys.getsizeof(int_tuple) / size:.1f} bytes/element")
    print(f"Set:   {sys.getsizeof(int_set) / size:.1f} bytes/element")
    print(f"Dict:  {sys.getsizeof(int_dict) / size:.1f} bytes/element")

analyze_space_complexity()
```

## Memory Efficiency

### Memory-Efficient Patterns

```python
import sys
from array import array
import itertools

# 1. Use generators for large datasets
def memory_efficient_processing():
    """Demonstrate memory-efficient data processing"""
    
    # Memory-heavy approach
    def process_large_dataset_bad(size):
        data = [i**2 for i in range(size)]  # Creates entire list in memory
        return sum(data)
    
    # Memory-efficient approach
    def process_large_dataset_good(size):
        data = (i**2 for i in range(size))  # Generator - constant memory
        return sum(data)
    
    size = 1000000
    
    # Compare memory usage
    import tracemalloc
    
    # Measure memory-heavy approach
    tracemalloc.start()
    result1 = process_large_dataset_bad(size)
    current1, peak1 = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Measure memory-efficient approach
    tracemalloc.start()
    result2 = process_large_dataset_good(size)
    current2, peak2 = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"Results are equal: {result1 == result2}")
    print(f"Memory-heavy peak usage: {peak1 / 1024 / 1024:.2f} MB")
    print(f"Memory-efficient peak usage: {peak2 / 1024 / 1024:.2f} MB")
    print(f"Memory saved: {(peak1 - peak2) / 1024 / 1024:.2f} MB")

memory_efficient_processing()

# 2. Use array for numeric data
def compare_list_vs_array():
    """Compare memory usage of list vs array for numeric data"""
    
    size = 10000
    
    # Python list (stores references to int objects)
    python_list = [i for i in range(size)]
    
    # Array (stores actual values)
    int_array = array('i', range(size))  # 'i' = signed int
    
    print(f"\nNumeric data storage comparison ({size:,} integers):")
    print(f"Python list: {sys.getsizeof(python_list):,} bytes")
    print(f"Array:       {sys.getsizeof(int_array):,} bytes")
    print(f"Space saved: {sys.getsizeof(python_list) - sys.getsizeof(int_array):,} bytes")
    print(f"Efficiency:  {sys.getsizeof(int_array) / sys.getsizeof(python_list):.2%}")

compare_list_vs_array()

# 3. Memory-efficient string operations
def efficient_string_operations():
    """Demonstrate efficient string operations"""
    
    # Inefficient string concatenation
    def concat_inefficient(words):
        result = ""
        for word in words:
            result += word + " "
        return result.strip()
    
    # Efficient string concatenation
    def concat_efficient(words):
        return " ".join(words)
    
    words = ["word"] * 1000
    
    import time
    
    # Time inefficient approach
    start = time.time()
    result1 = concat_inefficient(words)
    time1 = time.time() - start
    
    # Time efficient approach
    start = time.time()
    result2 = concat_efficient(words)
    time2 = time.time() - start
    
    print(f"\nString concatenation comparison:")
    print(f"Inefficient approach: {time1:.6f} seconds")
    print(f"Efficient approach:   {time2:.6f} seconds")
    print(f"Speedup: {time1/time2:.2f}x faster")

efficient_string_operations()

# 4. Use __slots__ for classes
class RegularClass:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class SlottedClass:
    __slots__ = ['x', 'y', 'z']
    
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

def compare_class_memory():
    """Compare memory usage of regular vs slotted classes"""
    
    regular_objects = [RegularClass(i, i+1, i+2) for i in range(1000)]
    slotted_objects = [SlottedClass(i, i+1, i+2) for i in range(1000)]
    
    regular_size = sum(sys.getsizeof(obj) + sys.getsizeof(obj.__dict__) 
                      for obj in regular_objects)
    slotted_size = sum(sys.getsizeof(obj) for obj in slotted_objects)
    
    print(f"\nClass memory comparison (1000 objects):")
    print(f"Regular classes: {regular_size:,} bytes")
    print(f"Slotted classes: {slotted_size:,} bytes")
    print(f"Memory saved: {regular_size - slotted_size:,} bytes")
    print(f"Efficiency: {slotted_size / regular_size:.2%}")

compare_class_memory()
```

## Advanced Patterns

### Data Structure Design Patterns

```python
from collections import defaultdict, Counter
from typing import List, Dict, Any, Optional
import heapq
import bisect

# 1. Cache/Memoization Pattern
class LRUCache:
    """Least Recently Used Cache implementation"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.order = []
    
    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            # Move to end (most recently used)
            self.order.remove(key)
            self.order.append(key)
            return self.cache[key]
        return None
    
    def put(self, key: str, value: Any) -> None:
        if key in self.cache:
            # Update existing
            self.cache[key] = value
            self.order.remove(key)
            self.order.append(key)
        else:
            # Add new
            if len(self.cache) >= self.capacity:
                # Remove least recently used
                oldest = self.order.pop(0)
                del self.cache[oldest]
            
            self.cache[key] = value
            self.order.append(key)
    
    def size(self) -> int:
        return len(self.cache)

# Example usage
cache = LRUCache(3)
cache.put("a", 1)
cache.put("b", 2)
cache.put("c", 3)
print(f"Cache size: {cache.size()}")
print(f"Get 'a': {cache.get('a')}")
cache.put("d", 4)  # This will evict 'b'
print(f"Get 'b': {cache.get('b')}")  # None

# 2. Trie (Prefix Tree) Pattern
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_word = False

class Trie:
    """Trie data structure for efficient string operations"""
    
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word: str) -> None:
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_word = True
    
    def search(self, word: str) -> bool:
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_word
    
    def starts_with(self, prefix: str) -> bool:
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True
    
    def find_words_with_prefix(self, prefix: str) -> List[str]:
        """Find all words that start with given prefix"""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        
        results = []
        self._dfs(node, prefix, results)
        return results
    
    def _dfs(self, node: TrieNode, current_word: str, results: List[str]) -> None:
        if node.is_end_word:
            results.append(current_word)
        
        for char, child_node in node.children.items():
            self._dfs(child_node, current_word + char, results)

# Example usage
trie = Trie()
words = ["apple", "app", "application", "apply", "banana", "band"]
for word in words:
    trie.insert(word)

print(f"\nTrie operations:")
print(f"Search 'app': {trie.search('app')}")
print(f"Starts with 'app': {trie.starts_with('app')}")
print(f"Words with prefix 'app': {trie.find_words_with_prefix('app')}")

# 3. Graph Adjacency List Pattern
class Graph:
    """Graph implementation using adjacency list"""
    
    def __init__(self, directed: bool = False):
        self.graph = defaultdict(list)
        self.directed = directed
    
    def add_edge(self, u: str, v: str, weight: int = 1) -> None:
        self.graph[u].append((v, weight))
        if not self.directed:
            self.graph[v].append((u, weight))
    
    def get_neighbors(self, vertex: str) -> List[tuple]:
        return self.graph[vertex]
    
    def bfs(self, start: str) -> List[str]:
        """Breadth-first search"""
        visited = set()
        queue = [start]
        result = []
        
        while queue:
            vertex = queue.pop(0)
            if vertex not in visited:
                visited.add(vertex)
                result.append(vertex)
                for neighbor, _ in self.graph[vertex]:
                    if neighbor not in visited:
                        queue.append(neighbor)
        
        return result
    
    def dfs(self, start: str) -> List[str]:
        """Depth-first search"""
        visited = set()
        result = []
        
        def dfs_recursive(vertex: str):
            visited.add(vertex)
            result.append(vertex)
            for neighbor, _ in self.graph[vertex]:
                if neighbor not in visited:
                    dfs_recursive(neighbor)
        
        dfs_recursive(start)
        return result

# Example usage
graph = Graph()
edges = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("D", "E")]
for u, v in edges:
    graph.add_edge(u, v)

print(f"\nGraph traversal:")
print(f"BFS from A: {graph.bfs('A')}")
print(f"DFS from A: {graph.dfs('A')}")

# 4. Priority Queue Pattern
class PriorityQueue:
    """Priority queue implementation using heapq"""
    
    def __init__(self):
        self.heap = []
        self.index = 0
    
    def push(self, item: Any, priority: int) -> None:
        # Use negative priority for max heap behavior
        # Include index to break ties and maintain insertion order
        heapq.heappush(self.heap, (priority, self.index, item))
        self.index += 1
    
    def pop(self) -> Any:
        if self.heap:
            return heapq.heappop(self.heap)[2]  # Return item only
        raise IndexError("pop from empty priority queue")
    
    def peek(self) -> Any:
        if self.heap:
            return self.heap[0][2]
        raise IndexError("peek from empty priority queue")
    
    def is_empty(self) -> bool:
        return len(self.heap) == 0
    
    def size(self) -> int:
        return len(self.heap)

# Example usage
pq = PriorityQueue()
pq.push("Low priority task", 3)
pq.push("High priority task", 1)
pq.push("Medium priority task", 2)

print(f"\nPriority Queue operations:")
while not pq.is_empty():
    print(f"Processing: {pq.pop()}")
```

## Custom Data Structures

### Implementing Custom Collections

```python
from typing import Iterator, Any, Optional

# 1. Custom Stack Implementation
class Stack:
    """Stack data structure implementation"""
    
    def __init__(self):
        self._items = []
    
    def push(self, item: Any) -> None:
        """Add item to top of stack"""
        self._items.append(item)
    
    def pop(self) -> Any:
        """Remove and return top item"""
        if self.is_empty():
            raise IndexError("pop from empty stack")
        return self._items.pop()
    
    def peek(self) -> Any:
        """Return top item without removing"""
        if self.is_empty():
            raise IndexError("peek from empty stack")
        return self._items[-1]
    
    def is_empty(self) -> bool:
        return len(self._items) == 0
    
    def size(self) -> int:
        return len(self._items)
    
    def __str__(self) -> str:
        return f"Stack({self._items})"

# 2. Custom Queue Implementation
class Queue:
    """Queue data structure implementation"""
    
    def __init__(self):
        self._items = []
    
    def enqueue(self, item: Any) -> None:
        """Add item to rear of queue"""
        self._items.append(item)
    
    def dequeue(self) -> Any:
        """Remove and return front item"""
        if self.is_empty():
            raise IndexError("dequeue from empty queue")
        return self._items.pop(0)
    
    def front(self) -> Any:
        """Return front item without removing"""
        if self.is_empty():
            raise IndexError("front from empty queue")
        return self._items[0]
    
    def is_empty(self) -> bool:
        return len(self._items) == 0
    
    def size(self) -> int:
        return len(self._items)
    
    def __str__(self) -> str:
        return f"Queue({self._items})"

# 3. Custom Linked List Implementation
class ListNode:
    def __init__(self, val: Any = 0, next_node: Optional['ListNode'] = None):
        self.val = val
        self.next = next_node

class LinkedList:
    """Singly linked list implementation"""
    
    def __init__(self):
        self.head = None
        self._size = 0
    
    def append(self, val: Any) -> None:
        """Add element to end of list"""
        new_node = ListNode(val)
        if not self.head:
            self.head = new_node
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node
        self._size += 1
    
    def prepend(self, val: Any) -> None:
        """Add element to beginning of list"""
        new_node = ListNode(val, self.head)
        self.head = new_node
        self._size += 1
    
    def delete(self, val: Any) -> bool:
        """Delete first occurrence of value"""
        if not self.head:
            return False
        
        if self.head.val == val:
            self.head = self.head.next
            self._size -= 1
            return True
        
        current = self.head
        while current.next:
            if current.next.val == val:
                current.next = current.next.next
                self._size -= 1
                return True
            current = current.next
        
        return False
    
    def find(self, val: Any) -> bool:
        """Check if value exists in list"""
        current = self.head
        while current:
            if current.val == val:
                return True
            current = current.next
        return False
    
    def size(self) -> int:
        return self._size
    
    def to_list(self) -> List[Any]:
        """Convert to Python list"""
        result = []
        current = self.head
        while current:
            result.append(current.val)
            current = current.next
        return result
    
    def __str__(self) -> str:
        return f"LinkedList({self.to_list()})"

# 4. Custom Binary Search Tree
class TreeNode:
    def __init__(self, val: Any = 0, left: Optional['TreeNode'] = None, 
                 right: Optional['TreeNode'] = None):
        self.val = val
        self.left = left
        self.right = right

class BinarySearchTree:
    """Binary Search Tree implementation"""
    
    def __init__(self):
        self.root = None
    
    def insert(self, val: Any) -> None:
        """Insert value into BST"""
        self.root = self._insert_recursive(self.root, val)
    
    def _insert_recursive(self, node: Optional[TreeNode], val: Any) -> TreeNode:
        if not node:
            return TreeNode(val)
        
        if val < node.val:
            node.left = self._insert_recursive(node.left, val)
        elif val > node.val:
            node.right = self._insert_recursive(node.right, val)
        
        return node
    
    def search(self, val: Any) -> bool:
        """Search for value in BST"""
        return self._search_recursive(self.root, val)
    
    def _search_recursive(self, node: Optional[TreeNode], val: Any) -> bool:
        if not node:
            return False
        
        if val == node.val:
            return True
        elif val < node.val:
            return self._search_recursive(node.left, val)
        else:
            return self._search_recursive(node.right, val)
    
    def inorder_traversal(self) -> List[Any]:
        """Return inorder traversal (sorted order)"""
        result = []
        self._inorder_recursive(self.root, result)
        return result
    
    def _inorder_recursive(self, node: Optional[TreeNode], result: List[Any]) -> None:
        if node:
            self._inorder_recursive(node.left, result)
            result.append(node.val)
            self._inorder_recursive(node.right, result)

# Example usage of custom data structures
print("\n" + "="*50)
print("CUSTOM DATA STRUCTURES EXAMPLES")
print("="*50)

# Stack example
stack = Stack()
for i in [1, 2, 3, 4, 5]:
    stack.push(i)
print(f"Stack: {stack}")
print(f"Pop: {stack.pop()}")
print(f"Peek: {stack.peek()}")

# Queue example
queue = Queue()
for i in [1, 2, 3, 4, 5]:
    queue.enqueue(i)
print(f"Queue: {queue}")
print(f"Dequeue: {queue.dequeue()}")
print(f"Front: {queue.front()}")

# Linked List example
ll = LinkedList()
for i in [1, 2, 3, 4, 5]:
    ll.append(i)
print(f"Linked List: {ll}")
print(f"Find 3: {ll.find(3)}")
ll.delete(3)
print(f"After deleting 3: {ll}")

# BST example
bst = BinarySearchTree()
values = [5, 3, 7, 2, 4, 6, 8]
for val in values:
    bst.insert(val)
print(f"BST inorder traversal: {bst.inorder_traversal()}")
print(f"Search 4: {bst.search(4)}")
print(f"Search 9: {bst.search(9)}")

print("\nData structures implementation completed!")
```
