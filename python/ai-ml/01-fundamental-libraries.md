# Fundamental Python Libraries for AI-ML

## Table of Contents
1. [NumPy - Numerical Computing](#numpy---numerical-computing)
2. [Pandas - Data Manipulation](#pandas---data-manipulation)
3. [Matplotlib - Data Visualization](#matplotlib---data-visualization)
4. [Seaborn - Statistical Visualization](#seaborn---statistical-visualization)
5. [SciPy - Scientific Computing](#scipy---scientific-computing)
6. [Jupyter Notebooks](#jupyter-notebooks)

## NumPy - Numerical Computing

NumPy (Numerical Python) is the foundation library for scientific computing in Python. It provides powerful N-dimensional array objects and mathematical functions.

### 1. Installation and Import
```python
# Installation
pip install numpy

# Import convention
import numpy as np
print(f"NumPy version: {np.__version__}")
```

### 2. Array Creation
```python
# Creating arrays from lists
arr1d = np.array([1, 2, 3, 4, 5])
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
arr3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

print(f"1D Array: {arr1d}")
print(f"2D Array:\n{arr2d}")
print(f"3D Array:\n{arr3d}")

# Array properties
print(f"Shape: {arr2d.shape}")
print(f"Dimensions: {arr2d.ndim}")
print(f"Size: {arr2d.size}")
print(f"Data type: {arr2d.dtype}")

# Creating arrays with built-in functions
zeros = np.zeros((3, 4))                    # Array of zeros
ones = np.ones((2, 3))                      # Array of ones
empty = np.empty((2, 2))                    # Uninitialized array
full = np.full((3, 3), 7)                   # Array filled with specific value
eye = np.eye(4)                             # Identity matrix
diag = np.diag([1, 2, 3, 4])              # Diagonal matrix

# Range arrays
arange = np.arange(0, 10, 2)               # [0, 2, 4, 6, 8]
linspace = np.linspace(0, 1, 5)            # [0, 0.25, 0.5, 0.75, 1]
logspace = np.logspace(0, 2, 3)            # [1, 10, 100]

# Random arrays
random_array = np.random.random((3, 3))     # Random values [0, 1)
random_int = np.random.randint(0, 10, (2, 3))  # Random integers
normal = np.random.normal(0, 1, (2, 3))     # Normal distribution

print(f"Random array:\n{random_array}")
print(f"Random integers:\n{random_int}")
```

### 3. Array Indexing and Slicing
```python
# 1D array indexing
arr = np.array([10, 20, 30, 40, 50])
print(f"Element at index 0: {arr[0]}")      # 10
print(f"Element at index -1: {arr[-1]}")    # 50
print(f"Slice [1:4]: {arr[1:4]}")          # [20, 30, 40]
print(f"Every 2nd element: {arr[::2]}")     # [10, 30, 50]

# 2D array indexing
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"Element at (1,2): {arr2d[1, 2]}")  # 6
print(f"Row 0: {arr2d[0, :]}")             # [1, 2, 3]
print(f"Column 1: {arr2d[:, 1]}")          # [2, 5, 8]
print(f"Subarray:\n{arr2d[0:2, 1:3]}")     # [[2, 3], [5, 6]]

# Boolean indexing
data = np.array([1, 2, 3, 4, 5, 6])
mask = data > 3
print(f"Boolean mask: {mask}")              # [False, False, False, True, True, True]
print(f"Filtered data: {data[mask]}")       # [4, 5, 6]
print(f"Data > 3: {data[data > 3]}")       # [4, 5, 6]

# Fancy indexing
arr = np.array([10, 20, 30, 40, 50])
indices = [0, 2, 4]
print(f"Elements at indices {indices}: {arr[indices]}")  # [10, 30, 50]

# Multi-dimensional fancy indexing
arr2d = np.random.randint(0, 10, (4, 4))
print(f"Original array:\n{arr2d}")
print(f"Elements at (0,1) and (2,3): {arr2d[[0, 2], [1, 3]]}")
```

### 4. Array Operations
```python
# Element-wise operations
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

# Arithmetic operations
print(f"Addition: {a + b}")                 # [6, 8, 10, 12]
print(f"Subtraction: {a - b}")              # [-4, -4, -4, -4]
print(f"Multiplication: {a * b}")           # [5, 12, 21, 32]
print(f"Division: {b / a}")                 # [5., 3., 2.33, 2.]
print(f"Power: {a ** 2}")                   # [1, 4, 9, 16]
print(f"Square root: {np.sqrt(a)}")         # [1., 1.41, 1.73, 2.]

# Comparison operations
print(f"a > 2: {a > 2}")                    # [False, False, True, True]
print(f"a == b: {a == b}")                  # [False, False, False, False]

# Matrix operations
matrix_a = np.array([[1, 2], [3, 4]])
matrix_b = np.array([[5, 6], [7, 8]])

# Matrix multiplication
print(f"Matrix multiplication:\n{np.dot(matrix_a, matrix_b)}")
print(f"Matrix multiplication (@ operator):\n{matrix_a @ matrix_b}")

# Element-wise multiplication
print(f"Element-wise multiplication:\n{matrix_a * matrix_b}")

# Universal functions (ufuncs)
data = np.array([-2, -1, 0, 1, 2])
print(f"Absolute values: {np.abs(data)}")
print(f"Exponential: {np.exp(data)}")
print(f"Logarithm: {np.log(np.abs(data) + 1)}")  # Adding 1 to avoid log(0)
print(f"Sine: {np.sin(data)}")
print(f"Cosine: {np.cos(data)}")
```

### 5. Statistical Operations
```python
# Sample data
data = np.random.normal(100, 15, 1000)  # Mean=100, std=15, 1000 samples

# Basic statistics
print(f"Mean: {np.mean(data):.2f}")
print(f"Median: {np.median(data):.2f}")
print(f"Standard deviation: {np.std(data):.2f}")
print(f"Variance: {np.var(data):.2f}")
print(f"Min: {np.min(data):.2f}")
print(f"Max: {np.max(data):.2f}")

# Percentiles
print(f"25th percentile: {np.percentile(data, 25):.2f}")
print(f"75th percentile: {np.percentile(data, 75):.2f}")

# 2D array statistics
matrix = np.random.randint(1, 10, (3, 4))
print(f"Matrix:\n{matrix}")
print(f"Sum of all elements: {np.sum(matrix)}")
print(f"Sum along axis 0 (columns): {np.sum(matrix, axis=0)}")
print(f"Sum along axis 1 (rows): {np.sum(matrix, axis=1)}")
print(f"Mean along axis 0: {np.mean(matrix, axis=0)}")

# Cumulative operations
arr = np.array([1, 2, 3, 4, 5])
print(f"Cumulative sum: {np.cumsum(arr)}")        # [1, 3, 6, 10, 15]
print(f"Cumulative product: {np.cumprod(arr)}")   # [1, 2, 6, 24, 120]
```

### 6. Array Manipulation
```python
# Reshaping arrays
arr = np.arange(12)
print(f"Original: {arr}")
print(f"Reshaped (3x4):\n{arr.reshape(3, 4)}")
print(f"Reshaped (2x6):\n{arr.reshape(2, 6)}")
print(f"Reshaped (auto dimension):\n{arr.reshape(-1, 3)}")  # -1 means "figure it out"

# Flattening arrays
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(f"Flattened: {matrix.flatten()}")
print(f"Ravel: {matrix.ravel()}")               # Similar to flatten but returns view if possible

# Transpose
print(f"Original matrix:\n{matrix}")
print(f"Transposed:\n{matrix.T}")
print(f"Transposed (function):\n{np.transpose(matrix)}")

# Concatenation
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

# Vertical stack (along axis 0)
vstack = np.vstack((a, b))
print(f"Vertical stack:\n{vstack}")

# Horizontal stack (along axis 1)
hstack = np.hstack((a, b))
print(f"Horizontal stack:\n{hstack}")

# General concatenation
concat_axis0 = np.concatenate((a, b), axis=0)
concat_axis1 = np.concatenate((a, b), axis=1)
print(f"Concatenate axis 0:\n{concat_axis0}")
print(f"Concatenate axis 1:\n{concat_axis1}")

# Splitting arrays
arr = np.arange(12)
split_equal = np.split(arr, 3)              # Split into 3 equal parts
print(f"Split into 3 parts: {split_equal}")

split_at = np.split(arr, [3, 7])            # Split at indices 3 and 7
print(f"Split at indices [3, 7]: {split_at}")

# 2D splitting
matrix = np.arange(16).reshape(4, 4)
hsplit = np.hsplit(matrix, 2)               # Horizontal split
vsplit = np.vsplit(matrix, 2)               # Vertical split
print(f"Horizontal split: {hsplit}")
print(f"Vertical split: {vsplit}")
```

### 7. Broadcasting
```python
# Broadcasting rules: NumPy can perform operations on arrays with different shapes
# Rule 1: If arrays have different dimensions, pad the smaller one with 1s on the left
# Rule 2: If shapes differ in any dimension, extend the dimension with size 1
# Rule 3: If shapes still don't match, raise an error

# Scalar and array
arr = np.array([1, 2, 3, 4])
result = arr + 10                           # Scalar broadcasts to array shape
print(f"Array + scalar: {result}")

# 1D array and 2D array
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
vector = np.array([10, 20, 30])
result = matrix + vector                    # Vector broadcasts across rows
print(f"Matrix + vector:\n{result}")

# Broadcasting with different shapes
a = np.arange(12).reshape(3, 4)
b = np.arange(4)
print(f"Shape of a: {a.shape}")            # (3, 4)
print(f"Shape of b: {b.shape}")            # (4,)
result = a + b                              # b broadcasts to (3, 4)
print(f"Broadcasted result:\n{result}")

# More complex broadcasting
a = np.arange(6).reshape(2, 3, 1)
b = np.arange(4).reshape(1, 1, 4)
print(f"Shape of a: {a.shape}")            # (2, 3, 1)
print(f"Shape of b: {b.shape}")            # (1, 1, 4)
result = a + b                              # Result shape: (2, 3, 4)
print(f"Result shape: {result.shape}")
```

### 8. Advanced Array Operations
```python
# Sorting
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
print(f"Original: {arr}")
print(f"Sorted: {np.sort(arr)}")
print(f"Sort indices: {np.argsort(arr)}")  # Indices that would sort the array

# 2D sorting
matrix = np.random.randint(1, 10, (3, 4))
print(f"Original matrix:\n{matrix}")
print(f"Sorted along axis 0:\n{np.sort(matrix, axis=0)}")
print(f"Sorted along axis 1:\n{np.sort(matrix, axis=1)}")

# Unique values
arr = np.array([1, 2, 2, 3, 3, 3, 4])
unique_vals, counts = np.unique(arr, return_counts=True)
print(f"Unique values: {unique_vals}")
print(f"Counts: {counts}")

# Set operations
a = np.array([1, 2, 3, 4, 5])
b = np.array([3, 4, 5, 6, 7])
print(f"Intersection: {np.intersect1d(a, b)}")
print(f"Union: {np.union1d(a, b)}")
print(f"Difference (a - b): {np.setdiff1d(a, b)}")
print(f"Symmetric difference: {np.setxor1d(a, b)}")

# Searching
arr = np.array([1, 3, 5, 7, 9, 11])
print(f"Where arr > 5: {np.where(arr > 5)}")
print(f"Elements > 5: {arr[np.where(arr > 5)]}")

# Conditional selection
print(f"Conditional: {np.where(arr > 5, arr, 0)}")  # Replace values <= 5 with 0

# argmax and argmin
print(f"Index of max: {np.argmax(arr)}")
print(f"Index of min: {np.argmin(arr)}")
```

## Pandas - Data Manipulation

Pandas is a powerful library for data manipulation and analysis, built on top of NumPy.

### 1. Installation and Import
```python
# Installation
pip install pandas

# Import convention
import pandas as pd
import numpy as np
print(f"Pandas version: {pd.__version__}")
```

### 2. Series - One-dimensional labeled array
```python
# Creating Series
s1 = pd.Series([1, 2, 3, 4, 5])
s2 = pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'])
s3 = pd.Series({'a': 1, 'b': 2, 'c': 3})

print(f"Series with default index:\n{s1}\n")
print(f"Series with custom index:\n{s2}\n")
print(f"Series from dictionary:\n{s3}\n")

# Series properties
print(f"Values: {s2.values}")
print(f"Index: {s2.index}")
print(f"Data type: {s2.dtype}")
print(f"Shape: {s2.shape}")
print(f"Size: {s2.size}")

# Accessing elements
print(f"Element 'b': {s2['b']}")
print(f"Elements ['a', 'c']: {s2[['a', 'c']]}")
print(f"First 3 elements: {s2[:3]}")

# Boolean indexing
print(f"Elements > 2: {s2[s2 > 2]}")

# Series operations
s4 = pd.Series([10, 20, 30, 40, 50], index=['a', 'b', 'c', 'd', 'e'])
print(f"Addition:\n{s2 + s4}")
print(f"Multiplication:\n{s2 * 2}")
print(f"Mathematical functions:\n{np.sqrt(s2)}")
```

### 3. DataFrame - Two-dimensional labeled data structure
```python
# Creating DataFrames
# From dictionary
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Age': [25, 30, 35, 28],
    'City': ['New York', 'London', 'Tokyo', 'Paris'],
    'Salary': [50000, 60000, 70000, 55000]
}
df = pd.DataFrame(data)
print(f"DataFrame from dictionary:\n{df}\n")

# From list of dictionaries
data_list = [
    {'Name': 'Alice', 'Age': 25, 'City': 'New York'},
    {'Name': 'Bob', 'Age': 30, 'City': 'London'},
    {'Name': 'Charlie', 'Age': 35, 'City': 'Tokyo'}
]
df2 = pd.DataFrame(data_list)
print(f"DataFrame from list of dictionaries:\n{df2}\n")

# From NumPy array
array_data = np.random.randn(4, 3)
df3 = pd.DataFrame(array_data, columns=['A', 'B', 'C'])
print(f"DataFrame from NumPy array:\n{df3}\n")

# DataFrame properties
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns}")
print(f"Index: {df.index}")
print(f"Data types:\n{df.dtypes}")
print(f"Info:")
df.info()
```

### 4. Data Selection and Indexing
```python
# Column selection
print(f"Single column (Series):\n{df['Name']}\n")
print(f"Single column (DataFrame):\n{df[['Name']]}\n")
print(f"Multiple columns:\n{df[['Name', 'Age']]}\n")

# Row selection
print(f"First row:\n{df.iloc[0]}\n")           # Integer-location based
print(f"Row by index:\n{df.loc[0]}\n")         # Label-based
print(f"First 2 rows:\n{df.head(2)}\n")
print(f"Last 2 rows:\n{df.tail(2)}\n")

# Specific element
print(f"Element at (0, 'Name'): {df.loc[0, 'Name']}")
print(f"Element at (1, 1): {df.iloc[1, 1]}")

# Slicing
print(f"Rows 1-2, Columns 'Name' to 'City':\n{df.loc[1:2, 'Name':'City']}\n")
print(f"Rows 0-1, Columns 0-2:\n{df.iloc[0:2, 0:3]}\n")

# Boolean indexing
high_salary = df[df['Salary'] > 55000]
print(f"High salary employees:\n{high_salary}\n")

# Multiple conditions
young_high_earners = df[(df['Age'] < 32) & (df['Salary'] > 55000)]
print(f"Young high earners:\n{young_high_earners}\n")

# Query method
query_result = df.query('Age > 28 and Salary < 65000')
print(f"Query result:\n{query_result}\n")
```

### 5. Data Manipulation
```python
# Adding new columns
df['Bonus'] = df['Salary'] * 0.1
df['Total_Compensation'] = df['Salary'] + df['Bonus']
print(f"DataFrame with new columns:\n{df}\n")

# Modifying existing columns
df['Age'] = df['Age'] + 1  # Everyone gets a year older
print(f"After age increment:\n{df}\n")

# Dropping columns
df_dropped = df.drop(['Bonus'], axis=1)  # axis=1 for columns
print(f"After dropping Bonus column:\n{df_dropped}\n")

# Dropping rows
df_dropped_rows = df.drop([0, 2])  # Drop rows with index 0 and 2
print(f"After dropping rows:\n{df_dropped_rows}\n")

# Renaming columns
df_renamed = df.rename(columns={'Name': 'Employee_Name', 'Age': 'Employee_Age'})
print(f"After renaming columns:\n{df_renamed}\n")

# Setting index
df_indexed = df.set_index('Name')
print(f"With Name as index:\n{df_indexed}\n")

# Resetting index
df_reset = df_indexed.reset_index()
print(f"After resetting index:\n{df_reset}\n")
```

### 6. Data Cleaning
```python
# Creating DataFrame with missing values
data_with_na = {
    'A': [1, 2, np.nan, 4, 5],
    'B': [np.nan, 2, 3, 4, np.nan],
    'C': [1, 2, 3, np.nan, 5],
    'D': ['a', 'b', 'c', 'd', 'e']
}
df_na = pd.DataFrame(data_with_na)
print(f"DataFrame with missing values:\n{df_na}\n")

# Detecting missing values
print(f"Is null:\n{df_na.isnull()}\n")
print(f"Not null:\n{df_na.notnull()}\n")
print(f"Missing values per column:\n{df_na.isnull().sum()}\n")

# Dropping missing values
print(f"Drop any row with NaN:\n{df_na.dropna()}\n")
print(f"Drop columns with any NaN:\n{df_na.dropna(axis=1)}\n")
print(f"Drop rows where all values are NaN:\n{df_na.dropna(how='all')}\n")

# Filling missing values
print(f"Fill with 0:\n{df_na.fillna(0)}\n")
print(f"Fill with column mean:\n{df_na.fillna(df_na.mean())}\n")
print(f"Forward fill:\n{df_na.fillna(method='ffill')}\n")
print(f"Backward fill:\n{df_na.fillna(method='bfill')}\n")

# Interpolation
print(f"Linear interpolation:\n{df_na.interpolate()}\n")

# Duplicates
data_with_dupes = pd.DataFrame({
    'A': [1, 2, 2, 3, 3],
    'B': [1, 2, 2, 3, 4]
})
print(f"DataFrame with duplicates:\n{data_with_dupes}\n")
print(f"Duplicate rows:\n{data_with_dupes.duplicated()}\n")
print(f"Drop duplicates:\n{data_with_dupes.drop_duplicates()}\n")
```

## 7. Advanced NumPy Operations

### Memory Layout and Performance

```python
import numpy as np
import time

# Memory layout - C vs Fortran order
arr_c = np.array([[1, 2, 3], [4, 5, 6]], order='C')  # Row-major (C-style)
arr_f = np.array([[1, 2, 3], [4, 5, 6]], order='F')  # Column-major (Fortran-style)

print(f"C-order flags: {arr_c.flags}")
print(f"F-order flags: {arr_f.flags}")

# Performance implications
large_arr = np.random.random((1000, 1000))

# Row-wise access (efficient for C-order)
start = time.time()
for i in range(1000):
    row_sum = np.sum(large_arr[i, :])
row_time = time.time() - start

# Column-wise access (less efficient for C-order)
start = time.time()
for j in range(1000):
    col_sum = np.sum(large_arr[:, j])
col_time = time.time() - start

print(f"Row-wise access time: {row_time:.4f}s")
print(f"Column-wise access time: {col_time:.4f}s")
```

### Broadcasting and Vectorization

```python
# Advanced broadcasting
a = np.array([1, 2, 3]).reshape(3, 1)  # (3, 1)
b = np.array([10, 20, 30, 40])         # (4,)
result = a + b  # Broadcasts to (3, 4)
print(f"Broadcasted result shape: {result.shape}")
print(f"Result:\n{result}")

# Vectorization vs loops
def slow_computation(arr):
    """Slow Python loop"""
    result = np.zeros_like(arr)
    for i in range(len(arr)):
        result[i] = arr[i] ** 2 + 2 * arr[i] + 1
    return result

def fast_computation(arr):
    """Vectorized NumPy"""
    return arr ** 2 + 2 * arr + 1

# Performance comparison
test_arr = np.random.random(1000000)

start = time.time()
slow_result = slow_computation(test_arr)
slow_time = time.time() - start

start = time.time()
fast_result = fast_computation(test_arr)
fast_time = time.time() - start

print(f"Slow computation time: {slow_time:.4f}s")
print(f"Fast computation time: {fast_time:.4f}s")
print(f"Speedup: {slow_time/fast_time:.1f}x")
```

### Custom NumPy Data Types

```python
# Structured arrays
dt = np.dtype([('name', 'U10'), ('age', 'i4'), ('salary', 'f8')])
employees = np.array([
    ('Alice', 25, 50000.0),
    ('Bob', 30, 60000.0),
    ('Charlie', 35, 70000.0)
], dtype=dt)

print(f"Structured array:\n{employees}")
print(f"Names: {employees['name']}")
print(f"Average age: {employees['age'].mean()}")

# Record arrays (more convenient)
rec_arr = np.rec.fromarrays([
    ['Alice', 'Bob', 'Charlie'],
    [25, 30, 35],
    [50000.0, 60000.0, 70000.0]
], names='name,age,salary')

print(f"Record array names: {rec_arr.name}")
print(f"High earners: {rec_arr[rec_arr.salary > 55000]}")
```

## 8. Advanced Pandas Techniques

### Multi-Index and Hierarchical Data

```python
import pandas as pd
import numpy as np

# Create MultiIndex DataFrame
arrays = [
    ['A', 'A', 'B', 'B'],
    ['one', 'two', 'one', 'two']
]
tuples = list(zip(*arrays))
index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])

df_multi = pd.DataFrame(np.random.randn(4, 2), index=index, columns=['X', 'Y'])
print(f"MultiIndex DataFrame:\n{df_multi}\n")

# Indexing MultiIndex
print(f"Level A data:\n{df_multi.loc['A']}\n")
print(f"Specific cell: {df_multi.loc[('A', 'one'), 'X']}\n")

# Stack and unstack
stacked = df_multi.stack()
print(f"Stacked:\n{stacked}\n")
unstacked = stacked.unstack()
print(f"Unstacked:\n{unstacked}\n")
```

### Advanced Groupby Operations

```python
# Sample data
df = pd.DataFrame({
    'Department': ['Sales', 'Sales', 'Marketing', 'Marketing', 'IT', 'IT'],
    'Employee': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank'],
    'Salary': [50000, 60000, 55000, 65000, 70000, 75000],
    'Bonus': [5000, 6000, 5500, 6500, 7000, 7500],
    'Years': [2, 5, 3, 7, 4, 6]
})

# Multiple aggregations
agg_funcs = {
    'Salary': ['mean', 'min', 'max'],
    'Bonus': 'sum',
    'Years': 'mean'
}
result = df.groupby('Department').agg(agg_funcs)
print(f"Multiple aggregations:\n{result}\n")

# Custom aggregation functions
def salary_range(series):
    return series.max() - series.min()

custom_agg = df.groupby('Department').agg({
    'Salary': [salary_range, 'count'],
    'Bonus': lambda x: x.sum() / len(x)  # Average bonus
})
print(f"Custom aggregations:\n{custom_agg}\n")

# Transform vs apply
df['Salary_zscore'] = df.groupby('Department')['Salary'].transform(
    lambda x: (x - x.mean()) / x.std()
)
print(f"Z-scores by department:\n{df[['Department', 'Employee', 'Salary', 'Salary_zscore']].round(2)}\n")
```

### Time Series Analysis

```python
# Create time series data
dates = pd.date_range('2023-01-01', periods=100, freq='D')
ts = pd.Series(np.random.randn(100).cumsum(), index=dates)

print(f"Time series head:\n{ts.head()}\n")

# Resampling
monthly = ts.resample('M').mean()
print(f"Monthly averages:\n{monthly}\n")

# Rolling statistics
rolling_mean = ts.rolling(window=7).mean()
rolling_std = ts.rolling(window=7).std()

print(f"7-day rolling mean (last 5):\n{rolling_mean.tail()}\n")

# Time zone handling
ts_utc = ts.tz_localize('UTC')
ts_est = ts_utc.tz_convert('US/Eastern')
print(f"UTC vs EST comparison:\n{pd.concat([ts_utc.head(3), ts_est.head(3)], axis=1, keys=['UTC', 'EST'])}\n")
```

## 9. Performance Optimization

### Efficient Data Operations

```python
import pandas as pd
import numpy as np
import time

# Create large dataset
n = 1000000
df_large = pd.DataFrame({
    'A': np.random.randn(n),
    'B': np.random.randint(0, 100, n),
    'C': np.random.choice(['X', 'Y', 'Z'], n)
})

# Method 1: Naive approach
start = time.time()
result1 = df_large[df_large['A'] > 0]['B'].sum()
naive_time = time.time() - start

# Method 2: Using query (faster for complex conditions)
start = time.time()
result2 = df_large.query('A > 0')['B'].sum()
query_time = time.time() - start

# Method 3: Using boolean indexing with loc
start = time.time()
mask = df_large['A'] > 0
result3 = df_large.loc[mask, 'B'].sum()
loc_time = time.time() - start

print(f"Naive approach: {naive_time:.4f}s")
print(f"Query approach: {query_time:.4f}s")
print(f"Loc approach: {loc_time:.4f}s")

# Memory usage optimization
print(f"\nOriginal memory usage: {df_large.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Optimize data types
df_optimized = df_large.copy()
df_optimized['B'] = df_optimized['B'].astype('int8')  # Smaller integer type
df_optimized['C'] = df_optimized['C'].astype('category')  # Categorical for repeated strings

print(f"Optimized memory usage: {df_optimized.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
```

### Chunking for Large Datasets

```python
# Simulate reading large CSV in chunks
def process_large_dataset(chunk_size=10000):
    """Process data in chunks to manage memory"""
    # Simulate large dataset
    total_rows = 100000
    results = []
    
    for start in range(0, total_rows, chunk_size):
        end = min(start + chunk_size, total_rows)
        
        # Simulate reading chunk
        chunk = pd.DataFrame({
            'value': np.random.randn(end - start),
            'category': np.random.choice(['A', 'B', 'C'], end - start)
        })
        
        # Process chunk
        chunk_result = chunk.groupby('category')['value'].mean()
        results.append(chunk_result)
    
    # Combine results
    final_result = pd.concat(results, axis=1).mean(axis=1)
    return final_result

chunk_result = process_large_dataset()
print(f"Chunked processing result:\n{chunk_result}")
```

## 10. Integration Patterns

### NumPy-Pandas Integration

```python
# Converting between NumPy and Pandas
df = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': [5, 6, 7, 8]
})

# DataFrame to NumPy
arr = df.values  # or df.to_numpy()
print(f"DataFrame as NumPy array:\n{arr}")

# NumPy operations on DataFrame
df['C'] = np.sqrt(df['A'] ** 2 + df['B'] ** 2)
print(f"DataFrame with NumPy operation:\n{df}")

# Using NumPy functions with Pandas
df['A_normalized'] = (df['A'] - np.mean(df['A'])) / np.std(df['A'])
print(f"Normalized column:\n{df['A_normalized']}")
```

### Matplotlib-Pandas Integration

```python
import matplotlib.pyplot as plt

# Direct plotting from Pandas
df = pd.DataFrame({
    'x': np.linspace(0, 10, 100),
    'y1': np.sin(np.linspace(0, 10, 100)),
    'y2': np.cos(np.linspace(0, 10, 100))
})

# Plot directly from DataFrame
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

df.plot(x='x', y=['y1', 'y2'], ax=ax1, title='Pandas Plot')
ax2.plot(df['x'], df['y1'], label='sin(x)')
ax2.plot(df['x'], df['y2'], label='cos(x)')
ax2.set_title('Matplotlib Plot')
ax2.legend()

plt.tight_layout()
plt.show()

# Time series plotting
ts_data = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=365),
    'value': np.random.randn(365).cumsum()
})
ts_data.set_index('date').plot(figsize=(10, 6), title='Time Series Data')
plt.show()
```

This completes the comprehensive AI-ML fundamentals documentation with advanced concepts and integration patterns.
