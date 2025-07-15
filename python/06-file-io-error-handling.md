# File I/O and Error Handling in Python

## Table of Contents
1. [File Operations](#file-operations)
2. [Reading Files](#reading-files)
3. [Writing Files](#writing-files)
4. [Working with CSV Files](#working-with-csv-files)
5. [Working with JSON Files](#working-with-json-files)
6. [Path and Directory Operations](#path-and-directory-operations)
7. [Exception Handling (Advanced)](#exception-handling-advanced)
8. [Logging](#logging)

## File Operations

### 1. Basic File Operations
```python
# Opening files - basic syntax
file = open("example.txt", "r")  # Read mode
content = file.read()
file.close()  # Always close files

# Better approach - using with statement (context manager)
with open("example.txt", "r") as file:
    content = file.read()
# File is automatically closed

# File modes
# "r"  - Read (default)
# "w"  - Write (overwrites existing file)
# "a"  - Append
# "x"  - Exclusive creation (fails if file exists)
# "b"  - Binary mode (e.g., "rb", "wb")
# "t"  - Text mode (default)
# "+"  - Read and write

# Examples of different modes
with open("data.txt", "w") as file:    # Write mode
    file.write("Hello, World!")

with open("data.txt", "a") as file:    # Append mode
    file.write("\nSecond line")

with open("data.txt", "r+") as file:   # Read and write
    content = file.read()
    file.write("\nThird line")
```

### 2. File Object Methods
```python
# File object attributes and methods
with open("example.txt", "r") as file:
    print(f"File name: {file.name}")
    print(f"File mode: {file.mode}")
    print(f"File closed: {file.closed}")
    print(f"File readable: {file.readable()}")
    print(f"File writable: {file.writable()}")
    print(f"File seekable: {file.seekable()}")

# File position operations
with open("example.txt", "r") as file:
    print(f"Current position: {file.tell()}")  # Get current position
    content = file.read(10)  # Read 10 characters
    print(f"Position after reading: {file.tell()}")
    file.seek(0)  # Go back to beginning
    print(f"Position after seek: {file.tell()}")
```

## Reading Files

### 1. Different Ways to Read Files
```python
# Method 1: Read entire file at once
with open("example.txt", "r") as file:
    content = file.read()
    print(content)

# Method 2: Read specific number of characters
with open("example.txt", "r") as file:
    chunk = file.read(50)  # Read first 50 characters
    print(chunk)

# Method 3: Read line by line
with open("example.txt", "r") as file:
    line = file.readline()  # Read one line
    while line:
        print(line.strip())  # strip() removes newline characters
        line = file.readline()

# Method 4: Read all lines into a list
with open("example.txt", "r") as file:
    lines = file.readlines()
    for line in lines:
        print(line.strip())

# Method 5: Iterate over file object (most Pythonic)
with open("example.txt", "r") as file:
    for line in file:
        print(line.strip())

# Method 6: Read in chunks (memory efficient for large files)
def read_in_chunks(file_path, chunk_size=1024):
    with open(file_path, "r") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            yield chunk

for chunk in read_in_chunks("large_file.txt"):
    process_chunk(chunk)
```

### 2. Reading Different File Types
```python
# Reading text files with encoding
with open("unicode_file.txt", "r", encoding="utf-8") as file:
    content = file.read()

# Reading binary files
with open("image.jpg", "rb") as file:
    binary_data = file.read()
    print(f"File size: {len(binary_data)} bytes")

# Reading files with error handling for encoding
try:
    with open("problematic_file.txt", "r", encoding="utf-8") as file:
        content = file.read()
except UnicodeDecodeError:
    # Try different encoding
    with open("problematic_file.txt", "r", encoding="latin-1") as file:
        content = file.read()

# Reading files line by line with line numbers
with open("example.txt", "r") as file:
    for line_num, line in enumerate(file, start=1):
        print(f"Line {line_num}: {line.strip()}")
```

### 3. Advanced Reading Techniques
```python
# Reading specific lines
def read_specific_lines(file_path, line_numbers):
    """Read specific line numbers from a file."""
    with open(file_path, "r") as file:
        lines = {}
        for current_line_num, line in enumerate(file, start=1):
            if current_line_num in line_numbers:
                lines[current_line_num] = line.strip()
            if len(lines) == len(line_numbers):
                break
    return lines

# Usage
specific_lines = read_specific_lines("data.txt", [1, 5, 10])

# Reading file backwards
def read_file_backwards(file_path):
    """Read file lines in reverse order."""
    with open(file_path, "r") as file:
        lines = file.readlines()
    for line in reversed(lines):
        yield line.strip()

for line in read_file_backwards("example.txt"):
    print(line)

# Reading large files efficiently
def read_large_file(file_path):
    """Memory-efficient reading of large files."""
    with open(file_path, "r") as file:
        for line in file:
            yield line.strip()

# Process one line at a time without loading entire file
for line in read_large_file("huge_file.txt"):
    if "search_term" in line:
        print(f"Found: {line}")
```

## Writing Files

### 1. Basic Writing Operations
```python
# Writing to a new file
with open("output.txt", "w") as file:
    file.write("Hello, World!\n")
    file.write("This is a second line.\n")

# Writing multiple lines at once
lines = ["Line 1\n", "Line 2\n", "Line 3\n"]
with open("output.txt", "w") as file:
    file.writelines(lines)

# Appending to existing file
with open("output.txt", "a") as file:
    file.write("This line is appended.\n")

# Writing with print function
with open("output.txt", "w") as file:
    print("Hello, World!", file=file)
    print("Second line", file=file)
    print("Number:", 42, file=file)
```

### 2. Advanced Writing Techniques
```python
# Writing formatted data
data = [
    {"name": "Alice", "age": 25, "city": "New York"},
    {"name": "Bob", "age": 30, "city": "San Francisco"},
    {"name": "Charlie", "age": 35, "city": "Chicago"}
]

with open("formatted_output.txt", "w") as file:
    file.write("Name\t\tAge\tCity\n")
    file.write("-" * 30 + "\n")
    for person in data:
        file.write(f"{person['name']:<10}\t{person['age']}\t{person['city']}\n")

# Writing with different encodings
text = "Hello, 世界! 🌍"
with open("unicode_output.txt", "w", encoding="utf-8") as file:
    file.write(text)

# Atomic writes (write to temporary file, then rename)
import tempfile
import os

def atomic_write(file_path, content):
    """Write to file atomically to prevent corruption."""
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, 
                                           dir=os.path.dirname(file_path))
    try:
        temp_file.write(content)
        temp_file.flush()
        os.fsync(temp_file.fileno())  # Force write to disk
        temp_file.close()
        os.replace(temp_file.name, file_path)  # Atomic rename
    except:
        os.unlink(temp_file.name)  # Clean up on error
        raise

atomic_write("important_data.txt", "Critical information")
```

### 3. Binary File Operations
```python
# Writing binary data
binary_data = b'\x89PNG\r\n\x1a\n'  # PNG file header
with open("sample.png", "wb") as file:
    file.write(binary_data)

# Copying binary files
def copy_binary_file(source, destination):
    """Copy a binary file in chunks."""
    with open(source, "rb") as src, open(destination, "wb") as dst:
        while True:
            chunk = src.read(4096)  # Read 4KB chunks
            if not chunk:
                break
            dst.write(chunk)

copy_binary_file("source_image.jpg", "copied_image.jpg")

# Reading and modifying binary files
with open("data.bin", "r+b") as file:
    file.seek(10)  # Go to position 10
    original_byte = file.read(1)
    file.seek(10)  # Go back to position 10
    file.write(b'\xFF')  # Write new byte
```

## Working with CSV Files

### 1. Basic CSV Operations
```python
import csv

# Writing CSV files
data = [
    ["Name", "Age", "City"],
    ["Alice", 25, "New York"],
    ["Bob", 30, "San Francisco"],
    ["Charlie", 35, "Chicago"]
]

with open("people.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(data)

# Reading CSV files
with open("people.csv", "r") as file:
    reader = csv.reader(file)
    for row in reader:
        print(row)

# Using DictWriter and DictReader
people = [
    {"name": "Alice", "age": 25, "city": "New York"},
    {"name": "Bob", "age": 30, "city": "San Francisco"},
    {"name": "Charlie", "age": 35, "city": "Chicago"}
]

# Writing with DictWriter
with open("people_dict.csv", "w", newline="") as file:
    fieldnames = ["name", "age", "city"]
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(people)

# Reading with DictReader
with open("people_dict.csv", "r") as file:
    reader = csv.DictReader(file)
    for row in reader:
        print(f"Name: {row['name']}, Age: {row['age']}, City: {row['city']}")
```

### 2. Advanced CSV Operations
```python
# Custom delimiters and quoting
with open("custom.csv", "w", newline="") as file:
    writer = csv.writer(file, delimiter=";", quotechar='"', 
                       quoting=csv.QUOTE_MINIMAL)
    writer.writerow(["Name", "Description", "Price"])
    writer.writerow(["Product A", "A great product; very useful", 29.99])

# Reading CSV with custom dialect
csv.register_dialect('custom', delimiter=';', quotechar='"')

with open("custom.csv", "r") as file:
    reader = csv.reader(file, dialect='custom')
    for row in reader:
        print(row)

# Handling CSV with different encodings
with open("international.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Name", "Country", "Language"])
    writer.writerow(["José", "España", "Español"])
    writer.writerow(["François", "France", "Français"])

# Processing large CSV files efficiently
def process_large_csv(file_path):
    """Process large CSV files without loading all into memory."""
    with open(file_path, "r") as file:
        reader = csv.DictReader(file)
        for row_num, row in enumerate(reader, start=1):
            # Process one row at a time
            if row_num % 10000 == 0:
                print(f"Processed {row_num} rows")
            
            # Example processing
            if float(row['price']) > 100:
                print(f"Expensive item: {row['name']}")

# Filtering and transforming CSV data
def filter_csv(input_file, output_file, condition_func):
    """Filter CSV rows based on a condition function."""
    with open(input_file, "r") as infile, open(output_file, "w", newline="") as outfile:
        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
        writer.writeheader()
        
        for row in reader:
            if condition_func(row):
                writer.writerow(row)

# Usage
filter_csv("people.csv", "adults.csv", lambda row: int(row['age']) >= 18)
```

## Working with JSON Files

### 1. Basic JSON Operations
```python
import json

# Writing JSON files
data = {
    "users": [
        {"id": 1, "name": "Alice", "email": "alice@example.com", "active": True},
        {"id": 2, "name": "Bob", "email": "bob@example.com", "active": False},
        {"id": 3, "name": "Charlie", "email": "charlie@example.com", "active": True}
    ],
    "total_users": 3,
    "last_updated": "2024-01-15"
}

# Write JSON to file
with open("users.json", "w") as file:
    json.dump(data, file, indent=2)

# Read JSON from file
with open("users.json", "r") as file:
    loaded_data = json.load(file)
    print(loaded_data["total_users"])

# Working with JSON strings
json_string = json.dumps(data, indent=2)
print(json_string)

parsed_data = json.loads(json_string)
print(parsed_data["users"][0]["name"])
```

### 2. Advanced JSON Operations
```python
from datetime import datetime
import json

# Custom JSON encoder for special types
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

# Data with datetime
data_with_date = {
    "event": "Meeting",
    "timestamp": datetime.now(),
    "participants": ["Alice", "Bob"]
}

# Serialize with custom encoder
with open("events.json", "w") as file:
    json.dump(data_with_date, file, cls=DateTimeEncoder, indent=2)

# Custom JSON decoder
def datetime_hook(json_dict):
    for key, value in json_dict.items():
        if key == "timestamp":
            try:
                json_dict[key] = datetime.fromisoformat(value)
            except (ValueError, TypeError):
                pass
    return json_dict

# Deserialize with custom decoder
with open("events.json", "r") as file:
    loaded_data = json.load(file, object_hook=datetime_hook)
    print(type(loaded_data["timestamp"]))  # <class 'datetime.datetime'>

# Handling JSON errors
def safe_json_load(file_path):
    """Safely load JSON with error handling."""
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"File {file_path} not found")
        return None
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in {file_path}: {e}")
        return None

# Pretty printing JSON
def pretty_print_json(data):
    """Pretty print JSON data."""
    print(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False))

# Merging JSON files
def merge_json_files(file1, file2, output_file):
    """Merge two JSON files."""
    with open(file1, "r") as f1, open(file2, "r") as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)
    
    # Merge dictionaries
    if isinstance(data1, dict) and isinstance(data2, dict):
        merged = {**data1, **data2}
    else:
        merged = [data1, data2]
    
    with open(output_file, "w") as outfile:
        json.dump(merged, outfile, indent=2)
```

## Path and Directory Operations

### 1. Using pathlib (Modern Approach)
```python
from pathlib import Path
import os

# Creating Path objects
current_dir = Path.cwd()  # Current working directory
home_dir = Path.home()    # User's home directory
file_path = Path("data") / "files" / "example.txt"  # Cross-platform path

print(f"Current directory: {current_dir}")
print(f"Home directory: {home_dir}")
print(f"File path: {file_path}")

# Path operations
file_path = Path("documents/data/example.txt")
print(f"Parent directory: {file_path.parent}")
print(f"File name: {file_path.name}")
print(f"File stem: {file_path.stem}")
print(f"File suffix: {file_path.suffix}")
print(f"All parts: {file_path.parts}")

# Checking path properties
print(f"Exists: {file_path.exists()}")
print(f"Is file: {file_path.is_file()}")
print(f"Is directory: {file_path.is_dir()}")
print(f"Is absolute: {file_path.is_absolute()}")

# Creating directories
new_dir = Path("new_folder")
new_dir.mkdir(exist_ok=True)  # Create if doesn't exist

nested_dir = Path("parent/child/grandchild")
nested_dir.mkdir(parents=True, exist_ok=True)  # Create parent directories
```

### 2. Directory Operations
```python
from pathlib import Path
import shutil
import os

# Listing directory contents
data_dir = Path("data")
if data_dir.exists():
    # List all files and directories
    for item in data_dir.iterdir():
        print(f"{'DIR' if item.is_dir() else 'FILE'}: {item.name}")
    
    # List only Python files
    python_files = list(data_dir.glob("*.py"))
    print(f"Python files: {python_files}")
    
    # Recursive search
    all_python_files = list(data_dir.rglob("*.py"))
    print(f"All Python files: {all_python_files}")

# Working with os module
for root, dirs, files in os.walk("data"):
    for file in files:
        file_path = os.path.join(root, file)
        print(f"File: {file_path}")

# File operations
source_file = Path("source.txt")
destination_file = Path("destination.txt")

# Copy file
if source_file.exists():
    shutil.copy2(source_file, destination_file)  # Copy with metadata

# Move/rename file
old_name = Path("old_name.txt")
new_name = Path("new_name.txt")
if old_name.exists():
    old_name.rename(new_name)

# Delete file
temp_file = Path("temp.txt")
if temp_file.exists():
    temp_file.unlink()  # Delete file

# Delete directory
temp_dir = Path("temp_directory")
if temp_dir.exists():
    shutil.rmtree(temp_dir)  # Delete directory and contents
```

### 3. File System Information
```python
from pathlib import Path
import os
import stat
from datetime import datetime

file_path = Path("example.txt")

if file_path.exists():
    # File statistics
    file_stat = file_path.stat()
    
    print(f"File size: {file_stat.st_size} bytes")
    print(f"Created: {datetime.fromtimestamp(file_stat.st_ctime)}")
    print(f"Modified: {datetime.fromtimestamp(file_stat.st_mtime)}")
    print(f"Accessed: {datetime.fromtimestamp(file_stat.st_atime)}")
    
    # File permissions
    permissions = file_stat.st_mode
    print(f"Is readable: {bool(permissions & stat.S_IREAD)}")
    print(f"Is writable: {bool(permissions & stat.S_IWRITE)}")
    print(f"Is executable: {bool(permissions & stat.S_IEXEC)}")

# Disk usage
def get_directory_size(path):
    """Calculate total size of directory."""
    total_size = 0
    for item in Path(path).rglob('*'):
        if item.is_file():
            total_size += item.stat().st_size
    return total_size

# Get available disk space
def get_disk_usage(path):
    """Get disk usage statistics."""
    statvfs = os.statvfs(path)
    
    total_bytes = statvfs.f_frsize * statvfs.f_blocks
    free_bytes = statvfs.f_frsize * statvfs.f_bavail
    used_bytes = total_bytes - free_bytes
    
    return {
        'total': total_bytes,
        'used': used_bytes,
        'free': free_bytes
    }
```

## Exception Handling (Advanced)

### 1. Creating Custom Exceptions
```python
# Base custom exception
class ApplicationError(Exception):
    """Base exception for application-specific errors."""
    pass

class ValidationError(ApplicationError):
    """Raised when data validation fails."""
    
    def __init__(self, message, field=None, value=None):
        super().__init__(message)
        self.field = field
        self.value = value
    
    def __str__(self):
        if self.field:
            return f"Validation error in field '{self.field}': {super().__str__()}"
        return super().__str__()

class DatabaseError(ApplicationError):
    """Raised when database operations fail."""
    
    def __init__(self, message, error_code=None):
        super().__init__(message)
        self.error_code = error_code

class FileProcessingError(ApplicationError):
    """Raised when file processing fails."""
    pass

# Using custom exceptions
def validate_email(email):
    """Validate email address."""
    if "@" not in email:
        raise ValidationError("Email must contain @ symbol", field="email", value=email)
    if "." not in email.split("@")[1]:
        raise ValidationError("Email domain must contain a dot", field="email", value=email)

def process_user_data(user_data):
    """Process user data with validation."""
    try:
        validate_email(user_data["email"])
        # Process user...
        return "User processed successfully"
    except ValidationError as e:
        print(f"Validation failed: {e}")
        print(f"Field: {e.field}, Value: {e.value}")
        raise  # Re-raise for higher-level handling
```

### 2. Exception Chaining
```python
def read_config_file(file_path):
    """Read configuration file with exception chaining."""
    try:
        with open(file_path, "r") as file:
            content = file.read()
    except FileNotFoundError as e:
        raise FileProcessingError(f"Configuration file not found: {file_path}") from e
    except PermissionError as e:
        raise FileProcessingError(f"Permission denied reading: {file_path}") from e
    
    try:
        import json
        config = json.loads(content)
        return config
    except json.JSONDecodeError as e:
        raise FileProcessingError(f"Invalid JSON in config file: {file_path}") from e

# Using the function
try:
    config = read_config_file("config.json")
except FileProcessingError as e:
    print(f"Error: {e}")
    print(f"Original cause: {e.__cause__}")
```

### 3. Context Managers for Exception Handling
```python
from contextlib import contextmanager
import sys

@contextmanager
def suppress_errors(*exceptions):
    """Context manager to suppress specific exceptions."""
    try:
        yield
    except exceptions as e:
        print(f"Suppressed error: {e}")

# Usage
with suppress_errors(FileNotFoundError, PermissionError):
    with open("nonexistent_file.txt", "r") as file:
        content = file.read()

@contextmanager
def error_handling_context(operation_name):
    """Context manager for standardized error handling."""
    print(f"Starting {operation_name}...")
    try:
        yield
        print(f"Completed {operation_name} successfully")
    except Exception as e:
        print(f"Error in {operation_name}: {e}")
        # Log error, send notification, etc.
        raise
    finally:
        print(f"Cleanup for {operation_name}")

# Usage
with error_handling_context("file processing"):
    # File processing code here
    with open("data.txt", "r") as file:
        process_file(file)
```

## Logging

### 1. Basic Logging
```python
import logging

# Basic configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='app.log'
)

# Create logger
logger = logging.getLogger(__name__)

# Different log levels
logger.debug("This is a debug message")
logger.info("This is an info message")
logger.warning("This is a warning message")
logger.error("This is an error message")
logger.critical("This is a critical message")

# Logging with variables
user_id = 12345
action = "login"
logger.info(f"User {user_id} performed action: {action}")

# Exception logging
try:
    result = 10 / 0
except ZeroDivisionError:
    logger.exception("Division by zero occurred")  # Includes traceback
```

### 2. Advanced Logging Configuration
```python
import logging
import logging.handlers
from datetime import datetime

# Custom formatter
class CustomFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'ENDC': '\033[0m'       # End color
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['ENDC'])
        record.levelname = f"{log_color}{record.levelname}{self.COLORS['ENDC']}"
        return super().format(record)

# Configure multiple handlers
def setup_logging():
    """Set up comprehensive logging configuration."""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # Console handler with colors
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = CustomFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    
    # File handler for all logs
    file_handler = logging.FileHandler('app.log')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    # Rotating file handler for large logs
    rotating_handler = logging.handlers.RotatingFileHandler(
        'app_rotating.log', maxBytes=1024*1024, backupCount=5
    )
    rotating_handler.setLevel(logging.WARNING)
    rotating_handler.setFormatter(file_formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(rotating_handler)
    
    return logger

# Usage
logger = setup_logging()
logger.info("Application started")
logger.error("This is an error message")
```

### 3. Structured Logging
```python
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 
                          'pathname', 'filename', 'module', 'lineno', 
                          'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process',
                          'message', 'exc_info', 'exc_text', 'stack_info']:
                log_entry[key] = value
        
        return json.dumps(log_entry)

# Set up structured logging
def setup_structured_logging():
    """Set up JSON-based structured logging."""
    logger = logging.getLogger('structured')
    handler = logging.FileHandler('structured.log')
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

# Usage with extra context
structured_logger = setup_structured_logging()
structured_logger.info("User action", extra={
    'user_id': 12345,
    'action': 'login',
    'ip_address': '192.168.1.1',
    'session_id': 'abc123'
})

# Decorator for function logging
def log_function_calls(logger):
    """Decorator to log function calls."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.info(f"Calling function {func.__name__}", extra={
                'function': func.__name__,
                'args': str(args),
                'kwargs': str(kwargs)
            })
            try:
                result = func(*args, **kwargs)
                logger.info(f"Function {func.__name__} completed successfully")
                return result
            except Exception as e:
                logger.error(f"Function {func.__name__} failed", extra={
                    'function': func.__name__,
                    'error': str(e)
                })
                raise
        return wrapper
    return decorator

# Usage
@log_function_calls(structured_logger)
def divide_numbers(a, b):
    return a / b

result = divide_numbers(10, 2)
```

## Best Practices

### 1. File Handling Best Practices
```python
# Always use context managers
# Good
with open("file.txt", "r") as file:
    content = file.read()

# Bad
file = open("file.txt", "r")
content = file.read()
file.close()  # Might not execute if exception occurs

# Handle encoding explicitly
with open("file.txt", "r", encoding="utf-8") as file:
    content = file.read()

# Use pathlib for path operations
from pathlib import Path
file_path = Path("data") / "subfolder" / "file.txt"

# Check file existence before operations
if file_path.exists():
    with file_path.open("r") as file:
        content = file.read()
```

### 2. Error Handling Best Practices
```python
# Be specific with exception handling
try:
    with open("file.txt", "r") as file:
        data = json.load(file)
except FileNotFoundError:
    # Handle missing file
    data = {}
except json.JSONDecodeError:
    # Handle invalid JSON
    data = {}
except PermissionError:
    # Handle permission issues
    raise

# Don't suppress exceptions unless necessary
try:
    risky_operation()
except SpecificException:
    # Handle specific case
    handle_error()
    # Don't use bare except: pass

# Use logging instead of print for error reporting
logger.error("Failed to process file", exc_info=True)
```

### 3. Performance Considerations
```python
# For large files, read in chunks
def process_large_file(file_path):
    with open(file_path, "r") as file:
        while True:
            chunk = file.read(8192)  # 8KB chunks
            if not chunk:
                break
            process_chunk(chunk)

# Use generators for memory efficiency
def read_lines(file_path):
    with open(file_path, "r") as file:
        for line in file:
            yield line.strip()

# Buffer writes for better performance
with open("output.txt", "w", buffering=8192) as file:
    for data in large_dataset:
        file.write(data)
```
