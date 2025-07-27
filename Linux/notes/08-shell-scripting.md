# Shell Scripting

## Introduction to Shell Scripting

Shell scripting is a powerful way to automate tasks in Linux. A shell script is a text file containing a sequence of commands that can be executed by the shell.

### Why Use Shell Scripts?
- **Automation**: Automate repetitive tasks
- **System Administration**: Manage system configurations
- **Task Scheduling**: Run tasks at specific times
- **Process Control**: Chain multiple commands together
- **Customization**: Create custom tools and utilities

## Basic Shell Script Structure

### Shebang Line
The first line of a shell script should specify which interpreter to use:

```bash
#!/bin/bash          # Use bash shell
#!/bin/sh            # Use POSIX shell
#!/usr/bin/env bash  # Use bash from PATH
```

### Basic Script Template
```bash
#!/bin/bash

# Script: example.sh
# Description: Example shell script template
# Author: Your Name
# Date: 2024-01-01

# Script body starts here
echo "Hello, World!"
```

### Making Scripts Executable
```bash
# Make script executable
chmod +x script.sh

# Run script
./script.sh

# Run script with bash explicitly
bash script.sh
```

## Variables

### Variable Declaration and Usage
```bash
#!/bin/bash

# Variable assignment (no spaces around =)
name="John"
age=25
directory="/home/user"

# Using variables
echo "Name: $name"
echo "Age: ${age}"
echo "Directory: $directory"

# Read-only variables
readonly PI=3.14159
declare -r CONSTANT="unchangeable"
```

### Special Variables
```bash
#!/bin/bash

echo "Script name: $0"
echo "First argument: $1"
echo "Second argument: $2"
echo "All arguments: $@"
echo "Number of arguments: $#"
echo "Exit status of last command: $?"
echo "Process ID: $$"
echo "Background process ID: $!"
```

### Environment Variables
```bash
#!/bin/bash

# Common environment variables
echo "Home directory: $HOME"
echo "Current user: $USER"
echo "Current directory: $PWD"
echo "PATH: $PATH"

# Setting environment variables
export MY_VAR="value"
export PATH="$PATH:/new/directory"
```

### Variable Scope
```bash
#!/bin/bash

# Global variable
global_var="I'm global"

function test_scope() {
    # Local variable
    local local_var="I'm local"
    echo "Inside function: $global_var"
    echo "Inside function: $local_var"
}

test_scope
echo "Outside function: $global_var"
# echo "Outside function: $local_var"  # This would be empty
```

## Input and Output

### Reading User Input
```bash
#!/bin/bash

# Simple input
echo "Enter your name: "
read name
echo "Hello, $name!"

# Input with prompt
read -p "Enter your age: " age
echo "You are $age years old."

# Silent input (for passwords)
read -s -p "Enter password: " password
echo
echo "Password entered."

# Reading multiple values
read -p "Enter your first and last name: " first last
echo "First: $first, Last: $last"

# Reading with timeout
if read -t 10 -p "Enter something (10 seconds): " input; then
    echo "You entered: $input"
else
    echo "Timeout!"
fi
```

### Command Line Arguments
```bash
#!/bin/bash

# Check if arguments provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <argument1> <argument2>"
    exit 1
fi

# Process arguments
echo "Script: $0"
echo "First argument: $1"
echo "Second argument: $2"

# Loop through all arguments
for arg in "$@"; do
    echo "Argument: $arg"
done

# Shift arguments
echo "Before shift: $1 $2 $3"
shift
echo "After shift: $1 $2"
```

### Output Redirection
```bash
#!/bin/bash

# Redirect output to file
echo "This goes to file" > output.txt

# Append to file
echo "This is appended" >> output.txt

# Redirect stderr
command_that_fails 2> error.log

# Redirect both stdout and stderr
command 2>&1 > all_output.log

# Redirect to /dev/null (discard output)
noisy_command > /dev/null 2>&1
```

## Control Structures

### Conditional Statements

#### if-else Statements
```bash
#!/bin/bash

# Basic if statement
if [ "$USER" = "root" ]; then
    echo "You are root user"
fi

# if-else statement
if [ -f "/etc/passwd" ]; then
    echo "Password file exists"
else
    echo "Password file not found"
fi

# if-elif-else statement
read -p "Enter a number: " num
if [ $num -gt 100 ]; then
    echo "Number is greater than 100"
elif [ $num -eq 100 ]; then
    echo "Number is exactly 100"
else
    echo "Number is less than 100"
fi

# Multiple conditions
if [ -f "file.txt" ] && [ -r "file.txt" ]; then
    echo "File exists and is readable"
fi

if [ "$USER" = "admin" ] || [ "$USER" = "root" ]; then
    echo "You have admin privileges"
fi
```

#### Test Operators
```bash
# File tests
[ -e file ]     # File exists
[ -f file ]     # Regular file
[ -d file ]     # Directory
[ -r file ]     # Readable
[ -w file ]     # Writable
[ -x file ]     # Executable
[ -s file ]     # File size > 0

# String tests
[ -z "$str" ]   # String is empty
[ -n "$str" ]   # String is not empty
[ "$str1" = "$str2" ]   # Strings equal
[ "$str1" != "$str2" ]  # Strings not equal

# Numeric tests
[ $num1 -eq $num2 ]  # Equal
[ $num1 -ne $num2 ]  # Not equal
[ $num1 -lt $num2 ]  # Less than
[ $num1 -le $num2 ]  # Less than or equal
[ $num1 -gt $num2 ]  # Greater than
[ $num1 -ge $num2 ]  # Greater than or equal
```

#### case Statements
```bash
#!/bin/bash

read -p "Enter a character: " char

case $char in
    [a-z])
        echo "Lowercase letter"
        ;;
    [A-Z])
        echo "Uppercase letter"
        ;;
    [0-9])
        echo "Digit"
        ;;
    *)
        echo "Special character"
        ;;
esac

# Menu example
echo "Choose an option:"
echo "1) List files"
echo "2) Show date"
echo "3) Exit"
read -p "Enter choice: " choice

case $choice in
    1)
        ls -la
        ;;
    2)
        date
        ;;
    3)
        echo "Goodbye!"
        exit 0
        ;;
    *)
        echo "Invalid option"
        ;;
esac
```

### Loops

#### for Loops
```bash
#!/bin/bash

# Basic for loop
for i in 1 2 3 4 5; do
    echo "Number: $i"
done

# For loop with range
for i in {1..10}; do
    echo "Count: $i"
done

# For loop with step
for i in {0..20..2}; do
    echo "Even number: $i"
done

# For loop with files
for file in *.txt; do
    echo "Processing: $file"
done

# For loop with command output
for user in $(cat /etc/passwd | cut -d: -f1); do
    echo "User: $user"
done

# C-style for loop
for ((i=1; i<=10; i++)); do
    echo "Iteration: $i"
done
```

#### while Loops
```bash
#!/bin/bash

# Basic while loop
counter=1
while [ $counter -le 5 ]; do
    echo "Counter: $counter"
    counter=$((counter + 1))
done

# Reading file line by line
while IFS= read -r line; do
    echo "Line: $line"
done < "input.txt"

# Infinite loop with break
while true; do
    read -p "Enter 'quit' to exit: " input
    if [ "$input" = "quit" ]; then
        break
    fi
    echo "You entered: $input"
done

# While loop with condition check
while [ -f "process.lock" ]; do
    echo "Waiting for process to finish..."
    sleep 5
done
```

#### until Loops
```bash
#!/bin/bash

# until loop (opposite of while)
counter=1
until [ $counter -gt 5 ]; do
    echo "Counter: $counter"
    counter=$((counter + 1))
done

# Wait until file exists
until [ -f "important_file.txt" ]; do
    echo "Waiting for file..."
    sleep 2
done
echo "File found!"
```

### Loop Control
```bash
#!/bin/bash

# break and continue
for i in {1..10}; do
    if [ $i -eq 3 ]; then
        continue  # Skip iteration
    fi
    if [ $i -eq 8 ]; then
        break    # Exit loop
    fi
    echo "Number: $i"
done
```

## Functions

### Function Definition and Usage
```bash
#!/bin/bash

# Function definition
greet() {
    echo "Hello, $1!"
}

# Function call
greet "World"
greet "Alice"

# Function with multiple parameters
calculate_sum() {
    local num1=$1
    local num2=$2
    local sum=$((num1 + num2))
    echo $sum
}

result=$(calculate_sum 10 20)
echo "Sum: $result"

# Function with return value
is_even() {
    local number=$1
    if [ $((number % 2)) -eq 0 ]; then
        return 0  # True
    else
        return 1  # False
    fi
}

if is_even 4; then
    echo "4 is even"
fi
```

### Advanced Function Features
```bash
#!/bin/bash

# Function with default parameters
greet_with_default() {
    local name=${1:-"World"}
    echo "Hello, $name!"
}

greet_with_default        # Uses default
greet_with_default "Bob"  # Uses provided name

# Function with variable arguments
print_all() {
    echo "Number of arguments: $#"
    for arg in "$@"; do
        echo "Argument: $arg"
    done
}

print_all one two three four

# Recursive function
factorial() {
    local n=$1
    if [ $n -le 1 ]; then
        echo 1
    else
        local prev=$(factorial $((n - 1)))
        echo $((n * prev))
    fi
}

echo "Factorial of 5: $(factorial 5)"
```

## Arrays

### Array Basics
```bash
#!/bin/bash

# Array declaration
fruits=("apple" "banana" "orange" "grape")

# Another way to declare
declare -a numbers
numbers[0]=10
numbers[1]=20
numbers[2]=30

# Array access
echo "First fruit: ${fruits[0]}"
echo "Second fruit: ${fruits[1]}"

# All elements
echo "All fruits: ${fruits[@]}"
echo "All fruits: ${fruits[*]}"

# Array length
echo "Number of fruits: ${#fruits[@]}"

# Array indices
echo "Indices: ${!fruits[@]}"
```

### Array Operations
```bash
#!/bin/bash

# Adding elements
fruits=("apple" "banana")
fruits+=("orange")           # Append single element
fruits+=("grape" "mango")    # Append multiple elements

# Removing elements
unset fruits[1]             # Remove element at index 1

# Looping through array
for fruit in "${fruits[@]}"; do
    echo "Fruit: $fruit"
done

# Looping with indices
for i in "${!fruits[@]}"; do
    echo "Index $i: ${fruits[i]}"
done

# Array slicing
echo "Elements 1-2: ${fruits[@]:1:2}"

# Search in array
search_element() {
    local element=$1
    shift
    local array=("$@")
    
    for item in "${array[@]}"; do
        if [ "$item" = "$element" ]; then
            return 0
        fi
    done
    return 1
}

if search_element "apple" "${fruits[@]}"; then
    echo "Found apple"
fi
```

### Associative Arrays (Bash 4+)
```bash
#!/bin/bash

# Declare associative array
declare -A person

# Set values
person[name]="John"
person[age]=30
person[city]="New York"

# Access values
echo "Name: ${person[name]}"
echo "Age: ${person[age]}"

# All keys
echo "Keys: ${!person[@]}"

# All values
echo "Values: ${person[@]}"

# Loop through associative array
for key in "${!person[@]}"; do
    echo "$key: ${person[$key]}"
done
```

## String Manipulation

### String Operations
```bash
#!/bin/bash

text="Hello World"

# String length
echo "Length: ${#text}"

# Substring extraction
echo "Substring: ${text:6:5}"    # From position 6, length 5
echo "From position: ${text:6}"  # From position 6 to end

# String replacement
echo "Replace: ${text/World/Universe}"     # Replace first occurrence
echo "Replace all: ${text//l/L}"           # Replace all occurrences

# String removal
filename="script.sh"
echo "Remove extension: ${filename%.sh}"   # Remove shortest match from end
echo "Get extension: ${filename#*.}"       # Remove shortest match from beginning

# Case conversion (Bash 4+)
echo "Uppercase: ${text^^}"
echo "Lowercase: ${text,,}"
echo "Capitalize: ${text^}"
```

### Pattern Matching
```bash
#!/bin/bash

# Wildcard matching
filename="document.pdf"
case $filename in
    *.txt)
        echo "Text file"
        ;;
    *.pdf)
        echo "PDF file"
        ;;
    *.doc|*.docx)
        echo "Word document"
        ;;
    *)
        echo "Unknown file type"
        ;;
esac

# Parameter expansion with patterns
path="/home/user/documents/file.txt"
echo "Directory: ${path%/*}"      # Remove filename
echo "Filename: ${path##*/}"      # Remove path
echo "Extension: ${path##*.}"     # Get extension
echo "Name only: ${path##*/}"     # Get filename
echo "Name without ext: ${path%.*}"  # Remove extension
```

## File Operations

### File Testing and Manipulation
```bash
#!/bin/bash

# File operations script
process_file() {
    local file=$1
    
    if [ ! -f "$file" ]; then
        echo "Error: File '$file' does not exist"
        return 1
    fi
    
    echo "File: $file"
    echo "Size: $(stat -c%s "$file") bytes"
    echo "Modified: $(stat -c%y "$file")"
    echo "Permissions: $(stat -c%A "$file")"
    echo "Owner: $(stat -c%U "$file")"
    echo "Group: $(stat -c%G "$file")"
    
    if [ -r "$file" ]; then
        echo "First 5 lines:"
        head -5 "$file"
    fi
}

# Usage
process_file "/etc/passwd"
```

### Directory Operations
```bash
#!/bin/bash

# Create directory structure
create_project_structure() {
    local project_name=$1
    
    if [ -z "$project_name" ]; then
        echo "Usage: create_project_structure <project_name>"
        return 1
    fi
    
    echo "Creating project structure for: $project_name"
    
    mkdir -p "$project_name"/{src,docs,tests,config}
    touch "$project_name"/README.md
    touch "$project_name"/src/main.sh
    touch "$project_name"/tests/test.sh
    
    echo "Project structure created:"
    tree "$project_name" 2>/dev/null || find "$project_name" -type f
}

# Find files by criteria
find_large_files() {
    local directory=${1:-"."}
    local size=${2:-"100M"}
    
    echo "Finding files larger than $size in $directory:"
    find "$directory" -type f -size +$size -exec ls -lh {} \; 2>/dev/null
}
```

## Error Handling

### Exit Codes and Error Checking
```bash
#!/bin/bash

# Set error handling options
set -e          # Exit on any error
set -u          # Exit on undefined variable
set -o pipefail # Exit on pipe failure

# Function to handle errors
error_exit() {
    echo "Error: $1" >&2
    exit 1
}

# Check command success
check_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        error_exit "Command '$1' not found"
    fi
}

# Safe file operations
safe_copy() {
    local src=$1
    local dest=$2
    
    [ $# -ne 2 ] && error_exit "Usage: safe_copy <source> <destination>"
    [ ! -f "$src" ] && error_exit "Source file '$src' does not exist"
    
    if cp "$src" "$dest"; then
        echo "Successfully copied $src to $dest"
    else
        error_exit "Failed to copy $src to $dest"
    fi
}

# Trap for cleanup
cleanup() {
    echo "Cleaning up..."
    rm -f /tmp/script.lock
}

trap cleanup EXIT

# Create lock file
echo $$ > /tmp/script.lock
```

### Input Validation
```bash
#!/bin/bash

# Validate numeric input
validate_number() {
    local input=$1
    
    if [[ ! "$input" =~ ^[0-9]+$ ]]; then
        echo "Error: '$input' is not a valid number"
        return 1
    fi
    
    return 0
}

# Validate email format
validate_email() {
    local email=$1
    local regex="^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    
    if [[ "$email" =~ $regex ]]; then
        return 0
    else
        echo "Error: '$email' is not a valid email address"
        return 1
    fi
}

# Validate file path
validate_path() {
    local path=$1
    
    if [ ! -e "$path" ]; then
        echo "Error: Path '$path' does not exist"
        return 1
    fi
    
    return 0
}

# Example usage
read -p "Enter a number: " num
if validate_number "$num"; then
    echo "Valid number: $num"
fi
```

## Advanced Features

### Command Substitution
```bash
#!/bin/bash

# Command substitution with $()
current_date=$(date)
echo "Current date: $current_date"

# Backticks (older syntax)
current_user=`whoami`
echo "Current user: $current_user"

# Nested command substitution
files_count=$(ls $(pwd) | wc -l)
echo "Files in current directory: $files_count"

# Using in conditions
if [ "$(id -u)" -eq 0 ]; then
    echo "Running as root"
fi
```

### Process Substitution
```bash
#!/bin/bash

# Compare output of two commands
diff <(ls /etc) <(ls /usr/etc) || true

# Use command output as input
while read -r line; do
    echo "Processing: $line"
done < <(find /var/log -name "*.log")
```

### Here Documents
```bash
#!/bin/bash

# Here document
cat << EOF
This is a here document.
It can span multiple lines.
Current user: $USER
Current date: $(date)
EOF

# Here document with function
send_email() {
    local recipient=$1
    local subject=$2
    
    mail "$recipient" << EOF
Subject: $subject

Dear User,

This is an automated message.

Best regards,
System Administrator
EOF
}

# Here string
grep "pattern" <<< "$variable"
```

### Parameter Expansion
```bash
#!/bin/bash

# Default values
name=${1:-"World"}                    # Use "World" if $1 is empty
config_file=${CONFIG_FILE:-"/etc/app.conf"}

# Required parameters
database_url=${DATABASE_URL:?"DATABASE_URL is required"}

# Conditional assignment
temp_dir=${TEMP_DIR:="/tmp/myapp"}

# Length and substrings
text="Hello World"
echo "Length: ${#text}"
echo "Substring: ${text:0:5}"
echo "Remove prefix: ${text#Hello }"
echo "Remove suffix: ${text% World}"
```

## Practical Examples

### System Information Script
```bash
#!/bin/bash

# system_info.sh - Display system information

echo "=== System Information ==="
echo "Hostname: $(hostname)"
echo "Operating System: $(lsb_release -d 2>/dev/null | cut -f2 || echo "Unknown")"
echo "Kernel Version: $(uname -r)"
echo "Architecture: $(uname -m)"
echo "Uptime: $(uptime -p 2>/dev/null || uptime)"
echo

echo "=== Hardware Information ==="
echo "CPU: $(lscpu | grep "Model name" | cut -d: -f2 | xargs)"
echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
echo "Disk Usage:"
df -h | grep -E '^/dev/' | awk '{print "  " $1 ": " $3 "/" $2 " (" $5 " used)"}'
echo

echo "=== Network Information ==="
echo "IP Addresses:"
ip addr show | grep "inet " | grep -v "127.0.0.1" | awk '{print "  " $2}'

echo "=== Current Users ==="
who
```

### Log Analysis Script
```bash
#!/bin/bash

# log_analyzer.sh - Analyze log files

analyze_log() {
    local log_file=$1
    local pattern=${2:-"ERROR"}
    
    if [ ! -f "$log_file" ]; then
        echo "Error: Log file '$log_file' not found"
        return 1
    fi
    
    echo "=== Log Analysis for $log_file ==="
    echo "File size: $(stat -c%s "$log_file" | numfmt --to=iec)"
    echo "Last modified: $(stat -c%y "$log_file")"
    echo "Total lines: $(wc -l < "$log_file")"
    echo
    
    echo "=== Pattern Analysis: $pattern ==="
    local count=$(grep -c "$pattern" "$log_file" 2>/dev/null || echo "0")
    echo "Occurrences of '$pattern': $count"
    
    if [ "$count" -gt 0 ]; then
        echo "Recent occurrences:"
        grep "$pattern" "$log_file" | tail -5
    fi
    echo
}

# Usage example
for log in /var/log/syslog /var/log/auth.log; do
    if [ -f "$log" ]; then
        analyze_log "$log" "FAILED"
    fi
done
```

### Backup Script
```bash
#!/bin/bash

# backup.sh - Automated backup script

BACKUP_SOURCE="/home"
BACKUP_DEST="/backup"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="backup_$TIMESTAMP"
LOG_FILE="/var/log/backup.log"

# Function to log messages
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Check if backup destination exists
if [ ! -d "$BACKUP_DEST" ]; then
    log_message "ERROR: Backup destination $BACKUP_DEST does not exist"
    exit 1
fi

# Check available space
available_space=$(df "$BACKUP_DEST" | awk 'NR==2 {print $4}')
required_space=$(du -s "$BACKUP_SOURCE" | awk '{print $1}')

if [ "$available_space" -lt "$required_space" ]; then
    log_message "ERROR: Insufficient space for backup"
    exit 1
fi

# Start backup
log_message "Starting backup of $BACKUP_SOURCE"

if tar -czf "$BACKUP_DEST/$BACKUP_NAME.tar.gz" -C "$BACKUP_SOURCE" .; then
    log_message "Backup completed successfully: $BACKUP_NAME.tar.gz"
    
    # Remove backups older than 7 days
    find "$BACKUP_DEST" -name "backup_*.tar.gz" -mtime +7 -delete
    log_message "Cleaned up old backups"
else
    log_message "ERROR: Backup failed"
    exit 1
fi

log_message "Backup process finished"
```

## Best Practices

### Script Writing Guidelines
1. **Use meaningful variable names**
2. **Add comments for complex logic**
3. **Validate input parameters**
4. **Handle errors gracefully**
5. **Use functions for reusable code**
6. **Quote variables to prevent word splitting**
7. **Use local variables in functions**
8. **Test scripts thoroughly**

### Performance Tips
1. **Avoid unnecessary subshells**
2. **Use built-in commands when possible**
3. **Minimize external command calls**
4. **Use arrays for multiple values**
5. **Cache expensive operations**

### Security Considerations
1. **Validate all inputs**
2. **Use absolute paths for commands**
3. **Avoid eval with user input**
4. **Set proper file permissions**
5. **Don't store passwords in scripts**
6. **Use temporary files securely**

### Debugging Tips
```bash
#!/bin/bash

# Enable debugging
set -x          # Print commands as they're executed
set -v          # Print input lines as they're read

# Conditional debugging
DEBUG=${DEBUG:-0}
debug() {
    if [ "$DEBUG" -eq 1 ]; then
        echo "DEBUG: $*" >&2
    fi
}

# Usage: DEBUG=1 ./script.sh
debug "This is a debug message"
```
