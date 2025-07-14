# 01. C++ Basics

## 📋 Overview
C++ is a general-purpose programming language that was developed by Bjarne Stroustrup at Bell Labs starting in 1979. It's an extension of the C programming language and is often referred to as "C with Classes."

## 🎯 Key Features of C++

### 1. **Object-Oriented Programming (OOP)**
- Supports classes and objects
- Encapsulation, Inheritance, Polymorphism
- Data abstraction

### 2. **Low-Level Control**
- Direct memory management
- Pointer arithmetic
- Hardware-level programming capabilities

### 3. **Performance**
- Compiled language
- Minimal runtime overhead
- Efficient execution

### 4. **Standard Library**
- Rich collection of pre-written functions
- STL (Standard Template Library)
- Extensive I/O facilities

## 🏗️ Basic Structure of a C++ Program

```cpp
// Preprocessor directive
#include <iostream>
using namespace std;

// Main function - entry point of the program
int main() {
    // Program statements
    cout << "Hello, World!" << endl;
    
    // Return statement
    return 0;
}
```

### Breakdown:
- **`#include <iostream>`**: Preprocessor directive to include input/output stream library
- **`using namespace std`**: Allows using standard library functions without `std::` prefix
- **`int main()`**: Main function where program execution begins
- **`cout`**: Output stream object for displaying data
- **`endl`**: Manipulator that inserts newline and flushes the buffer
- **`return 0`**: Indicates successful program termination

## 📝 Comments in C++

### Single-line Comments
```cpp
// This is a single-line comment
int x = 5; // Comment at the end of a line
```

### Multi-line Comments
```cpp
/*
This is a multi-line comment
that spans across multiple lines
*/
```

### Documentation Comments (Doxygen style)
```cpp
/**
 * @brief Brief description of function
 * @param param1 Description of parameter
 * @return Description of return value
 */
```

## 🔧 Preprocessor Directives

### Common Preprocessor Directives
```cpp
#include <iostream>    // Include system header
#include "myheader.h"  // Include user-defined header
#define PI 3.14159     // Define a macro
#ifdef DEBUG           // Conditional compilation
    // Debug code
#endif
```

### Header Guards
```cpp
#ifndef MYHEADER_H
#define MYHEADER_H
// Header content
#endif
```

## 🌐 Namespaces

### What is a Namespace?
A namespace is a declarative region that provides a scope to the identifiers inside it.

```cpp
// Defining a namespace
namespace MyNamespace {
    int value = 100;
    void display() {
        cout << "Value: " << value << endl;
    }
}

// Using namespace elements
int main() {
    MyNamespace::display();           // Using scope resolution
    cout << MyNamespace::value;       // Accessing variable
    
    using MyNamespace::value;         // Using declaration
    cout << value;                    // Now can use directly
    
    using namespace MyNamespace;      // Using directive
    display();                        // Now can use directly
    
    return 0;
}
```

### Standard Namespace
```cpp
// Instead of using namespace std;
std::cout << "Hello World" << std::endl;

// Or use specific elements
using std::cout;
using std::endl;
cout << "Hello World" << endl;
```

## 🔍 Identifiers and Keywords

### Identifiers
Rules for naming identifiers:
- Must start with a letter or underscore
- Can contain letters, digits, and underscores
- Case-sensitive
- Cannot be a keyword

```cpp
// Valid identifiers
int age;
float _temperature;
char firstName;
int student123;

// Invalid identifiers
int 123student;  // Cannot start with digit
int class;       // 'class' is a keyword
int first-name;  // Cannot contain hyphen
```

### Keywords (Reserved Words)
```cpp
// Some common C++ keywords
alignas     alignof     and         and_eq      asm
auto        bitand      bitor       bool        break
case        catch       char        char16_t    char32_t
class       compl       const       constexpr   const_cast
continue    decltype    default     delete      do
double      dynamic_cast else       enum        explicit
export      extern      false       float       for
friend      goto        if          inline      int
long        mutable     namespace   new         noexcept
not         not_eq      nullptr     operator    or
or_eq       private     protected   public      register
return      short       signed      sizeof      static
static_assert static_cast struct    switch      template
this        thread_local throw      true        try
typedef     typeid      typename    union       unsigned
using       virtual     void        volatile    wchar_t
while       xor         xor_eq
```

## 🚀 Program Compilation Process

### Steps in C++ Compilation:

1. **Preprocessing**
   - Handles preprocessor directives
   - Includes header files
   - Expands macros

2. **Compilation**
   - Converts source code to assembly language
   - Checks syntax and semantics

3. **Assembly**
   - Converts assembly code to machine code (object files)

4. **Linking**
   - Combines object files and libraries
   - Creates executable file

```bash
# Compilation command
g++ -o program program.cpp

# Step-by-step compilation
g++ -E program.cpp -o program.i     # Preprocessing
g++ -S program.i -o program.s        # Compilation
g++ -c program.s -o program.o        # Assembly
g++ program.o -o program             # Linking
```

## 💡 Best Practices

### 1. **Code Organization**
```cpp
// Good practice
#include <iostream>
#include <vector>

using namespace std;

int main() {
    // Clear, readable code
    vector<int> numbers = {1, 2, 3, 4, 5};
    
    for (const auto& num : numbers) {
        cout << num << " ";
    }
    cout << endl;
    
    return 0;
}
```

### 2. **Naming Conventions**
```cpp
// Variables and functions: camelCase or snake_case
int studentAge;          // camelCase
int student_age;         // snake_case

// Constants: UPPER_CASE
const int MAX_SIZE = 100;

// Classes: PascalCase
class StudentRecord {
    // ...
};
```

### 3. **Code Comments**
```cpp
// Use comments to explain WHY, not WHAT
int calculateGPA(const vector<int>& grades) {
    // Using weighted average algorithm for GPA calculation
    // This accounts for credit hours per course
    int total = 0;
    for (const auto& grade : grades) {
        total += grade;
    }
    return total / grades.size();
}
```

## 🔗 Related Topics
- [Data Types & Variables](./02-data-types-variables.md)
- [Operators](./03-operators.md)
- [Control Structures](./04-control-structures.md)

---
*Next: [Data Types & Variables](./02-data-types-variables.md)*
