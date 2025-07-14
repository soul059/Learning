# 02. Data Types & Variables

## 📋 Overview
Data types specify the type of data that variables can store. C++ supports various built-in data types and allows you to create user-defined data types.

## 🔢 Fundamental Data Types

### 1. **Integer Types**

| Type | Size (bytes) | Range | Description |
|------|--------------|-------|-------------|
| `char` | 1 | -128 to 127 | Single character |
| `signed char` | 1 | -128 to 127 | Explicitly signed char |
| `unsigned char` | 1 | 0 to 255 | Unsigned character |
| `short` | 2 | -32,768 to 32,767 | Short integer |
| `unsigned short` | 2 | 0 to 65,535 | Unsigned short |
| `int` | 4 | -2,147,483,648 to 2,147,483,647 | Standard integer |
| `unsigned int` | 4 | 0 to 4,294,967,295 | Unsigned integer |
| `long` | 4/8 | Platform dependent | Long integer |
| `long long` | 8 | Very large range | 64-bit integer |

```cpp
#include <iostream>
#include <climits>
using namespace std;

int main() {
    // Integer type examples
    char letter = 'A';
    short temperature = -15;
    int age = 25;
    long population = 1000000L;
    long long bigNumber = 9223372036854775807LL;
    
    // Display sizes and ranges
    cout << "Size of int: " << sizeof(int) << " bytes" << endl;
    cout << "Range of int: " << INT_MIN << " to " << INT_MAX << endl;
    
    return 0;
}
```

### 2. **Floating-Point Types**

| Type | Size (bytes) | Precision | Range |
|------|--------------|-----------|-------|
| `float` | 4 | ~7 digits | ±3.4 × 10^±38 |
| `double` | 8 | ~15 digits | ±1.7 × 10^±308 |
| `long double` | 12/16 | ~19 digits | Extended precision |

```cpp
#include <iostream>
#include <iomanip>
using namespace std;

int main() {
    float pi_f = 3.14159f;              // f suffix for float
    double pi_d = 3.14159265358979;     // default is double
    long double pi_ld = 3.14159265358979323846L; // L suffix
    
    // Set precision for output
    cout << setprecision(15);
    cout << "Float PI: " << pi_f << endl;
    cout << "Double PI: " << pi_d << endl;
    cout << "Long Double PI: " << pi_ld << endl;
    
    return 0;
}
```

### 3. **Boolean Type**
```cpp
bool isValid = true;
bool isComplete = false;

// Boolean expressions
bool result = (5 > 3);  // true
bool comparison = (10 == 5);  // false

// In conditional statements
if (isValid) {
    cout << "Data is valid" << endl;
}
```

### 4. **Character Types**
```cpp
char ch = 'A';                    // Single character
char newline = '\n';              // Escape sequence
char tab = '\t';                  // Tab character
char backslash = '\\';            // Backslash

// Wide character types
wchar_t wideChar = L'Ω';          // Wide character
char16_t utf16Char = u'π';        // UTF-16
char32_t utf32Char = U'🌟';       // UTF-32
```

## 📝 Variable Declaration and Initialization

### Declaration Syntax
```cpp
// Basic declaration
dataType variableName;

// Multiple variables of same type
int x, y, z;

// Declaration with initialization
int age = 25;
double salary = 50000.50;
char grade = 'A';
bool isStudent = true;
```

### Initialization Methods
```cpp
// 1. Copy initialization
int x = 10;

// 2. Direct initialization
int y(20);

// 3. Uniform initialization (C++11)
int z{30};
int w{};        // Initialized to 0

// 4. List initialization
int numbers{1, 2, 3, 4, 5};  // Error for single variable
```

### Variable Scope
```cpp
#include <iostream>
using namespace std;

int globalVar = 100;  // Global scope

int main() {
    int localVar = 50;  // Local scope
    
    {
        int blockVar = 25;  // Block scope
        cout << globalVar << endl;  // Accessible
        cout << localVar << endl;   // Accessible
        cout << blockVar << endl;   // Accessible
    }
    
    // cout << blockVar;  // Error: not accessible
    
    return 0;
}
```

## 🔒 Constants

### 1. **const Keyword**
```cpp
const int MAX_SIZE = 100;
const double PI = 3.14159;
const char GRADE = 'A';

// const with pointers
const int* ptr1;        // Pointer to constant int
int* const ptr2 = &x;   // Constant pointer to int
const int* const ptr3 = &x;  // Constant pointer to constant int
```

### 2. **constexpr (C++11)**
```cpp
constexpr int SIZE = 10;           // Compile-time constant
constexpr double AREA = PI * 5 * 5; // Computed at compile time

constexpr int factorial(int n) {
    return (n <= 1) ? 1 : n * factorial(n - 1);
}

constexpr int fact5 = factorial(5);  // Computed at compile time
```

### 3. **#define Preprocessor**
```cpp
#define MAX_STUDENTS 50
#define PI 3.14159
#define SQUARE(x) ((x) * (x))

// Usage
int array[MAX_STUDENTS];
double area = PI * SQUARE(radius);
```

### 4. **enum Constants**
```cpp
enum Color {
    RED,      // 0
    GREEN,    // 1
    BLUE      // 2
};

enum class Status : int {
    PENDING = 1,
    APPROVED = 2,
    REJECTED = 3
};

Color favoriteColor = RED;
Status orderStatus = Status::PENDING;
```

## 🎯 Type Modifiers

### Sign Modifiers
```cpp
signed int x = -100;      // Can store negative values
unsigned int y = 100;     // Only positive values

signed char ch1 = -50;    // Range: -128 to 127
unsigned char ch2 = 200;  // Range: 0 to 255
```

### Size Modifiers
```cpp
short int shortNum = 1000;
long int longNum = 1000000L;
long long int bigNum = 1000000000000LL;

// Equivalent declarations
short shortNum2 = 1000;
long longNum2 = 1000000L;
long long bigNum2 = 1000000000000LL;
```

## 🔄 Type Conversion

### 1. **Implicit Conversion (Type Promotion)**
```cpp
int x = 10;
double y = 3.14;
double result = x + y;  // x is promoted to double

char ch = 'A';
int ascii = ch;         // char promoted to int (65)

bool flag = 5;          // Any non-zero value becomes true
int num = true;         // true becomes 1
```

### 2. **Explicit Conversion (Type Casting)**

#### C-style Cast
```cpp
double pi = 3.14159;
int intPi = (int)pi;    // C-style cast
```

#### C++ Cast Operators
```cpp
// static_cast - Safe compile-time conversion
double d = 3.14;
int i = static_cast<int>(d);

// dynamic_cast - Runtime type checking (for polymorphism)
Base* basePtr = new Derived();
Derived* derivedPtr = dynamic_cast<Derived*>(basePtr);

// const_cast - Remove/add const qualifier
const int x = 10;
int* ptr = const_cast<int*>(&x);

// reinterpret_cast - Low-level reinterpretation
int num = 65;
char* charPtr = reinterpret_cast<char*>(&num);
```

## 📏 sizeof Operator

```cpp
#include <iostream>
using namespace std;

int main() {
    cout << "Size of data types:" << endl;
    cout << "char: " << sizeof(char) << " bytes" << endl;
    cout << "int: " << sizeof(int) << " bytes" << endl;
    cout << "float: " << sizeof(float) << " bytes" << endl;
    cout << "double: " << sizeof(double) << " bytes" << endl;
    cout << "long long: " << sizeof(long long) << " bytes" << endl;
    
    int arr[10];
    cout << "Array size: " << sizeof(arr) << " bytes" << endl;
    cout << "Array elements: " << sizeof(arr)/sizeof(arr[0]) << endl;
    
    return 0;
}
```

## 🎲 Auto Keyword (C++11)

```cpp
auto x = 10;        // x is int
auto y = 3.14;      // y is double
auto z = 'A';       // z is char
auto w = true;      // w is bool

// With complex types
auto numbers = {1, 2, 3, 4, 5};  // std::initializer_list<int>

// Function return type deduction
auto add(int a, int b) -> int {
    return a + b;
}

// Or in C++14
auto multiply(int a, int b) {
    return a * b;  // Return type deduced as int
}
```

## 🔗 User-Defined Data Types

### 1. **typedef**
```cpp
typedef unsigned long ulong;
typedef int* IntPtr;

ulong bigNumber = 1000000UL;
IntPtr ptr = &someInt;
```

### 2. **using (C++11 - Type Alias)**
```cpp
using ulong = unsigned long;
using IntPtr = int*;
using StringVector = std::vector<std::string>;

StringVector names = {"Alice", "Bob", "Charlie"};
```

### 3. **struct**
```cpp
struct Point {
    int x;
    int y;
};

Point p1 = {10, 20};
Point p2{30, 40};  // Uniform initialization
```

### 4. **enum class (C++11)**
```cpp
enum class Color : int {
    RED = 1,
    GREEN = 2,
    BLUE = 3
};

enum class Size {
    SMALL,
    MEDIUM,
    LARGE
};

Color favoriteColor = Color::RED;
Size shirtSize = Size::MEDIUM;

// Type-safe - no implicit conversion
// int x = Color::RED;  // Error
int x = static_cast<int>(Color::RED);  // OK
```

## 💡 Best Practices

### 1. **Choose Appropriate Data Types**
```cpp
// Use appropriate size
int count = 0;              // For counting
long long bigNumber = 1e18; // For very large numbers
double precise = 3.14159;   // For precision

// Use unsigned for non-negative values
unsigned int arraySize = 100;
size_t stringLength = str.length();
```

### 2. **Initialize Variables**
```cpp
// Good practice - always initialize
int count = 0;
double sum = 0.0;
bool isValid = false;
string name = "";

// Use uniform initialization
int x{0};
double y{0.0};
```

### 3. **Use const When Possible**
```cpp
const int MAX_ATTEMPTS = 3;
const string DATABASE_URL = "localhost:5432";

void processData(const vector<int>& data) {
    // Function won't modify data
}
```

### 4. **Prefer auto for Complex Types**
```cpp
// Instead of this
std::map<std::string, std::vector<int>>::iterator it = myMap.begin();

// Use this
auto it = myMap.begin();
```

## 🔗 Related Topics
- [Operators](./03-operators.md)
- [Arrays & Strings](./06-arrays-strings.md)
- [Pointers & References](./07-pointers-references.md)

---
*Previous: [Basics](./01-basics.md) | Next: [Operators](./03-operators.md)*
