# 03. Operators

## 📋 Overview
Operators are special symbols that perform specific operations on one, two, or three operands, and then return a result. C++ provides a rich set of operators that can be categorized into different types.

## ➕ Arithmetic Operators

### Basic Arithmetic Operators
| Operator | Name | Description | Example |
|----------|------|-------------|---------|
| `+` | Addition | Adds two operands | `a + b` |
| `-` | Subtraction | Subtracts second operand from first | `a - b` |
| `*` | Multiplication | Multiplies two operands | `a * b` |
| `/` | Division | Divides first operand by second | `a / b` |
| `%` | Modulus | Returns remainder of division | `a % b` |

```cpp
#include <iostream>
using namespace std;

int main() {
    int a = 15, b = 4;
    
    cout << "a = " << a << ", b = " << b << endl;
    cout << "a + b = " << (a + b) << endl;  // 19
    cout << "a - b = " << (a - b) << endl;  // 11
    cout << "a * b = " << (a * b) << endl;  // 60
    cout << "a / b = " << (a / b) << endl;  // 3 (integer division)
    cout << "a % b = " << (a % b) << endl;  // 3 (remainder)
    
    // Floating-point division
    double x = 15.0, y = 4.0;
    cout << "x / y = " << (x / y) << endl;  // 3.75
    
    return 0;
}
```

### Unary Arithmetic Operators
```cpp
int x = 5;
int y = -x;     // Unary minus: y = -5
int z = +x;     // Unary plus: z = 5

// Increment and Decrement
int a = 10;
int pre_inc = ++a;   // Pre-increment: a = 11, pre_inc = 11
int post_inc = a++;  // Post-increment: post_inc = 11, a = 12

int b = 10;
int pre_dec = --b;   // Pre-decrement: b = 9, pre_dec = 9
int post_dec = b--;  // Post-decrement: post_dec = 9, b = 8
```

## 🔄 Assignment Operators

### Basic Assignment
```cpp
int x = 10;        // Simple assignment
```

### Compound Assignment Operators
| Operator | Equivalent | Description |
|----------|------------|-------------|
| `+=` | `a = a + b` | Addition assignment |
| `-=` | `a = a - b` | Subtraction assignment |
| `*=` | `a = a * b` | Multiplication assignment |
| `/=` | `a = a / b` | Division assignment |
| `%=` | `a = a % b` | Modulus assignment |

```cpp
int x = 10;
x += 5;   // x = x + 5; x becomes 15
x -= 3;   // x = x - 3; x becomes 12
x *= 2;   // x = x * 2; x becomes 24
x /= 4;   // x = x / 4; x becomes 6
x %= 4;   // x = x % 4; x becomes 2

// Bitwise compound assignments
x |= 8;   // x = x | 8
x &= 15;  // x = x & 15
x ^= 7;   // x = x ^ 7
x <<= 2;  // x = x << 2
x >>= 1;  // x = x >> 1
```

## ⚖️ Comparison Operators

| Operator | Name | Description | Example |
|----------|------|-------------|---------|
| `==` | Equal to | Checks if operands are equal | `a == b` |
| `!=` | Not equal to | Checks if operands are not equal | `a != b` |
| `<` | Less than | Checks if left is less than right | `a < b` |
| `>` | Greater than | Checks if left is greater than right | `a > b` |
| `<=` | Less than or equal | Checks if left is ≤ right | `a <= b` |
| `>=` | Greater than or equal | Checks if left is ≥ right | `a >= b` |

```cpp
#include <iostream>
using namespace std;

int main() {
    int a = 10, b = 20, c = 10;
    
    cout << "a = " << a << ", b = " << b << ", c = " << c << endl;
    
    cout << "a == b: " << (a == b) << endl;  // false (0)
    cout << "a == c: " << (a == c) << endl;  // true (1)
    cout << "a != b: " << (a != b) << endl;  // true (1)
    cout << "a < b: " << (a < b) << endl;    // true (1)
    cout << "a > b: " << (a > b) << endl;    // false (0)
    cout << "a <= c: " << (a <= c) << endl;  // true (1)
    cout << "b >= a: " << (b >= a) << endl;  // true (1)
    
    return 0;
}
```

### String Comparison
```cpp
#include <string>
using namespace std;

string str1 = "apple";
string str2 = "banana";
string str3 = "apple";

bool isEqual = (str1 == str3);      // true
bool isLess = (str1 < str2);        // true (lexicographical)
bool isGreater = (str2 > str1);     // true
```

## 🔗 Logical Operators

| Operator | Name | Description | Example |
|----------|------|-------------|---------|
| `&&` | Logical AND | Returns true if both operands are true | `a && b` |
| `\|\|` | Logical OR | Returns true if at least one operand is true | `a \|\| b` |
| `!` | Logical NOT | Returns opposite boolean value | `!a` |

```cpp
#include <iostream>
using namespace std;

int main() {
    bool a = true, b = false, c = true;
    
    cout << "a = " << a << ", b = " << b << ", c = " << c << endl;
    
    // Logical AND
    cout << "a && b: " << (a && b) << endl;  // false
    cout << "a && c: " << (a && c) << endl;  // true
    
    // Logical OR
    cout << "a || b: " << (a || b) << endl;  // true
    cout << "b || false: " << (b || false) << endl;  // false
    
    // Logical NOT
    cout << "!a: " << (!a) << endl;  // false
    cout << "!b: " << (!b) << endl;  // true
    
    // Short-circuit evaluation
    int x = 0;
    if (b && (++x > 0)) {  // ++x is not executed because b is false
        cout << "This won't print" << endl;
    }
    cout << "x = " << x << endl;  // x is still 0
    
    return 0;
}
```

### Practical Examples
```cpp
// Validating user input
int age;
cout << "Enter age: ";
cin >> age;

if (age >= 18 && age <= 65) {
    cout << "Eligible for job" << endl;
}

// Multiple conditions
bool isStudent = true;
bool hasID = true;
double gpa = 3.5;

if ((isStudent && hasID) && (gpa >= 3.0)) {
    cout << "Eligible for scholarship" << endl;
}
```

## 🔢 Bitwise Operators

| Operator | Name | Description | Example |
|----------|------|-------------|---------|
| `&` | Bitwise AND | AND operation on each bit | `a & b` |
| `\|` | Bitwise OR | OR operation on each bit | `a \| b` |
| `^` | Bitwise XOR | XOR operation on each bit | `a ^ b` |
| `~` | Bitwise NOT | Complement (invert) all bits | `~a` |
| `<<` | Left shift | Shift bits left | `a << 2` |
| `>>` | Right shift | Shift bits right | `a >> 2` |

```cpp
#include <iostream>
#include <bitset>
using namespace std;

int main() {
    unsigned int a = 12;  // Binary: 1100
    unsigned int b = 10;  // Binary: 1010
    
    cout << "a = " << bitset<8>(a) << " (" << a << ")" << endl;
    cout << "b = " << bitset<8>(b) << " (" << b << ")" << endl;
    
    // Bitwise AND
    unsigned int and_result = a & b;  // 1000 = 8
    cout << "a & b = " << bitset<8>(and_result) << " (" << and_result << ")" << endl;
    
    // Bitwise OR
    unsigned int or_result = a | b;   // 1110 = 14
    cout << "a | b = " << bitset<8>(or_result) << " (" << or_result << ")" << endl;
    
    // Bitwise XOR
    unsigned int xor_result = a ^ b;  // 0110 = 6
    cout << "a ^ b = " << bitset<8>(xor_result) << " (" << xor_result << ")" << endl;
    
    // Bitwise NOT
    unsigned int not_result = ~a;     // Inverts all bits
    cout << "~a = " << bitset<8>(not_result) << " (" << not_result << ")" << endl;
    
    // Left shift
    unsigned int left_shift = a << 2;  // 110000 = 48
    cout << "a << 2 = " << bitset<8>(left_shift) << " (" << left_shift << ")" << endl;
    
    // Right shift
    unsigned int right_shift = a >> 2; // 11 = 3
    cout << "a >> 2 = " << bitset<8>(right_shift) << " (" << right_shift << ")" << endl;
    
    return 0;
}
```

### Practical Applications of Bitwise Operators
```cpp
// Setting, clearing, and toggling bits
unsigned int flags = 0;

// Set bit at position 2
flags |= (1 << 2);  // flags = 00000100

// Clear bit at position 2
flags &= ~(1 << 2); // flags = 00000000

// Toggle bit at position 2
flags ^= (1 << 2);  // flags = 00000100

// Check if bit at position 2 is set
bool isBitSet = (flags & (1 << 2)) != 0;

// Fast multiplication and division by powers of 2
int x = 16;
int multiply_by_4 = x << 2;  // x * 4 = 64
int divide_by_8 = x >> 3;    // x / 8 = 2
```

## 🎯 Conditional (Ternary) Operator

```cpp
// Syntax: condition ? value_if_true : value_if_false
int a = 10, b = 20;
int max = (a > b) ? a : b;  // max = 20

// More examples
string result = (age >= 18) ? "Adult" : "Minor";

// Nested ternary operators (use sparingly)
string grade = (score >= 90) ? "A" : 
               (score >= 80) ? "B" : 
               (score >= 70) ? "C" : 
               (score >= 60) ? "D" : "F";

// Ternary vs if-else
// Ternary (good for simple assignments)
int absValue = (x < 0) ? -x : x;

// If-else (better for complex operations)
if (x < 0) {
    absValue = -x;
    cout << "Number was negative" << endl;
} else {
    absValue = x;
    cout << "Number was positive" << endl;
}
```

## 📐 sizeof Operator

```cpp
#include <iostream>
using namespace std;

int main() {
    int arr[10];
    double d = 3.14;
    
    cout << "sizeof(int): " << sizeof(int) << " bytes" << endl;
    cout << "sizeof(double): " << sizeof(double) << " bytes" << endl;
    cout << "sizeof(arr): " << sizeof(arr) << " bytes" << endl;
    cout << "sizeof(d): " << sizeof(d) << " bytes" << endl;
    
    // Number of elements in array
    int arraySize = sizeof(arr) / sizeof(arr[0]);
    cout << "Array has " << arraySize << " elements" << endl;
    
    return 0;
}
```

## 🎪 Other Operators

### 1. **Comma Operator**
```cpp
int a, b, c;
a = (b = 3, c = 4, b + c);  // a = 7, expressions evaluated left to right

// More practical use in for loops
for (int i = 0, j = 10; i < j; i++, j--) {
    cout << i << " " << j << endl;
}
```

### 2. **Scope Resolution Operator (::)**
```cpp
int x = 10;  // Global variable

int main() {
    int x = 20;  // Local variable
    
    cout << "Local x: " << x << endl;     // 20
    cout << "Global x: " << ::x << endl;  // 10
    
    return 0;
}

// With namespaces
std::cout << "Hello" << std::endl;

// With classes
class MyClass {
public:
    static int value;
};
int MyClass::value = 100;  // Definition
```

### 3. **Member Access Operators**
```cpp
struct Point {
    int x, y;
    void display() { cout << "(" << x << ", " << y << ")" << endl; }
};

Point p1 = {10, 20};
Point* ptr = &p1;

// Dot operator for objects
p1.x = 30;
p1.display();

// Arrow operator for pointers
ptr->y = 40;
ptr->display();

// Equivalent to arrow operator
(*ptr).x = 50;
```

### 4. **Subscript Operator**
```cpp
int arr[5] = {1, 2, 3, 4, 5};
int value = arr[2];  // value = 3

// Equivalent pointer arithmetic
int sameValue = *(arr + 2);  // Also equals 3

string str = "Hello";
char firstChar = str[0];  // 'H'
```

## 📊 Operator Precedence and Associativity

### Precedence Table (High to Low)
| Precedence | Operators | Associativity |
|------------|-----------|---------------|
| 1 | `::` | Left to right |
| 2 | `() [] -> . ++ --` (postfix) | Left to right |
| 3 | `++ -- + - ! ~ sizeof` (prefix) | Right to left |
| 4 | `* / %` | Left to right |
| 5 | `+ -` | Left to right |
| 6 | `<< >>` | Left to right |
| 7 | `< <= > >=` | Left to right |
| 8 | `== !=` | Left to right |
| 9 | `&` | Left to right |
| 10 | `^` | Left to right |
| 11 | `\|` | Left to right |
| 12 | `&&` | Left to right |
| 13 | `\|\|` | Left to right |
| 14 | `?:` | Right to left |
| 15 | `= += -= *= /= %=` etc. | Right to left |
| 16 | `,` | Left to right |

### Examples with Parentheses
```cpp
int result;

// Without parentheses (follows precedence)
result = 2 + 3 * 4;      // result = 14 (3*4 first, then +2)

// With parentheses (overrides precedence)
result = (2 + 3) * 4;    // result = 20 (2+3 first, then *4)

// Complex expression
result = 2 + 3 * 4 > 10 && 5 < 8;  // result = 1 (true)
// Evaluation: ((2 + (3 * 4)) > 10) && (5 < 8)
//            = (14 > 10) && (5 < 8)
//            = true && true = true
```

## ⚡ Short-Circuit Evaluation

```cpp
#include <iostream>
using namespace std;

bool expensive_function() {
    cout << "Expensive function called!" << endl;
    return true;
}

int main() {
    bool flag = false;
    
    // Logical AND - second expression not evaluated if first is false
    if (flag && expensive_function()) {
        cout << "Both conditions true" << endl;
    }
    // expensive_function() is NOT called because flag is false
    
    flag = true;
    // Logical OR - second expression not evaluated if first is true
    if (flag || expensive_function()) {
        cout << "At least one condition true" << endl;
    }
    // expensive_function() is NOT called because flag is true
    
    return 0;
}
```

## 💡 Best Practices

### 1. **Use Parentheses for Clarity**
```cpp
// Unclear precedence
if (a == b && c == d || e == f) { /* ... */ }

// Clear with parentheses
if ((a == b && c == d) || (e == f)) { /* ... */ }
```

### 2. **Prefer Prefix Increment/Decrement**
```cpp
// For simple variables, both are equivalent
int i = 5;
++i;  // Preferred style
i++;  // Also fine

// For complex objects (like iterators), prefix is more efficient
for (auto it = vec.begin(); it != vec.end(); ++it) {
    // ++it is preferred over it++
}
```

### 3. **Be Careful with Floating-Point Comparisons**
```cpp
double a = 0.1 + 0.2;
double b = 0.3;

// DON'T do this
if (a == b) { /* May not work as expected */ }

// DO this instead
const double EPSILON = 1e-9;
if (abs(a - b) < EPSILON) {
    cout << "Numbers are equal" << endl;
}
```

### 4. **Use Appropriate Operators for Different Types**
```cpp
// For booleans
bool isReady = true;
if (isReady) { /* ... */ }           // Good
if (isReady == true) { /* ... */ }   // Redundant

// For pointers
int* ptr = nullptr;
if (ptr) { /* ... */ }               // Good
if (ptr != nullptr) { /* ... */ }    // Also clear and explicit
```

## 🔗 Related Topics
- [Data Types & Variables](./02-data-types-variables.md)
- [Control Structures](./04-control-structures.md)
- [Operator Overloading](./11-operator-overloading.md)

---
*Previous: [Data Types & Variables](./02-data-types-variables.md) | Next: [Control Structures](./04-control-structures.md)*
