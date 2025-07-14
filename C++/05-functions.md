# 05. Functions

## 📋 Overview
Functions are blocks of code that perform specific tasks. They help in code organization, reusability, and modularity. C++ supports both built-in functions and user-defined functions.

## 🏗️ Function Basics

### Function Definition Syntax
```cpp
return_type function_name(parameter_list) {
    // Function body
    // Statements
    return value; // if return_type is not void
}
```

### Simple Function Example
```cpp
#include <iostream>
using namespace std;

// Function to add two numbers
int add(int a, int b) {
    int sum = a + b;
    return sum;
}

int main() {
    int x = 5, y = 3;
    int result = add(x, y);  // Function call
    cout << "Sum: " << result << endl;
    
    return 0;
}
```

### Function Declaration (Prototype)
```cpp
#include <iostream>
using namespace std;

// Function declarations (prototypes)
int add(int a, int b);
void greet(string name);
double calculateArea(double radius);

int main() {
    // Function calls
    int sum = add(10, 20);
    greet("Alice");
    double area = calculateArea(5.0);
    
    cout << "Sum: " << sum << endl;
    cout << "Area: " << area << endl;
    
    return 0;
}

// Function definitions
int add(int a, int b) {
    return a + b;
}

void greet(string name) {
    cout << "Hello, " << name << "!" << endl;
}

double calculateArea(double radius) {
    const double PI = 3.14159;
    return PI * radius * radius;
}
```

## 📥 Function Parameters

### 1. **Pass by Value**
```cpp
#include <iostream>
using namespace std;

void increment(int x) {
    x++;  // Only modifies the local copy
    cout << "Inside function: " << x << endl;
}

int main() {
    int num = 5;
    cout << "Before function call: " << num << endl;
    increment(num);
    cout << "After function call: " << num << endl;  // num is still 5
    
    return 0;
}
```

### 2. **Pass by Reference**
```cpp
void incrementByReference(int& x) {
    x++;  // Modifies the original variable
    cout << "Inside function: " << x << endl;
}

int main() {
    int num = 5;
    cout << "Before function call: " << num << endl;
    incrementByReference(num);
    cout << "After function call: " << num << endl;  // num is now 6
    
    return 0;
}
```

### 3. **Pass by Pointer**
```cpp
void incrementByPointer(int* x) {
    (*x)++;  // Modifies the value at the address
    cout << "Inside function: " << *x << endl;
}

int main() {
    int num = 5;
    cout << "Before function call: " << num << endl;
    incrementByPointer(&num);  // Pass address of num
    cout << "After function call: " << num << endl;  // num is now 6
    
    return 0;
}
```

### 4. **const Parameters**
```cpp
// Function that doesn't modify the parameter
void printArray(const int arr[], int size) {
    for (int i = 0; i < size; i++) {
        cout << arr[i] << " ";
        // arr[i] = 0;  // Error: cannot modify const parameter
    }
    cout << endl;
}

// Function with const reference (efficient for large objects)
void displayString(const string& str) {
    cout << "String: " << str << endl;
    // str[0] = 'X';  // Error: cannot modify const reference
}
```

## 🎯 Default Parameters

```cpp
#include <iostream>
using namespace std;

// Function with default parameters
void printInfo(string name, int age = 25, string city = "Unknown") {
    cout << "Name: " << name << endl;
    cout << "Age: " << age << endl;
    cout << "City: " << city << endl;
    cout << "---" << endl;
}

int main() {
    printInfo("Alice");                    // Uses default age and city
    printInfo("Bob", 30);                  // Uses default city
    printInfo("Charlie", 28, "New York");  // No defaults used
    
    return 0;
}
```

### Rules for Default Parameters
```cpp
// Good: Default parameters at the end
void function1(int a, int b = 10, int c = 20);

// Bad: Default parameter before non-default
// void function2(int a = 5, int b);  // Error

// Good: All parameters after first default must have defaults
void function3(int a, int b = 10, int c = 20, int d = 30);

// Function call examples
function1(5);        // a=5, b=10, c=20
function1(5, 15);    // a=5, b=15, c=20
function1(5, 15, 25); // a=5, b=15, c=25
```

## 🔄 Function Overloading

```cpp
#include <iostream>
using namespace std;

// Function overloading - same name, different parameters
int add(int a, int b) {
    cout << "Adding two integers" << endl;
    return a + b;
}

double add(double a, double b) {
    cout << "Adding two doubles" << endl;
    return a + b;
}

int add(int a, int b, int c) {
    cout << "Adding three integers" << endl;
    return a + b + c;
}

string add(string a, string b) {
    cout << "Concatenating two strings" << endl;
    return a + b;
}

int main() {
    cout << add(5, 3) << endl;           // Calls int version
    cout << add(5.5, 3.2) << endl;      // Calls double version
    cout << add(1, 2, 3) << endl;       // Calls three-parameter version
    cout << add("Hello, ", "World!") << endl; // Calls string version
    
    return 0;
}
```

### Function Overloading Rules
```cpp
// Valid overloads - different parameter types
void print(int x);
void print(double x);
void print(string x);

// Valid overloads - different number of parameters
void calculate(int a);
void calculate(int a, int b);
void calculate(int a, int b, int c);

// Valid overloads - different parameter order
void process(int a, double b);
void process(double a, int b);

// Invalid overload - only return type differs
// int getValue();
// double getValue();  // Error: redefinition

// Invalid overload - only parameter names differ
// void func(int x);
// void func(int y);  // Error: redefinition
```

## 🔁 Recursion

### Basic Recursion
```cpp
#include <iostream>
using namespace std;

// Factorial using recursion
long long factorial(int n) {
    // Base case
    if (n <= 1) {
        return 1;
    }
    // Recursive case
    return n * factorial(n - 1);
}

int main() {
    int num = 5;
    cout << "Factorial of " << num << " is " << factorial(num) << endl;
    
    return 0;
}
```

### More Recursion Examples
```cpp
// Fibonacci sequence
int fibonacci(int n) {
    if (n <= 1) {
        return n;
    }
    return fibonacci(n - 1) + fibonacci(n - 2);
}

// Sum of digits
int sumOfDigits(int n) {
    if (n == 0) {
        return 0;
    }
    return (n % 10) + sumOfDigits(n / 10);
}

// Power function
double power(double base, int exponent) {
    if (exponent == 0) {
        return 1;
    }
    if (exponent < 0) {
        return 1.0 / power(base, -exponent);
    }
    return base * power(base, exponent - 1);
}

// Binary search (recursive)
int binarySearch(int arr[], int left, int right, int target) {
    if (left > right) {
        return -1;  // Not found
    }
    
    int mid = left + (right - left) / 2;
    
    if (arr[mid] == target) {
        return mid;
    }
    else if (arr[mid] > target) {
        return binarySearch(arr, left, mid - 1, target);
    }
    else {
        return binarySearch(arr, mid + 1, right, target);
    }
}
```

### Tail Recursion
```cpp
// Non-tail recursive factorial
int factorial_normal(int n) {
    if (n <= 1) return 1;
    return n * factorial_normal(n - 1);  // Operation after recursive call
}

// Tail recursive factorial
int factorial_tail(int n, int accumulator = 1) {
    if (n <= 1) return accumulator;
    return factorial_tail(n - 1, n * accumulator);  // No operation after call
}
```

## 🎪 Advanced Function Features

### 1. **inline Functions**
```cpp
#include <iostream>
using namespace std;

// inline function - suggestion to compiler for optimization
inline int square(int x) {
    return x * x;
}

inline int max(int a, int b) {
    return (a > b) ? a : b;
}

int main() {
    cout << "Square of 5: " << square(5) << endl;
    cout << "Max of 10 and 20: " << max(10, 20) << endl;
    
    return 0;
}
```

### 2. **Function Pointers**
```cpp
#include <iostream>
using namespace std;

// Functions to be pointed to
int add(int a, int b) { return a + b; }
int subtract(int a, int b) { return a - b; }
int multiply(int a, int b) { return a * b; }

int main() {
    // Declare function pointer
    int (*operation)(int, int);
    
    // Point to different functions
    operation = add;
    cout << "Addition: " << operation(10, 5) << endl;
    
    operation = subtract;
    cout << "Subtraction: " << operation(10, 5) << endl;
    
    operation = multiply;
    cout << "Multiplication: " << operation(10, 5) << endl;
    
    // Array of function pointers
    int (*operations[])(int, int) = {add, subtract, multiply};
    string names[] = {"Add", "Subtract", "Multiply"};
    
    for (int i = 0; i < 3; i++) {
        cout << names[i] << ": " << operations[i](8, 4) << endl;
    }
    
    return 0;
}
```

### 3. **Lambda Functions (C++11)**
```cpp
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int main() {
    // Basic lambda
    auto greet = []() {
        cout << "Hello from lambda!" << endl;
    };
    greet();
    
    // Lambda with parameters
    auto add = [](int a, int b) {
        return a + b;
    };
    cout << "Sum: " << add(5, 3) << endl;
    
    // Lambda with capture
    int multiplier = 10;
    auto multiply = [multiplier](int x) {
        return x * multiplier;
    };
    cout << "5 * 10 = " << multiply(5) << endl;
    
    // Lambda with capture by reference
    int counter = 0;
    auto increment = [&counter]() {
        counter++;
    };
    increment();
    increment();
    cout << "Counter: " << counter << endl;
    
    // Using lambda with STL algorithms
    vector<int> numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    // Count even numbers
    int evenCount = count_if(numbers.begin(), numbers.end(), 
                            [](int n) { return n % 2 == 0; });
    cout << "Even numbers count: " << evenCount << endl;
    
    return 0;
}
```

## 📚 Function Examples and Utilities

### 1. **Mathematical Functions**
```cpp
#include <iostream>
#include <cmath>
using namespace std;

// Check if number is prime
bool isPrime(int n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    
    for (int i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) {
            return false;
        }
    }
    return true;
}

// Greatest Common Divisor
int gcd(int a, int b) {
    if (b == 0) return a;
    return gcd(b, a % b);
}

// Least Common Multiple
int lcm(int a, int b) {
    return (a * b) / gcd(a, b);
}

// Check if number is perfect
bool isPerfectNumber(int n) {
    int sum = 1;  // 1 is always a divisor
    for (int i = 2; i * i <= n; i++) {
        if (n % i == 0) {
            sum += i;
            if (i != n / i) {  // Avoid counting square root twice
                sum += n / i;
            }
        }
    }
    return sum == n && n > 1;
}
```

### 2. **String Functions**
```cpp
#include <iostream>
#include <string>
#include <algorithm>
using namespace std;

// Convert string to uppercase
string toUpper(string str) {
    transform(str.begin(), str.end(), str.begin(), ::toupper);
    return str;
}

// Convert string to lowercase
string toLower(string str) {
    transform(str.begin(), str.end(), str.begin(), ::tolower);
    return str;
}

// Reverse a string
string reverseString(string str) {
    reverse(str.begin(), str.end());
    return str;
}

// Check if string is palindrome
bool isPalindrome(string str) {
    string reversed = reverseString(toLower(str));
    return toLower(str) == reversed;
}

// Count words in string
int countWords(const string& str) {
    int count = 0;
    bool inWord = false;
    
    for (char c : str) {
        if (c != ' ' && c != '\t' && c != '\n') {
            if (!inWord) {
                count++;
                inWord = true;
            }
        } else {
            inWord = false;
        }
    }
    return count;
}
```

### 3. **Array Functions**
```cpp
#include <iostream>
using namespace std;

// Print array
void printArray(const int arr[], int size) {
    for (int i = 0; i < size; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;
}

// Find maximum element
int findMax(const int arr[], int size) {
    int max = arr[0];
    for (int i = 1; i < size; i++) {
        if (arr[i] > max) {
            max = arr[i];
        }
    }
    return max;
}

// Find minimum element
int findMin(const int arr[], int size) {
    int min = arr[0];
    for (int i = 1; i < size; i++) {
        if (arr[i] < min) {
            min = arr[i];
        }
    }
    return min;
}

// Calculate average
double calculateAverage(const int arr[], int size) {
    double sum = 0;
    for (int i = 0; i < size; i++) {
        sum += arr[i];
    }
    return sum / size;
}

// Linear search
int linearSearch(const int arr[], int size, int target) {
    for (int i = 0; i < size; i++) {
        if (arr[i] == target) {
            return i;  // Return index if found
        }
    }
    return -1;  // Return -1 if not found
}

// Bubble sort
void bubbleSort(int arr[], int size) {
    for (int i = 0; i < size - 1; i++) {
        for (int j = 0; j < size - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                // Swap elements
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}
```

### 4. **Utility Functions**
```cpp
#include <iostream>
#include <cstdlib>
#include <ctime>
using namespace std;

// Generate random number in range
int randomRange(int min, int max) {
    return min + rand() % (max - min + 1);
}

// Swap two values
template<typename T>
void swapValues(T& a, T& b) {
    T temp = a;
    a = b;
    b = temp;
}

// Convert Celsius to Fahrenheit
double celsiusToFahrenheit(double celsius) {
    return (celsius * 9.0 / 5.0) + 32.0;
}

// Convert Fahrenheit to Celsius
double fahrenheitToCelsius(double fahrenheit) {
    return (fahrenheit - 32.0) * 5.0 / 9.0;
}

// Validate input range
bool isInRange(int value, int min, int max) {
    return value >= min && value <= max;
}

// Get valid input from user
int getValidInput(int min, int max) {
    int input;
    do {
        cout << "Enter a number between " << min << " and " << max << ": ";
        cin >> input;
        if (!isInRange(input, min, max)) {
            cout << "Invalid input! Please try again." << endl;
        }
    } while (!isInRange(input, min, max));
    
    return input;
}
```

## 💡 Best Practices

### 1. **Function Design Principles**
```cpp
// Good: Single responsibility
double calculateCircleArea(double radius) {
    const double PI = 3.14159;
    return PI * radius * radius;
}

// Good: Descriptive naming
bool isEligibleForVoting(int age) {
    return age >= 18;
}

// Good: Use const for parameters that shouldn't change
void displayMessage(const string& message) {
    cout << message << endl;
}
```

### 2. **Error Handling**
```cpp
#include <stdexcept>
using namespace std;

// Function with error checking
double divide(double a, double b) {
    if (b == 0) {
        throw invalid_argument("Division by zero");
    }
    return a / b;
}

// Function with return code
bool safeDivide(double a, double b, double& result) {
    if (b == 0) {
        return false;  // Indicate error
    }
    result = a / b;
    return true;  // Indicate success
}
```

### 3. **Function Documentation**
```cpp
/**
 * @brief Calculates the factorial of a number
 * @param n The number to calculate factorial for (must be non-negative)
 * @return The factorial of n, or -1 if n is negative
 */
long long factorial(int n) {
    if (n < 0) return -1;  // Error case
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}
```

### 4. **Function Organization**
```cpp
// Group related functions
namespace MathUtils {
    double add(double a, double b) { return a + b; }
    double subtract(double a, double b) { return a - b; }
    double multiply(double a, double b) { return a * b; }
    double divide(double a, double b) { return a / b; }
}

namespace StringUtils {
    string trim(const string& str) { /* implementation */ }
    string toUpper(const string& str) { /* implementation */ }
    string toLower(const string& str) { /* implementation */ }
}
```

## 🔗 Related Topics
- [Control Structures](./04-control-structures.md)
- [Arrays & Strings](./06-arrays-strings.md)
- [Pointers & References](./07-pointers-references.md)
- [Templates](./12-templates.md)

---
*Previous: [Control Structures](./04-control-structures.md) | Next: [Arrays & Strings](./06-arrays-strings.md)*
