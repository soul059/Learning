# 11. Operator Overloading

## 📋 Overview
Operator overloading allows you to define custom behavior for operators when applied to user-defined types. It enables objects to be used with standard operators like `+`, `-`, `*`, `==`, etc., making your classes more intuitive and natural to use.

## 🎯 Basic Operator Overloading

### 1. **Arithmetic Operators**
```cpp
#include <iostream>
using namespace std;

class Complex {
private:
    double real, imag;

public:
    // Constructors
    Complex(double r = 0, double i = 0) : real(r), imag(i) {}
    
    // Addition operator
    Complex operator+(const Complex& other) const {
        return Complex(real + other.real, imag + other.imag);
    }
    
    // Subtraction operator
    Complex operator-(const Complex& other) const {
        return Complex(real - other.real, imag - other.imag);
    }
    
    // Multiplication operator
    Complex operator*(const Complex& other) const {
        return Complex(
            real * other.real - imag * other.imag,
            real * other.imag + imag * other.real
        );
    }
    
    // Division operator
    Complex operator/(const Complex& other) const {
        double denominator = other.real * other.real + other.imag * other.imag;
        if (denominator == 0) {
            throw runtime_error("Division by zero");
        }
        return Complex(
            (real * other.real + imag * other.imag) / denominator,
            (imag * other.real - real * other.imag) / denominator
        );
    }
    
    // Unary minus operator
    Complex operator-() const {
        return Complex(-real, -imag);
    }
    
    // Compound assignment operators
    Complex& operator+=(const Complex& other) {
        real += other.real;
        imag += other.imag;
        return *this;
    }
    
    Complex& operator-=(const Complex& other) {
        real -= other.real;
        imag -= other.imag;
        return *this;
    }
    
    // Display function
    void display() const {
        cout << real;
        if (imag >= 0) cout << " + " << imag << "i";
        else cout << " - " << (-imag) << "i";
    }
    
    // Getters
    double getReal() const { return real; }
    double getImag() const { return imag; }
};

int main() {
    Complex c1(3, 4);
    Complex c2(1, 2);
    
    cout << "c1 = "; c1.display(); cout << endl;
    cout << "c2 = "; c2.display(); cout << endl;
    
    Complex sum = c1 + c2;
    cout << "c1 + c2 = "; sum.display(); cout << endl;
    
    Complex diff = c1 - c2;
    cout << "c1 - c2 = "; diff.display(); cout << endl;
    
    Complex product = c1 * c2;
    cout << "c1 * c2 = "; product.display(); cout << endl;
    
    Complex quotient = c1 / c2;
    cout << "c1 / c2 = "; quotient.display(); cout << endl;
    
    Complex neg = -c1;
    cout << "-c1 = "; neg.display(); cout << endl;
    
    c1 += c2;
    cout << "After c1 += c2, c1 = "; c1.display(); cout << endl;
    
    return 0;
}
```

### 2. **Comparison Operators**
```cpp
#include <iostream>
#include <cmath>
using namespace std;

class Point {
private:
    double x, y;

public:
    Point(double x = 0, double y = 0) : x(x), y(y) {}
    
    // Equality operator
    bool operator==(const Point& other) const {
        const double epsilon = 1e-9;
        return abs(x - other.x) < epsilon && abs(y - other.y) < epsilon;
    }
    
    // Inequality operator
    bool operator!=(const Point& other) const {
        return !(*this == other);
    }
    
    // Less than operator (for sorting)
    bool operator<(const Point& other) const {
        if (x != other.x) return x < other.x;
        return y < other.y;
    }
    
    // Less than or equal
    bool operator<=(const Point& other) const {
        return (*this < other) || (*this == other);
    }
    
    // Greater than operator
    bool operator>(const Point& other) const {
        return !(*this <= other);
    }
    
    // Greater than or equal
    bool operator>=(const Point& other) const {
        return !(*this < other);
    }
    
    // Distance from origin (for comparison)
    double distanceFromOrigin() const {
        return sqrt(x * x + y * y);
    }
    
    void display() const {
        cout << "(" << x << ", " << y << ")";
    }
    
    double getX() const { return x; }
    double getY() const { return y; }
};

int main() {
    Point p1(3, 4);
    Point p2(3, 4);
    Point p3(1, 2);
    Point p4(5, 6);
    
    cout << "p1 = "; p1.display(); cout << endl;
    cout << "p2 = "; p2.display(); cout << endl;
    cout << "p3 = "; p3.display(); cout << endl;
    cout << "p4 = "; p4.display(); cout << endl;
    
    cout << "\nComparison results:" << endl;
    cout << "p1 == p2: " << (p1 == p2) << endl;
    cout << "p1 != p3: " << (p1 != p3) << endl;
    cout << "p3 < p1: " << (p3 < p1) << endl;
    cout << "p1 < p4: " << (p1 < p4) << endl;
    cout << "p4 > p1: " << (p4 > p1) << endl;
    
    return 0;
}
```

## 🔄 Stream Operators

### Input and Output Stream Operators
```cpp
#include <iostream>
#include <string>
#include <sstream>
using namespace std;

class Fraction {
private:
    int numerator, denominator;
    
    // Helper function to find GCD
    int gcd(int a, int b) const {
        return b == 0 ? a : gcd(b, a % b);
    }
    
    // Simplify the fraction
    void simplify() {
        if (denominator == 0) {
            throw invalid_argument("Denominator cannot be zero");
        }
        
        if (denominator < 0) {
            numerator = -numerator;
            denominator = -denominator;
        }
        
        int g = gcd(abs(numerator), abs(denominator));
        numerator /= g;
        denominator /= g;
    }

public:
    Fraction(int num = 0, int den = 1) : numerator(num), denominator(den) {
        simplify();
    }
    
    // Arithmetic operators
    Fraction operator+(const Fraction& other) const {
        return Fraction(
            numerator * other.denominator + other.numerator * denominator,
            denominator * other.denominator
        );
    }
    
    Fraction operator-(const Fraction& other) const {
        return Fraction(
            numerator * other.denominator - other.numerator * denominator,
            denominator * other.denominator
        );
    }
    
    Fraction operator*(const Fraction& other) const {
        return Fraction(
            numerator * other.numerator,
            denominator * other.denominator
        );
    }
    
    Fraction operator/(const Fraction& other) const {
        return Fraction(
            numerator * other.denominator,
            denominator * other.numerator
        );
    }
    
    // Comparison operators
    bool operator==(const Fraction& other) const {
        return numerator == other.numerator && denominator == other.denominator;
    }
    
    bool operator<(const Fraction& other) const {
        return numerator * other.denominator < other.numerator * denominator;
    }
    
    // Output stream operator (friend function)
    friend ostream& operator<<(ostream& os, const Fraction& f) {
        if (f.denominator == 1) {
            os << f.numerator;
        } else {
            os << f.numerator << "/" << f.denominator;
        }
        return os;
    }
    
    // Input stream operator (friend function)
    friend istream& operator>>(istream& is, Fraction& f) {
        string input;
        is >> input;
        
        size_t slashPos = input.find('/');
        if (slashPos != string::npos) {
            f.numerator = stoi(input.substr(0, slashPos));
            f.denominator = stoi(input.substr(slashPos + 1));
        } else {
            f.numerator = stoi(input);
            f.denominator = 1;
        }
        
        f.simplify();
        return is;
    }
    
    double toDecimal() const {
        return static_cast<double>(numerator) / denominator;
    }
};

int main() {
    Fraction f1(3, 4);
    Fraction f2(1, 2);
    
    cout << "f1 = " << f1 << endl;
    cout << "f2 = " << f2 << endl;
    
    cout << "f1 + f2 = " << (f1 + f2) << endl;
    cout << "f1 - f2 = " << (f1 - f2) << endl;
    cout << "f1 * f2 = " << (f1 * f2) << endl;
    cout << "f1 / f2 = " << (f1 / f2) << endl;
    
    cout << "f1 as decimal: " << f1.toDecimal() << endl;
    
    cout << "\nEnter a fraction (e.g., 3/4 or 5): ";
    Fraction userFraction;
    cin >> userFraction;
    cout << "You entered: " << userFraction << endl;
    cout << "As decimal: " << userFraction.toDecimal() << endl;
    
    return 0;
}
```

## 🔢 Increment and Decrement Operators

### Pre/Post Increment and Decrement
```cpp
#include <iostream>
using namespace std;

class Counter {
private:
    int value;

public:
    Counter(int v = 0) : value(v) {}
    
    // Pre-increment operator (++counter)
    Counter& operator++() {
        ++value;
        return *this;
    }
    
    // Post-increment operator (counter++)
    Counter operator++(int) {
        Counter temp(*this);  // Save current state
        ++value;              // Increment
        return temp;          // Return old state
    }
    
    // Pre-decrement operator (--counter)
    Counter& operator--() {
        --value;
        return *this;
    }
    
    // Post-decrement operator (counter--)
    Counter operator--(int) {
        Counter temp(*this);  // Save current state
        --value;              // Decrement
        return temp;          // Return old state
    }
    
    // Addition operator
    Counter operator+(const Counter& other) const {
        return Counter(value + other.value);
    }
    
    // Compound assignment
    Counter& operator+=(int n) {
        value += n;
        return *this;
    }
    
    // Output operator
    friend ostream& operator<<(ostream& os, const Counter& c) {
        os << c.value;
        return os;
    }
    
    int getValue() const { return value; }
};

int main() {
    Counter c1(5);
    Counter c2(10);
    
    cout << "Initial values:" << endl;
    cout << "c1 = " << c1 << endl;
    cout << "c2 = " << c2 << endl;
    
    cout << "\nPre-increment:" << endl;
    cout << "++c1 = " << ++c1 << endl;
    cout << "c1 is now = " << c1 << endl;
    
    cout << "\nPost-increment:" << endl;
    cout << "c2++ = " << c2++ << endl;
    cout << "c2 is now = " << c2 << endl;
    
    cout << "\nPre-decrement:" << endl;
    cout << "--c1 = " << --c1 << endl;
    cout << "c1 is now = " << c1 << endl;
    
    cout << "\nPost-decrement:" << endl;
    cout << "c2-- = " << c2-- << endl;
    cout << "c2 is now = " << c2 << endl;
    
    // Demonstrate difference between pre and post
    Counter c3(0);
    cout << "\nDemonstrating pre vs post increment:" << endl;
    cout << "c3 = " << c3 << endl;
    cout << "++c3 + ++c3 = " << (++c3 + ++c3) << endl;  // 1 + 2 = 3
    
    Counter c4(0);
    cout << "c4 = " << c4 << endl;
    cout << "c4++ + c4++ = " << (c4++ + c4++) << endl;  // 0 + 1 = 1
    cout << "c4 is now = " << c4 << endl;
    
    return 0;
}
```

## 🎯 Subscript and Function Call Operators

### Array-like Access and Function Objects
```cpp
#include <iostream>
#include <vector>
#include <stdexcept>
using namespace std;

class Matrix {
private:
    vector<vector<int>> data;
    int rows, cols;

public:
    Matrix(int r, int c) : rows(r), cols(c) {
        data.resize(rows, vector<int>(cols, 0));
    }
    
    // Subscript operator for row access
    vector<int>& operator[](int row) {
        if (row < 0 || row >= rows) {
            throw out_of_range("Row index out of bounds");
        }
        return data[row];
    }
    
    // Const version of subscript operator
    const vector<int>& operator[](int row) const {
        if (row < 0 || row >= rows) {
            throw out_of_range("Row index out of bounds");
        }
        return data[row];
    }
    
    // Function call operator for matrix access
    int& operator()(int row, int col) {
        if (row < 0 || row >= rows || col < 0 || col >= cols) {
            throw out_of_range("Index out of bounds");
        }
        return data[row][col];
    }
    
    // Const version of function call operator
    const int& operator()(int row, int col) const {
        if (row < 0 || row >= rows || col < 0 || col >= cols) {
            throw out_of_range("Index out of bounds");
        }
        return data[row][col];
    }
    
    // Matrix addition
    Matrix operator+(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw invalid_argument("Matrix dimensions must match");
        }
        
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result(i, j) = (*this)(i, j) + other(i, j);
            }
        }
        return result;
    }
    
    // Matrix multiplication
    Matrix operator*(const Matrix& other) const {
        if (cols != other.rows) {
            throw invalid_argument("Invalid dimensions for matrix multiplication");
        }
        
        Matrix result(rows, other.cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < other.cols; j++) {
                for (int k = 0; k < cols; k++) {
                    result(i, j) += (*this)(i, k) * other(k, j);
                }
            }
        }
        return result;
    }
    
    void display() const {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                cout << (*this)(i, j) << "\t";
            }
            cout << endl;
        }
    }
    
    int getRows() const { return rows; }
    int getCols() const { return cols; }
};

// Function object for mathematical operations
class MathFunction {
private:
    string operation;

public:
    MathFunction(string op) : operation(op) {}
    
    // Function call operator - makes object callable
    double operator()(double x, double y) const {
        if (operation == "add") return x + y;
        if (operation == "multiply") return x * y;
        if (operation == "power") return pow(x, y);
        if (operation == "max") return max(x, y);
        if (operation == "min") return min(x, y);
        throw invalid_argument("Unknown operation");
    }
    
    // Overload for single argument
    double operator()(double x) const {
        if (operation == "square") return x * x;
        if (operation == "sqrt") return sqrt(x);
        if (operation == "abs") return abs(x);
        throw invalid_argument("Operation requires two arguments");
    }
};

int main() {
    // Matrix operations
    cout << "=== Matrix Operations ===" << endl;
    Matrix m1(2, 3);
    Matrix m2(2, 3);
    
    // Fill matrices using different access methods
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            m1(i, j) = i + j + 1;      // Using function call operator
            m2[i][j] = (i + 1) * (j + 1);  // Using subscript operator
        }
    }
    
    cout << "Matrix m1:" << endl;
    m1.display();
    
    cout << "\nMatrix m2:" << endl;
    m2.display();
    
    Matrix sum = m1 + m2;
    cout << "\nm1 + m2:" << endl;
    sum.display();
    
    // Function objects
    cout << "\n=== Function Objects ===" << endl;
    MathFunction adder("add");
    MathFunction multiplier("multiply");
    MathFunction squarer("square");
    
    cout << "adder(5, 3) = " << adder(5, 3) << endl;
    cout << "multiplier(4, 7) = " << multiplier(4, 7) << endl;
    cout << "squarer(6) = " << squarer(6) << endl;
    
    // Using function objects in algorithms (simulation)
    vector<double> numbers = {1, 2, 3, 4, 5};
    cout << "\nSquaring numbers: ";
    for (double num : numbers) {
        cout << squarer(num) << " ";
    }
    cout << endl;
    
    return 0;
}
```

## 🔧 Assignment and Type Conversion Operators

### Copy Assignment and Type Conversion
```cpp
#include <iostream>
#include <string>
#include <cstring>
using namespace std;

class String {
private:
    char* data;
    size_t length;
    
    void cleanup() {
        delete[] data;
        data = nullptr;
        length = 0;
    }

public:
    // Default constructor
    String() : data(nullptr), length(0) {
        data = new char[1];
        data[0] = '\0';
    }
    
    // Constructor from C-string
    String(const char* str) {
        length = strlen(str);
        data = new char[length + 1];
        strcpy(data, str);
    }
    
    // Constructor from std::string (type conversion)
    String(const string& str) {
        length = str.length();
        data = new char[length + 1];
        strcpy(data, str.c_str());
    }
    
    // Copy constructor
    String(const String& other) {
        length = other.length;
        data = new char[length + 1];
        strcpy(data, other.data);
    }
    
    // Copy assignment operator
    String& operator=(const String& other) {
        if (this != &other) {  // Self-assignment check
            cleanup();         // Clean up current data
            
            length = other.length;
            data = new char[length + 1];
            strcpy(data, other.data);
        }
        return *this;
    }
    
    // Assignment from C-string
    String& operator=(const char* str) {
        cleanup();
        length = strlen(str);
        data = new char[length + 1];
        strcpy(data, str);
        return *this;
    }
    
    // Assignment from std::string
    String& operator=(const string& str) {
        cleanup();
        length = str.length();
        data = new char[length + 1];
        strcpy(data, str.c_str());
        return *this;
    }
    
    // Destructor
    ~String() {
        cleanup();
    }
    
    // String concatenation
    String operator+(const String& other) const {
        String result;
        result.length = length + other.length;
        result.data = new char[result.length + 1];
        
        strcpy(result.data, data);
        strcat(result.data, other.data);
        
        return result;
    }
    
    String& operator+=(const String& other) {
        *this = *this + other;
        return *this;
    }
    
    // Subscript operator
    char& operator[](size_t index) {
        if (index >= length) {
            throw out_of_range("Index out of bounds");
        }
        return data[index];
    }
    
    const char& operator[](size_t index) const {
        if (index >= length) {
            throw out_of_range("Index out of bounds");
        }
        return data[index];
    }
    
    // Comparison operators
    bool operator==(const String& other) const {
        return strcmp(data, other.data) == 0;
    }
    
    bool operator<(const String& other) const {
        return strcmp(data, other.data) < 0;
    }
    
    // Type conversion operators
    operator const char*() const {
        return data;
    }
    
    operator string() const {
        return string(data);
    }
    
    operator bool() const {
        return length > 0;
    }
    
    // Stream operators
    friend ostream& operator<<(ostream& os, const String& str) {
        os << str.data;
        return os;
    }
    
    friend istream& operator>>(istream& is, String& str) {
        string temp;
        is >> temp;
        str = temp;
        return is;
    }
    
    size_t size() const { return length; }
    bool empty() const { return length == 0; }
    const char* c_str() const { return data; }
};

int main() {
    String s1("Hello");
    String s2(" World");
    String s3;
    
    cout << "s1 = " << s1 << endl;
    cout << "s2 = " << s2 << endl;
    
    // String concatenation
    s3 = s1 + s2;
    cout << "s3 = s1 + s2 = " << s3 << endl;
    
    // Assignment operators
    String s4;
    s4 = "C++ Programming";  // Assignment from C-string
    cout << "s4 = " << s4 << endl;
    
    string stdStr = "Standard String";
    String s5 = stdStr;     // Constructor from std::string
    cout << "s5 = " << s5 << endl;
    
    // Subscript operator
    cout << "First character of s1: " << s1[0] << endl;
    s1[0] = 'h';  // Modify character
    cout << "After changing first char: " << s1 << endl;
    
    // Comparison
    String s6("hello");
    cout << "s1 == s6: " << (s1 == s6) << endl;
    cout << "s1 < s4: " << (s1 < s4) << endl;
    
    // Type conversion
    const char* cStr = s1;  // Implicit conversion to const char*
    cout << "As C-string: " << cStr << endl;
    
    string stdString = s1;  // Implicit conversion to std::string
    cout << "As std::string: " << stdString << endl;
    
    // Boolean conversion
    String empty;
    cout << "s1 is " << (s1 ? "not empty" : "empty") << endl;
    cout << "empty is " << (empty ? "not empty" : "empty") << endl;
    
    // Compound assignment
    s1 += s2;
    cout << "After s1 += s2: " << s1 << endl;
    
    return 0;
}
```

## 🎪 Advanced Operator Overloading

### 1. **Smart Pointer Implementation**
```cpp
#include <iostream>
using namespace std;

template<typename T>
class SmartPtr {
private:
    T* ptr;

public:
    // Constructor
    explicit SmartPtr(T* p = nullptr) : ptr(p) {}
    
    // Destructor
    ~SmartPtr() {
        delete ptr;
    }
    
    // Copy constructor (transfer ownership)
    SmartPtr(SmartPtr& other) : ptr(other.ptr) {
        other.ptr = nullptr;  // Transfer ownership
    }
    
    // Assignment operator (transfer ownership)
    SmartPtr& operator=(SmartPtr& other) {
        if (this != &other) {
            delete ptr;           // Clean up current resource
            ptr = other.ptr;      // Transfer ownership
            other.ptr = nullptr;
        }
        return *this;
    }
    
    // Dereference operator
    T& operator*() const {
        if (!ptr) throw runtime_error("Dereferencing null pointer");
        return *ptr;
    }
    
    // Arrow operator
    T* operator->() const {
        if (!ptr) throw runtime_error("Accessing null pointer");
        return ptr;
    }
    
    // Boolean conversion
    operator bool() const {
        return ptr != nullptr;
    }
    
    // Get raw pointer
    T* get() const {
        return ptr;
    }
    
    // Release ownership
    T* release() {
        T* temp = ptr;
        ptr = nullptr;
        return temp;
    }
    
    // Reset with new pointer
    void reset(T* p = nullptr) {
        delete ptr;
        ptr = p;
    }
};

class Resource {
private:
    string name;
    int value;

public:
    Resource(string n, int v) : name(n), value(v) {
        cout << "Resource " << name << " created" << endl;
    }
    
    ~Resource() {
        cout << "Resource " << name << " destroyed" << endl;
    }
    
    void display() const {
        cout << "Resource: " << name << ", Value: " << value << endl;
    }
    
    void setValue(int v) { value = v; }
    int getValue() const { return value; }
    string getName() const { return name; }
};

int main() {
    cout << "=== Smart Pointer Demo ===" << endl;
    
    // Create smart pointer
    SmartPtr<Resource> ptr1(new Resource("Resource1", 100));
    
    // Use dereference operator
    (*ptr1).display();
    
    // Use arrow operator
    ptr1->setValue(200);
    ptr1->display();
    
    // Transfer ownership
    SmartPtr<Resource> ptr2 = ptr1;  // ptr1 becomes null
    
    if (ptr1) {
        cout << "ptr1 is valid" << endl;
    } else {
        cout << "ptr1 is null" << endl;
    }
    
    if (ptr2) {
        cout << "ptr2 is valid" << endl;
        ptr2->display();
    }
    
    // Create another resource
    SmartPtr<Resource> ptr3(new Resource("Resource2", 300));
    ptr3->display();
    
    return 0;
}
```

### 2. **Iterator Implementation**
```cpp
#include <iostream>
#include <vector>
using namespace std;

template<typename T>
class SimpleVector {
private:
    T* data;
    size_t capacity;
    size_t size_;

public:
    class Iterator {
    private:
        T* ptr;

    public:
        Iterator(T* p) : ptr(p) {}
        
        // Dereference operator
        T& operator*() const {
            return *ptr;
        }
        
        // Arrow operator
        T* operator->() const {
            return ptr;
        }
        
        // Pre-increment
        Iterator& operator++() {
            ++ptr;
            return *this;
        }
        
        // Post-increment
        Iterator operator++(int) {
            Iterator temp = *this;
            ++ptr;
            return temp;
        }
        
        // Pre-decrement
        Iterator& operator--() {
            --ptr;
            return *this;
        }
        
        // Post-decrement
        Iterator operator--(int) {
            Iterator temp = *this;
            --ptr;
            return temp;
        }
        
        // Addition operator
        Iterator operator+(int n) const {
            return Iterator(ptr + n);
        }
        
        // Subtraction operator
        Iterator operator-(int n) const {
            return Iterator(ptr - n);
        }
        
        // Difference operator
        ptrdiff_t operator-(const Iterator& other) const {
            return ptr - other.ptr;
        }
        
        // Compound assignment
        Iterator& operator+=(int n) {
            ptr += n;
            return *this;
        }
        
        Iterator& operator-=(int n) {
            ptr -= n;
            return *this;
        }
        
        // Subscript operator
        T& operator[](int index) const {
            return *(ptr + index);
        }
        
        // Comparison operators
        bool operator==(const Iterator& other) const {
            return ptr == other.ptr;
        }
        
        bool operator!=(const Iterator& other) const {
            return ptr != other.ptr;
        }
        
        bool operator<(const Iterator& other) const {
            return ptr < other.ptr;
        }
        
        bool operator<=(const Iterator& other) const {
            return ptr <= other.ptr;
        }
        
        bool operator>(const Iterator& other) const {
            return ptr > other.ptr;
        }
        
        bool operator>=(const Iterator& other) const {
            return ptr >= other.ptr;
        }
    };

    // Constructor
    SimpleVector(size_t cap = 10) : capacity(cap), size_(0) {
        data = new T[capacity];
    }
    
    // Destructor
    ~SimpleVector() {
        delete[] data;
    }
    
    // Add element
    void push_back(const T& value) {
        if (size_ >= capacity) {
            throw overflow_error("Vector is full");
        }
        data[size_++] = value;
    }
    
    // Subscript operator
    T& operator[](size_t index) {
        if (index >= size_) {
            throw out_of_range("Index out of bounds");
        }
        return data[index];
    }
    
    const T& operator[](size_t index) const {
        if (index >= size_) {
            throw out_of_range("Index out of bounds");
        }
        return data[index];
    }
    
    // Iterator methods
    Iterator begin() {
        return Iterator(data);
    }
    
    Iterator end() {
        return Iterator(data + size_);
    }
    
    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
};

int main() {
    cout << "=== Custom Iterator Demo ===" << endl;
    
    SimpleVector<int> vec;
    
    // Add some elements
    for (int i = 1; i <= 5; i++) {
        vec.push_back(i * 10);
    }
    
    cout << "Vector contents using subscript operator:" << endl;
    for (size_t i = 0; i < vec.size(); i++) {
        cout << "vec[" << i << "] = " << vec[i] << endl;
    }
    
    cout << "\nVector contents using iterator:" << endl;
    for (auto it = vec.begin(); it != vec.end(); ++it) {
        cout << *it << " ";
    }
    cout << endl;
    
    cout << "\nUsing range-based for loop:" << endl;
    for (auto value : vec) {
        cout << value << " ";
    }
    cout << endl;
    
    // Iterator arithmetic
    auto it = vec.begin();
    cout << "\nIterator arithmetic:" << endl;
    cout << "First element: " << *it << endl;
    cout << "Third element: " << *(it + 2) << endl;
    cout << "Last element: " << *(vec.end() - 1) << endl;
    
    // Modify through iterator
    cout << "\nModifying through iterator:" << endl;
    for (auto it = vec.begin(); it != vec.end(); ++it) {
        *it += 5;  // Add 5 to each element
    }
    
    for (auto value : vec) {
        cout << value << " ";
    }
    cout << endl;
    
    return 0;
}
```

## 💡 Best Practices and Guidelines

### 1. **Operator Overloading Rules**
```cpp
#include <iostream>
using namespace std;

class BestPractices {
private:
    int value;

public:
    BestPractices(int v = 0) : value(v) {}
    
    // 1. Return types should be consistent with built-in types
    // Binary operators should return by value
    BestPractices operator+(const BestPractices& other) const {
        return BestPractices(value + other.value);
    }
    
    // 2. Assignment operators should return reference to *this
    BestPractices& operator+=(const BestPractices& other) {
        value += other.value;
        return *this;
    }
    
    // 3. Prefix operators return reference, postfix return by value
    BestPractices& operator++() {      // Prefix
        ++value;
        return *this;
    }
    
    BestPractices operator++(int) {    // Postfix
        BestPractices temp(*this);
        ++value;
        return temp;
    }
    
    // 4. Comparison operators should be const
    bool operator==(const BestPractices& other) const {
        return value == other.value;
    }
    
    // 5. Stream operators should be friend functions
    friend ostream& operator<<(ostream& os, const BestPractices& obj) {
        os << obj.value;
        return os;
    }
    
    // 6. Self-assignment protection in assignment operators
    BestPractices& operator=(const BestPractices& other) {
        if (this != &other) {  // Self-assignment check
            value = other.value;
        }
        return *this;
    }
    
    int getValue() const { return value; }
};

// Don't overload operators that don't make sense
// For example, don't overload % for a Person class

// Be consistent with semantics
// If you overload +, also overload +=
// If you overload ==, also overload !=

int main() {
    BestPractices a(10);
    BestPractices b(20);
    
    cout << "a = " << a << endl;
    cout << "b = " << b << endl;
    
    BestPractices c = a + b;
    cout << "a + b = " << c << endl;
    
    a += b;
    cout << "After a += b, a = " << a << endl;
    
    cout << "++a = " << ++a << endl;
    cout << "a++ = " << a++ << endl;
    cout << "Now a = " << a << endl;
    
    return 0;
}
```

### 2. **Common Mistakes to Avoid**
```cpp
#include <iostream>
using namespace std;

class BadExample {
public:
    int value;
    
    // MISTAKE 1: Wrong return type for binary operators
    // Should return by value, not reference
    // BadExample& operator+(const BadExample& other) { ... }
    
    // MISTAKE 2: Not making comparison operators const
    // bool operator==(const BadExample& other) { ... }
    
    // MISTAKE 3: Returning wrong type from assignment
    // void operator=(const BadExample& other) { ... }
    
    // MISTAKE 4: No self-assignment check
    // BadExample& operator=(const BadExample& other) {
    //     delete[] somePointer;
    //     somePointer = new int[other.size];
    //     // If other is *this, we just deleted our own data!
    // }
    
    // MISTAKE 5: Inconsistent operators
    // Defining + but not +=, or == but not !=
};

class GoodExample {
private:
    int value;

public:
    GoodExample(int v = 0) : value(v) {}
    
    // CORRECT: Return by value for binary operators
    GoodExample operator+(const GoodExample& other) const {
        return GoodExample(value + other.value);
    }
    
    // CORRECT: Make comparison operators const
    bool operator==(const GoodExample& other) const {
        return value == other.value;
    }
    
    // CORRECT: Return reference from assignment
    GoodExample& operator=(const GoodExample& other) {
        if (this != &other) {  // Self-assignment check
            value = other.value;
        }
        return *this;
    }
    
    // CORRECT: Consistent operator pairs
    bool operator!=(const GoodExample& other) const {
        return !(*this == other);
    }
    
    GoodExample& operator+=(const GoodExample& other) {
        value += other.value;
        return *this;
    }
    
    friend ostream& operator<<(ostream& os, const GoodExample& obj) {
        os << obj.value;
        return os;
    }
};

int main() {
    GoodExample a(5);
    GoodExample b(10);
    
    cout << "Demonstrating correct operator overloading:" << endl;
    cout << "a = " << a << ", b = " << b << endl;
    cout << "a + b = " << (a + b) << endl;
    cout << "a == b: " << (a == b) << endl;
    cout << "a != b: " << (a != b) << endl;
    
    a += b;
    cout << "After a += b, a = " << a << endl;
    
    return 0;
}
```

## 🔗 Related Topics
- [OOP Basics](./08-oop.md)
- [Polymorphism](./10-polymorphism.md)
- [Templates](./12-templates.md)
- [Advanced Topics](./17-advanced-topics.md)

---
*Previous: [Polymorphism](./10-polymorphism.md) | Next: [Templates](./12-templates.md)*
