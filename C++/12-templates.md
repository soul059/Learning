# 12. Templates

## 📋 Overview
Templates are a powerful C++ feature that enables generic programming. They allow you to write code that works with any data type without sacrificing type safety or performance. Templates are resolved at compile time, creating type-specific versions of your generic code.

## 🎯 Function Templates

### 1. **Basic Function Templates**
```cpp
#include <iostream>
#include <vector>
#include <string>
using namespace std;

// Basic function template
template<typename T>
T maximum(T a, T b) {
    return (a > b) ? a : b;
}

// Function template with multiple parameters
template<typename T, typename U>
auto add(T a, U b) -> decltype(a + b) {
    return a + b;
}

// Template with non-type parameter
template<typename T, int SIZE>
void printArray(const T (&arr)[SIZE]) {
    cout << "Array of size " << SIZE << ": ";
    for (int i = 0; i < SIZE; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;
}

// Template specialization
template<>
const char* maximum<const char*>(const char* a, const char* b) {
    return (strcmp(a, b) > 0) ? a : b;
}

int main() {
    // Template argument deduction
    cout << "Maximum of 10 and 20: " << maximum(10, 20) << endl;
    cout << "Maximum of 3.14 and 2.71: " << maximum(3.14, 2.71) << endl;
    cout << "Maximum of 'a' and 'z': " << maximum('a', 'z') << endl;
    
    // Explicit template arguments
    cout << "Maximum<double>(5, 7.5): " << maximum<double>(5, 7.5) << endl;
    
    // Multiple type parameters
    cout << "Add int and double: " << add(5, 3.14) << endl;
    cout << "Add double and int: " << add(2.5, 10) << endl;
    
    // Non-type template parameters
    int intArray[] = {1, 2, 3, 4, 5};
    double doubleArray[] = {1.1, 2.2, 3.3};
    
    printArray(intArray);
    printArray(doubleArray);
    
    // Template specialization
    const char* str1 = "hello";
    const char* str2 = "world";
    cout << "Maximum string: " << maximum(str1, str2) << endl;
    
    return 0;
}
```

### 2. **Advanced Function Templates**
```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <type_traits>
using namespace std;

// Template with default arguments
template<typename T, typename Compare = less<T>>
T findExtreme(const vector<T>& vec, Compare comp = Compare{}) {
    if (vec.empty()) {
        throw runtime_error("Empty vector");
    }
    
    T result = vec[0];
    for (const auto& element : vec) {
        if (comp(result, element)) {
            result = element;
        }
    }
    return result;
}

// Variadic template function
template<typename T>
T sum(T value) {
    return value;
}

template<typename T, typename... Args>
T sum(T first, Args... args) {
    return first + sum(args...);
}

// Template with SFINAE (Substitution Failure Is Not An Error)
template<typename T>
typename enable_if<is_integral<T>::value, T>::type
multiply(T a, T b) {
    cout << "Integer multiplication: ";
    return a * b;
}

template<typename T>
typename enable_if<is_floating_point<T>::value, T>::type
multiply(T a, T b) {
    cout << "Floating-point multiplication: ";
    return a * b;
}

// Perfect forwarding
template<typename T, typename... Args>
unique_ptr<T> make_unique(Args&&... args) {
    return unique_ptr<T>(new T(forward<Args>(args)...));
}

class Person {
private:
    string name;
    int age;

public:
    Person(const string& n, int a) : name(n), age(a) {
        cout << "Person created: " << name << ", " << age << endl;
    }
    
    void display() const {
        cout << "Name: " << name << ", Age: " << age << endl;
    }
};

int main() {
    // Template with default arguments
    vector<int> numbers = {5, 2, 8, 1, 9, 3};
    
    cout << "Maximum: " << findExtreme(numbers) << endl;  // Uses default less<int>
    cout << "Minimum: " << findExtreme(numbers, greater<int>{}) << endl;
    
    // Variadic templates
    cout << "Sum of 1, 2, 3, 4, 5: " << sum(1, 2, 3, 4, 5) << endl;
    cout << "Sum of 1.1, 2.2, 3.3: " << sum(1.1, 2.2, 3.3) << endl;
    
    // SFINAE
    cout << multiply(5, 6) << endl;      // Calls integer version
    cout << multiply(3.14, 2.5) << endl; // Calls floating-point version
    
    // Perfect forwarding
    auto person = make_unique<Person>("Alice", 30);
    person->display();
    
    return 0;
}
```

## 🏗️ Class Templates

### 1. **Basic Class Templates**
```cpp
#include <iostream>
#include <stdexcept>
using namespace std;

template<typename T, int CAPACITY = 100>
class Stack {
private:
    T data[CAPACITY];
    int top;

public:
    Stack() : top(-1) {}
    
    void push(const T& item) {
        if (isFull()) {
            throw overflow_error("Stack overflow");
        }
        data[++top] = item;
    }
    
    T pop() {
        if (isEmpty()) {
            throw underflow_error("Stack underflow");
        }
        return data[top--];
    }
    
    const T& peek() const {
        if (isEmpty()) {
            throw underflow_error("Stack is empty");
        }
        return data[top];
    }
    
    bool isEmpty() const {
        return top == -1;
    }
    
    bool isFull() const {
        return top == CAPACITY - 1;
    }
    
    int size() const {
        return top + 1;
    }
    
    void display() const {
        cout << "Stack contents (top to bottom): ";
        for (int i = top; i >= 0; i--) {
            cout << data[i] << " ";
        }
        cout << endl;
    }
};

// Template specialization for bool
template<int CAPACITY>
class Stack<bool, CAPACITY> {
private:
    unsigned char data[(CAPACITY + 7) / 8];  // Bit packing
    int count;

public:
    Stack() : count(0) {
        fill(data, data + sizeof(data), 0);
    }
    
    void push(bool item) {
        if (count >= CAPACITY) {
            throw overflow_error("Stack overflow");
        }
        
        int byteIndex = count / 8;
        int bitIndex = count % 8;
        
        if (item) {
            data[byteIndex] |= (1 << bitIndex);
        } else {
            data[byteIndex] &= ~(1 << bitIndex);
        }
        count++;
    }
    
    bool pop() {
        if (count == 0) {
            throw underflow_error("Stack underflow");
        }
        
        count--;
        int byteIndex = count / 8;
        int bitIndex = count % 8;
        
        return (data[byteIndex] & (1 << bitIndex)) != 0;
    }
    
    bool isEmpty() const {
        return count == 0;
    }
    
    int size() const {
        return count;
    }
};

int main() {
    // Integer stack with default capacity
    Stack<int> intStack;
    
    cout << "=== Integer Stack Demo ===" << endl;
    for (int i = 1; i <= 5; i++) {
        intStack.push(i * 10);
    }
    
    intStack.display();
    cout << "Stack size: " << intStack.size() << endl;
    
    while (!intStack.isEmpty()) {
        cout << "Popped: " << intStack.pop() << endl;
    }
    
    // String stack with custom capacity
    Stack<string, 5> stringStack;
    
    cout << "\n=== String Stack Demo ===" << endl;
    stringStack.push("First");
    stringStack.push("Second");
    stringStack.push("Third");
    
    stringStack.display();
    
    // Boolean stack (specialized version)
    Stack<bool, 10> boolStack;
    
    cout << "\n=== Boolean Stack Demo ===" << endl;
    boolStack.push(true);
    boolStack.push(false);
    boolStack.push(true);
    boolStack.push(true);
    
    cout << "Boolean stack size: " << boolStack.size() << endl;
    
    while (!boolStack.isEmpty()) {
        cout << "Popped: " << (boolStack.pop() ? "true" : "false") << endl;
    }
    
    return 0;
}
```

### 2. **Advanced Class Templates**
```cpp
#include <iostream>
#include <memory>
#include <initializer_list>
using namespace std;

template<typename T>
class Vector {
private:
    T* data;
    size_t capacity;
    size_t size_;
    
    void reallocate(size_t newCapacity) {
        T* newData = static_cast<T*>(operator new(newCapacity * sizeof(T)));
        
        // Move or copy existing elements
        for (size_t i = 0; i < size_; i++) {
            new(newData + i) T(move(data[i]));
            data[i].~T();
        }
        
        operator delete(data);
        data = newData;
        capacity = newCapacity;
    }

public:
    // Iterator class
    class Iterator {
    private:
        T* ptr;

    public:
        Iterator(T* p) : ptr(p) {}
        
        T& operator*() const { return *ptr; }
        T* operator->() const { return ptr; }
        
        Iterator& operator++() { ++ptr; return *this; }
        Iterator operator++(int) { Iterator temp = *this; ++ptr; return temp; }
        
        Iterator& operator--() { --ptr; return *this; }
        Iterator operator--(int) { Iterator temp = *this; --ptr; return temp; }
        
        Iterator operator+(int n) const { return Iterator(ptr + n); }
        Iterator operator-(int n) const { return Iterator(ptr - n); }
        
        ptrdiff_t operator-(const Iterator& other) const { return ptr - other.ptr; }
        
        bool operator==(const Iterator& other) const { return ptr == other.ptr; }
        bool operator!=(const Iterator& other) const { return ptr != other.ptr; }
        bool operator<(const Iterator& other) const { return ptr < other.ptr; }
        bool operator>(const Iterator& other) const { return ptr > other.ptr; }
    };
    
    // Constructors
    Vector() : data(nullptr), capacity(0), size_(0) {}
    
    explicit Vector(size_t count) : capacity(count), size_(count) {
        data = static_cast<T*>(operator new(capacity * sizeof(T)));
        for (size_t i = 0; i < size_; i++) {
            new(data + i) T();
        }
    }
    
    Vector(size_t count, const T& value) : capacity(count), size_(count) {
        data = static_cast<T*>(operator new(capacity * sizeof(T)));
        for (size_t i = 0; i < size_; i++) {
            new(data + i) T(value);
        }
    }
    
    Vector(initializer_list<T> init) : capacity(init.size()), size_(init.size()) {
        data = static_cast<T*>(operator new(capacity * sizeof(T)));
        size_t i = 0;
        for (const auto& item : init) {
            new(data + i) T(item);
            ++i;
        }
    }
    
    // Copy constructor
    Vector(const Vector& other) : capacity(other.capacity), size_(other.size_) {
        data = static_cast<T*>(operator new(capacity * sizeof(T)));
        for (size_t i = 0; i < size_; i++) {
            new(data + i) T(other.data[i]);
        }
    }
    
    // Move constructor
    Vector(Vector&& other) noexcept 
        : data(other.data), capacity(other.capacity), size_(other.size_) {
        other.data = nullptr;
        other.capacity = 0;
        other.size_ = 0;
    }
    
    // Destructor
    ~Vector() {
        clear();
        operator delete(data);
    }
    
    // Assignment operators
    Vector& operator=(const Vector& other) {
        if (this != &other) {
            clear();
            capacity = other.capacity;
            size_ = other.size_;
            data = static_cast<T*>(operator new(capacity * sizeof(T)));
            for (size_t i = 0; i < size_; i++) {
                new(data + i) T(other.data[i]);
            }
        }
        return *this;
    }
    
    Vector& operator=(Vector&& other) noexcept {
        if (this != &other) {
            clear();
            operator delete(data);
            
            data = other.data;
            capacity = other.capacity;
            size_ = other.size_;
            
            other.data = nullptr;
            other.capacity = 0;
            other.size_ = 0;
        }
        return *this;
    }
    
    // Element access
    T& operator[](size_t index) {
        return data[index];
    }
    
    const T& operator[](size_t index) const {
        return data[index];
    }
    
    T& at(size_t index) {
        if (index >= size_) {
            throw out_of_range("Index out of bounds");
        }
        return data[index];
    }
    
    const T& at(size_t index) const {
        if (index >= size_) {
            throw out_of_range("Index out of bounds");
        }
        return data[index];
    }
    
    T& front() { return data[0]; }
    const T& front() const { return data[0]; }
    
    T& back() { return data[size_ - 1]; }
    const T& back() const { return data[size_ - 1]; }
    
    // Modifiers
    void push_back(const T& value) {
        if (size_ >= capacity) {
            reallocate(capacity == 0 ? 1 : capacity * 2);
        }
        new(data + size_) T(value);
        ++size_;
    }
    
    void push_back(T&& value) {
        if (size_ >= capacity) {
            reallocate(capacity == 0 ? 1 : capacity * 2);
        }
        new(data + size_) T(move(value));
        ++size_;
    }
    
    template<typename... Args>
    void emplace_back(Args&&... args) {
        if (size_ >= capacity) {
            reallocate(capacity == 0 ? 1 : capacity * 2);
        }
        new(data + size_) T(forward<Args>(args)...);
        ++size_;
    }
    
    void pop_back() {
        if (size_ > 0) {
            --size_;
            data[size_].~T();
        }
    }
    
    void clear() {
        for (size_t i = 0; i < size_; i++) {
            data[i].~T();
        }
        size_ = 0;
    }
    
    void reserve(size_t newCapacity) {
        if (newCapacity > capacity) {
            reallocate(newCapacity);
        }
    }
    
    // Capacity
    size_t size() const { return size_; }
    size_t getCapacity() const { return capacity; }
    bool empty() const { return size_ == 0; }
    
    // Iterators
    Iterator begin() { return Iterator(data); }
    Iterator end() { return Iterator(data + size_); }
    
    const Iterator begin() const { return Iterator(data); }
    const Iterator end() const { return Iterator(data + size_); }
};

class ComplexObject {
private:
    string name;
    int value;

public:
    ComplexObject(const string& n = "default", int v = 0) : name(n), value(v) {
        cout << "ComplexObject created: " << name << "(" << value << ")" << endl;
    }
    
    ComplexObject(const ComplexObject& other) : name(other.name), value(other.value) {
        cout << "ComplexObject copied: " << name << "(" << value << ")" << endl;
    }
    
    ComplexObject(ComplexObject&& other) noexcept : name(move(other.name)), value(other.value) {
        cout << "ComplexObject moved: " << name << "(" << value << ")" << endl;
        other.value = 0;
    }
    
    ~ComplexObject() {
        cout << "ComplexObject destroyed: " << name << "(" << value << ")" << endl;
    }
    
    ComplexObject& operator=(const ComplexObject& other) {
        if (this != &other) {
            name = other.name;
            value = other.value;
            cout << "ComplexObject assigned: " << name << "(" << value << ")" << endl;
        }
        return *this;
    }
    
    void display() const {
        cout << name << "(" << value << ")";
    }
    
    string getName() const { return name; }
    int getValue() const { return value; }
};

int main() {
    cout << "=== Custom Vector Demo ===" << endl;
    
    // Test with built-in types
    Vector<int> intVec{1, 2, 3, 4, 5};
    
    cout << "Integer vector: ";
    for (auto it = intVec.begin(); it != intVec.end(); ++it) {
        cout << *it << " ";
    }
    cout << endl;
    
    intVec.push_back(6);
    intVec.emplace_back(7);
    
    cout << "After adding elements: ";
    for (size_t i = 0; i < intVec.size(); i++) {
        cout << intVec[i] << " ";
    }
    cout << endl;
    
    // Test with complex objects
    cout << "\n=== Complex Object Vector ===" << endl;
    Vector<ComplexObject> objVec;
    
    objVec.emplace_back("Object1", 100);
    objVec.emplace_back("Object2", 200);
    objVec.push_back(ComplexObject("Object3", 300));
    
    cout << "Complex object vector: ";
    for (const auto& obj : objVec) {
        obj.display();
        cout << " ";
    }
    cout << endl;
    
    // Test copy and move
    cout << "\n=== Copy and Move Operations ===" << endl;
    Vector<ComplexObject> copyVec = objVec;  // Copy constructor
    Vector<ComplexObject> moveVec = move(objVec);  // Move constructor
    
    cout << "Original vector size after move: " << objVec.size() << endl;
    cout << "Moved vector size: " << moveVec.size() << endl;
    
    return 0;
}
```

## 🔧 Template Specialization

### 1. **Full and Partial Specialization**
```cpp
#include <iostream>
#include <string>
#include <vector>
using namespace std;

// Primary template
template<typename T, typename U>
class Pair {
private:
    T first;
    U second;

public:
    Pair(const T& f, const U& s) : first(f), second(s) {}
    
    void display() const {
        cout << "Generic Pair: (" << first << ", " << second << ")" << endl;
    }
    
    T getFirst() const { return first; }
    U getSecond() const { return second; }
};

// Full specialization for <string, string>
template<>
class Pair<string, string> {
private:
    string first, second;

public:
    Pair(const string& f, const string& s) : first(f), second(s) {}
    
    void display() const {
        cout << "String Pair: \"" << first << "\" and \"" << second << "\"" << endl;
    }
    
    string getConcatenated() const {
        return first + " " + second;
    }
    
    string getFirst() const { return first; }
    string getSecond() const { return second; }
};

// Partial specialization for pointer types
template<typename T>
class Pair<T*, T*> {
private:
    T* first;
    T* second;

public:
    Pair(T* f, T* s) : first(f), second(s) {}
    
    void display() const {
        cout << "Pointer Pair: ";
        if (first && second) {
            cout << "(*" << *first << ", *" << *second << ")";
        }
        cout << " [addresses: " << first << ", " << second << "]" << endl;
    }
    
    T* getFirst() const { return first; }
    T* getSecond() const { return second; }
    
    // Special method for pointer pairs
    void swap() {
        T* temp = first;
        first = second;
        second = temp;
    }
};

// Template specialization for function templates
template<typename T>
void printType(const T& value) {
    cout << "Generic type: " << value << " (type: " << typeid(T).name() << ")" << endl;
}

// Specialization for const char*
template<>
void printType<const char*>(const char* const& value) {
    cout << "C-string: \"" << value << "\"" << endl;
}

// Specialization for vectors
template<typename T>
void printType(const vector<T>& vec) {
    cout << "Vector of " << vec.size() << " elements: [";
    for (size_t i = 0; i < vec.size(); i++) {
        cout << vec[i];
        if (i < vec.size() - 1) cout << ", ";
    }
    cout << "]" << endl;
}

int main() {
    cout << "=== Template Specialization Demo ===" << endl;
    
    // Primary template
    Pair<int, double> intDoublePair(42, 3.14);
    intDoublePair.display();
    
    // Full specialization
    Pair<string, string> stringPair("Hello", "World");
    stringPair.display();
    cout << "Concatenated: " << stringPair.getConcatenated() << endl;
    
    // Partial specialization
    int a = 10, b = 20;
    Pair<int, int> pointerPair(&a, &b);
    pointerPair.display();
    
    cout << "Before swap:" << endl;
    pointerPair.display();
    pointerPair.swap();
    cout << "After swap:" << endl;
    pointerPair.display();
    
    // Function template specialization
    cout << "\n=== Function Template Specialization ===" << endl;
    printType(42);
    printType(3.14);
    printType("Hello, World!");
    
    vector<int> numbers = {1, 2, 3, 4, 5};
    printType(numbers);
    
    return 0;
}
```

### 2. **Template Metaprogramming**
```cpp
#include <iostream>
#include <type_traits>
using namespace std;

// Compile-time factorial calculation
template<int N>
struct Factorial {
    static constexpr int value = N * Factorial<N - 1>::value;
};

// Specialization for base case
template<>
struct Factorial<0> {
    static constexpr int value = 1;
};

// Type traits example
template<typename T>
struct TypeInfo {
    static constexpr bool is_pointer = false;
    static constexpr bool is_reference = false;
    static constexpr bool is_const = false;
    static const char* name() { return "unknown"; }
};

// Specializations for different types
template<typename T>
struct TypeInfo<T*> {
    static constexpr bool is_pointer = true;
    static constexpr bool is_reference = false;
    static constexpr bool is_const = false;
    static const char* name() { return "pointer"; }
};

template<typename T>
struct TypeInfo<T&> {
    static constexpr bool is_pointer = false;
    static constexpr bool is_reference = true;
    static constexpr bool is_const = false;
    static const char* name() { return "reference"; }
};

template<typename T>
struct TypeInfo<const T> {
    static constexpr bool is_pointer = false;
    static constexpr bool is_reference = false;
    static constexpr bool is_const = true;
    static const char* name() { return "const"; }
};

// SFINAE example - enable_if
template<typename T>
typename enable_if<is_integral<T>::value, T>::type
process(T value) {
    cout << "Processing integral type: " << value << endl;
    return value * 2;
}

template<typename T>
typename enable_if<is_floating_point<T>::value, T>::type
process(T value) {
    cout << "Processing floating-point type: " << value << endl;
    return value / 2.0;
}

// Template alias
template<typename T>
using Vector = std::vector<T>;

template<typename K, typename V>
using Map = std::map<K, V>;

// Variable templates (C++14)
template<typename T>
constexpr bool is_smart_pointer_v = false;

template<typename T>
constexpr bool is_smart_pointer_v<std::unique_ptr<T>> = true;

template<typename T>
constexpr bool is_smart_pointer_v<std::shared_ptr<T>> = true;

// Concept simulation (pre-C++20)
template<typename T>
struct has_size {
private:
    template<typename U>
    static auto test(int) -> decltype(declval<U>().size(), true_type{});
    
    template<typename>
    static false_type test(...);

public:
    static constexpr bool value = decltype(test<T>(0))::value;
};

template<typename Container>
typename enable_if<has_size<Container>::value, size_t>::type
getSize(const Container& c) {
    return c.size();
}

int main() {
    cout << "=== Template Metaprogramming ===" << endl;
    
    // Compile-time calculations
    cout << "Factorial<5> = " << Factorial<5>::value << endl;
    cout << "Factorial<0> = " << Factorial<0>::value << endl;
    
    // Type traits
    cout << "\n=== Type Information ===" << endl;
    cout << "int: pointer=" << TypeInfo<int>::is_pointer 
         << ", reference=" << TypeInfo<int>::is_reference 
         << ", const=" << TypeInfo<int>::is_const << endl;
    
    cout << "int*: pointer=" << TypeInfo<int*>::is_pointer 
         << ", name=" << TypeInfo<int*>::name() << endl;
    
    cout << "int&: reference=" << TypeInfo<int&>::is_reference 
         << ", name=" << TypeInfo<int&>::name() << endl;
    
    cout << "const int: const=" << TypeInfo<const int>::is_const 
         << ", name=" << TypeInfo<const int>::name() << endl;
    
    // SFINAE
    cout << "\n=== SFINAE Examples ===" << endl;
    cout << "Result: " << process(42) << endl;        // Integral
    cout << "Result: " << process(3.14) << endl;      // Floating-point
    
    // Template aliases
    Vector<int> vec = {1, 2, 3, 4, 5};
    cout << "\nVector size: " << vec.size() << endl;
    
    // Concept simulation
    cout << "Vector has size method: " << has_size<Vector<int>>::value << endl;
    cout << "Int has size method: " << has_size<int>::value << endl;
    
    if constexpr (has_size<Vector<int>>::value) {
        cout << "Getting size of vector: " << getSize(vec) << endl;
    }
    
    return 0;
}
```

## 🎪 Practical Examples

### 1. **Generic Data Structures**
```cpp
#include <iostream>
#include <functional>
#include <vector>
using namespace std;

// Generic Binary Search Tree
template<typename T, typename Compare = less<T>>
class BST {
private:
    struct Node {
        T data;
        Node* left;
        Node* right;
        
        Node(const T& value) : data(value), left(nullptr), right(nullptr) {}
    };
    
    Node* root;
    Compare comp;
    
    void insert(Node*& node, const T& value) {
        if (!node) {
            node = new Node(value);
            return;
        }
        
        if (comp(value, node->data)) {
            insert(node->left, value);
        } else if (comp(node->data, value)) {
            insert(node->right, value);
        }
        // Equal values are ignored
    }
    
    bool search(Node* node, const T& value) const {
        if (!node) return false;
        
        if (comp(value, node->data)) {
            return search(node->left, value);
        } else if (comp(node->data, value)) {
            return search(node->right, value);
        } else {
            return true;  // Found
        }
    }
    
    void inorder(Node* node, vector<T>& result) const {
        if (node) {
            inorder(node->left, result);
            result.push_back(node->data);
            inorder(node->right, result);
        }
    }
    
    void cleanup(Node* node) {
        if (node) {
            cleanup(node->left);
            cleanup(node->right);
            delete node;
        }
    }
    
    Node* findMin(Node* node) const {
        while (node && node->left) {
            node = node->left;
        }
        return node;
    }
    
    Node* remove(Node* node, const T& value) {
        if (!node) return nullptr;
        
        if (comp(value, node->data)) {
            node->left = remove(node->left, value);
        } else if (comp(node->data, value)) {
            node->right = remove(node->right, value);
        } else {
            // Node to be deleted found
            if (!node->left) {
                Node* temp = node->right;
                delete node;
                return temp;
            } else if (!node->right) {
                Node* temp = node->left;
                delete node;
                return temp;
            } else {
                // Node with two children
                Node* temp = findMin(node->right);
                node->data = temp->data;
                node->right = remove(node->right, temp->data);
            }
        }
        return node;
    }

public:
    BST(Compare c = Compare{}) : root(nullptr), comp(c) {}
    
    ~BST() {
        cleanup(root);
    }
    
    void insert(const T& value) {
        insert(root, value);
    }
    
    bool search(const T& value) const {
        return search(root, value);
    }
    
    void remove(const T& value) {
        root = remove(root, value);
    }
    
    vector<T> getSorted() const {
        vector<T> result;
        inorder(root, result);
        return result;
    }
    
    bool empty() const {
        return root == nullptr;
    }
};

// Generic Hash Table
template<typename K, typename V, typename Hash = hash<K>>
class HashTable {
private:
    struct Entry {
        K key;
        V value;
        bool deleted;
        
        Entry() : deleted(true) {}
        Entry(const K& k, const V& v) : key(k), value(v), deleted(false) {}
    };
    
    vector<Entry> table;
    size_t capacity;
    size_t size_;
    Hash hasher;
    
    size_t hashFunction(const K& key) const {
        return hasher(key) % capacity;
    }
    
    void rehash() {
        vector<Entry> oldTable = move(table);
        capacity *= 2;
        size_ = 0;
        table = vector<Entry>(capacity);
        
        for (const auto& entry : oldTable) {
            if (!entry.deleted) {
                insert(entry.key, entry.value);
            }
        }
    }

public:
    HashTable(size_t cap = 16) : capacity(cap), size_(0), table(capacity) {}
    
    void insert(const K& key, const V& value) {
        if (size_ >= capacity * 0.75) {
            rehash();
        }
        
        size_t index = hashFunction(key);
        
        // Linear probing
        while (!table[index].deleted && table[index].key != key) {
            index = (index + 1) % capacity;
        }
        
        if (table[index].deleted) {
            size_++;
        }
        
        table[index] = Entry(key, value);
    }
    
    bool find(const K& key, V& value) const {
        size_t index = hashFunction(key);
        size_t originalIndex = index;
        
        do {
            if (table[index].deleted) {
                if (table[index].key == key) {
                    // This slot was used but is now deleted
                    index = (index + 1) % capacity;
                    continue;
                }
                return false;  // Empty slot, key not found
            }
            
            if (table[index].key == key) {
                value = table[index].value;
                return true;
            }
            
            index = (index + 1) % capacity;
        } while (index != originalIndex);
        
        return false;
    }
    
    bool remove(const K& key) {
        size_t index = hashFunction(key);
        size_t originalIndex = index;
        
        do {
            if (table[index].deleted && table[index].key != key) {
                return false;  // Empty slot, key not found
            }
            
            if (!table[index].deleted && table[index].key == key) {
                table[index].deleted = true;
                size_--;
                return true;
            }
            
            index = (index + 1) % capacity;
        } while (index != originalIndex);
        
        return false;
    }
    
    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    
    void display() const {
        cout << "Hash Table contents:" << endl;
        for (size_t i = 0; i < capacity; i++) {
            if (!table[i].deleted) {
                cout << "[" << i << "] " << table[i].key << " -> " << table[i].value << endl;
            }
        }
    }
};

int main() {
    cout << "=== Generic BST Demo ===" << endl;
    
    // Integer BST
    BST<int> intBST;
    vector<int> values = {50, 30, 70, 20, 40, 60, 80};
    
    for (int val : values) {
        intBST.insert(val);
    }
    
    cout << "Sorted values: ";
    auto sorted = intBST.getSorted();
    for (int val : sorted) {
        cout << val << " ";
    }
    cout << endl;
    
    cout << "Search 40: " << (intBST.search(40) ? "Found" : "Not found") << endl;
    cout << "Search 25: " << (intBST.search(25) ? "Found" : "Not found") << endl;
    
    // String BST with custom comparator
    BST<string, greater<string>> stringBST;  // Reverse order
    vector<string> words = {"apple", "banana", "cherry", "date", "elderberry"};
    
    for (const string& word : words) {
        stringBST.insert(word);
    }
    
    cout << "\nReverse sorted words: ";
    auto sortedWords = stringBST.getSorted();
    for (const string& word : sortedWords) {
        cout << word << " ";
    }
    cout << endl;
    
    cout << "\n=== Generic Hash Table Demo ===" << endl;
    
    HashTable<string, int> ageTable;
    
    ageTable.insert("Alice", 25);
    ageTable.insert("Bob", 30);
    ageTable.insert("Charlie", 35);
    ageTable.insert("Diana", 28);
    
    ageTable.display();
    
    int age;
    if (ageTable.find("Bob", age)) {
        cout << "Bob's age: " << age << endl;
    }
    
    if (ageTable.find("Eve", age)) {
        cout << "Eve's age: " << age << endl;
    } else {
        cout << "Eve not found" << endl;
    }
    
    ageTable.remove("Charlie");
    cout << "\nAfter removing Charlie:" << endl;
    ageTable.display();
    
    return 0;
}
```

### 2. **Template-based Utilities**
```cpp
#include <iostream>
#include <memory>
#include <chrono>
#include <functional>
#include <tuple>
using namespace std;
using namespace std::chrono;

// Timer utility template
template<typename TimeUnit = milliseconds>
class Timer {
private:
    high_resolution_clock::time_point start_time;
    string name;

public:
    Timer(const string& timer_name = "Timer") : name(timer_name) {
        start();
    }
    
    void start() {
        start_time = high_resolution_clock::now();
    }
    
    auto elapsed() const {
        auto end_time = high_resolution_clock::now();
        return duration_cast<TimeUnit>(end_time - start_time).count();
    }
    
    void report() const {
        cout << name << " elapsed: " << elapsed();
        if (is_same_v<TimeUnit, nanoseconds>) cout << " ns";
        else if (is_same_v<TimeUnit, microseconds>) cout << " μs";
        else if (is_same_v<TimeUnit, milliseconds>) cout << " ms";
        else if (is_same_v<TimeUnit, seconds>) cout << " s";
        cout << endl;
    }
    
    ~Timer() {
        if (name != "Timer") {  // Don't auto-report for default timers
            report();
        }
    }
};

// Benchmark function template
template<typename Func, typename... Args>
auto benchmark(const string& name, Func&& func, Args&&... args) {
    Timer<microseconds> timer(name);
    
    if constexpr (is_void_v<invoke_result_t<Func, Args...>>) {
        func(forward<Args>(args)...);
        timer.report();
    } else {
        auto result = func(forward<Args>(args)...);
        timer.report();
        return result;
    }
}

// Generic Observer Pattern
template<typename EventType>
class Observable {
private:
    vector<function<void(const EventType&)>> observers;

public:
    void subscribe(function<void(const EventType&)> observer) {
        observers.push_back(observer);
    }
    
    void notify(const EventType& event) {
        for (auto& observer : observers) {
            observer(event);
        }
    }
    
    size_t observerCount() const {
        return observers.size();
    }
};

// Event types
struct ClickEvent {
    int x, y;
    string button;
    
    ClickEvent(int x, int y, string btn) : x(x), y(y), button(btn) {}
};

struct KeyEvent {
    char key;
    bool shift, ctrl;
    
    KeyEvent(char k, bool s = false, bool c = false) : key(k), shift(s), ctrl(c) {}
};

// Generic Factory Pattern
template<typename BaseType, typename KeyType = string>
class Factory {
private:
    map<KeyType, function<unique_ptr<BaseType>()>> creators;

public:
    template<typename DerivedType>
    void registerType(const KeyType& key) {
        creators[key] = []() {
            return make_unique<DerivedType>();
        };
    }
    
    template<typename DerivedType, typename... Args>
    void registerType(const KeyType& key, Args&&... args) {
        creators[key] = [args...]() {
            return make_unique<DerivedType>(args...);
        };
    }
    
    unique_ptr<BaseType> create(const KeyType& key) {
        auto it = creators.find(key);
        if (it != creators.end()) {
            return it->second();
        }
        return nullptr;
    }
    
    vector<KeyType> getRegisteredTypes() const {
        vector<KeyType> types;
        for (const auto& pair : creators) {
            types.push_back(pair.first);
        }
        return types;
    }
};

// Example classes for factory
class Shape {
public:
    virtual void draw() const = 0;
    virtual double area() const = 0;
    virtual ~Shape() = default;
};

class Circle : public Shape {
private:
    double radius;

public:
    Circle(double r = 1.0) : radius(r) {}
    
    void draw() const override {
        cout << "Drawing circle with radius " << radius << endl;
    }
    
    double area() const override {
        return 3.14159 * radius * radius;
    }
};

class Rectangle : public Shape {
private:
    double width, height;

public:
    Rectangle(double w = 1.0, double h = 1.0) : width(w), height(h) {}
    
    void draw() const override {
        cout << "Drawing rectangle " << width << "x" << height << endl;
    }
    
    double area() const override {
        return width * height;
    }
};

// Test functions for benchmarking
void expensiveOperation() {
    // Simulate some work
    for (volatile int i = 0; i < 1000000; ++i) {
        // Do nothing, just consume time
    }
}

int fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

int main() {
    cout << "=== Template Utilities Demo ===" << endl;
    
    // Timer demo
    cout << "\n=== Timer Demo ===" << endl;
    {
        Timer<milliseconds> timer("Expensive Operation");
        expensiveOperation();
    }  // Timer reports automatically on destruction
    
    // Benchmark demo
    cout << "\n=== Benchmark Demo ===" << endl;
    auto result = benchmark("Fibonacci(35)", fibonacci, 35);
    cout << "Fibonacci(35) = " << result << endl;
    
    benchmark("Another Expensive Operation", expensiveOperation);
    
    // Observer pattern demo
    cout << "\n=== Observer Pattern Demo ===" << endl;
    Observable<ClickEvent> clickObservable;
    Observable<KeyEvent> keyObservable;
    
    // Subscribe to click events
    clickObservable.subscribe([](const ClickEvent& event) {
        cout << "Click detected at (" << event.x << ", " << event.y 
             << ") with " << event.button << " button" << endl;
    });
    
    clickObservable.subscribe([](const ClickEvent& event) {
        if (event.button == "left") {
            cout << "Left click handler: Processing click..." << endl;
        }
    });
    
    // Subscribe to key events
    keyObservable.subscribe([](const KeyEvent& event) {
        cout << "Key pressed: " << event.key;
        if (event.shift) cout << " (with Shift)";
        if (event.ctrl) cout << " (with Ctrl)";
        cout << endl;
    });
    
    // Trigger events
    clickObservable.notify(ClickEvent(100, 200, "left"));
    clickObservable.notify(ClickEvent(50, 75, "right"));
    keyObservable.notify(KeyEvent('A', true, false));
    keyObservable.notify(KeyEvent('C', false, true));
    
    // Factory pattern demo
    cout << "\n=== Factory Pattern Demo ===" << endl;
    Factory<Shape> shapeFactory;
    
    shapeFactory.registerType<Circle>("circle");
    shapeFactory.registerType<Rectangle>("rectangle");
    
    auto registeredTypes = shapeFactory.getRegisteredTypes();
    cout << "Registered shape types: ";
    for (const auto& type : registeredTypes) {
        cout << type << " ";
    }
    cout << endl;
    
    auto circle = shapeFactory.create("circle");
    auto rectangle = shapeFactory.create("rectangle");
    auto unknown = shapeFactory.create("triangle");
    
    if (circle) {
        circle->draw();
        cout << "Area: " << circle->area() << endl;
    }
    
    if (rectangle) {
        rectangle->draw();
        cout << "Area: " << rectangle->area() << endl;
    }
    
    if (!unknown) {
        cout << "Triangle shape not registered" << endl;
    }
    
    return 0;
}
```

## 💡 Best Practices

### Template Design Guidelines
```cpp
#include <iostream>
#include <type_traits>
using namespace std;

// 1. Use meaningful template parameter names
template<typename ElementType, typename Allocator = std::allocator<ElementType>>
class GoodContainer {
    // Implementation
};

// 2. Provide default template arguments when appropriate
template<typename T, size_t BufferSize = 1024>
class Buffer {
    T data[BufferSize];
public:
    constexpr size_t size() const { return BufferSize; }
};

// 3. Use SFINAE for conditional compilation
template<typename T>
typename enable_if<is_arithmetic<T>::value, T>::type
safe_divide(T a, T b) {
    if (b == 0) {
        throw invalid_argument("Division by zero");
    }
    return a / b;
}

// 4. Use concepts (C++20) or concept simulation for constraints
template<typename T>
struct is_container {
private:
    template<typename U>
    static auto test(int) -> decltype(
        declval<U>().begin(),
        declval<U>().end(),
        declval<U>().size(),
        true_type{}
    );
    
    template<typename>
    static false_type test(...);

public:
    static constexpr bool value = decltype(test<T>(0))::value;
};

template<typename Container>
typename enable_if<is_container<Container>::value, void>::type
print_container(const Container& c) {
    cout << "Container with " << c.size() << " elements: ";
    for (const auto& item : c) {
        cout << item << " ";
    }
    cout << endl;
}

// 5. Use perfect forwarding for generic functions
template<typename T, typename... Args>
unique_ptr<T> make_unique_safe(Args&&... args) {
    return unique_ptr<T>(new T(forward<Args>(args)...));
}

// 6. Provide explicit instantiation declarations when needed
extern template class Buffer<int, 512>;  // Declaration
// template class Buffer<int, 512>;      // Definition (in .cpp file)

int main() {
    cout << "=== Template Best Practices Demo ===" << endl;
    
    // Use meaningful names and defaults
    Buffer<int> defaultBuffer;    // Uses default size 1024
    Buffer<double, 256> customBuffer;  // Custom size
    
    cout << "Default buffer size: " << defaultBuffer.size() << endl;
    cout << "Custom buffer size: " << customBuffer.size() << endl;
    
    // SFINAE in action
    try {
        cout << "10 / 3 = " << safe_divide(10.0, 3.0) << endl;
        cout << "10 / 0 = " << safe_divide(10, 0) << endl;  // Will throw
    } catch (const exception& e) {
        cout << "Error: " << e.what() << endl;
    }
    
    // Container concept simulation
    vector<int> vec = {1, 2, 3, 4, 5};
    print_container(vec);
    
    // Perfect forwarding
    auto obj = make_unique_safe<string>("Hello, Templates!");
    cout << "Created string: " << *obj << endl;
    
    return 0;
}
```

## 🔗 Related Topics
- [Functions](./03-functions.md)
- [OOP Basics](./08-oop.md)
- [STL](./15-stl.md)
- [Advanced Topics](./17-advanced-topics.md)

---
*Previous: [Operator Overloading](./11-operator-overloading.md) | Next: [Exception Handling](./13-exception-handling.md)*
