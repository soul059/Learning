# 07. Pointers & References

## 📋 Overview
Pointers and references are fundamental concepts in C++ that provide indirect access to memory locations. They enable efficient memory management, dynamic allocation, and advanced programming techniques.

## 🎯 Pointers

### 1. **Pointer Basics**

#### What is a Pointer?
A pointer is a variable that stores the memory address of another variable.

```cpp
#include <iostream>
using namespace std;

int main() {
    int x = 42;          // Regular variable
    int* ptr = &x;       // Pointer to int, stores address of x
    
    cout << "Value of x: " << x << endl;           // 42
    cout << "Address of x: " << &x << endl;        // Memory address
    cout << "Value of ptr: " << ptr << endl;       // Same address as &x
    cout << "Value pointed by ptr: " << *ptr << endl; // 42 (dereferencing)
    
    // Modify value through pointer
    *ptr = 100;
    cout << "New value of x: " << x << endl;       // 100
    
    return 0;
}
```

#### Pointer Declaration and Initialization
```cpp
int main() {
    int num = 10;
    
    // Different ways to declare pointers
    int* ptr1;           // Declaration (uninitialized)
    int *ptr2;           // Alternative syntax
    int* ptr3, ptr4;     // ptr3 is pointer, ptr4 is int (be careful!)
    int *ptr5, *ptr6;    // Both are pointers
    
    // Initialize pointers
    ptr1 = &num;         // Point to num
    ptr2 = nullptr;      // Null pointer (C++11)
    ptr3 = NULL;         // Null pointer (older style)
    
    // Check for null before dereferencing
    if (ptr1 != nullptr) {
        cout << "Value: " << *ptr1 << endl;
    }
    
    return 0;
}
```

### 2. **Pointer Arithmetic**
```cpp
#include <iostream>
using namespace std;

int main() {
    int arr[5] = {10, 20, 30, 40, 50};
    int* ptr = arr;  // Points to first element
    
    cout << "Array using pointer arithmetic:" << endl;
    for (int i = 0; i < 5; i++) {
        cout << "arr[" << i << "] = " << *(ptr + i) << endl;
    }
    
    // Increment/decrement operations
    cout << "\nPointer increment operations:" << endl;
    ptr = arr;  // Reset to beginning
    
    cout << "ptr points to: " << *ptr << endl;        // 10
    ptr++;
    cout << "After ptr++: " << *ptr << endl;          // 20
    ptr += 2;
    cout << "After ptr += 2: " << *ptr << endl;       // 40
    ptr--;
    cout << "After ptr--: " << *ptr << endl;          // 30
    
    // Pointer difference
    int* start = arr;
    int* end = &arr[4];
    cout << "Distance between pointers: " << (end - start) << endl; // 4
    
    return 0;
}
```

### 3. **Pointers and Arrays**
```cpp
#include <iostream>
using namespace std;

void printArray(int* arr, int size) {
    for (int i = 0; i < size; i++) {
        cout << arr[i] << " ";  // or *(arr + i)
    }
    cout << endl;
}

int main() {
    int numbers[5] = {1, 2, 3, 4, 5};
    
    // Array name is a pointer to first element
    cout << "Array address: " << numbers << endl;
    cout << "First element address: " << &numbers[0] << endl;
    
    // Different ways to access array elements
    cout << "numbers[2] = " << numbers[2] << endl;     // Array notation
    cout << "*(numbers + 2) = " << *(numbers + 2) << endl; // Pointer arithmetic
    
    int* ptr = numbers;
    cout << "ptr[2] = " << ptr[2] << endl;             // Pointer as array
    
    // Pass array to function
    printArray(numbers, 5);
    printArray(ptr, 5);
    
    return 0;
}
```

### 4. **Pointers to Pointers**
```cpp
#include <iostream>
using namespace std;

int main() {
    int value = 42;
    int* ptr = &value;      // Pointer to int
    int** ptrToPtr = &ptr;  // Pointer to pointer to int
    
    cout << "Value: " << value << endl;              // 42
    cout << "Via ptr: " << *ptr << endl;             // 42
    cout << "Via ptrToPtr: " << **ptrToPtr << endl;  // 42
    
    // Addresses
    cout << "Address of value: " << &value << endl;
    cout << "Address stored in ptr: " << ptr << endl;
    cout << "Address of ptr: " << &ptr << endl;
    cout << "Address stored in ptrToPtr: " << ptrToPtr << endl;
    
    // Modify value through pointer to pointer
    **ptrToPtr = 100;
    cout << "New value: " << value << endl;          // 100
    
    return 0;
}
```

### 5. **Dynamic Memory Allocation**
```cpp
#include <iostream>
using namespace std;

int main() {
    // Allocate single integer
    int* ptr = new int(42);
    cout << "Dynamically allocated value: " << *ptr << endl;
    delete ptr;  // Free memory
    ptr = nullptr;  // Good practice
    
    // Allocate array
    int size;
    cout << "Enter array size: ";
    cin >> size;
    
    int* arr = new int[size];
    
    // Initialize array
    for (int i = 0; i < size; i++) {
        arr[i] = i * 10;
    }
    
    // Print array
    cout << "Dynamic array: ";
    for (int i = 0; i < size; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;
    
    delete[] arr;  // Free array memory
    arr = nullptr;
    
    return 0;
}
```

### 6. **Function Pointers**
```cpp
#include <iostream>
using namespace std;

// Functions to be pointed to
int add(int a, int b) { return a + b; }
int subtract(int a, int b) { return a - b; }
int multiply(int a, int b) { return a * b; }

// Function that takes function pointer as parameter
int calculate(int x, int y, int (*operation)(int, int)) {
    return operation(x, y);
}

int main() {
    // Declare function pointer
    int (*mathOp)(int, int);
    
    // Assign function to pointer
    mathOp = add;
    cout << "10 + 5 = " << mathOp(10, 5) << endl;
    
    mathOp = subtract;
    cout << "10 - 5 = " << mathOp(10, 5) << endl;
    
    // Use with function parameter
    cout << "Using calculate function:" << endl;
    cout << "Add: " << calculate(8, 3, add) << endl;
    cout << "Multiply: " << calculate(8, 3, multiply) << endl;
    
    // Array of function pointers
    int (*operations[])(int, int) = {add, subtract, multiply};
    string names[] = {"Add", "Subtract", "Multiply"};
    
    for (int i = 0; i < 3; i++) {
        cout << names[i] << ": " << operations[i](6, 4) << endl;
    }
    
    return 0;
}
```

## 📎 References

### 1. **Reference Basics**
```cpp
#include <iostream>
using namespace std;

int main() {
    int x = 10;
    int& ref = x;    // Reference to x (must be initialized)
    
    cout << "x = " << x << endl;           // 10
    cout << "ref = " << ref << endl;       // 10
    
    // Modify through reference
    ref = 20;
    cout << "After ref = 20:" << endl;
    cout << "x = " << x << endl;           // 20
    cout << "ref = " << ref << endl;       // 20
    
    // Reference and original have same address
    cout << "Address of x: " << &x << endl;
    cout << "Address of ref: " << &ref << endl;  // Same as &x
    
    return 0;
}
```

### 2. **References vs Pointers**
```cpp
#include <iostream>
using namespace std;

int main() {
    int a = 10, b = 20;
    
    // Pointer
    int* ptr = &a;
    cout << "Pointer to a: " << *ptr << endl;    // 10
    ptr = &b;  // Can point to different variable
    cout << "Pointer to b: " << *ptr << endl;    // 20
    
    // Reference
    int& ref = a;  // Must be initialized
    cout << "Reference to a: " << ref << endl;   // 10
    // ref = &b;  // Error: cannot reassign reference
    ref = b;      // This copies value of b to a
    cout << "After ref = b, a = " << a << endl;  // 20
    
    return 0;
}
```

### 3. **Function Parameters: Pass by Reference**
```cpp
#include <iostream>
using namespace std;

// Pass by value
void swapByValue(int a, int b) {
    int temp = a;
    a = b;
    b = temp;
    cout << "Inside swapByValue: a=" << a << ", b=" << b << endl;
}

// Pass by reference
void swapByReference(int& a, int& b) {
    int temp = a;
    a = b;
    b = temp;
    cout << "Inside swapByReference: a=" << a << ", b=" << b << endl;
}

// Pass by pointer
void swapByPointer(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
    cout << "Inside swapByPointer: a=" << *a << ", b=" << *b << endl;
}

int main() {
    int x = 10, y = 20;
    
    cout << "Original: x=" << x << ", y=" << y << endl;
    
    swapByValue(x, y);
    cout << "After swapByValue: x=" << x << ", y=" << y << endl;  // No change
    
    swapByReference(x, y);
    cout << "After swapByReference: x=" << x << ", y=" << y << endl;  // Swapped
    
    swapByPointer(&x, &y);
    cout << "After swapByPointer: x=" << x << ", y=" << y << endl;  // Swapped back
    
    return 0;
}
```

### 4. **const References**
```cpp
#include <iostream>
#include <string>
using namespace std;

// Efficient parameter passing for large objects
void printString(const string& str) {
    cout << "String: " << str << endl;
    // str[0] = 'X';  // Error: cannot modify const reference
}

// Return reference for chaining
string& getString() {
    static string str = "Hello";  // Static to ensure lifetime
    return str;
}

int main() {
    string message = "Hello, World!";
    printString(message);  // No copying of string
    
    // const reference to literal
    const int& ref = 42;  // OK: extends lifetime of temporary
    cout << "Reference to literal: " << ref << endl;
    
    // Function returning reference
    getString() = "Modified";  // Can modify through non-const reference
    cout << "Modified string: " << getString() << endl;
    
    return 0;
}
```

## 🔧 Advanced Pointer Concepts

### 1. **Smart Pointers (C++11)**
```cpp
#include <iostream>
#include <memory>
using namespace std;

class Resource {
public:
    Resource(int id) : id_(id) {
        cout << "Resource " << id_ << " created" << endl;
    }
    
    ~Resource() {
        cout << "Resource " << id_ << " destroyed" << endl;
    }
    
    void use() {
        cout << "Using resource " << id_ << endl;
    }

private:
    int id_;
};

int main() {
    // unique_ptr - exclusive ownership
    {
        unique_ptr<Resource> ptr1 = make_unique<Resource>(1);
        ptr1->use();
        
        // unique_ptr<Resource> ptr2 = ptr1;  // Error: no copy
        unique_ptr<Resource> ptr2 = move(ptr1);  // Transfer ownership
        
        if (!ptr1) {
            cout << "ptr1 is now null" << endl;
        }
        
        ptr2->use();
    }  // Resource automatically destroyed
    
    // shared_ptr - shared ownership
    {
        shared_ptr<Resource> ptr1 = make_shared<Resource>(2);
        cout << "Reference count: " << ptr1.use_count() << endl;  // 1
        
        {
            shared_ptr<Resource> ptr2 = ptr1;  // Shared ownership
            cout << "Reference count: " << ptr1.use_count() << endl;  // 2
            ptr2->use();
        }  // ptr2 destroyed, reference count decreases
        
        cout << "Reference count: " << ptr1.use_count() << endl;  // 1
    }  // Resource destroyed when last shared_ptr goes out of scope
    
    return 0;
}
```

### 2. **Pointer Safety and Best Practices**
```cpp
#include <iostream>
using namespace std;

// Dangerous: returning pointer to local variable
int* getBadPointer() {
    int local = 42;
    return &local;  // Danger: local variable destroyed
}

// Safe: returning pointer to static variable
int* getGoodPointer() {
    static int value = 42;
    return &value;  // OK: static variable persists
}

// Safe: dynamic allocation (caller must delete)
int* getDynamicPointer() {
    return new int(42);  // Caller responsible for deletion
}

int main() {
    // Always initialize pointers
    int* ptr = nullptr;
    
    // Check for null before dereferencing
    if (ptr != nullptr) {
        cout << *ptr << endl;
    }
    
    // Avoid double deletion
    ptr = new int(10);
    delete ptr;
    ptr = nullptr;  // Prevent accidental reuse
    // delete ptr;  // Safe: deleting nullptr is OK
    
    // Avoid memory leaks
    int* arr = new int[100];
    // ... use array ...
    delete[] arr;  // Don't forget to delete arrays
    arr = nullptr;
    
    // Use RAII and smart pointers when possible
    auto smart_ptr = make_unique<int>(42);
    // No manual delete needed
    
    return 0;
}
```

## 🎯 Practical Examples

### 1. **Dynamic Array Management**
```cpp
#include <iostream>
using namespace std;

class DynamicArray {
private:
    int* data;
    int size;
    int capacity;
    
public:
    DynamicArray(int initial_capacity = 10) 
        : size(0), capacity(initial_capacity) {
        data = new int[capacity];
    }
    
    ~DynamicArray() {
        delete[] data;
    }
    
    void push_back(int value) {
        if (size >= capacity) {
            resize();
        }
        data[size++] = value;
    }
    
    int get(int index) const {
        if (index >= 0 && index < size) {
            return data[index];
        }
        throw out_of_range("Index out of bounds");
    }
    
    int getSize() const { return size; }
    
private:
    void resize() {
        capacity *= 2;
        int* new_data = new int[capacity];
        
        for (int i = 0; i < size; i++) {
            new_data[i] = data[i];
        }
        
        delete[] data;
        data = new_data;
    }
};

int main() {
    DynamicArray arr;
    
    for (int i = 1; i <= 15; i++) {
        arr.push_back(i * 10);
    }
    
    cout << "Array contents: ";
    for (int i = 0; i < arr.getSize(); i++) {
        cout << arr.get(i) << " ";
    }
    cout << endl;
    
    return 0;
}
```

### 2. **Linked List Implementation**
```cpp
#include <iostream>
using namespace std;

struct Node {
    int data;
    Node* next;
    
    Node(int value) : data(value), next(nullptr) {}
};

class LinkedList {
private:
    Node* head;
    
public:
    LinkedList() : head(nullptr) {}
    
    ~LinkedList() {
        clear();
    }
    
    void push_front(int value) {
        Node* new_node = new Node(value);
        new_node->next = head;
        head = new_node;
    }
    
    void push_back(int value) {
        Node* new_node = new Node(value);
        
        if (!head) {
            head = new_node;
            return;
        }
        
        Node* current = head;
        while (current->next) {
            current = current->next;
        }
        current->next = new_node;
    }
    
    void display() const {
        Node* current = head;
        cout << "List: ";
        while (current) {
            cout << current->data << " -> ";
            current = current->next;
        }
        cout << "NULL" << endl;
    }
    
    bool remove(int value) {
        if (!head) return false;
        
        if (head->data == value) {
            Node* to_delete = head;
            head = head->next;
            delete to_delete;
            return true;
        }
        
        Node* current = head;
        while (current->next && current->next->data != value) {
            current = current->next;
        }
        
        if (current->next) {
            Node* to_delete = current->next;
            current->next = current->next->next;
            delete to_delete;
            return true;
        }
        
        return false;
    }
    
    void clear() {
        while (head) {
            Node* to_delete = head;
            head = head->next;
            delete to_delete;
        }
    }
};

int main() {
    LinkedList list;
    
    list.push_back(10);
    list.push_back(20);
    list.push_front(5);
    list.push_back(30);
    
    list.display();
    
    list.remove(20);
    cout << "After removing 20:" << endl;
    list.display();
    
    return 0;
}
```

### 3. **Function Callback System**
```cpp
#include <iostream>
#include <vector>
using namespace std;

// Event types
enum EventType { BUTTON_CLICK, KEY_PRESS, MOUSE_MOVE };

// Callback function type
typedef void (*EventCallback)(EventType, int);

// Event system class
class EventSystem {
private:
    vector<EventCallback> callbacks;
    
public:
    void registerCallback(EventCallback callback) {
        callbacks.push_back(callback);
    }
    
    void triggerEvent(EventType type, int data) {
        cout << "Triggering event..." << endl;
        for (EventCallback callback : callbacks) {
            callback(type, data);
        }
    }
};

// Event handler functions
void onButtonClick(EventType type, int buttonId) {
    if (type == BUTTON_CLICK) {
        cout << "Button " << buttonId << " was clicked!" << endl;
    }
}

void onKeyPress(EventType type, int keyCode) {
    if (type == KEY_PRESS) {
        cout << "Key " << keyCode << " was pressed!" << endl;
    }
}

void logAllEvents(EventType type, int data) {
    cout << "LOG: Event type " << type << " with data " << data << endl;
}

int main() {
    EventSystem eventSys;
    
    // Register event handlers
    eventSys.registerCallback(onButtonClick);
    eventSys.registerCallback(onKeyPress);
    eventSys.registerCallback(logAllEvents);
    
    // Trigger events
    eventSys.triggerEvent(BUTTON_CLICK, 1);
    eventSys.triggerEvent(KEY_PRESS, 65);  // 'A' key
    eventSys.triggerEvent(MOUSE_MOVE, 100);
    
    return 0;
}
```

## 💡 Best Practices

### 1. **Pointer Safety**
```cpp
// Always initialize pointers
int* ptr = nullptr;  // Not: int* ptr;

// Check for null before dereferencing
if (ptr != nullptr) {
    *ptr = 42;
}

// Set to null after deletion
delete ptr;
ptr = nullptr;

// Use smart pointers for automatic memory management
auto smart_ptr = make_unique<int>(42);
```

### 2. **Reference Guidelines**
```cpp
// Use const references for read-only parameters
void processData(const vector<int>& data) {
    // Efficient: no copying, cannot modify
}

// Return references for chaining operations
class Counter {
    int value = 0;
public:
    Counter& increment() { ++value; return *this; }
    Counter& decrement() { --value; return *this; }
    int get() const { return value; }
};

// Usage: counter.increment().increment().decrement();
```

### 3. **Memory Management**
```cpp
// Prefer RAII (Resource Acquisition Is Initialization)
class FileHandler {
    FILE* file;
public:
    FileHandler(const char* filename) {
        file = fopen(filename, "r");
        if (!file) throw runtime_error("Cannot open file");
    }
    
    ~FileHandler() {
        if (file) fclose(file);
    }
    
    // Disable copy constructor and assignment
    FileHandler(const FileHandler&) = delete;
    FileHandler& operator=(const FileHandler&) = delete;
};
```

## 🔗 Related Topics
- [Arrays & Strings](./06-arrays-strings.md)
- [Memory Management](./16-memory-management.md)
- [Object-Oriented Programming](./08-oop.md)
- [STL (Standard Template Library)](./15-stl.md)

---
*Previous: [Arrays & Strings](./06-arrays-strings.md) | Next: [Object-Oriented Programming](./08-oop.md)*
