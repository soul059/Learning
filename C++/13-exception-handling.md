# 13. Exception Handling

## 📋 Overview
Exception handling in C++ provides a robust mechanism for dealing with runtime errors and exceptional situations. It allows programs to gracefully handle errors without terminating abruptly, enabling better error management and program reliability.

## 🎯 Basic Exception Handling

### 1. **Try-Catch-Throw Mechanism**
```cpp
#include <iostream>
#include <stdexcept>
#include <string>
using namespace std;

// Function that throws exceptions
double divide(double a, double b) {
    if (b == 0) {
        throw invalid_argument("Division by zero is not allowed");
    }
    if (a < 0 || b < 0) {
        throw domain_error("Negative numbers not supported in this context");
    }
    return a / b;
}

int factorial(int n) {
    if (n < 0) {
        throw invalid_argument("Factorial of negative number is undefined");
    }
    if (n > 20) {
        throw overflow_error("Factorial too large to compute");
    }
    
    int result = 1;
    for (int i = 2; i <= n; i++) {
        result *= i;
    }
    return result;
}

void demonstrateBasicExceptions() {
    cout << "=== Basic Exception Handling ===" << endl;
    
    // Example 1: Handling specific exceptions
    try {
        cout << "Attempting to divide 10 by 2: ";
        double result = divide(10, 2);
        cout << result << endl;
        
        cout << "Attempting to divide 10 by 0: ";
        result = divide(10, 0);  // This will throw
        cout << result << endl;  // This won't execute
    } catch (const invalid_argument& e) {
        cout << "Invalid argument error: " << e.what() << endl;
    } catch (const domain_error& e) {
        cout << "Domain error: " << e.what() << endl;
    }
    
    // Example 2: Multiple catch blocks
    vector<int> testValues = {5, -1, 25};
    
    for (int val : testValues) {
        try {
            cout << "Factorial of " << val << " = " << factorial(val) << endl;
        } catch (const invalid_argument& e) {
            cout << "Error computing factorial of " << val << ": " << e.what() << endl;
        } catch (const overflow_error& e) {
            cout << "Overflow computing factorial of " << val << ": " << e.what() << endl;
        }
    }
}

int main() {
    demonstrateBasicExceptions();
    return 0;
}
```

### 2. **Standard Exception Hierarchy**
```cpp
#include <iostream>
#include <exception>
#include <stdexcept>
#include <memory>
#include <vector>
using namespace std;

void demonstrateStandardExceptions() {
    cout << "\n=== Standard Exception Types ===" << endl;
    
    try {
        // logic_error family
        throw invalid_argument("This is an invalid_argument");
    } catch (const logic_error& e) {
        cout << "Caught logic_error: " << e.what() << endl;
    }
    
    try {
        // runtime_error family
        throw runtime_error("This is a runtime_error");
    } catch (const runtime_error& e) {
        cout << "Caught runtime_error: " << e.what() << endl;
    }
    
    try {
        // out_of_range exception
        vector<int> vec = {1, 2, 3};
        cout << "Accessing vec.at(10): ";
        cout << vec.at(10) << endl;  // Will throw out_of_range
    } catch (const out_of_range& e) {
        cout << "Out of range error: " << e.what() << endl;
    }
    
    try {
        // bad_alloc exception
        cout << "Attempting huge memory allocation..." << endl;
        vector<int> huge_vector(SIZE_MAX);  // Will likely throw bad_alloc
    } catch (const bad_alloc& e) {
        cout << "Memory allocation failed: " << e.what() << endl;
    }
    
    try {
        // length_error exception
        string str;
        str.resize(str.max_size() + 1);  // Will throw length_error
    } catch (const length_error& e) {
        cout << "Length error: " << e.what() << endl;
    }
}

// Exception hierarchy demonstration
void catchHierarchy() {
    cout << "\n=== Exception Hierarchy ===" << endl;
    
    try {
        // Throw different types of exceptions
        int choice = 2;
        
        switch (choice) {
            case 1:
                throw invalid_argument("Invalid argument");
            case 2:
                throw out_of_range("Out of range");
            case 3:
                throw runtime_error("Runtime error");
            default:
                throw exception();
        }
    } catch (const logic_error& e) {
        cout << "Caught logic_error (base class): " << e.what() << endl;
    } catch (const runtime_error& e) {
        cout << "Caught runtime_error: " << e.what() << endl;
    } catch (const exception& e) {
        cout << "Caught generic exception: " << e.what() << endl;
    }
}

int main() {
    demonstrateStandardExceptions();
    catchHierarchy();
    return 0;
}
```

## 🔧 Custom Exceptions

### 1. **Creating Custom Exception Classes**
```cpp
#include <iostream>
#include <exception>
#include <string>
#include <sstream>
using namespace std;

// Base custom exception class
class CustomException : public exception {
protected:
    string message;

public:
    CustomException(const string& msg) : message(msg) {}
    
    virtual const char* what() const noexcept override {
        return message.c_str();
    }
    
    virtual ~CustomException() = default;
};

// Specific custom exceptions
class FileException : public CustomException {
private:
    string filename;
    int errorCode;

public:
    FileException(const string& file, int code, const string& msg) 
        : CustomException(msg), filename(file), errorCode(code) {
        ostringstream oss;
        oss << "File Error [" << errorCode << "] in '" << filename << "': " << msg;
        message = oss.str();
    }
    
    const string& getFilename() const { return filename; }
    int getErrorCode() const { return errorCode; }
};

class NetworkException : public CustomException {
private:
    string host;
    int port;

public:
    NetworkException(const string& h, int p, const string& msg)
        : CustomException(msg), host(h), port(p) {
        ostringstream oss;
        oss << "Network Error connecting to " << host << ":" << port << " - " << msg;
        message = oss.str();
    }
    
    const string& getHost() const { return host; }
    int getPort() const { return port; }
};

class ValidationException : public CustomException {
private:
    string fieldName;
    string invalidValue;

public:
    ValidationException(const string& field, const string& value, const string& reason)
        : CustomException(reason), fieldName(field), invalidValue(value) {
        ostringstream oss;
        oss << "Validation Error in field '" << fieldName 
            << "' with value '" << invalidValue << "': " << reason;
        message = oss.str();
    }
    
    const string& getFieldName() const { return fieldName; }
    const string& getInvalidValue() const { return invalidValue; }
};

// Classes that use custom exceptions
class FileManager {
public:
    void openFile(const string& filename) {
        if (filename.empty()) {
            throw FileException(filename, 101, "Filename cannot be empty");
        }
        if (filename.length() > 255) {
            throw FileException(filename, 102, "Filename too long");
        }
        if (filename.find("..") != string::npos) {
            throw FileException(filename, 103, "Invalid path traversal");
        }
        
        // Simulate file opening failure
        if (filename == "nonexistent.txt") {
            throw FileException(filename, 404, "File not found");
        }
        
        cout << "File '" << filename << "' opened successfully" << endl;
    }
};

class NetworkClient {
public:
    void connect(const string& host, int port) {
        if (host.empty()) {
            throw NetworkException(host, port, "Host cannot be empty");
        }
        if (port <= 0 || port > 65535) {
            throw NetworkException(host, port, "Invalid port number");
        }
        
        // Simulate connection failure
        if (host == "invalid.host.com") {
            throw NetworkException(host, port, "Host not reachable");
        }
        
        cout << "Connected to " << host << ":" << port << " successfully" << endl;
    }
};

class UserValidator {
public:
    void validateUser(const string& username, const string& email, int age) {
        if (username.empty()) {
            throw ValidationException("username", username, "Username cannot be empty");
        }
        if (username.length() < 3) {
            throw ValidationException("username", username, "Username must be at least 3 characters");
        }
        if (email.find('@') == string::npos) {
            throw ValidationException("email", email, "Email must contain @ symbol");
        }
        if (age < 0 || age > 150) {
            throw ValidationException("age", to_string(age), "Age must be between 0 and 150");
        }
        
        cout << "User validation successful for " << username << endl;
    }
};

int main() {
    cout << "=== Custom Exception Demo ===" << endl;
    
    FileManager fm;
    NetworkClient nc;
    UserValidator uv;
    
    // Test FileException
    vector<string> filenames = {"valid.txt", "", "verylongfilename" + string(300, 'x') + ".txt", 
                               "../../../etc/passwd", "nonexistent.txt"};
    
    for (const string& filename : filenames) {
        try {
            fm.openFile(filename);
        } catch (const FileException& e) {
            cout << "File Error: " << e.what() << endl;
            cout << "  Error Code: " << e.getErrorCode() << endl;
            cout << "  Filename: '" << e.getFilename() << "'" << endl;
        }
    }
    
    cout << "\n--- Network Tests ---" << endl;
    
    // Test NetworkException
    vector<pair<string, int>> connections = {
        {"google.com", 80},
        {"", 8080},
        {"localhost", -1},
        {"example.com", 70000},
        {"invalid.host.com", 443}
    };
    
    for (const auto& conn : connections) {
        try {
            nc.connect(conn.first, conn.second);
        } catch (const NetworkException& e) {
            cout << "Network Error: " << e.what() << endl;
            cout << "  Host: " << e.getHost() << endl;
            cout << "  Port: " << e.getPort() << endl;
        }
    }
    
    cout << "\n--- Validation Tests ---" << endl;
    
    // Test ValidationException
    vector<tuple<string, string, int>> users = {
        {"alice", "alice@example.com", 25},
        {"", "bob@example.com", 30},
        {"bo", "charlie@example.com", 35},
        {"dave", "invalid-email", 40},
        {"eve", "eve@example.com", -5},
        {"frank", "frank@example.com", 200}
    };
    
    for (const auto& user : users) {
        try {
            uv.validateUser(get<0>(user), get<1>(user), get<2>(user));
        } catch (const ValidationException& e) {
            cout << "Validation Error: " << e.what() << endl;
            cout << "  Field: " << e.getFieldName() << endl;
            cout << "  Value: " << e.getInvalidValue() << endl;
        }
    }
    
    return 0;
}
```

### 2. **Exception Safety and RAII**
```cpp
#include <iostream>
#include <memory>
#include <fstream>
#include <vector>
using namespace std;

// Resource management with RAII
class FileResource {
private:
    string filename;
    ofstream file;

public:
    FileResource(const string& name) : filename(name) {
        cout << "Opening file: " << filename << endl;
        file.open(filename);
        if (!file.is_open()) {
            throw runtime_error("Failed to open file: " + filename);
        }
    }
    
    ~FileResource() {
        if (file.is_open()) {
            cout << "Closing file: " << filename << endl;
            file.close();
        }
    }
    
    // Delete copy constructor and assignment to prevent copying
    FileResource(const FileResource&) = delete;
    FileResource& operator=(const FileResource&) = delete;
    
    void write(const string& data) {
        if (!file.is_open()) {
            throw runtime_error("File is not open");
        }
        file << data << endl;
        if (file.fail()) {
            throw runtime_error("Failed to write to file");
        }
    }
};

class MemoryResource {
private:
    unique_ptr<int[]> data;
    size_t size;

public:
    MemoryResource(size_t s) : size(s) {
        cout << "Allocating memory for " << size << " integers" << endl;
        data = make_unique<int[]>(size);
        if (!data) {
            throw bad_alloc();
        }
    }
    
    ~MemoryResource() {
        cout << "Deallocating memory" << endl;
        // Automatic cleanup with unique_ptr
    }
    
    void setData(size_t index, int value) {
        if (index >= size) {
            throw out_of_range("Index out of bounds");
        }
        data[index] = value;
    }
    
    int getData(size_t index) const {
        if (index >= size) {
            throw out_of_range("Index out of bounds");
        }
        return data[index];
    }
    
    size_t getSize() const { return size; }
};

// Exception-safe class design
class ExceptionSafeContainer {
private:
    vector<unique_ptr<string>> items;

public:
    // Strong exception safety guarantee
    void addItem(const string& item) {
        auto newItem = make_unique<string>(item);  // May throw bad_alloc
        items.push_back(move(newItem));            // Strong guarantee
    }
    
    // Basic exception safety guarantee
    void removeItem(size_t index) {
        if (index >= items.size()) {
            throw out_of_range("Index out of bounds");
        }
        items.erase(items.begin() + index);
    }
    
    // No-throw guarantee
    size_t size() const noexcept {
        return items.size();
    }
    
    // No-throw guarantee
    bool empty() const noexcept {
        return items.empty();
    }
    
    // Strong exception safety with copy-and-swap
    void swap(ExceptionSafeContainer& other) noexcept {
        items.swap(other.items);
    }
    
    const string& getItem(size_t index) const {
        if (index >= items.size()) {
            throw out_of_range("Index out of bounds");
        }
        return *items[index];
    }
};

// Function demonstrating exception safety levels
void demonstrateExceptionSafety() {
    cout << "\n=== Exception Safety Demo ===" << endl;
    
    try {
        // RAII ensures cleanup even if exceptions occur
        FileResource file("test_output.txt");
        MemoryResource memory(100);
        
        file.write("Starting processing...");
        
        // Fill memory with some data
        for (size_t i = 0; i < memory.getSize(); i++) {
            memory.setData(i, i * 2);
        }
        
        file.write("Memory initialized successfully");
        
        // Simulate an error condition
        if (memory.getData(50) == 100) {
            throw runtime_error("Simulated processing error");
        }
        
        file.write("Processing completed successfully");
        
    } catch (const exception& e) {
        cout << "Error during processing: " << e.what() << endl;
        cout << "Resources will be cleaned up automatically" << endl;
    }
    // File and memory are automatically cleaned up here
}

void demonstrateStrongExceptionSafety() {
    cout << "\n=== Strong Exception Safety Demo ===" << endl;
    
    ExceptionSafeContainer container;
    
    try {
        container.addItem("Item 1");
        container.addItem("Item 2");
        container.addItem("Item 3");
        
        cout << "Container has " << container.size() << " items" << endl;
        
        // This might throw an exception
        container.addItem("Item 4");
        
        cout << "Successfully added Item 4" << endl;
        
        // Try to access invalid index
        cout << "Accessing item at index 10: " << container.getItem(10) << endl;
        
    } catch (const exception& e) {
        cout << "Exception caught: " << e.what() << endl;
        cout << "Container state remains valid with " << container.size() << " items" << endl;
        
        // Container is still in a valid state
        for (size_t i = 0; i < container.size(); i++) {
            cout << "Item " << i << ": " << container.getItem(i) << endl;
        }
    }
}

int main() {
    demonstrateExceptionSafety();
    demonstrateStrongExceptionSafety();
    return 0;
}
```

## 🔄 Advanced Exception Handling

### 1. **Exception Specifications and noexcept**
```cpp
#include <iostream>
#include <vector>
#include <type_traits>
using namespace std;

// Function with noexcept specification
int safeAdd(int a, int b) noexcept {
    return a + b;
}

// Conditional noexcept
template<typename T>
void safeSwap(T& a, T& b) noexcept(is_nothrow_move_constructible_v<T> && 
                                   is_nothrow_move_assignable_v<T>) {
    T temp = move(a);
    a = move(b);
    b = move(temp);
}

// Function that might throw
double riskyDivision(double a, double b) {
    if (b == 0) {
        throw invalid_argument("Division by zero");
    }
    return a / b;
}

// Wrapper that converts exceptions to error codes
enum class ErrorCode {
    Success,
    DivisionByZero,
    UnknownError
};

pair<double, ErrorCode> safeDivision(double a, double b) noexcept {
    try {
        double result = riskyDivision(a, b);
        return {result, ErrorCode::Success};
    } catch (const invalid_argument&) {
        return {0.0, ErrorCode::DivisionByZero};
    } catch (...) {
        return {0.0, ErrorCode::UnknownError};
    }
}

class NoThrowClass {
private:
    int value;

public:
    NoThrowClass(int v) noexcept : value(v) {}
    
    NoThrowClass(const NoThrowClass& other) noexcept : value(other.value) {}
    
    NoThrowClass& operator=(const NoThrowClass& other) noexcept {
        value = other.value;
        return *this;
    }
    
    int getValue() const noexcept { return value; }
    
    void setValue(int v) noexcept { value = v; }
};

class MightThrowClass {
private:
    string data;

public:
    MightThrowClass(const string& d) : data(d) {
        if (d.empty()) {
            throw invalid_argument("Data cannot be empty");
        }
    }
    
    MightThrowClass(const MightThrowClass& other) : data(other.data) {}
    
    string getData() const { return data; }
};

void demonstrateNoexcept() {
    cout << "=== noexcept Demonstration ===" << endl;
    
    // Check if functions are noexcept
    cout << "safeAdd is noexcept: " << noexcept(safeAdd(1, 2)) << endl;
    cout << "riskyDivision is noexcept: " << noexcept(riskyDivision(1.0, 2.0)) << endl;
    
    // Test noexcept swap
    int a = 10, b = 20;
    cout << "Before swap: a=" << a << ", b=" << b << endl;
    safeSwap(a, b);
    cout << "After swap: a=" << a << ", b=" << b << endl;
    
    // Check noexcept properties of types
    cout << "\nType properties:" << endl;
    cout << "NoThrowClass move constructor is noexcept: " 
         << is_nothrow_move_constructible_v<NoThrowClass> << endl;
    cout << "MightThrowClass move constructor is noexcept: " 
         << is_nothrow_move_constructible_v<MightThrowClass> << endl;
    
    // Test safe division
    vector<pair<double, double>> testCases = {{10, 2}, {5, 0}, {7, 3}};
    
    for (const auto& test : testCases) {
        auto [result, error] = safeDivision(test.first, test.second);
        cout << test.first << " / " << test.second << " = ";
        
        switch (error) {
            case ErrorCode::Success:
                cout << result << endl;
                break;
            case ErrorCode::DivisionByZero:
                cout << "Error: Division by zero" << endl;
                break;
            case ErrorCode::UnknownError:
                cout << "Error: Unknown error" << endl;
                break;
        }
    }
}

int main() {
    demonstrateNoexcept();
    return 0;
}
```

### 2. **Exception Propagation and Rethrowing**
```cpp
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
using namespace std;

// Nested exception handling
class DatabaseConnection {
public:
    void connect() {
        throw runtime_error("Database connection failed");
    }
    
    void executeQuery(const string& query) {
        throw runtime_error("Query execution failed: " + query);
    }
};

class BusinessLogic {
private:
    DatabaseConnection db;

public:
    void processUser(const string& username) {
        try {
            db.connect();
            db.executeQuery("SELECT * FROM users WHERE name = '" + username + "'");
        } catch (const runtime_error& e) {
            // Add context and rethrow
            throw runtime_error("Failed to process user '" + username + "': " + e.what());
        }
    }
    
    void batchProcess(const vector<string>& usernames) {
        vector<string> failed_users;
        
        for (const string& username : usernames) {
            try {
                processUser(username);
                cout << "Successfully processed user: " << username << endl;
            } catch (const runtime_error& e) {
                cout << "Failed to process user " << username << ": " << e.what() << endl;
                failed_users.push_back(username);
            }
        }
        
        if (!failed_users.empty()) {
            string error_msg = "Batch processing completed with failures for users: ";
            for (size_t i = 0; i < failed_users.size(); ++i) {
                error_msg += failed_users[i];
                if (i < failed_users.size() - 1) error_msg += ", ";
            }
            throw runtime_error(error_msg);
        }
    }
};

// Exception aggregation
class TaskProcessor {
private:
    struct TaskResult {
        int taskId;
        bool success;
        string error;
        
        TaskResult(int id, bool s, const string& e = "") 
            : taskId(id), success(s), error(e) {}
    };

public:
    vector<TaskResult> processTasks(const vector<int>& taskIds) {
        vector<TaskResult> results;
        
        for (int taskId : taskIds) {
            try {
                processTask(taskId);
                results.emplace_back(taskId, true);
            } catch (const exception& e) {
                results.emplace_back(taskId, false, e.what());
            }
        }
        
        return results;
    }

private:
    void processTask(int taskId) {
        // Simulate different types of failures
        if (taskId % 3 == 0) {
            throw invalid_argument("Task " + to_string(taskId) + " has invalid parameters");
        }
        if (taskId % 5 == 0) {
            throw runtime_error("Task " + to_string(taskId) + " encountered runtime error");
        }
        if (taskId > 100) {
            throw out_of_range("Task ID " + to_string(taskId) + " is out of valid range");
        }
        
        // Simulate processing time
        this_thread::sleep_for(chrono::milliseconds(10));
        cout << "Task " << taskId << " completed successfully" << endl;
    }
};

// Exception chaining with nested exceptions (C++11)
void demonstrateNestedException() {
    cout << "\n=== Nested Exception Demo ===" << endl;
    
    try {
        try {
            throw runtime_error("Low-level database error");
        } catch (...) {
            throw_with_nested(runtime_error("High-level business logic error"));
        }
    } catch (const exception& e) {
        cout << "Caught exception: " << e.what() << endl;
        
        // Unpack nested exceptions
        auto nested = dynamic_cast<const nested_exception*>(&e);
        if (nested) {
            try {
                nested->rethrow_nested();
            } catch (const exception& nested_e) {
                cout << "Nested exception: " << nested_e.what() << endl;
            }
        }
    }
}

// Function to print exception hierarchy
void printException(const exception& e, int level = 0) {
    cout << string(level * 2, ' ') << "Exception: " << e.what() << endl;
    
    try {
        rethrow_if_nested(e);
    } catch (const exception& nested) {
        printException(nested, level + 1);
    }
}

int main() {
    cout << "=== Exception Propagation Demo ===" << endl;
    
    BusinessLogic bl;
    vector<string> users = {"alice", "bob", "charlie", "dave"};
    
    try {
        bl.batchProcess(users);
    } catch (const runtime_error& e) {
        cout << "\nBatch processing error: " << e.what() << endl;
    }
    
    cout << "\n=== Task Processing Demo ===" << endl;
    
    TaskProcessor processor;
    vector<int> taskIds = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 105};
    
    auto results = processor.processTasks(taskIds);
    
    cout << "\nTask Processing Results:" << endl;
    int successful = 0, failed = 0;
    
    for (const auto& result : results) {
        if (result.success) {
            successful++;
        } else {
            failed++;
            cout << "Task " << result.taskId << " failed: " << result.error << endl;
        }
    }
    
    cout << "\nSummary: " << successful << " successful, " << failed << " failed" << endl;
    
    demonstrateNestedException();
    
    // Demonstrate exception hierarchy printing
    cout << "\n=== Exception Hierarchy Demo ===" << endl;
    try {
        try {
            try {
                throw invalid_argument("Inner exception");
            } catch (...) {
                throw_with_nested(runtime_error("Middle exception"));
            }
        } catch (...) {
            throw_with_nested(logic_error("Outer exception"));
        }
    } catch (const exception& e) {
        printException(e);
    }
    
    return 0;
}
```

## 🎯 Practical Examples

### 1. **Configuration Parser with Exception Handling**
```cpp
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <regex>
using namespace std;

class ConfigException : public exception {
private:
    string message;

public:
    ConfigException(const string& msg) : message(msg) {}
    const char* what() const noexcept override { return message.c_str(); }
};

class ConfigParser {
private:
    map<string, string> config;
    string filename;

public:
    ConfigParser(const string& file) : filename(file) {}
    
    void load() {
        ifstream file(filename);
        if (!file.is_open()) {
            throw ConfigException("Cannot open config file: " + filename);
        }
        
        string line;
        int lineNumber = 0;
        
        try {
            while (getline(file, line)) {
                lineNumber++;
                parseLine(line, lineNumber);
            }
        } catch (const ConfigException& e) {
            throw ConfigException("Error in " + filename + " at line " + 
                                to_string(lineNumber) + ": " + e.what());
        }
        
        cout << "Successfully loaded " << config.size() << " configuration entries" << endl;
    }
    
    string get(const string& key) const {
        auto it = config.find(key);
        if (it == config.end()) {
            throw ConfigException("Configuration key not found: " + key);
        }
        return it->second;
    }
    
    template<typename T>
    T get(const string& key) const {
        string value = get(key);
        
        try {
            if constexpr (is_same_v<T, int>) {
                return stoi(value);
            } else if constexpr (is_same_v<T, double>) {
                return stod(value);
            } else if constexpr (is_same_v<T, bool>) {
                if (value == "true" || value == "1") return true;
                if (value == "false" || value == "0") return false;
                throw invalid_argument("Invalid boolean value");
            } else {
                return value;  // String type
            }
        } catch (const invalid_argument&) {
            throw ConfigException("Cannot convert value '" + value + 
                                "' to requested type for key: " + key);
        }
    }
    
    bool has(const string& key) const {
        return config.find(key) != config.end();
    }
    
    void printAll() const {
        cout << "Configuration entries:" << endl;
        for (const auto& pair : config) {
            cout << "  " << pair.first << " = " << pair.second << endl;
        }
    }

private:
    void parseLine(const string& line, int lineNumber) {
        // Skip empty lines and comments
        string trimmed = trim(line);
        if (trimmed.empty() || trimmed[0] == '#') {
            return;
        }
        
        // Parse key=value format
        regex pattern(R"(^\s*([^=\s]+)\s*=\s*(.+?)\s*$)");
        smatch matches;
        
        if (!regex_match(trimmed, matches, pattern)) {
            throw ConfigException("Invalid syntax, expected key=value format");
        }
        
        string key = matches[1];
        string value = matches[2];
        
        // Remove quotes if present
        if (value.length() >= 2 && value[0] == '"' && value.back() == '"') {
            value = value.substr(1, value.length() - 2);
        }
        
        config[key] = value;
    }
    
    string trim(const string& str) const {
        size_t start = str.find_first_not_of(" \t\r\n");
        if (start == string::npos) return "";
        
        size_t end = str.find_last_not_of(" \t\r\n");
        return str.substr(start, end - start + 1);
    }
};

// Application configuration manager
class AppConfig {
private:
    ConfigParser parser;
    
public:
    AppConfig(const string& configFile) : parser(configFile) {
        try {
            parser.load();
        } catch (const ConfigException& e) {
            throw ConfigException("Failed to initialize application configuration: " + string(e.what()));
        }
    }
    
    string getDatabaseUrl() const {
        return parser.get("database.url");
    }
    
    int getDatabasePort() const {
        return parser.get<int>("database.port");
    }
    
    string getLogLevel() const {
        return parser.has("log.level") ? parser.get("log.level") : "INFO";
    }
    
    int getMaxConnections() const {
        return parser.has("max.connections") ? parser.get<int>("max.connections") : 100;
    }
    
    bool isDebugMode() const {
        return parser.has("debug.enabled") ? parser.get<bool>("debug.enabled") : false;
    }
    
    void validateConfiguration() const {
        vector<string> requiredKeys = {
            "database.url",
            "database.port",
            "app.name"
        };
        
        for (const string& key : requiredKeys) {
            if (!parser.has(key)) {
                throw ConfigException("Required configuration key missing: " + key);
            }
        }
        
        // Validate port range
        int port = getDatabasePort();
        if (port <= 0 || port > 65535) {
            throw ConfigException("Database port must be between 1 and 65535");
        }
        
        // Validate log level
        string logLevel = getLogLevel();
        vector<string> validLevels = {"DEBUG", "INFO", "WARN", "ERROR"};
        if (find(validLevels.begin(), validLevels.end(), logLevel) == validLevels.end()) {
            throw ConfigException("Invalid log level: " + logLevel);
        }
        
        cout << "Configuration validation passed" << endl;
    }
};

// Create a sample config file for testing
void createSampleConfig() {
    ofstream file("app.config");
    file << "# Application Configuration\n";
    file << "app.name = \"My Application\"\n";
    file << "app.version = \"1.0.0\"\n";
    file << "\n";
    file << "# Database settings\n";
    file << "database.url = \"localhost\"\n";
    file << "database.port = 5432\n";
    file << "database.name = \"myapp_db\"\n";
    file << "\n";
    file << "# Logging\n";
    file << "log.level = INFO\n";
    file << "log.file = \"/var/log/myapp.log\"\n";
    file << "\n";
    file << "# Other settings\n";
    file << "max.connections = 50\n";
    file << "debug.enabled = false\n";
}

int main() {
    cout << "=== Configuration Parser Demo ===" << endl;
    
    // Create sample config file
    createSampleConfig();
    
    try {
        AppConfig config("app.config");
        
        cout << "\nConfiguration loaded successfully!" << endl;
        cout << "App Name: " << config.parser.get("app.name") << endl;
        cout << "Database URL: " << config.getDatabaseUrl() << endl;
        cout << "Database Port: " << config.getDatabasePort() << endl;
        cout << "Log Level: " << config.getLogLevel() << endl;
        cout << "Max Connections: " << config.getMaxConnections() << endl;
        cout << "Debug Mode: " << (config.isDebugMode() ? "enabled" : "disabled") << endl;
        
        config.validateConfiguration();
        
    } catch (const ConfigException& e) {
        cout << "Configuration Error: " << e.what() << endl;
        return 1;
    }
    
    // Test error cases
    cout << "\n=== Testing Error Cases ===" << endl;
    
    try {
        AppConfig badConfig("nonexistent.config");
    } catch (const ConfigException& e) {
        cout << "Expected error: " << e.what() << endl;
    }
    
    try {
        ConfigParser parser("app.config");
        parser.load();
        
        // Try to get non-existent key
        string value = parser.get("nonexistent.key");
    } catch (const ConfigException& e) {
        cout << "Expected error: " << e.what() << endl;
    }
    
    try {
        ConfigParser parser("app.config");
        parser.load();
        
        // Try to convert string to int
        int value = parser.get<int>("app.name");
    } catch (const ConfigException& e) {
        cout << "Expected error: " << e.what() << endl;
    }
    
    return 0;
}
```

### 2. **Network Client with Robust Error Handling**
```cpp
#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <random>
using namespace std;

// Network-related exceptions
class NetworkException : public exception {
protected:
    string message;

public:
    NetworkException(const string& msg) : message(msg) {}
    const char* what() const noexcept override { return message.c_str(); }
};

class ConnectionException : public NetworkException {
public:
    ConnectionException(const string& host, int port, const string& reason)
        : NetworkException("Connection failed to " + host + ":" + to_string(port) + " - " + reason) {}
};

class TimeoutException : public NetworkException {
public:
    TimeoutException(int timeoutMs) 
        : NetworkException("Operation timed out after " + to_string(timeoutMs) + "ms") {}
};

class ProtocolException : public NetworkException {
public:
    ProtocolException(const string& operation, const string& details)
        : NetworkException("Protocol error during " + operation + ": " + details) {}
};

// Retry policy
struct RetryPolicy {
    int maxRetries;
    int baseDelayMs;
    double backoffMultiplier;
    int maxDelayMs;
    
    RetryPolicy(int retries = 3, int delay = 100, double multiplier = 2.0, int maxDelay = 5000)
        : maxRetries(retries), baseDelayMs(delay), backoffMultiplier(multiplier), maxDelayMs(maxDelay) {}
    
    int getDelay(int attempt) const {
        int delay = baseDelayMs * pow(backoffMultiplier, attempt);
        return min(delay, maxDelayMs);
    }
};

// Network client with exception handling
class NetworkClient {
private:
    string host;
    int port;
    bool connected;
    RetryPolicy retryPolicy;
    mutable mt19937 rng;

public:
    NetworkClient(const string& h, int p, const RetryPolicy& policy = RetryPolicy())
        : host(h), port(p), connected(false), retryPolicy(policy), rng(random_device{}()) {}
    
    void connect() {
        if (connected) {
            throw logic_error("Already connected");
        }
        
        executeWithRetry("connect", [this]() {
            simulateConnection();
        });
        
        connected = true;
        cout << "Successfully connected to " << host << ":" << port << endl;
    }
    
    void disconnect() {
        if (!connected) {
            cout << "Already disconnected" << endl;
            return;
        }
        
        try {
            simulateDisconnection();
            connected = false;
            cout << "Disconnected from " << host << ":" << port << endl;
        } catch (const NetworkException& e) {
            cout << "Warning during disconnect: " << e.what() << endl;
            connected = false;  // Force disconnection even if there's an error
        }
    }
    
    string sendRequest(const string& request) {
        if (!connected) {
            throw logic_error("Not connected to server");
        }
        
        string response;
        executeWithRetry("send request", [&]() {
            response = simulateRequest(request);
        });
        
        return response;
    }
    
    bool isConnected() const {
        return connected;
    }
    
    ~NetworkClient() {
        try {
            if (connected) {
                disconnect();
            }
        } catch (...) {
            // Destructor should not throw
        }
    }

private:
    void simulateConnection() {
        // Simulate various connection failures
        uniform_int_distribution<int> dist(1, 10);
        int result = dist(rng);
        
        if (result <= 2) {
            throw ConnectionException(host, port, "Host unreachable");
        } else if (result <= 4) {
            throw TimeoutException(5000);
        } else if (result <= 5) {
            throw ConnectionException(host, port, "Connection refused");
        }
        
        // Simulate connection time
        this_thread::sleep_for(chrono::milliseconds(100));
    }
    
    void simulateDisconnection() {
        uniform_int_distribution<int> dist(1, 10);
        if (dist(rng) <= 1) {
            throw NetworkException("Error during disconnection");
        }
        
        this_thread::sleep_for(chrono::milliseconds(50));
    }
    
    string simulateRequest(const string& request) {
        uniform_int_distribution<int> dist(1, 10);
        int result = dist(rng);
        
        if (result <= 1) {
            throw TimeoutException(3000);
        } else if (result <= 2) {
            throw ProtocolException("send", "Invalid request format");
        } else if (result <= 3) {
            throw NetworkException("Connection lost during request");
        }
        
        // Simulate processing time
        this_thread::sleep_for(chrono::milliseconds(50));
        
        return "Response to: " + request;
    }
    
    template<typename Func>
    void executeWithRetry(const string& operation, Func func) {
        int attempt = 0;
        
        while (attempt <= retryPolicy.maxRetries) {
            try {
                func();
                if (attempt > 0) {
                    cout << "Operation '" << operation << "' succeeded on attempt " << (attempt + 1) << endl;
                }
                return;
            } catch (const NetworkException& e) {
                attempt++;
                
                if (attempt > retryPolicy.maxRetries) {
                    throw NetworkException("Operation '" + operation + "' failed after " + 
                                         to_string(retryPolicy.maxRetries + 1) + " attempts. Last error: " + e.what());
                }
                
                int delay = retryPolicy.getDelay(attempt - 1);
                cout << "Attempt " << attempt << " failed for '" << operation << "': " << e.what() 
                     << ". Retrying in " << delay << "ms..." << endl;
                
                this_thread::sleep_for(chrono::milliseconds(delay));
            }
        }
    }
};

// Connection pool with exception handling
class ConnectionPool {
private:
    vector<unique_ptr<NetworkClient>> connections;
    string host;
    int port;
    size_t maxSize;
    RetryPolicy retryPolicy;

public:
    ConnectionPool(const string& h, int p, size_t maxConnections = 5)
        : host(h), port(p), maxSize(maxConnections) {}
    
    shared_ptr<NetworkClient> getConnection() {
        // Try to reuse existing connection
        for (auto it = connections.begin(); it != connections.end(); ++it) {
            if (*it && (*it)->isConnected()) {
                auto connection = shared_ptr<NetworkClient>(it->release());
                connections.erase(it);
                return connection;
            }
        }
        
        // Create new connection
        auto connection = make_unique<NetworkClient>(host, port, retryPolicy);
        
        try {
            connection->connect();
            return shared_ptr<NetworkClient>(connection.release());
        } catch (const NetworkException& e) {
            throw ConnectionException(host, port, "Pool failed to create connection: " + string(e.what()));
        }
    }
    
    void returnConnection(shared_ptr<NetworkClient> connection) {
        if (connection && connection->isConnected() && connections.size() < maxSize) {
            connections.push_back(unique_ptr<NetworkClient>(connection.get()));
            connection.reset();  // Release shared ownership
        }
    }
    
    size_t size() const {
        return connections.size();
    }
    
    void closeAll() {
        for (auto& conn : connections) {
            if (conn) {
                try {
                    conn->disconnect();
                } catch (const exception& e) {
                    cout << "Error closing connection: " << e.what() << endl;
                }
            }
        }
        connections.clear();
    }
    
    ~ConnectionPool() {
        closeAll();
    }
};

int main() {
    cout << "=== Network Client Demo ===" << endl;
    
    // Test individual client
    try {
        RetryPolicy policy(3, 200, 1.5, 2000);  // 3 retries, exponential backoff
        NetworkClient client("example.com", 8080, policy);
        
        client.connect();
        
        vector<string> requests = {"GET /api/data", "POST /api/update", "GET /api/status"};
        
        for (const string& request : requests) {
            try {
                string response = client.sendRequest(request);
                cout << "Request: " << request << " -> Response: " << response << endl;
            } catch (const NetworkException& e) {
                cout << "Request failed: " << e.what() << endl;
            }
        }
        
        client.disconnect();
        
    } catch (const NetworkException& e) {
        cout << "Network error: " << e.what() << endl;
    } catch (const exception& e) {
        cout << "Unexpected error: " << e.what() << endl;
    }
    
    cout << "\n=== Connection Pool Demo ===" << endl;
    
    try {
        ConnectionPool pool("api.example.com", 443, 3);
        
        vector<shared_ptr<NetworkClient>> activeConnections;
        
        // Get multiple connections
        for (int i = 0; i < 3; i++) {
            try {
                auto conn = pool.getConnection();
                activeConnections.push_back(conn);
                cout << "Obtained connection " << (i + 1) << endl;
            } catch (const NetworkException& e) {
                cout << "Failed to get connection " << (i + 1) << ": " << e.what() << endl;
            }
        }
        
        // Use connections
        for (size_t i = 0; i < activeConnections.size(); i++) {
            try {
                string response = activeConnections[i]->sendRequest("GET /api/endpoint" + to_string(i));
                cout << "Connection " << (i + 1) << " response: " << response << endl;
            } catch (const NetworkException& e) {
                cout << "Request failed on connection " << (i + 1) << ": " << e.what() << endl;
            }
        }
        
        // Return connections to pool
        for (auto& conn : activeConnections) {
            pool.returnConnection(conn);
        }
        
        cout << "Pool size after returning connections: " << pool.size() << endl;
        
    } catch (const exception& e) {
        cout << "Pool error: " << e.what() << endl;
    }
    
    return 0;
}
```

## 💡 Best Practices

### Exception Handling Guidelines
```cpp
#include <iostream>
#include <memory>
#include <vector>
using namespace std;

// 1. Use RAII for resource management
class GoodResourceManager {
private:
    unique_ptr<int[]> data;
    size_t size;

public:
    GoodResourceManager(size_t s) : size(s) {
        data = make_unique<int[]>(size);
        cout << "Resource allocated" << endl;
    }
    
    ~GoodResourceManager() {
        cout << "Resource automatically cleaned up" << endl;
    }
    
    void doWork() {
        // Even if this throws, destructor will clean up
        if (size == 0) {
            throw invalid_argument("Cannot work with zero size");
        }
        
        for (size_t i = 0; i < size; i++) {
            data[i] = i;
        }
    }
};

// 2. Catch exceptions by const reference
void demonstrateCatchByReference() {
    try {
        throw runtime_error("Test exception");
    } catch (const exception& e) {  // Good: catch by const reference
        cout << "Caught: " << e.what() << endl;
    }
    // Bad: catch (exception e) - copies the exception object
}

// 3. Order catch blocks from most specific to most general
void demonstrateCatchOrdering() {
    try {
        throw out_of_range("Index out of bounds");
    } catch (const out_of_range& e) {      // Most specific first
        cout << "Out of range: " << e.what() << endl;
    } catch (const logic_error& e) {       // More general
        cout << "Logic error: " << e.what() << endl;
    } catch (const exception& e) {         // Most general last
        cout << "General exception: " << e.what() << endl;
    }
}

// 4. Use appropriate exception types
enum class ValidationError {
    Empty,
    TooShort,
    TooLong,
    InvalidCharacters
};

class ValidationException : public invalid_argument {
private:
    ValidationError errorType;

public:
    ValidationException(ValidationError type, const string& field, const string& msg)
        : invalid_argument("Validation error in " + field + ": " + msg), errorType(type) {}
    
    ValidationError getErrorType() const { return errorType; }
};

void validateUsername(const string& username) {
    if (username.empty()) {
        throw ValidationException(ValidationError::Empty, "username", "Username cannot be empty");
    }
    if (username.length() < 3) {
        throw ValidationException(ValidationError::TooShort, "username", "Username too short");
    }
    if (username.length() > 20) {
        throw ValidationException(ValidationError::TooLong, "username", "Username too long");
    }
}

// 5. Provide strong exception safety guarantees
class ExceptionSafeVector {
private:
    vector<unique_ptr<string>> data;

public:
    // Strong exception safety: either succeeds completely or leaves object unchanged
    void addItems(const vector<string>& items) {
        vector<unique_ptr<string>> temp;  // Temporary storage
        
        // Prepare all items first (may throw)
        for (const string& item : items) {
            temp.push_back(make_unique<string>(item));
        }
        
        // Only modify the actual data if everything succeeded
        for (auto& item : temp) {
            data.push_back(move(item));
        }
    }
    
    size_t size() const noexcept { return data.size(); }
    
    const string& get(size_t index) const {
        if (index >= data.size()) {
            throw out_of_range("Index out of bounds");
        }
        return *data[index];
    }
};

// 6. Don't throw exceptions from destructors
class SafeDestructor {
private:
    bool hasResource;

public:
    SafeDestructor() : hasResource(true) {}
    
    ~SafeDestructor() {
        try {
            if (hasResource) {
                // Cleanup that might fail
                cleanup();
            }
        } catch (...) {
            // Log the error but don't rethrow
            cout << "Error during cleanup (logged, not thrown)" << endl;
        }
    }

private:
    void cleanup() {
        // Simulated cleanup that might throw
        hasResource = false;
    }
};

int main() {
    cout << "=== Exception Handling Best Practices ===" << endl;
    
    // RAII demonstration
    try {
        GoodResourceManager manager(0);  // Will throw
        manager.doWork();
    } catch (const invalid_argument& e) {
        cout << "Exception caught, but resources were cleaned up automatically" << endl;
    }
    
    // Catch by reference
    demonstrateCatchByReference();
    
    // Proper catch ordering
    demonstrateCatchOrdering();
    
    // Appropriate exception types
    vector<string> testUsernames = {"", "ab", "validuser", string(25, 'x')};
    
    for (const string& username : testUsernames) {
        try {
            validateUsername(username);
            cout << "Username '" << username << "' is valid" << endl;
        } catch (const ValidationException& e) {
            cout << "Validation failed: " << e.what() << endl;
        }
    }
    
    // Exception safety
    try {
        ExceptionSafeVector vec;
        vector<string> items1 = {"item1", "item2", "item3"};
        vec.addItems(items1);
        cout << "Added " << vec.size() << " items successfully" << endl;
        
        // This might fail, but vec remains in valid state
        vector<string> items2 = {"item4", "item5"};
        vec.addItems(items2);
        cout << "Total items: " << vec.size() << endl;
        
    } catch (const exception& e) {
        cout << "Exception during vector operations: " << e.what() << endl;
    }
    
    // Safe destructor
    {
        SafeDestructor obj;
        // Destructor will be called here and won't throw
    }
    
    return 0;
}
```

## 🔗 Related Topics
- [Functions](./03-functions.md)
- [OOP Basics](./08-oop.md)
- [Templates](./12-templates.md)
- [File I/O](./14-file-io.md)

---
*Previous: [Templates](./12-templates.md) | Next: [File I/O](./14-file-io.md)*
