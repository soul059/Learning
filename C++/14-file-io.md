# 14. File I/O

## 📋 Overview
File Input/Output operations in C++ are handled through the iostream library, which provides powerful stream-based I/O capabilities. C++ offers both C-style file operations and modern stream-based approaches for reading from and writing to files.

## 📁 Basic File Operations

### 1. **File Streams - ifstream, ofstream, fstream**
```cpp
#include <iostream>
#include <fstream>
#include <string>
using namespace std;

void demonstrateBasicFileIO() {
    cout << "=== Basic File I/O Operations ===" << endl;
    
    // Writing to a file using ofstream
    {
        ofstream outFile("example.txt");
        
        if (!outFile.is_open()) {
            cout << "Error: Could not open file for writing" << endl;
            return;
        }
        
        outFile << "Hello, File I/O!" << endl;
        outFile << "This is line 2" << endl;
        outFile << "Numbers: " << 42 << " " << 3.14159 << endl;
        
        cout << "Data written to example.txt" << endl;
    }  // File automatically closed when outFile goes out of scope
    
    // Reading from a file using ifstream
    {
        ifstream inFile("example.txt");
        
        if (!inFile.is_open()) {
            cout << "Error: Could not open file for reading" << endl;
            return;
        }
        
        string line;
        cout << "\nReading from example.txt:" << endl;
        
        while (getline(inFile, line)) {
            cout << line << endl;
        }
    }
    
    // Using fstream for both reading and writing
    {
        fstream file("example.txt", ios::in | ios::out | ios::app);
        
        if (!file.is_open()) {
            cout << "Error: Could not open file for read/write" << endl;
            return;
        }
        
        // Append more data
        file << "Appended line" << endl;
        
        // Go back to beginning to read
        file.seekg(0, ios::beg);
        
        string line;
        cout << "\nFile contents after appending:" << endl;
        
        while (getline(file, line)) {
            cout << line << endl;
        }
    }
}

// File opening modes demonstration
void demonstrateFileModes() {
    cout << "\n=== File Opening Modes ===" << endl;
    
    // Create initial file
    {
        ofstream file("modes_test.txt");
        file << "Original content" << endl;
        file << "Line 2" << endl;
    }
    
    // ios::app - Append mode
    {
        ofstream file("modes_test.txt", ios::app);
        file << "Appended content" << endl;
    }
    
    // ios::trunc - Truncate mode (default for ofstream)
    {
        ofstream file("modes_test.txt", ios::trunc);
        file << "This replaces all previous content" << endl;
    }
    
    // ios::ate - At end mode
    {
        ofstream file("modes_test.txt", ios::ate);
        file << "Added at end" << endl;
    }
    
    // Read and display final content
    {
        ifstream file("modes_test.txt");
        string line;
        cout << "Final content of modes_test.txt:" << endl;
        while (getline(file, line)) {
            cout << line << endl;
        }
    }
}

int main() {
    demonstrateBasicFileIO();
    demonstrateFileModes();
    return 0;
}
```

### 2. **Error Handling in File Operations**
```cpp
#include <iostream>
#include <fstream>
#include <stdexcept>
using namespace std;

class FileException : public exception {
private:
    string message;

public:
    FileException(const string& msg) : message(msg) {}
    const char* what() const noexcept override {
        return message.c_str();
    }
};

class SafeFileHandler {
public:
    static void writeToFile(const string& filename, const string& content) {
        ofstream file(filename);
        
        if (!file.is_open()) {
            throw FileException("Cannot open file for writing: " + filename);
        }
        
        file << content;
        
        if (file.fail()) {
            throw FileException("Error writing to file: " + filename);
        }
        
        cout << "Successfully wrote to " << filename << endl;
    }
    
    static string readFromFile(const string& filename) {
        ifstream file(filename);
        
        if (!file.is_open()) {
            throw FileException("Cannot open file for reading: " + filename);
        }
        
        string content;
        string line;
        
        while (getline(file, line)) {
            content += line + "\n";
        }
        
        if (file.bad()) {
            throw FileException("Error reading from file: " + filename);
        }
        
        return content;
    }
    
    static bool fileExists(const string& filename) {
        ifstream file(filename);
        return file.good();
    }
    
    static size_t getFileSize(const string& filename) {
        ifstream file(filename, ios::binary | ios::ate);
        
        if (!file.is_open()) {
            throw FileException("Cannot open file to get size: " + filename);
        }
        
        return file.tellg();
    }
};

void demonstrateErrorHandling() {
    cout << "=== File Error Handling ===" << endl;
    
    try {
        // Test writing
        SafeFileHandler::writeToFile("test_output.txt", "Hello, safe file I/O!\nThis is a test file.\n");
        
        // Test reading
        string content = SafeFileHandler::readFromFile("test_output.txt");
        cout << "File content:\n" << content << endl;
        
        // Test file size
        size_t size = SafeFileHandler::getFileSize("test_output.txt");
        cout << "File size: " << size << " bytes" << endl;
        
        // Test file existence
        cout << "File exists: " << (SafeFileHandler::fileExists("test_output.txt") ? "Yes" : "No") << endl;
        cout << "Non-existent file: " << (SafeFileHandler::fileExists("nonexistent.txt") ? "Yes" : "No") << endl;
        
        // Test error case
        SafeFileHandler::readFromFile("nonexistent.txt");
        
    } catch (const FileException& e) {
        cout << "File error: " << e.what() << endl;
    }
}

int main() {
    demonstrateErrorHandling();
    return 0;
}
```

## 📊 Reading and Writing Different Data Types

### 1. **Binary File I/O**
```cpp
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
using namespace std;

struct Person {
    char name[50];
    int age;
    double salary;
    
    Person() = default;
    
    Person(const string& n, int a, double s) : age(a), salary(s) {
        strncpy(name, n.c_str(), sizeof(name) - 1);
        name[sizeof(name) - 1] = '\0';
    }
    
    void display() const {
        cout << "Name: " << name << ", Age: " << age << ", Salary: $" << salary << endl;
    }
};

class BinaryFileManager {
public:
    static void writePersons(const string& filename, const vector<Person>& persons) {
        ofstream file(filename, ios::binary);
        
        if (!file.is_open()) {
            throw runtime_error("Cannot open file for binary writing: " + filename);
        }
        
        // Write number of persons first
        size_t count = persons.size();
        file.write(reinterpret_cast<const char*>(&count), sizeof(count));
        
        // Write each person
        for (const auto& person : persons) {
            file.write(reinterpret_cast<const char*>(&person), sizeof(Person));
        }
        
        if (file.fail()) {
            throw runtime_error("Error writing binary data to: " + filename);
        }
        
        cout << "Successfully wrote " << count << " persons to " << filename << endl;
    }
    
    static vector<Person> readPersons(const string& filename) {
        ifstream file(filename, ios::binary);
        
        if (!file.is_open()) {
            throw runtime_error("Cannot open file for binary reading: " + filename);
        }
        
        // Read number of persons
        size_t count;
        file.read(reinterpret_cast<char*>(&count), sizeof(count));
        
        if (file.fail()) {
            throw runtime_error("Error reading count from: " + filename);
        }
        
        vector<Person> persons;
        persons.reserve(count);
        
        // Read each person
        for (size_t i = 0; i < count; i++) {
            Person person;
            file.read(reinterpret_cast<char*>(&person), sizeof(Person));
            
            if (file.fail()) {
                throw runtime_error("Error reading person data from: " + filename);
            }
            
            persons.push_back(person);
        }
        
        cout << "Successfully read " << count << " persons from " << filename << endl;
        return persons;
    }
    
    static void appendPerson(const string& filename, const Person& person) {
        // First, read existing data
        vector<Person> persons;
        
        try {
            persons = readPersons(filename);
        } catch (const runtime_error&) {
            // File might not exist, start with empty vector
        }
        
        // Add new person
        persons.push_back(person);
        
        // Write all data back
        writePersons(filename, persons);
    }
};

void demonstrateBinaryIO() {
    cout << "=== Binary File I/O ===" << endl;
    
    try {
        // Create sample data
        vector<Person> persons = {
            Person("Alice Johnson", 30, 75000.0),
            Person("Bob Smith", 25, 65000.0),
            Person("Charlie Brown", 35, 85000.0),
            Person("Diana Prince", 28, 90000.0)
        };
        
        // Write to binary file
        BinaryFileManager::writePersons("employees.bin", persons);
        
        // Read from binary file
        vector<Person> readPersons = BinaryFileManager::readPersons("employees.bin");
        
        cout << "\nPersons read from file:" << endl;
        for (const auto& person : readPersons) {
            person.display();
        }
        
        // Append a new person
        cout << "\nAppending new person..." << endl;
        BinaryFileManager::appendPerson("employees.bin", Person("Eve Wilson", 32, 95000.0));
        
        // Read again to verify
        readPersons = BinaryFileManager::readPersons("employees.bin");
        cout << "\nUpdated file contents:" << endl;
        for (const auto& person : readPersons) {
            person.display();
        }
        
    } catch (const runtime_error& e) {
        cout << "Binary I/O error: " << e.what() << endl;
    }
}

int main() {
    demonstrateBinaryIO();
    return 0;
}
```

### 2. **Formatted Text I/O**
```cpp
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <sstream>
using namespace std;

struct Product {
    int id;
    string name;
    double price;
    int quantity;
    
    Product() = default;
    Product(int i, const string& n, double p, int q) : id(i), name(n), price(p), quantity(q) {}
};

class FormattedFileManager {
public:
    static void writeProducts(const string& filename, const vector<Product>& products) {
        ofstream file(filename);
        
        if (!file.is_open()) {
            throw runtime_error("Cannot open file for writing: " + filename);
        }
        
        // Write header
        file << "# Product Inventory" << endl;
        file << "# Format: ID|Name|Price|Quantity" << endl;
        file << "# Generated on: " << getCurrentTimestamp() << endl;
        file << "---" << endl;
        
        // Write products with formatting
        for (const auto& product : products) {
            file << product.id << "|" 
                 << product.name << "|" 
                 << fixed << setprecision(2) << product.price << "|" 
                 << product.quantity << endl;
        }
        
        cout << "Wrote " << products.size() << " products to " << filename << endl;
    }
    
    static vector<Product> readProducts(const string& filename) {
        ifstream file(filename);
        
        if (!file.is_open()) {
            throw runtime_error("Cannot open file for reading: " + filename);
        }
        
        vector<Product> products;
        string line;
        
        // Skip header lines (until "---")
        while (getline(file, line)) {
            if (line == "---") break;
        }
        
        // Read product data
        while (getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;  // Skip empty lines and comments
            
            Product product = parseProductLine(line);
            products.push_back(product);
        }
        
        cout << "Read " << products.size() << " products from " << filename << endl;
        return products;
    }
    
    static void writeFormattedReport(const string& filename, const vector<Product>& products) {
        ofstream file(filename);
        
        if (!file.is_open()) {
            throw runtime_error("Cannot open file for writing report: " + filename);
        }
        
        // Write formatted report
        file << "PRODUCT INVENTORY REPORT" << endl;
        file << "========================" << endl;
        file << "Generated: " << getCurrentTimestamp() << endl << endl;
        
        file << left << setw(5) << "ID" 
             << left << setw(20) << "Product Name" 
             << right << setw(10) << "Price" 
             << right << setw(10) << "Quantity" 
             << right << setw(12) << "Total Value" << endl;
        
        file << string(67, '-') << endl;
        
        double grandTotal = 0.0;
        
        for (const auto& product : products) {
            double totalValue = product.price * product.quantity;
            grandTotal += totalValue;
            
            file << left << setw(5) << product.id
                 << left << setw(20) << product.name
                 << right << setw(9) << fixed << setprecision(2) << "$" << product.price
                 << right << setw(10) << product.quantity
                 << right << setw(11) << fixed << setprecision(2) << "$" << totalValue << endl;
        }
        
        file << string(67, '-') << endl;
        file << right << setw(55) << "GRAND TOTAL: $" << fixed << setprecision(2) << grandTotal << endl;
        
        cout << "Generated formatted report: " << filename << endl;
    }

private:
    static Product parseProductLine(const string& line) {
        stringstream ss(line);
        string item;
        Product product;
        
        // Parse ID
        if (!getline(ss, item, '|')) {
            throw runtime_error("Invalid product line format: " + line);
        }
        product.id = stoi(item);
        
        // Parse Name
        if (!getline(ss, product.name, '|')) {
            throw runtime_error("Invalid product line format: " + line);
        }
        
        // Parse Price
        if (!getline(ss, item, '|')) {
            throw runtime_error("Invalid product line format: " + line);
        }
        product.price = stod(item);
        
        // Parse Quantity
        if (!getline(ss, item)) {
            throw runtime_error("Invalid product line format: " + line);
        }
        product.quantity = stoi(item);
        
        return product;
    }
    
    static string getCurrentTimestamp() {
        // Simplified timestamp - in real code, use proper date/time library
        return "2024-01-15 10:30:00";
    }
};

void demonstrateFormattedIO() {
    cout << "=== Formatted Text I/O ===" << endl;
    
    try {
        // Create sample products
        vector<Product> products = {
            Product(101, "Wireless Mouse", 29.99, 50),
            Product(102, "Mechanical Keyboard", 89.95, 25),
            Product(103, "USB Cable", 12.50, 100),
            Product(104, "Monitor Stand", 45.00, 15),
            Product(105, "Webcam HD", 79.99, 30)
        };
        
        // Write products to file
        FormattedFileManager::writeProducts("inventory.txt", products);
        
        // Read products back
        vector<Product> readProducts = FormattedFileManager::readProducts("inventory.txt");
        
        cout << "\nProducts read from file:" << endl;
        for (const auto& product : readProducts) {
            cout << "ID: " << product.id 
                 << ", Name: " << product.name 
                 << ", Price: $" << fixed << setprecision(2) << product.price 
                 << ", Qty: " << product.quantity << endl;
        }
        
        // Generate formatted report
        FormattedFileManager::writeFormattedReport("inventory_report.txt", readProducts);
        
        // Display the report
        cout << "\nGenerated report content:" << endl;
        ifstream reportFile("inventory_report.txt");
        string line;
        while (getline(reportFile, line)) {
            cout << line << endl;
        }
        
    } catch (const exception& e) {
        cout << "Formatted I/O error: " << e.what() << endl;
    }
}

int main() {
    demonstrateFormattedIO();
    return 0;
}
```

## 🔄 Advanced File Operations

### 1. **Random Access and File Positioning**
```cpp
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
using namespace std;

struct Record {
    int id;
    char name[32];
    double value;
    
    Record() : id(0), value(0.0) {
        memset(name, 0, sizeof(name));
    }
    
    Record(int i, const string& n, double v) : id(i), value(v) {
        strncpy(name, n.c_str(), sizeof(name) - 1);
        name[sizeof(name) - 1] = '\0';
    }
    
    void display() const {
        cout << "ID: " << setw(3) << id 
             << ", Name: " << setw(15) << name 
             << ", Value: " << fixed << setprecision(2) << value << endl;
    }
};

class RandomAccessFile {
private:
    string filename;
    
public:
    RandomAccessFile(const string& fname) : filename(fname) {}
    
    void createFile(const vector<Record>& records) {
        ofstream file(filename, ios::binary);
        
        if (!file.is_open()) {
            throw runtime_error("Cannot create file: " + filename);
        }
        
        for (const auto& record : records) {
            file.write(reinterpret_cast<const char*>(&record), sizeof(Record));
        }
        
        cout << "Created file with " << records.size() << " records" << endl;
    }
    
    Record readRecord(size_t index) {
        ifstream file(filename, ios::binary);
        
        if (!file.is_open()) {
            throw runtime_error("Cannot open file for reading: " + filename);
        }
        
        // Calculate position
        streampos position = index * sizeof(Record);
        file.seekg(position);
        
        Record record;
        file.read(reinterpret_cast<char*>(&record), sizeof(Record));
        
        if (file.fail()) {
            throw runtime_error("Error reading record at index " + to_string(index));
        }
        
        return record;
    }
    
    void writeRecord(size_t index, const Record& record) {
        fstream file(filename, ios::binary | ios::in | ios::out);
        
        if (!file.is_open()) {
            throw runtime_error("Cannot open file for writing: " + filename);
        }
        
        // Calculate position
        streampos position = index * sizeof(Record);
        file.seekp(position);
        
        file.write(reinterpret_cast<const char*>(&record), sizeof(Record));
        
        if (file.fail()) {
            throw runtime_error("Error writing record at index " + to_string(index));
        }
        
        cout << "Updated record at index " << index << endl;
    }
    
    void insertRecord(size_t index, const Record& record) {
        // Read all records
        vector<Record> records = readAllRecords();
        
        // Insert new record
        if (index > records.size()) {
            index = records.size();  // Append at end
        }
        
        records.insert(records.begin() + index, record);
        
        // Write all records back
        createFile(records);
        
        cout << "Inserted record at index " << index << endl;
    }
    
    void deleteRecord(size_t index) {
        vector<Record> records = readAllRecords();
        
        if (index >= records.size()) {
            throw runtime_error("Index out of bounds: " + to_string(index));
        }
        
        records.erase(records.begin() + index);
        createFile(records);
        
        cout << "Deleted record at index " << index << endl;
    }
    
    vector<Record> readAllRecords() {
        ifstream file(filename, ios::binary);
        
        if (!file.is_open()) {
            throw runtime_error("Cannot open file for reading: " + filename);
        }
        
        vector<Record> records;
        Record record;
        
        while (file.read(reinterpret_cast<char*>(&record), sizeof(Record))) {
            records.push_back(record);
        }
        
        return records;
    }
    
    size_t getRecordCount() {
        ifstream file(filename, ios::binary | ios::ate);
        
        if (!file.is_open()) {
            return 0;
        }
        
        streampos fileSize = file.tellg();
        return fileSize / sizeof(Record);
    }
    
    void displayAllRecords() {
        vector<Record> records = readAllRecords();
        
        cout << "All records in file:" << endl;
        for (size_t i = 0; i < records.size(); i++) {
            cout << "[" << i << "] ";
            records[i].display();
        }
        cout << "Total records: " << records.size() << endl;
    }
};

void demonstrateRandomAccess() {
    cout << "=== Random Access File Operations ===" << endl;
    
    try {
        RandomAccessFile raf("records.bin");
        
        // Create initial file
        vector<Record> initialRecords = {
            Record(1, "Alice", 1000.0),
            Record(2, "Bob", 1500.0),
            Record(3, "Charlie", 1200.0),
            Record(4, "Diana", 1800.0),
            Record(5, "Eve", 1400.0)
        };
        
        raf.createFile(initialRecords);
        raf.displayAllRecords();
        
        // Read specific record
        cout << "\nReading record at index 2:" << endl;
        Record record = raf.readRecord(2);
        record.display();
        
        // Update a record
        cout << "\nUpdating record at index 1:" << endl;
        Record updatedRecord(2, "Robert", 2000.0);
        raf.writeRecord(1, updatedRecord);
        raf.displayAllRecords();
        
        // Insert a new record
        cout << "\nInserting record at index 2:" << endl;
        Record newRecord(6, "Frank", 1600.0);
        raf.insertRecord(2, newRecord);
        raf.displayAllRecords();
        
        // Delete a record
        cout << "\nDeleting record at index 0:" << endl;
        raf.deleteRecord(0);
        raf.displayAllRecords();
        
        cout << "\nTotal records in file: " << raf.getRecordCount() << endl;
        
    } catch (const exception& e) {
        cout << "Random access error: " << e.what() << endl;
    }
}

int main() {
    demonstrateRandomAccess();
    return 0;
}
```

### 2. **CSV File Processing**
```cpp
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <iomanip>
using namespace std;

struct Employee {
    int id;
    string firstName;
    string lastName;
    string department;
    double salary;
    string hireDate;
    
    Employee() = default;
    
    Employee(int i, const string& fn, const string& ln, const string& dept, 
             double sal, const string& date)
        : id(i), firstName(fn), lastName(ln), department(dept), salary(sal), hireDate(date) {}
    
    string getFullName() const {
        return firstName + " " + lastName;
    }
    
    void display() const {
        cout << left << setw(4) << id 
             << left << setw(20) << getFullName()
             << left << setw(15) << department
             << right << setw(10) << fixed << setprecision(2) << salary
             << "  " << hireDate << endl;
    }
};

class CSVProcessor {
public:
    static void writeCSV(const string& filename, const vector<Employee>& employees) {
        ofstream file(filename);
        
        if (!file.is_open()) {
            throw runtime_error("Cannot open CSV file for writing: " + filename);
        }
        
        // Write header
        file << "ID,FirstName,LastName,Department,Salary,HireDate" << endl;
        
        // Write employee data
        for (const auto& emp : employees) {
            file << emp.id << ","
                 << escapeCSVField(emp.firstName) << ","
                 << escapeCSVField(emp.lastName) << ","
                 << escapeCSVField(emp.department) << ","
                 << fixed << setprecision(2) << emp.salary << ","
                 << emp.hireDate << endl;
        }
        
        cout << "Wrote " << employees.size() << " employees to " << filename << endl;
    }
    
    static vector<Employee> readCSV(const string& filename) {
        ifstream file(filename);
        
        if (!file.is_open()) {
            throw runtime_error("Cannot open CSV file for reading: " + filename);
        }
        
        vector<Employee> employees;
        string line;
        
        // Skip header
        if (!getline(file, line)) {
            throw runtime_error("Empty CSV file or cannot read header");
        }
        
        int lineNumber = 2;  // Start from line 2 (after header)
        
        while (getline(file, line)) {
            try {
                Employee emp = parseCSVLine(line);
                employees.push_back(emp);
            } catch (const exception& e) {
                cout << "Warning: Error parsing line " << lineNumber << ": " << e.what() << endl;
            }
            lineNumber++;
        }
        
        cout << "Read " << employees.size() << " employees from " << filename << endl;
        return employees;
    }
    
    static void generateReport(const string& filename, const vector<Employee>& employees) {
        ofstream file(filename);
        
        if (!file.is_open()) {
            throw runtime_error("Cannot create report file: " + filename);
        }
        
        // Calculate statistics
        auto salaryStats = calculateSalaryStatistics(employees);
        auto deptStats = calculateDepartmentStatistics(employees);
        
        // Write report
        file << "EMPLOYEE REPORT" << endl;
        file << "===============" << endl << endl;
        
        file << "SUMMARY STATISTICS" << endl;
        file << "------------------" << endl;
        file << "Total Employees: " << employees.size() << endl;
        file << "Average Salary: $" << fixed << setprecision(2) << salaryStats.average << endl;
        file << "Minimum Salary: $" << fixed << setprecision(2) << salaryStats.minimum << endl;
        file << "Maximum Salary: $" << fixed << setprecision(2) << salaryStats.maximum << endl;
        file << "Total Payroll: $" << fixed << setprecision(2) << salaryStats.total << endl << endl;
        
        file << "DEPARTMENT BREAKDOWN" << endl;
        file << "--------------------" << endl;
        for (const auto& dept : deptStats) {
            file << dept.first << ": " << dept.second.count << " employees, "
                 << "Avg Salary: $" << fixed << setprecision(2) << dept.second.avgSalary << endl;
        }
        file << endl;
        
        file << "EMPLOYEE LISTING" << endl;
        file << "----------------" << endl;
        file << left << setw(4) << "ID" 
             << left << setw(20) << "Name"
             << left << setw(15) << "Department"
             << right << setw(10) << "Salary"
             << "  Hire Date" << endl;
        file << string(65, '-') << endl;
        
        for (const auto& emp : employees) {
            file << left << setw(4) << emp.id 
                 << left << setw(20) << emp.getFullName()
                 << left << setw(15) << emp.department
                 << right << setw(9) << fixed << setprecision(2) << "$" << emp.salary
                 << "  " << emp.hireDate << endl;
        }
        
        cout << "Generated report: " << filename << endl;
    }

private:
    static string escapeCSVField(const string& field) {
        if (field.find(',') != string::npos || field.find('"') != string::npos) {
            string escaped = "\"";
            for (char c : field) {
                if (c == '"') escaped += "\"\"";  // Escape quotes
                else escaped += c;
            }
            escaped += "\"";
            return escaped;
        }
        return field;
    }
    
    static Employee parseCSVLine(const string& line) {
        vector<string> fields = splitCSVLine(line);
        
        if (fields.size() != 6) {
            throw runtime_error("Expected 6 fields, got " + to_string(fields.size()));
        }
        
        Employee emp;
        emp.id = stoi(fields[0]);
        emp.firstName = fields[1];
        emp.lastName = fields[2];
        emp.department = fields[3];
        emp.salary = stod(fields[4]);
        emp.hireDate = fields[5];
        
        return emp;
    }
    
    static vector<string> splitCSVLine(const string& line) {
        vector<string> fields;
        stringstream ss(line);
        string field;
        bool inQuotes = false;
        
        for (size_t i = 0; i < line.length(); i++) {
            char c = line[i];
            
            if (c == '"') {
                inQuotes = !inQuotes;
            } else if (c == ',' && !inQuotes) {
                fields.push_back(field);
                field.clear();
            } else {
                field += c;
            }
        }
        
        fields.push_back(field);  // Add last field
        return fields;
    }
    
    struct SalaryStats {
        double average, minimum, maximum, total;
    };
    
    struct DepartmentStats {
        int count;
        double avgSalary;
    };
    
    static SalaryStats calculateSalaryStatistics(const vector<Employee>& employees) {
        if (employees.empty()) {
            return {0, 0, 0, 0};
        }
        
        double total = 0;
        double min_sal = employees[0].salary;
        double max_sal = employees[0].salary;
        
        for (const auto& emp : employees) {
            total += emp.salary;
            min_sal = min(min_sal, emp.salary);
            max_sal = max(max_sal, emp.salary);
        }
        
        return {total / employees.size(), min_sal, max_sal, total};
    }
    
    static map<string, DepartmentStats> calculateDepartmentStatistics(const vector<Employee>& employees) {
        map<string, vector<double>> deptSalaries;
        
        for (const auto& emp : employees) {
            deptSalaries[emp.department].push_back(emp.salary);
        }
        
        map<string, DepartmentStats> stats;
        for (const auto& dept : deptSalaries) {
            double total = 0;
            for (double salary : dept.second) {
                total += salary;
            }
            
            stats[dept.first] = {
                static_cast<int>(dept.second.size()),
                total / dept.second.size()
            };
        }
        
        return stats;
    }
};

void demonstrateCSVProcessing() {
    cout << "=== CSV File Processing ===" << endl;
    
    try {
        // Create sample employee data
        vector<Employee> employees = {
            Employee(101, "John", "Doe", "Engineering", 85000.0, "2020-01-15"),
            Employee(102, "Jane", "Smith", "Marketing", 72000.0, "2019-03-22"),
            Employee(103, "Bob", "Johnson", "Engineering", 90000.0, "2018-07-10"),
            Employee(104, "Alice", "Brown", "HR", 65000.0, "2021-02-28"),
            Employee(105, "Charlie", "Wilson", "Engineering", 95000.0, "2017-11-05"),
            Employee(106, "Diana", "Davis", "Marketing", 78000.0, "2020-09-14"),
            Employee(107, "Eve", "Miller", "Finance", 82000.0, "2019-06-30"),
            Employee(108, "Frank", "Garcia", "Engineering", 88000.0, "2021-01-12"),
            Employee(109, "Grace", "Lee", "HR", 70000.0, "2020-04-18"),
            Employee(110, "Henry", "Wang", "Finance", 85000.0, "2018-12-03")
        };
        
        // Write to CSV
        CSVProcessor::writeCSV("employees.csv", employees);
        
        // Read from CSV
        vector<Employee> readEmployees = CSVProcessor::readCSV("employees.csv");
        
        cout << "\nEmployees read from CSV:" << endl;
        cout << left << setw(4) << "ID" 
             << left << setw(20) << "Name"
             << left << setw(15) << "Department"
             << right << setw(10) << "Salary"
             << "  Hire Date" << endl;
        cout << string(65, '-') << endl;
        
        for (const auto& emp : readEmployees) {
            emp.display();
        }
        
        // Generate report
        CSVProcessor::generateReport("employee_report.txt", readEmployees);
        
        // Display report
        cout << "\nGenerated report content:" << endl;
        cout << string(50, '=') << endl;
        
        ifstream reportFile("employee_report.txt");
        string line;
        while (getline(reportFile, line)) {
            cout << line << endl;
        }
        
    } catch (const exception& e) {
        cout << "CSV processing error: " << e.what() << endl;
    }
}

int main() {
    demonstrateCSVProcessing();
    return 0;
}
```

## 🔧 File Utilities and Management

### 1. **File System Operations**
```cpp
#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <chrono>
#include <iomanip>
using namespace std;
namespace fs = std::filesystem;

class FileSystemManager {
public:
    static bool fileExists(const string& path) {
        return fs::exists(path);
    }
    
    static bool createDirectory(const string& path) {
        try {
            return fs::create_directories(path);
        } catch (const fs::filesystem_error& e) {
            cout << "Error creating directory: " << e.what() << endl;
            return false;
        }
    }
    
    static bool copyFile(const string& source, const string& destination) {
        try {
            return fs::copy_file(source, destination);
        } catch (const fs::filesystem_error& e) {
            cout << "Error copying file: " << e.what() << endl;
            return false;
        }
    }
    
    static bool moveFile(const string& source, const string& destination) {
        try {
            fs::rename(source, destination);
            return true;
        } catch (const fs::filesystem_error& e) {
            cout << "Error moving file: " << e.what() << endl;
            return false;
        }
    }
    
    static bool deleteFile(const string& path) {
        try {
            return fs::remove(path);
        } catch (const fs::filesystem_error& e) {
            cout << "Error deleting file: " << e.what() << endl;
            return false;
        }
    }
    
    static uintmax_t getFileSize(const string& path) {
        try {
            return fs::file_size(path);
        } catch (const fs::filesystem_error& e) {
            cout << "Error getting file size: " << e.what() << endl;
            return static_cast<uintmax_t>(-1);
        }
    }
    
    static void listDirectory(const string& path) {
        try {
            cout << "Contents of directory: " << path << endl;
            cout << string(50, '-') << endl;
            
            for (const auto& entry : fs::directory_iterator(path)) {
                cout << (entry.is_directory() ? "[DIR]  " : "[FILE] ");
                cout << left << setw(30) << entry.path().filename().string();
                
                if (entry.is_regular_file()) {
                    cout << right << setw(10) << entry.file_size() << " bytes";
                }
                cout << endl;
            }
        } catch (const fs::filesystem_error& e) {
            cout << "Error listing directory: " << e.what() << endl;
        }
    }
    
    static void findFiles(const string& directory, const string& pattern) {
        try {
            cout << "Files matching pattern '" << pattern << "' in " << directory << ":" << endl;
            
            for (const auto& entry : fs::recursive_directory_iterator(directory)) {
                if (entry.is_regular_file()) {
                    string filename = entry.path().filename().string();
                    if (filename.find(pattern) != string::npos) {
                        cout << entry.path().string() << " (" << entry.file_size() << " bytes)" << endl;
                    }
                }
            }
        } catch (const fs::filesystem_error& e) {
            cout << "Error searching files: " << e.what() << endl;
        }
    }
    
    static void createSampleFiles() {
        // Create directory structure
        createDirectory("sample_project");
        createDirectory("sample_project/src");
        createDirectory("sample_project/docs");
        createDirectory("sample_project/tests");
        
        // Create sample files
        vector<pair<string, string>> files = {
            {"sample_project/README.md", "# Sample Project\nThis is a sample project for file I/O demonstration."},
            {"sample_project/src/main.cpp", "#include <iostream>\nint main() { return 0; }"},
            {"sample_project/src/utils.h", "#pragma once\nvoid utility_function();"},
            {"sample_project/src/utils.cpp", "#include \"utils.h\"\nvoid utility_function() {}"},
            {"sample_project/docs/manual.txt", "User Manual\n============\nThis is the user manual."},
            {"sample_project/tests/test_main.cpp", "#include <cassert>\nint main() { return 0; }"}
        };
        
        for (const auto& file : files) {
            ofstream out(file.first);
            out << file.second << endl;
        }
        
        cout << "Created sample project structure" << endl;
    }
};

class FileBackupManager {
private:
    string backupDir;

public:
    FileBackupManager(const string& dir) : backupDir(dir) {
        FileSystemManager::createDirectory(backupDir);
    }
    
    void backupFile(const string& sourcePath) {
        if (!FileSystemManager::fileExists(sourcePath)) {
            cout << "Source file does not exist: " << sourcePath << endl;
            return;
        }
        
        fs::path source(sourcePath);
        string timestamp = getCurrentTimestamp();
        string backupName = source.stem().string() + "_" + timestamp + source.extension().string();
        string backupPath = backupDir + "/" + backupName;
        
        if (FileSystemManager::copyFile(sourcePath, backupPath)) {
            cout << "Backed up " << sourcePath << " to " << backupPath << endl;
        }
    }
    
    void restoreFile(const string& backupName, const string& restorePath) {
        string backupPath = backupDir + "/" + backupName;
        
        if (FileSystemManager::copyFile(backupPath, restorePath)) {
            cout << "Restored " << backupName << " to " << restorePath << endl;
        }
    }
    
    void listBackups() {
        cout << "Available backups:" << endl;
        FileSystemManager::listDirectory(backupDir);
    }

private:
    string getCurrentTimestamp() {
        auto now = chrono::system_clock::now();
        auto time_t = chrono::system_clock::to_time_t(now);
        auto tm = *localtime(&time_t);
        
        ostringstream oss;
        oss << put_time(&tm, "%Y%m%d_%H%M%S");
        return oss.str();
    }
};

void demonstrateFileSystemOperations() {
    cout << "=== File System Operations ===" << endl;
    
    // Create sample project structure
    FileSystemManager::createSampleFiles();
    
    // List contents
    cout << "\nProject structure:" << endl;
    FileSystemManager::listDirectory("sample_project");
    
    cout << "\nSource directory:" << endl;
    FileSystemManager::listDirectory("sample_project/src");
    
    // File operations
    cout << "\nFile operations:" << endl;
    
    // Check file existence
    cout << "README.md exists: " << (FileSystemManager::fileExists("sample_project/README.md") ? "Yes" : "No") << endl;
    
    // Get file size
    auto size = FileSystemManager::getFileSize("sample_project/README.md");
    if (size != static_cast<uintmax_t>(-1)) {
        cout << "README.md size: " << size << " bytes" << endl;
    }
    
    // Copy file
    if (FileSystemManager::copyFile("sample_project/README.md", "README_copy.md")) {
        cout << "Copied README.md to README_copy.md" << endl;
    }
    
    // Find files
    cout << "\nFinding .cpp files:" << endl;
    FileSystemManager::findFiles("sample_project", ".cpp");
    
    // Backup demonstration
    cout << "\n=== Backup Manager Demo ===" << endl;
    
    FileBackupManager backupMgr("backups");
    
    // Create a test file to backup
    {
        ofstream testFile("test_document.txt");
        testFile << "This is a test document.\nIt contains important data.\nVersion 1.0" << endl;
    }
    
    // Backup the file
    backupMgr.backupFile("test_document.txt");
    
    // Modify the original file
    {
        ofstream testFile("test_document.txt");
        testFile << "This is a test document.\nIt contains important data.\nVersion 2.0 - Modified!" << endl;
    }
    
    // Backup again
    backupMgr.backupFile("test_document.txt");
    
    // List backups
    backupMgr.listBackups();
    
    cout << "\nCleaning up..." << endl;
    
    // Clean up
    FileSystemManager::deleteFile("README_copy.md");
    FileSystemManager::deleteFile("test_document.txt");
    
    try {
        fs::remove_all("sample_project");
        fs::remove_all("backups");
        cout << "Cleaned up sample files and directories" << endl;
    } catch (const fs::filesystem_error& e) {
        cout << "Error during cleanup: " << e.what() << endl;
    }
}

int main() {
    demonstrateFileSystemOperations();
    return 0;
}
```

## 💡 Best Practices

### File I/O Best Practices
```cpp
#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
using namespace std;

// 1. RAII for file handling
class SafeFileReader {
private:
    unique_ptr<ifstream> file;
    string filename;

public:
    SafeFileReader(const string& fname) : filename(fname) {
        file = make_unique<ifstream>(filename);
        if (!file->is_open()) {
            throw runtime_error("Cannot open file: " + filename);
        }
    }
    
    string readLine() {
        string line;
        if (!getline(*file, line)) {
            throw runtime_error("Cannot read line from file: " + filename);
        }
        return line;
    }
    
    vector<string> readAllLines() {
        vector<string> lines;
        string line;
        
        file->seekg(0, ios::beg);  // Start from beginning
        
        while (getline(*file, line)) {
            lines.push_back(line);
        }
        
        return lines;
    }
    
    bool hasMoreData() {
        return file->good() && !file->eof();
    }
    
    // File automatically closed when object is destroyed
};

// 2. Exception-safe file operations
class SecureFileWriter {
private:
    string filename;
    string tempFilename;

public:
    SecureFileWriter(const string& fname) : filename(fname) {
        tempFilename = filename + ".tmp";
    }
    
    void writeData(const vector<string>& data) {
        // Write to temporary file first
        {
            ofstream tempFile(tempFilename);
            if (!tempFile.is_open()) {
                throw runtime_error("Cannot create temporary file: " + tempFilename);
            }
            
            for (const string& line : data) {
                tempFile << line << endl;
                if (tempFile.fail()) {
                    throw runtime_error("Error writing to temporary file");
                }
            }
        }  // Temporary file closed here
        
        // Atomically replace original file
        if (rename(tempFilename.c_str(), filename.c_str()) != 0) {
            remove(tempFilename.c_str());  // Clean up temp file
            throw runtime_error("Cannot replace original file");
        }
        
        cout << "Successfully wrote " << data.size() << " lines to " << filename << endl;
    }
    
    ~SecureFileWriter() {
        // Clean up temporary file if it exists
        remove(tempFilename.c_str());
    }
};

// 3. Buffered file operations for performance
class BufferedFileProcessor {
private:
    static const size_t BUFFER_SIZE = 8192;  // 8KB buffer

public:
    static void copyLargeFile(const string& source, const string& destination) {
        ifstream src(source, ios::binary);
        ofstream dst(destination, ios::binary);
        
        if (!src.is_open()) {
            throw runtime_error("Cannot open source file: " + source);
        }
        
        if (!dst.is_open()) {
            throw runtime_error("Cannot create destination file: " + destination);
        }
        
        // Use custom buffer for better performance
        vector<char> buffer(BUFFER_SIZE);
        
        while (src.read(buffer.data(), BUFFER_SIZE) || src.gcount() > 0) {
            dst.write(buffer.data(), src.gcount());
            
            if (dst.fail()) {
                throw runtime_error("Error writing to destination file");
            }
        }
        
        cout << "Successfully copied " << source << " to " << destination << endl;
    }
    
    static size_t countLines(const string& filename) {
        ifstream file(filename);
        
        if (!file.is_open()) {
            throw runtime_error("Cannot open file: " + filename);
        }
        
        // Use buffer for efficient line counting
        size_t lineCount = 0;
        string line;
        
        while (getline(file, line)) {
            lineCount++;
        }
        
        return lineCount;
    }
};

// 4. File validation and error checking
class FileValidator {
public:
    struct ValidationResult {
        bool isValid;
        vector<string> errors;
        
        ValidationResult(bool valid = true) : isValid(valid) {}
        
        void addError(const string& error) {
            errors.push_back(error);
            isValid = false;
        }
    };
    
    static ValidationResult validateTextFile(const string& filename) {
        ValidationResult result;
        
        // Check if file exists and is readable
        ifstream file(filename);
        if (!file.is_open()) {
            result.addError("File cannot be opened: " + filename);
            return result;
        }
        
        // Check file size
        file.seekg(0, ios::end);
        streampos fileSize = file.tellg();
        file.seekg(0, ios::beg);
        
        if (fileSize == 0) {
            result.addError("File is empty");
        } else if (fileSize > 100 * 1024 * 1024) {  // 100MB limit
            result.addError("File is too large (>100MB)");
        }
        
        // Validate content
        string line;
        int lineNumber = 0;
        
        while (getline(file, line)) {
            lineNumber++;
            
            // Check for very long lines
            if (line.length() > 1000) {
                result.addError("Line " + to_string(lineNumber) + " is too long");
            }
            
            // Check for non-printable characters (except common whitespace)
            for (char c : line) {
                if (c < 32 && c != '\t' && c != '\n' && c != '\r') {
                    result.addError("Line " + to_string(lineNumber) + " contains non-printable character");
                    break;
                }
            }
        }
        
        return result;
    }
};

// 5. Configuration file manager
class ConfigManager {
private:
    map<string, string> config;
    string configFile;

public:
    ConfigManager(const string& filename) : configFile(filename) {
        loadConfig();
    }
    
    void set(const string& key, const string& value) {
        config[key] = value;
    }
    
    string get(const string& key, const string& defaultValue = "") const {
        auto it = config.find(key);
        return (it != config.end()) ? it->second : defaultValue;
    }
    
    void save() {
        SecureFileWriter writer(configFile);
        
        vector<string> lines;
        lines.push_back("# Configuration file");
        lines.push_back("# Generated automatically");
        lines.push_back("");
        
        for (const auto& pair : config) {
            lines.push_back(pair.first + "=" + pair.second);
        }
        
        writer.writeData(lines);
    }

private:
    void loadConfig() {
        try {
            SafeFileReader reader(configFile);
            
            while (reader.hasMoreData()) {
                string line = reader.readLine();
                
                // Skip comments and empty lines
                if (line.empty() || line[0] == '#') continue;
                
                size_t eqPos = line.find('=');
                if (eqPos != string::npos) {
                    string key = line.substr(0, eqPos);
                    string value = line.substr(eqPos + 1);
                    config[key] = value;
                }
            }
            
            cout << "Loaded " << config.size() << " configuration entries" << endl;
            
        } catch (const runtime_error&) {
            cout << "Config file not found, starting with empty configuration" << endl;
        }
    }
};

void demonstrateBestPractices() {
    cout << "=== File I/O Best Practices ===" << endl;
    
    // Create test data
    vector<string> testData = {
        "Line 1: Important data",
        "Line 2: More important data",
        "Line 3: Critical information",
        "Line 4: Essential content",
        "Line 5: Final line"
    };
    
    try {
        // 1. Safe file writing
        cout << "1. Secure file writing:" << endl;
        SecureFileWriter writer("safe_output.txt");
        writer.writeData(testData);
        
        // 2. Safe file reading
        cout << "\n2. Safe file reading:" << endl;
        SafeFileReader reader("safe_output.txt");
        vector<string> readData = reader.readAllLines();
        
        cout << "Read " << readData.size() << " lines:" << endl;
        for (const string& line : readData) {
            cout << "  " << line << endl;
        }
        
        // 3. File validation
        cout << "\n3. File validation:" << endl;
        auto validation = FileValidator::validateTextFile("safe_output.txt");
        
        if (validation.isValid) {
            cout << "File validation passed" << endl;
        } else {
            cout << "File validation failed:" << endl;
            for (const string& error : validation.errors) {
                cout << "  Error: " << error << endl;
            }
        }
        
        // 4. Performance operations
        cout << "\n4. Performance operations:" << endl;
        size_t lineCount = BufferedFileProcessor::countLines("safe_output.txt");
        cout << "File contains " << lineCount << " lines" << endl;
        
        BufferedFileProcessor::copyLargeFile("safe_output.txt", "safe_output_copy.txt");
        
        // 5. Configuration management
        cout << "\n5. Configuration management:" << endl;
        ConfigManager config("app.conf");
        
        config.set("database.host", "localhost");
        config.set("database.port", "5432");
        config.set("app.debug", "true");
        config.save();
        
        cout << "Database host: " << config.get("database.host") << endl;
        cout << "Database port: " << config.get("database.port") << endl;
        cout << "Debug mode: " << config.get("app.debug") << endl;
        
        // Clean up
        remove("safe_output.txt");
        remove("safe_output_copy.txt");
        remove("app.conf");
        
    } catch (const exception& e) {
        cout << "Error: " << e.what() << endl;
    }
}

int main() {
    demonstrateBestPractices();
    return 0;
}
```

## 🔗 Related Topics
- [Arrays and Strings](./05-arrays-strings.md)
- [Exception Handling](./13-exception-handling.md)
- [STL](./15-stl.md)
- [Advanced Topics](./17-advanced-topics.md)

---
*Previous: [Exception Handling](./13-exception-handling.md) | Next: [Memory Management](./16-memory-management.md)*
