# 08. Object-Oriented Programming (OOP)

## 📋 Overview
Object-Oriented Programming (OOP) is a programming paradigm that organizes code around objects and classes. C++ supports all fundamental OOP concepts: Encapsulation, Inheritance, Polymorphism, and Abstraction.

## 🏗️ Classes and Objects

### 1. **Basic Class Definition**
```cpp
#include <iostream>
#include <string>
using namespace std;

class Student {
private:
    // Data members (attributes)
    string name;
    int age;
    double gpa;

public:
    // Constructor
    Student(string n, int a, double g) {
        name = n;
        age = a;
        gpa = g;
    }
    
    // Member functions (methods)
    void displayInfo() {
        cout << "Name: " << name << endl;
        cout << "Age: " << age << endl;
        cout << "GPA: " << gpa << endl;
    }
    
    // Getter methods
    string getName() const { return name; }
    int getAge() const { return age; }
    double getGPA() const { return gpa; }
    
    // Setter methods
    void setName(string n) { name = n; }
    void setAge(int a) { 
        if (a > 0) age = a; 
    }
    void setGPA(double g) { 
        if (g >= 0.0 && g <= 4.0) gpa = g; 
    }
};

int main() {
    // Create objects
    Student student1("Alice", 20, 3.8);
    Student student2("Bob", 19, 3.5);
    
    // Use objects
    student1.displayInfo();
    cout << "---" << endl;
    student2.displayInfo();
    
    // Modify through setters
    student1.setGPA(3.9);
    cout << "\nAfter GPA update:" << endl;
    student1.displayInfo();
    
    return 0;
}
```

### 2. **Access Specifiers**
```cpp
class AccessExample {
private:
    int privateVar;        // Only accessible within class

protected:
    int protectedVar;      // Accessible in class and derived classes

public:
    int publicVar;         // Accessible everywhere

public:
    AccessExample(int priv, int prot, int pub) 
        : privateVar(priv), protectedVar(prot), publicVar(pub) {}
    
    void displayPrivate() {
        cout << "Private: " << privateVar << endl;  // OK
    }
    
    // Friend function can access private members
    friend void friendFunction(const AccessExample& obj);
};

void friendFunction(const AccessExample& obj) {
    cout << "Friend accessing private: " << obj.privateVar << endl;  // OK
}

int main() {
    AccessExample obj(10, 20, 30);
    
    cout << "Public: " << obj.publicVar << endl;        // OK
    // cout << obj.privateVar;   // Error: private member
    // cout << obj.protectedVar; // Error: protected member
    
    obj.displayPrivate();      // OK: public method
    friendFunction(obj);       // OK: friend function
    
    return 0;
}
```

## 🏗️ Constructors and Destructors

### 1. **Types of Constructors**
```cpp
#include <iostream>
#include <string>
using namespace std;

class Person {
private:
    string name;
    int age;
    double* salary;  // Dynamic memory for demonstration

public:
    // 1. Default constructor
    Person() {
        name = "Unknown";
        age = 0;
        salary = new double(0.0);
        cout << "Default constructor called" << endl;
    }
    
    // 2. Parameterized constructor
    Person(string n, int a, double s) {
        name = n;
        age = a;
        salary = new double(s);
        cout << "Parameterized constructor called for " << name << endl;
    }
    
    // 3. Copy constructor
    Person(const Person& other) {
        name = other.name;
        age = other.age;
        salary = new double(*(other.salary));  // Deep copy
        cout << "Copy constructor called for " << name << endl;
    }
    
    // 4. Move constructor (C++11)
    Person(Person&& other) noexcept {
        name = move(other.name);
        age = other.age;
        salary = other.salary;
        other.salary = nullptr;  // Transfer ownership
        cout << "Move constructor called for " << name << endl;
    }
    
    // Destructor
    ~Person() {
        cout << "Destructor called for " << name << endl;
        delete salary;  // Clean up dynamic memory
    }
    
    // Assignment operator
    Person& operator=(const Person& other) {
        if (this != &other) {  // Self-assignment check
            name = other.name;
            age = other.age;
            delete salary;  // Clean up existing memory
            salary = new double(*(other.salary));  // Deep copy
        }
        cout << "Assignment operator called for " << name << endl;
        return *this;
    }
    
    void display() const {
        cout << "Name: " << name << ", Age: " << age 
             << ", Salary: " << *salary << endl;
    }
};

int main() {
    Person p1;                           // Default constructor
    Person p2("Alice", 25, 50000.0);     // Parameterized constructor
    Person p3 = p2;                      // Copy constructor
    Person p4("Bob", 30, 60000.0);       // Parameterized constructor
    
    p1 = p4;                             // Assignment operator
    
    p1.display();
    p2.display();
    p3.display();
    p4.display();
    
    return 0;
}  // Destructors called in reverse order
```

### 2. **Constructor Initialization Lists**
```cpp
class Rectangle {
private:
    const int length;  // const member must be initialized
    const int width;   // const member must be initialized
    int& areaRef;      // reference member must be initialized
    int area;

public:
    // Constructor with initialization list
    Rectangle(int l, int w, int& ref) 
        : length(l), width(w), areaRef(ref), area(l * w) {
        // Constructor body (after initialization)
        cout << "Rectangle created: " << length << "x" << width << endl;
    }
    
    // Delegating constructor (C++11)
    Rectangle(int side, int& ref) : Rectangle(side, side, ref) {
        cout << "Square created using delegating constructor" << endl;
    }
    
    void display() const {
        cout << "Dimensions: " << length << "x" << width 
             << ", Area: " << area << endl;
    }
};

int main() {
    int tempVar = 0;
    Rectangle rect(10, 5, tempVar);
    Rectangle square(7, tempVar);
    
    rect.display();
    square.display();
    
    return 0;
}
```

## 🎭 Encapsulation

### Data Hiding and Access Control
```cpp
#include <iostream>
#include <string>
using namespace std;

class BankAccount {
private:
    string accountNumber;
    double balance;
    string pin;
    
    // Private helper method
    bool validatePin(const string& inputPin) const {
        return pin == inputPin;
    }

public:
    BankAccount(string accNum, string accountPin, double initialBalance = 0.0) 
        : accountNumber(accNum), pin(accountPin), balance(initialBalance) {
        if (initialBalance < 0) {
            balance = 0;
        }
    }
    
    // Public interface methods
    bool deposit(double amount, const string& inputPin) {
        if (!validatePin(inputPin)) {
            cout << "Invalid PIN!" << endl;
            return false;
        }
        
        if (amount > 0) {
            balance += amount;
            cout << "Deposited $" << amount << endl;
            return true;
        }
        
        cout << "Invalid deposit amount!" << endl;
        return false;
    }
    
    bool withdraw(double amount, const string& inputPin) {
        if (!validatePin(inputPin)) {
            cout << "Invalid PIN!" << endl;
            return false;
        }
        
        if (amount > 0 && amount <= balance) {
            balance -= amount;
            cout << "Withdrew $" << amount << endl;
            return true;
        }
        
        cout << "Invalid withdrawal amount or insufficient funds!" << endl;
        return false;
    }
    
    double getBalance(const string& inputPin) const {
        if (validatePin(inputPin)) {
            return balance;
        }
        
        cout << "Invalid PIN!" << endl;
        return -1;  // Error indicator
    }
    
    string getAccountNumber() const {
        return accountNumber;  // Account number can be public
    }
};

int main() {
    BankAccount account("123456789", "1234", 1000.0);
    
    cout << "Account: " << account.getAccountNumber() << endl;
    cout << "Balance: $" << account.getBalance("1234") << endl;
    
    account.deposit(500, "1234");
    account.withdraw(200, "1234");
    account.withdraw(2000, "1234");  // Should fail
    
    cout << "Final balance: $" << account.getBalance("1234") << endl;
    
    return 0;
}
```

## 🔧 Static Members

### Static Data Members and Methods
```cpp
#include <iostream>
#include <string>
using namespace std;

class Counter {
private:
    static int objectCount;  // Shared by all objects
    int instanceId;

public:
    Counter() {
        instanceId = ++objectCount;
        cout << "Counter object " << instanceId << " created" << endl;
    }
    
    ~Counter() {
        cout << "Counter object " << instanceId << " destroyed" << endl;
        --objectCount;
    }
    
    // Static method - can only access static members
    static int getObjectCount() {
        return objectCount;
        // return instanceId;  // Error: cannot access non-static member
    }
    
    int getInstanceId() const {
        return instanceId;
    }
    
    // Static method with no object required
    static void printInfo() {
        cout << "Total Counter objects: " << objectCount << endl;
    }
};

// Definition of static member (required)
int Counter::objectCount = 0;

class MathUtils {
public:
    static const double PI;
    
    static double circleArea(double radius) {
        return PI * radius * radius;
    }
    
    static double circleCircumference(double radius) {
        return 2 * PI * radius;
    }
    
    static int factorial(int n) {
        if (n <= 1) return 1;
        return n * factorial(n - 1);
    }
};

// Definition of static const member
const double MathUtils::PI = 3.14159265359;

int main() {
    cout << "Initial count: " << Counter::getObjectCount() << endl;
    
    Counter c1;
    Counter c2;
    
    {
        Counter c3;
        Counter::printInfo();
    }  // c3 destroyed here
    
    Counter::printInfo();
    
    // Using static utility methods
    cout << "\nMath utilities:" << endl;
    cout << "PI = " << MathUtils::PI << endl;
    cout << "Circle area (r=5): " << MathUtils::circleArea(5) << endl;
    cout << "Factorial of 5: " << MathUtils::factorial(5) << endl;
    
    return 0;
}  // c1 and c2 destroyed here
```

## 👥 Friend Functions and Classes

```cpp
#include <iostream>
using namespace std;

class Box {
private:
    double length, width, height;

public:
    Box(double l, double w, double h) : length(l), width(w), height(h) {}
    
    // Friend function declaration
    friend double calculateVolume(const Box& box);
    friend class BoxInspector;
    
    // Friend function for operator overloading
    friend ostream& operator<<(ostream& os, const Box& box);
};

// Friend function definition
double calculateVolume(const Box& box) {
    // Can access private members
    return box.length * box.width * box.height;
}

// Friend class
class BoxInspector {
public:
    static void inspectBox(const Box& box) {
        cout << "Box inspection:" << endl;
        cout << "Length: " << box.length << endl;  // Access private member
        cout << "Width: " << box.width << endl;
        cout << "Height: " << box.height << endl;
    }
    
    static bool compareBoxes(const Box& box1, const Box& box2) {
        return (box1.length * box1.width * box1.height) > 
               (box2.length * box2.width * box2.height);
    }
};

// Friend operator overloading
ostream& operator<<(ostream& os, const Box& box) {
    os << "Box(" << box.length << ", " << box.width << ", " << box.height << ")";
    return os;
}

int main() {
    Box box1(10, 5, 3);
    Box box2(8, 6, 4);
    
    cout << "Box 1: " << box1 << endl;
    cout << "Box 2: " << box2 << endl;
    
    cout << "Volume of box1: " << calculateVolume(box1) << endl;
    
    BoxInspector::inspectBox(box1);
    
    if (BoxInspector::compareBoxes(box1, box2)) {
        cout << "Box 1 is larger than Box 2" << endl;
    } else {
        cout << "Box 2 is larger than or equal to Box 1" << endl;
    }
    
    return 0;
}
```

## 🎯 Practical Examples

### 1. **Student Management System**
```cpp
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
using namespace std;

class Student {
private:
    static int nextId;
    int studentId;
    string name;
    vector<double> grades;

public:
    Student(const string& studentName) 
        : studentId(++nextId), name(studentName) {}
    
    void addGrade(double grade) {
        if (grade >= 0.0 && grade <= 100.0) {
            grades.push_back(grade);
        }
    }
    
    double calculateGPA() const {
        if (grades.empty()) return 0.0;
        
        double sum = 0.0;
        for (double grade : grades) {
            sum += grade;
        }
        return sum / grades.size();
    }
    
    void displayInfo() const {
        cout << "ID: " << studentId << ", Name: " << name;
        cout << ", GPA: " << calculateGPA() << endl;
    }
    
    // Getters
    int getId() const { return studentId; }
    const string& getName() const { return name; }
    
    // For sorting
    bool operator<(const Student& other) const {
        return calculateGPA() > other.calculateGPA();  // Sort by GPA descending
    }
};

int Student::nextId = 0;

class StudentManager {
private:
    vector<Student> students;

public:
    void addStudent(const string& name) {
        students.emplace_back(name);
        cout << "Added student: " << name << endl;
    }
    
    Student* findStudent(int id) {
        for (auto& student : students) {
            if (student.getId() == id) {
                return &student;
            }
        }
        return nullptr;
    }
    
    void displayAllStudents() const {
        cout << "\n=== All Students ===" << endl;
        for (const auto& student : students) {
            student.displayInfo();
        }
    }
    
    void displayTopStudents(int count = 3) {
        vector<Student> sortedStudents = students;
        sort(sortedStudents.begin(), sortedStudents.end());
        
        cout << "\n=== Top " << count << " Students ===" << endl;
        for (int i = 0; i < min(count, (int)sortedStudents.size()); i++) {
            sortedStudents[i].displayInfo();
        }
    }
};

int main() {
    StudentManager manager;
    
    // Add students
    manager.addStudent("Alice Johnson");
    manager.addStudent("Bob Smith");
    manager.addStudent("Charlie Brown");
    
    // Add grades
    Student* alice = manager.findStudent(1);
    if (alice) {
        alice->addGrade(95.0);
        alice->addGrade(87.5);
        alice->addGrade(92.0);
    }
    
    Student* bob = manager.findStudent(2);
    if (bob) {
        bob->addGrade(78.0);
        bob->addGrade(85.5);
        bob->addGrade(81.0);
    }
    
    Student* charlie = manager.findStudent(3);
    if (charlie) {
        charlie->addGrade(98.0);
        charlie->addGrade(94.5);
        charlie->addGrade(96.0);
    }
    
    manager.displayAllStudents();
    manager.displayTopStudents();
    
    return 0;
}
```

### 2. **Library Book System**
```cpp
#include <iostream>
#include <string>
#include <vector>
#include <ctime>
using namespace std;

class Book {
private:
    static int nextBookId;
    int bookId;
    string title;
    string author;
    string isbn;
    bool isAvailable;
    time_t borrowDate;

public:
    Book(const string& bookTitle, const string& bookAuthor, const string& bookIsbn)
        : bookId(++nextBookId), title(bookTitle), author(bookAuthor), 
          isbn(bookIsbn), isAvailable(true), borrowDate(0) {}
    
    bool borrowBook() {
        if (isAvailable) {
            isAvailable = false;
            borrowDate = time(nullptr);
            return true;
        }
        return false;
    }
    
    bool returnBook() {
        if (!isAvailable) {
            isAvailable = true;
            borrowDate = 0;
            return true;
        }
        return false;
    }
    
    void displayInfo() const {
        cout << "ID: " << bookId << ", Title: " << title 
             << ", Author: " << author << ", ISBN: " << isbn;
        cout << ", Status: " << (isAvailable ? "Available" : "Borrowed");
        
        if (!isAvailable && borrowDate > 0) {
            cout << ", Borrowed: " << ctime(&borrowDate);
        } else {
            cout << endl;
        }
    }
    
    // Getters
    int getId() const { return bookId; }
    const string& getTitle() const { return title; }
    const string& getAuthor() const { return author; }
    bool getAvailability() const { return isAvailable; }
};

int Book::nextBookId = 0;

class Library {
private:
    vector<Book> books;
    string libraryName;

public:
    Library(const string& name) : libraryName(name) {}
    
    void addBook(const string& title, const string& author, const string& isbn) {
        books.emplace_back(title, author, isbn);
        cout << "Added book: " << title << endl;
    }
    
    Book* findBookById(int id) {
        for (auto& book : books) {
            if (book.getId() == id) {
                return &book;
            }
        }
        return nullptr;
    }
    
    vector<Book*> searchByTitle(const string& title) {
        vector<Book*> results;
        for (auto& book : books) {
            if (book.getTitle().find(title) != string::npos) {
                results.push_back(&book);
            }
        }
        return results;
    }
    
    void displayAllBooks() const {
        cout << "\n=== " << libraryName << " Catalog ===" << endl;
        for (const auto& book : books) {
            book.displayInfo();
        }
    }
    
    void displayAvailableBooks() const {
        cout << "\n=== Available Books ===" << endl;
        for (const auto& book : books) {
            if (book.getAvailability()) {
                book.displayInfo();
            }
        }
    }
};

int main() {
    Library library("City Central Library");
    
    // Add books
    library.addBook("The Great Gatsby", "F. Scott Fitzgerald", "978-0-7432-7356-5");
    library.addBook("To Kill a Mockingbird", "Harper Lee", "978-0-06-112008-4");
    library.addBook("1984", "George Orwell", "978-0-452-28423-4");
    library.addBook("The Catcher in the Rye", "J.D. Salinger", "978-0-316-76948-0");
    
    library.displayAllBooks();
    
    // Borrow a book
    Book* book = library.findBookById(2);
    if (book && book->borrowBook()) {
        cout << "\nSuccessfully borrowed: " << book->getTitle() << endl;
    }
    
    library.displayAvailableBooks();
    
    // Return the book
    if (book && book->returnBook()) {
        cout << "\nSuccessfully returned: " << book->getTitle() << endl;
    }
    
    library.displayAvailableBooks();
    
    return 0;
}
```

## 💡 Best Practices

### 1. **Class Design Principles**
```cpp
// 1. Keep classes focused (Single Responsibility Principle)
class Calculator {
public:
    double add(double a, double b) { return a + b; }
    double subtract(double a, double b) { return a - b; }
    // Don't add unrelated methods like displayMenu()
};

// 2. Use const correctness
class Point {
private:
    double x, y;
public:
    Point(double x, double y) : x(x), y(y) {}
    
    double getX() const { return x; }  // Const method
    double getY() const { return y; }
    
    void setX(double newX) { x = newX; }  // Non-const method
    void setY(double newY) { y = newY; }
    
    double distanceFrom(const Point& other) const {  // Const parameter and method
        return sqrt((x - other.x) * (x - other.x) + (y - other.y) * (y - other.y));
    }
};

// 3. Initialize members in initialization list
class Person {
private:
    const string name;  // const member
    int age;
public:
    Person(const string& n, int a) : name(n), age(a) {}  // Initialization list
};
```

### 2. **Resource Management**
```cpp
// RAII (Resource Acquisition Is Initialization)
class FileManager {
private:
    FILE* file;
public:
    FileManager(const char* filename, const char* mode) {
        file = fopen(filename, mode);
        if (!file) {
            throw runtime_error("Failed to open file");
        }
    }
    
    ~FileManager() {
        if (file) {
            fclose(file);
        }
    }
    
    // Disable copy constructor and assignment operator
    FileManager(const FileManager&) = delete;
    FileManager& operator=(const FileManager&) = delete;
    
    // Enable move constructor and assignment (C++11)
    FileManager(FileManager&& other) noexcept : file(other.file) {
        other.file = nullptr;
    }
    
    FileManager& operator=(FileManager&& other) noexcept {
        if (this != &other) {
            if (file) fclose(file);
            file = other.file;
            other.file = nullptr;
        }
        return *this;
    }
    
    FILE* get() const { return file; }
};
```

## 🔗 Related Topics
- [Inheritance](./09-inheritance.md)
- [Polymorphism](./10-polymorphism.md)
- [Operator Overloading](./11-operator-overloading.md)
- [Templates](./12-templates.md)

---
*Previous: [Pointers & References](./07-pointers-references.md) | Next: [Inheritance](./09-inheritance.md)*
