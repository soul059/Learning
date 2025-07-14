# 09. Inheritance

## 📋 Overview
Inheritance is a fundamental principle of Object-Oriented Programming that allows a class to inherit properties and methods from another class. It promotes code reusability and establishes an "is-a" relationship between classes.

## 🏗️ Basic Inheritance

### 1. **Simple Inheritance**
```cpp
#include <iostream>
#include <string>
using namespace std;

// Base class (Parent class)
class Animal {
protected:
    string name;
    int age;

public:
    Animal(string n, int a) : name(n), age(a) {
        cout << "Animal constructor called for " << name << endl;
    }
    
    ~Animal() {
        cout << "Animal destructor called for " << name << endl;
    }
    
    void eat() {
        cout << name << " is eating." << endl;
    }
    
    void sleep() {
        cout << name << " is sleeping." << endl;
    }
    
    void displayInfo() {
        cout << "Name: " << name << ", Age: " << age << endl;
    }
    
    // Getter methods
    string getName() const { return name; }
    int getAge() const { return age; }
};

// Derived class (Child class)
class Dog : public Animal {
private:
    string breed;

public:
    Dog(string n, int a, string b) : Animal(n, a), breed(b) {
        cout << "Dog constructor called for " << name << endl;
    }
    
    ~Dog() {
        cout << "Dog destructor called for " << name << endl;
    }
    
    void bark() {
        cout << name << " is barking: Woof! Woof!" << endl;
    }
    
    void fetch() {
        cout << name << " is fetching the ball." << endl;
    }
    
    void displayInfo() {  // Method overriding
        Animal::displayInfo();  // Call base class method
        cout << "Breed: " << breed << endl;
    }
    
    string getBreed() const { return breed; }
};

int main() {
    Dog myDog("Buddy", 3, "Golden Retriever");
    
    // Using inherited methods
    myDog.eat();
    myDog.sleep();
    
    // Using derived class methods
    myDog.bark();
    myDog.fetch();
    
    // Method overriding
    myDog.displayInfo();
    
    return 0;
}
```

### 2. **Access Specifiers in Inheritance**
```cpp
#include <iostream>
using namespace std;

class Base {
private:
    int privateVar;

protected:
    int protectedVar;

public:
    int publicVar;
    
    Base() : privateVar(1), protectedVar(2), publicVar(3) {}
    
    void displayBase() {
        cout << "Private: " << privateVar << endl;      // Accessible
        cout << "Protected: " << protectedVar << endl;  // Accessible
        cout << "Public: " << publicVar << endl;        // Accessible
    }
};

// Public Inheritance
class PublicDerived : public Base {
public:
    void displayDerived() {
        // cout << privateVar;    // Error: not accessible
        cout << "Protected: " << protectedVar << endl;  // Accessible
        cout << "Public: " << publicVar << endl;        // Accessible
    }
};

// Protected Inheritance
class ProtectedDerived : protected Base {
public:
    void displayDerived() {
        // cout << privateVar;    // Error: not accessible
        cout << "Protected: " << protectedVar << endl;  // Accessible
        cout << "Public: " << publicVar << endl;        // Accessible (but becomes protected)
    }
};

// Private Inheritance
class PrivateDerived : private Base {
public:
    void displayDerived() {
        // cout << privateVar;    // Error: not accessible
        cout << "Protected: " << protectedVar << endl;  // Accessible (but becomes private)
        cout << "Public: " << publicVar << endl;        // Accessible (but becomes private)
    }
};

int main() {
    PublicDerived pubObj;
    ProtectedDerived protObj;
    PrivateDerived privObj;
    
    // Public inheritance - base class public members remain public
    pubObj.publicVar = 10;      // OK
    pubObj.displayBase();       // OK
    
    // Protected inheritance - base class public members become protected
    // protObj.publicVar = 20;  // Error: protected member
    // protObj.displayBase();   // Error: protected member
    
    // Private inheritance - base class public members become private
    // privObj.publicVar = 30;  // Error: private member
    // privObj.displayBase();   // Error: private member
    
    return 0;
}
```

## 🔄 Types of Inheritance

### 1. **Single Inheritance**
```cpp
#include <iostream>
#include <string>
using namespace std;

class Vehicle {
protected:
    string brand;
    int year;

public:
    Vehicle(string b, int y) : brand(b), year(y) {}
    
    void start() {
        cout << brand << " vehicle is starting..." << endl;
    }
    
    void stop() {
        cout << brand << " vehicle is stopping..." << endl;
    }
};

class Car : public Vehicle {
private:
    int doors;

public:
    Car(string b, int y, int d) : Vehicle(b, y), doors(d) {}
    
    void drive() {
        cout << brand << " car is driving with " << doors << " doors." << endl;
    }
    
    void park() {
        cout << brand << " car is parking." << endl;
    }
};

int main() {
    Car myCar("Toyota", 2023, 4);
    
    myCar.start();  // Inherited method
    myCar.drive();  // Own method
    myCar.park();   // Own method
    myCar.stop();   // Inherited method
    
    return 0;
}
```

### 2. **Multiple Inheritance**
```cpp
#include <iostream>
#include <string>
using namespace std;

class Flyable {
public:
    void fly() {
        cout << "Flying in the sky!" << endl;
    }
    
    virtual void takeOff() {
        cout << "Taking off..." << endl;
    }
};

class Swimmable {
public:
    void swim() {
        cout << "Swimming in water!" << endl;
    }
    
    virtual void dive() {
        cout << "Diving underwater..." << endl;
    }
};

class Duck : public Flyable, public Swimmable {
private:
    string name;

public:
    Duck(string n) : name(n) {}
    
    void quack() {
        cout << name << " says: Quack! Quack!" << endl;
    }
    
    // Override methods from both base classes
    void takeOff() override {
        cout << name << " is taking off from water!" << endl;
    }
    
    void dive() override {
        cout << name << " is diving for food!" << endl;
    }
};

int main() {
    Duck duck("Donald");
    
    duck.quack();   // Own method
    duck.fly();     // From Flyable
    duck.swim();    // From Swimmable
    duck.takeOff(); // Overridden method
    duck.dive();    // Overridden method
    
    return 0;
}
```

### 3. **Multilevel Inheritance**
```cpp
#include <iostream>
#include <string>
using namespace std;

// Base class
class LivingBeing {
protected:
    bool isAlive;

public:
    LivingBeing() : isAlive(true) {
        cout << "LivingBeing constructor" << endl;
    }
    
    void breathe() {
        cout << "Breathing..." << endl;
    }
    
    void reproduce() {
        cout << "Reproducing..." << endl;
    }
};

// Intermediate class
class Animal : public LivingBeing {
protected:
    string species;

public:
    Animal(string s) : species(s) {
        cout << "Animal constructor for " << species << endl;
    }
    
    void move() {
        cout << species << " is moving." << endl;
    }
    
    void eat() {
        cout << species << " is eating." << endl;
    }
};

// Derived class
class Mammal : public Animal {
protected:
    bool hasFur;

public:
    Mammal(string s, bool fur) : Animal(s), hasFur(fur) {
        cout << "Mammal constructor" << endl;
    }
    
    void feedMilk() {
        cout << species << " is feeding milk to offspring." << endl;
    }
    
    void regulateTemperature() {
        cout << species << " is regulating body temperature." << endl;
    }
};

// Final derived class
class Dog : public Mammal {
private:
    string name;

public:
    Dog(string n) : Mammal("Canine", true), name(n) {
        cout << "Dog constructor for " << name << endl;
    }
    
    void bark() {
        cout << name << " is barking!" << endl;
    }
    
    void wagTail() {
        cout << name << " is wagging tail!" << endl;
    }
};

int main() {
    Dog myDog("Rex");
    
    // Methods from all levels of inheritance
    myDog.breathe();              // From LivingBeing
    myDog.move();                 // From Animal
    myDog.feedMilk();             // From Mammal
    myDog.bark();                 // From Dog
    
    return 0;
}
```

### 4. **Hierarchical Inheritance**
```cpp
#include <iostream>
#include <string>
using namespace std;

class Shape {
protected:
    string color;

public:
    Shape(string c) : color(c) {}
    
    void setColor(string c) { color = c; }
    string getColor() const { return color; }
    
    virtual void display() {
        cout << "Shape with color: " << color << endl;
    }
    
    virtual double area() = 0;  // Pure virtual function
};

class Circle : public Shape {
private:
    double radius;

public:
    Circle(string c, double r) : Shape(c), radius(r) {}
    
    double area() override {
        return 3.14159 * radius * radius;
    }
    
    void display() override {
        cout << "Circle - Color: " << color << ", Radius: " << radius << endl;
    }
};

class Rectangle : public Shape {
private:
    double length, width;

public:
    Rectangle(string c, double l, double w) : Shape(c), length(l), width(w) {}
    
    double area() override {
        return length * width;
    }
    
    void display() override {
        cout << "Rectangle - Color: " << color << ", Length: " << length 
             << ", Width: " << width << endl;
    }
};

class Triangle : public Shape {
private:
    double base, height;

public:
    Triangle(string c, double b, double h) : Shape(c), base(b), height(h) {}
    
    double area() override {
        return 0.5 * base * height;
    }
    
    void display() override {
        cout << "Triangle - Color: " << color << ", Base: " << base 
             << ", Height: " << height << endl;
    }
};

int main() {
    Circle circle("Red", 5.0);
    Rectangle rect("Blue", 10.0, 6.0);
    Triangle tri("Green", 8.0, 4.0);
    
    // Array of base class pointers
    Shape* shapes[] = {&circle, &rect, &tri};
    
    cout << "Shape Information:" << endl;
    for (int i = 0; i < 3; i++) {
        shapes[i]->display();
        cout << "Area: " << shapes[i]->area() << endl;
        cout << "---" << endl;
    }
    
    return 0;
}
```

## 💎 Diamond Problem and Virtual Inheritance

### The Diamond Problem
```cpp
#include <iostream>
using namespace std;

class Base {
public:
    int value;
    Base(int v) : value(v) {
        cout << "Base constructor: " << value << endl;
    }
    
    void display() {
        cout << "Base value: " << value << endl;
    }
};

class Left : public Base {
public:
    Left(int v) : Base(v) {
        cout << "Left constructor" << endl;
    }
};

class Right : public Base {
public:
    Right(int v) : Base(v) {
        cout << "Right constructor" << endl;
    }
};

// This creates ambiguity - which Base::value?
class Bottom : public Left, public Right {
public:
    Bottom(int l, int r) : Left(l), Right(r) {
        cout << "Bottom constructor" << endl;
    }
    
    void showValues() {
        cout << "Left base value: " << Left::value << endl;
        cout << "Right base value: " << Right::value << endl;
        // cout << value;  // Error: ambiguous
    }
};

int main() {
    Bottom obj(10, 20);
    obj.showValues();
    
    // obj.display();  // Error: ambiguous
    obj.Left::display();   // Specify which one
    obj.Right::display();  // Specify which one
    
    return 0;
}
```

### Virtual Inheritance Solution
```cpp
#include <iostream>
using namespace std;

class Base {
public:
    int value;
    Base(int v) : value(v) {
        cout << "Base constructor: " << value << endl;
    }
    
    void display() {
        cout << "Base value: " << value << endl;
    }
};

// Virtual inheritance
class Left : public virtual Base {
public:
    Left(int v) : Base(v) {
        cout << "Left constructor" << endl;
    }
};

class Right : public virtual Base {
public:
    Right(int v) : Base(v) {
        cout << "Right constructor" << endl;
    }
};

class Bottom : public Left, public Right {
public:
    // Must directly initialize virtual base
    Bottom(int v) : Base(v), Left(v), Right(v) {
        cout << "Bottom constructor" << endl;
    }
    
    void showValue() {
        cout << "Shared base value: " << value << endl;  // No ambiguity
    }
};

int main() {
    Bottom obj(42);
    obj.showValue();
    obj.display();  // No ambiguity
    
    return 0;
}
```

## 🔧 Constructor and Destructor Order

### Order of Execution
```cpp
#include <iostream>
using namespace std;

class Base1 {
public:
    Base1() { cout << "Base1 constructor" << endl; }
    ~Base1() { cout << "Base1 destructor" << endl; }
};

class Base2 {
public:
    Base2() { cout << "Base2 constructor" << endl; }
    ~Base2() { cout << "Base2 destructor" << endl; }
};

class Derived : public Base1, public Base2 {
public:
    Derived() { cout << "Derived constructor" << endl; }
    ~Derived() { cout << "Derived destructor" << endl; }
};

class SubDerived : public Derived {
public:
    SubDerived() { cout << "SubDerived constructor" << endl; }
    ~SubDerived() { cout << "SubDerived destructor" << endl; }
};

int main() {
    cout << "Creating SubDerived object:" << endl;
    {
        SubDerived obj;
    }
    cout << "Object destroyed" << endl;
    
    /* Output:
    Creating SubDerived object:
    Base1 constructor
    Base2 constructor
    Derived constructor
    SubDerived constructor
    SubDerived destructor
    Derived destructor
    Base2 destructor
    Base1 destructor
    Object destroyed
    */
    
    return 0;
}
```

## 🎯 Practical Examples

### 1. **Employee Management System**
```cpp
#include <iostream>
#include <string>
#include <vector>
using namespace std;

class Employee {
protected:
    static int nextId;
    int employeeId;
    string name;
    string department;
    double baseSalary;

public:
    Employee(string n, string dept, double salary) 
        : employeeId(++nextId), name(n), department(dept), baseSalary(salary) {}
    
    virtual ~Employee() = default;
    
    virtual double calculateSalary() const {
        return baseSalary;
    }
    
    virtual void displayInfo() const {
        cout << "ID: " << employeeId << ", Name: " << name 
             << ", Department: " << department 
             << ", Base Salary: $" << baseSalary << endl;
    }
    
    // Getters
    int getId() const { return employeeId; }
    string getName() const { return name; }
    string getDepartment() const { return department; }
    double getBaseSalary() const { return baseSalary; }
};

int Employee::nextId = 0;

class Manager : public Employee {
private:
    double bonus;
    int teamSize;

public:
    Manager(string n, string dept, double salary, double b, int team) 
        : Employee(n, dept, salary), bonus(b), teamSize(team) {}
    
    double calculateSalary() const override {
        return baseSalary + bonus + (teamSize * 100);  // $100 per team member
    }
    
    void displayInfo() const override {
        Employee::displayInfo();
        cout << "Role: Manager, Bonus: $" << bonus 
             << ", Team Size: " << teamSize 
             << ", Total Salary: $" << calculateSalary() << endl;
    }
    
    void conductMeeting() {
        cout << name << " is conducting a team meeting." << endl;
    }
};

class Developer : public Employee {
private:
    string programmingLanguage;
    int projectsCompleted;

public:
    Developer(string n, string dept, double salary, string lang, int projects) 
        : Employee(n, dept, salary), programmingLanguage(lang), projectsCompleted(projects) {}
    
    double calculateSalary() const override {
        return baseSalary + (projectsCompleted * 500);  // $500 per project
    }
    
    void displayInfo() const override {
        Employee::displayInfo();
        cout << "Role: Developer, Language: " << programmingLanguage 
             << ", Projects: " << projectsCompleted 
             << ", Total Salary: $" << calculateSalary() << endl;
    }
    
    void writeCode() {
        cout << name << " is writing " << programmingLanguage << " code." << endl;
    }
};

class Intern : public Employee {
private:
    string university;
    int duration;  // months

public:
    Intern(string n, string dept, double salary, string uni, int dur) 
        : Employee(n, dept, salary), university(uni), duration(dur) {}
    
    double calculateSalary() const override {
        return baseSalary * 0.8;  // 20% less than base salary
    }
    
    void displayInfo() const override {
        Employee::displayInfo();
        cout << "Role: Intern, University: " << university 
             << ", Duration: " << duration << " months"
             << ", Total Salary: $" << calculateSalary() << endl;
    }
    
    void attendTraining() {
        cout << name << " is attending training sessions." << endl;
    }
};

class Company {
private:
    vector<Employee*> employees;
    string companyName;

public:
    Company(string name) : companyName(name) {}
    
    ~Company() {
        for (Employee* emp : employees) {
            delete emp;
        }
    }
    
    void addEmployee(Employee* emp) {
        employees.push_back(emp);
        cout << "Added employee: " << emp->getName() << endl;
    }
    
    void displayAllEmployees() const {
        cout << "\n=== " << companyName << " Employee List ===" << endl;
        for (const Employee* emp : employees) {
            emp->displayInfo();
            cout << "---" << endl;
        }
    }
    
    double calculateTotalPayroll() const {
        double total = 0;
        for (const Employee* emp : employees) {
            total += emp->calculateSalary();
        }
        return total;
    }
    
    void displayPayrollSummary() const {
        cout << "\n=== Payroll Summary ===" << endl;
        cout << "Total employees: " << employees.size() << endl;
        cout << "Total payroll: $" << calculateTotalPayroll() << endl;
        cout << "Average salary: $" << calculateTotalPayroll() / employees.size() << endl;
    }
};

int main() {
    Company company("TechCorp");
    
    // Add different types of employees
    company.addEmployee(new Manager("Alice Johnson", "Engineering", 80000, 15000, 8));
    company.addEmployee(new Developer("Bob Smith", "Engineering", 70000, "C++", 12));
    company.addEmployee(new Developer("Carol Davis", "Engineering", 65000, "Python", 8));
    company.addEmployee(new Intern("David Wilson", "Engineering", 30000, "MIT", 6));
    company.addEmployee(new Manager("Eve Brown", "Marketing", 75000, 10000, 5));
    
    company.displayAllEmployees();
    company.displayPayrollSummary();
    
    return 0;
}
```

### 2. **Vehicle Management System**
```cpp
#include <iostream>
#include <string>
#include <vector>
using namespace std;

class Vehicle {
protected:
    string make, model;
    int year;
    double price;

public:
    Vehicle(string mk, string md, int yr, double pr) 
        : make(mk), model(md), year(yr), price(pr) {}
    
    virtual ~Vehicle() = default;
    
    virtual void start() {
        cout << year << " " << make << " " << model << " is starting..." << endl;
    }
    
    virtual void stop() {
        cout << year << " " << make << " " << model << " is stopping..." << endl;
    }
    
    virtual double calculateInsurance() const {
        return price * 0.05;  // 5% of price
    }
    
    virtual void displayInfo() const {
        cout << year << " " << make << " " << model << " - $" << price << endl;
    }
    
    string getMake() const { return make; }
    string getModel() const { return model; }
    int getYear() const { return year; }
    double getPrice() const { return price; }
};

class Car : public Vehicle {
private:
    int doors;
    string fuelType;

public:
    Car(string mk, string md, int yr, double pr, int d, string fuel) 
        : Vehicle(mk, md, yr, pr), doors(d), fuelType(fuel) {}
    
    void drive() {
        cout << "Driving the " << doors << "-door " << make << " " << model << endl;
    }
    
    double calculateInsurance() const override {
        double base = Vehicle::calculateInsurance();
        return (doors == 2) ? base * 1.2 : base;  // Sports cars cost more
    }
    
    void displayInfo() const override {
        Vehicle::displayInfo();
        cout << "Type: Car, Doors: " << doors << ", Fuel: " << fuelType << endl;
    }
};

class Motorcycle : public Vehicle {
private:
    int engineCC;
    bool hasSidecar;

public:
    Motorcycle(string mk, string md, int yr, double pr, int cc, bool sidecar) 
        : Vehicle(mk, md, yr, pr), engineCC(cc), hasSidecar(sidecar) {}
    
    void ride() {
        cout << "Riding the " << engineCC << "cc " << make << " " << model << endl;
    }
    
    double calculateInsurance() const override {
        double base = Vehicle::calculateInsurance();
        return base * 1.5;  // Motorcycles are riskier
    }
    
    void displayInfo() const override {
        Vehicle::displayInfo();
        cout << "Type: Motorcycle, Engine: " << engineCC << "cc, Sidecar: " 
             << (hasSidecar ? "Yes" : "No") << endl;
    }
};

class Truck : public Vehicle {
private:
    double cargoCapacity;  // in tons
    bool isCommercial;

public:
    Truck(string mk, string md, int yr, double pr, double capacity, bool commercial) 
        : Vehicle(mk, md, yr, pr), cargoCapacity(capacity), isCommercial(commercial) {}
    
    void loadCargo() {
        cout << "Loading cargo into " << make << " " << model 
             << " (Capacity: " << cargoCapacity << " tons)" << endl;
    }
    
    double calculateInsurance() const override {
        double base = Vehicle::calculateInsurance();
        return isCommercial ? base * 2.0 : base * 1.3;
    }
    
    void displayInfo() const override {
        Vehicle::displayInfo();
        cout << "Type: Truck, Capacity: " << cargoCapacity << " tons, Commercial: " 
             << (isCommercial ? "Yes" : "No") << endl;
    }
};

class VehicleFleet {
private:
    vector<Vehicle*> vehicles;
    string fleetName;

public:
    VehicleFleet(string name) : fleetName(name) {}
    
    ~VehicleFleet() {
        for (Vehicle* vehicle : vehicles) {
            delete vehicle;
        }
    }
    
    void addVehicle(Vehicle* vehicle) {
        vehicles.push_back(vehicle);
        cout << "Added to fleet: " << vehicle->getYear() << " " 
             << vehicle->getMake() << " " << vehicle->getModel() << endl;
    }
    
    void displayFleet() const {
        cout << "\n=== " << fleetName << " Fleet ===" << endl;
        for (const Vehicle* vehicle : vehicles) {
            vehicle->displayInfo();
            cout << "Insurance: $" << vehicle->calculateInsurance() << endl;
            cout << "---" << endl;
        }
    }
    
    double getTotalValue() const {
        double total = 0;
        for (const Vehicle* vehicle : vehicles) {
            total += vehicle->getPrice();
        }
        return total;
    }
    
    double getTotalInsurance() const {
        double total = 0;
        for (const Vehicle* vehicle : vehicles) {
            total += vehicle->calculateInsurance();
        }
        return total;
    }
};

int main() {
    VehicleFleet fleet("City Transport");
    
    // Add different types of vehicles
    fleet.addVehicle(new Car("Toyota", "Camry", 2023, 25000, 4, "Gasoline"));
    fleet.addVehicle(new Car("Ford", "Mustang", 2022, 35000, 2, "Gasoline"));
    fleet.addVehicle(new Motorcycle("Harley-Davidson", "Street 750", 2023, 12000, 750, false));
    fleet.addVehicle(new Truck("Ford", "F-150", 2023, 40000, 2.5, false));
    fleet.addVehicle(new Truck("Volvo", "VNL", 2022, 80000, 40, true));
    
    fleet.displayFleet();
    
    cout << "\nFleet Summary:" << endl;
    cout << "Total Fleet Value: $" << fleet.getTotalValue() << endl;
    cout << "Total Insurance Cost: $" << fleet.getTotalInsurance() << endl;
    
    return 0;
}
```

## 💡 Best Practices

### 1. **Use Virtual Destructors**
```cpp
class Base {
public:
    virtual ~Base() = default;  // Virtual destructor for proper cleanup
    virtual void doSomething() = 0;
};

class Derived : public Base {
private:
    int* data;
public:
    Derived() : data(new int[100]) {}
    ~Derived() { delete[] data; }  // Will be called properly
    void doSomething() override {}
};
```

### 2. **Prefer Composition over Inheritance**
```cpp
// Instead of inheritance for "has-a" relationships
class Engine {
public:
    void start() { cout << "Engine starting" << endl; }
    void stop() { cout << "Engine stopping" << endl; }
};

class Car {
private:
    Engine engine;  // Composition
public:
    void start() { engine.start(); }
    void stop() { engine.stop(); }
};
```

### 3. **Use Protected Members Carefully**
```cpp
class Base {
protected:
    void protectedMethod() {  // Only for derived classes
        // Implementation
    }
    
private:
    void privateMethod() {    // Internal implementation only
        // Implementation
    }
    
public:
    void publicMethod() {     // Public interface
        // Implementation
    }
};
```

## 🔗 Related Topics
- [Object-Oriented Programming](./08-oop.md)
- [Polymorphism](./10-polymorphism.md)
- [Advanced Topics](./17-advanced-topics.md)

---
*Previous: [Object-Oriented Programming](./08-oop.md) | Next: [Polymorphism](./10-polymorphism.md)*
