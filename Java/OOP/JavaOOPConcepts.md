# Object-Oriented Programming (OOP) in Java: A Detailed Guide

This document provides a deep dive into the core principles of Object-Oriented Programming (OOP) in Java, complete with real-world examples to illustrate each concept.

---

## Introduction to OOP

Object-Oriented Programming is a programming paradigm based on the concept of "objects", which can contain data in the form of fields (often known as attributes or properties) and code in the form of procedures (often known as methods). The primary purpose of OOP is to increase the flexibility and maintainability of programs.

Java is a pure object-oriented language, and its core principles are:
1.  **Encapsulation**
2.  **Inheritance**
3.  **Polymorphism**
4.  **Abstraction**

---

## 1. Encapsulation

Encapsulation is the mechanism of bundling the data (variables) and the code acting on the data (methods) together as a single unit. It's about hiding the internal state of an object from the outside and only exposing the necessary functionality. This prevents external code from accidentally or maliciously corrupting the object's state.

**How it's achieved in Java:**
-   Declare the variables of a class as `private`.
-   Provide `public` setter and getter methods to modify and view the variables' values.

### Real-Time Example: A Bank Account

A bank account is a perfect real-world example. You don't want anyone to be able to arbitrarily change your account balance. Instead, you interact with it through defined methods like `deposit` and `withdraw`, which can contain important logic.

```java
// BankAccount.java

public class BankAccount {
    // private variable - hidden from other classes
    private double balance;
    private final String accountNumber;

    public BankAccount(String accountNumber, double initialBalance) {
        this.accountNumber = accountNumber;
        if (initialBalance >= 0) {
            this.balance = initialBalance;
        } else {
            this.balance = 0;
            System.out.println("Initial balance cannot be negative. Set to 0.");
        }
    }

    // public getter method - provides read-only access
    public double getBalance() {
        return this.balance;
    }

    public String getAccountNumber() {
        return this.accountNumber;
    }

    // public method to deposit money
    public void deposit(double amount) {
        if (amount > 0) {
            this.balance += amount;
            System.out.println("Deposited: " + amount);
        } else {
            System.out.println("Deposit amount must be positive.");
        }
    }

    // public method to withdraw money
    public void withdraw(double amount) {
        if (amount > 0 && amount <= this.balance) {
            this.balance -= amount;
            System.out.println("Withdrew: " + amount);
        } else if (amount > this.balance) {
            System.out.println("Withdrawal failed. Insufficient funds.");
        } else {
            System.out.println("Withdrawal amount must be positive.");
        }
    }
}

// Main.java
public class Main {
    public static void main(String[] args) {
        BankAccount myAccount = new BankAccount("123456789", 500.00);

        // Cannot access balance directly:
        // myAccount.balance = 1000000; // This will cause a compile error

        // Interact through public methods
        System.out.println("Current Balance: " + myAccount.getBalance()); // 500.00
        myAccount.deposit(200.00); // Deposited: 200.0
        myAccount.withdraw(100.00); // Withdrew: 100.0
        System.out.println("Final Balance: " + myAccount.getBalance()); // 600.00

        myAccount.withdraw(700.00); // Withdrawal failed. Insufficient funds.
    }
}
```

---

## 2. Inheritance

Inheritance is a mechanism wherein one class acquires the properties (fields) and behaviors (methods) of another. It creates a parent-child relationship between classes. The class that inherits is called the **subclass** (or child class), and the class it inherits from is the **superclass** (or parent class).

**Why use it?**
-   For code reusability.
-   For method overriding (runtime polymorphism).

**How it's achieved in Java:**
-   Using the `extends` keyword.

### Real-Time Example: Vehicle Hierarchy

Consider a system for managing different types of vehicles. All vehicles share common attributes (like `brand`, `speed`) and behaviors (like `start()`, `stop()`). Instead of rewriting this code for every vehicle type, we can use inheritance.

```java
// Vehicle.java (Superclass)
public class Vehicle {
    protected String brand; // protected allows subclass access

    public Vehicle(String brand) {
        this.brand = brand;
    }

    public void start() {
        System.out.println("The vehicle's engine starts.");
    }

    public void stop() {
        System.out.println("The vehicle's engine stops.");
    }

    public String getBrand() {
        return brand;
    }
}

// Car.java (Subclass)
public class Car extends Vehicle {
    private int numberOfDoors;

    public Car(String brand, int numberOfDoors) {
        super(brand); // Call the superclass constructor
        this.numberOfDoors = numberOfDoors;
    }

    // Car-specific method
    public void openTrunk() {
        System.out.println("The car's trunk is open.");
    }
}

// Motorcycle.java (Subclass)
public class Motorcycle extends Vehicle {
    public Motorcycle(String brand) {
        super(brand);
    }

    // Overriding the start method for a more specific behavior
    @Override
    public void start() {
        System.out.println("The " + brand + " motorcycle roars to life!");
    }
}

// Main.java
public class Main {
    public static void main(String[] args) {
        Car myCar = new Car("Toyota", 4);
        myCar.start(); // Inherited method
        System.out.println("Brand: " + myCar.getBrand()); // Inherited method
        myCar.openTrunk(); // Specific method

        System.out.println("---");

        Motorcycle myBike = new Motorcycle("Harley-Davidson");
        myBike.start(); // Overridden method
        myBike.stop(); // Inherited method
    }
}
```

---

## 3. Polymorphism

Polymorphism, meaning "many forms," is the ability of a variable, function, or object to take on multiple forms. In Java, it allows us to perform a single action in different ways.

There are two types of polymorphism in Java:

1.  **Compile-time Polymorphism (Method Overloading):** This is achieved by having multiple methods with the same name but different parameters (either number of parameters or type of parameters) in the same class.
2.  **Runtime Polymorphism (Method Overriding):** This is achieved when a subclass has a method with the same name, parameters, and return type as a method in its superclass. The method to be executed is determined at runtime.

### Real-Time Example: A Payment Processing System

Imagine an e-commerce application that accepts various payment methods: Credit Card, PayPal, or Cryptocurrency. You can use polymorphism to process payments without needing to know the specific type of payment at compile time.

```java
// PaymentMethod.java (Interface for Abstraction and Polymorphism)
public interface PaymentMethod {
    void pay(double amount);
}

// CreditCardPayment.java
public class CreditCardPayment implements PaymentMethod {
    private String cardNumber;

    public CreditCardPayment(String cardNumber) {
        this.cardNumber = cardNumber;
    }

    @Override
    public void pay(double amount) {
        System.out.println("Processing credit card payment of $" + amount + " with card " + cardNumber);
        // Add logic for credit card transaction
    }
}

// PayPalPayment.java
public class PayPalPayment implements PaymentMethod {
    private String email;

    public PayPalPayment(String email) {
        this.email = email;
    }

    @Override
    public void pay(double amount) {
        System.out.println("Processing PayPal payment of $" + amount + " for user " + email);
        // Add logic for PayPal transaction
    }
}

// Main.java
public class Main {
    public static void main(String[] args) {
        // A single type 'PaymentMethod' can hold different object types
        PaymentMethod creditCard = new CreditCardPayment("1111-2222-3333-4444");
        PaymentMethod payPal = new PayPalPayment("user@example.com");

        processOrder(creditCard, 150.75);
        processOrder(payPal, 88.50);
    }

    // This method works with any class that implements PaymentMethod
    public static void processOrder(PaymentMethod paymentMethod, double amount) {
        System.out.println("Starting payment process...");
        // The correct 'pay' method is called at RUNTIME
        paymentMethod.pay(amount);
        System.out.println("Payment process finished.");
        System.out.println("---");
    }
}
```

---

## 4. Abstraction

Abstraction is the concept of hiding the complex implementation details and showing only the essential features of the object. It helps in managing complexity.

**How it's achieved in Java:**
-   **Abstract Classes:** A class declared with the `abstract` keyword. It can have both abstract (methods without a body) and concrete methods. It cannot be instantiated.
-   **Interfaces:** A blueprint of a class. It can only have abstract methods (before Java 8) and static, final constants. A class `implements` an interface, thereby inheriting the abstract methods.

### Real-Time Example: Database Connection

When your application needs to connect to a database, you want a common way to interact with it, regardless of whether it's MySQL, PostgreSQL, or Oracle. Abstraction allows you to define a contract for what a "database connection" should do, without worrying about the specific implementation details of each database type.

```java
// DatabaseConnector.java (Abstract Class)
public abstract class DatabaseConnector {
    // Concrete method - shared by all subclasses
    public void getVersion() {
        System.out.println("Using standard DB API version 1.0");
    }

    // Abstract methods - must be implemented by subclasses
    public abstract void connect();
    public abstract void disconnect();
    public abstract void executeQuery(String query);
}

// MySqlConnector.java
public class MySqlConnector extends DatabaseConnector {
    @Override
    public void connect() {
        System.out.println("Connecting to MySQL database...");
        // MySQL specific connection logic
    }

    @Override
    public void disconnect() {
        System.out.println("Disconnecting from MySQL database.");
    }

    @Override
    public void executeQuery(String query) {
        System.out.println("Executing MySQL query: " + query);
    }
}

// PostgreSqlConnector.java
public class PostgreSqlConnector extends DatabaseConnector {
    @Override
    public void connect() {
        System.out.println("Initializing connection to PostgreSQL...");
        // PostgreSQL specific connection logic
    }

    @Override
    public void disconnect() {
        System.out.println("Closing PostgreSQL connection.");
    }

    @Override
    public void executeQuery(String query) {
        System.out.println("Running PostgreSQL query: " + query);
    }
}

// Main.java
public class Main {
    public static void main(String[] args) {
        // You can't do: new DatabaseConnector(); // Compile error

        DatabaseConnector myDb = new MySqlConnector();
        runDatabaseOperations(myDb);

        System.out.println("\nSwitching database...\n");

        DatabaseConnector pgDb = new PostgreSqlConnector();
        runDatabaseOperations(pgDb);
    }

    // This method doesn't care about the specific DB type, only the abstraction
    public static void runDatabaseOperations(DatabaseConnector db) {
        db.connect();
        db.executeQuery("SELECT * FROM users");
        db.disconnect();
    }
}
```
