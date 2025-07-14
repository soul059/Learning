# Core Java Concepts

A comprehensive guide to the core concepts of the Java programming language.

---

## 1. Introduction to Java

### What is Java?
Java is a high-level, class-based, object-oriented programming language that is designed to have as few implementation dependencies as possible. It is a general-purpose programming language intended to let application developers "write once, run anywhere" (WORA), meaning that compiled Java code can run on all platforms that support Java without the need for recompilation.

### Key Features of Java
- **Platform Independent**: Java code is compiled into an intermediate format called bytecode, which can be executed on any machine that has a Java Virtual Machine (JVM).
- **Object-Oriented**: Everything in Java is an object, which has some data and behavior. Java supports the core OOP concepts like Encapsulation, Inheritance, Polymorphism, and Abstraction.
- **Simple**: Java was designed to be easy to learn and is similar in syntax to C++.
- **Secure**: Java's security features include a security manager that defines access policies for applications.
- **Robust**: Java makes an effort to eliminate error-prone situations by emphasizing compile-time error checking and runtime checking.
- **Multithreaded**: Java has built-in support for multithreaded programming, which allows you to write programs that can do many things simultaneously.
- **High Performance**: With the use of Just-In-Time (JIT) compilers, Java enables high performance.

---

## 2. Java Basics

### Basic Syntax
- **Case-Sensitivity**: Java is case-sensitive. `myVariable` and `myvariable` are different.
- **Class Names**: The first letter should be in uppercase.
- **Method Names**: Should start with a lowercase letter.
- **File Names**: The file name must match the class name.

### Data Types
Java has two categories of data types:

1.  **Primitive Data Types**:
    - `byte`: 1 byte, range -128 to 127
    - `short`: 2 bytes, range -32,768 to 32,767
    - `int`: 4 bytes, range -2^31 to 2^31-1
    - `long`: 8 bytes, range -2^63 to 2^63-1
    - `float`: 4 bytes, for fractional numbers, 6-7 decimal digits
    - `double`: 8 bytes, for fractional numbers, ~15 decimal digits
    - `char`: 2 bytes, for a single character
    - `boolean`: 1 bit, `true` or `false`

2.  **Non-Primitive (Reference) Data Types**:
    - **Classes**: User-defined blueprints for objects.
    - **Interfaces**: A contract for a class.
    - **Arrays**: A collection of similar data types.
    - **Strings**: A sequence of characters.

### Variables
A variable is a container that holds a value.
- **Instance Variables**: Declared in a class but outside a method. Each object of the class has its own copy.
- **Static (Class) Variables**: Declared with the `static` keyword. There is only one copy of this variable, shared among all objects of the class.
- **Local Variables**: Declared inside a method. They are only accessible within that method.

### Operators
- **Arithmetic**: `+`, `-`, `*`, `/`, `%`
- **Relational**: `==`, `!=`, `>`, `<`, `>=`, `<=`
- **Logical**: `&&` (AND), `||` (OR), `!` (NOT)
- **Assignment**: `=`, `+=`, `-=`, `*=`, `/=`
- **Unary**: `++` (increment), `--` (decrement)
- **Bitwise**: `&`, `|`, `^`, `~`

### Control Flow
- **Conditional Statements**:
    - `if`, `if-else`, `if-else-if`
    - `switch`
- **Looping Statements**:
    - `for` loop
    - `while` loop
    - `do-while` loop
- **Branching Statements**:
    - `break`: Exits the loop or switch.
    - `continue`: Skips the current iteration of a loop.
    - `return`: Exits the method.

---

## 3. Object-Oriented Programming (OOP) in Java

OOP is a methodology to design a program using classes and objects.

### The Four Pillars of OOP

1.  **Encapsulation**:
    - The bundling of data (attributes) and methods (behaviors) that operate on the data into a single unit (a class).
    - It restricts direct access to some of an object's components, which is a means of preventing accidental interference and misuse of the methods.
    - Achieved using `private` access modifiers and providing `public` getter/setter methods.
    ```java
    public class Person {
        private String name; // private data

        public String getName() { // getter
            return name;
        }

        public void setName(String newName) { // setter
            this.name = newName;
        }
    }
    ```

2.  **Inheritance**:
    - A mechanism where a new class (subclass/child) inherits attributes and methods from an existing class (superclass/parent).
    - Promotes code reusability.
    - Achieved using the `extends` keyword.
    ```java
    class Animal {
        void eat() {
            System.out.println("This animal eats food.");
        }
    }

    class Dog extends Animal { // Dog inherits from Animal
        void bark() {
            System.out.println("The dog barks.");
        }
    }
    ```

3.  **Polymorphism**:
    - The ability of an object to take on many forms. The most common use of polymorphism in OOP occurs when a parent class reference is used to refer to a child class object.
    - Two types:
        - **Compile-time Polymorphism (Method Overloading)**: A class has multiple methods with the same name but different parameters.
        - **Runtime Polymorphism (Method Overriding)**: A subclass provides a specific implementation for a method that is already defined in its superclass.
    ```java
    // Method Overloading
    class Adder {
        int add(int a, int b) { return a + b; }
        double add(double a, double b) { return a + b; }
    }

    // Method Overriding
    class Animal {
        void makeSound() { System.out.println("Animal sound"); }
    }
    class Cat extends Animal {
        @Override
        void makeSound() { System.out.println("Meow"); }
    }
    ```

4.  **Abstraction**:
    - Hiding the implementation details and showing only the functionality to the user.
    - Achieved using `abstract` classes and `interfaces`.
    - **Abstract Class**: A class that cannot be instantiated. It can have abstract and non-abstract methods.
    - **Interface**: A blueprint of a class. It has static constants and abstract methods only. A class can implement multiple interfaces.
    ```java
    // Abstract Class
    abstract class Shape {
        abstract void draw(); // abstract method
    }

    // Interface
    interface Drawable {
        void draw();
    }
    ```

### Classes and Objects
- **Class**: A blueprint for creating objects.
- **Object**: An instance of a class.

### Constructors
A special method used to initialize objects. It is called when an object of a class is created.
- It must have the same name as the class.
- It does not have a return type.
- Can be overloaded.

---

## 4. Java Collections Framework

A set of classes and interfaces that implement commonly reusable collection data structures.

- **`Collection` Interface**: Root interface.
- **`List` Interface**: Ordered collection, allows duplicates.
    - `ArrayList`: Dynamic array.
    - `LinkedList`: Doubly-linked list.
- **`Set` Interface**: Unordered collection, does not allow duplicates.
    - `HashSet`: Uses a hash table for storage.
    - `TreeSet`: Stores elements in a sorted order.
- **`Queue` Interface**: FIFO (First-In, First-Out) data structure.
    - `PriorityQueue`: Elements are ordered based on their natural ordering or by a `Comparator`.
- **`Map` Interface**: Key-value pairs. Keys must be unique.
    - `HashMap`: Unordered, allows one `null` key.
    - `TreeMap`: Sorted by key.

---

## 5. Exception Handling

An event that disrupts the normal flow of the program.

- **`try`**: The block of code to be monitored for exceptions.
- **`catch`**: Catches the exception thrown by the `try` block.
- **`finally`**: Always executes, whether an exception is handled or not.
- **`throw`**: Used to manually throw an exception.
- **`throws`**: Declares the exceptions that can be thrown by a method.

### Checked vs. Unchecked Exceptions
- **Checked Exceptions**: Checked at compile-time (e.g., `IOException`, `SQLException`). Must be handled or declared.
- **Unchecked Exceptions (Runtime Exceptions)**: Not checked at compile-time (e.g., `NullPointerException`, `ArrayIndexOutOfBoundsException`).

---

## 6. Multithreading

Executing multiple threads (lightweight processes) simultaneously.

- **Creating a Thread**:
    1.  Extend the `Thread` class.
    2.  Implement the `Runnable` interface (preferred).
- **Thread Lifecycle**: New, Runnable, Running, Blocked/Waiting, Terminated.
- **Synchronization**: A mechanism to control access of multiple threads to any shared resource.
    - `synchronized` keyword (for methods or blocks).
    - `java.util.concurrent.locks.Lock` interface.

---

## 7. I/O Streams

Java uses streams to perform I/O.

- **Byte Streams**: For handling I/O of 8-bit bytes (`InputStream`, `OutputStream`).
- **Character Streams**: For handling I/O of 16-bit Unicode characters (`Reader`, `Writer`).
- **`File` Class**: Represents a file or directory path.

---

## 8. Java 8+ Features

Significant changes were introduced in Java 8.

- **Lambda Expressions**: A short block of code which takes in parameters and returns a value.
  ```java
  (parameter1, parameter2) -> expression
  ```
- **Functional Interfaces**: An interface with exactly one abstract method. `@FunctionalInterface` annotation.
- **Stream API**: For processing collections of objects in a functional style. Supports operations like `filter`, `map`, `reduce`.
- **`Optional` Class**: A container object which may or may not contain a non-null value. Helps in avoiding `NullPointerException`.
