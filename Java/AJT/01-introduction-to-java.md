# Introduction to Java

## What is Java?

Java is a high-level, object-oriented programming language developed by Sun Microsystems (now owned by Oracle) in 1995. It follows the principle of "Write Once, Run Anywhere" (WORA), meaning Java code can run on any platform that has a Java Virtual Machine (JVM).

## Key Features of Java

### 1. Platform Independence
- Java code is compiled into bytecode, which runs on the JVM
- JVM acts as an intermediary between the operating system and Java application
- Same Java program can run on Windows, Linux, macOS, etc.

### 2. Object-Oriented Programming (OOP)
- Everything in Java is an object (except primitive data types)
- Supports all OOP principles: Encapsulation, Inheritance, Polymorphism, Abstraction

### 3. Automatic Memory Management
- Garbage collection automatically manages memory
- Prevents memory leaks and reduces programmer burden

### 4. Strong Type Checking
- Compile-time and runtime checking
- Helps catch errors early in development

### 5. Multi-threading Support
- Built-in support for concurrent programming
- Allows multiple threads to run simultaneously

### 6. Security
- Built-in security features
- Bytecode verification
- Security manager controls access to system resources

### 7. Rich Standard Library
- Extensive API for common programming tasks
- Collections framework, I/O operations, networking, etc.

## Java Architecture

```
Source Code (.java) → Compiler (javac) → Bytecode (.class) → JVM → Machine Code
```

### Components:

1. **Java Development Kit (JDK)**: Complete development environment
2. **Java Runtime Environment (JRE)**: Runtime environment for executing Java programs
3. **Java Virtual Machine (JVM)**: Executes Java bytecode

## Java Program Structure

```java
// Package declaration (optional)
package com.example;

// Import statements (optional)
import java.util.Scanner;

// Class declaration
public class HelloWorld {
    // Main method - entry point of program
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}
```

## Basic Syntax Rules

1. **Case Sensitivity**: Java is case-sensitive
2. **Class Names**: Should start with uppercase letter
3. **Method Names**: Should start with lowercase letter (camelCase)
4. **File Names**: Must match the public class name
5. **Semicolons**: Required at the end of statements
6. **Blocks**: Defined by curly braces `{}`

## Comments

```java
// Single-line comment

/*
 * Multi-line comment
 * Can span multiple lines
 */

/**
 * JavaDoc comment
 * Used for documentation
 * @param args command line arguments
 */
```

## Reserved Words (Keywords)

Java has 53 reserved words that cannot be used as identifiers:

```
abstract    assert      boolean     break       byte        case
catch       char        class       const       continue    default
do          double      else        enum        extends     final
finally     float       for         goto        if          implements
import      instanceof  int         interface   long        native
new         package     private     protected   public      return
short       static      strictfp    super       switch      synchronized
this        throw       throws      transient   try         void
volatile    while
```

## Hello World Program

```java
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}
```

### Compilation and Execution:
```bash
javac HelloWorld.java  # Compiles to HelloWorld.class
java HelloWorld        # Runs the program
```

## Java Editions

1. **Java SE (Standard Edition)**: Core Java platform
2. **Java EE (Enterprise Edition)**: Enterprise applications
3. **Java ME (Micro Edition)**: Mobile and embedded devices
4. **JavaFX**: Rich internet applications

## History and Versions

- **1995**: Java 1.0 (Oak)
- **1997**: Java 1.1
- **1998**: Java 1.2 (J2SE)
- **2000**: Java 1.3
- **2002**: Java 1.4
- **2004**: Java 5.0 (major update)
- **2006**: Java 6
- **2011**: Java 7
- **2014**: Java 8 (LTS)
- **2017**: Java 9
- **2018**: Java 10, 11 (LTS)
- **2019**: Java 12, 13
- **2020**: Java 14, 15
- **2021**: Java 16, 17 (LTS)
- **2022**: Java 18, 19
- **2023**: Java 20, 21 (LTS)

## Setting Up Java Development Environment

### 1. Install JDK
- Download from Oracle or OpenJDK
- Set JAVA_HOME environment variable
- Add to PATH

### 2. IDE Options
- **IntelliJ IDEA**: Popular commercial IDE
- **Eclipse**: Free, open-source IDE
- **NetBeans**: Free IDE from Apache
- **Visual Studio Code**: Lightweight editor with Java extensions

### 3. Build Tools
- **Maven**: Project management and build automation
- **Gradle**: Build automation tool
- **Ant**: Java-based build tool

## Best Practices

1. Follow naming conventions
2. Use meaningful variable and method names
3. Keep classes and methods small and focused
4. Write clear comments and documentation
5. Handle exceptions properly
6. Use appropriate access modifiers
7. Follow DRY (Don't Repeat Yourself) principle
8. Write unit tests
