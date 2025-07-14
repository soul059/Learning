# Java Data Types and Variables

## Variables

A variable is a container that holds data that can be changed during program execution. In Java, every variable must be declared with a specific data type.

### Variable Declaration Syntax:
```java
dataType variableName;
dataType variableName = initialValue;
```

### Variable Naming Rules:
1. Must start with letter, underscore (_), or dollar sign ($)
2. Cannot start with a digit
3. Cannot be a Java keyword
4. Case-sensitive
5. No spaces allowed

### Naming Conventions:
- Use camelCase for variable names: `firstName`, `totalAmount`
- Use descriptive names: `studentAge` instead of `a`
- Constants use UPPER_CASE: `MAX_SIZE`, `PI`

## Data Types

Java has two categories of data types:

### 1. Primitive Data Types
Built-in data types provided by Java language.

#### Numeric Types

**Integer Types:**
```java
byte b = 127;           // 8-bit, range: -128 to 127
short s = 32767;        // 16-bit, range: -32,768 to 32,767
int i = 2147483647;     // 32-bit, range: -2^31 to 2^31-1
long l = 9223372036854775807L; // 64-bit, range: -2^63 to 2^63-1
```

**Floating-Point Types:**
```java
float f = 3.14f;        // 32-bit IEEE 754 floating point
double d = 3.141592653589793; // 64-bit IEEE 754 floating point
```

#### Character Type
```java
char c = 'A';           // 16-bit Unicode character
char unicode = '\u0041'; // Unicode representation of 'A'
char digit = '5';
```

#### Boolean Type
```java
boolean isTrue = true;
boolean isFalse = false;
```

### 2. Non-Primitive (Reference) Data Types
- Classes
- Interfaces
- Arrays
- Strings

## Data Type Characteristics

| Data Type | Size | Range | Default Value |
|-----------|------|-------|---------------|
| byte | 1 byte | -128 to 127 | 0 |
| short | 2 bytes | -32,768 to 32,767 | 0 |
| int | 4 bytes | -2,147,483,648 to 2,147,483,647 | 0 |
| long | 8 bytes | -9,223,372,036,854,775,808 to 9,223,372,036,854,775,807 | 0L |
| float | 4 bytes | ±3.40282347E+38F | 0.0f |
| double | 8 bytes | ±1.79769313486231570E+308 | 0.0d |
| char | 2 bytes | 0 to 65,536 (Unicode) | '\u0000' |
| boolean | 1 bit | true or false | false |

## Type Casting

### Implicit Casting (Widening)
Automatic conversion from smaller to larger data type:
```java
int i = 100;
long l = i;     // int to long
float f = l;    // long to float
double d = f;   // float to double
```

### Explicit Casting (Narrowing)
Manual conversion from larger to smaller data type:
```java
double d = 100.04;
long l = (long) d;    // 100
int i = (int) l;      // 100
short s = (short) i;  // 100
```

### Casting Examples:
```java
// Potential data loss
int i = 257;
byte b = (byte) i;    // Result: 1 (due to overflow)

// Character and numeric casting
char c = 'A';
int ascii = c;        // 65 (ASCII value of 'A')
char fromInt = (char) 66;  // 'B'
```

## Variable Types

### 1. Local Variables
```java
public void method() {
    int localVar = 10;  // Local variable
    // Must be initialized before use
}
```

### 2. Instance Variables (Non-static)
```java
public class Student {
    String name;        // Instance variable
    int age;           // Instance variable
}
```

### 3. Class Variables (Static)
```java
public class Student {
    static String school = "ABC School";  // Class variable
    static int totalStudents = 0;         // Class variable
}
```

## Constants

### Using `final` keyword:
```java
final int MAX_SIZE = 100;
final double PI = 3.14159;
final String COMPANY_NAME = "TechCorp";
```

### Static Constants:
```java
public class Constants {
    public static final int MAX_USERS = 1000;
    public static final String DATABASE_URL = "jdbc:mysql://localhost";
}
```

## Literals

### Integer Literals:
```java
int decimal = 42;        // Decimal
int binary = 0b101010;   // Binary (Java 7+)
int octal = 052;         // Octal
int hex = 0x2A;          // Hexadecimal
long bigNumber = 123456789L;  // Long literal
```

### Floating-Point Literals:
```java
double d1 = 123.45;
double d2 = 1.2345e2;    // Scientific notation
float f1 = 123.45f;      // Float literal
float f2 = 1.2345e2f;
```

### Character Literals:
```java
char c1 = 'A';
char c2 = '\n';          // Newline
char c3 = '\t';          // Tab
char c4 = '\\';          // Backslash
char c5 = '\'';          // Single quote
char c6 = '\u0041';      // Unicode
```

### String Literals:
```java
String s1 = "Hello World";
String s2 = "Line 1\nLine 2";
String s3 = "Path: C:\\Users\\John";
```

### Boolean Literals:
```java
boolean b1 = true;
boolean b2 = false;
```

## Escape Sequences

| Escape Sequence | Description |
|----------------|-------------|
| \n | Newline |
| \t | Tab |
| \r | Carriage return |
| \\ | Backslash |
| \' | Single quote |
| \" | Double quote |
| \b | Backspace |
| \f | Form feed |
| \0 | Null character |

## Variable Scope

### Block Scope:
```java
public void method() {
    int x = 10;         // Method scope
    
    if (true) {
        int y = 20;     // Block scope
        System.out.println(x);  // Accessible
        System.out.println(y);  // Accessible
    }
    
    // System.out.println(y);  // Error: y not accessible
}
```

### Class Scope:
```java
public class Example {
    private int instanceVar = 10;    // Instance variable
    private static int classVar = 20; // Class variable
    
    public void method() {
        System.out.println(instanceVar);  // Accessible
        System.out.println(classVar);     // Accessible
    }
}
```

## Wrapper Classes

Each primitive type has a corresponding wrapper class:

| Primitive | Wrapper Class |
|-----------|---------------|
| byte | Byte |
| short | Short |
| int | Integer |
| long | Long |
| float | Float |
| double | Double |
| char | Character |
| boolean | Boolean |

### Autoboxing and Unboxing:
```java
// Autoboxing: primitive to wrapper
Integer i = 10;         // int to Integer
Double d = 3.14;        // double to Double

// Unboxing: wrapper to primitive
int primitive = i;      // Integer to int
double primDouble = d;  // Double to double

// Null values possible with wrapper classes
Integer nullInteger = null;  // Valid
// int nullInt = null;   // Error: primitives cannot be null
```

## Memory Allocation

### Stack Memory:
- Stores local variables and method calls
- Automatic memory management
- Faster access
- Limited size

### Heap Memory:
- Stores objects and instance variables
- Garbage collection manages memory
- Slower access than stack
- Larger size

## Examples and Best Practices

### Variable Declaration Examples:
```java
public class VariableExamples {
    // Class variables
    static final String COMPANY = "TechCorp";
    static int employeeCount = 0;
    
    // Instance variables
    private String name;
    private int age;
    private double salary;
    
    public void demonstrateVariables() {
        // Local variables
        int localInt = 100;
        String localString = "Hello";
        boolean isActive = true;
        
        // Array variables
        int[] numbers = {1, 2, 3, 4, 5};
        String[] names = new String[10];
        
        // Type casting examples
        double price = 99.99;
        int roundedPrice = (int) price;  // 99
        
        // Using wrapper classes
        Integer boxedInt = Integer.valueOf(42);
        int unboxedInt = boxedInt.intValue();
    }
}
```

### Best Practices:
1. Initialize variables before use
2. Use appropriate data types (don't use `int` for small numbers if `byte` suffices)
3. Use meaningful variable names
4. Declare variables in the smallest scope needed
5. Use constants for fixed values
6. Be careful with floating-point precision
7. Avoid magic numbers - use named constants instead
