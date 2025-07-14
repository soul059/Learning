# Understanding Methods in Java

A method in Java is a block of code that performs a specific task. It runs only when it is called. You can pass data, known as parameters, into a method. Methods are used to perform certain actions, and they are also known as functions.

The primary advantages of using methods are:
-   **Code Reusability**: Define the code once, and use it many times.
-   **Readability**: By breaking down a complex problem into smaller, manageable pieces, methods make the code easier to read and understand.
-   **Maintainability**: It's easier to fix and maintain code that is well-organized into methods.

---

## 1. Method Declaration

The declaration of a method provides its signature, which includes the method's name, return type, and a list of parameters.

### Syntax

```java
accessModifier returnType methodName(parameterList) {
    // method body: block of code
    return value; // if returnType is not void
}
```

### Components of a Method Declaration

1.  **`accessModifier`**: (e.g., `public`, `private`, `protected`, or default) - It defines the visibility of the method.
    -   `public`: The method is accessible from any other class.
    -   `protected`: The method is accessible within the same package and subclasses.
    -   `default` (no modifier): The method is accessible only within the same package.
    -   `private`: The method is accessible only within the same class.

2.  **`returnType`**: The data type of the value returned by the method. If the method does not return any value, its return type is `void`.

3.  **`methodName`**: A unique name that identifies the method. It follows the `camelCase` convention (e.g., `calculateSum`).

4.  **`parameterList`**: A comma-separated list of input parameters, preceded by their data types, enclosed in parentheses `()`. If there are no parameters, you must use empty parentheses.

5.  **`method body`**: The block of code, enclosed in curly braces `{}`, that needs to be executed to perform the intended action.

### Example

```java
public class Calculator {
    // A public method that takes two integers and returns their sum
    public int add(int num1, int num2) {
        int sum = num1 + num2;
        return sum;
    }

    // A private method with no return value (void)
    private void displayMessage(String message) {
        System.out.println(message);
    }
}
```

---

## 2. Calling a Method

To use a method, you need to call it. If the method is not `static`, you must first create an object of the class in which the method is defined.

```java
public class Main {
    public static void main(String[] args) {
        // Create an object of the Calculator class
        Calculator myCalculator = new Calculator();

        // Call the 'add' method on the object
        int result = myCalculator.add(10, 25);

        System.out.println("The sum is: " + result); // Output: The sum is: 35
    }
}
```

---

## 3. Types of Methods

### a) Pre-defined Methods

These are the methods that are already defined in the Java Class Library (JCL). For example, `println()`, `sqrt()`, `length()`.

```java
public class PredefinedExample {
    public static void main(String[] args) {
        // Using the sqrt() method from the Math class
        double number = 64;
        double squareRoot = Math.sqrt(number);
        System.out.println("Square root of " + number + " is: " + squareRoot);
    }
}
```

### b) User-defined Methods

These are methods created by the programmer to perform specific tasks. The `add()` method in the `Calculator` class is a user-defined method.

---

## 4. Static Methods

A method declared with the `static` keyword is a static method.
-   It belongs to the class rather than the object of a class.
-   It can be called without creating an instance of the class, by using the class name.
-   It can only access static data and call other static methods directly.

### Example

```java
public class Utility {
    // A static method
    public static int max(int a, int b) {
        return (a > b) ? a : b;
    }
}

public class Main {
    public static void main(String[] args) {
        // Calling the static method without creating an object
        int maximum = Utility.max(100, 200);
        System.out.println("The maximum value is: " + maximum); // Output: 200
    }
}
```

---

## 5. Instance Methods

Any method that is not declared as `static` is an instance method.
-   It belongs to the object of a class.
-   It must be called through an object of its class.
-   It can access both instance variables and static variables.

The `add()` method in the first `Calculator` example is an instance method.

---

## 6. Method Overloading

If a class has multiple methods having the same name but different in parameters (number of arguments, type of arguments, or both), it is known as method overloading. It increases the readability of the program.

### Example

```java
public class Display {
    public void show(String name) {
        System.out.println("Displaying string: " + name);
    }

    public void show(int number) {
        System.out.println("Displaying integer: " + number);
    }

    public void show(String name, int number) {
        System.out.println("Displaying: " + name + " and " + number);
    }
}

public class Main {
    public static void main(String[] args) {
        Display d = new Display();
        d.show("Hello");       // Calls the first method
        d.show(123);           // Calls the second method
        d.show("World", 456);  // Calls the third method
    }
}
```

---

## 7. Constructors

A constructor is a special type of method used to initialize an object.
-   It has the same name as its class.
-   It does not have an explicit return type, not even `void`.
-   It is called automatically when an object is created.

### Example

```java
public class Car {
    String model;

    // Constructor
    public Car(String modelName) {
        this.model = modelName; // Initialize the model variable
        System.out.println("Car object created for model: " + this.model);
    }

    public static void main(String[] args) {
        // The constructor is called when the object is created
        Car myCar = new Car("Tesla Model S");
    }
}
```

---

## 8. The `main` Method

The `main()` method is the entry point for any Java application. The Java Virtual Machine (JVM) starts the execution of a Java program from this method.

```java
public static void main(String[] args) {
    // Code execution begins here
}
```
-   `public`: It is an access modifier, which means it can be called by the JVM from anywhere.
-   `static`: It can be called without creating an object of the class.
-   `void`: It does not return any value.
-   `main`: The name of the method, configured in the JVM.
-   `String[] args`: An array of strings to store command-line arguments.
