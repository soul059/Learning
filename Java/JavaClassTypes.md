# A Guide to Different Class Types in Java

In Java, a class is the fundamental building block of an Object-Oriented Program. While most developers are familiar with the standard top-level class, Java supports several other specialized types of classes, each with a specific purpose. This guide explores them in detail.

---

## 1. Standard (Top-Level) Class

This is the most common type of class. It's a class defined in a `.java` file, typically as `public`, and is not enclosed within any other class.

```java
// A standard, top-level class
public class Car {
    private String model;

    public Car(String model) {
        this.model = model;
    }

    public void drive() {
        System.out.println("The " + model + " is driving.");
    }
}
```

---

## 2. Nested Classes

A nested class is a class defined within another class. They are used to group classes that belong together, which increases encapsulation and creates more readable and maintainable code.

There are two main categories of nested classes:
-   **Static Nested Classes**
-   **Inner Classes** (non-static nested classes)

### a) Static Nested Class

A class that is declared `static` inside another class.
-   **Behavior**: It behaves like a regular top-level class but is enclosed within the namespace of the outer class.
-   **Access**: It cannot access the instance members (non-static fields and methods) of the outer class directly. It can only access them through an object of the outer class. It can, however, access all static members of the outer class.

**When to use**: When you need a helper class that is strongly associated with its outer class but does not need access to its instance members.

```java
public class EnclosingClass {
    private static String staticOuterField = "Static Outer Field";
    private String instanceOuterField = "Instance Outer Field";

    // Static nested class
    public static class StaticNestedClass {
        public void display() {
            // Can access static members of the outer class
            System.out.println(staticOuterField);

            // Cannot access instance members directly
            // System.out.println(instanceOuterField); // COMPILE ERROR

            // Can access instance members through an object
            EnclosingClass outerObject = new EnclosingClass();
            System.out.println(outerObject.instanceOuterField);
        }
    }
}

// To instantiate a static nested class:
// EnclosingClass.StaticNestedClass nestedObject = new EnclosingClass.StaticNestedClass();
```

### b) Inner Classes (Non-Static)

An inner class is a non-static class defined inside another class. An instance of an inner class is implicitly associated with an instance of its outer class.

#### i. Member Inner Class
A class defined at the same level as instance variables.
-   **Behavior**: Each instance of a member inner class is tied to an instance of the outer class.
-   **Access**: It has full access to all members (static and instance) of the outer class, including private ones.

```java
public class Outer {
    private String message = "Hello from Outer";

    // Member inner class
    public class Inner {
        public void display() {
            // Can access instance members of the outer class directly
            System.out.println(message);
        }
    }
}

// To instantiate an inner class, you need an instance of the outer class:
// Outer outerObject = new Outer();
// Outer.Inner innerObject = outerObject.new Inner();
```

#### ii. Local Inner Class
A class defined within a method or a block of code.
-   **Behavior**: Its scope is restricted to the block where it is defined. It is rarely used.
-   **Access**: It can access the members of the outer class and also the `final` or *effectively final* local variables of the method it's in.

```java
public class MethodContainer {
    public void myMethod() {
        final String localVariable = "Local Variable";

        // Local inner class
        class LocalInner {
            public void print() {
                System.out.println("Inside local inner class: " + localVariable);
            }
        }

        // Create and use the local inner class within the method
        LocalInner local = new LocalInner();
        local.print();
    }
}
```

#### iii. Anonymous Inner Class
A local inner class that has no name. It is declared and instantiated in a single expression.
-   **Behavior**: They are used to create a one-time-use object, typically for implementing an interface or extending a class on the fly. They are heavily used in older Java code for event listeners.
-   **Syntax**: The syntax can be a bit unusual. You are creating an object of an interface or class and providing the implementation for its methods at the same time.

```java
interface Greeting {
    void sayHello();
}

public class AnonymousExample {
    public void displayGreeting() {
        // Anonymous inner class implementing the Greeting interface
        Greeting englishGreeting = new Greeting() {
            @Override
            public void sayHello() {
                System.out.println("Hello!");
            }
        };

        englishGreeting.sayHello();
    }
}
```
**Note**: With Java 8 and later, lambda expressions are often used as a more concise alternative to anonymous inner classes for implementing functional interfaces.

---

## 3. Wrapper Classes

For each primitive data type in Java, there is a corresponding wrapper class. These classes "wrap" the primitive value in an object.
-   **Purpose**: To use primitive types in contexts that require objects, such as in Java Collections (`ArrayList<int>` is not allowed, but `ArrayList<Integer>` is). They also provide useful utility methods.
-   **Autoboxing/Unboxing**: Java automatically converts between primitives and their wrapper classes (e.g., `int` to `Integer` and vice-versa).

| Primitive | Wrapper Class |
|-----------|---------------|
| `int`     | `Integer`     |
| `char`    | `Character`   |
| `double`  | `Double`      |
| `boolean` | `Boolean`     |
| ...and so on for `byte`, `short`, `long`, `float`. |

---

## 4. Abstract Classes

A class declared with the `abstract` keyword.
-   **Behavior**: It cannot be instantiated. It is designed to be subclassed.
-   **Content**: It can contain both abstract methods (methods without a body) and concrete methods (regular methods with a body).
-   **Purpose**: To provide a common base for a group of related subclasses, enforcing a contract through its abstract methods.

```java
public abstract class Animal {
    // Concrete method
    public void sleep() {
        System.out.println("Sleeping...");
    }
    // Abstract method - must be implemented by subclasses
    public abstract void makeSound();
}
```

---

## 5. Final Classes

A class declared with the `final` keyword.
-   **Behavior**: It cannot be extended (subclassed).
-   **Purpose**: To prevent inheritance, often for security or design reasons. For example, the `String` class is `final` so that its behavior cannot be altered.

```java
public final class ImmutableData {
    // This class cannot be extended
}
```

---

## 6. Enum Classes

An `enum` is a special type of class used to define a collection of constants.
-   **Behavior**: `enum` constants are implicitly `public`, `static`, and `final`. An `enum` can have constructors, methods, and fields.
-   **Purpose**: To provide a type-safe way to handle a fixed set of constants, which is much better than using `static final` integer constants.

```java
public enum Day {
    SUNDAY, MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY;
}
```

---

## 7. Record Classes (Java 16+)

A `record` is a special, concise syntax for declaring classes that are transparent carriers for immutable data.
-   **Behavior**: The compiler automatically generates the constructor, private final fields, `getters`, `equals()`, `hashCode()`, and `toString()` methods.
-   **Purpose**: To drastically reduce the boilerplate code needed for simple data-holding classes like DTOs (Data Transfer Objects).

```java
// A record class
public record Point(int x, int y) { }

// The above is roughly equivalent to a final class with:
// - private final int x;
// - private final int y;
// - a canonical constructor
// - getters x() and y()
// - equals(), hashCode(), and toString()
```
