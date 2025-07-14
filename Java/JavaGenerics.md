# A Guide to Java Generics

Generics were introduced in Java 5 to provide **type safety** at compile time. They allow types (classes and interfaces) to be parameters when defining classes, interfaces, and methods. By using generics, you can create classes, interfaces, and methods that work with different data types while providing compile-time type checking.

---

## Why Use Generics?

1.  **Stronger Type Checks at Compile Time**: The Java compiler applies strong type checking to generic code and issues errors if the code violates type safety. Fixing compile-time errors is easier than fixing runtime errors, which can be difficult to find.
2.  **Elimination of Casts**: Before generics, you had to cast every object you read from a collection. This was verbose and error-prone. Generics eliminate the need for this.
3.  **Enabling Programmers to Implement Generic Algorithms**: By using generics, programmers can implement algorithms that work on collections of different types, can be customized, and are type-safe and easier to read.

### Example: Without vs. With Generics

**Without Generics (Pre-Java 5):**
```java
// No type safety - can add any object
List list = new ArrayList();
list.add("hello");
list.add(123); // This is a potential problem

// Requires an explicit cast
String s = (String) list.get(0);

// Throws ClassCastException at runtime
Integer i = (Integer) list.get(0); // Error!
```

**With Generics:**
```java
// Type safety - only Strings can be added
List<String> list = new ArrayList<>();
list.add("hello");
// list.add(123); // This line causes a COMPILE-TIME error!

// No cast needed
String s = list.get(0);
```

---

## Generic Classes

A generic class is defined with a type parameter section in angle brackets `<>`. This type parameter can then be used like a regular type within the class.

**Syntax**: `class name<T1, T2, ..., Tn> { /* ... */ }`

-   `T1, T2, ... Tn` are the type parameters.

### Example: A Generic `Box` Class

This class can be used to store an object of any type.

```java
public class Box<T> {
    // T stands for "Type"
    private T t;

    public void set(T t) {
        this.t = t;
    }

    public T get() {
        return t;
    }
}

// --- Usage ---
// Create a Box for Integers
Box<Integer> integerBox = new Box<>();
integerBox.set(10);
Integer someInteger = integerBox.get(); // No cast needed

// Create a Box for Strings
Box<String> stringBox = new Box<>();
stringBox.set("Hello World");
String someString = stringBox.get(); // No cast needed
```

---

## Generic Methods

You can also write generic methods that have their own type parameters. The type parameter's scope is limited to the method where it is declared.

**Syntax**: `<T> returnType methodName(T parameter)`

### Example: A Generic `printArray` Method

```java
public class GenericMethodExample {

    // A generic method
    public static <E> void printArray(E[] inputArray) {
        // Display array elements
        for (E element : inputArray) {
            System.out.printf("%s ", element);
        }
        System.out.println();
    }

    public static void main(String args[]) {
        // Create arrays of different types
        Integer[] intArray = { 1, 2, 3, 4, 5 };
        Double[] doubleArray = { 1.1, 2.2, 3.3, 4.4 };
        Character[] charArray = { 'H', 'E', 'L', 'L', 'O' };

        System.out.println("Array integerArray contains:");
        printArray(intArray);   // Pass an Integer array

        System.out.println("\nArray doubleArray contains:");
        printArray(doubleArray); // Pass a Double array

        System.out.println("\nArray characterArray contains:");
        printArray(charArray);  // Pass a Character array
    }
}
```

---

## Bounded Type Parameters

Sometimes you want to restrict the types that can be used as type arguments in a parameterized type. For example, a method that operates on numbers might only want to accept instances of `Number` or its subclasses. This is what bounded type parameters are for.

To declare a bounded type parameter, list the type parameter's name, followed by the `extends` keyword, followed by its upper bound.

**Syntax**: `<T extends UpperBound>`

### Example: A Method that Works on Numbers

```java
public class Stats<T extends Number> {
    private T[] nums; // Array of Number or its subclasses

    public Stats(T[] o) {
        nums = o;
    }

    // Return the double representation of the average
    public double average() {
        double sum = 0.0;
        for (int i = 0; i < nums.length; i++) {
            sum += nums[i].doubleValue(); // All Number objects have doubleValue()
        }
        return sum / nums.length;
    }
}

// --- Usage ---
Integer[] inums = { 1, 2, 3, 4, 5 };
Stats<Integer> iStats = new Stats<>(inums);
System.out.println("Average of integers: " + iStats.average());

Double[] dnums = { 1.1, 2.2, 3.3, 4.4, 5.5 };
Stats<Double> dStats = new Stats<>(dnums);
System.out.println("Average of doubles: " + dStats.average());

// This will not compile because String does not extend Number
// String[] snums = { "1", "2", "3" };
// Stats<String> sStats = new Stats<>(snums);
```

---

## Wildcards

In generic code, the question mark (`?`), called the **wildcard**, represents an unknown type. The wildcard can be used in a variety of situations: as the type of a parameter, field, or local variable; sometimes as a return type.

### 1. Upper Bounded Wildcards (`? extends Type`)

This wildcard restricts the unknown type to be a specific type or a subtype of that type. It is used when you want to read from a generic structure.

**Example**: A method that can take a list of any subclass of `Number`.
```java
public static void process(List<? extends Number> list) {
    for (Number n : list) {
        System.out.println(n);
    }
    // list.add(1); // Compile-time error: You can't add to a list with an extends wildcard.
}
```

### 2. Lower Bounded Wildcards (`? super Type`)

This wildcard restricts the unknown type to be a specific type or a supertype of that type. It is used when you want to add to a generic structure.

**Example**: A method that can add `Integer` objects to a list of `Integer` or a list of its supertypes (`Number`, `Object`).
```java
public static void addIntegers(List<? super Integer> list) {
    list.add(10);
    list.add(20);
    // Object o = list.get(0); // You can only read Objects safely.
}
```

### 3. Unbounded Wildcards (`?`)

This wildcard means that the code is written to work with any type. It's useful when the methods in your generic class don't depend on the type parameter.

**Example**: A method to print a list of any type.
```java
public static void printList(List<?> list) {
    for (Object elem : list) {
        System.out.print(elem + " ");
    }
    System.out.println();
}
```
---

## Type Erasure

Generics are implemented by the Java compiler as a front-end conversion called **type erasure**. In short, the compiler:
1.  Replaces all generic type parameters in generic types with their bounds or with `Object` if the type parameters are unbounded.
2.  Inserts type casts if necessary to preserve type safety.
3.  Generates bridge methods to preserve polymorphism in extended generic types.

This means that generic type information is not available at runtime. For example, `List<String>` and `List<Integer>` are both just `List` at runtime.
