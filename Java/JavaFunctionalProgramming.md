# A Guide to Functional Programming in Java

For much of its history, Java was a strictly Object-Oriented language. However, with the release of Java 8, a new paradigm was introduced: **Functional Programming**. This guide explores the core concepts of functional programming and how they are implemented in modern Java.

---

## 1. What is Functional Programming?

Functional Programming (FP) is a programming paradigm that treats computation as the evaluation of mathematical functions and avoids changing-state and mutable data. It's built on a few core principles:

-   **Pure Functions**: A pure function's return value is the same for the same arguments. It has no observable side effects, such as modifying a global variable or performing I/O.
-   **Immutability**: Data is immutable, meaning it cannot be changed after it's created. Instead of modifying existing data structures, you create new ones.
-   **First-Class Functions**: Functions are treated like any other variable. They can be passed as arguments to other functions, returned from functions, and stored in data structures.
-   **Higher-Order Functions**: These are functions that either take other functions as arguments or return them as results.

---

## 2. Key Functional Features in Java 8+

Java 8 introduced several features that form the foundation of functional programming in the language.

### a) Lambda Expressions

A lambda expression is a short, anonymous function that can be treated as a value. It allows you to write code in a more concise and functional style.

**Syntax:**
`(parameters) -> { body }`

-   If there is a single parameter, parentheses are optional: `param -> { body }`
-   If the body is a single expression, curly braces and the `return` keyword are optional: `(a, b) -> a + b`

**Example:**
```java
// Before Java 8 (using an anonymous class)
Runnable oldRunnable = new Runnable() {
    @Override
    public void run() {
        System.out.println("Running in the old way.");
    }
};

// With a lambda expression
Runnable newRunnable = () -> System.out.println("Running with a lambda!");
```

### b) Functional Interfaces

A functional interface is an interface that contains **exactly one abstract method**. It serves as the target type for a lambda expression. The `@FunctionalInterface` annotation can be used to ensure this contract is met at compile time.

Java provides a set of standard functional interfaces in the `java.util.function` package:
-   **`Predicate<T>`**: Takes an argument and returns a boolean. (e.g., `t -> t > 10`)
-   **`Function<T, R>`**: Takes an argument of type `T` and returns a result of type `R`. (e.g., `s -> s.length()`)
-   **`Consumer<T>`**: Takes an argument and performs an action but returns nothing (void). (e.g., `s -> System.out.println(s)`)
-   **`Supplier<T>`**: Takes no arguments but returns a value. (e.g., `() -> new ArrayList<>()`)

```java
import java.util.function.Predicate;

public class PredicateExample {
    public static void main(String[] args) {
        Predicate<String> isLongerThan5 = (s) -> s.length() > 5;

        System.out.println(isLongerThan5.test("hello"));    // false
        System.out.println(isLongerThan5.test("hello world")); // true
    }
}
```

---

## 3. The Stream API

The Stream API is arguably the most important functional feature added to Java. A **stream** is a sequence of elements from a source that supports aggregate operations.

**Key Characteristics of Streams:**
-   They don't store data. They operate on a data source, like a `Collection`.
-   They are immutable. Stream operations return a new stream, leaving the original source unchanged.
-   Operations are often **lazy**, meaning they are not executed until a terminal operation is invoked.

### Stream Pipeline

A stream pipeline consists of:
1.  A **source** (e.g., a `List` or an array).
2.  Zero or more **intermediate operations** (e.g., `filter`, `map`). These are lazy and return a new stream.
3.  A **terminal operation** (e.g., `forEach`, `collect`). This triggers the execution of the pipeline and produces a result.

### Common Stream Operations

-   **`filter(Predicate<T>)`**: Returns a stream consisting of the elements that match the given predicate.
-   **`map(Function<T, R>)`**: Returns a stream consisting of the results of applying the given function to the elements of this stream. (Transforms elements).
-   **`sorted()`**: Returns a stream with the elements sorted.
-   **`forEach(Consumer<T>)`**: Performs an action for each element of the stream. (Terminal operation).
-   **`collect(Collector)`**: Performs a mutable reduction operation on the elements of this stream. This is often used to put the results into a `List`, `Set`, or `Map`. (Terminal operation).
-   **`reduce()`**: Performs a reduction on the elements of the stream, using an associative accumulation function, and returns an `Optional` result. (Terminal operation).

### Example: A Complete Stream Pipeline

```java
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class StreamExample {
    public static void main(String[] args) {
        List<String> names = Arrays.asList("Alice", "Bob", "Charlie", "Anna", "Alex");

        // Find all names that start with 'A', convert them to uppercase, sort them,
        // and collect them into a new list.
        List<String> result = names.stream() // 1. Get stream from source
            .filter(name -> name.startsWith("A")) // 2. Intermediate op: filter
            .map(String::toUpperCase)             // 3. Intermediate op: transform
            .sorted()                             // 4. Intermediate op: sort
            .collect(Collectors.toList());        // 5. Terminal op: collect results

        System.out.println(result); // Output: [ALEX, ALICE, ANNA]
    }
}
```

---

## 4. Method References

A method reference is a compact, easy-to-read shorthand for a lambda expression that only calls an existing method.

There are four kinds of method references:
1.  **Reference to a static method**: `ClassName::staticMethodName`
2.  **Reference to an instance method of a particular object**: `object::instanceMethodName`
3.  **Reference to an instance method of an arbitrary object of a particular type**: `ClassName::instanceMethodName`
4.  **Reference to a constructor**: `ClassName::new`

**Example:**
```java
List<String> names = Arrays.asList("a", "b", "c");

// Using a lambda expression
names.forEach(s -> System.out.println(s));

// Using a method reference (more concise)
names.forEach(System.out::println);
```

---

## 5. The `Optional` Class

`Optional` is a container object which may or may not contain a non-null value. Its purpose is to provide a better way to handle `null` values, avoiding `NullPointerException` and creating more expressive APIs.

### Key Methods:
-   `Optional.of(value)`: Creates an `Optional` with a non-null value.
-   `Optional.ofNullable(value)`: Creates an `Optional` that can hold a null value.
-   `isPresent()`: Returns `true` if a value is present.
-   `ifPresent(Consumer)`: Executes the consumer if a value is present.
-   `orElse(defaultValue)`: Returns the value if present, otherwise returns a default value.
-   `orElseThrow(Supplier)`: Returns the value if present, otherwise throws an exception.

**Example:**
```java
import java.util.Optional;

public class OptionalExample {
    public static Optional<String> findUserById(int id) {
        if (id == 1) {
            return Optional.of("Alice");
        }
        return Optional.empty(); // Represents an absent value
    }

    public static void main(String[] args) {
        // Using Optional to avoid NullPointerException
        String userName = findUserById(2).orElse("Unknown User");
        System.out.println(userName); // Output: Unknown User

        // Using ifPresent for actions
        findUserById(1).ifPresent(name -> System.out.println("Found user: " + name));
    }
}
```
By embracing these features, developers can write more concise, readable, and maintainable Java code that is less prone to errors like null pointer exceptions.
