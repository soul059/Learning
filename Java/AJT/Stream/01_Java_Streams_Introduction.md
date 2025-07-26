# Java Streams - Introduction and Fundamentals

## 1. What are Java Streams?

Java Streams, introduced in Java 8, provide a powerful and functional approach to processing collections of data. Streams represent a sequence of elements and support various operations to transform, filter, and process data in a declarative way.

### Key Characteristics
- **Functional Programming**: Operations are expressed as functions rather than imperative loops
- **Lazy Evaluation**: Operations are not executed until a terminal operation is called
- **Pipeline Processing**: Operations can be chained together to form processing pipelines
- **Immutable**: Original data source is not modified; operations create new streams
- **Parallel Processing**: Easy parallelization for performance improvements

### Stream vs Collections
| Aspect | Collections | Streams |
|--------|-------------|---------|
| **Storage** | Store elements | Process elements |
| **Computation** | Eager (immediate) | Lazy (on-demand) |
| **Iteration** | External (explicit loops) | Internal (hidden iteration) |
| **Reusability** | Multiple iterations | Single use |
| **Modification** | Mutable | Immutable operations |

## 2. Stream Creation

### From Collections
```java
import java.util.*;
import java.util.stream.*;

public class StreamCreationDemo {
    public static void main(String[] args) {
        // From List
        List<String> names = Arrays.asList("Alice", "Bob", "Charlie", "Diana");
        Stream<String> nameStream = names.stream();
        
        // From Set
        Set<Integer> numbers = Set.of(1, 2, 3, 4, 5);
        Stream<Integer> numberStream = numbers.stream();
        
        // From Map (keys or values)
        Map<String, Integer> ages = Map.of("Alice", 25, "Bob", 30, "Charlie", 35);
        Stream<String> keyStream = ages.keySet().stream();
        Stream<Integer> valueStream = ages.values().stream();
        
        // Print streams
        System.out.println("Names:");
        names.stream().forEach(System.out::println);
        
        System.out.println("\nNumbers:");
        numbers.stream().forEach(System.out::println);
        
        System.out.println("\nAge keys:");
        ages.keySet().stream().forEach(System.out::println);
    }
}
```

### From Arrays
```java
import java.util.Arrays;
import java.util.stream.Stream;

public class ArrayStreamDemo {
    public static void main(String[] args) {
        // From array using Arrays.stream()
        String[] fruits = {"apple", "banana", "cherry", "date"};
        Stream<String> fruitStream = Arrays.stream(fruits);
        
        // From array using Stream.of()
        Stream<String> fruitStream2 = Stream.of(fruits);
        
        // From primitive arrays
        int[] numbers = {1, 2, 3, 4, 5};
        IntStream intStream = Arrays.stream(numbers);
        
        // Print results
        System.out.println("Fruits from Arrays.stream():");
        Arrays.stream(fruits).forEach(System.out::println);
        
        System.out.println("\nNumbers from primitive array:");
        Arrays.stream(numbers).forEach(System.out::println);
        
        // Stream of individual elements
        Stream<String> colors = Stream.of("red", "green", "blue");
        System.out.println("\nColors:");
        colors.forEach(System.out::println);
    }
}
```

### Using Stream Builders and Generators
```java
import java.util.stream.Stream;
import java.util.Random;

public class StreamBuildersDemo {
    public static void main(String[] args) {
        // Using Stream.builder()
        Stream<String> builtStream = Stream.<String>builder()
            .add("first")
            .add("second")
            .add("third")
            .build();
        
        System.out.println("Built stream:");
        builtStream.forEach(System.out::println);
        
        // Empty stream
        Stream<String> emptyStream = Stream.empty();
        System.out.println("Empty stream count: " + emptyStream.count());
        
        // Infinite streams with generate()
        Stream<Double> randomNumbers = Stream.generate(Math::random)
            .limit(5); // Limit to avoid infinite execution
        
        System.out.println("\nRandom numbers:");
        randomNumbers.forEach(System.out::println);
        
        // Infinite streams with iterate()
        Stream<Integer> evenNumbers = Stream.iterate(0, n -> n + 2)
            .limit(10);
        
        System.out.println("\nFirst 10 even numbers:");
        evenNumbers.forEach(System.out::println);
        
        // Iterate with condition (Java 9+)
        Stream<Integer> numbersUpTo100 = Stream.iterate(1, n -> n <= 100, n -> n + 1);
        System.out.println("\nSum of numbers 1-100: " + 
            numbersUpTo100.mapToInt(Integer::intValue).sum());
    }
}
```

### Primitive Streams
```java
import java.util.stream.IntStream;
import java.util.stream.LongStream;
import java.util.stream.DoubleStream;

public class PrimitiveStreamsDemo {
    public static void main(String[] args) {
        // IntStream
        IntStream intRange = IntStream.range(1, 6); // 1 to 5
        System.out.println("IntStream range 1-5:");
        intRange.forEach(System.out::println);
        
        // IntStream inclusive
        IntStream intRangeInclusive = IntStream.rangeClosed(1, 5); // 1 to 5 inclusive
        System.out.println("\nIntStream rangeClosed 1-5:");
        intRangeInclusive.forEach(System.out::println);
        
        // LongStream
        LongStream longStream = LongStream.of(1000000L, 2000000L, 3000000L);
        System.out.println("\nLongStream:");
        longStream.forEach(System.out::println);
        
        // DoubleStream
        DoubleStream doubleStream = DoubleStream.of(1.1, 2.2, 3.3, 4.4);
        System.out.println("\nDoubleStream:");
        doubleStream.forEach(System.out::println);
        
        // Converting between streams
        IntStream numbers = IntStream.range(1, 6);
        DoubleStream doubles = numbers.asDoubleStream();
        System.out.println("\nConverted to DoubleStream:");
        doubles.forEach(System.out::println);
        
        // Specialized operations
        IntStream factorialBase = IntStream.rangeClosed(1, 5);
        int product = factorialBase.reduce(1, (a, b) -> a * b);
        System.out.println("\nFactorial of 5: " + product);
        
        // Statistical operations
        IntStream stats = IntStream.of(10, 20, 30, 40, 50);
        System.out.println("\nStatistics for [10,20,30,40,50]:");
        System.out.println("Sum: " + stats.sum());
        
        IntStream stats2 = IntStream.of(10, 20, 30, 40, 50);
        System.out.println("Average: " + stats2.average().orElse(0.0));
        
        IntStream stats3 = IntStream.of(10, 20, 30, 40, 50);
        System.out.println("Max: " + stats3.max().orElse(0));
        
        IntStream stats4 = IntStream.of(10, 20, 30, 40, 50);
        System.out.println("Min: " + stats4.min().orElse(0));
    }
}
```

## 3. Stream Operations Overview

Stream operations are divided into two categories:

### Intermediate Operations
- **Lazy**: Not executed until a terminal operation is invoked
- **Return Stream**: Can be chained together
- **Examples**: `filter()`, `map()`, `sorted()`, `distinct()`

### Terminal Operations
- **Eager**: Trigger the execution of the stream pipeline
- **Return Result**: Produce a final result or side effect
- **Examples**: `collect()`, `forEach()`, `reduce()`, `count()`

### Basic Operations Example
```java
import java.util.*;
import java.util.stream.Collectors;

public class BasicOperationsDemo {
    public static void main(String[] args) {
        List<String> words = Arrays.asList(
            "apple", "banana", "cherry", "date", "elderberry", "fig", "grape"
        );
        
        System.out.println("Original list: " + words);
        
        // Chain of intermediate operations with terminal operation
        List<String> result = words.stream()
            .filter(word -> word.length() > 4)      // Intermediate: filter
            .map(String::toUpperCase)               // Intermediate: transform
            .sorted()                               // Intermediate: sort
            .collect(Collectors.toList());          // Terminal: collect
        
        System.out.println("Filtered, uppercased, and sorted: " + result);
        
        // Another example with different operations
        long count = words.stream()
            .filter(word -> word.startsWith("a") || word.startsWith("e"))
            .count();                               // Terminal: count
        
        System.out.println("Words starting with 'a' or 'e': " + count);
        
        // Find first example
        Optional<String> firstLongWord = words.stream()
            .filter(word -> word.length() > 6)
            .findFirst();                           // Terminal: find first
        
        System.out.println("First word longer than 6 characters: " + 
            firstLongWord.orElse("None found"));
        
        // ForEach example (terminal operation)
        System.out.println("Words with their lengths:");
        words.stream()
            .forEach(word -> System.out.println(word + ": " + word.length()));
    }
}
```

## 4. Stream Processing Pipeline

### Understanding the Pipeline
```java
import java.util.*;
import java.util.stream.Collectors;

public class StreamPipelineDemo {
    public static void main(String[] args) {
        List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        
        System.out.println("Original numbers: " + numbers);
        
        // Complex pipeline example
        List<String> result = numbers.stream()
            // Step 1: Filter even numbers
            .filter(n -> {
                System.out.println("Filtering: " + n + " (even: " + (n % 2 == 0) + ")");
                return n % 2 == 0;
            })
            // Step 2: Square the numbers
            .map(n -> {
                int squared = n * n;
                System.out.println("Mapping: " + n + " -> " + squared);
                return squared;
            })
            // Step 3: Filter numbers greater than 10
            .filter(n -> {
                System.out.println("Second filter: " + n + " (>10: " + (n > 10) + ")");
                return n > 10;
            })
            // Step 4: Convert to String
            .map(n -> {
                String str = "Number: " + n;
                System.out.println("Converting to string: " + n + " -> " + str);
                return str;
            })
            // Step 5: Sort (strings are sorted lexicographically)
            .sorted()
            // Terminal operation: collect to list
            .collect(Collectors.toList());
        
        System.out.println("\nFinal result: " + result);
        
        // Demonstrating lazy evaluation
        System.out.println("\n=== Lazy Evaluation Demo ===");
        Stream<Integer> lazyStream = numbers.stream()
            .filter(n -> {
                System.out.println("This won't print until terminal operation");
                return n > 5;
            });
        
        System.out.println("Stream created but no output above because no terminal operation yet");
        
        // Now trigger with terminal operation
        long count = lazyStream.count();
        System.out.println("Count: " + count);
    }
}
```

### Performance Considerations
```java
import java.util.*;
import java.util.stream.Collectors;

public class StreamPerformanceDemo {
    public static void main(String[] args) {
        // Create a large dataset
        List<Integer> largeList = new ArrayList<>();
        for (int i = 1; i <= 1000000; i++) {
            largeList.add(i);
        }
        
        // Sequential processing
        long startTime = System.currentTimeMillis();
        List<Integer> sequentialResult = largeList.stream()
            .filter(n -> n % 2 == 0)
            .map(n -> n * n)
            .filter(n -> n > 1000)
            .collect(Collectors.toList());
        long sequentialTime = System.currentTimeMillis() - startTime;
        
        // Parallel processing
        startTime = System.currentTimeMillis();
        List<Integer> parallelResult = largeList.parallelStream()
            .filter(n -> n % 2 == 0)
            .map(n -> n * n)
            .filter(n -> n > 1000)
            .collect(Collectors.toList());
        long parallelTime = System.currentTimeMillis() - startTime;
        
        System.out.println("Sequential processing time: " + sequentialTime + "ms");
        System.out.println("Parallel processing time: " + parallelTime + "ms");
        System.out.println("Results equal: " + sequentialResult.equals(parallelResult));
        System.out.println("Result size: " + sequentialResult.size());
        
        // Short-circuiting operations
        startTime = System.currentTimeMillis();
        Optional<Integer> firstLarge = largeList.stream()
            .filter(n -> n > 500000)
            .findFirst();  // Short-circuits after finding first match
        long shortCircuitTime = System.currentTimeMillis() - startTime;
        
        System.out.println("Short-circuit time: " + shortCircuitTime + "ms");
        System.out.println("First large number: " + firstLarge.orElse(-1));
    }
}
```

## 5. Stream Lifecycle and Best Practices

### Stream Lifecycle
```java
import java.util.*;
import java.util.stream.Stream;

public class StreamLifecycleDemo {
    public static void main(String[] args) {
        List<String> words = Arrays.asList("hello", "world", "java", "streams");
        
        // 1. Create stream
        Stream<String> stream = words.stream();
        
        // 2. Add intermediate operations (lazy - not executed yet)
        Stream<String> processedStream = stream
            .filter(word -> word.length() > 4)
            .map(String::toUpperCase);
        
        // 3. Terminal operation (triggers execution)
        processedStream.forEach(System.out::println);
        
        // Stream is now consumed and cannot be reused
        try {
            processedStream.forEach(System.out::println); // This will throw exception
        } catch (IllegalStateException e) {
            System.out.println("Error: " + e.getMessage());
        }
        
        // Create new stream for reuse
        System.out.println("\nCreating new stream for reuse:");
        words.stream()
            .filter(word -> word.length() > 4)
            .map(String::toUpperCase)
            .forEach(System.out::println);
    }
}
```

### Best Practices
```java
import java.util.*;
import java.util.stream.Collectors;

public class StreamBestPracticesDemo {
    public static void main(String[] args) {
        List<Person> people = Arrays.asList(
            new Person("Alice", 25, "Engineer"),
            new Person("Bob", 30, "Manager"),
            new Person("Charlie", 35, "Engineer"),
            new Person("Diana", 28, "Designer"),
            new Person("Eve", 32, "Manager")
        );
        
        // 1. Prefer method references over lambdas when possible
        System.out.println("Names (using method reference):");
        people.stream()
            .map(Person::getName)  // Method reference instead of p -> p.getName()
            .forEach(System.out::println);
        
        // 2. Use appropriate terminal operations
        System.out.println("\nEngineers count: " + 
            people.stream()
                .filter(p -> "Engineer".equals(p.getJob()))
                .count());  // Use count() instead of collecting and getting size
        
        // 3. Use Optional properly
        Optional<Person> oldestPerson = people.stream()
            .max(Comparator.comparing(Person::getAge));
        
        oldestPerson.ifPresent(p -> 
            System.out.println("Oldest person: " + p.getName()));
        
        // 4. Avoid side effects in lambda expressions
        // BAD: Don't modify external state
        List<String> sideEffectList = new ArrayList<>();
        people.stream()
            .filter(p -> p.getAge() > 30)
            .forEach(p -> sideEffectList.add(p.getName())); // Side effect!
        
        // GOOD: Use collectors instead
        List<String> names = people.stream()
            .filter(p -> p.getAge() > 30)
            .map(Person::getName)
            .collect(Collectors.toList());
        
        System.out.println("People over 30: " + names);
        
        // 5. Consider parallel streams for large datasets with CPU-intensive operations
        List<Integer> largeList = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        
        // For small datasets, sequential is often faster
        int sequentialSum = largeList.stream()
            .mapToInt(Integer::intValue)
            .sum();
        
        // For large datasets with heavy computation, consider parallel
        int parallelSum = largeList.parallelStream()
            .mapToInt(Integer::intValue)
            .sum();
        
        System.out.println("Sequential sum: " + sequentialSum);
        System.out.println("Parallel sum: " + parallelSum);
    }
    
    static class Person {
        private String name;
        private int age;
        private String job;
        
        public Person(String name, int age, String job) {
            this.name = name;
            this.age = age;
            this.job = job;
        }
        
        public String getName() { return name; }
        public int getAge() { return age; }
        public String getJob() { return job; }
        
        @Override
        public String toString() {
            return name + "(" + age + ", " + job + ")";
        }
    }
}
```

## Summary

### Key Takeaways

1. **Functional Programming**: Streams enable declarative programming style
2. **Lazy Evaluation**: Operations are deferred until terminal operation
3. **Pipeline Processing**: Chain operations together for complex transformations
4. **Immutability**: Original data remains unchanged
5. **Performance**: Consider parallel streams for large datasets

### When to Use Streams

**Use Streams when:**
- Processing collections with multiple transformations
- Need functional programming style
- Want to leverage parallel processing
- Working with data pipelines

**Avoid Streams when:**
- Simple single operations (use traditional loops)
- Heavy side effects are needed
- Performance is critical and overhead matters
- Working with small datasets where traditional loops are clearer

### Common Patterns

- **Filter-Map-Collect**: Most common pattern for data transformation
- **Reduce**: For aggregation operations
- **Group By**: For categorizing data
- **Find First/Any**: For searching operations
- **Parallel Processing**: For performance-critical operations

Streams provide a powerful and expressive way to work with data in Java, making code more readable and maintainable while offering opportunities for performance optimization through parallelization.
