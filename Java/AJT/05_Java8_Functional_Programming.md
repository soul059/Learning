# Java 8+ Features and Functional Programming

## 1. Lambda Expressions

### Introduction
Lambda expressions provide a way to represent a function as an object, enabling functional programming concepts in Java.

### Basic Syntax
```java
// Before Java 8 - Anonymous class
Runnable oldWay = new Runnable() {
    @Override
    public void run() {
        System.out.println("Running in old way");
    }
};

// Java 8 - Lambda expression
Runnable newWay = () -> System.out.println("Running with lambda");

// Lambda with parameters
Comparator<String> oldComparator = new Comparator<String>() {
    @Override
    public int compare(String s1, String s2) {
        return s1.compareTo(s2);
    }
};

Comparator<String> newComparator = (s1, s2) -> s1.compareTo(s2);
// Or even shorter with method reference
Comparator<String> shortest = String::compareTo;
```

### Lambda Syntax Variations
```java
public class LambdaSyntax {
    public static void main(String[] args) {
        // No parameters
        Runnable noParams = () -> System.out.println("No parameters");
        
        // Single parameter (parentheses optional)
        Consumer<String> singleParam = s -> System.out.println(s);
        Consumer<String> singleParamWithParens = (s) -> System.out.println(s);
        
        // Multiple parameters
        BiFunction<Integer, Integer, Integer> add = (a, b) -> a + b;
        
        // Block body
        BiFunction<Integer, Integer, Integer> complexOperation = (a, b) -> {
            int sum = a + b;
            int product = a * b;
            return sum > product ? sum : product;
        };
        
        // Type inference
        BiFunction<String, String, Integer> compare = (s1, s2) -> s1.compareTo(s2);
        
        // Explicit types (when needed)
        BiFunction<String, String, Integer> explicitTypes = 
            (String s1, String s2) -> s1.compareTo(s2);
    }
}
```

### Functional Interfaces
```java
// Custom functional interface
@FunctionalInterface
interface Calculator {
    int calculate(int a, int b);
    
    // Default methods are allowed
    default void printResult(int result) {
        System.out.println("Result: " + result);
    }
    
    // Static methods are allowed
    static Calculator getAdder() {
        return (a, b) -> a + b;
    }
}

public class FunctionalInterfaceExample {
    public static void main(String[] args) {
        // Using lambda with custom functional interface
        Calculator adder = (a, b) -> a + b;
        Calculator multiplier = (a, b) -> a * b;
        Calculator subtractor = (a, b) -> a - b;
        
        System.out.println("Addition: " + adder.calculate(10, 5));
        System.out.println("Multiplication: " + multiplier.calculate(10, 5));
        System.out.println("Subtraction: " + subtractor.calculate(10, 5));
        
        // Using static method
        Calculator staticAdder = Calculator.getAdder();
        staticAdder.printResult(staticAdder.calculate(20, 30));
    }
}
```

## 2. Built-in Functional Interfaces

### Common Functional Interfaces
```java
import java.util.function.*;

public class BuiltInFunctionalInterfaces {
    public static void main(String[] args) {
        // Predicate<T> - boolean test(T t)
        Predicate<String> isEmpty = String::isEmpty;
        Predicate<String> isLong = s -> s.length() > 10;
        Predicate<String> isEmptyOrLong = isEmpty.or(isLong);
        
        System.out.println("Is 'Hello' empty or long: " + isEmptyOrLong.test("Hello"));
        
        // Function<T, R> - R apply(T t)
        Function<String, Integer> stringLength = String::length;
        Function<Integer, String> intToString = Object::toString;
        Function<String, String> lengthAsString = stringLength.andThen(intToString);
        
        System.out.println("Length of 'Hello' as string: " + lengthAsString.apply("Hello"));
        
        // Consumer<T> - void accept(T t)
        Consumer<String> printer = System.out::println;
        Consumer<String> upperPrinter = s -> System.out.println(s.toUpperCase());
        Consumer<String> bothPrinters = printer.andThen(upperPrinter);
        
        bothPrinters.accept("Hello World");
        
        // Supplier<T> - T get()
        Supplier<String> stringSupplier = () -> "Generated String";
        Supplier<Double> randomSupplier = Math::random;
        
        System.out.println("Supplied string: " + stringSupplier.get());
        System.out.println("Random number: " + randomSupplier.get());
        
        // BiFunction<T, U, R> - R apply(T t, U u)
        BiFunction<String, String, String> concatenator = (s1, s2) -> s1 + s2;
        BiFunction<String, String, String> betterConcatenator = String::concat;
        
        System.out.println("Concatenated: " + concatenator.apply("Hello", " World"));
        
        // UnaryOperator<T> extends Function<T, T>
        UnaryOperator<String> toUpperCase = String::toUpperCase;
        UnaryOperator<Integer> square = x -> x * x;
        
        System.out.println("Upper case: " + toUpperCase.apply("hello"));
        System.out.println("Square of 5: " + square.apply(5));
        
        // BinaryOperator<T> extends BiFunction<T, T, T>
        BinaryOperator<Integer> adder = Integer::sum;
        BinaryOperator<String> stringAdder = (s1, s2) -> s1 + s2;
        
        System.out.println("Sum: " + adder.apply(10, 20));
    }
}
```

### Primitive Functional Interfaces
```java
import java.util.function.*;

public class PrimitiveFunctionalInterfaces {
    public static void main(String[] args) {
        // IntPredicate - avoids boxing/unboxing
        IntPredicate isEven = x -> x % 2 == 0;
        IntPredicate isPositive = x -> x > 0;
        IntPredicate isEvenAndPositive = isEven.and(isPositive);
        
        System.out.println("Is 4 even and positive: " + isEvenAndPositive.test(4));
        
        // IntFunction<R>
        IntFunction<String> intToString = Integer::toString;
        IntFunction<Double> intToDouble = x -> (double) x;
        
        // ToIntFunction<T>
        ToIntFunction<String> stringToLength = String::length;
        
        // IntSupplier
        IntSupplier randomInt = () -> (int) (Math.random() * 100);
        
        // IntConsumer
        IntConsumer intPrinter = System.out::println;
        
        // IntUnaryOperator
        IntUnaryOperator doubler = x -> x * 2;
        IntUnaryOperator squarer = x -> x * x;
        IntUnaryOperator doubleAndSquare = doubler.andThen(squarer);
        
        System.out.println("Double and square of 3: " + doubleAndSquare.applyAsInt(3));
        
        // IntBinaryOperator
        IntBinaryOperator intAdder = Integer::sum;
        IntBinaryOperator intMultiplier = (a, b) -> a * b;
        
        System.out.println("Sum: " + intAdder.applyAsInt(10, 20));
        System.out.println("Product: " + intMultiplier.applyAsInt(10, 20));
    }
}
```

## 3. Method References

### Types of Method References
```java
import java.util.*;
import java.util.function.*;

public class MethodReferences {
    public static void main(String[] args) {
        List<String> names = Arrays.asList("Alice", "Bob", "Charlie", "David");
        
        // 1. Static method reference
        // Lambda: s -> Integer.parseInt(s)
        Function<String, Integer> parser = Integer::parseInt;
        
        // 2. Instance method reference of a particular object
        String prefix = "Name: ";
        // Lambda: s -> prefix.concat(s)
        Function<String, String> prefixer = prefix::concat;
        
        // 3. Instance method reference of an arbitrary object
        // Lambda: s -> s.toUpperCase()
        Function<String, String> upperCaser = String::toUpperCase;
        
        // 4. Constructor reference
        // Lambda: () -> new ArrayList<>()
        Supplier<List<String>> listSupplier = ArrayList::new;
        // Lambda: capacity -> new ArrayList<>(capacity)
        IntFunction<List<String>> listWithCapacity = ArrayList::new;
        
        // Examples with collections
        names.stream()
             .map(String::toUpperCase)  // Method reference
             .forEach(System.out::println); // Method reference
        
        // Equivalent with lambdas
        names.stream()
             .map(s -> s.toUpperCase())
             .forEach(s -> System.out.println(s));
    }
}
```

### Constructor References
```java
import java.util.function.*;

class Person {
    private String name;
    private int age;
    
    public Person() {
        this("Unknown", 0);
    }
    
    public Person(String name) {
        this(name, 0);
    }
    
    public Person(String name, int age) {
        this.name = name;
        this.age = age;
    }
    
    @Override
    public String toString() {
        return "Person{name='" + name + "', age=" + age + '}';
    }
}

public class ConstructorReferences {
    public static void main(String[] args) {
        // Different constructor references
        Supplier<Person> defaultConstructor = Person::new;
        Function<String, Person> nameConstructor = Person::new;
        BiFunction<String, Integer, Person> fullConstructor = Person::new;
        
        Person p1 = defaultConstructor.get();
        Person p2 = nameConstructor.apply("Alice");
        Person p3 = fullConstructor.apply("Bob", 25);
        
        System.out.println(p1);
        System.out.println(p2);
        System.out.println(p3);
        
        // Array constructor reference
        IntFunction<String[]> stringArrayConstructor = String[]::new;
        String[] strings = stringArrayConstructor.apply(5);
        System.out.println("Array length: " + strings.length);
    }
}
```

## 4. Stream API

### Creating Streams
```java
import java.util.*;
import java.util.stream.*;

public class CreatingStreams {
    public static void main(String[] args) {
        // From collections
        List<String> list = Arrays.asList("a", "b", "c");
        Stream<String> streamFromList = list.stream();
        
        // From arrays
        String[] array = {"x", "y", "z"};
        Stream<String> streamFromArray = Arrays.stream(array);
        
        // From individual values
        Stream<String> streamFromValues = Stream.of("1", "2", "3");
        
        // Empty stream
        Stream<String> emptyStream = Stream.empty();
        
        // Infinite streams
        Stream<Integer> infiniteStream = Stream.iterate(0, n -> n + 2);
        Stream<Double> randomStream = Stream.generate(Math::random);
        
        // Range streams (IntStream, LongStream, DoubleStream)
        IntStream range = IntStream.range(1, 10); // 1 to 9
        IntStream rangeClosed = IntStream.rangeClosed(1, 10); // 1 to 10
        
        // From files
        try {
            Stream<String> lines = Files.lines(Paths.get("file.txt"));
        } catch (IOException e) {
            // Handle exception
        }
        
        // Stream builder
        Stream<String> builderStream = Stream.<String>builder()
            .add("a")
            .add("b")
            .add("c")
            .build();
        
        // Print first 10 even numbers
        Stream.iterate(0, n -> n + 2)
              .limit(10)
              .forEach(System.out::println);
    }
}
```

### Intermediate Operations
```java
import java.util.*;
import java.util.stream.*;

public class IntermediateOperations {
    public static void main(String[] args) {
        List<String> words = Arrays.asList("apple", "banana", "cherry", "date", "elderberry");
        
        // filter - keeps elements that match predicate
        words.stream()
             .filter(word -> word.length() > 5)
             .forEach(System.out::println);
        
        // map - transforms each element
        words.stream()
             .map(String::toUpperCase)
             .forEach(System.out::println);
        
        // flatMap - flattens nested structures
        List<List<String>> nestedList = Arrays.asList(
            Arrays.asList("a", "b"),
            Arrays.asList("c", "d"),
            Arrays.asList("e", "f")
        );
        
        nestedList.stream()
                  .flatMap(List::stream)
                  .forEach(System.out::println);
        
        // distinct - removes duplicates
        Arrays.asList("a", "b", "a", "c", "b")
              .stream()
              .distinct()
              .forEach(System.out::println);
        
        // sorted - sorts elements
        words.stream()
             .sorted()
             .forEach(System.out::println);
        
        words.stream()
             .sorted(Comparator.comparing(String::length))
             .forEach(System.out::println);
        
        // peek - performs action without changing stream (for debugging)
        words.stream()
             .filter(word -> word.startsWith("a"))
             .peek(word -> System.out.println("Filtered: " + word))
             .map(String::toUpperCase)
             .peek(word -> System.out.println("Mapped: " + word))
             .collect(Collectors.toList());
        
        // limit - limits stream size
        Stream.iterate(1, n -> n + 1)
              .limit(5)
              .forEach(System.out::println);
        
        // skip - skips first n elements
        words.stream()
             .skip(2)
             .forEach(System.out::println);
        
        // takeWhile (Java 9+) - takes elements while predicate is true
        Stream.of(1, 2, 3, 4, 5, 6)
              .takeWhile(n -> n < 4)
              .forEach(System.out::println); // 1, 2, 3
        
        // dropWhile (Java 9+) - drops elements while predicate is true
        Stream.of(1, 2, 3, 4, 5, 6)
              .dropWhile(n -> n < 4)
              .forEach(System.out::println); // 4, 5, 6
    }
}
```

### Terminal Operations
```java
import java.util.*;
import java.util.stream.*;

public class TerminalOperations {
    public static void main(String[] args) {
        List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        
        // forEach - performs action on each element
        numbers.stream()
               .filter(n -> n % 2 == 0)
               .forEach(System.out::println);
        
        // collect - collects elements into a collection
        List<Integer> evenNumbers = numbers.stream()
                                          .filter(n -> n % 2 == 0)
                                          .collect(Collectors.toList());
        
        Set<Integer> evenSet = numbers.stream()
                                     .filter(n -> n % 2 == 0)
                                     .collect(Collectors.toSet());
        
        String joinedNumbers = numbers.stream()
                                     .map(Object::toString)
                                     .collect(Collectors.joining(", "));
        
        // reduce - reduces stream to single value
        Optional<Integer> sum = numbers.stream()
                                      .reduce((a, b) -> a + b);
        
        Integer sumWithIdentity = numbers.stream()
                                        .reduce(0, (a, b) -> a + b);
        
        Integer product = numbers.stream()
                                .reduce(1, (a, b) -> a * b);
        
        // min/max
        Optional<Integer> min = numbers.stream().min(Integer::compareTo);
        Optional<Integer> max = numbers.stream().max(Integer::compareTo);
        
        // count
        long count = numbers.stream()
                           .filter(n -> n > 5)
                           .count();
        
        // anyMatch, allMatch, noneMatch
        boolean anyEven = numbers.stream().anyMatch(n -> n % 2 == 0);
        boolean allPositive = numbers.stream().allMatch(n -> n > 0);
        boolean noneNegative = numbers.stream().noneMatch(n -> n < 0);
        
        // findFirst, findAny
        Optional<Integer> first = numbers.stream()
                                        .filter(n -> n > 5)
                                        .findFirst();
        
        Optional<Integer> any = numbers.stream()
                                      .filter(n -> n > 5)
                                      .findAny();
        
        // toArray
        Integer[] array = numbers.stream()
                                .filter(n -> n % 2 == 0)
                                .toArray(Integer[]::new);
        
        System.out.println("Sum: " + sum.orElse(0));
        System.out.println("Min: " + min.orElse(0));
        System.out.println("Max: " + max.orElse(0));
        System.out.println("Count > 5: " + count);
        System.out.println("Any even: " + anyEven);
        System.out.println("All positive: " + allPositive);
        System.out.println("None negative: " + noneNegative);
        System.out.println("Joined: " + joinedNumbers);
    }
}
```

### Advanced Stream Operations
```java
import java.util.*;
import java.util.stream.*;

public class AdvancedStreamOperations {
    public static void main(String[] args) {
        // Grouping
        List<String> words = Arrays.asList("apple", "banana", "cherry", "avocado", "blueberry");
        
        Map<Character, List<String>> groupedByFirstLetter = words.stream()
            .collect(Collectors.groupingBy(word -> word.charAt(0)));
        
        System.out.println("Grouped by first letter: " + groupedByFirstLetter);
        
        // Partitioning
        Map<Boolean, List<String>> partitionedByLength = words.stream()
            .collect(Collectors.partitioningBy(word -> word.length() > 5));
        
        System.out.println("Partitioned by length > 5: " + partitionedByLength);
        
        // Collecting statistics
        List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        
        IntSummaryStatistics stats = numbers.stream()
            .mapToInt(Integer::intValue)
            .summaryStatistics();
        
        System.out.println("Statistics: " + stats);
        System.out.println("Average: " + stats.getAverage());
        System.out.println("Count: " + stats.getCount());
        System.out.println("Max: " + stats.getMax());
        System.out.println("Min: " + stats.getMin());
        System.out.println("Sum: " + stats.getSum());
        
        // Custom collectors
        String result = words.stream()
            .collect(Collector.of(
                StringBuilder::new,                    // supplier
                (sb, s) -> sb.append(s).append(", "), // accumulator
                StringBuilder::append,                 // combiner
                sb -> sb.toString()                   // finisher
            ));
        
        System.out.println("Custom collector result: " + result);
        
        // Parallel streams
        long parallelSum = numbers.parallelStream()
                                 .mapToLong(Integer::longValue)
                                 .sum();
        
        System.out.println("Parallel sum: " + parallelSum);
        
        // Complex example: Group people by age group
        List<Person> people = Arrays.asList(
            new Person("Alice", 25),
            new Person("Bob", 35),
            new Person("Charlie", 45),
            new Person("David", 15),
            new Person("Eve", 55)
        );
        
        Map<String, List<Person>> groupedByAgeCategory = people.stream()
            .collect(Collectors.groupingBy(person -> {
                if (person.getAge() < 18) return "Minor";
                else if (person.getAge() < 65) return "Adult";
                else return "Senior";
            }));
        
        System.out.println("Grouped by age category: " + groupedByAgeCategory);
    }
    
    static class Person {
        private String name;
        private int age;
        
        public Person(String name, int age) {
            this.name = name;
            this.age = age;
        }
        
        public String getName() { return name; }
        public int getAge() { return age; }
        
        @Override
        public String toString() {
            return name + "(" + age + ")";
        }
    }
}
```

## 5. Optional Class

### Basic Optional Usage
```java
import java.util.*;

public class OptionalBasics {
    public static void main(String[] args) {
        // Creating Optional objects
        Optional<String> empty = Optional.empty();
        Optional<String> nonEmpty = Optional.of("Hello");
        Optional<String> nullable = Optional.ofNullable(getString());
        
        // Checking if value is present
        if (nonEmpty.isPresent()) {
            System.out.println("Value: " + nonEmpty.get());
        }
        
        // Using ifPresent
        nonEmpty.ifPresent(System.out::println);
        
        // Using orElse and orElseGet
        String value1 = empty.orElse("Default Value");
        String value2 = empty.orElseGet(() -> "Generated Default");
        
        // Using orElseThrow
        try {
            String value3 = empty.orElseThrow(() -> new RuntimeException("No value present"));
        } catch (RuntimeException e) {
            System.out.println("Exception: " + e.getMessage());
        }
        
        // Chaining operations
        Optional<String> result = Optional.of("  Hello World  ")
            .filter(s -> !s.isEmpty())
            .map(String::trim)
            .map(String::toUpperCase);
        
        result.ifPresent(System.out::println);
        
        // flatMap for nested Optionals
        Optional<Optional<String>> nested = Optional.of(Optional.of("Nested"));
        Optional<String> flattened = nested.flatMap(opt -> opt);
        
        flattened.ifPresent(System.out::println);
    }
    
    private static String getString() {
        return Math.random() > 0.5 ? "Random String" : null;
    }
}
```

### Advanced Optional Patterns
```java
import java.util.*;

public class AdvancedOptional {
    
    // Repository pattern with Optional
    static class UserRepository {
        private Map<Long, User> users = new HashMap<>();
        
        public UserRepository() {
            users.put(1L, new User(1L, "Alice", "alice@example.com"));
            users.put(2L, new User(2L, "Bob", "bob@example.com"));
        }
        
        public Optional<User> findById(Long id) {
            return Optional.ofNullable(users.get(id));
        }
        
        public Optional<User> findByEmail(String email) {
            return users.values().stream()
                       .filter(user -> email.equals(user.getEmail()))
                       .findFirst();
        }
    }
    
    static class User {
        private Long id;
        private String name;
        private String email;
        
        public User(Long id, String name, String email) {
            this.id = id;
            this.name = name;
            this.email = email;
        }
        
        // Getters
        public Long getId() { return id; }
        public String getName() { return name; }
        public String getEmail() { return email; }
        
        @Override
        public String toString() {
            return "User{id=" + id + ", name='" + name + "', email='" + email + "'}";
        }
    }
    
    public static void main(String[] args) {
        UserRepository repository = new UserRepository();
        
        // Safe user retrieval
        Long userId = 1L;
        String userName = repository.findById(userId)
                                   .map(User::getName)
                                   .orElse("Unknown User");
        
        System.out.println("User name: " + userName);
        
        // Chaining multiple operations
        String userInfo = repository.findById(2L)
                                   .filter(user -> user.getEmail().contains("@"))
                                   .map(user -> user.getName() + " (" + user.getEmail() + ")")
                                   .orElse("Invalid user");
        
        System.out.println("User info: " + userInfo);
        
        // Optional chaining with flatMap
        Optional<String> emailDomain = repository.findById(1L)
                                                .map(User::getEmail)
                                                .map(email -> email.substring(email.indexOf('@') + 1));
        
        emailDomain.ifPresent(domain -> System.out.println("Email domain: " + domain));
        
        // Using Optional in streams
        List<Long> userIds = Arrays.asList(1L, 2L, 3L, 4L);
        
        List<User> validUsers = userIds.stream()
                                      .map(repository::findById)
                                      .filter(Optional::isPresent)
                                      .map(Optional::get)
                                      .collect(Collectors.toList());
        
        // Better approach with flatMap
        List<User> validUsers2 = userIds.stream()
                                       .map(repository::findById)
                                       .flatMap(Optional::stream) // Java 9+
                                       .collect(Collectors.toList());
        
        System.out.println("Valid users: " + validUsers);
    }
}
```

## 6. Default and Static Methods in Interfaces

### Default Methods
```java
interface Vehicle {
    // Abstract method
    void start();
    
    // Default method
    default void honk() {
        System.out.println("Beep beep!");
    }
    
    default void stop() {
        System.out.println("Vehicle stopped");
    }
    
    // Static method
    static void checkLicense() {
        System.out.println("License checked");
    }
}

interface ElectricVehicle extends Vehicle {
    void charge();
    
    // Override default method
    @Override
    default void stop() {
        System.out.println("Electric vehicle stopped silently");
    }
    
    // New default method
    default void displayBatteryLevel() {
        System.out.println("Battery level: 80%");
    }
}

class Car implements Vehicle {
    @Override
    public void start() {
        System.out.println("Car started with ignition");
    }
    
    // Can override default method
    @Override
    public void honk() {
        System.out.println("Car horn: HONK!");
    }
}

class Tesla implements ElectricVehicle {
    @Override
    public void start() {
        System.out.println("Tesla started silently");
    }
    
    @Override
    public void charge() {
        System.out.println("Tesla charging at supercharger");
    }
}

public class DefaultMethodsExample {
    public static void main(String[] args) {
        Car car = new Car();
        car.start();
        car.honk();
        car.stop(); // Uses default implementation
        
        Tesla tesla = new Tesla();
        tesla.start();
        tesla.honk(); // Uses inherited default implementation
        tesla.stop(); // Uses ElectricVehicle's override
        tesla.charge();
        tesla.displayBatteryLevel();
        
        // Static method call
        Vehicle.checkLicense();
    }
}
```

### Resolving Default Method Conflicts
```java
interface A {
    default void doSomething() {
        System.out.println("From A");
    }
}

interface B {
    default void doSomething() {
        System.out.println("From B");
    }
}

// Must resolve conflict
class C implements A, B {
    @Override
    public void doSomething() {
        // Option 1: Choose one
        A.super.doSomething();
        
        // Option 2: Call both
        // A.super.doSomething();
        // B.super.doSomething();
        
        // Option 3: Provide own implementation
        // System.out.println("From C");
    }
}

// Abstract class takes precedence over interface
abstract class AbstractClass {
    public void doSomething() {
        System.out.println("From AbstractClass");
    }
}

class D extends AbstractClass implements A {
    // No need to override - AbstractClass method takes precedence
}
```

## 7. Date and Time API (java.time)

### Basic Date and Time Classes
```java
import java.time.*;
import java.time.format.DateTimeFormatter;
import java.time.temporal.ChronoUnit;

public class DateTimeBasics {
    public static void main(String[] args) {
        // Current date and time
        LocalDate today = LocalDate.now();
        LocalTime now = LocalTime.now();
        LocalDateTime dateTime = LocalDateTime.now();
        ZonedDateTime zonedDateTime = ZonedDateTime.now();
        
        System.out.println("Today: " + today);
        System.out.println("Now: " + now);
        System.out.println("Date Time: " + dateTime);
        System.out.println("Zoned Date Time: " + zonedDateTime);
        
        // Creating specific dates and times
        LocalDate specificDate = LocalDate.of(2023, Month.DECEMBER, 25);
        LocalTime specificTime = LocalTime.of(14, 30, 0);
        LocalDateTime specificDateTime = LocalDateTime.of(2023, 12, 25, 14, 30, 0);
        
        // Parsing from strings
        LocalDate parsedDate = LocalDate.parse("2023-12-25");
        LocalTime parsedTime = LocalTime.parse("14:30:00");
        LocalDateTime parsedDateTime = LocalDateTime.parse("2023-12-25T14:30:00");
        
        // Formatting
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("dd/MM/yyyy");
        String formattedDate = today.format(formatter);
        System.out.println("Formatted date: " + formattedDate);
        
        // Arithmetic operations
        LocalDate tomorrow = today.plusDays(1);
        LocalDate nextWeek = today.plusWeeks(1);
        LocalDate nextMonth = today.plusMonths(1);
        LocalDate nextYear = today.plusYears(1);
        
        LocalDate yesterday = today.minusDays(1);
        
        // Duration and Period
        LocalDateTime start = LocalDateTime.of(2023, 1, 1, 10, 0);
        LocalDateTime end = LocalDateTime.of(2023, 1, 1, 15, 30);
        
        Duration duration = Duration.between(start, end);
        System.out.println("Duration: " + duration.toHours() + " hours");
        
        Period period = Period.between(LocalDate.of(2020, 1, 1), today);
        System.out.println("Period: " + period.getYears() + " years, " + 
                          period.getMonths() + " months, " + 
                          period.getDays() + " days");
        
        // Temporal adjusters
        LocalDate firstDayOfMonth = today.withDayOfMonth(1);
        LocalDate lastDayOfMonth = today.withDayOfMonth(today.lengthOfMonth());
        LocalDate firstDayOfYear = today.withDayOfYear(1);
        
        System.out.println("First day of month: " + firstDayOfMonth);
        System.out.println("Last day of month: " + lastDayOfMonth);
    }
}
```

### Working with Time Zones
```java
import java.time.*;
import java.time.zone.ZoneRules;

public class TimeZoneExample {
    public static void main(String[] args) {
        // Different ways to create ZonedDateTime
        ZonedDateTime nowInSystemZone = ZonedDateTime.now();
        ZonedDateTime nowInUTC = ZonedDateTime.now(ZoneOffset.UTC);
        ZonedDateTime nowInNY = ZonedDateTime.now(ZoneId.of("America/New_York"));
        ZonedDateTime nowInTokyo = ZonedDateTime.now(ZoneId.of("Asia/Tokyo"));
        
        System.out.println("System zone: " + nowInSystemZone);
        System.out.println("UTC: " + nowInUTC);
        System.out.println("New York: " + nowInNY);
        System.out.println("Tokyo: " + nowInTokyo);
        
        // Converting between time zones
        ZonedDateTime utcTime = ZonedDateTime.now(ZoneOffset.UTC);
        ZonedDateTime nyTime = utcTime.withZoneSameInstant(ZoneId.of("America/New_York"));
        ZonedDateTime tokyoTime = utcTime.withZoneSameInstant(ZoneId.of("Asia/Tokyo"));
        
        System.out.println("Same instant in different zones:");
        System.out.println("UTC: " + utcTime);
        System.out.println("NY: " + nyTime);
        System.out.println("Tokyo: " + tokyoTime);
        
        // Working with Instant (machine time)
        Instant instant = Instant.now();
        System.out.println("Instant: " + instant);
        
        // Convert between Instant and ZonedDateTime
        ZonedDateTime fromInstant = instant.atZone(ZoneId.systemDefault());
        Instant backToInstant = fromInstant.toInstant();
        
        // OffsetDateTime
        OffsetDateTime offsetDateTime = OffsetDateTime.now(ZoneOffset.of("+05:30"));
        System.out.println("Offset DateTime: " + offsetDateTime);
        
        // Zone information
        ZoneId zone = ZoneId.of("America/New_York");
        ZoneRules rules = zone.getRules();
        System.out.println("Is daylight savings: " + rules.isDaylightSavings(instant));
    }
}
```

## 8. CompletableFuture

### Basic CompletableFuture Usage
```java
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;

public class CompletableFutureBasics {
    public static void main(String[] args) throws ExecutionException, InterruptedException {
        // Simple async computation
        CompletableFuture<String> future = CompletableFuture.supplyAsync(() -> {
            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            return "Hello from async computation";
        });
        
        // Non-blocking - continues immediately
        System.out.println("This prints immediately");
        
        // Blocking - waits for result
        String result = future.get();
        System.out.println("Result: " + result);
        
        // Async with no return value
        CompletableFuture<Void> voidFuture = CompletableFuture.runAsync(() -> {
            System.out.println("Running async task");
        });
        
        voidFuture.get(); // Wait for completion
    }
}
```

### CompletableFuture Composition
```java
import java.util.concurrent.CompletableFuture;

public class CompletableFutureComposition {
    public static void main(String[] args) {
        // Sequential composition with thenApply
        CompletableFuture<String> future1 = CompletableFuture
            .supplyAsync(() -> "Hello")
            .thenApply(s -> s + " World")
            .thenApply(String::toUpperCase);
        
        future1.thenAccept(System.out::println);
        
        // Sequential composition with thenCompose (flatten nested futures)
        CompletableFuture<String> future2 = CompletableFuture
            .supplyAsync(() -> "Hello")
            .thenCompose(s -> CompletableFuture.supplyAsync(() -> s + " World"));
        
        future2.thenAccept(System.out::println);
        
        // Combining two independent futures
        CompletableFuture<String> greetingFuture = CompletableFuture.supplyAsync(() -> "Hello");
        CompletableFuture<String> nameFuture = CompletableFuture.supplyAsync(() -> "World");
        
        CompletableFuture<String> combinedFuture = greetingFuture
            .thenCombine(nameFuture, (greeting, name) -> greeting + " " + name);
        
        combinedFuture.thenAccept(System.out::println);
        
        // Running multiple futures and waiting for all
        CompletableFuture<String> future3 = CompletableFuture.supplyAsync(() -> "Task 1");
        CompletableFuture<String> future4 = CompletableFuture.supplyAsync(() -> "Task 2");
        CompletableFuture<String> future5 = CompletableFuture.supplyAsync(() -> "Task 3");
        
        CompletableFuture<Void> allFutures = CompletableFuture.allOf(future3, future4, future5);
        
        allFutures.thenRun(() -> {
            try {
                System.out.println("All tasks completed:");
                System.out.println(future3.get());
                System.out.println(future4.get());
                System.out.println(future5.get());
            } catch (Exception e) {
                e.printStackTrace();
            }
        });
        
        // Running multiple futures and getting first completed
        CompletableFuture<String> fastestFuture = CompletableFuture.anyOf(future3, future4, future5)
            .thenApply(result -> (String) result);
        
        fastestFuture.thenAccept(result -> System.out.println("Fastest result: " + result));
        
        // Exception handling
        CompletableFuture<String> futureWithException = CompletableFuture
            .supplyAsync(() -> {
                if (Math.random() > 0.5) {
                    throw new RuntimeException("Random error");
                }
                return "Success";
            })
            .handle((result, exception) -> {
                if (exception != null) {
                    return "Error: " + exception.getMessage();
                }
                return result;
            });
        
        futureWithException.thenAccept(System.out::println);
        
        // Wait for all examples to complete
        try {
            Thread.sleep(3000);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
}
```

## Summary

Java 8+ introduced significant functional programming features:

- **Lambda Expressions**: Concise way to write anonymous functions
- **Functional Interfaces**: Interfaces with single abstract method
- **Method References**: Shorthand for lambda expressions
- **Stream API**: Functional-style operations on collections
- **Optional**: Better null handling
- **Default Methods**: Interface evolution without breaking compatibility
- **New Date/Time API**: Comprehensive and thread-safe date/time handling
- **CompletableFuture**: Powerful asynchronous programming

These features enable:
- More concise and readable code
- Better support for parallel processing
- Functional programming paradigms
- Improved API design and evolution
- Better handling of common patterns (null checks, async operations)

Key principles:
- Immutability and side-effect-free functions
- Composition over inheritance
- Declarative over imperative style
- Lazy evaluation and efficient processing
- Better error handling and resource management
