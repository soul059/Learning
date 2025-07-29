# Java Method References - Complete Guide

## Table of Contents
1. [Introduction to Method References](#introduction-to-method-references)
2. [Types of Method References](#types-of-method-references)
3. [Static Method References](#static-method-references)
4. [Instance Method References](#instance-method-references)
5. [Constructor References](#constructor-references)
6. [Array Constructor References](#array-constructor-references)
7. [Method References vs Lambda Expressions](#method-references-vs-lambda-expressions)
8. [Advanced Examples and Use Cases](#advanced-examples-and-use-cases)
9. [Best Practices](#best-practices)
10. [Common Pitfalls](#common-pitfalls)

## 1. Introduction to Method References

Method references provide a way to refer to methods without invoking them. They are a shorthand notation for lambda expressions that call a single method. Method references use the `::` (double colon) operator to separate the class or instance from the method name.

### Why Use Method References?
- **Readability**: More concise and readable than lambda expressions
- **Reusability**: Can reference existing methods without rewriting logic
- **Performance**: Slightly better performance in some cases
- **Maintainability**: Changes to the referenced method automatically apply

### Basic Syntax
```java
// Lambda expression
Function<String, Integer> lambda = s -> Integer.parseInt(s);

// Method reference
Function<String, Integer> methodRef = Integer::parseInt;
```

## 2. Types of Method References

Java supports four types of method references:

1. **Static Method References**: `ClassName::staticMethodName`
2. **Instance Method References of a Particular Object**: `instance::instanceMethodName`
3. **Instance Method References of an Arbitrary Object**: `ClassName::instanceMethodName`
4. **Constructor References**: `ClassName::new`

```java
import java.util.*;
import java.util.function.*;

public class MethodReferenceTypes {
    public static void main(String[] args) {
        // 1. Static method reference
        Function<String, Integer> parseInt = Integer::parseInt;
        
        // 2. Instance method reference of particular object
        String prefix = "Hello ";
        UnaryOperator<String> prefixer = prefix::concat;
        
        // 3. Instance method reference of arbitrary object
        Function<String, String> toUpper = String::toUpperCase;
        
        // 4. Constructor reference
        Supplier<List<String>> listSupplier = ArrayList::new;
        
        // Examples
        System.out.println(parseInt.apply("123"));           // 123
        System.out.println(prefixer.apply("World"));         // Hello World
        System.out.println(toUpper.apply("java"));           // JAVA
        System.out.println(listSupplier.get());              // []
    }
}
```

## 3. Static Method References

Static method references refer to static methods of a class using the syntax `ClassName::staticMethodName`.

### Basic Static Method References
```java
import java.util.function.*;
import java.util.Arrays;
import java.util.List;

public class StaticMethodReferences {
    
    // Custom static methods
    public static String formatName(String name) {
        return "Mr/Ms. " + name.toUpperCase();
    }
    
    public static boolean isEven(Integer number) {
        return number % 2 == 0;
    }
    
    public static int multiply(int a, int b) {
        return a * b;
    }
    
    public static void main(String[] args) {
        List<String> names = Arrays.asList("alice", "bob", "charlie");
        List<String> numbers = Arrays.asList("1", "2", "3", "4", "5");
        List<Integer> integers = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        
        // Built-in static methods
        // Integer::parseInt
        numbers.stream()
               .map(Integer::parseInt)
               .forEach(System.out::println);
        
        // Math::abs
        List<Integer> negatives = Arrays.asList(-1, -2, 3, -4, 5);
        negatives.stream()
                 .map(Math::abs)
                 .forEach(System.out::println);
        
        // String::valueOf
        integers.stream()
                .map(String::valueOf)
                .forEach(System.out::println);
        
        // Custom static methods
        // StaticMethodReferences::formatName
        names.stream()
             .map(StaticMethodReferences::formatName)
             .forEach(System.out::println);
        
        // StaticMethodReferences::isEven
        integers.stream()
                .filter(StaticMethodReferences::isEven)
                .forEach(System.out::println);
        
        // With BiFunction for methods with two parameters
        BiFunction<Integer, Integer, Integer> multiplier = StaticMethodReferences::multiply;
        System.out.println(multiplier.apply(5, 3)); // 15
    }
}
```

### Common Static Method References
```java
import java.util.*;
import java.util.function.*;
import java.util.stream.Collectors;

public class CommonStaticMethodReferences {
    public static void main(String[] args) {
        List<String> stringNumbers = Arrays.asList("10", "20", "30", "40");
        List<Double> doubleNumbers = Arrays.asList(1.5, 2.7, 3.9, 4.1);
        
        // Parsing methods
        Function<String, Integer> parseInt = Integer::parseInt;
        Function<String, Double> parseDouble = Double::parseDouble;
        Function<String, Boolean> parseBoolean = Boolean::parseBoolean;
        
        // Math operations
        Function<Double, Double> sqrt = Math::sqrt;
        Function<Double, Double> abs = Math::abs;
        Function<Double, Long> round = Math::round;
        
        // String operations
        Function<Object, String> toString = String::valueOf;
        
        // Collections operations
        BinaryOperator<Integer> max = Integer::max;
        BinaryOperator<Integer> min = Integer::min;
        BinaryOperator<Integer> sum = Integer::sum;
        
        // Examples
        List<Integer> parsedInts = stringNumbers.stream()
                                               .map(parseInt)
                                               .collect(Collectors.toList());
        System.out.println("Parsed integers: " + parsedInts);
        
        List<Long> roundedNumbers = doubleNumbers.stream()
                                                .map(round)
                                                .collect(Collectors.toList());
        System.out.println("Rounded numbers: " + roundedNumbers);
        
        Optional<Integer> maxValue = parsedInts.stream()
                                              .reduce(max);
        System.out.println("Max value: " + maxValue.orElse(0));
    }
}
```

## 4. Instance Method References

### Instance Method References of a Particular Object

These references call an instance method on a specific object instance.

```java
import java.util.*;
import java.util.function.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

public class ParticularObjectMethodReferences {
    
    public static void main(String[] args) {
        List<String> words = Arrays.asList("hello", "world", "java", "programming");
        
        // String instance methods
        String prefix = ">>> ";
        String suffix = " <<<";
        
        // UnaryOperator for single parameter, same return type
        UnaryOperator<String> addPrefix = prefix::concat;
        Function<String, String> addSuffix = suffix::concat;
        
        // Predicate for boolean returning methods
        String searchTerm = "java";
        Predicate<String> containsJava = searchTerm::equals;
        
        // Examples
        words.stream()
             .map(addPrefix)
             .map(word -> word + suffix) // Could also use suffix::concat in a different way
             .forEach(System.out::println);
        
        // Using object methods
        StringBuilder sb = new StringBuilder();
        Consumer<String> appender = sb::append;
        
        words.forEach(appender);
        System.out.println("StringBuilder result: " + sb.toString());
        
        // Date formatting example
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
        Function<LocalDateTime, String> dateFormatter = formatter::format;
        
        LocalDateTime now = LocalDateTime.now();
        System.out.println("Formatted date: " + dateFormatter.apply(now));
        
        // Collection methods
        List<String> targetList = new ArrayList<>();
        Consumer<String> addToList = targetList::add;
        
        words.stream()
             .filter(word -> word.length() > 4)
             .forEach(addToList);
        
        System.out.println("Filtered words: " + targetList);
    }
}
```

### Instance Method References of an Arbitrary Object

These references call an instance method on an arbitrary object of a specific type that will be provided later.

```java
import java.util.*;
import java.util.function.*;
import java.util.stream.Collectors;

public class ArbitraryObjectMethodReferences {
    
    public static void main(String[] args) {
        List<String> words = Arrays.asList("Hello", "World", "Java", "Programming", "Method", "References");
        List<Person> people = Arrays.asList(
            new Person("Alice", 25),
            new Person("Bob", 30),
            new Person("Charlie", 35)
        );
        
        // String methods on arbitrary String objects
        Function<String, String> toUpperCase = String::toUpperCase;
        Function<String, String> toLowerCase = String::toLowerCase;
        Function<String, Integer> getLength = String::length;
        Predicate<String> isEmpty = String::isEmpty;
        Function<String, String> trim = String::trim;
        
        // Examples with String methods
        List<String> upperCaseWords = words.stream()
                                          .map(toUpperCase)
                                          .collect(Collectors.toList());
        System.out.println("Uppercase: " + upperCaseWords);
        
        List<Integer> wordLengths = words.stream()
                                        .map(getLength)
                                        .collect(Collectors.toList());
        System.out.println("Word lengths: " + wordLengths);
        
        // Sorting with method references
        List<String> sortedByLength = words.stream()
                                          .sorted(Comparator.comparing(String::length))
                                          .collect(Collectors.toList());
        System.out.println("Sorted by length: " + sortedByLength);
        
        // Using with custom objects
        Function<Person, String> getName = Person::getName;
        Function<Person, Integer> getAge = Person::getAge;
        
        List<String> names = people.stream()
                                  .map(getName)
                                  .collect(Collectors.toList());
        System.out.println("Names: " + names);
        
        // Sorting people by age
        List<Person> sortedByAge = people.stream()
                                        .sorted(Comparator.comparing(Person::getAge))
                                        .collect(Collectors.toList());
        System.out.println("Sorted by age: " + sortedByAge);
        
        // Chaining comparisons
        List<Person> sortedByAgeAndName = people.stream()
                                               .sorted(Comparator.comparing(Person::getAge)
                                                       .thenComparing(Person::getName))
                                               .collect(Collectors.toList());
        System.out.println("Sorted by age and name: " + sortedByAgeAndName);
        
        // Method references with different functional interfaces
        ToIntFunction<String> stringToLength = String::length;
        ToDoubleFunction<Person> personToAge = person -> person.getAge().doubleValue();
        
        OptionalInt maxLength = words.stream()
                                    .mapToInt(stringToLength)
                                    .max();
        System.out.println("Max word length: " + maxLength.orElse(0));
    }
}

class Person {
    private String name;
    private Integer age;
    
    public Person(String name, Integer age) {
        this.name = name;
        this.age = age;
    }
    
    public String getName() {
        return name;
    }
    
    public Integer getAge() {
        return age;
    }
    
    @Override
    public String toString() {
        return "Person{name='" + name + "', age=" + age + '}';
    }
}
```

## 5. Constructor References

Constructor references allow you to reference constructors using the `::new` syntax.

### Basic Constructor References
```java
import java.util.*;
import java.util.function.*;
import java.util.stream.Collectors;

public class ConstructorReferences {
    
    public static void main(String[] args) {
        // Simple constructor references
        Supplier<ArrayList<String>> listSupplier = ArrayList::new;
        Supplier<HashMap<String, Integer>> mapSupplier = HashMap::new;
        Supplier<StringBuilder> stringBuilderSupplier = StringBuilder::new;
        
        // Constructor with parameters
        Function<String, StringBuilder> stringBuilderWithContent = StringBuilder::new;
        IntFunction<ArrayList<String>> listWithCapacity = ArrayList::new;
        BiFunction<String, Integer, Person> personConstructor = Person::new;
        
        // Examples
        List<String> newList = listSupplier.get();
        Map<String, Integer> newMap = mapSupplier.get();
        StringBuilder sb1 = stringBuilderSupplier.get();
        StringBuilder sb2 = stringBuilderWithContent.apply("Initial content");
        List<String> listWithSize = listWithCapacity.apply(10);
        Person person = personConstructor.apply("John", 25);
        
        System.out.println("New list: " + newList);
        System.out.println("StringBuilder with content: " + sb2.toString());
        System.out.println("List capacity (size): " + listWithSize.size());
        System.out.println("Person: " + person);
        
        // Using constructor references in streams
        List<String> names = Arrays.asList("Alice", "Bob", "Charlie");
        List<Integer> ages = Arrays.asList(25, 30, 35);
        
        // Creating objects from stream data
        List<Person> people = names.stream()
                                  .map(name -> new Person(name, 0)) // Lambda
                                  .collect(Collectors.toList());
        
        // Better approach with zip-like operation (Java doesn't have built-in zip)
        List<Person> peopleWithAges = IntStream.range(0, Math.min(names.size(), ages.size()))
                                              .mapToObj(i -> new Person(names.get(i), ages.get(i)))
                                              .collect(Collectors.toList());
        
        System.out.println("People: " + people);
        System.out.println("People with ages: " + peopleWithAges);
    }
}
```

### Advanced Constructor References
```java
import java.util.*;
import java.util.function.*;
import java.util.stream.Collectors;

// Class with multiple constructors
class Employee {
    private String name;
    private String department;
    private double salary;
    private int experience;
    
    // No-arg constructor
    public Employee() {
        this("Unknown", "General", 0.0, 0);
    }
    
    // Name only constructor
    public Employee(String name) {
        this(name, "General", 0.0, 0);
    }
    
    // Name and department constructor
    public Employee(String name, String department) {
        this(name, department, 0.0, 0);
    }
    
    // Full constructor
    public Employee(String name, String department, double salary, int experience) {
        this.name = name;
        this.department = department;
        this.salary = salary;
        this.experience = experience;
    }
    
    // Getters
    public String getName() { return name; }
    public String getDepartment() { return department; }
    public double getSalary() { return salary; }
    public int getExperience() { return experience; }
    
    @Override
    public String toString() {
        return String.format("Employee{name='%s', dept='%s', salary=%.2f, exp=%d}", 
                           name, department, salary, experience);
    }
}

public class AdvancedConstructorReferences {
    
    public static void main(String[] args) {
        // Different constructor references for different functional interfaces
        Supplier<Employee> defaultEmployee = Employee::new;
        Function<String, Employee> employeeWithName = Employee::new;
        BiFunction<String, String, Employee> employeeWithDept = Employee::new;
        
        // Custom functional interface for 4 parameters
        @FunctionalInterface
        interface QuadFunction<T, U, V, W, R> {
            R apply(T t, U u, V v, W w);
        }
        
        QuadFunction<String, String, Double, Integer, Employee> fullEmployee = Employee::new;
        
        // Examples
        Employee emp1 = defaultEmployee.get();
        Employee emp2 = employeeWithName.apply("John");
        Employee emp3 = employeeWithDept.apply("Jane", "Engineering");
        Employee emp4 = fullEmployee.apply("Bob", "Sales", 50000.0, 5);
        
        System.out.println("Default: " + emp1);
        System.out.println("With name: " + emp2);
        System.out.println("With dept: " + emp3);
        System.out.println("Full: " + emp4);
        
        // Using in collections
        List<String> names = Arrays.asList("Alice", "Bob", "Charlie", "Diana");
        List<String> departments = Arrays.asList("IT", "HR", "Finance", "Marketing");
        
        // Creating employees from names
        List<Employee> employees = names.stream()
                                       .map(Employee::new)
                                       .collect(Collectors.toList());
        
        System.out.println("\nEmployees from names:");
        employees.forEach(System.out::println);
        
        // Creating employees with names and departments
        List<Employee> employeesWithDept = IntStream.range(0, Math.min(names.size(), departments.size()))
                                                   .mapToObj(i -> new Employee(names.get(i), departments.get(i)))
                                                   .collect(Collectors.toList());
        
        System.out.println("\nEmployees with departments:");
        employeesWithDept.forEach(System.out::println);
    }
}
```

## 6. Array Constructor References

Array constructor references are used to create arrays using method references.

```java
import java.util.*;
import java.util.function.*;
import java.util.stream.Stream;

public class ArrayConstructorReferences {
    
    public static void main(String[] args) {
        // Basic array constructor references
        IntFunction<String[]> stringArrayConstructor = String[]::new;
        IntFunction<Integer[]> integerArrayConstructor = Integer[]::new;
        IntFunction<Person[]> personArrayConstructor = Person[]::new;
        
        // Creating arrays
        String[] stringArray = stringArrayConstructor.apply(5);
        Integer[] integerArray = integerArrayConstructor.apply(10);
        Person[] personArray = personArrayConstructor.apply(3);
        
        System.out.println("String array length: " + stringArray.length);
        System.out.println("Integer array length: " + integerArray.length);
        System.out.println("Person array length: " + personArray.length);
        
        // Using with streams to collect to arrays
        List<String> words = Arrays.asList("hello", "world", "java", "programming");
        
        // Convert stream to array using constructor reference
        String[] wordsArray = words.stream()
                                   .map(String::toUpperCase)
                                   .toArray(String[]::new);
        
        System.out.println("Words array: " + Arrays.toString(wordsArray));
        
        // Creating and populating arrays
        Integer[] numbers = Stream.iterate(1, n -> n + 1)
                                 .limit(10)
                                 .toArray(Integer[]::new);
        
        System.out.println("Numbers array: " + Arrays.toString(numbers));
        
        // Multi-dimensional arrays
        IntFunction<String[][]> string2DArrayConstructor = String[][]::new;
        String[][] matrix = string2DArrayConstructor.apply(3);
        
        // Initialize the matrix
        for (int i = 0; i < matrix.length; i++) {
            matrix[i] = new String[3];
            Arrays.fill(matrix[i], "cell_" + i);
        }
        
        System.out.println("Matrix:");
        for (String[] row : matrix) {
            System.out.println(Arrays.toString(row));
        }
        
        // Custom object arrays from streams
        List<String> names = Arrays.asList("Alice", "Bob", "Charlie");
        Person[] people = names.stream()
                              .map(name -> new Person(name, 25))
                              .toArray(Person[]::new);
        
        System.out.println("People array: " + Arrays.toString(people));
        
        // Parallel processing with arrays
        String[] parallelResult = words.parallelStream()
                                      .map(String::toUpperCase)
                                      .sorted()
                                      .toArray(String[]::new);
        
        System.out.println("Parallel result: " + Arrays.toString(parallelResult));
    }
}
```

## 7. Method References vs Lambda Expressions

Understanding when to use method references versus lambda expressions is crucial for writing clean, maintainable code.

```java
import java.util.*;
import java.util.function.*;
import java.util.stream.Collectors;

public class MethodReferencesVsLambdas {
    
    public static void main(String[] args) {
        List<String> words = Arrays.asList("apple", "banana", "cherry", "date");
        List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        
        System.out.println("=== When to use Method References ===");
        
        // 1. When lambda just calls a single method
        // Good: Method reference
        words.stream()
             .map(String::toUpperCase)
             .forEach(System.out::println);
        
        // Less preferred: Lambda
        words.stream()
             .map(s -> s.toUpperCase())
             .forEach(s -> System.out.println(s));
        
        // 2. When using existing static methods
        // Good: Method reference
        List<String> stringNumbers = Arrays.asList("1", "2", "3");
        List<Integer> parsed = stringNumbers.stream()
                                           .map(Integer::parseInt)
                                           .collect(Collectors.toList());
        
        // Less preferred: Lambda
        List<Integer> parsedLambda = stringNumbers.stream()
                                                 .map(s -> Integer.parseInt(s))
                                                 .collect(Collectors.toList());
        
        System.out.println("\n=== When to use Lambda Expressions ===");
        
        // 1. When you need to perform multiple operations
        // Good: Lambda
        List<String> processed = words.stream()
                                     .map(s -> s.toUpperCase().trim().substring(0, Math.min(3, s.length())))
                                     .collect(Collectors.toList());
        System.out.println("Processed: " + processed);
        
        // 2. When you need to add logic
        // Good: Lambda
        List<Integer> evenSquares = numbers.stream()
                                          .filter(n -> n % 2 == 0)
                                          .map(n -> n * n)
                                          .collect(Collectors.toList());
        System.out.println("Even squares: " + evenSquares);
        
        // 3. When capturing local variables
        String prefix = "Item: ";
        List<String> prefixed = words.stream()
                                    .map(word -> prefix + word) // Captures local variable
                                    .collect(Collectors.toList());
        System.out.println("Prefixed: " + prefixed);
        
        System.out.println("\n=== Readability Comparison ===");
        
        // Method references are more readable for simple operations
        Comparator<String> byLength1 = Comparator.comparing(String::length);
        Comparator<String> byLength2 = Comparator.comparing(s -> s.length());
        
        // Lambda expressions are more readable for complex operations
        Comparator<String> complex1 = (s1, s2) -> {
            int lengthCompare = Integer.compare(s1.length(), s2.length());
            return lengthCompare != 0 ? lengthCompare : s1.compareTo(s2);
        };
        
        // This would be harder to read as method references
        Predicate<String> complexPredicate = s -> s.length() > 3 && 
                                                 s.startsWith("a") && 
                                                 !s.contains("e");
        
        System.out.println("\n=== Performance Considerations ===");
        
        // Both are generally equivalent in performance
        // Method references might have slight advantage in some JVM implementations
        
        long start = System.nanoTime();
        for (int i = 0; i < 1000000; i++) {
            words.stream().map(String::toUpperCase).count();
        }
        long methodRefTime = System.nanoTime() - start;
        
        start = System.nanoTime();
        for (int i = 0; i < 1000000; i++) {
            words.stream().map(s -> s.toUpperCase()).count();
        }
        long lambdaTime = System.nanoTime() - start;
        
        System.out.println("Method reference time: " + methodRefTime / 1000000 + "ms");
        System.out.println("Lambda time: " + lambdaTime / 1000000 + "ms");
    }
}
```

### Guidelines for Choosing Between Method References and Lambdas

```java
import java.util.*;
import java.util.function.*;

public class ChoiceGuidelines {
    
    // Custom methods for examples
    public static String formatString(String s) {
        return "[" + s.toUpperCase() + "]";
    }
    
    public static boolean isLongWord(String word) {
        return word.length() > 5;
    }
    
    public static void main(String[] args) {
        List<String> words = Arrays.asList("java", "programming", "method", "reference", "lambda");
        
        // ✅ Use Method References When:
        
        // 1. Calling a single method with no additional logic
        words.stream()
             .map(String::toUpperCase)    // ✅ Good
             .forEach(System.out::println); // ✅ Good
        
        // 2. Using static methods
        List<String> numbers = Arrays.asList("1", "2", "3");
        numbers.stream()
               .map(Integer::parseInt)     // ✅ Good
               .forEach(System.out::println);
        
        // 3. Using constructors
        Function<String, StringBuilder> sbCreator = StringBuilder::new; // ✅ Good
        
        // 4. Using methods on specific instances
        String prefix = ">> ";
        Function<String, String> prefixer = prefix::concat; // ✅ Good
        
        // 5. Using custom static methods
        words.stream()
             .map(ChoiceGuidelines::formatString)  // ✅ Good
             .filter(ChoiceGuidelines::isLongWord) // ✅ Good (if isLongWord matches the logic)
             .forEach(System.out::println);
        
        // ✅ Use Lambda Expressions When:
        
        // 1. Need multiple operations
        words.stream()
             .map(s -> s.toUpperCase().trim()) // ✅ Good
             .forEach(System.out::println);
        
        // 2. Need conditional logic
        words.stream()
             .filter(s -> s.length() > 4 && s.startsWith("p")) // ✅ Good
             .forEach(System.out::println);
        
        // 3. Need to capture local variables
        int minLength = 5;
        words.stream()
             .filter(s -> s.length() >= minLength) // ✅ Good
             .forEach(System.out::println);
        
        // 4. Need custom complex logic
        words.stream()
             .map(s -> s.substring(0, Math.min(3, s.length())) + "...") // ✅ Good
             .forEach(System.out::println);
        
        // 5. Need to handle exceptions
        List<String> fileNames = Arrays.asList("file1.txt", "file2.txt");
        fileNames.stream()
                 .map(name -> {
                     try {
                         return name.toUpperCase();
                     } catch (Exception e) {
                         return "ERROR";
                     }
                 }) // ✅ Good - exception handling
                 .forEach(System.out::println);
    }
}
```

## 8. Advanced Examples and Use Cases

### Method References in Stream Processing
```java
import java.util.*;
import java.util.function.*;
import java.util.stream.Collectors;

public class AdvancedStreamExamples {
    
    public static void main(String[] args) {
        List<Employee> employees = Arrays.asList(
            new Employee("Alice", "Engineering", 75000, 3),
            new Employee("Bob", "Sales", 60000, 2),
            new Employee("Charlie", "Engineering", 80000, 5),
            new Employee("Diana", "Marketing", 65000, 4),
            new Employee("Eve", "Sales", 70000, 3)
        );
        
        // Complex stream operations using method references
        System.out.println("=== Advanced Stream Processing ===");
        
        // 1. Grouping with method references
        Map<String, List<Employee>> byDepartment = employees.stream()
                .collect(Collectors.groupingBy(Employee::getDepartment));
        
        byDepartment.forEach((dept, emps) -> {
            System.out.println(dept + ": " + emps.stream()
                                                .map(Employee::getName)
                                                .collect(Collectors.joining(", ")));
        });
        
        // 2. Finding statistics using method references
        DoubleSummaryStatistics salaryStats = employees.stream()
                .mapToDouble(Employee::getSalary)
                .summaryStatistics();
        
        System.out.println("\nSalary Statistics:");
        System.out.println("Average: " + salaryStats.getAverage());
        System.out.println("Max: " + salaryStats.getMax());
        System.out.println("Min: " + salaryStats.getMin());
        
        // 3. Complex sorting with method references
        List<Employee> sorted = employees.stream()
                .sorted(Comparator.comparing(Employee::getDepartment)
                        .thenComparing(Employee::getSalary, Comparator.reverseOrder())
                        .thenComparing(Employee::getName))
                .collect(Collectors.toList());
        
        System.out.println("\nSorted employees:");
        sorted.forEach(System.out::println);
        
        // 4. Partitioning with method references
        Map<Boolean, List<Employee>> partitioned = employees.stream()
                .collect(Collectors.partitioningBy(emp -> emp.getSalary() > 70000));
        
        System.out.println("\nHigh earners:");
        partitioned.get(true).forEach(System.out::println);
        
        // 5. Custom collectors with method references
        String employeeNames = employees.stream()
                .map(Employee::getName)
                .collect(Collectors.joining(", ", "[", "]"));
        
        System.out.println("\nAll employee names: " + employeeNames);
        
        // 6. Parallel processing with method references
        double avgSalaryParallel = employees.parallelStream()
                .mapToDouble(Employee::getSalary)
                .average()
                .orElse(0.0);
        
        System.out.println("Average salary (parallel): " + avgSalaryParallel);
    }
}
```

### Method References with Optional
```java
import java.util.*;
import java.util.function.*;

public class MethodReferencesWithOptional {
    
    public static void main(String[] args) {
        List<String> words = Arrays.asList("hello", "world", null, "java", "", "programming");
        
        // Using method references with Optional operations
        System.out.println("=== Method References with Optional ===");
        
        // 1. Filter and map with method references
        words.stream()
             .filter(Objects::nonNull)           // Static method reference
             .filter(Predicate.not(String::isEmpty)) // Instance method reference
             .map(String::toUpperCase)           // Instance method reference
             .forEach(System.out::println);      // Instance method reference
        
        // 2. Finding with method references
        Optional<String> longestWord = words.stream()
                .filter(Objects::nonNull)
                .max(Comparator.comparing(String::length));
        
        longestWord.ifPresent(System.out::println);
        
        // 3. Optional operations with method references
        Optional<String> result = Optional.of("  hello world  ")
                .filter(Predicate.not(String::isEmpty))
                .map(String::trim)
                .map(String::toUpperCase);
        
        result.ifPresent(System.out::println);
        
        // 4. FlatMap with method references
        List<Optional<String>> optionals = Arrays.asList(
            Optional.of("first"),
            Optional.empty(),
            Optional.of("second"),
            Optional.of("third")
        );
        
        List<String> values = optionals.stream()
                .flatMap(Optional::stream)  // Method reference to stream()
                .map(String::toUpperCase)
                .collect(Collectors.toList());
        
        System.out.println("Extracted values: " + values);
        
        // 5. OrElse operations
        String defaultValue = Optional.ofNullable((String) null)
                .orElseGet(String::new);  // Constructor reference
        
        System.out.println("Default value: '" + defaultValue + "'");
    }
}
```

### Method References in Concurrent Programming
```java
import java.util.*;
import java.util.concurrent.*;
import java.util.function.*;

public class ConcurrentMethodReferences {
    
    private static final ExecutorService executor = Executors.newFixedThreadPool(4);
    
    public static String processString(String input) {
        // Simulate some processing
        try {
            Thread.sleep(100);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        return input.toUpperCase() + "_PROCESSED";
    }
    
    public static void main(String[] args) throws InterruptedException, ExecutionException {
        List<String> data = Arrays.asList("item1", "item2", "item3", "item4", "item5");
        
        System.out.println("=== Concurrent Processing with Method References ===");
        
        // 1. CompletableFuture with method references
        List<CompletableFuture<String>> futures = data.stream()
                .map(item -> CompletableFuture.supplyAsync(() -> processString(item), executor))
                .collect(Collectors.toList());
        
        // Wait for all to complete and collect results
        CompletableFuture<List<String>> allFutures = CompletableFuture
                .allOf(futures.toArray(new CompletableFuture[0]))
                .thenApply(v -> futures.stream()
                        .map(CompletableFuture::join)  // Method reference to join()
                        .collect(Collectors.toList()));
        
        List<String> results = allFutures.get();
        System.out.println("Processed results: " + results);
        
        // 2. Using method references in async operations
        CompletableFuture<String> pipeline = CompletableFuture
                .supplyAsync(() -> "hello world", executor)
                .thenApply(String::toUpperCase)        // Method reference
                .thenApply(String::trim)               // Method reference
                .thenCompose(s -> CompletableFuture.supplyAsync(() -> processString(s), executor));
        
        System.out.println("Pipeline result: " + pipeline.get());
        
        // 3. Exception handling with method references
        CompletableFuture<String> withErrorHandling = CompletableFuture
                .supplyAsync(() -> {
                    if (Math.random() > 0.5) {
                        throw new RuntimeException("Random error");
                    }
                    return "success";
                }, executor)
                .exceptionally(Throwable::getMessage)  // Method reference to getMessage()
                .thenApply(String::toUpperCase);
        
        System.out.println("With error handling: " + withErrorHandling.get());
        
        executor.shutdown();
        executor.awaitTermination(1, TimeUnit.SECONDS);
    }
}
```

## 9. Best Practices

### Code Organization and Readability
```java
import java.util.*;
import java.util.function.*;
import java.util.stream.Collectors;

public class MethodReferenceBestPractices {
    
    // ✅ Best Practice 1: Use meaningful method names for method references
    public static boolean isAdult(Person person) {
        return person.getAge() >= 18;
    }
    
    public static String formatPersonName(Person person) {
        return person.getName().toUpperCase() + " (" + person.getAge() + ")";
    }
    
    // ✅ Best Practice 2: Create utility methods for complex operations
    public static class StringUtils {
        public static String normalizeWhitespace(String input) {
            return input.trim().replaceAll("\\s+", " ");
        }
        
        public static boolean isNotEmpty(String input) {
            return input != null && !input.isEmpty();
        }
    }
    
    public static void main(String[] args) {
        List<Person> people = Arrays.asList(
            new Person("Alice", 25),
            new Person("Bob", 17),
            new Person("Charlie", 30)
        );
        
        List<String> texts = Arrays.asList("  hello  world  ", "", "java   programming", null);
        
        System.out.println("=== Best Practices Examples ===");
        
        // ✅ Good: Use meaningful method references
        List<Person> adults = people.stream()
                .filter(MethodReferenceBestPractices::isAdult)  // Clear intent
                .collect(Collectors.toList());
        
        List<String> formattedNames = people.stream()
                .map(MethodReferenceBestPractices::formatPersonName)  // Clear purpose
                .collect(Collectors.toList());
        
        // ✅ Good: Use utility methods for reusable logic
        List<String> cleanTexts = texts.stream()
                .filter(Objects::nonNull)
                .filter(StringUtils::isNotEmpty)
                .map(StringUtils::normalizeWhitespace)
                .collect(Collectors.toList());
        
        System.out.println("Adults: " + adults);
        System.out.println("Formatted names: " + formattedNames);
        System.out.println("Clean texts: " + cleanTexts);
        
        // ❌ Avoid: Don't overuse method references when lambdas are clearer
        // Bad - unclear what this does
        // people.stream().filter(p -> someComplexCondition(p)).collect(toList());
        
        // ✅ Good: Use lambda when logic is simple and specific
        List<Person> youngAdults = people.stream()
                .filter(p -> p.getAge() >= 18 && p.getAge() <= 25)  // Clear inline logic
                .collect(Collectors.toList());
        
        System.out.println("Young adults: " + youngAdults);
        
        // ✅ Best Practice 3: Chain method references appropriately
        Map<Boolean, List<String>> partitionedNames = people.stream()
                .filter(Objects::nonNull)
                .collect(Collectors.partitioningBy(
                    MethodReferenceBestPractices::isAdult,
                    Collectors.mapping(Person::getName, Collectors.toList())
                ));
        
        System.out.println("Partitioned by adult status: " + partitionedNames);
    }
}
```

### Performance Considerations
```java
import java.util.*;
import java.util.function.*;
import java.util.stream.IntStream;

public class PerformanceBestPractices {
    
    // Reuse method references when possible
    private static final Function<String, String> TO_UPPER = String::toUpperCase;
    private static final Predicate<String> IS_NOT_EMPTY = Predicate.not(String::isEmpty);
    private static final IntFunction<String[]> STRING_ARRAY_CREATOR = String[]::new;
    
    public static void main(String[] args) {
        List<String> largeList = generateLargeList(100000);
        
        System.out.println("=== Performance Best Practices ===");
        
        // ✅ Best Practice 1: Reuse method references
        long start = System.nanoTime();
        for (int i = 0; i < 100; i++) {
            largeList.stream()
                    .filter(Objects::nonNull)
                    .filter(IS_NOT_EMPTY)  // Reused predicate
                    .map(TO_UPPER)         // Reused function
                    .toArray(STRING_ARRAY_CREATOR);  // Reused array constructor
        }
        long reuseTime = System.nanoTime() - start;
        
        // ❌ Less efficient: Creating new method references each time
        start = System.nanoTime();
        for (int i = 0; i < 100; i++) {
            largeList.stream()
                    .filter(Objects::nonNull)
                    .filter(Predicate.not(String::isEmpty))  // New instance each time
                    .map(String::toUpperCase)                 // New instance each time
                    .toArray(String[]::new);                  // New instance each time
        }
        long newTime = System.nanoTime() - start;
        
        System.out.println("Reused method references: " + reuseTime / 1000000 + "ms");
        System.out.println("New method references: " + newTime / 1000000 + "ms");
        
        // ✅ Best Practice 2: Use appropriate stream operations
        // For large datasets, consider parallel streams with method references
        start = System.nanoTime();
        long count = largeList.parallelStream()
                .filter(Objects::nonNull)
                .filter(IS_NOT_EMPTY)
                .map(TO_UPPER)
                .count();
        long parallelTime = System.nanoTime() - start;
        
        start = System.nanoTime();
        long count2 = largeList.stream()
                .filter(Objects::nonNull)
                .filter(IS_NOT_EMPTY)
                .map(TO_UPPER)
                .count();
        long sequentialTime = System.nanoTime() - start;
        
        System.out.println("Parallel processing: " + parallelTime / 1000000 + "ms (count: " + count + ")");
        System.out.println("Sequential processing: " + sequentialTime / 1000000 + "ms (count: " + count2 + ")");
    }
    
    private static List<String> generateLargeList(int size) {
        return IntStream.range(0, size)
                .mapToObj(i -> i % 10 == 0 ? null : "item_" + i)
                .collect(ArrayList::new, ArrayList::add, ArrayList::addAll);
    }
}
```

## 10. Common Pitfalls

### Type Inference Issues
```java
import java.util.*;
import java.util.function.*;

public class CommonPitfalls {
    
    public static void main(String[] args) {
        System.out.println("=== Common Pitfalls and Solutions ===");
        
        // ❌ Pitfall 1: Ambiguous method references
        List<String> strings = Arrays.asList("1", "2", "3");
        
        // This might be ambiguous in some contexts
        // Function<String, ?> ambiguous = Integer::valueOf;  // Could be Integer or int
        
        // ✅ Solution: Be explicit with types
        Function<String, Integer> explicit = Integer::valueOf;
        ToIntFunction<String> primitiveVersion = Integer::parseInt;
        
        // ❌ Pitfall 2: Null pointer exceptions with method references
        List<String> withNulls = Arrays.asList("hello", null, "world");
        
        try {
            // This will throw NPE
            withNulls.stream()
                    .map(String::toUpperCase)  // NPE when processing null
                    .forEach(System.out::println);
        } catch (NullPointerException e) {
            System.out.println("NPE caught: " + e.getMessage());
        }
        
        // ✅ Solution: Filter nulls first
        withNulls.stream()
                .filter(Objects::nonNull)
                .map(String::toUpperCase)
                .forEach(System.out::println);
        
        // ❌ Pitfall 3: Incorrect method reference type
        List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5);
        
        // Wrong: This doesn't compile
        // Predicate<Integer> isEven = Integer::intValue;  // Wrong return type
        
        // ✅ Solution: Use correct functional interface
        ToIntFunction<Integer> toInt = Integer::intValue;
        Predicate<Integer> isEven = n -> n % 2 == 0;  // Use lambda for custom logic
        
        // ❌ Pitfall 4: Method reference to overloaded methods
        // This can be ambiguous
        // Function<String, ?> overloaded = String::valueOf;  // Multiple valueOf methods
        
        // ✅ Solution: Use lambda for clarity or be more specific
        Function<Object, String> objectToString = String::valueOf;
        Function<int[], String> arrayToString = Arrays::toString;
        
        // ❌ Pitfall 5: Capturing instance state incorrectly
        List<String> items = Arrays.asList("a", "b", "c");
        StringBuilder sb = new StringBuilder();
        
        // This creates a new StringBuilder for each element (not what we want)
        // items.forEach(StringBuilder::new);  // Wrong!
        
        // ✅ Solution: Use instance method reference correctly
        items.forEach(sb::append);  // Appends to the same StringBuilder
        System.out.println("StringBuilder result: " + sb.toString());
        
        // ❌ Pitfall 6: Performance anti-patterns
        // Don't create new instances unnecessarily
        for (int i = 0; i < 1000; i++) {
            // Bad: Creates new predicate each time
            // strings.stream().filter(s -> Objects.nonNull(s));
        }
        
        // ✅ Solution: Reuse method references
        Predicate<String> notNull = Objects::nonNull;
        for (int i = 0; i < 1000; i++) {
            strings.stream().filter(notNull);
        }
        
        System.out.println("All pitfalls demonstrated and resolved!");
    }
}
```

### Exception Handling with Method References
```java
import java.util.*;
import java.util.function.*;

public class ExceptionHandlingPitfalls {
    
    // Method that throws checked exception
    public static int parseIntWithException(String s) throws NumberFormatException {
        return Integer.parseInt(s);
    }
    
    // Wrapper for safe parsing
    public static Optional<Integer> safeParse(String s) {
        try {
            return Optional.of(Integer.parseInt(s));
        } catch (NumberFormatException e) {
            return Optional.empty();
        }
    }
    
    public static void main(String[] args) {
        List<String> inputs = Arrays.asList("1", "2", "invalid", "4", "5");
        
        System.out.println("=== Exception Handling with Method References ===");
        
        // ❌ Problem: Method references don't handle exceptions well
        try {
            inputs.stream()
                  .map(Integer::parseInt)  // Will throw exception on "invalid"
                  .forEach(System.out::println);
        } catch (NumberFormatException e) {
            System.out.println("Exception caught: " + e.getMessage());
        }
        
        // ✅ Solution 1: Use wrapper methods
        List<Optional<Integer>> safeResults = inputs.stream()
                .map(ExceptionHandlingPitfalls::safeParse)
                .collect(Collectors.toList());
        
        System.out.println("Safe parsing results: " + safeResults);
        
        // Extract only successful results
        List<Integer> successfulResults = inputs.stream()
                .map(ExceptionHandlingPitfalls::safeParse)
                .filter(Optional::isPresent)
                .map(Optional::get)
                .collect(Collectors.toList());
        
        System.out.println("Successful results: " + successfulResults);
        
        // ✅ Solution 2: Use lambda for exception handling
        List<Integer> lambdaResults = inputs.stream()
                .map(s -> {
                    try {
                        return Integer.parseInt(s);
                    } catch (NumberFormatException e) {
                        return -1; // Default value
                    }
                })
                .collect(Collectors.toList());
        
        System.out.println("Lambda with exception handling: " + lambdaResults);
        
        // ✅ Solution 3: Create utility functional interfaces
        @FunctionalInterface
        interface ThrowingFunction<T, R> {
            R apply(T t) throws Exception;
            
            static <T, R> Function<T, Optional<R>> safe(ThrowingFunction<T, R> function) {
                return t -> {
                    try {
                        return Optional.of(function.apply(t));
                    } catch (Exception e) {
                        return Optional.empty();
                    }
                };
            }
        }
        
        List<Optional<Integer>> utilityResults = inputs.stream()
                .map(ThrowingFunction.safe(Integer::parseInt))
                .collect(Collectors.toList());
        
        System.out.println("Utility function results: " + utilityResults);
    }
}
```

## Conclusion

Method references are a powerful feature in Java that can make your code more readable and maintainable when used appropriately. They provide a concise way to refer to methods without invoking them and work seamlessly with functional interfaces and the Stream API.

### Key Takeaways:

1. **Use method references for simple, single-method operations**
2. **Use lambdas for complex logic or when capturing local variables**
3. **Consider readability and maintainability over brevity**
4. **Be aware of type inference limitations and null safety**
5. **Reuse method references for better performance**
6. **Handle exceptions appropriately when using method references**

### Quick Reference:

| Type | Syntax | Example | Equivalent Lambda |
|------|--------|---------|-------------------|
| Static Method | `Class::staticMethod` | `Integer::parseInt` | `s -> Integer.parseInt(s)` |
| Instance Method (Particular Object) | `instance::instanceMethod` | `str::length` | `() -> str.length()` |
| Instance Method (Arbitrary Object) | `Class::instanceMethod` | `String::length` | `s -> s.length()` |
| Constructor | `Class::new` | `ArrayList::new` | `() -> new ArrayList<>()` |

Method references are an essential tool for writing clean, functional-style Java code. Practice using them in your streams and functional programming code to become proficient with this feature.
