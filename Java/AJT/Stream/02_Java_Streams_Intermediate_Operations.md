# Java Streams - Intermediate Operations

## 1. Filter Operations

The `filter()` operation allows you to select elements based on a predicate condition.

### Basic Filtering
```java
import java.util.*;
import java.util.stream.Collectors;

public class FilterOperationsDemo {
    public static void main(String[] args) {
        List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        
        // Filter even numbers
        List<Integer> evenNumbers = numbers.stream()
            .filter(n -> n % 2 == 0)
            .collect(Collectors.toList());
        System.out.println("Even numbers: " + evenNumbers);
        
        // Filter numbers greater than 5
        List<Integer> greaterThanFive = numbers.stream()
            .filter(n -> n > 5)
            .collect(Collectors.toList());
        System.out.println("Numbers > 5: " + greaterThanFive);
        
        // Multiple filter conditions
        List<Integer> evenAndGreaterThanFive = numbers.stream()
            .filter(n -> n % 2 == 0)
            .filter(n -> n > 5)
            .collect(Collectors.toList());
        System.out.println("Even and > 5: " + evenAndGreaterThanFive);
        
        // Combined filter with logical operators
        List<Integer> combinedFilter = numbers.stream()
            .filter(n -> n % 2 == 0 && n > 5)
            .collect(Collectors.toList());
        System.out.println("Combined filter: " + combinedFilter);
    }
}
```

### Advanced Filtering with Objects
```java
import java.util.*;
import java.util.stream.Collectors;

public class AdvancedFilteringDemo {
    public static void main(String[] args) {
        List<Employee> employees = Arrays.asList(
            new Employee("Alice", 25, "Engineering", 75000),
            new Employee("Bob", 30, "Marketing", 60000),
            new Employee("Charlie", 35, "Engineering", 85000),
            new Employee("Diana", 28, "HR", 55000),
            new Employee("Eve", 32, "Engineering", 90000),
            new Employee("Frank", 45, "Marketing", 70000)
        );
        
        // Filter by department
        List<Employee> engineers = employees.stream()
            .filter(emp -> "Engineering".equals(emp.getDepartment()))
            .collect(Collectors.toList());
        System.out.println("Engineers: " + engineers);
        
        // Filter by salary range
        List<Employee> highEarners = employees.stream()
            .filter(emp -> emp.getSalary() > 70000)
            .collect(Collectors.toList());
        System.out.println("High earners (>70k): " + highEarners);
        
        // Complex filtering conditions
        List<Employee> seniorEngineers = employees.stream()
            .filter(emp -> "Engineering".equals(emp.getDepartment()))
            .filter(emp -> emp.getAge() > 30)
            .filter(emp -> emp.getSalary() > 80000)
            .collect(Collectors.toList());
        System.out.println("Senior engineers: " + seniorEngineers);
        
        // Filter with method references
        List<Employee> experiencedEmployees = employees.stream()
            .filter(Employee::isExperienced)  // Custom method
            .collect(Collectors.toList());
        System.out.println("Experienced employees: " + experiencedEmployees);
        
        // Filter with null checking
        List<String> departments = Arrays.asList("Engineering", null, "Marketing", "HR", null);
        List<String> validDepartments = departments.stream()
            .filter(Objects::nonNull)
            .collect(Collectors.toList());
        System.out.println("Valid departments: " + validDepartments);
    }
    
    static class Employee {
        private String name;
        private int age;
        private String department;
        private double salary;
        
        public Employee(String name, int age, String department, double salary) {
            this.name = name;
            this.age = age;
            this.department = department;
            this.salary = salary;
        }
        
        public String getName() { return name; }
        public int getAge() { return age; }
        public String getDepartment() { return department; }
        public double getSalary() { return salary; }
        
        public boolean isExperienced() {
            return age > 30 && salary > 65000;
        }
        
        @Override
        public String toString() {
            return String.format("%s(%d, %s, %.0f)", name, age, department, salary);
        }
    }
}
```

## 2. Map Operations

The `map()` operation transforms each element of the stream using a provided function.

### Basic Mapping
```java
import java.util.*;
import java.util.stream.Collectors;

public class MapOperationsDemo {
    public static void main(String[] args) {
        List<String> words = Arrays.asList("hello", "world", "java", "streams");
        
        // Transform to uppercase
        List<String> upperWords = words.stream()
            .map(String::toUpperCase)
            .collect(Collectors.toList());
        System.out.println("Uppercase: " + upperWords);
        
        // Transform to word lengths
        List<Integer> lengths = words.stream()
            .map(String::length)
            .collect(Collectors.toList());
        System.out.println("Lengths: " + lengths);
        
        // Transform numbers
        List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5);
        List<Integer> squares = numbers.stream()
            .map(n -> n * n)
            .collect(Collectors.toList());
        System.out.println("Squares: " + squares);
        
        // Chain transformations
        List<String> processedWords = words.stream()
            .map(String::toUpperCase)
            .map(word -> "PREFIX_" + word)
            .map(word -> word + "_SUFFIX")
            .collect(Collectors.toList());
        System.out.println("Processed: " + processedWords);
    }
}
```

### Object Transformation
```java
import java.util.*;
import java.util.stream.Collectors;

public class ObjectMappingDemo {
    public static void main(String[] args) {
        List<Employee> employees = Arrays.asList(
            new Employee("Alice", 25, "Engineering", 75000),
            new Employee("Bob", 30, "Marketing", 60000),
            new Employee("Charlie", 35, "Engineering", 85000)
        );
        
        // Extract specific fields
        List<String> names = employees.stream()
            .map(Employee::getName)
            .collect(Collectors.toList());
        System.out.println("Names: " + names);
        
        List<String> departments = employees.stream()
            .map(Employee::getDepartment)
            .distinct()  // Remove duplicates
            .collect(Collectors.toList());
        System.out.println("Unique departments: " + departments);
        
        // Transform to different objects
        List<EmployeeSummary> summaries = employees.stream()
            .map(emp -> new EmployeeSummary(emp.getName(), emp.getDepartment()))
            .collect(Collectors.toList());
        System.out.println("Summaries: " + summaries);
        
        // Complex transformations
        List<String> employeeInfo = employees.stream()
            .map(emp -> String.format("%s works in %s earning $%.0f", 
                emp.getName(), emp.getDepartment(), emp.getSalary()))
            .collect(Collectors.toList());
        System.out.println("Employee info:");
        employeeInfo.forEach(System.out::println);
        
        // Mathematical transformations
        List<Double> salariesInK = employees.stream()
            .map(Employee::getSalary)
            .map(salary -> salary / 1000.0)
            .collect(Collectors.toList());
        System.out.println("Salaries in K: " + salariesInK);
    }
    
    static class Employee {
        private String name;
        private int age;
        private String department;
        private double salary;
        
        public Employee(String name, int age, String department, double salary) {
            this.name = name;
            this.age = age;
            this.department = department;
            this.salary = salary;
        }
        
        public String getName() { return name; }
        public int getAge() { return age; }
        public String getDepartment() { return department; }
        public double getSalary() { return salary; }
    }
    
    static class EmployeeSummary {
        private String name;
        private String department;
        
        public EmployeeSummary(String name, String department) {
            this.name = name;
            this.department = department;
        }
        
        @Override
        public String toString() {
            return name + " in " + department;
        }
    }
}
```

## 3. FlatMap Operations

The `flatMap()` operation is used to flatten nested structures and create a single stream from multiple streams.

### Basic FlatMap
```java
import java.util.*;
import java.util.stream.Collectors;

public class FlatMapDemo {
    public static void main(String[] args) {
        // Flatten list of lists
        List<List<Integer>> listOfLists = Arrays.asList(
            Arrays.asList(1, 2, 3),
            Arrays.asList(4, 5, 6),
            Arrays.asList(7, 8, 9)
        );
        
        List<Integer> flattened = listOfLists.stream()
            .flatMap(List::stream)
            .collect(Collectors.toList());
        System.out.println("Flattened: " + flattened);
        
        // Flatten strings to characters
        List<String> words = Arrays.asList("hello", "world");
        List<String> characters = words.stream()
            .flatMap(word -> Arrays.stream(word.split("")))
            .collect(Collectors.toList());
        System.out.println("Characters: " + characters);
        
        // Flatten and transform
        List<String> uppercaseChars = words.stream()
            .flatMap(word -> Arrays.stream(word.split("")))
            .map(String::toUpperCase)
            .distinct()
            .sorted()
            .collect(Collectors.toList());
        System.out.println("Unique uppercase chars: " + uppercaseChars);
    }
}
```

### Advanced FlatMap Use Cases
```java
import java.util.*;
import java.util.stream.Collectors;

public class AdvancedFlatMapDemo {
    public static void main(String[] args) {
        List<Department> departments = Arrays.asList(
            new Department("Engineering", Arrays.asList("Alice", "Bob", "Charlie")),
            new Department("Marketing", Arrays.asList("Diana", "Eve")),
            new Department("HR", Arrays.asList("Frank"))
        );
        
        // Flatten department employees
        List<String> allEmployees = departments.stream()
            .flatMap(dept -> dept.getEmployees().stream())
            .collect(Collectors.toList());
        System.out.println("All employees: " + allEmployees);
        
        // Create employee-department pairs
        List<String> employeeDeptPairs = departments.stream()
            .flatMap(dept -> dept.getEmployees().stream()
                .map(emp -> emp + " - " + dept.getName()))
            .collect(Collectors.toList());
        System.out.println("Employee-Department pairs:");
        employeeDeptPairs.forEach(System.out::println);
        
        // Flatten optional values
        List<Optional<String>> optionals = Arrays.asList(
            Optional.of("value1"),
            Optional.empty(),
            Optional.of("value2"),
            Optional.empty(),
            Optional.of("value3")
        );
        
        List<String> presentValues = optionals.stream()
            .flatMap(Optional::stream)  // Java 9+ feature
            .collect(Collectors.toList());
        System.out.println("Present values: " + presentValues);
        
        // Alternative for Java 8
        List<String> presentValuesJava8 = optionals.stream()
            .filter(Optional::isPresent)
            .map(Optional::get)
            .collect(Collectors.toList());
        System.out.println("Present values (Java 8): " + presentValuesJava8);
        
        // Flatten arrays
        String[][] arrays = {
            {"a", "b", "c"},
            {"d", "e"},
            {"f", "g", "h", "i"}
        };
        
        List<String> flattenedArray = Arrays.stream(arrays)
            .flatMap(Arrays::stream)
            .collect(Collectors.toList());
        System.out.println("Flattened array: " + flattenedArray);
    }
    
    static class Department {
        private String name;
        private List<String> employees;
        
        public Department(String name, List<String> employees) {
            this.name = name;
            this.employees = employees;
        }
        
        public String getName() { return name; }
        public List<String> getEmployees() { return employees; }
    }
}
```

## 4. Sorting Operations

The `sorted()` operation allows you to sort stream elements.

### Basic Sorting
```java
import java.util.*;
import java.util.stream.Collectors;

public class SortingDemo {
    public static void main(String[] args) {
        List<String> words = Arrays.asList("banana", "apple", "cherry", "date");
        List<Integer> numbers = Arrays.asList(5, 2, 8, 1, 9, 3);
        
        // Natural sorting (alphabetical for strings)
        List<String> sortedWords = words.stream()
            .sorted()
            .collect(Collectors.toList());
        System.out.println("Sorted words: " + sortedWords);
        
        // Natural sorting (ascending for numbers)
        List<Integer> sortedNumbers = numbers.stream()
            .sorted()
            .collect(Collectors.toList());
        System.out.println("Sorted numbers: " + sortedNumbers);
        
        // Reverse sorting
        List<String> reverseSortedWords = words.stream()
            .sorted(Comparator.reverseOrder())
            .collect(Collectors.toList());
        System.out.println("Reverse sorted words: " + reverseSortedWords);
        
        // Custom sorting by length
        List<String> sortedByLength = words.stream()
            .sorted(Comparator.comparing(String::length))
            .collect(Collectors.toList());
        System.out.println("Sorted by length: " + sortedByLength);
        
        // Multiple sorting criteria
        List<String> multiSorted = words.stream()
            .sorted(Comparator.comparing(String::length)
                .thenComparing(Comparator.naturalOrder()))
            .collect(Collectors.toList());
        System.out.println("Sorted by length then alphabetically: " + multiSorted);
    }
}
```

### Complex Object Sorting
```java
import java.util.*;
import java.util.stream.Collectors;

public class ComplexSortingDemo {
    public static void main(String[] args) {
        List<Employee> employees = Arrays.asList(
            new Employee("Alice", 25, "Engineering", 75000),
            new Employee("Bob", 30, "Marketing", 60000),
            new Employee("Charlie", 35, "Engineering", 85000),
            new Employee("Diana", 28, "HR", 55000),
            new Employee("Eve", 32, "Engineering", 90000),
            new Employee("Alice", 40, "Marketing", 70000)  // Same name, different person
        );
        
        // Sort by single field
        List<Employee> sortedByAge = employees.stream()
            .sorted(Comparator.comparing(Employee::getAge))
            .collect(Collectors.toList());
        System.out.println("Sorted by age:");
        sortedByAge.forEach(System.out::println);
        
        // Sort by salary (descending)
        List<Employee> sortedBySalaryDesc = employees.stream()
            .sorted(Comparator.comparing(Employee::getSalary).reversed())
            .collect(Collectors.toList());
        System.out.println("\nSorted by salary (descending):");
        sortedBySalaryDesc.forEach(System.out::println);
        
        // Sort by multiple fields
        List<Employee> multiFieldSort = employees.stream()
            .sorted(Comparator.comparing(Employee::getDepartment)
                .thenComparing(Employee::getName)
                .thenComparing(Employee::getAge))
            .collect(Collectors.toList());
        System.out.println("\nSorted by department, then name, then age:");
        multiFieldSort.forEach(System.out::println);
        
        // Custom comparator
        List<Employee> customSort = employees.stream()
            .sorted((e1, e2) -> {
                // First by department
                int deptComparison = e1.getDepartment().compareTo(e2.getDepartment());
                if (deptComparison != 0) return deptComparison;
                
                // Then by salary (descending)
                return Double.compare(e2.getSalary(), e1.getSalary());
            })
            .collect(Collectors.toList());
        System.out.println("\nCustom sort (dept asc, salary desc):");
        customSort.forEach(System.out::println);
        
        // Null-safe sorting
        List<String> wordsWithNulls = Arrays.asList("apple", null, "banana", null, "cherry");
        List<String> nullSafeSorted = wordsWithNulls.stream()
            .sorted(Comparator.nullsLast(Comparator.naturalOrder()))
            .collect(Collectors.toList());
        System.out.println("\nNull-safe sorting: " + nullSafeSorted);
    }
    
    static class Employee {
        private String name;
        private int age;
        private String department;
        private double salary;
        
        public Employee(String name, int age, String department, double salary) {
            this.name = name;
            this.age = age;
            this.department = department;
            this.salary = salary;
        }
        
        public String getName() { return name; }
        public int getAge() { return age; }
        public String getDepartment() { return department; }
        public double getSalary() { return salary; }
        
        @Override
        public String toString() {
            return String.format("%s(%d, %s, $%.0f)", name, age, department, salary);
        }
    }
}
```

## 5. Other Intermediate Operations

### Distinct
```java
import java.util.*;
import java.util.stream.Collectors;

public class DistinctDemo {
    public static void main(String[] args) {
        List<Integer> numbersWithDuplicates = Arrays.asList(1, 2, 2, 3, 3, 3, 4, 4, 5);
        
        // Remove duplicates
        List<Integer> uniqueNumbers = numbersWithDuplicates.stream()
            .distinct()
            .collect(Collectors.toList());
        System.out.println("Unique numbers: " + uniqueNumbers);
        
        // Distinct with objects (uses equals() method)
        List<Person> people = Arrays.asList(
            new Person("Alice", 25),
            new Person("Bob", 30),
            new Person("Alice", 25),  // Duplicate
            new Person("Charlie", 35),
            new Person("Bob", 30)     // Duplicate
        );
        
        List<Person> uniquePeople = people.stream()
            .distinct()
            .collect(Collectors.toList());
        System.out.println("Unique people: " + uniquePeople);
        
        // Distinct by specific field
        List<Person> distinctByName = people.stream()
            .collect(Collectors.toMap(
                Person::getName,
                p -> p,
                (existing, replacement) -> existing))  // Keep first occurrence
            .values()
            .stream()
            .collect(Collectors.toList());
        System.out.println("Distinct by name: " + distinctByName);
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
        public boolean equals(Object obj) {
            if (this == obj) return true;
            if (obj == null || getClass() != obj.getClass()) return false;
            Person person = (Person) obj;
            return age == person.age && Objects.equals(name, person.name);
        }
        
        @Override
        public int hashCode() {
            return Objects.hash(name, age);
        }
        
        @Override
        public String toString() {
            return name + "(" + age + ")";
        }
    }
}
```

### Limit and Skip
```java
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class LimitSkipDemo {
    public static void main(String[] args) {
        List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        
        // Limit - take first n elements
        List<Integer> firstFive = numbers.stream()
            .limit(5)
            .collect(Collectors.toList());
        System.out.println("First 5: " + firstFive);
        
        // Skip - skip first n elements
        List<Integer> skipFirstThree = numbers.stream()
            .skip(3)
            .collect(Collectors.toList());
        System.out.println("Skip first 3: " + skipFirstThree);
        
        // Combine skip and limit for pagination
        List<Integer> page2 = numbers.stream()
            .skip(3)    // Skip first 3 (page 1)
            .limit(3)   // Take next 3 (page 2)
            .collect(Collectors.toList());
        System.out.println("Page 2 (elements 4-6): " + page2);
        
        // Infinite stream with limit
        List<Integer> firstTenEvenNumbers = IntStream.iterate(0, n -> n + 2)
            .limit(10)
            .boxed()
            .collect(Collectors.toList());
        System.out.println("First 10 even numbers: " + firstTenEvenNumbers);
        
        // Practical pagination example
        List<String> items = Arrays.asList(
            "Item1", "Item2", "Item3", "Item4", "Item5",
            "Item6", "Item7", "Item8", "Item9", "Item10"
        );
        
        int pageSize = 3;
        int pageNumber = 2; // 0-based
        
        List<String> pageItems = items.stream()
            .skip(pageNumber * pageSize)
            .limit(pageSize)
            .collect(Collectors.toList());
        System.out.println("Page " + pageNumber + ": " + pageItems);
    }
}
```

### Peek (for debugging)
```java
import java.util.*;
import java.util.stream.Collectors;

public class PeekDemo {
    public static void main(String[] args) {
        List<String> words = Arrays.asList("apple", "banana", "cherry", "date");
        
        // Use peek for debugging/logging
        List<String> result = words.stream()
            .peek(word -> System.out.println("Processing: " + word))
            .filter(word -> word.length() > 5)
            .peek(word -> System.out.println("After filter: " + word))
            .map(String::toUpperCase)
            .peek(word -> System.out.println("After map: " + word))
            .collect(Collectors.toList());
        
        System.out.println("Final result: " + result);
        
        // Peek for side effects (use carefully)
        List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5);
        List<String> debugInfo = new ArrayList<>();
        
        List<Integer> processed = numbers.stream()
            .peek(n -> debugInfo.add("Processing: " + n))
            .filter(n -> n % 2 == 0)
            .peek(n -> debugInfo.add("Filtered: " + n))
            .map(n -> n * n)
            .peek(n -> debugInfo.add("Squared: " + n))
            .collect(Collectors.toList());
        
        System.out.println("\nDebug information:");
        debugInfo.forEach(System.out::println);
        System.out.println("Processed: " + processed);
    }
}
```

## 6. Chaining Intermediate Operations

### Complex Processing Pipeline
```java
import java.util.*;
import java.util.stream.Collectors;

public class ComplexPipelineDemo {
    public static void main(String[] args) {
        List<Order> orders = Arrays.asList(
            new Order("Alice", "Electronics", 1200.0, 2),
            new Order("Bob", "Books", 45.0, 3),
            new Order("Charlie", "Electronics", 800.0, 1),
            new Order("Diana", "Clothing", 150.0, 4),
            new Order("Eve", "Books", 75.0, 2),
            new Order("Frank", "Electronics", 2000.0, 1),
            new Order("Grace", "Clothing", 200.0, 3)
        );
        
        // Complex pipeline: High-value electronics orders by repeat customers
        List<OrderSummary> highValueElectronics = orders.stream()
            .filter(order -> "Electronics".equals(order.getCategory()))
            .filter(order -> order.getAmount() > 1000.0)
            .filter(order -> order.getQuantity() >= 1)
            .map(order -> new OrderSummary(
                order.getCustomer(),
                order.getAmount(),
                order.getAmount() * 1.1)) // Add 10% tax
            .sorted(Comparator.comparing(OrderSummary::getTotalWithTax).reversed())
            .collect(Collectors.toList());
        
        System.out.println("High-value electronics orders:");
        highValueElectronics.forEach(System.out::println);
        
        // Another complex pipeline: Customer spending analysis
        Map<String, Double> customerTotals = orders.stream()
            .filter(order -> order.getAmount() > 50.0)  // Minimum order amount
            .collect(Collectors.groupingBy(
                Order::getCustomer,
                Collectors.summingDouble(Order::getAmount)));
        
        List<CustomerSpending> topSpenders = customerTotals.entrySet().stream()
            .map(entry -> new CustomerSpending(entry.getKey(), entry.getValue()))
            .sorted(Comparator.comparing(CustomerSpending::getTotalSpent).reversed())
            .limit(3)
            .collect(Collectors.toList());
        
        System.out.println("\nTop 3 spenders:");
        topSpenders.forEach(System.out::println);
        
        // Category analysis pipeline
        Map<String, Long> categoryPopularity = orders.stream()
            .filter(order -> order.getQuantity() > 1)  // Multiple items
            .collect(Collectors.groupingBy(
                Order::getCategory,
                Collectors.counting()));
        
        System.out.println("\nCategory popularity (multi-item orders): " + categoryPopularity);
    }
    
    static class Order {
        private String customer;
        private String category;
        private double amount;
        private int quantity;
        
        public Order(String customer, String category, double amount, int quantity) {
            this.customer = customer;
            this.category = category;
            this.amount = amount;
            this.quantity = quantity;
        }
        
        public String getCustomer() { return customer; }
        public String getCategory() { return category; }
        public double getAmount() { return amount; }
        public int getQuantity() { return quantity; }
        
        @Override
        public String toString() {
            return String.format("%s: %s $%.2f (qty: %d)", 
                customer, category, amount, quantity);
        }
    }
    
    static class OrderSummary {
        private String customer;
        private double amount;
        private double totalWithTax;
        
        public OrderSummary(String customer, double amount, double totalWithTax) {
            this.customer = customer;
            this.amount = amount;
            this.totalWithTax = totalWithTax;
        }
        
        public String getCustomer() { return customer; }
        public double getAmount() { return amount; }
        public double getTotalWithTax() { return totalWithTax; }
        
        @Override
        public String toString() {
            return String.format("%s: $%.2f (with tax: $%.2f)", 
                customer, amount, totalWithTax);
        }
    }
    
    static class CustomerSpending {
        private String customer;
        private double totalSpent;
        
        public CustomerSpending(String customer, double totalSpent) {
            this.customer = customer;
            this.totalSpent = totalSpent;
        }
        
        public String getCustomer() { return customer; }
        public double getTotalSpent() { return totalSpent; }
        
        @Override
        public String toString() {
            return String.format("%s: $%.2f", customer, totalSpent);
        }
    }
}
```

## Summary

### Intermediate Operations Characteristics

1. **Lazy Evaluation**: Not executed until terminal operation
2. **Chainable**: Can be linked together to form processing pipelines
3. **Return Streams**: Allow further operations
4. **Stateless vs Stateful**: Some operations (like `sorted()`) require seeing all elements

### Performance Considerations

- **Order Matters**: Place cheap operations (like `filter()`) before expensive ones (like `sorted()`)
- **Short-circuiting**: Use operations like `limit()` to avoid processing unnecessary elements
- **Avoid Side Effects**: Keep operations pure for predictable behavior
- **Consider Parallel Streams**: For CPU-intensive operations on large datasets

### Best Practices

1. **Filter Early**: Place `filter()` operations as early as possible
2. **Use Method References**: More readable than lambda expressions when possible
3. **Chain Logically**: Organize operations in logical order
4. **Avoid Unnecessary Boxing**: Use primitive streams when appropriate
5. **Consider Readability**: Break complex pipelines into smaller, named operations

Intermediate operations provide the building blocks for powerful data processing pipelines, enabling elegant and efficient data transformations in Java applications.
