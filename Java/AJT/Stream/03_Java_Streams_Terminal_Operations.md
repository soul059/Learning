# Java Streams - Terminal Operations

## 1. Collection Operations

Terminal operations that collect stream elements into collections or other data structures.

### Basic Collection with collect()
```java
import java.util.*;
import java.util.stream.Collectors;

public class CollectOperationsDemo {
    public static void main(String[] args) {
        List<String> words = Arrays.asList("apple", "banana", "cherry", "date", "elderberry");
        
        // Collect to List
        List<String> filteredList = words.stream()
            .filter(word -> word.length() > 5)
            .collect(Collectors.toList());
        System.out.println("Filtered to List: " + filteredList);
        
        // Collect to Set (removes duplicates)
        List<String> wordsWithDuplicates = Arrays.asList("apple", "banana", "apple", "cherry", "banana");
        Set<String> uniqueWords = wordsWithDuplicates.stream()
            .collect(Collectors.toSet());
        System.out.println("Unique words in Set: " + uniqueWords);
        
        // Collect to specific List implementation
        LinkedList<String> linkedList = words.stream()
            .filter(word -> word.startsWith("a") || word.startsWith("e"))
            .collect(Collectors.toCollection(LinkedList::new));
        System.out.println("LinkedList: " + linkedList);
        
        // Collect to TreeSet (sorted)
        TreeSet<String> sortedSet = words.stream()
            .collect(Collectors.toCollection(TreeSet::new));
        System.out.println("TreeSet (sorted): " + sortedSet);
        
        // Collect to Array
        String[] wordsArray = words.stream()
            .filter(word -> word.length() <= 5)
            .toArray(String[]::new);
        System.out.println("Array: " + Arrays.toString(wordsArray));
    }
}
```

### Advanced Collection Operations
```java
import java.util.*;
import java.util.stream.Collectors;

public class AdvancedCollectDemo {
    public static void main(String[] args) {
        List<Employee> employees = Arrays.asList(
            new Employee("Alice", "Engineering", 75000, 25),
            new Employee("Bob", "Marketing", 60000, 30),
            new Employee("Charlie", "Engineering", 85000, 35),
            new Employee("Diana", "HR", 55000, 28),
            new Employee("Eve", "Engineering", 90000, 32)
        );
        
        // Collect to Map - name to salary
        Map<String, Double> nameSalaryMap = employees.stream()
            .collect(Collectors.toMap(
                Employee::getName,
                Employee::getSalary));
        System.out.println("Name-Salary Map: " + nameSalaryMap);
        
        // Collect to Map with custom value transformation
        Map<String, String> nameInfoMap = employees.stream()
            .collect(Collectors.toMap(
                Employee::getName,
                emp -> emp.getDepartment() + " - $" + emp.getSalary()));
        System.out.println("Name-Info Map: " + nameInfoMap);
        
        // Group by department
        Map<String, List<Employee>> byDepartment = employees.stream()
            .collect(Collectors.groupingBy(Employee::getDepartment));
        System.out.println("Grouped by Department:");
        byDepartment.forEach((dept, empList) -> {
            System.out.println("  " + dept + ": " + empList.size() + " employees");
        });
        
        // Group by department and collect names only
        Map<String, List<String>> departmentNames = employees.stream()
            .collect(Collectors.groupingBy(
                Employee::getDepartment,
                Collectors.mapping(Employee::getName, Collectors.toList())));
        System.out.println("Department Names: " + departmentNames);
        
        // Group by department and calculate average salary
        Map<String, Double> departmentAvgSalary = employees.stream()
            .collect(Collectors.groupingBy(
                Employee::getDepartment,
                Collectors.averagingDouble(Employee::getSalary)));
        System.out.println("Average Salary by Department: " + departmentAvgSalary);
        
        // Partition by condition (salary > 70000)
        Map<Boolean, List<Employee>> partitioned = employees.stream()
            .collect(Collectors.partitioningBy(emp -> emp.getSalary() > 70000));
        System.out.println("High earners: " + partitioned.get(true).size());
        System.out.println("Regular earners: " + partitioned.get(false).size());
    }
    
    static class Employee {
        private String name;
        private String department;
        private double salary;
        private int age;
        
        public Employee(String name, String department, double salary, int age) {
            this.name = name;
            this.department = department;
            this.salary = salary;
            this.age = age;
        }
        
        public String getName() { return name; }
        public String getDepartment() { return department; }
        public double getSalary() { return salary; }
        public int getAge() { return age; }
        
        @Override
        public String toString() {
            return name + "(" + department + ", $" + salary + ")";
        }
    }
}
```

## 2. Reduction Operations

Terminal operations that combine stream elements into a single result.

### Basic Reduction with reduce()
```java
import java.util.*;
import java.util.stream.IntStream;

public class ReduceOperationsDemo {
    public static void main(String[] args) {
        List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        
        // Sum using reduce
        Optional<Integer> sum = numbers.stream()
            .reduce((a, b) -> a + b);
        System.out.println("Sum: " + sum.orElse(0));
        
        // Sum with initial value
        Integer sumWithIdentity = numbers.stream()
            .reduce(0, (a, b) -> a + b);
        System.out.println("Sum with identity: " + sumWithIdentity);
        
        // Product
        Optional<Integer> product = numbers.stream()
            .reduce((a, b) -> a * b);
        System.out.println("Product: " + product.orElse(1));
        
        // Maximum value
        Optional<Integer> max = numbers.stream()
            .reduce(Integer::max);
        System.out.println("Maximum: " + max.orElse(0));
        
        // Minimum value
        Optional<Integer> min = numbers.stream()
            .reduce(Integer::min);
        System.out.println("Minimum: " + min.orElse(0));
        
        // String concatenation
        List<String> words = Arrays.asList("Hello", "World", "Java", "Streams");
        String concatenated = words.stream()
            .reduce("", (a, b) -> a + " " + b);
        System.out.println("Concatenated: " + concatenated.trim());
        
        // Complex reduction - finding longest word
        Optional<String> longestWord = words.stream()
            .reduce((w1, w2) -> w1.length() > w2.length() ? w1 : w2);
        System.out.println("Longest word: " + longestWord.orElse("None"));
    }
}
```

### Advanced Reduction Examples
```java
import java.util.*;

public class AdvancedReduceDemo {
    public static void main(String[] args) {
        List<Employee> employees = Arrays.asList(
            new Employee("Alice", "Engineering", 75000),
            new Employee("Bob", "Marketing", 60000),
            new Employee("Charlie", "Engineering", 85000),
            new Employee("Diana", "HR", 55000),
            new Employee("Eve", "Engineering", 90000)
        );
        
        // Find highest paid employee
        Optional<Employee> highestPaid = employees.stream()
            .reduce((e1, e2) -> e1.getSalary() > e2.getSalary() ? e1 : e2);
        System.out.println("Highest paid: " + highestPaid.orElse(null));
        
        // Calculate total salary
        double totalSalary = employees.stream()
            .map(Employee::getSalary)
            .reduce(0.0, Double::sum);
        System.out.println("Total salary: $" + totalSalary);
        
        // Combine employee names
        String allNames = employees.stream()
            .map(Employee::getName)
            .reduce("", (names, name) -> 
                names.isEmpty() ? name : names + ", " + name);
        System.out.println("All names: " + allNames);
        
        // Complex object reduction - create summary
        EmployeeSummary summary = employees.stream()
            .reduce(
                new EmployeeSummary(0, 0.0, ""),
                (sum, emp) -> new EmployeeSummary(
                    sum.getCount() + 1,
                    sum.getTotalSalary() + emp.getSalary(),
                    sum.getNames().isEmpty() ? emp.getName() : sum.getNames() + ", " + emp.getName()
                ),
                (sum1, sum2) -> new EmployeeSummary(
                    sum1.getCount() + sum2.getCount(),
                    sum1.getTotalSalary() + sum2.getTotalSalary(),
                    sum1.getNames() + ", " + sum2.getNames()
                )
            );
        
        System.out.println("Summary: " + summary);
        System.out.println("Average salary: $" + (summary.getTotalSalary() / summary.getCount()));
    }
    
    static class Employee {
        private String name;
        private String department;
        private double salary;
        
        public Employee(String name, String department, double salary) {
            this.name = name;
            this.department = department;
            this.salary = salary;
        }
        
        public String getName() { return name; }
        public String getDepartment() { return department; }
        public double getSalary() { return salary; }
        
        @Override
        public String toString() {
            return name + "(" + department + ", $" + salary + ")";
        }
    }
    
    static class EmployeeSummary {
        private int count;
        private double totalSalary;
        private String names;
        
        public EmployeeSummary(int count, double totalSalary, String names) {
            this.count = count;
            this.totalSalary = totalSalary;
            this.names = names;
        }
        
        public int getCount() { return count; }
        public double getTotalSalary() { return totalSalary; }
        public String getNames() { return names; }
        
        @Override
        public String toString() {
            return String.format("Count: %d, Total Salary: $%.2f, Names: %s", 
                count, totalSalary, names);
        }
    }
}
```

## 3. Finding Operations

Terminal operations for finding specific elements in the stream.

### Find Operations
```java
import java.util.*;

public class FindOperationsDemo {
    public static void main(String[] args) {
        List<String> words = Arrays.asList("apple", "banana", "cherry", "date", "elderberry");
        List<Integer> numbers = Arrays.asList(1, 3, 5, 8, 9, 12, 15);
        
        // findFirst - returns first element (if any)
        Optional<String> firstWord = words.stream()
            .filter(word -> word.startsWith("c"))
            .findFirst();
        System.out.println("First word starting with 'c': " + firstWord.orElse("Not found"));
        
        // findAny - returns any element (useful for parallel streams)
        Optional<String> anyLongWord = words.stream()
            .filter(word -> word.length() > 6)
            .findAny();
        System.out.println("Any long word: " + anyLongWord.orElse("Not found"));
        
        // Find first even number
        Optional<Integer> firstEven = numbers.stream()
            .filter(n -> n % 2 == 0)
            .findFirst();
        System.out.println("First even number: " + firstEven.orElse(-1));
        
        // Find any number greater than 10
        Optional<Integer> anyGreaterThan10 = numbers.stream()
            .filter(n -> n > 10)
            .findAny();
        System.out.println("Any number > 10: " + anyGreaterThan10.orElse(-1));
        
        // Working with Optional results
        words.stream()
            .filter(word -> word.length() > 10)
            .findFirst()
            .ifPresent(word -> System.out.println("Found very long word: " + word));
        
        // Using orElse with findFirst
        String result = words.stream()
            .filter(word -> word.startsWith("z"))
            .findFirst()
            .orElse("No word found starting with 'z'");
        System.out.println(result);
        
        // Using orElseThrow
        try {
            String mandatoryResult = words.stream()
                .filter(word -> word.startsWith("z"))
                .findFirst()
                .orElseThrow(() -> new RuntimeException("Required word not found"));
        } catch (RuntimeException e) {
            System.out.println("Exception caught: " + e.getMessage());
        }
    }
}
```

### Complex Finding with Objects
```java
import java.util.*;

public class ComplexFindDemo {
    public static void main(String[] args) {
        List<Product> products = Arrays.asList(
            new Product("Laptop", "Electronics", 1200.0, 5),
            new Product("Book", "Education", 25.0, 50),
            new Product("Headphones", "Electronics", 150.0, 20),
            new Product("Desk", "Furniture", 300.0, 8),
            new Product("Pen", "Education", 2.0, 100)
        );
        
        // Find first expensive product
        Optional<Product> expensiveProduct = products.stream()
            .filter(p -> p.getPrice() > 500.0)
            .findFirst();
        
        expensiveProduct.ifPresent(p -> 
            System.out.println("First expensive product: " + p));
        
        // Find any low-stock item
        Optional<Product> lowStockItem = products.stream()
            .filter(p -> p.getStock() < 10)
            .findAny();
        
        System.out.println("Low stock item: " + lowStockItem.orElse(null));
        
        // Find product by exact name
        String searchName = "Book";
        Optional<Product> foundProduct = products.stream()
            .filter(p -> searchName.equals(p.getName()))
            .findFirst();
        
        foundProduct.ifPresentOrElse(
            p -> System.out.println("Found: " + p),
            () -> System.out.println("Product '" + searchName + "' not found")
        );
        
        // Find cheapest product in category
        String category = "Electronics";
        Optional<Product> cheapestInCategory = products.stream()
            .filter(p -> category.equals(p.getCategory()))
            .min(Comparator.comparing(Product::getPrice));
        
        System.out.println("Cheapest in " + category + ": " + 
            cheapestInCategory.orElse(null));
        
        // Find product with highest stock
        Optional<Product> highestStock = products.stream()
            .max(Comparator.comparing(Product::getStock));
        
        highestStock.ifPresent(p -> 
            System.out.println("Highest stock: " + p));
    }
    
    static class Product {
        private String name;
        private String category;
        private double price;
        private int stock;
        
        public Product(String name, String category, double price, int stock) {
            this.name = name;
            this.category = category;
            this.price = price;
            this.stock = stock;
        }
        
        public String getName() { return name; }
        public String getCategory() { return category; }
        public double getPrice() { return price; }
        public int getStock() { return stock; }
        
        @Override
        public String toString() {
            return String.format("%s (%s) - $%.2f (Stock: %d)", 
                name, category, price, stock);
        }
    }
}
```

## 4. Matching Operations

Terminal operations that test whether stream elements match certain conditions.

### All Match, Any Match, None Match
```java
import java.util.*;

public class MatchingOperationsDemo {
    public static void main(String[] args) {
        List<Integer> numbers = Arrays.asList(2, 4, 6, 8, 10);
        List<Integer> mixedNumbers = Arrays.asList(1, 2, 3, 4, 5);
        List<String> words = Arrays.asList("hello", "world", "java", "programming");
        
        // allMatch - all elements must satisfy the predicate
        boolean allEven = numbers.stream()
            .allMatch(n -> n % 2 == 0);
        System.out.println("All numbers are even: " + allEven);
        
        boolean allPositive = mixedNumbers.stream()
            .allMatch(n -> n > 0);
        System.out.println("All numbers are positive: " + allPositive);
        
        boolean allLongWords = words.stream()
            .allMatch(word -> word.length() > 10);
        System.out.println("All words are long: " + allLongWords);
        
        // anyMatch - at least one element must satisfy the predicate
        boolean anyEven = mixedNumbers.stream()
            .anyMatch(n -> n % 2 == 0);
        System.out.println("Any number is even: " + anyEven);
        
        boolean anyLongWord = words.stream()
            .anyMatch(word -> word.length() > 6);
        System.out.println("Any word is long: " + anyLongWord);
        
        // noneMatch - no element should satisfy the predicate
        boolean noneNegative = numbers.stream()
            .noneMatch(n -> n < 0);
        System.out.println("No negative numbers: " + noneNegative);
        
        boolean noneStartsWithZ = words.stream()
            .noneMatch(word -> word.startsWith("z"));
        System.out.println("No word starts with 'z': " + noneStartsWithZ);
        
        // Empty stream behavior
        List<Integer> emptyList = Arrays.asList();
        System.out.println("Empty stream - allMatch: " + 
            emptyList.stream().allMatch(n -> n > 0));  // true
        System.out.println("Empty stream - anyMatch: " + 
            emptyList.stream().anyMatch(n -> n > 0));  // false
        System.out.println("Empty stream - noneMatch: " + 
            emptyList.stream().noneMatch(n -> n > 0)); // true
    }
}
```

### Advanced Matching with Objects
```java
import java.util.*;

public class AdvancedMatchingDemo {
    public static void main(String[] args) {
        List<Employee> employees = Arrays.asList(
            new Employee("Alice", "Engineering", 75000, 25),
            new Employee("Bob", "Marketing", 60000, 30),
            new Employee("Charlie", "Engineering", 85000, 35),
            new Employee("Diana", "HR", 55000, 28),
            new Employee("Eve", "Engineering", 90000, 32)
        );
        
        // Check if all employees earn above minimum wage
        boolean allAboveMinimum = employees.stream()
            .allMatch(emp -> emp.getSalary() > 50000);
        System.out.println("All employees earn above $50k: " + allAboveMinimum);
        
        // Check if any employee is in HR
        boolean anyInHR = employees.stream()
            .anyMatch(emp -> "HR".equals(emp.getDepartment()));
        System.out.println("Any employee in HR: " + anyInHR);
        
        // Check if no employee is over 40
        boolean noneOver40 = employees.stream()
            .noneMatch(emp -> emp.getAge() > 40);
        System.out.println("No employee over 40: " + noneOver40);
        
        // Check if all engineers earn above average
        double averageSalary = employees.stream()
            .mapToDouble(Employee::getSalary)
            .average()
            .orElse(0.0);
        
        boolean allEngineersAboveAverage = employees.stream()
            .filter(emp -> "Engineering".equals(emp.getDepartment()))
            .allMatch(emp -> emp.getSalary() > averageSalary);
        
        System.out.println("Average salary: $" + averageSalary);
        System.out.println("All engineers above average: " + allEngineersAboveAverage);
        
        // Complex matching conditions
        boolean hasHighPaidYoungEmployee = employees.stream()
            .anyMatch(emp -> emp.getAge() < 30 && emp.getSalary() > 70000);
        System.out.println("Has high-paid young employee: " + hasHighPaidYoungEmployee);
        
        // Validation use case
        boolean isValidEmployeeData = employees.stream()
            .allMatch(emp -> 
                emp.getName() != null && !emp.getName().isEmpty() &&
                emp.getDepartment() != null && !emp.getDepartment().isEmpty() &&
                emp.getSalary() > 0 &&
                emp.getAge() > 0);
        System.out.println("All employee data is valid: " + isValidEmployeeData);
    }
    
    static class Employee {
        private String name;
        private String department;
        private double salary;
        private int age;
        
        public Employee(String name, String department, double salary, int age) {
            this.name = name;
            this.department = department;
            this.salary = salary;
            this.age = age;
        }
        
        public String getName() { return name; }
        public String getDepartment() { return department; }
        public double getSalary() { return salary; }
        public int getAge() { return age; }
        
        @Override
        public String toString() {
            return String.format("%s (%s, %d, $%.0f)", name, department, age, salary);
        }
    }
}
```

## 5. Aggregation Operations

Terminal operations that perform calculations on stream elements.

### Count, Min, Max
```java
import java.util.*;
import java.util.stream.IntStream;

public class AggregationOperationsDemo {
    public static void main(String[] args) {
        List<String> words = Arrays.asList("apple", "banana", "cherry", "date", "elderberry");
        List<Integer> numbers = Arrays.asList(5, 2, 8, 1, 9, 3, 7, 4, 6);
        
        // Count operations
        long totalWords = words.stream().count();
        System.out.println("Total words: " + totalWords);
        
        long longWords = words.stream()
            .filter(word -> word.length() > 5)
            .count();
        System.out.println("Long words: " + longWords);
        
        // Min and Max with Comparator
        Optional<String> shortestWord = words.stream()
            .min(Comparator.comparing(String::length));
        System.out.println("Shortest word: " + shortestWord.orElse("None"));
        
        Optional<String> longestWord = words.stream()
            .max(Comparator.comparing(String::length));
        System.out.println("Longest word: " + longestWord.orElse("None"));
        
        Optional<String> firstAlphabetically = words.stream()
            .min(Comparator.naturalOrder());
        System.out.println("First alphabetically: " + firstAlphabetically.orElse("None"));
        
        // Numeric operations
        Optional<Integer> minNumber = numbers.stream().min(Integer::compareTo);
        Optional<Integer> maxNumber = numbers.stream().max(Integer::compareTo);
        
        System.out.println("Min number: " + minNumber.orElse(0));
        System.out.println("Max number: " + maxNumber.orElse(0));
        
        // Specialized numeric stream operations
        IntStream intStream = numbers.stream().mapToInt(Integer::intValue);
        System.out.println("Sum: " + intStream.sum());
        
        OptionalDouble average = numbers.stream()
            .mapToInt(Integer::intValue)
            .average();
        System.out.println("Average: " + average.orElse(0.0));
        
        IntSummaryStatistics stats = numbers.stream()
            .mapToInt(Integer::intValue)
            .summaryStatistics();
        
        System.out.println("Statistics: " + stats);
        System.out.println("  Count: " + stats.getCount());
        System.out.println("  Sum: " + stats.getSum());
        System.out.println("  Min: " + stats.getMin());
        System.out.println("  Max: " + stats.getMax());
        System.out.println("  Average: " + stats.getAverage());
    }
}
```

### Advanced Aggregation Examples
```java
import java.util.*;
import java.util.stream.Collectors;

public class AdvancedAggregationDemo {
    public static void main(String[] args) {
        List<Sale> sales = Arrays.asList(
            new Sale("Alice", "Electronics", 1200.0),
            new Sale("Bob", "Books", 45.0),
            new Sale("Charlie", "Electronics", 800.0),
            new Sale("Diana", "Clothing", 150.0),
            new Sale("Eve", "Books", 75.0),
            new Sale("Frank", "Electronics", 2000.0),
            new Sale("Alice", "Clothing", 200.0)
        );
        
        // Total sales count
        long totalSales = sales.stream().count();
        System.out.println("Total sales: " + totalSales);
        
        // Highest sale amount
        Optional<Sale> highestSale = sales.stream()
            .max(Comparator.comparing(Sale::getAmount));
        System.out.println("Highest sale: " + highestSale.orElse(null));
        
        // Total revenue
        double totalRevenue = sales.stream()
            .mapToDouble(Sale::getAmount)
            .sum();
        System.out.println("Total revenue: $" + totalRevenue);
        
        // Average sale amount
        OptionalDouble averageSale = sales.stream()
            .mapToDouble(Sale::getAmount)
            .average();
        System.out.println("Average sale: $" + averageSale.orElse(0.0));
        
        // Sales by category
        Map<String, Long> salesByCategory = sales.stream()
            .collect(Collectors.groupingBy(
                Sale::getCategory,
                Collectors.counting()));
        System.out.println("Sales by category: " + salesByCategory);
        
        // Revenue by category
        Map<String, Double> revenueByCategory = sales.stream()
            .collect(Collectors.groupingBy(
                Sale::getCategory,
                Collectors.summingDouble(Sale::getAmount)));
        System.out.println("Revenue by category: " + revenueByCategory);
        
        // Top performer (by revenue)
        Optional<Map.Entry<String, Double>> topPerformer = sales.stream()
            .collect(Collectors.groupingBy(
                Sale::getSalesperson,
                Collectors.summingDouble(Sale::getAmount)))
            .entrySet()
            .stream()
            .max(Map.Entry.comparingByValue());
        
        topPerformer.ifPresent(entry -> 
            System.out.println("Top performer: " + entry.getKey() + 
                " with $" + entry.getValue()));
        
        // Category with highest average sale
        Optional<Map.Entry<String, Double>> highestAvgCategory = sales.stream()
            .collect(Collectors.groupingBy(
                Sale::getCategory,
                Collectors.averagingDouble(Sale::getAmount)))
            .entrySet()
            .stream()
            .max(Map.Entry.comparingByValue());
        
        highestAvgCategory.ifPresent(entry ->
            System.out.println("Highest avg category: " + entry.getKey() + 
                " with avg $" + String.format("%.2f", entry.getValue())));
        
        // Statistical summary
        DoubleSummaryStatistics salesStats = sales.stream()
            .mapToDouble(Sale::getAmount)
            .summaryStatistics();
        
        System.out.println("\nSales Statistics:");
        System.out.println("  Count: " + salesStats.getCount());
        System.out.println("  Total: $" + salesStats.getSum());
        System.out.println("  Average: $" + String.format("%.2f", salesStats.getAverage()));
        System.out.println("  Min: $" + salesStats.getMin());
        System.out.println("  Max: $" + salesStats.getMax());
    }
    
    static class Sale {
        private String salesperson;
        private String category;
        private double amount;
        
        public Sale(String salesperson, String category, double amount) {
            this.salesperson = salesperson;
            this.category = category;
            this.amount = amount;
        }
        
        public String getSalesperson() { return salesperson; }
        public String getCategory() { return category; }
        public double getAmount() { return amount; }
        
        @Override
        public String toString() {
            return String.format("%s sold %s for $%.2f", 
                salesperson, category, amount);
        }
    }
}
```

## 6. ForEach Operations

Terminal operations for performing actions on each stream element.

### Basic ForEach
```java
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

public class ForEachDemo {
    public static void main(String[] args) {
        List<String> words = Arrays.asList("apple", "banana", "cherry", "date");
        List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5);
        
        // Simple forEach - print each element
        System.out.println("Words:");
        words.stream().forEach(System.out::println);
        
        // forEach with custom action
        System.out.println("\nUppercase words:");
        words.stream()
            .map(String::toUpperCase)
            .forEach(word -> System.out.println("-> " + word));
        
        // forEach with side effects (be careful!)
        List<String> results = new ArrayList<>();
        words.stream()
            .filter(word -> word.length() > 5)
            .forEach(word -> results.add(word.toUpperCase()));
        System.out.println("Filtered results: " + results);
        
        // forEachOrdered - maintains order in parallel streams
        System.out.println("\nParallel forEach (order not guaranteed):");
        numbers.parallelStream()
            .forEach(n -> System.out.print(n + " "));
        
        System.out.println("\n\nParallel forEachOrdered (order guaranteed):");
        numbers.parallelStream()
            .forEachOrdered(n -> System.out.print(n + " "));
        
        // Complex forEach operations
        System.out.println("\n\nWord analysis:");
        AtomicInteger counter = new AtomicInteger(0);
        words.stream()
            .forEach(word -> {
                int index = counter.incrementAndGet();
                System.out.println(index + ". " + word + 
                    " (length: " + word.length() + ")");
            });
    }
}
```

### Practical ForEach Examples
```java
import java.util.*;
import java.io.PrintWriter;
import java.io.StringWriter;

public class PracticalForEachDemo {
    public static void main(String[] args) {
        List<Employee> employees = Arrays.asList(
            new Employee("Alice", "Engineering", 75000),
            new Employee("Bob", "Marketing", 60000),
            new Employee("Charlie", "Engineering", 85000),
            new Employee("Diana", "HR", 55000)
        );
        
        // Generate report
        System.out.println("Employee Report:");
        System.out.println("================");
        employees.stream()
            .sorted(Comparator.comparing(Employee::getDepartment)
                .thenComparing(Employee::getName))
            .forEach(emp -> System.out.printf("%-12s %-15s $%,.2f%n",
                emp.getName(), emp.getDepartment(), emp.getSalary()));
        
        // Conditional actions
        System.out.println("\nHigh earners (>70k):");
        employees.stream()
            .filter(emp -> emp.getSalary() > 70000)
            .forEach(emp -> System.out.println("🌟 " + emp.getName() + 
                " earns $" + emp.getSalary()));
        
        // Side effects for logging/debugging
        List<String> processLog = new ArrayList<>();
        double totalSalary = employees.stream()
            .peek(emp -> processLog.add("Processing: " + emp.getName()))
            .mapToDouble(Employee::getSalary)
            .peek(salary -> processLog.add("Salary: $" + salary))
            .sum();
        
        System.out.println("\nProcessing log:");
        processLog.forEach(System.out::println);
        System.out.println("Total salary: $" + totalSalary);
        
        // Building strings/output
        StringWriter output = new StringWriter();
        PrintWriter writer = new PrintWriter(output);
        
        employees.stream()
            .filter(emp -> "Engineering".equals(emp.getDepartment()))
            .forEach(emp -> writer.println("Engineer: " + emp.getName()));
        
        System.out.println("\nEngineers:");
        System.out.print(output.toString());
        
        // Updating external collections (careful with side effects)
        Map<String, List<String>> departmentEmployees = new HashMap<>();
        employees.stream()
            .forEach(emp -> {
                departmentEmployees.computeIfAbsent(emp.getDepartment(), 
                    k -> new ArrayList<>()).add(emp.getName());
            });
        
        System.out.println("Department grouping: " + departmentEmployees);
    }
    
    static class Employee {
        private String name;
        private String department;
        private double salary;
        
        public Employee(String name, String department, double salary) {
            this.name = name;
            this.department = department;
            this.salary = salary;
        }
        
        public String getName() { return name; }
        public String getDepartment() { return department; }
        public double getSalary() { return salary; }
        
        @Override
        public String toString() {
            return name + " (" + department + ", $" + salary + ")";
        }
    }
}
```

## Summary

### Terminal Operations Categories

1. **Collection Operations**: `collect()`, `toArray()`
2. **Reduction Operations**: `reduce()`, `sum()`, `min()`, `max()`
3. **Search Operations**: `findFirst()`, `findAny()`
4. **Matching Operations**: `allMatch()`, `anyMatch()`, `noneMatch()`
5. **Aggregation Operations**: `count()`, statistical operations
6. **Iteration Operations**: `forEach()`, `forEachOrdered()`

### Key Characteristics

- **Eager Evaluation**: Execute immediately and trigger the entire stream pipeline
- **Consume Stream**: After terminal operation, stream cannot be reused
- **Return Results**: Produce final values, collections, or perform side effects
- **Short-circuiting**: Some operations (`findFirst`, `anyMatch`) can terminate early

### Best Practices

1. **Choose Right Operation**: Use specific terminal operations rather than generic ones
2. **Handle Optionals**: Properly handle `Optional` results from operations like `findFirst()`
3. **Avoid Side Effects**: Prefer collecting results over modifying external state
4. **Consider Performance**: Some operations are more efficient than others
5. **Use Parallel Carefully**: Some operations benefit more from parallelization than others

### Performance Tips

- **Short-circuiting**: Use `findFirst()`, `anyMatch()` when you don't need all results
- **Primitive Streams**: Use `IntStream`, `LongStream`, `DoubleStream` for numeric operations
- **Specialized Collectors**: Use appropriate collectors for better performance
- **Parallel Streams**: Consider for CPU-intensive operations on large datasets

Terminal operations are the culmination of stream processing, converting the lazy intermediate operations into concrete results or actions.
