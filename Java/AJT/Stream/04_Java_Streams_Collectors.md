# Java Streams - Collectors and Advanced Collection Operations

## 1. Understanding Collectors

Collectors are used with the `collect()` terminal operation to perform mutable reduction operations on streams. They provide powerful ways to accumulate stream elements into collections, strings, or other data structures.

### Basic Built-in Collectors
```java
import java.util.*;
import java.util.stream.Collectors;

public class BasicCollectorsDemo {
    public static void main(String[] args) {
        List<String> words = Arrays.asList("apple", "banana", "cherry", "date", "elderberry");
        List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        
        // Collect to List
        List<String> longWords = words.stream()
            .filter(word -> word.length() > 5)
            .collect(Collectors.toList());
        System.out.println("Long words (List): " + longWords);
        
        // Collect to Set
        Set<Integer> evenNumbers = numbers.stream()
            .filter(n -> n % 2 == 0)
            .collect(Collectors.toSet());
        System.out.println("Even numbers (Set): " + evenNumbers);
        
        // Collect to specific collection type
        LinkedList<String> linkedList = words.stream()
            .collect(Collectors.toCollection(LinkedList::new));
        System.out.println("LinkedList: " + linkedList);
        
        TreeSet<String> treeSet = words.stream()
            .collect(Collectors.toCollection(TreeSet::new));
        System.out.println("TreeSet (sorted): " + treeSet);
        
        // Collect to Array
        String[] wordArray = words.stream()
            .filter(word -> word.startsWith("a") || word.startsWith("e"))
            .toArray(String[]::new);
        System.out.println("Array: " + Arrays.toString(wordArray));
        
        // Joining strings
        String joinedWords = words.stream()
            .collect(Collectors.joining());
        System.out.println("Joined: " + joinedWords);
        
        String joinedWithDelimiter = words.stream()
            .collect(Collectors.joining(", "));
        System.out.println("Joined with delimiter: " + joinedWithDelimiter);
        
        String joinedWithPrefixSuffix = words.stream()
            .collect(Collectors.joining(", ", "[", "]"));
        System.out.println("Joined with prefix/suffix: " + joinedWithPrefixSuffix);
    }
}
```

### Map Collectors
```java
import java.util.*;
import java.util.stream.Collectors;
import java.util.function.Function;

public class MapCollectorsDemo {
    public static void main(String[] args) {
        List<Person> people = Arrays.asList(
            new Person("Alice", 25, "Engineering"),
            new Person("Bob", 30, "Marketing"),
            new Person("Charlie", 35, "Engineering"),
            new Person("Diana", 28, "HR"),
            new Person("Eve", 32, "Marketing")
        );
        
        // Basic toMap - key to value
        Map<String, Integer> nameToAge = people.stream()
            .collect(Collectors.toMap(
                Person::getName,
                Person::getAge));
        System.out.println("Name to Age: " + nameToAge);
        
        // toMap with value transformation
        Map<String, String> nameToInfo = people.stream()
            .collect(Collectors.toMap(
                Person::getName,
                person -> person.getDepartment() + " (" + person.getAge() + ")"));
        System.out.println("Name to Info: " + nameToInfo);
        
        // toMap with duplicate key handling
        List<Person> peopleWithDuplicates = Arrays.asList(
            new Person("Alice", 25, "Engineering"),
            new Person("Bob", 30, "Marketing"),
            new Person("Alice", 26, "Engineering")  // Duplicate name
        );
        
        Map<String, Integer> nameToAgeHandled = peopleWithDuplicates.stream()
            .collect(Collectors.toMap(
                Person::getName,
                Person::getAge,
                Integer::max));  // Keep the maximum age for duplicates
        System.out.println("Name to Age (max): " + nameToAgeHandled);
        
        // toMap with specific Map implementation
        LinkedHashMap<String, String> orderedMap = people.stream()
            .collect(Collectors.toMap(
                Person::getName,
                Person::getDepartment,
                (existing, replacement) -> existing,  // Keep first
                LinkedHashMap::new));
        System.out.println("Ordered Map: " + orderedMap);
        
        // Create Map with identity as key
        Map<String, Person> nameToPersonMap = people.stream()
            .collect(Collectors.toMap(
                Person::getName,
                Function.identity()));
        System.out.println("Name to Person: " + nameToPersonMap);
        
        // Complex key generation
        Map<String, Person> deptAgeKey = people.stream()
            .collect(Collectors.toMap(
                person -> person.getDepartment() + "_" + person.getAge(),
                Function.identity()));
        System.out.println("Department_Age to Person: " + deptAgeKey);
    }
    
    static class Person {
        private String name;
        private int age;
        private String department;
        
        public Person(String name, int age, String department) {
            this.name = name;
            this.age = age;
            this.department = department;
        }
        
        public String getName() { return name; }
        public int getAge() { return age; }
        public String getDepartment() { return department; }
        
        @Override
        public String toString() {
            return name + "(" + age + ", " + department + ")";
        }
    }
}
```

## 2. Grouping Collectors

Grouping is one of the most powerful features of collectors, allowing you to categorize stream elements.

### Basic Grouping
```java
import java.util.*;
import java.util.stream.Collectors;

public class GroupingDemo {
    public static void main(String[] args) {
        List<Employee> employees = Arrays.asList(
            new Employee("Alice", "Engineering", 75000, 25),
            new Employee("Bob", "Marketing", 60000, 30),
            new Employee("Charlie", "Engineering", 85000, 35),
            new Employee("Diana", "HR", 55000, 28),
            new Employee("Eve", "Engineering", 90000, 32),
            new Employee("Frank", "Marketing", 70000, 40)
        );
        
        // Basic grouping by department
        Map<String, List<Employee>> byDepartment = employees.stream()
            .collect(Collectors.groupingBy(Employee::getDepartment));
        
        System.out.println("Grouped by Department:");
        byDepartment.forEach((dept, empList) -> {
            System.out.println("  " + dept + ":");
            empList.forEach(emp -> System.out.println("    " + emp));
        });
        
        // Grouping by age range
        Map<String, List<Employee>> byAgeRange = employees.stream()
            .collect(Collectors.groupingBy(emp -> {
                if (emp.getAge() < 30) return "Young";
                else if (emp.getAge() < 40) return "Middle";
                else return "Senior";
            }));
        
        System.out.println("\nGrouped by Age Range:");
        byAgeRange.forEach((range, empList) -> {
            System.out.println("  " + range + ": " + empList.size() + " employees");
        });
        
        // Grouping by salary bracket
        Map<String, List<Employee>> bySalaryBracket = employees.stream()
            .collect(Collectors.groupingBy(emp -> {
                if (emp.getSalary() < 60000) return "Low";
                else if (emp.getSalary() < 80000) return "Medium";
                else return "High";
            }));
        
        System.out.println("\nGrouped by Salary Bracket:");
        bySalaryBracket.forEach((bracket, empList) -> {
            System.out.println("  " + bracket + " (" + empList.size() + "): " + 
                empList.stream().map(Employee::getName).collect(Collectors.joining(", ")));
        });
        
        // Grouping with specific Map type
        LinkedHashMap<String, List<Employee>> orderedGrouping = employees.stream()
            .collect(Collectors.groupingBy(
                Employee::getDepartment,
                LinkedHashMap::new,
                Collectors.toList()));
        
        System.out.println("\nOrdered Grouping: " + orderedGrouping.keySet());
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
            return String.format("%s (age %d, $%.0f)", name, age, salary);
        }
    }
}
```

### Advanced Grouping with Downstream Collectors
```java
import java.util.*;
import java.util.stream.Collectors;

public class AdvancedGroupingDemo {
    public static void main(String[] args) {
        List<Sale> sales = Arrays.asList(
            new Sale("Alice", "Electronics", 1200.0, "Q1"),
            new Sale("Bob", "Books", 45.0, "Q1"),
            new Sale("Charlie", "Electronics", 800.0, "Q1"),
            new Sale("Alice", "Clothing", 150.0, "Q1"),
            new Sale("Diana", "Books", 75.0, "Q2"),
            new Sale("Eve", "Electronics", 2000.0, "Q2"),
            new Sale("Frank", "Clothing", 300.0, "Q2"),
            new Sale("Alice", "Electronics", 1500.0, "Q2")
        );
        
        // Group by category and count
        Map<String, Long> categoryCount = sales.stream()
            .collect(Collectors.groupingBy(
                Sale::getCategory,
                Collectors.counting()));
        System.out.println("Sales count by category: " + categoryCount);
        
        // Group by category and sum amounts
        Map<String, Double> categoryRevenue = sales.stream()
            .collect(Collectors.groupingBy(
                Sale::getCategory,
                Collectors.summingDouble(Sale::getAmount)));
        System.out.println("Revenue by category: " + categoryRevenue);
        
        // Group by category and get average amount
        Map<String, Double> categoryAverage = sales.stream()
            .collect(Collectors.groupingBy(
                Sale::getCategory,
                Collectors.averagingDouble(Sale::getAmount)));
        System.out.println("Average sale by category: " + categoryAverage);
        
        // Group by category and collect salesperson names
        Map<String, Set<String>> categorySalespeople = sales.stream()
            .collect(Collectors.groupingBy(
                Sale::getCategory,
                Collectors.mapping(Sale::getSalesperson, Collectors.toSet())));
        System.out.println("Salespeople by category: " + categorySalespeople);
        
        // Group by category and get statistics
        Map<String, DoubleSummaryStatistics> categoryStats = sales.stream()
            .collect(Collectors.groupingBy(
                Sale::getCategory,
                Collectors.summarizingDouble(Sale::getAmount)));
        
        System.out.println("\nCategory Statistics:");
        categoryStats.forEach((category, stats) -> {
            System.out.printf("%s: count=%d, sum=%.2f, avg=%.2f, min=%.2f, max=%.2f%n",
                category, stats.getCount(), stats.getSum(), stats.getAverage(),
                stats.getMin(), stats.getMax());
        });
        
        // Group by salesperson and find their best sale
        Map<String, Optional<Sale>> bestSaleByPerson = sales.stream()
            .collect(Collectors.groupingBy(
                Sale::getSalesperson,
                Collectors.maxBy(Comparator.comparing(Sale::getAmount))));
        
        System.out.println("\nBest sale by person:");
        bestSaleByPerson.forEach((person, sale) -> {
            System.out.println("  " + person + ": " + sale.orElse(null));
        });
        
        // Group by quarter and category (nested grouping)
        Map<String, Map<String, Double>> quarterCategoryRevenue = sales.stream()
            .collect(Collectors.groupingBy(
                Sale::getQuarter,
                Collectors.groupingBy(
                    Sale::getCategory,
                    Collectors.summingDouble(Sale::getAmount))));
        
        System.out.println("\nRevenue by Quarter and Category:");
        quarterCategoryRevenue.forEach((quarter, categories) -> {
            System.out.println("  " + quarter + ":");
            categories.forEach((category, revenue) -> {
                System.out.printf("    %s: $%.2f%n", category, revenue);
            });
        });
    }
    
    static class Sale {
        private String salesperson;
        private String category;
        private double amount;
        private String quarter;
        
        public Sale(String salesperson, String category, double amount, String quarter) {
            this.salesperson = salesperson;
            this.category = category;
            this.amount = amount;
            this.quarter = quarter;
        }
        
        public String getSalesperson() { return salesperson; }
        public String getCategory() { return category; }
        public double getAmount() { return amount; }
        public String getQuarter() { return quarter; }
        
        @Override
        public String toString() {
            return String.format("%s: %s $%.2f (%s)", 
                salesperson, category, amount, quarter);
        }
    }
}
```

## 3. Partitioning Collectors

Partitioning is a special case of grouping where elements are divided into two groups based on a predicate.

### Basic Partitioning
```java
import java.util.*;
import java.util.stream.Collectors;

public class PartitioningDemo {
    public static void main(String[] args) {
        List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        List<String> words = Arrays.asList("apple", "banana", "cherry", "date", "elderberry");
        
        // Partition numbers into even and odd
        Map<Boolean, List<Integer>> evenOddPartition = numbers.stream()
            .collect(Collectors.partitioningBy(n -> n % 2 == 0));
        
        System.out.println("Even numbers: " + evenOddPartition.get(true));
        System.out.println("Odd numbers: " + evenOddPartition.get(false));
        
        // Partition words by length
        Map<Boolean, List<String>> wordLengthPartition = words.stream()
            .collect(Collectors.partitioningBy(word -> word.length() > 5));
        
        System.out.println("Long words: " + wordLengthPartition.get(true));
        System.out.println("Short words: " + wordLengthPartition.get(false));
        
        // Partition with downstream collector - count
        Map<Boolean, Long> evenOddCount = numbers.stream()
            .collect(Collectors.partitioningBy(
                n -> n % 2 == 0,
                Collectors.counting()));
        
        System.out.println("Even count: " + evenOddCount.get(true));
        System.out.println("Odd count: " + evenOddCount.get(false));
        
        // Partition with downstream collector - sum
        Map<Boolean, Integer> evenOddSum = numbers.stream()
            .collect(Collectors.partitioningBy(
                n -> n % 2 == 0,
                Collectors.summingInt(Integer::intValue)));
        
        System.out.println("Even sum: " + evenOddSum.get(true));
        System.out.println("Odd sum: " + evenOddSum.get(false));
    }
}
```

### Advanced Partitioning Examples
```java
import java.util.*;
import java.util.stream.Collectors;

public class AdvancedPartitioningDemo {
    public static void main(String[] args) {
        List<Student> students = Arrays.asList(
            new Student("Alice", 85, "Computer Science"),
            new Student("Bob", 72, "Mathematics"),
            new Student("Charlie", 95, "Computer Science"),
            new Student("Diana", 68, "Physics"),
            new Student("Eve", 88, "Mathematics"),
            new Student("Frank", 91, "Computer Science"),
            new Student("Grace", 77, "Physics")
        );
        
        // Partition by passing grade (>= 75)
        Map<Boolean, List<Student>> passingGrades = students.stream()
            .collect(Collectors.partitioningBy(student -> student.getGrade() >= 75));
        
        System.out.println("Passing students:");
        passingGrades.get(true).forEach(System.out::println);
        
        System.out.println("\nFailing students:");
        passingGrades.get(false).forEach(System.out::println);
        
        // Partition and get statistics
        Map<Boolean, DoubleSummaryStatistics> gradeStats = students.stream()
            .collect(Collectors.partitioningBy(
                student -> student.getGrade() >= 75,
                Collectors.summarizingDouble(Student::getGrade)));
        
        System.out.println("\nPassing students grade stats: " + gradeStats.get(true));
        System.out.println("Failing students grade stats: " + gradeStats.get(false));
        
        // Partition and get best student in each group
        Map<Boolean, Optional<Student>> bestInGroup = students.stream()
            .collect(Collectors.partitioningBy(
                student -> student.getGrade() >= 75,
                Collectors.maxBy(Comparator.comparing(Student::getGrade))));
        
        System.out.println("\nBest passing student: " + bestInGroup.get(true).orElse(null));
        System.out.println("Best failing student: " + bestInGroup.get(false).orElse(null));
        
        // Partition by major (CS vs non-CS)
        Map<Boolean, List<String>> csMajors = students.stream()
            .collect(Collectors.partitioningBy(
                student -> "Computer Science".equals(student.getMajor()),
                Collectors.mapping(Student::getName, Collectors.toList())));
        
        System.out.println("\nCS students: " + csMajors.get(true));
        System.out.println("Non-CS students: " + csMajors.get(false));
        
        // Multiple criteria partitioning
        Map<Boolean, Map<Boolean, List<Student>>> complexPartition = students.stream()
            .collect(Collectors.partitioningBy(
                student -> student.getGrade() >= 80,  // High grade
                Collectors.partitioningBy(
                    student -> "Computer Science".equals(student.getMajor()))));  // CS major
        
        System.out.println("\nHigh grade CS students: " + 
            complexPartition.get(true).get(true).size());
        System.out.println("High grade non-CS students: " + 
            complexPartition.get(true).get(false).size());
        System.out.println("Low grade CS students: " + 
            complexPartition.get(false).get(true).size());
        System.out.println("Low grade non-CS students: " + 
            complexPartition.get(false).get(false).size());
    }
    
    static class Student {
        private String name;
        private double grade;
        private String major;
        
        public Student(String name, double grade, String major) {
            this.name = name;
            this.grade = grade;
            this.major = major;
        }
        
        public String getName() { return name; }
        public double getGrade() { return grade; }
        public String getMajor() { return major; }
        
        @Override
        public String toString() {
            return String.format("%s (%.1f, %s)", name, grade, major);
        }
    }
}
```

## 4. Custom Collectors

Creating your own collectors for specialized collection operations.

### Building Custom Collectors
```java
import java.util.*;
import java.util.function.*;
import java.util.stream.Collector;
import java.util.stream.Collectors;

public class CustomCollectorsDemo {
    
    // Custom collector to collect into a comma-separated string
    public static Collector<String, ?, String> toCommaSeparatedString() {
        return Collector.of(
            StringBuilder::new,                    // Supplier: create accumulator
            (sb, str) -> {                        // Accumulator: add element
                if (sb.length() > 0) sb.append(", ");
                sb.append(str);
            },
            (sb1, sb2) -> {                       // Combiner: combine accumulators
                if (sb1.length() > 0 && sb2.length() > 0) sb1.append(", ");
                return sb1.append(sb2);
            },
            StringBuilder::toString               // Finisher: final transformation
        );
    }
    
    // Custom collector to collect into a Set with size limit
    public static <T> Collector<T, ?, Set<T>> toLimitedSet(int maxSize) {
        return Collector.of(
            HashSet::new,
            (set, element) -> {
                if (set.size() < maxSize) {
                    set.add(element);
                }
            },
            (set1, set2) -> {
                Set<T> result = new HashSet<>(set1);
                for (T element : set2) {
                    if (result.size() >= maxSize) break;
                    result.add(element);
                }
                return result;
            }
        );
    }
    
    // Custom collector for statistics
    public static <T> Collector<T, ?, Statistics<T>> toStatistics(
            Function<T, Double> valueExtractor,
            Comparator<T> comparator) {
        
        return Collector.of(
            () -> new Statistics<>(valueExtractor, comparator),
            Statistics::accept,
            Statistics::combine,
            Function.identity()
        );
    }
    
    public static void main(String[] args) {
        List<String> words = Arrays.asList("apple", "banana", "cherry", "date", "elderberry");
        List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        
        // Using custom comma-separated collector
        String commaSeparated = words.stream()
            .filter(word -> word.length() > 4)
            .collect(toCommaSeparatedString());
        System.out.println("Comma separated: " + commaSeparated);
        
        // Using limited set collector
        Set<Integer> limitedSet = numbers.stream()
            .filter(n -> n % 2 == 0)
            .collect(toLimitedSet(3));
        System.out.println("Limited set (max 3): " + limitedSet);
        
        // Using custom statistics collector
        List<Product> products = Arrays.asList(
            new Product("Laptop", 1200.0),
            new Product("Phone", 800.0),
            new Product("Tablet", 400.0),
            new Product("Watch", 200.0),
            new Product("Headphones", 150.0)
        );
        
        Statistics<Product> priceStats = products.stream()
            .collect(toStatistics(
                Product::getPrice,
                Comparator.comparing(Product::getPrice)));
        
        System.out.println("Product statistics: " + priceStats);
        
        // Combining custom collectors
        Map<Boolean, String> expensiveProducts = products.stream()
            .collect(Collectors.partitioningBy(
                p -> p.getPrice() > 500,
                Collector.of(
                    StringBuilder::new,
                    (sb, product) -> {
                        if (sb.length() > 0) sb.append(", ");
                        sb.append(product.getName());
                    },
                    (sb1, sb2) -> {
                        if (sb1.length() > 0 && sb2.length() > 0) sb1.append(", ");
                        return sb1.append(sb2);
                    },
                    StringBuilder::toString
                )
            ));
        
        System.out.println("Expensive products: " + expensiveProducts.get(true));
        System.out.println("Affordable products: " + expensiveProducts.get(false));
    }
    
    // Statistics accumulator class
    static class Statistics<T> {
        private int count = 0;
        private double sum = 0.0;
        private T min = null;
        private T max = null;
        private final Function<T, Double> valueExtractor;
        private final Comparator<T> comparator;
        
        public Statistics(Function<T, Double> valueExtractor, Comparator<T> comparator) {
            this.valueExtractor = valueExtractor;
            this.comparator = comparator;
        }
        
        public void accept(T element) {
            count++;
            sum += valueExtractor.apply(element);
            
            if (min == null || comparator.compare(element, min) < 0) {
                min = element;
            }
            if (max == null || comparator.compare(element, max) > 0) {
                max = element;
            }
        }
        
        public Statistics<T> combine(Statistics<T> other) {
            Statistics<T> result = new Statistics<>(valueExtractor, comparator);
            result.count = this.count + other.count;
            result.sum = this.sum + other.sum;
            
            if (this.min != null && other.min != null) {
                result.min = comparator.compare(this.min, other.min) <= 0 ? this.min : other.min;
            } else {
                result.min = this.min != null ? this.min : other.min;
            }
            
            if (this.max != null && other.max != null) {
                result.max = comparator.compare(this.max, other.max) >= 0 ? this.max : other.max;
            } else {
                result.max = this.max != null ? this.max : other.max;
            }
            
            return result;
        }
        
        public double getAverage() {
            return count > 0 ? sum / count : 0.0;
        }
        
        @Override
        public String toString() {
            return String.format("Statistics{count=%d, sum=%.2f, avg=%.2f, min=%s, max=%s}",
                count, sum, getAverage(), min, max);
        }
    }
    
    static class Product {
        private String name;
        private double price;
        
        public Product(String name, double price) {
            this.name = name;
            this.price = price;
        }
        
        public String getName() { return name; }
        public double getPrice() { return price; }
        
        @Override
        public String toString() {
            return name + "($" + price + ")";
        }
    }
}
```

## 5. Parallel Collection Operations

Understanding how collectors work with parallel streams.

### Parallel Collection Examples
```java
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class ParallelCollectionDemo {
    public static void main(String[] args) {
        // Generate large dataset
        List<Integer> largeList = IntStream.rangeClosed(1, 1000000)
            .boxed()
            .collect(Collectors.toList());
        
        // Sequential vs Parallel collection
        long startTime = System.currentTimeMillis();
        Map<String, List<Integer>> sequentialGrouping = largeList.stream()
            .collect(Collectors.groupingBy(n -> {
                if (n % 3 == 0) return "Divisible by 3";
                else if (n % 2 == 0) return "Even";
                else return "Odd";
            }));
        long sequentialTime = System.currentTimeMillis() - startTime;
        
        startTime = System.currentTimeMillis();
        Map<String, List<Integer>> parallelGrouping = largeList.parallelStream()
            .collect(Collectors.groupingBy(n -> {
                if (n % 3 == 0) return "Divisible by 3";
                else if (n % 2 == 0) return "Even";
                else return "Odd";
            }));
        long parallelTime = System.currentTimeMillis() - startTime;
        
        System.out.println("Sequential grouping time: " + sequentialTime + "ms");
        System.out.println("Parallel grouping time: " + parallelTime + "ms");
        System.out.println("Results equal: " + 
            sequentialGrouping.keySet().equals(parallelGrouping.keySet()));
        
        // Thread-safe collection for parallel operations
        startTime = System.currentTimeMillis();
        ConcurrentHashMap<String, List<Integer>> concurrentMap = largeList.parallelStream()
            .collect(Collectors.groupingByConcurrent(n -> {
                if (n % 5 == 0) return "Divisible by 5";
                else if (n % 3 == 0) return "Divisible by 3";
                else if (n % 2 == 0) return "Even";
                else return "Odd";
            }));
        long concurrentTime = System.currentTimeMillis() - startTime;
        
        System.out.println("Concurrent grouping time: " + concurrentTime + "ms");
        System.out.println("Group sizes: " + 
            concurrentMap.entrySet().stream()
                .collect(Collectors.toMap(
                    Map.Entry::getKey,
                    entry -> entry.getValue().size())));
        
        // Parallel reduction operations
        List<Employee> employees = generateEmployees(100000);
        
        startTime = System.currentTimeMillis();
        Map<String, Double> sequentialSalaries = employees.stream()
            .collect(Collectors.groupingBy(
                Employee::getDepartment,
                Collectors.averagingDouble(Employee::getSalary)));
        long seqSalaryTime = System.currentTimeMillis() - startTime;
        
        startTime = System.currentTimeMillis();
        Map<String, Double> parallelSalaries = employees.parallelStream()
            .collect(Collectors.groupingBy(
                Employee::getDepartment,
                Collectors.averagingDouble(Employee::getSalary)));
        long parSalaryTime = System.currentTimeMillis() - startTime;
        
        System.out.println("\nSalary calculation:");
        System.out.println("Sequential time: " + seqSalaryTime + "ms");
        System.out.println("Parallel time: " + parSalaryTime + "ms");
        System.out.println("Average salaries: " + sequentialSalaries);
    }
    
    private static List<Employee> generateEmployees(int count) {
        List<Employee> employees = new ArrayList<>();
        String[] departments = {"Engineering", "Marketing", "HR", "Finance", "Operations"};
        Random random = new Random();
        
        for (int i = 0; i < count; i++) {
            employees.add(new Employee(
                "Employee" + i,
                departments[random.nextInt(departments.length)],
                50000 + random.nextDouble() * 100000,
                22 + random.nextInt(40)
            ));
        }
        return employees;
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
    }
}
```

## Summary

### Key Collector Types

1. **Basic Collectors**: `toList()`, `toSet()`, `toMap()`
2. **Joining Collectors**: `joining()` with delimiters
3. **Grouping Collectors**: `groupingBy()` with downstream collectors
4. **Partitioning Collectors**: `partitioningBy()` for binary classification
5. **Statistical Collectors**: `summarizing()`, `averaging()`, `counting()`
6. **Custom Collectors**: User-defined accumulation logic

### Performance Considerations

- **Parallel Collection**: Use `groupingByConcurrent()` for thread-safe parallel operations
- **Memory Usage**: Consider memory requirements for large groupings
- **Custom Collectors**: Can be optimized for specific use cases
- **Downstream Collectors**: Combine multiple collection operations efficiently

### Best Practices

1. **Choose Appropriate Collector**: Use the most specific collector for your needs
2. **Combine Collectors**: Use downstream collectors for complex aggregations
3. **Handle Duplicates**: Specify merge functions for `toMap()` operations
4. **Consider Thread Safety**: Use concurrent collectors for parallel streams
5. **Custom Logic**: Create custom collectors for specialized requirements

Collectors provide a powerful and flexible way to transform stream data into various collection types and perform complex aggregation operations, making them essential for effective stream processing in Java.
