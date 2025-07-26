# Java Streams - Real-World Examples and Use Cases

## 1. Data Processing and Analysis

### Sales Data Analysis
```java
import java.util.*;
import java.util.stream.Collectors;
import java.time.LocalDate;
import java.time.Month;

public class SalesDataAnalysisDemo {
    public static void main(String[] args) {
        List<SaleRecord> sales = generateSalesData();
        
        // 1. Total revenue by month
        System.out.println("=== Monthly Revenue Analysis ===");
        Map<Month, Double> monthlyRevenue = sales.stream()
            .collect(Collectors.groupingBy(
                sale -> sale.getDate().getMonth(),
                Collectors.summingDouble(SaleRecord::getAmount)
            ));
        
        monthlyRevenue.entrySet().stream()
            .sorted(Map.Entry.comparingByKey())
            .forEach(entry -> System.out.printf("%s: $%.2f%n", 
                entry.getKey(), entry.getValue()));
        
        // 2. Top 5 products by revenue
        System.out.println("\n=== Top 5 Products by Revenue ===");
        sales.stream()
            .collect(Collectors.groupingBy(
                SaleRecord::getProduct,
                Collectors.summingDouble(SaleRecord::getAmount)))
            .entrySet().stream()
            .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
            .limit(5)
            .forEach(entry -> System.out.printf("%-15s: $%.2f%n", 
                entry.getKey(), entry.getValue()));
        
        // 3. Sales representative performance
        System.out.println("\n=== Sales Rep Performance ===");
        Map<String, SalesStats> repStats = sales.stream()
            .collect(Collectors.groupingBy(
                SaleRecord::getSalesRep,
                Collectors.collectingAndThen(
                    Collectors.toList(),
                    list -> new SalesStats(
                        list.size(),
                        list.stream().mapToDouble(SaleRecord::getAmount).sum(),
                        list.stream().mapToDouble(SaleRecord::getAmount).average().orElse(0)
                    )
                )
            ));
        
        repStats.entrySet().stream()
            .sorted(Map.Entry.<String, SalesStats>comparingByValue(
                Comparator.comparing(SalesStats::getTotalRevenue)).reversed())
            .forEach(entry -> System.out.printf("%-12s: %s%n", 
                entry.getKey(), entry.getValue()));
        
        // 4. Regional analysis
        System.out.println("\n=== Regional Performance ===");
        Map<String, DoubleSummaryStatistics> regionalStats = sales.stream()
            .collect(Collectors.groupingBy(
                SaleRecord::getRegion,
                Collectors.summarizingDouble(SaleRecord::getAmount)
            ));
        
        regionalStats.forEach((region, stats) -> {
            System.out.printf("%s:%n", region);
            System.out.printf("  Total Sales: %d%n", stats.getCount());
            System.out.printf("  Revenue: $%.2f%n", stats.getSum());
            System.out.printf("  Average: $%.2f%n", stats.getAverage());
            System.out.printf("  Range: $%.2f - $%.2f%n", stats.getMin(), stats.getMax());
            System.out.println();
        });
        
        // 5. Quarterly trends
        System.out.println("=== Quarterly Trends ===");
        Map<Integer, Double> quarterlyRevenue = sales.stream()
            .collect(Collectors.groupingBy(
                sale -> (sale.getDate().getMonthValue() - 1) / 3 + 1, // Quarter calculation
                Collectors.summingDouble(SaleRecord::getAmount)
            ));
        
        quarterlyRevenue.entrySet().stream()
            .sorted(Map.Entry.comparingByKey())
            .forEach(entry -> System.out.printf("Q%d: $%.2f%n", 
                entry.getKey(), entry.getValue()));
        
        // 6. High-value customers (customers with total purchases > $10,000)
        System.out.println("\n=== High-Value Customers ===");
        sales.stream()
            .collect(Collectors.groupingBy(
                SaleRecord::getCustomer,
                Collectors.summingDouble(SaleRecord::getAmount)))
            .entrySet().stream()
            .filter(entry -> entry.getValue() > 10000)
            .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
            .forEach(entry -> System.out.printf("%-20s: $%.2f%n", 
                entry.getKey(), entry.getValue()));
    }
    
    private static List<SaleRecord> generateSalesData() {
        String[] products = {"Laptop", "Smartphone", "Tablet", "Headphones", "Smartwatch", "Speaker"};
        String[] regions = {"North", "South", "East", "West"};
        String[] salesReps = {"Alice Johnson", "Bob Smith", "Carol Davis", "David Wilson", "Eve Brown"};
        String[] customers = {"TechCorp", "InnovateLLC", "FutureTech", "DataSystems", "CloudWorks", 
                             "NetSolutions", "InfoTech", "DigitalPro", "TechVision", "SmartSystems"};
        
        Random random = new Random(42);
        List<SaleRecord> sales = new ArrayList<>();
        
        for (int i = 0; i < 1000; i++) {
            LocalDate date = LocalDate.of(2023, random.nextInt(12) + 1, random.nextInt(28) + 1);
            String product = products[random.nextInt(products.length)];
            String region = regions[random.nextInt(regions.length)];
            String salesRep = salesReps[random.nextInt(salesReps.length)];
            String customer = customers[random.nextInt(customers.length)];
            double amount = 100 + random.nextDouble() * 9900; // $100 - $10,000
            
            sales.add(new SaleRecord(date, product, region, salesRep, customer, amount));
        }
        
        return sales;
    }
    
    static class SaleRecord {
        private LocalDate date;
        private String product;
        private String region;
        private String salesRep;
        private String customer;
        private double amount;
        
        public SaleRecord(LocalDate date, String product, String region, 
                         String salesRep, String customer, double amount) {
            this.date = date;
            this.product = product;
            this.region = region;
            this.salesRep = salesRep;
            this.customer = customer;
            this.amount = amount;
        }
        
        // Getters
        public LocalDate getDate() { return date; }
        public String getProduct() { return product; }
        public String getRegion() { return region; }
        public String getSalesRep() { return salesRep; }
        public String getCustomer() { return customer; }
        public double getAmount() { return amount; }
        
        @Override
        public String toString() {
            return String.format("%s: %s sold %s to %s for $%.2f in %s", 
                date, salesRep, product, customer, amount, region);
        }
    }
    
    static class SalesStats {
        private int salesCount;
        private double totalRevenue;
        private double averageSale;
        
        public SalesStats(int salesCount, double totalRevenue, double averageSale) {
            this.salesCount = salesCount;
            this.totalRevenue = totalRevenue;
            this.averageSale = averageSale;
        }
        
        public int getSalesCount() { return salesCount; }
        public double getTotalRevenue() { return totalRevenue; }
        public double getAverageSale() { return averageSale; }
        
        @Override
        public String toString() {
            return String.format("Sales: %d, Revenue: $%.2f, Avg: $%.2f", 
                salesCount, totalRevenue, averageSale);
        }
    }
}
```

### Log File Analysis
```java
import java.util.*;
import java.util.stream.Collectors;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.regex.Pattern;

public class LogAnalysisDemo {
    public static void main(String[] args) {
        List<LogEntry> logs = generateLogData();
        
        // 1. Error rate analysis
        System.out.println("=== Error Rate Analysis ===");
        Map<LogLevel, Long> levelCounts = logs.stream()
            .collect(Collectors.groupingBy(
                LogEntry::getLevel,
                Collectors.counting()));
        
        long totalLogs = logs.size();
        levelCounts.forEach((level, count) -> {
            double percentage = (count * 100.0) / totalLogs;
            System.out.printf("%-7s: %5d (%.2f%%)%n", level, count, percentage);
        });
        
        // 2. Top error sources
        System.out.println("\n=== Top Error Sources ===");
        logs.stream()
            .filter(log -> log.getLevel() == LogLevel.ERROR)
            .collect(Collectors.groupingBy(
                LogEntry::getSource,
                Collectors.counting()))
            .entrySet().stream()
            .sorted(Map.Entry.<String, Long>comparingByValue().reversed())
            .limit(5)
            .forEach(entry -> System.out.printf("%-20s: %d errors%n", 
                entry.getKey(), entry.getValue()));
        
        // 3. Hourly activity pattern
        System.out.println("\n=== Hourly Activity Pattern ===");
        Map<Integer, Long> hourlyActivity = logs.stream()
            .collect(Collectors.groupingBy(
                log -> log.getTimestamp().getHour(),
                Collectors.counting()));
        
        hourlyActivity.entrySet().stream()
            .sorted(Map.Entry.comparingByKey())
            .forEach(entry -> {
                String bar = "*".repeat((int) (entry.getValue() / 10)); // Scale for display
                System.out.printf("%02d:00 [%4d] %s%n", 
                    entry.getKey(), entry.getValue(), bar);
            });
        
        // 4. Find suspicious patterns (multiple errors from same source in short time)
        System.out.println("\n=== Suspicious Error Patterns ===");
        Map<String, List<LogEntry>> errorsBySource = logs.stream()
            .filter(log -> log.getLevel() == LogLevel.ERROR)
            .collect(Collectors.groupingBy(LogEntry::getSource));
        
        errorsBySource.entrySet().stream()
            .filter(entry -> entry.getValue().size() > 5) // More than 5 errors
            .forEach(entry -> {
                String source = entry.getKey();
                List<LogEntry> errors = entry.getValue();
                
                // Check if errors occurred within a short time window
                LocalDateTime firstError = errors.stream()
                    .map(LogEntry::getTimestamp)
                    .min(LocalDateTime::compareTo)
                    .orElse(null);
                
                LocalDateTime lastError = errors.stream()
                    .map(LogEntry::getTimestamp)
                    .max(LocalDateTime::compareTo)
                    .orElse(null);
                
                if (firstError != null && lastError != null) {
                    long minutesDiff = java.time.Duration.between(firstError, lastError).toMinutes();
                    if (minutesDiff < 60) { // Within an hour
                        System.out.printf("%s: %d errors in %d minutes%n", 
                            source, errors.size(), minutesDiff);
                    }
                }
            });
        
        // 5. Performance analysis (response times)
        System.out.println("\n=== Performance Analysis ===");
        OptionalDouble avgResponseTime = logs.stream()
            .filter(log -> log.getResponseTime() > 0)
            .mapToDouble(LogEntry::getResponseTime)
            .average();
        
        if (avgResponseTime.isPresent()) {
            System.out.printf("Average response time: %.2f ms%n", avgResponseTime.getAsDouble());
        }
        
        // Slow requests (>1000ms)
        long slowRequests = logs.stream()
            .filter(log -> log.getResponseTime() > 1000)
            .count();
        
        System.out.printf("Slow requests (>1000ms): %d%n", slowRequests);
        
        // 6. Search for specific patterns in messages
        System.out.println("\n=== Database Connection Issues ===");
        Pattern dbPattern = Pattern.compile("database|connection|timeout", Pattern.CASE_INSENSITIVE);
        
        logs.stream()
            .filter(log -> dbPattern.matcher(log.getMessage()).find())
            .limit(5)
            .forEach(log -> System.out.printf("[%s] %s: %s%n", 
                log.getTimestamp().format(DateTimeFormatter.ofPattern("HH:mm:ss")),
                log.getLevel(), log.getMessage()));
    }
    
    private static List<LogEntry> generateLogData() {
        String[] sources = {"UserService", "OrderService", "PaymentService", "NotificationService", 
                           "DatabaseConnector", "AuthService", "ReportService"};
        String[] messages = {
            "User login successful",
            "Order processed successfully", 
            "Payment completed",
            "Database connection timeout",
            "Invalid user credentials",
            "Service unavailable",
            "Memory usage high",
            "Request processed",
            "Cache miss",
            "API rate limit exceeded"
        };
        
        Random random = new Random(42);
        List<LogEntry> logs = new ArrayList<>();
        LocalDateTime baseTime = LocalDateTime.now().minusHours(24);
        
        for (int i = 0; i < 5000; i++) {
            LocalDateTime timestamp = baseTime.plusMinutes(random.nextInt(1440)); // 24 hours
            String source = sources[random.nextInt(sources.length)];
            String message = messages[random.nextInt(messages.length)];
            
            // Bias toward INFO level, some WARN and ERROR
            LogLevel level;
            int levelRand = random.nextInt(100);
            if (levelRand < 70) level = LogLevel.INFO;
            else if (levelRand < 90) level = LogLevel.WARN;
            else level = LogLevel.ERROR;
            
            // Add response time for some entries
            int responseTime = level == LogLevel.ERROR ? 
                random.nextInt(3000) + 500 : // Errors tend to be slower
                random.nextInt(500) + 50;    // Normal requests
            
            logs.add(new LogEntry(timestamp, level, source, message, responseTime));
        }
        
        return logs;
    }
    
    enum LogLevel {
        INFO, WARN, ERROR
    }
    
    static class LogEntry {
        private LocalDateTime timestamp;
        private LogLevel level;
        private String source;
        private String message;
        private int responseTime;
        
        public LogEntry(LocalDateTime timestamp, LogLevel level, String source, 
                       String message, int responseTime) {
            this.timestamp = timestamp;
            this.level = level;
            this.source = source;
            this.message = message;
            this.responseTime = responseTime;
        }
        
        public LocalDateTime getTimestamp() { return timestamp; }
        public LogLevel getLevel() { return level; }
        public String getSource() { return source; }
        public String getMessage() { return message; }
        public int getResponseTime() { return responseTime; }
        
        @Override
        public String toString() {
            return String.format("[%s] %s %s: %s", timestamp, level, source, message);
        }
    }
}
```

## 2. File Processing and Text Analysis

### CSV Data Processing
```java
import java.util.*;
import java.util.stream.Collectors;
import java.io.*;
import java.nio.file.*;

public class CSVProcessingDemo {
    public static void main(String[] args) {
        // Simulate CSV data processing
        List<EmployeeRecord> employees = loadEmployeeData();
        
        // 1. Department salary analysis
        System.out.println("=== Department Salary Analysis ===");
        Map<String, DoubleSummaryStatistics> deptSalaryStats = employees.stream()
            .collect(Collectors.groupingBy(
                EmployeeRecord::getDepartment,
                Collectors.summarizingDouble(EmployeeRecord::getSalary)
            ));
        
        deptSalaryStats.forEach((dept, stats) -> {
            System.out.printf("%s:%n", dept);
            System.out.printf("  Employees: %d%n", stats.getCount());
            System.out.printf("  Total: $%.2f%n", stats.getSum());
            System.out.printf("  Average: $%.2f%n", stats.getAverage());
            System.out.printf("  Range: $%.2f - $%.2f%n", stats.getMin(), stats.getMax());
            System.out.println();
        });
        
        // 2. Age demographics
        System.out.println("=== Age Demographics ===");
        Map<String, Long> ageGroups = employees.stream()
            .collect(Collectors.groupingBy(
                emp -> {
                    int age = emp.getAge();
                    if (age < 30) return "Under 30";
                    else if (age < 40) return "30-39";
                    else if (age < 50) return "40-49";
                    else return "50+";
                },
                Collectors.counting()
            ));
        
        ageGroups.forEach((group, count) -> 
            System.out.printf("%-10s: %d employees%n", group, count));
        
        // 3. Export filtered data
        System.out.println("\n=== High Earners Export ===");
        List<String> highEarnersCSV = employees.stream()
            .filter(emp -> emp.getSalary() > 80000)
            .sorted(Comparator.comparing(EmployeeRecord::getSalary).reversed())
            .map(emp -> String.format("%s,%s,%d,%.2f,%s", 
                emp.getName(), emp.getDepartment(), emp.getAge(), emp.getSalary(), emp.getLocation()))
            .collect(Collectors.toList());
        
        System.out.println("High earners CSV format:");
        System.out.println("Name,Department,Age,Salary,Location");
        highEarnersCSV.stream().limit(5).forEach(System.out::println);
        System.out.printf("... and %d more records%n", highEarnersCSV.size() - 5);
        
        // 4. Data validation
        System.out.println("\n=== Data Validation ===");
        List<String> validationErrors = employees.stream()
            .filter(emp -> emp.getName() == null || emp.getName().trim().isEmpty() ||
                          emp.getSalary() <= 0 || emp.getAge() < 18 || emp.getAge() > 100)
            .map(emp -> "Invalid record: " + emp)
            .collect(Collectors.toList());
        
        if (validationErrors.isEmpty()) {
            System.out.println("All records are valid.");
        } else {
            System.out.println("Validation errors found:");
            validationErrors.forEach(System.out::println);
        }
        
        // 5. Location-based analysis
        System.out.println("\n=== Location Analysis ===");
        Map<String, List<EmployeeRecord>> byLocation = employees.stream()
            .collect(Collectors.groupingBy(EmployeeRecord::getLocation));
        
        byLocation.forEach((location, empList) -> {
            double avgSalary = empList.stream()
                .mapToDouble(EmployeeRecord::getSalary)
                .average().orElse(0);
            
            System.out.printf("%-15s: %d employees, avg salary: $%.2f%n", 
                location, empList.size(), avgSalary);
        });
        
        // 6. Generate summary report
        System.out.println("\n=== Summary Report ===");
        generateSummaryReport(employees);
    }
    
    private static List<EmployeeRecord> loadEmployeeData() {
        String[] departments = {"Engineering", "Marketing", "Sales", "HR", "Finance", "Operations"};
        String[] locations = {"New York", "San Francisco", "Chicago", "Austin", "Seattle", "Boston"};
        String[] firstNames = {"John", "Jane", "Mike", "Sarah", "David", "Lisa", "Alex", "Emma"};
        String[] lastNames = {"Smith", "Johnson", "Brown", "Davis", "Wilson", "Miller", "Garcia", "Martinez"};
        
        Random random = new Random(42);
        List<EmployeeRecord> employees = new ArrayList<>();
        
        for (int i = 0; i < 500; i++) {
            String name = firstNames[random.nextInt(firstNames.length)] + " " + 
                         lastNames[random.nextInt(lastNames.length)];
            String department = departments[random.nextInt(departments.length)];
            int age = 22 + random.nextInt(43); // 22-65
            double salary = 40000 + random.nextDouble() * 120000; // $40k-$160k
            String location = locations[random.nextInt(locations.length)];
            
            employees.add(new EmployeeRecord(name, department, age, salary, location));
        }
        
        return employees;
    }
    
    private static void generateSummaryReport(List<EmployeeRecord> employees) {
        int totalEmployees = employees.size();
        double totalPayroll = employees.stream().mapToDouble(EmployeeRecord::getSalary).sum();
        double avgSalary = totalPayroll / totalEmployees;
        
        OptionalDouble avgAge = employees.stream().mapToInt(EmployeeRecord::getAge).average();
        
        long uniqueDepartments = employees.stream()
            .map(EmployeeRecord::getDepartment)
            .distinct()
            .count();
        
        long uniqueLocations = employees.stream()
            .map(EmployeeRecord::getLocation)
            .distinct()
            .count();
        
        System.out.printf("Total Employees: %d%n", totalEmployees);
        System.out.printf("Total Payroll: $%.2f%n", totalPayroll);
        System.out.printf("Average Salary: $%.2f%n", avgSalary);
        System.out.printf("Average Age: %.1f%n", avgAge.orElse(0));
        System.out.printf("Departments: %d%n", uniqueDepartments);
        System.out.printf("Locations: %d%n", uniqueLocations);
    }
    
    static class EmployeeRecord {
        private String name;
        private String department;
        private int age;
        private double salary;
        private String location;
        
        public EmployeeRecord(String name, String department, int age, double salary, String location) {
            this.name = name;
            this.department = department;
            this.age = age;
            this.salary = salary;
            this.location = location;
        }
        
        public String getName() { return name; }
        public String getDepartment() { return department; }
        public int getAge() { return age; }
        public double getSalary() { return salary; }
        public String getLocation() { return location; }
        
        @Override
        public String toString() {
            return String.format("%s (%s, %d, $%.2f, %s)", 
                name, department, age, salary, location);
        }
    }
}
```

### Text Analysis and Processing
```java
import java.util.*;
import java.util.stream.Collectors;
import java.util.function.Function;
import java.util.regex.Pattern;

public class TextAnalysisDemo {
    public static void main(String[] args) {
        String text = getSampleText();
        
        // 1. Word frequency analysis
        System.out.println("=== Word Frequency Analysis ===");
        Map<String, Long> wordFreq = Arrays.stream(text.toLowerCase().split("\\W+"))
            .filter(word -> !word.isEmpty())
            .filter(word -> word.length() > 2) // Filter out short words
            .collect(Collectors.groupingBy(
                Function.identity(),
                Collectors.counting()
            ));
        
        System.out.println("Top 10 most frequent words:");
        wordFreq.entrySet().stream()
            .sorted(Map.Entry.<String, Long>comparingByValue().reversed())
            .limit(10)
            .forEach(entry -> System.out.printf("%-12s: %d%n", entry.getKey(), entry.getValue()));
        
        // 2. Sentence analysis
        System.out.println("\n=== Sentence Analysis ===");
        String[] sentences = text.split("[.!?]+");
        
        DoubleSummaryStatistics sentenceStats = Arrays.stream(sentences)
            .filter(s -> !s.trim().isEmpty())
            .mapToDouble(s -> s.trim().split("\\s+").length) // Words per sentence
            .summaryStatistics();
        
        System.out.printf("Total sentences: %d%n", (int) sentenceStats.getCount());
        System.out.printf("Average words per sentence: %.2f%n", sentenceStats.getAverage());
        System.out.printf("Shortest sentence: %.0f words%n", sentenceStats.getMin());
        System.out.printf("Longest sentence: %.0f words%n", sentenceStats.getMax());
        
        // 3. Character analysis
        System.out.println("\n=== Character Analysis ===");
        Map<Character, Long> charFreq = text.toLowerCase().chars()
            .filter(Character::isLetter)
            .mapToObj(c -> (char) c)
            .collect(Collectors.groupingBy(
                Function.identity(),
                Collectors.counting()
            ));
        
        System.out.println("Top 5 most frequent letters:");
        charFreq.entrySet().stream()
            .sorted(Map.Entry.<Character, Long>comparingByValue().reversed())
            .limit(5)
            .forEach(entry -> System.out.printf("'%c': %d%n", entry.getKey(), entry.getValue()));
        
        // 4. Reading level analysis (simplified)
        System.out.println("\n=== Reading Level Analysis ===");
        long totalWords = Arrays.stream(text.split("\\W+"))
            .filter(word -> !word.isEmpty())
            .count();
        
        long totalSentences = Arrays.stream(sentences)
            .filter(s -> !s.trim().isEmpty())
            .count();
        
        long totalSyllables = Arrays.stream(text.split("\\W+"))
            .filter(word -> !word.isEmpty())
            .mapToLong(TextAnalysisDemo::countSyllables)
            .sum();
        
        // Flesch Reading Ease (simplified)
        double fleschScore = 206.835 - (1.015 * totalWords / totalSentences) - 
                           (84.6 * totalSyllables / totalWords);
        
        System.out.printf("Total words: %d%n", totalWords);
        System.out.printf("Total sentences: %d%n", totalSentences);
        System.out.printf("Total syllables: %d%n", totalSyllables);
        System.out.printf("Flesch Reading Ease: %.2f%n", fleschScore);
        
        // 5. Find patterns
        System.out.println("\n=== Pattern Analysis ===");
        
        // Find email-like patterns
        Pattern emailPattern = Pattern.compile("\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b");
        long emailCount = emailPattern.matcher(text).results().count();
        System.out.printf("Email-like patterns found: %d%n", emailCount);
        
        // Find numbers
        Pattern numberPattern = Pattern.compile("\\b\\d+\\b");
        List<String> numbers = numberPattern.matcher(text)
            .results()
            .map(match -> match.group())
            .collect(Collectors.toList());
        System.out.printf("Numbers found: %s%n", numbers);
        
        // 6. Keyword extraction (simple approach)
        System.out.println("\n=== Keyword Extraction ===");
        Set<String> stopWords = Set.of("the", "and", "or", "but", "in", "on", "at", "to", "for", 
                                      "of", "with", "by", "is", "are", "was", "were", "been", "be", 
                                      "have", "has", "had", "do", "does", "did", "will", "would", 
                                      "could", "should", "may", "might", "can", "this", "that", 
                                      "these", "those", "a", "an");
        
        List<String> keywords = Arrays.stream(text.toLowerCase().split("\\W+"))
            .filter(word -> word.length() > 3)
            .filter(word -> !stopWords.contains(word))
            .collect(Collectors.groupingBy(
                Function.identity(),
                Collectors.counting()
            ))
            .entrySet().stream()
            .filter(entry -> entry.getValue() > 1) // Appears more than once
            .sorted(Map.Entry.<String, Long>comparingByValue().reversed())
            .map(Map.Entry::getKey)
            .limit(10)
            .collect(Collectors.toList());
        
        System.out.println("Potential keywords: " + keywords);
        
        // 7. Text summary statistics
        System.out.println("\n=== Text Summary ===");
        int totalCharacters = text.length();
        int totalCharactersNoSpaces = text.replaceAll("\\s", "").length();
        int totalParagraphs = text.split("\\n\\s*\\n").length;
        
        System.out.printf("Characters (with spaces): %d%n", totalCharacters);
        System.out.printf("Characters (no spaces): %d%n", totalCharactersNoSpaces);
        System.out.printf("Words: %d%n", totalWords);
        System.out.printf("Sentences: %d%n", totalSentences);
        System.out.printf("Paragraphs: %d%n", totalParagraphs);
    }
    
    private static int countSyllables(String word) {
        // Simple syllable counting (very basic)
        String vowels = "aeiouy";
        word = word.toLowerCase();
        int count = 0;
        boolean previousWasVowel = false;
        
        for (int i = 0; i < word.length(); i++) {
            boolean isVowel = vowels.indexOf(word.charAt(i)) != -1;
            if (isVowel && !previousWasVowel) {
                count++;
            }
            previousWasVowel = isVowel;
        }
        
        if (word.endsWith("e")) count--;
        if (count == 0) count = 1;
        
        return count;
    }
    
    private static String getSampleText() {
        return """
            Java Streams provide a powerful and expressive way to process collections of data. 
            They enable developers to write more readable and maintainable code by using 
            functional programming concepts. Streams support both sequential and parallel 
            processing, making them suitable for a wide range of applications.
            
            The Stream API was introduced in Java 8 as part of the major language update. 
            It allows for lazy evaluation, which means operations are not executed until 
            a terminal operation is called. This leads to more efficient processing, 
            especially when dealing with large datasets.
            
            Common operations include filtering, mapping, reducing, and collecting. 
            These operations can be chained together to create complex data processing 
            pipelines. The collect() method is particularly useful for accumulating 
            results into collections or other data structures.
            
            Parallel streams can significantly improve performance for CPU-intensive 
            operations on large datasets. However, they should be used carefully, 
            considering factors such as thread safety and the overhead of parallelization.
            
            Best practices include avoiding side effects in lambda expressions, 
            using appropriate collectors, and considering the characteristics of 
            the data source when deciding between sequential and parallel processing.
            """;
    }
}
```

## 3. Web API Data Processing

### JSON-like Data Processing
```java
import java.util.*;
import java.util.stream.Collectors;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

public class APIDataProcessingDemo {
    public static void main(String[] args) {
        List<APIResponse> responses = generateAPIResponses();
        
        // 1. Response time analysis
        System.out.println("=== API Response Time Analysis ===");
        DoubleSummaryStatistics responseStats = responses.stream()
            .mapToDouble(APIResponse::getResponseTime)
            .summaryStatistics();
        
        System.out.printf("Total requests: %d%n", (int) responseStats.getCount());
        System.out.printf("Average response time: %.2f ms%n", responseStats.getAverage());
        System.out.printf("Min response time: %.2f ms%n", responseStats.getMin());
        System.out.printf("Max response time: %.2f ms%n", responseStats.getMax());
        
        // 2. Status code distribution
        System.out.println("\n=== Status Code Distribution ===");
        Map<Integer, Long> statusCounts = responses.stream()
            .collect(Collectors.groupingBy(
                APIResponse::getStatusCode,
                Collectors.counting()
            ));
        
        statusCounts.entrySet().stream()
            .sorted(Map.Entry.comparingByKey())
            .forEach(entry -> {
                String status = getStatusDescription(entry.getKey());
                double percentage = (entry.getValue() * 100.0) / responses.size();
                System.out.printf("%d %-20s: %d (%.2f%%)%n", 
                    entry.getKey(), status, entry.getValue(), percentage);
            });
        
        // 3. Endpoint performance analysis
        System.out.println("\n=== Endpoint Performance ===");
        Map<String, DoubleSummaryStatistics> endpointStats = responses.stream()
            .collect(Collectors.groupingBy(
                APIResponse::getEndpoint,
                Collectors.summarizingDouble(APIResponse::getResponseTime)
            ));
        
        endpointStats.entrySet().stream()
            .sorted((e1, e2) -> Double.compare(e2.getValue().getAverage(), e1.getValue().getAverage()))
            .forEach(entry -> {
                String endpoint = entry.getKey();
                DoubleSummaryStatistics stats = entry.getValue();
                System.out.printf("%-20s: avg=%.2fms, count=%d, max=%.2fms%n",
                    endpoint, stats.getAverage(), stats.getCount(), stats.getMax());
            });
        
        // 4. Error analysis
        System.out.println("\n=== Error Analysis ===");
        List<APIResponse> errors = responses.stream()
            .filter(response -> response.getStatusCode() >= 400)
            .collect(Collectors.toList());
        
        System.out.printf("Total errors: %d (%.2f%% of all requests)%n", 
            errors.size(), (errors.size() * 100.0) / responses.size());
        
        Map<String, Long> errorsByEndpoint = errors.stream()
            .collect(Collectors.groupingBy(
                APIResponse::getEndpoint,
                Collectors.counting()
            ));
        
        System.out.println("Errors by endpoint:");
        errorsByEndpoint.entrySet().stream()
            .sorted(Map.Entry.<String, Long>comparingByValue().reversed())
            .forEach(entry -> System.out.printf("  %-20s: %d errors%n", 
                entry.getKey(), entry.getValue()));
        
        // 5. Time-based analysis
        System.out.println("\n=== Hourly Request Pattern ===");
        Map<Integer, Long> hourlyRequests = responses.stream()
            .collect(Collectors.groupingBy(
                response -> response.getTimestamp().getHour(),
                Collectors.counting()
            ));
        
        hourlyRequests.entrySet().stream()
            .sorted(Map.Entry.comparingByKey())
            .forEach(entry -> {
                String bar = "*".repeat((int) (entry.getValue() / 20)); // Scale for display
                System.out.printf("%02d:00 [%4d] %s%n", 
                    entry.getKey(), entry.getValue(), bar);
            });
        
        // 6. SLA compliance (responses under 1000ms)
        System.out.println("\n=== SLA Compliance ===");
        long fastResponses = responses.stream()
            .mapToDouble(APIResponse::getResponseTime)
            .filter(time -> time < 1000)
            .count();
        
        double slaCompliance = (fastResponses * 100.0) / responses.size();
        System.out.printf("Responses under 1000ms: %d/%d (%.2f%%)%n", 
            fastResponses, responses.size(), slaCompliance);
        
        // 7. User agent analysis
        System.out.println("\n=== Top User Agents ===");
        responses.stream()
            .collect(Collectors.groupingBy(
                APIResponse::getUserAgent,
                Collectors.counting()
            ))
            .entrySet().stream()
            .sorted(Map.Entry.<String, Long>comparingByValue().reversed())
            .limit(5)
            .forEach(entry -> System.out.printf("%-30s: %d requests%n", 
                entry.getKey(), entry.getValue()));
        
        // 8. Generate alert conditions
        System.out.println("\n=== Alert Conditions ===");
        generateAlerts(responses);
    }
    
    private static List<APIResponse> generateAPIResponses() {
        String[] endpoints = {"/api/users", "/api/orders", "/api/products", "/api/auth", 
                             "/api/search", "/api/analytics", "/api/reports"};
        String[] userAgents = {"Chrome/91.0", "Firefox/89.0", "Safari/14.1", "Edge/91.0", 
                              "Mobile Safari", "Chrome Mobile", "API Client"};
        
        Random random = new Random(42);
        List<APIResponse> responses = new ArrayList<>();
        LocalDateTime baseTime = LocalDateTime.now().minusHours(24);
        
        for (int i = 0; i < 10000; i++) {
            LocalDateTime timestamp = baseTime.plusMinutes(random.nextInt(1440));
            String endpoint = endpoints[random.nextInt(endpoints.length)];
            String userAgent = userAgents[random.nextInt(userAgents.length)];
            
            // Bias toward successful responses
            int statusCode;
            int statusRand = random.nextInt(100);
            if (statusRand < 85) statusCode = 200;
            else if (statusRand < 92) statusCode = 404;
            else if (statusRand < 97) statusCode = 400;
            else statusCode = 500;
            
            // Response time influenced by status code
            double responseTime;
            if (statusCode == 200) {
                responseTime = 50 + random.nextGaussian() * 200; // Normal: ~50-450ms
            } else {
                responseTime = 200 + random.nextGaussian() * 500; // Errors are slower
            }
            responseTime = Math.max(10, responseTime); // Minimum 10ms
            
            responses.add(new APIResponse(timestamp, endpoint, statusCode, responseTime, userAgent));
        }
        
        return responses;
    }
    
    private static String getStatusDescription(int statusCode) {
        return switch (statusCode) {
            case 200 -> "(OK)";
            case 400 -> "(Bad Request)";
            case 401 -> "(Unauthorized)";
            case 404 -> "(Not Found)";
            case 500 -> "(Internal Error)";
            default -> "(Other)";
        };
    }
    
    private static void generateAlerts(List<APIResponse> responses) {
        // Alert 1: High error rate in last hour
        LocalDateTime oneHourAgo = LocalDateTime.now().minusHours(1);
        List<APIResponse> recentResponses = responses.stream()
            .filter(response -> response.getTimestamp().isAfter(oneHourAgo))
            .collect(Collectors.toList());
        
        if (!recentResponses.isEmpty()) {
            long recentErrors = recentResponses.stream()
                .mapToInt(APIResponse::getStatusCode)
                .filter(code -> code >= 400)
                .count();
            
            double errorRate = (recentErrors * 100.0) / recentResponses.size();
            if (errorRate > 5.0) {
                System.out.printf("🚨 HIGH ERROR RATE: %.2f%% in last hour%n", errorRate);
            }
        }
        
        // Alert 2: Slow endpoint
        Map<String, Double> endpointAvgTimes = responses.stream()
            .filter(response -> response.getTimestamp().isAfter(oneHourAgo))
            .collect(Collectors.groupingBy(
                APIResponse::getEndpoint,
                Collectors.averagingDouble(APIResponse::getResponseTime)
            ));
        
        endpointAvgTimes.entrySet().stream()
            .filter(entry -> entry.getValue() > 1000)
            .forEach(entry -> System.out.printf("🐌 SLOW ENDPOINT: %s (%.2fms avg)%n", 
                entry.getKey(), entry.getValue()));
        
        // Alert 3: Traffic spike
        Map<Integer, Long> recentHourlyTraffic = recentResponses.stream()
            .collect(Collectors.groupingBy(
                response -> response.getTimestamp().getHour(),
                Collectors.counting()
            ));
        
        OptionalLong maxTraffic = recentHourlyTraffic.values().stream()
            .mapToLong(Long::longValue)
            .max();
        
        if (maxTraffic.isPresent() && maxTraffic.getAsLong() > 1000) {
            System.out.printf("📈 TRAFFIC SPIKE: %d requests in one hour%n", maxTraffic.getAsLong());
        }
    }
    
    static class APIResponse {
        private LocalDateTime timestamp;
        private String endpoint;
        private int statusCode;
        private double responseTime;
        private String userAgent;
        
        public APIResponse(LocalDateTime timestamp, String endpoint, int statusCode, 
                          double responseTime, String userAgent) {
            this.timestamp = timestamp;
            this.endpoint = endpoint;
            this.statusCode = statusCode;
            this.responseTime = responseTime;
            this.userAgent = userAgent;
        }
        
        public LocalDateTime getTimestamp() { return timestamp; }
        public String getEndpoint() { return endpoint; }
        public int getStatusCode() { return statusCode; }
        public double getResponseTime() { return responseTime; }
        public String getUserAgent() { return userAgent; }
        
        @Override
        public String toString() {
            return String.format("[%s] %s %d %.2fms %s", 
                timestamp.format(DateTimeFormatter.ofPattern("HH:mm:ss")),
                endpoint, statusCode, responseTime, userAgent);
        }
    }
}
```

## Summary

### Real-World Applications of Java Streams

1. **Data Analytics**: Processing large datasets, calculating statistics, generating reports
2. **Log Analysis**: Parsing log files, finding patterns, monitoring system health
3. **File Processing**: CSV parsing, data transformation, batch processing
4. **API Monitoring**: Response time analysis, error tracking, performance monitoring
5. **Business Intelligence**: Sales analysis, customer segmentation, trend analysis
6. **Text Processing**: Content analysis, keyword extraction, sentiment analysis

### Key Benefits in Real-World Scenarios

- **Readability**: Complex data transformations expressed clearly
- **Maintainability**: Easy to modify and extend processing logic
- **Performance**: Parallel processing for large datasets
- **Composability**: Chain operations to build complex pipelines
- **Debugging**: Easy to add logging and monitoring
- **Testing**: Individual operations can be tested in isolation

### Best Practices for Production Use

1. **Error Handling**: Use Optional and proper exception handling
2. **Performance Monitoring**: Track stream operation performance
3. **Memory Management**: Be aware of memory usage with large datasets
4. **Parallel Processing**: Use parallel streams judiciously
5. **Data Validation**: Validate input data before processing
6. **Logging**: Add appropriate logging for troubleshooting
7. **Testing**: Write comprehensive tests for stream operations

These examples demonstrate how Java Streams can be effectively used to solve real-world data processing challenges, from simple filtering and aggregation to complex analytics and monitoring scenarios.
