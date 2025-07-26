# Java Streams - Parallel Streams and Performance

## 1. Understanding Parallel Streams

Parallel streams allow you to process data using multiple threads, potentially improving performance for CPU-intensive operations on large datasets.

### Basic Parallel Stream Creation
```java
import java.util.*;
import java.util.stream.IntStream;
import java.util.stream.Collectors;

public class ParallelStreamBasicsDemo {
    public static void main(String[] args) {
        List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        
        // Creating parallel streams
        
        // Method 1: From collection using parallelStream()
        numbers.parallelStream()
            .forEach(n -> System.out.println(Thread.currentThread().getName() + ": " + n));
        
        System.out.println("\n" + "=".repeat(50) + "\n");
        
        // Method 2: Converting sequential stream to parallel
        numbers.stream()
            .parallel()
            .forEach(n -> System.out.println(Thread.currentThread().getName() + ": " + n));
        
        System.out.println("\n" + "=".repeat(50) + "\n");
        
        // Method 3: Parallel range
        IntStream.range(1, 11)
            .parallel()
            .forEach(n -> System.out.println(Thread.currentThread().getName() + ": " + n));
        
        // Check if stream is parallel
        boolean isParallel = numbers.parallelStream().isParallel();
        System.out.println("\nIs parallel: " + isParallel);
        
        // Convert back to sequential
        List<Integer> sequentialResult = numbers.parallelStream()
            .sequential()  // Convert to sequential
            .map(n -> n * 2)
            .collect(Collectors.toList());
        
        System.out.println("Sequential result: " + sequentialResult);
        
        // Parallel operations maintain order with forEachOrdered
        System.out.println("\nParallel with ordered output:");
        numbers.parallelStream()
            .map(n -> n * n)
            .forEachOrdered(System.out::println);
    }
}
```

### Parallel vs Sequential Comparison
```java
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class ParallelVsSequentialDemo {
    public static void main(String[] args) {
        // Generate large dataset
        List<Integer> largeList = IntStream.rangeClosed(1, 10_000_000)
            .boxed()
            .collect(Collectors.toList());
        
        System.out.println("Dataset size: " + largeList.size());
        System.out.println("Available processors: " + Runtime.getRuntime().availableProcessors());
        
        // Test 1: Simple filtering and mapping
        System.out.println("\n=== Test 1: Filter and Map ===");
        
        long startTime = System.currentTimeMillis();
        long sequentialCount = largeList.stream()
            .filter(n -> n % 2 == 0)
            .mapToLong(n -> (long) n * n)
            .filter(n -> n > 1000)
            .count();
        long sequentialTime = System.currentTimeMillis() - startTime;
        
        startTime = System.currentTimeMillis();
        long parallelCount = largeList.parallelStream()
            .filter(n -> n % 2 == 0)
            .mapToLong(n -> (long) n * n)
            .filter(n -> n > 1000)
            .count();
        long parallelTime = System.currentTimeMillis() - startTime;
        
        System.out.println("Sequential result: " + sequentialCount + " (time: " + sequentialTime + "ms)");
        System.out.println("Parallel result: " + parallelCount + " (time: " + parallelTime + "ms)");
        System.out.println("Speedup: " + (double) sequentialTime / parallelTime + "x");
        
        // Test 2: CPU-intensive operation
        System.out.println("\n=== Test 2: CPU-intensive (isPrime) ===");
        
        List<Integer> testNumbers = IntStream.rangeClosed(10000, 10100)
            .boxed()
            .collect(Collectors.toList());
        
        startTime = System.currentTimeMillis();
        long sequentialPrimes = testNumbers.stream()
            .filter(ParallelVsSequentialDemo::isPrime)
            .count();
        long sequentialPrimeTime = System.currentTimeMillis() - startTime;
        
        startTime = System.currentTimeMillis();
        long parallelPrimes = testNumbers.parallelStream()
            .filter(ParallelVsSequentialDemo::isPrime)
            .count();
        long parallelPrimeTime = System.currentTimeMillis() - startTime;
        
        System.out.println("Sequential primes: " + sequentialPrimes + " (time: " + sequentialPrimeTime + "ms)");
        System.out.println("Parallel primes: " + parallelPrimes + " (time: " + parallelPrimeTime + "ms)");
        System.out.println("Speedup: " + (double) sequentialPrimeTime / parallelPrimeTime + "x");
        
        // Test 3: Collection operations
        System.out.println("\n=== Test 3: Collection Operations ===");
        
        startTime = System.currentTimeMillis();
        Map<Integer, List<Integer>> sequentialGrouping = largeList.stream()
            .limit(100_000)
            .collect(Collectors.groupingBy(n -> n % 10));
        long sequentialGroupTime = System.currentTimeMillis() - startTime;
        
        startTime = System.currentTimeMillis();
        Map<Integer, List<Integer>> parallelGrouping = largeList.parallelStream()
            .limit(100_000)
            .collect(Collectors.groupingBy(n -> n % 10));
        long parallelGroupTime = System.currentTimeMillis() - startTime;
        
        System.out.println("Sequential grouping time: " + sequentialGroupTime + "ms");
        System.out.println("Parallel grouping time: " + parallelGroupTime + "ms");
        System.out.println("Groups created: " + sequentialGrouping.size());
        System.out.println("Results match: " + sequentialGrouping.keySet().equals(parallelGrouping.keySet()));
    }
    
    // CPU-intensive operation for testing
    private static boolean isPrime(int number) {
        if (number < 2) return false;
        for (int i = 2; i <= Math.sqrt(number); i++) {
            if (number % i == 0) return false;
        }
        return true;
    }
}
```

## 2. When to Use Parallel Streams

### Good Candidates for Parallel Processing
```java
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class ParallelStreamCandidatesDemo {
    public static void main(String[] args) {
        // Good candidate 1: Large datasets with CPU-intensive operations
        System.out.println("=== Good Candidate 1: CPU-intensive operations ===");
        
        List<Integer> largeNumbers = IntStream.rangeClosed(1, 100_000)
            .boxed()
            .collect(Collectors.toList());
        
        long startTime = System.currentTimeMillis();
        double sequentialAvg = largeNumbers.stream()
            .mapToDouble(n -> Math.sqrt(n * n * n))  // CPU-intensive calculation
            .average()
            .orElse(0.0);
        long sequentialTime = System.currentTimeMillis() - startTime;
        
        startTime = System.currentTimeMillis();
        double parallelAvg = largeNumbers.parallelStream()
            .mapToDouble(n -> Math.sqrt(n * n * n))  // CPU-intensive calculation
            .average()
            .orElse(0.0);
        long parallelTime = System.currentTimeMillis() - startTime;
        
        System.out.println("Sequential time: " + sequentialTime + "ms");
        System.out.println("Parallel time: " + parallelTime + "ms");
        System.out.println("Results equal: " + (Math.abs(sequentialAvg - parallelAvg) < 0.001));
        
        // Good candidate 2: Independent operations (no shared state)
        System.out.println("\n=== Good Candidate 2: Independent operations ===");
        
        List<String> texts = Arrays.asList("hello", "world", "java", "parallel", "streams");
        Collections.addAll(texts, "performance", "optimization", "concurrent", "processing");
        
        // Replicate to create larger dataset
        texts = IntStream.range(0, 10000)
            .mapToObj(i -> texts.get(i % texts.size()))
            .collect(Collectors.toList());
        
        startTime = System.currentTimeMillis();
        List<String> sequentialProcessed = texts.stream()
            .filter(s -> s.length() > 4)
            .map(String::toUpperCase)
            .map(s -> s + "_PROCESSED")
            .sorted()
            .collect(Collectors.toList());
        long sequentialTextTime = System.currentTimeMillis() - startTime;
        
        startTime = System.currentTimeMillis();
        List<String> parallelProcessed = texts.parallelStream()
            .filter(s -> s.length() > 4)
            .map(String::toUpperCase)
            .map(s -> s + "_PROCESSED")
            .sorted()
            .collect(Collectors.toList());
        long parallelTextTime = System.currentTimeMillis() - startTime;
        
        System.out.println("Sequential text processing: " + sequentialTextTime + "ms");
        System.out.println("Parallel text processing: " + parallelTextTime + "ms");
        System.out.println("Results equal: " + sequentialProcessed.equals(parallelProcessed));
        
        // Good candidate 3: Stateless operations
        System.out.println("\n=== Good Candidate 3: Reduction operations ===");
        
        startTime = System.currentTimeMillis();
        OptionalInt sequentialMax = largeNumbers.stream()
            .mapToInt(Integer::intValue)
            .max();
        long sequentialMaxTime = System.currentTimeMillis() - startTime;
        
        startTime = System.currentTimeMillis();
        OptionalInt parallelMax = largeNumbers.parallelStream()
            .mapToInt(Integer::intValue)
            .max();
        long parallelMaxTime = System.currentTimeMillis() - startTime;
        
        System.out.println("Sequential max: " + sequentialMaxTime + "ms");
        System.out.println("Parallel max: " + parallelMaxTime + "ms");
        System.out.println("Results equal: " + sequentialMax.equals(parallelMax));
    }
}
```

### Poor Candidates for Parallel Processing
```java
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class ParallelStreamPoorCandidatesDemo {
    public static void main(String[] args) {
        // Poor candidate 1: Small datasets
        System.out.println("=== Poor Candidate 1: Small datasets ===");
        
        List<Integer> smallList = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        
        long startTime = System.nanoTime();
        int sequentialSum = smallList.stream()
            .mapToInt(Integer::intValue)
            .sum();
        long sequentialTime = System.nanoTime() - startTime;
        
        startTime = System.nanoTime();
        int parallelSum = smallList.parallelStream()
            .mapToInt(Integer::intValue)
            .sum();
        long parallelTime = System.nanoTime() - startTime;
        
        System.out.println("Sequential time: " + sequentialTime + "ns");
        System.out.println("Parallel time: " + parallelTime + "ns");
        System.out.println("Parallel is slower by: " + (parallelTime - sequentialTime) + "ns");
        
        // Poor candidate 2: Operations with shared mutable state
        System.out.println("\n=== Poor Candidate 2: Shared mutable state (DANGEROUS) ===");
        
        List<Integer> numbers = IntStream.rangeClosed(1, 1000).boxed().collect(Collectors.toList());
        List<Integer> sharedList = new ArrayList<>();  // Shared mutable state - DANGEROUS!
        
        // This is INCORRECT and may produce inconsistent results
        numbers.parallelStream()
            .filter(n -> n % 2 == 0)
            .forEach(sharedList::add);  // Race condition!
        
        System.out.println("Shared list size (may vary): " + sharedList.size());
        System.out.println("Expected size: " + numbers.stream().mapToInt(n -> n % 2 == 0 ? 1 : 0).sum());
        
        // Correct approach - use collectors
        List<Integer> correctList = numbers.parallelStream()
            .filter(n -> n % 2 == 0)
            .collect(Collectors.toList());
        System.out.println("Correct list size: " + correctList.size());
        
        // Poor candidate 3: Order-dependent operations
        System.out.println("\n=== Poor Candidate 3: Order-dependent operations ===");
        
        List<String> words = Arrays.asList("apple", "banana", "cherry", "date", "elderberry");
        
        // Sequential - order preserved
        System.out.println("Sequential forEach (order preserved):");
        words.stream()
            .forEach(word -> System.out.print(word + " "));
        
        System.out.println("\n\nParallel forEach (order NOT preserved):");
        words.parallelStream()
            .forEach(word -> System.out.print(word + " "));
        
        System.out.println("\n\nParallel forEachOrdered (order preserved but slower):");
        words.parallelStream()
            .forEachOrdered(word -> System.out.print(word + " "));
        
        // Poor candidate 4: I/O bound operations
        System.out.println("\n\n=== Poor Candidate 4: I/O simulation ===");
        
        List<String> urls = Arrays.asList("url1", "url2", "url3", "url4", "url5");
        
        startTime = System.currentTimeMillis();
        List<String> sequentialResults = urls.stream()
            .map(ParallelStreamPoorCandidatesDemo::simulateHttpCall)
            .collect(Collectors.toList());
        long sequentialIOTime = System.currentTimeMillis() - startTime;
        
        startTime = System.currentTimeMillis();
        List<String> parallelResults = urls.parallelStream()
            .map(ParallelStreamPoorCandidatesDemo::simulateHttpCall)
            .collect(Collectors.toList());
        long parallelIOTime = System.currentTimeMillis() - startTime;
        
        System.out.println("Sequential I/O time: " + sequentialIOTime + "ms");
        System.out.println("Parallel I/O time: " + parallelIOTime + "ms");
        System.out.println("Note: For I/O operations, consider CompletableFuture instead");
    }
    
    private static String simulateHttpCall(String url) {
        try {
            Thread.sleep(100);  // Simulate I/O delay
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        return "Response from " + url;
    }
}
```

## 3. Thread Safety and Parallel Streams

### Thread Safety Considerations
```java
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class ThreadSafetyDemo {
    public static void main(String[] args) {
        // Problem: Non-thread-safe operations
        System.out.println("=== Thread Safety Issues ===");
        
        List<Integer> numbers = IntStream.rangeClosed(1, 10000).boxed().collect(Collectors.toList());
        
        // WRONG: Using non-thread-safe collection
        List<Integer> unsafeList = new ArrayList<>();
        numbers.parallelStream()
            .filter(n -> n % 2 == 0)
            .forEach(unsafeList::add);  // Race condition!
        
        System.out.println("Unsafe list size: " + unsafeList.size());
        System.out.println("Expected size: " + numbers.stream().mapToInt(n -> n % 2 == 0 ? 1 : 0).sum());
        
        // CORRECT: Using thread-safe collection
        List<Integer> safeList = new CopyOnWriteArrayList<>();
        numbers.parallelStream()
            .filter(n -> n % 2 == 0)
            .forEach(safeList::add);
        
        System.out.println("Safe list size: " + safeList.size());
        
        // BETTER: Using collectors (recommended)
        List<Integer> collectedList = numbers.parallelStream()
            .filter(n -> n % 2 == 0)
            .collect(Collectors.toList());
        
        System.out.println("Collected list size: " + collectedList.size());
        
        // Atomic operations for counters
        System.out.println("\n=== Atomic Operations ===");
        
        AtomicInteger atomicCounter = new AtomicInteger(0);
        int[] unsafeCounter = {0};  // Not thread-safe
        
        numbers.parallelStream()
            .filter(n -> n % 3 == 0)
            .forEach(n -> {
                atomicCounter.incrementAndGet();  // Thread-safe
                unsafeCounter[0]++;               // NOT thread-safe
            });
        
        System.out.println("Atomic counter: " + atomicCounter.get());
        System.out.println("Unsafe counter: " + unsafeCounter[0]);
        System.out.println("Expected count: " + numbers.stream().mapToInt(n -> n % 3 == 0 ? 1 : 0).sum());
        
        // Thread-safe maps
        System.out.println("\n=== Thread-safe Maps ===");
        
        Map<String, Integer> unsafeMap = new HashMap<>();
        Map<String, Integer> safeMap = new ConcurrentHashMap<>();
        
        List<String> words = Arrays.asList("apple", "banana", "apple", "cherry", "banana", "apple");
        // Replicate to create larger dataset
        words = IntStream.range(0, 1000)
            .mapToObj(i -> words.get(i % words.size()))
            .collect(Collectors.toList());
        
        // Using unsafe map (may lose updates)
        words.parallelStream()
            .forEach(word -> unsafeMap.merge(word, 1, Integer::sum));
        
        // Using safe map
        words.parallelStream()
            .forEach(word -> safeMap.merge(word, 1, Integer::sum));
        
        System.out.println("Unsafe map: " + unsafeMap);
        System.out.println("Safe map: " + safeMap);
        
        // Best approach: Use collectors
        Map<String, Long> countMap = words.parallelStream()
            .collect(Collectors.groupingBy(
                word -> word,
                Collectors.counting()));
        
        System.out.println("Collector map: " + countMap);
    }
}
```

### Common Thread Safety Patterns
```java
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.LongAdder;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class ThreadSafetyPatternsDemo {
    public static void main(String[] args) {
        List<Transaction> transactions = generateTransactions(100_000);
        
        // Pattern 1: Use atomic variables for simple aggregations
        System.out.println("=== Pattern 1: Atomic Variables ===");
        
        AtomicLong totalAmount = new AtomicLong(0);
        LongAdder transactionCount = new LongAdder();
        
        long startTime = System.currentTimeMillis();
        transactions.parallelStream()
            .filter(t -> t.getAmount() > 100)
            .forEach(t -> {
                totalAmount.addAndGet((long) t.getAmount());
                transactionCount.increment();
            });
        long atomicTime = System.currentTimeMillis() - startTime;
        
        System.out.println("Total amount (atomic): " + totalAmount.get());
        System.out.println("Count (adder): " + transactionCount.sum());
        System.out.println("Time: " + atomicTime + "ms");
        
        // Pattern 2: Use concurrent collections
        System.out.println("\n=== Pattern 2: Concurrent Collections ===");
        
        ConcurrentHashMap<String, LongAdder> categoryTotals = new ConcurrentHashMap<>();
        
        startTime = System.currentTimeMillis();
        transactions.parallelStream()
            .forEach(t -> {
                categoryTotals.computeIfAbsent(t.getCategory(), k -> new LongAdder())
                    .add((long) t.getAmount());
            });
        long concurrentTime = System.currentTimeMillis() - startTime;
        
        System.out.println("Category totals:");
        categoryTotals.forEach((category, adder) -> 
            System.out.println("  " + category + ": " + adder.sum()));
        System.out.println("Time: " + concurrentTime + "ms");
        
        // Pattern 3: Use collectors (recommended)
        System.out.println("\n=== Pattern 3: Collectors (Recommended) ===");
        
        startTime = System.currentTimeMillis();
        Map<String, Double> collectorTotals = transactions.parallelStream()
            .collect(Collectors.groupingBy(
                Transaction::getCategory,
                Collectors.summingDouble(Transaction::getAmount)));
        long collectorTime = System.currentTimeMillis() - startTime;
        
        System.out.println("Collector totals: " + collectorTotals);
        System.out.println("Time: " + collectorTime + "ms");
        
        // Pattern 4: Reduce operations
        System.out.println("\n=== Pattern 4: Reduce Operations ===");
        
        startTime = System.currentTimeMillis();
        double totalWithReduce = transactions.parallelStream()
            .filter(t -> "Electronics".equals(t.getCategory()))
            .mapToDouble(Transaction::getAmount)
            .reduce(0.0, Double::sum);
        long reduceTime = System.currentTimeMillis() - startTime;
        
        System.out.println("Electronics total (reduce): " + totalWithReduce);
        System.out.println("Time: " + reduceTime + "ms");
        
        // Pattern 5: Custom thread-safe accumulator
        System.out.println("\n=== Pattern 5: Custom Accumulator ===");
        
        startTime = System.currentTimeMillis();
        TransactionStatistics stats = transactions.parallelStream()
            .filter(t -> t.getAmount() > 50)
            .collect(
                TransactionStatistics::new,
                TransactionStatistics::accept,
                TransactionStatistics::combine);
        long customTime = System.currentTimeMillis() - startTime;
        
        System.out.println("Custom statistics: " + stats);
        System.out.println("Time: " + customTime + "ms");
    }
    
    private static List<Transaction> generateTransactions(int count) {
        String[] categories = {"Electronics", "Clothing", "Books", "Food", "Travel"};
        Random random = new Random();
        
        return IntStream.range(0, count)
            .mapToObj(i -> new Transaction(
                "TXN" + i,
                categories[random.nextInt(categories.length)],
                random.nextDouble() * 500 + 10))
            .collect(Collectors.toList());
    }
    
    static class Transaction {
        private final String id;
        private final String category;
        private final double amount;
        
        public Transaction(String id, String category, double amount) {
            this.id = id;
            this.category = category;
            this.amount = amount;
        }
        
        public String getId() { return id; }
        public String getCategory() { return category; }
        public double getAmount() { return amount; }
        
        @Override
        public String toString() {
            return String.format("%s: %s $%.2f", id, category, amount);
        }
    }
    
    static class TransactionStatistics {
        private volatile int count = 0;
        private volatile double sum = 0.0;
        private volatile double min = Double.MAX_VALUE;
        private volatile double max = Double.MIN_VALUE;
        
        public synchronized void accept(Transaction transaction) {
            count++;
            sum += transaction.getAmount();
            min = Math.min(min, transaction.getAmount());
            max = Math.max(max, transaction.getAmount());
        }
        
        public synchronized TransactionStatistics combine(TransactionStatistics other) {
            TransactionStatistics result = new TransactionStatistics();
            result.count = this.count + other.count;
            result.sum = this.sum + other.sum;
            result.min = Math.min(this.min, other.min);
            result.max = Math.max(this.max, other.max);
            return result;
        }
        
        public double getAverage() {
            return count > 0 ? sum / count : 0.0;
        }
        
        @Override
        public String toString() {
            return String.format("Stats{count=%d, sum=%.2f, avg=%.2f, min=%.2f, max=%.2f}",
                count, sum, getAverage(), min, max);
        }
    }
}
```

## 4. Performance Optimization and Best Practices

### ForkJoinPool Configuration
```java
import java.util.*;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class ForkJoinPoolDemo {
    public static void main(String[] args) {
        List<Integer> largeList = IntStream.rangeClosed(1, 1_000_000)
            .boxed()
            .collect(Collectors.toList());
        
        System.out.println("Available processors: " + Runtime.getRuntime().availableProcessors());
        System.out.println("Default ForkJoinPool parallelism: " + 
            ForkJoinPool.commonPool().getParallelism());
        
        // Test with default common pool
        long startTime = System.currentTimeMillis();
        long defaultResult = largeList.parallelStream()
            .filter(n -> isPrime(n))
            .count();
        long defaultTime = System.currentTimeMillis() - startTime;
        
        System.out.println("Default pool result: " + defaultResult + " (time: " + defaultTime + "ms)");
        
        // Test with custom ForkJoinPool
        ForkJoinPool customThreadPool = new ForkJoinPool(2);
        try {
            startTime = System.currentTimeMillis();
            long customResult = customThreadPool.submit(() ->
                largeList.parallelStream()
                    .filter(n -> isPrime(n))
                    .count()
            ).get();
            long customTime = System.currentTimeMillis() - startTime;
            
            System.out.println("Custom pool (2 threads) result: " + customResult + 
                " (time: " + customTime + "ms)");
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            customThreadPool.shutdown();
        }
        
        // Test with larger custom pool
        ForkJoinPool largeThreadPool = new ForkJoinPool(8);
        try {
            startTime = System.currentTimeMillis();
            long largeResult = largeThreadPool.submit(() ->
                largeList.parallelStream()
                    .filter(n -> isPrime(n))
                    .count()
            ).get();
            long largeTime = System.currentTimeMillis() - startTime;
            
            System.out.println("Large pool (8 threads) result: " + largeResult + 
                " (time: " + largeTime + "ms)");
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            largeThreadPool.shutdown();
        }
        
        // Demonstrate system property for default parallelism
        System.out.println("\nTo change default parallelism, use:");
        System.out.println("-Djava.util.concurrent.ForkJoinPool.common.parallelism=N");
    }
    
    private static boolean isPrime(int number) {
        if (number < 2) return false;
        if (number == 2) return true;
        if (number % 2 == 0) return false;
        
        for (int i = 3; i <= Math.sqrt(number); i += 2) {
            if (number % i == 0) return false;
        }
        return true;
    }
}
```

### Stream Spliterator and Performance
```java
import java.util.*;
import java.util.stream.IntStream;
import java.util.stream.StreamSupport;

public class SpliteratorPerformanceDemo {
    public static void main(String[] args) {
        // Different data structures have different splitting characteristics
        
        // ArrayList - excellent for parallel processing (random access)
        List<Integer> arrayList = new ArrayList<>();
        IntStream.rangeClosed(1, 1_000_000).forEach(arrayList::add);
        
        // LinkedList - poor for parallel processing (sequential access)
        List<Integer> linkedList = new LinkedList<>();
        IntStream.rangeClosed(1, 1_000_000).forEach(linkedList::add);
        
        // Array - excellent for parallel processing
        int[] array = IntStream.rangeClosed(1, 1_000_000).toArray();
        
        // HashSet - good for parallel processing
        Set<Integer> hashSet = new HashSet<>();
        IntStream.rangeClosed(1, 1_000_000).forEach(hashSet::add);
        
        System.out.println("=== Performance Comparison ===");
        
        // Test ArrayList
        long startTime = System.currentTimeMillis();
        long arrayListResult = arrayList.parallelStream()
            .filter(n -> n % 1000 == 0)
            .count();
        long arrayListTime = System.currentTimeMillis() - startTime;
        
        // Test LinkedList
        startTime = System.currentTimeMillis();
        long linkedListResult = linkedList.parallelStream()
            .filter(n -> n % 1000 == 0)
            .count();
        long linkedListTime = System.currentTimeMillis() - startTime;
        
        // Test Array
        startTime = System.currentTimeMillis();
        long arrayResult = Arrays.stream(array)
            .parallel()
            .filter(n -> n % 1000 == 0)
            .count();
        long arrayTime = System.currentTimeMillis() - startTime;
        
        // Test HashSet
        startTime = System.currentTimeMillis();
        long hashSetResult = hashSet.parallelStream()
            .filter(n -> n % 1000 == 0)
            .count();
        long hashSetTime = System.currentTimeMillis() - startTime;
        
        System.out.println("ArrayList: " + arrayListResult + " (time: " + arrayListTime + "ms)");
        System.out.println("LinkedList: " + linkedListResult + " (time: " + linkedListTime + "ms)");
        System.out.println("Array: " + arrayResult + " (time: " + arrayTime + "ms)");
        System.out.println("HashSet: " + hashSetResult + " (time: " + hashSetTime + "ms)");
        
        // Spliterator characteristics
        System.out.println("\n=== Spliterator Characteristics ===");
        
        Spliterator<Integer> arrayListSpliterator = arrayList.spliterator();
        Spliterator<Integer> linkedListSpliterator = linkedList.spliterator();
        
        System.out.println("ArrayList spliterator characteristics: " + 
            getCharacteristics(arrayListSpliterator));
        System.out.println("LinkedList spliterator characteristics: " + 
            getCharacteristics(linkedListSpliterator));
        
        System.out.println("ArrayList estimated size: " + arrayListSpliterator.estimateSize());
        System.out.println("LinkedList estimated size: " + linkedListSpliterator.estimateSize());
    }
    
    private static String getCharacteristics(Spliterator<?> spliterator) {
        List<String> characteristics = new ArrayList<>();
        
        if (spliterator.hasCharacteristics(Spliterator.ORDERED))
            characteristics.add("ORDERED");
        if (spliterator.hasCharacteristics(Spliterator.DISTINCT))
            characteristics.add("DISTINCT");
        if (spliterator.hasCharacteristics(Spliterator.SORTED))
            characteristics.add("SORTED");
        if (spliterator.hasCharacteristics(Spliterator.SIZED))
            characteristics.add("SIZED");
        if (spliterator.hasCharacteristics(Spliterator.NONNULL))
            characteristics.add("NONNULL");
        if (spliterator.hasCharacteristics(Spliterator.IMMUTABLE))
            characteristics.add("IMMUTABLE");
        if (spliterator.hasCharacteristics(Spliterator.CONCURRENT))
            characteristics.add("CONCURRENT");
        if (spliterator.hasCharacteristics(Spliterator.SUBSIZED))
            characteristics.add("SUBSIZED");
        
        return String.join(", ", characteristics);
    }
}
```

### Performance Best Practices
```java
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class PerformanceBestPracticesDemo {
    public static void main(String[] args) {
        // Best Practice 1: Consider data size
        System.out.println("=== Best Practice 1: Data Size Considerations ===");
        
        // Small dataset - sequential is often faster
        List<Integer> smallList = IntStream.rangeClosed(1, 100).boxed().collect(Collectors.toList());
        
        long startTime = System.nanoTime();
        int sequentialSum = smallList.stream().mapToInt(Integer::intValue).sum();
        long sequentialTime = System.nanoTime() - startTime;
        
        startTime = System.nanoTime();
        int parallelSum = smallList.parallelStream().mapToInt(Integer::intValue).sum();
        long parallelTime = System.nanoTime() - startTime;
        
        System.out.println("Small dataset (100 elements):");
        System.out.println("  Sequential: " + sequentialTime + "ns");
        System.out.println("  Parallel: " + parallelTime + "ns");
        System.out.println("  Sequential is faster: " + (parallelTime > sequentialTime));
        
        // Best Practice 2: Avoid boxing/unboxing
        System.out.println("\n=== Best Practice 2: Avoid Boxing ===");
        
        List<Integer> numbers = IntStream.rangeClosed(1, 1_000_000).boxed().collect(Collectors.toList());
        
        // With boxing (slower)
        startTime = System.currentTimeMillis();
        OptionalDouble avgWithBoxing = numbers.parallelStream()
            .mapToDouble(Integer::doubleValue)  // Boxing/unboxing
            .average();
        long boxingTime = System.currentTimeMillis() - startTime;
        
        // Without boxing (faster)
        startTime = System.currentTimeMillis();
        OptionalDouble avgWithoutBoxing = numbers.parallelStream()
            .mapToInt(Integer::intValue)        // Primitive stream
            .average();
        long primitiveTime = System.currentTimeMillis() - startTime;
        
        System.out.println("With boxing: " + boxingTime + "ms");
        System.out.println("With primitives: " + primitiveTime + "ms");
        
        // Best Practice 3: Order operations efficiently
        System.out.println("\n=== Best Practice 3: Operation Order ===");
        
        List<String> words = generateWords(100_000);
        
        // Inefficient: expensive operation first
        startTime = System.currentTimeMillis();
        long inefficientCount = words.parallelStream()
            .map(String::toUpperCase)           // Expensive operation first
            .filter(s -> s.length() > 5)       // Cheap filter after
            .count();
        long inefficientTime = System.currentTimeMillis() - startTime;
        
        // Efficient: cheap operation first
        startTime = System.currentTimeMillis();
        long efficientCount = words.parallelStream()
            .filter(s -> s.length() > 5)       // Cheap filter first
            .map(String::toUpperCase)           // Expensive operation after
            .count();
        long efficientTime = System.currentTimeMillis() - startTime;
        
        System.out.println("Inefficient order: " + inefficientTime + "ms");
        System.out.println("Efficient order: " + efficientTime + "ms");
        System.out.println("Results equal: " + (inefficientCount == efficientCount));
        
        // Best Practice 4: Use appropriate collectors
        System.out.println("\n=== Best Practice 4: Collector Choice ===");
        
        // Regular grouping
        startTime = System.currentTimeMillis();
        Map<Integer, List<String>> regularGrouping = words.parallelStream()
            .collect(Collectors.groupingBy(String::length));
        long regularTime = System.currentTimeMillis() - startTime;
        
        // Concurrent grouping (better for parallel)
        startTime = System.currentTimeMillis();
        Map<Integer, List<String>> concurrentGrouping = words.parallelStream()
            .collect(Collectors.groupingByConcurrent(String::length));
        long concurrentTime = System.currentTimeMillis() - startTime;
        
        System.out.println("Regular grouping: " + regularTime + "ms");
        System.out.println("Concurrent grouping: " + concurrentTime + "ms");
        System.out.println("Groups count equal: " + 
            (regularGrouping.size() == concurrentGrouping.size()));
        
        // Best Practice 5: Consider short-circuiting
        System.out.println("\n=== Best Practice 5: Short-circuiting ===");
        
        startTime = System.currentTimeMillis();
        boolean anyLongWord = words.parallelStream()
            .anyMatch(s -> s.length() > 15);    // Short-circuits on first match
        long shortCircuitTime = System.currentTimeMillis() - startTime;
        
        startTime = System.currentTimeMillis();
        long longWordCount = words.parallelStream()
            .filter(s -> s.length() > 15)
            .count();                           // Processes all elements
        long fullProcessTime = System.currentTimeMillis() - startTime;
        
        System.out.println("Short-circuit (anyMatch): " + shortCircuitTime + "ms");
        System.out.println("Full process (count): " + fullProcessTime + "ms");
        System.out.println("Found long words: " + anyLongWord + " (count: " + longWordCount + ")");
    }
    
    private static List<String> generateWords(int count) {
        String[] words = {"apple", "banana", "cherry", "date", "elderberry", "fig", "grape", "honeydew"};
        Random random = new Random();
        
        return IntStream.range(0, count)
            .mapToObj(i -> words[random.nextInt(words.length)] + i)
            .collect(Collectors.toList());
    }
}
```

## Summary

### When to Use Parallel Streams

**Use Parallel Streams When:**
- Large datasets (typically > 10,000 elements)
- CPU-intensive operations
- Independent operations (no shared state)
- Stateless operations
- Operations that benefit from parallelization (filtering, mapping, reduction)

**Avoid Parallel Streams When:**
- Small datasets
- I/O-bound operations
- Operations with side effects
- Order-dependent operations
- Short-running operations

### Performance Optimization Tips

1. **Data Structure Choice**: Use ArrayList, arrays, or other random-access structures
2. **Operation Order**: Place cheap operations (filters) before expensive ones (maps)
3. **Avoid Boxing**: Use primitive streams when working with numbers
4. **Thread Safety**: Use collectors instead of manual accumulation
5. **Pool Configuration**: Consider custom ForkJoinPool for specific needs
6. **Short-circuiting**: Use operations like `findFirst()`, `anyMatch()` when possible

### Thread Safety Guidelines

- **Use Collectors**: Preferred way to accumulate results
- **Atomic Operations**: For simple counters and accumulators
- **Concurrent Collections**: For complex shared state
- **Avoid Shared Mutable State**: Design operations to be independent
- **Synchronization**: Only when absolutely necessary and carefully implemented

Parallel streams can provide significant performance improvements when used correctly, but they require careful consideration of the workload characteristics and thread safety requirements.
