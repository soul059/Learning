# Collections Framework in Java

## 1. Overview of Collections Framework

The Java Collections Framework provides a unified architecture for representing and manipulating collections of objects. It includes interfaces, implementations, and algorithms.

### Core Interfaces Hierarchy
```
Collection<E>
├── List<E>
├── Set<E>
│   └── SortedSet<E>
│       └── NavigableSet<E>
└── Queue<E>
    └── Deque<E>

Map<K,V>
└── SortedMap<K,V>
    └── NavigableMap<K,V>
```

## 2. List Interface and Implementations

### ArrayList
Dynamic array implementation with automatic resizing.

```java
import java.util.*;

public class ArrayListExample {
    public static void main(String[] args) {
        // Creation and basic operations
        List<String> fruits = new ArrayList<>();
        
        // Adding elements
        fruits.add("Apple");
        fruits.add("Banana");
        fruits.add("Cherry");
        fruits.add(1, "Blueberry"); // Insert at index
        
        // Accessing elements
        String first = fruits.get(0);
        int size = fruits.size();
        
        // Modifying elements
        fruits.set(2, "Cranberry"); // Replace element at index
        
        // Removing elements
        fruits.remove("Banana");
        fruits.remove(0); // Remove by index
        
        // Bulk operations
        List<String> moreFruits = Arrays.asList("Orange", "Grape");
        fruits.addAll(moreFruits);
        
        // Iteration
        for (String fruit : fruits) {
            System.out.println(fruit);
        }
        
        // Using streams (Java 8+)
        fruits.stream()
              .filter(fruit -> fruit.startsWith("C"))
              .forEach(System.out::println);
    }
}
```

#### ArrayList Performance Characteristics
- **Access**: O(1) - Random access by index
- **Search**: O(n) - Linear search
- **Insertion**: O(1) amortized at end, O(n) at arbitrary position
- **Deletion**: O(1) at end, O(n) at arbitrary position

### LinkedList
Doubly-linked list implementation.

```java
public class LinkedListExample {
    public static void main(String[] args) {
        LinkedList<Integer> numbers = new LinkedList<>();
        
        // Adding elements
        numbers.add(1);
        numbers.add(2);
        numbers.add(3);
        
        // LinkedList specific methods
        numbers.addFirst(0);
        numbers.addLast(4);
        
        // Queue operations
        numbers.offer(5);    // Add to end
        Integer head = numbers.poll();  // Remove from beginning
        Integer peek = numbers.peek();  // Look at first element
        
        // Stack operations
        numbers.push(10);    // Add to beginning
        Integer top = numbers.pop();    // Remove from beginning
        
        // Deque operations
        numbers.offerFirst(-1);
        numbers.offerLast(100);
        Integer first = numbers.pollFirst();
        Integer last = numbers.pollLast();
        
        System.out.println(numbers);
    }
}
```

#### LinkedList Performance Characteristics
- **Access**: O(n) - Sequential access required
- **Search**: O(n) - Linear search
- **Insertion**: O(1) at beginning/end, O(n) at arbitrary position
- **Deletion**: O(1) at beginning/end, O(n) at arbitrary position

### Vector (Legacy - Synchronized)
```java
public class VectorExample {
    public static void main(String[] args) {
        Vector<String> vector = new Vector<>();
        
        // Similar to ArrayList but synchronized
        vector.add("Element 1");
        vector.add("Element 2");
        
        // Vector specific methods
        vector.addElement("Element 3");
        vector.insertElementAt("Element 0", 0);
        vector.removeElement("Element 1");
        
        // Enumeration (legacy iteration)
        Enumeration<String> enumeration = vector.elements();
        while (enumeration.hasMoreElements()) {
            System.out.println(enumeration.nextElement());
        }
    }
}
```

## 3. Set Interface and Implementations

### HashSet
Hash table based implementation - unordered, no duplicates.

```java
public class HashSetExample {
    public static void main(String[] args) {
        Set<String> countries = new HashSet<>();
        
        // Adding elements
        countries.add("USA");
        countries.add("Canada");
        countries.add("Mexico");
        countries.add("USA"); // Duplicate - won't be added
        
        // Checking existence
        boolean hasUSA = countries.contains("USA");
        
        // Set operations
        Set<String> europeanCountries = new HashSet<>();
        europeanCountries.add("France");
        europeanCountries.add("Germany");
        europeanCountries.add("Spain");
        
        // Union
        Set<String> allCountries = new HashSet<>(countries);
        allCountries.addAll(europeanCountries);
        
        // Intersection
        Set<String> intersection = new HashSet<>(countries);
        intersection.retainAll(europeanCountries);
        
        // Difference
        Set<String> difference = new HashSet<>(countries);
        difference.removeAll(europeanCountries);
        
        System.out.println("Countries: " + countries);
        System.out.println("Size: " + countries.size());
    }
}
```

### LinkedHashSet
Maintains insertion order.

```java
public class LinkedHashSetExample {
    public static void main(String[] args) {
        Set<Integer> numbers = new LinkedHashSet<>();
        
        numbers.add(3);
        numbers.add(1);
        numbers.add(4);
        numbers.add(1); // Duplicate
        numbers.add(5);
        
        // Maintains insertion order: [3, 1, 4, 5]
        System.out.println(numbers);
    }
}
```

### TreeSet
Sorted set implementation using Red-Black tree.

```java
public class TreeSetExample {
    public static void main(String[] args) {
        TreeSet<Integer> sortedNumbers = new TreeSet<>();
        
        sortedNumbers.add(5);
        sortedNumbers.add(2);
        sortedNumbers.add(8);
        sortedNumbers.add(1);
        sortedNumbers.add(9);
        
        // Automatically sorted: [1, 2, 5, 8, 9]
        System.out.println(sortedNumbers);
        
        // NavigableSet methods
        Integer first = sortedNumbers.first();
        Integer last = sortedNumbers.last();
        Integer lower = sortedNumbers.lower(5);  // Largest element < 5
        Integer higher = sortedNumbers.higher(5); // Smallest element > 5
        
        // Subset operations
        SortedSet<Integer> headSet = sortedNumbers.headSet(5);    // < 5
        SortedSet<Integer> tailSet = sortedNumbers.tailSet(5);    // >= 5
        SortedSet<Integer> subSet = sortedNumbers.subSet(2, 8);   // [2, 8)
        
        // Descending iteration
        NavigableSet<Integer> descendingSet = sortedNumbers.descendingSet();
        System.out.println("Descending: " + descendingSet);
    }
}
```

### Custom Objects in TreeSet
```java
class Person implements Comparable<Person> {
    private String name;
    private int age;
    
    public Person(String name, int age) {
        this.name = name;
        this.age = age;
    }
    
    @Override
    public int compareTo(Person other) {
        // Primary sort by age, secondary by name
        int ageComparison = Integer.compare(this.age, other.age);
        return ageComparison != 0 ? ageComparison : this.name.compareTo(other.name);
    }
    
    @Override
    public String toString() {
        return name + "(" + age + ")";
    }
}

// Using custom comparator
TreeSet<Person> peopleByName = new TreeSet<>(Comparator.comparing(Person::getName));
TreeSet<Person> peopleByAge = new TreeSet<>(Comparator.comparing(Person::getAge).reversed());
```

## 4. Queue Interface and Implementations

### PriorityQueue
Heap-based priority queue.

```java
public class PriorityQueueExample {
    public static void main(String[] args) {
        // Natural ordering (min-heap for integers)
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();
        
        minHeap.offer(5);
        minHeap.offer(2);
        minHeap.offer(8);
        minHeap.offer(1);
        
        // Poll elements in priority order
        while (!minHeap.isEmpty()) {
            System.out.println(minHeap.poll()); // 1, 2, 5, 8
        }
        
        // Max-heap using custom comparator
        PriorityQueue<Integer> maxHeap = new PriorityQueue<>(Collections.reverseOrder());
        maxHeap.addAll(Arrays.asList(5, 2, 8, 1));
        
        while (!maxHeap.isEmpty()) {
            System.out.println(maxHeap.poll()); // 8, 5, 2, 1
        }
        
        // Custom objects with priority
        PriorityQueue<Task> taskQueue = new PriorityQueue<>(
            Comparator.comparing(Task::getPriority).reversed()
        );
        
        taskQueue.offer(new Task("Low priority task", 1));
        taskQueue.offer(new Task("High priority task", 10));
        taskQueue.offer(new Task("Medium priority task", 5));
        
        while (!taskQueue.isEmpty()) {
            System.out.println(taskQueue.poll());
        }
    }
}

class Task {
    private String description;
    private int priority;
    
    public Task(String description, int priority) {
        this.description = description;
        this.priority = priority;
    }
    
    public int getPriority() { return priority; }
    
    @Override
    public String toString() {
        return description + " (Priority: " + priority + ")";
    }
}
```

### ArrayDeque
Resizable array implementation of Deque interface.

```java
public class ArrayDequeExample {
    public static void main(String[] args) {
        ArrayDeque<String> deque = new ArrayDeque<>();
        
        // Add to both ends
        deque.addFirst("First");
        deque.addLast("Last");
        deque.offerFirst("New First");
        deque.offerLast("New Last");
        
        // Remove from both ends
        String first = deque.removeFirst();
        String last = deque.removeLast();
        
        // Peek at both ends
        String peekFirst = deque.peekFirst();
        String peekLast = deque.peekLast();
        
        // Use as stack
        deque.push("Stack Element 1");
        deque.push("Stack Element 2");
        String popped = deque.pop();
        
        // Use as queue
        deque.offer("Queue Element 1");
        deque.offer("Queue Element 2");
        String polled = deque.poll();
        
        System.out.println(deque);
    }
}
```

## 5. Map Interface and Implementations

### HashMap
Hash table based implementation.

```java
public class HashMapExample {
    public static void main(String[] args) {
        Map<String, Integer> studentGrades = new HashMap<>();
        
        // Adding key-value pairs
        studentGrades.put("Alice", 95);
        studentGrades.put("Bob", 87);
        studentGrades.put("Charlie", 92);
        studentGrades.put("Alice", 98); // Updates existing value
        
        // Accessing values
        Integer aliceGrade = studentGrades.get("Alice");
        Integer defaultGrade = studentGrades.getOrDefault("David", 0);
        
        // Checking existence
        boolean hasAlice = studentGrades.containsKey("Alice");
        boolean hasGrade95 = studentGrades.containsValue(95);
        
        // Iteration methods
        // 1. Key set
        for (String student : studentGrades.keySet()) {
            System.out.println(student + ": " + studentGrades.get(student));
        }
        
        // 2. Entry set (more efficient)
        for (Map.Entry<String, Integer> entry : studentGrades.entrySet()) {
            System.out.println(entry.getKey() + ": " + entry.getValue());
        }
        
        // 3. Values
        for (Integer grade : studentGrades.values()) {
            System.out.println("Grade: " + grade);
        }
        
        // 4. Using streams
        studentGrades.entrySet().stream()
            .filter(entry -> entry.getValue() > 90)
            .forEach(entry -> System.out.println(entry.getKey() + " has high grade"));
        
        // Advanced operations (Java 8+)
        studentGrades.putIfAbsent("David", 85);
        studentGrades.computeIfAbsent("Eve", k -> 0);
        studentGrades.computeIfPresent("Alice", (k, v) -> v + 5);
        studentGrades.merge("Bob", 5, Integer::sum); // Add 5 to Bob's grade
    }
}
```

### LinkedHashMap
Maintains insertion or access order.

```java
public class LinkedHashMapExample {
    public static void main(String[] args) {
        // Insertion order
        Map<String, String> insertionOrder = new LinkedHashMap<>();
        insertionOrder.put("c", "Cherry");
        insertionOrder.put("a", "Apple");
        insertionOrder.put("b", "Banana");
        
        System.out.println("Insertion order: " + insertionOrder);
        
        // Access order LRU cache
        Map<String, String> lruCache = new LinkedHashMap<String, String>(16, 0.75f, true) {
            @Override
            protected boolean removeEldestEntry(Map.Entry<String, String> eldest) {
                return size() > 3; // Keep only 3 elements
            }
        };
        
        lruCache.put("1", "First");
        lruCache.put("2", "Second");
        lruCache.put("3", "Third");
        lruCache.get("1"); // Access first element
        lruCache.put("4", "Fourth"); // This will remove "2" (least recently used)
        
        System.out.println("LRU Cache: " + lruCache);
    }
}
```

### TreeMap
Sorted map implementation using Red-Black tree.

```java
public class TreeMapExample {
    public static void main(String[] args) {
        TreeMap<String, Integer> sortedMap = new TreeMap<>();
        
        sortedMap.put("Zebra", 1);
        sortedMap.put("Apple", 2);
        sortedMap.put("Banana", 3);
        sortedMap.put("Cherry", 4);
        
        // Automatically sorted by keys
        System.out.println(sortedMap); // {Apple=2, Banana=3, Cherry=4, Zebra=1}
        
        // NavigableMap methods
        String firstKey = sortedMap.firstKey();
        String lastKey = sortedMap.lastKey();
        String lowerKey = sortedMap.lowerKey("Cherry");
        String higherKey = sortedMap.higherKey("Cherry");
        
        // Subset operations
        SortedMap<String, Integer> headMap = sortedMap.headMap("Cherry");
        SortedMap<String, Integer> tailMap = sortedMap.tailMap("Cherry");
        SortedMap<String, Integer> subMap = sortedMap.subMap("Apple", "Zebra");
        
        // Descending map
        NavigableMap<String, Integer> descendingMap = sortedMap.descendingMap();
        System.out.println("Descending: " + descendingMap);
    }
}
```

## 6. Concurrent Collections

### ConcurrentHashMap
Thread-safe hash map without synchronizing the entire map.

```java
import java.util.concurrent.*;

public class ConcurrentHashMapExample {
    public static void main(String[] args) {
        ConcurrentHashMap<String, Integer> concurrentMap = new ConcurrentHashMap<>();
        
        // Thread-safe operations
        concurrentMap.put("key1", 1);
        concurrentMap.putIfAbsent("key2", 2);
        
        // Atomic operations
        concurrentMap.compute("key1", (k, v) -> v == null ? 1 : v + 1);
        concurrentMap.computeIfAbsent("key3", k -> 3);
        concurrentMap.computeIfPresent("key1", (k, v) -> v * 2);
        
        // Parallel bulk operations
        concurrentMap.forEach(1, (k, v) -> System.out.println(k + "=" + v));
        
        Integer sum = concurrentMap.reduceValues(1, Integer::sum);
        System.out.println("Sum of values: " + sum);
    }
}
```

### BlockingQueue Implementations
```java
public class BlockingQueueExample {
    public static void main(String[] args) throws InterruptedException {
        // ArrayBlockingQueue - bounded queue
        BlockingQueue<String> boundedQueue = new ArrayBlockingQueue<>(10);
        
        // LinkedBlockingQueue - optionally bounded
        BlockingQueue<String> unboundedQueue = new LinkedBlockingQueue<>();
        
        // PriorityBlockingQueue - unbounded priority queue
        BlockingQueue<Integer> priorityQueue = new PriorityBlockingQueue<>();
        
        // Producer-Consumer example
        BlockingQueue<Integer> queue = new LinkedBlockingQueue<>();
        
        // Producer thread
        Thread producer = new Thread(() -> {
            try {
                for (int i = 1; i <= 10; i++) {
                    queue.put(i);
                    System.out.println("Produced: " + i);
                    Thread.sleep(100);
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        });
        
        // Consumer thread
        Thread consumer = new Thread(() -> {
            try {
                while (true) {
                    Integer item = queue.take();
                    System.out.println("Consumed: " + item);
                    Thread.sleep(200);
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        });
        
        producer.start();
        consumer.start();
        
        producer.join();
        Thread.sleep(5000);
        consumer.interrupt();
    }
}
```

## 7. Utility Classes

### Collections Utility Class
```java
public class CollectionsUtilityExample {
    public static void main(String[] args) {
        List<Integer> numbers = Arrays.asList(5, 2, 8, 1, 9, 3);
        
        // Sorting
        Collections.sort(numbers);
        Collections.sort(numbers, Collections.reverseOrder());
        
        // Searching (on sorted list)
        int index = Collections.binarySearch(numbers, 5);
        
        // Shuffling
        Collections.shuffle(numbers);
        
        // Min/Max
        Integer min = Collections.min(numbers);
        Integer max = Collections.max(numbers);
        
        // Frequency
        int frequency = Collections.frequency(numbers, 5);
        
        // Replace all
        Collections.replaceAll(numbers, 5, 50);
        
        // Immutable collections
        List<String> immutableList = Collections.unmodifiableList(
            Arrays.asList("a", "b", "c")
        );
        
        // Synchronized collections
        List<String> synchronizedList = Collections.synchronizedList(new ArrayList<>());
        
        // Empty collections
        List<String> emptyList = Collections.emptyList();
        Set<String> emptySet = Collections.emptySet();
        Map<String, String> emptyMap = Collections.emptyMap();
        
        // Singleton collections
        List<String> singletonList = Collections.singletonList("only");
        Set<String> singletonSet = Collections.singleton("only");
        Map<String, String> singletonMap = Collections.singletonMap("key", "value");
    }
}
```

### Arrays Utility Class
```java
public class ArraysUtilityExample {
    public static void main(String[] args) {
        int[] numbers = {5, 2, 8, 1, 9, 3};
        
        // Sorting
        Arrays.sort(numbers);
        
        // Binary search
        int index = Arrays.binarySearch(numbers, 5);
        
        // Fill array
        int[] filled = new int[10];
        Arrays.fill(filled, 42);
        
        // Copy arrays
        int[] copy = Arrays.copyOf(numbers, numbers.length);
        int[] partialCopy = Arrays.copyOfRange(numbers, 1, 4);
        
        // Compare arrays
        boolean equal = Arrays.equals(numbers, copy);
        
        // Convert to list
        List<Integer> list = Arrays.asList(1, 2, 3, 4, 5);
        
        // String representation
        System.out.println(Arrays.toString(numbers));
        
        // Multidimensional arrays
        int[][] matrix = {{1, 2}, {3, 4}};
        System.out.println(Arrays.deepToString(matrix));
        
        // Stream from array (Java 8+)
        Arrays.stream(numbers)
              .filter(n -> n > 5)
              .forEach(System.out::println);
    }
}
```

## 8. Performance Comparison

### Time Complexity Summary

| Operation | ArrayList | LinkedList | HashSet | TreeSet | HashMap | TreeMap |
|-----------|-----------|------------|---------|---------|---------|---------|
| Access    | O(1)      | O(n)       | N/A     | N/A     | O(1)    | O(log n)|
| Search    | O(n)      | O(n)       | O(1)    | O(log n)| O(1)    | O(log n)|
| Insertion | O(1)*     | O(1)**     | O(1)    | O(log n)| O(1)    | O(log n)|
| Deletion  | O(n)      | O(1)**     | O(1)    | O(log n)| O(1)    | O(log n)|

*Amortized for end insertion, O(n) for arbitrary position  
**At known position, O(n) for arbitrary position

### When to Use Which Collection

```java
public class CollectionSelection {
    public void demonstrateUseCases() {
        // Use ArrayList when:
        // - Need indexed access
        // - Frequent reading, infrequent insertion/deletion
        // - Need to maintain insertion order
        List<String> readHeavyList = new ArrayList<>();
        
        // Use LinkedList when:
        // - Frequent insertion/deletion at beginning/end
        // - Don't need random access
        // - Implementing queue/stack
        List<String> insertionHeavyList = new LinkedList<>();
        
        // Use HashSet when:
        // - Need fast lookup, no duplicates
        // - Don't care about order
        Set<String> uniqueItems = new HashSet<>();
        
        // Use TreeSet when:
        // - Need sorted unique elements
        // - Need range operations
        Set<String> sortedUniqueItems = new TreeSet<>();
        
        // Use HashMap when:
        // - Need fast key-value lookup
        // - Don't care about order
        Map<String, String> fastLookup = new HashMap<>();
        
        // Use TreeMap when:
        // - Need sorted key-value pairs
        // - Need range operations on keys
        Map<String, String> sortedMap = new TreeMap<>();
        
        // Use LinkedHashMap when:
        // - Need predictable iteration order
        // - Building LRU cache
        Map<String, String> orderedMap = new LinkedHashMap<>();
    }
}
```

## 9. Best Practices

### 1. Choose the Right Collection
```java
// Good - specific interface type
List<String> names = new ArrayList<>();
Set<Integer> uniqueNumbers = new HashSet<>();
Map<String, User> userCache = new HashMap<>();

// Less flexible - concrete type
ArrayList<String> names = new ArrayList<>();
```

### 2. Initialize with Capacity When Known
```java
// Good - avoid resizing
List<String> list = new ArrayList<>(1000);
Map<String, String> map = new HashMap<>(1000);

// Default capacity might be too small
List<String> list = new ArrayList<>(); // Initial capacity: 10
```

### 3. Use Immutable Collections When Appropriate
```java
// Java 9+ - Immutable collections
List<String> immutableList = List.of("a", "b", "c");
Set<Integer> immutableSet = Set.of(1, 2, 3);
Map<String, Integer> immutableMap = Map.of("a", 1, "b", 2);

// Before Java 9
List<String> immutableList = Collections.unmodifiableList(
    Arrays.asList("a", "b", "c")
);
```

### 4. Implement equals() and hashCode() for Custom Objects
```java
public class Person {
    private String name;
    private int age;
    
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
}
```

## Summary

The Java Collections Framework provides:
- **Interfaces**: Define contracts for different collection types
- **Implementations**: Concrete classes with different performance characteristics
- **Algorithms**: Utility methods for common operations
- **Thread Safety**: Concurrent collections for multi-threaded environments

Key takeaways:
- Choose collections based on your access patterns
- Consider thread safety requirements
- Use appropriate interfaces for flexibility
- Understand performance implications of different implementations
- Leverage utility classes for common operations
