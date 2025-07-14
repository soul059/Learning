# Data Structures in Java: A Comprehensive Guide

A data structure is a way of organizing and storing data in a computer so that it can be accessed and modified efficiently. It's a fundamental concept in computer science, and choosing the right data structure for a task can make a huge difference in a program's performance and scalability.

Java provides a rich set of built-in data structures through its **Java Collections Framework (JCF)**.

---

## 1. What is the Java Collections Framework (JCF)?

The JCF is a set of classes and interfaces that implement commonly reusable collection data structures. It provides a unified architecture for representing and manipulating collections, allowing them to be manipulated independently of the details of their representation.

The core interfaces of the JCF are:
-   **`Collection`**: The root interface.
-   **`List`**: An ordered collection (a sequence) that can contain duplicate elements.
-   **`Set`**: A collection that cannot contain duplicate elements.
-   **`Queue`**: A collection used to hold multiple elements prior to processing (typically FIFO).
-   **`Map`**: An object that maps keys to values. It cannot contain duplicate keys.

---

## 2. Linear Data Structures

Linear data structures arrange elements in a sequential order.

### a) Arrays

The most basic data structure. An array is a fixed-size container of elements of the same type, stored in contiguous memory locations.

-   **Characteristics**: Fixed size, fast access by index (`O(1)`), slow insertion/deletion in the middle (`O(n)`).
-   **When to use**: When you know the number of elements beforehand and need frequent random access.

```java
// Declaring and initializing an array of integers
int[] numbers = new int[5];
numbers[0] = 10;
numbers[1] = 20;
// ...

// Accessing an element
int firstNumber = numbers[0]; // Fast access

System.out.println("Array elements:");
for (int i = 0; i < numbers.length; i++) {
    System.out.println(numbers[i]);
}
```

### b) `ArrayList` (Dynamic Array)

`ArrayList` is the resizable-array implementation of the `List` interface. It provides a dynamic array that can grow as needed.

-   **Characteristics**: Dynamic size, ordered, allows duplicates, fast access by index (`O(1)`), slower insertion/deletion (`O(n)`).
-   **When to use**: The default choice for a list. Use it when you need a dynamic list with fast random access.

```java
import java.util.ArrayList;
import java.util.List;

public class ArrayListExample {
    public static void main(String[] args) {
        List<String> fruits = new ArrayList<>();
        fruits.add("Apple");
        fruits.add("Banana");
        fruits.add("Orange");

        System.out.println("Fruit at index 1: " + fruits.get(1)); // Banana
        fruits.remove("Banana");
        System.out.println("ArrayList after removal: " + fruits); // [Apple, Orange]
    }
}
```

### c) `LinkedList`

`LinkedList` is the doubly-linked list implementation of the `List` and `Queue` interfaces. Each element (node) stores a reference to the previous and next element.

-   **Characteristics**: Dynamic size, ordered, allows duplicates, fast insertion/deletion at ends (`O(1)`) and in the middle (`O(1)` if you have a reference to the node), slow random access (`O(n)`).
-   **When to use**: When you have a large list and need frequent insertions and deletions, especially at the beginning or end. It's also a great choice for implementing stacks and queues.

```java
import java.util.LinkedList;
import java.util.List;

public class LinkedListExample {
    public static void main(String[] args) {
        LinkedList<String> animals = new LinkedList<>();
        animals.add("Dog");
        animals.addFirst("Cat"); // Fast insertion at the beginning
        animals.addLast("Horse"); // Fast insertion at the end

        System.out.println("LinkedList: " + animals); // [Cat, Dog, Horse]
        animals.removeFirst();
        System.out.println("After removing first: " + animals); // [Dog, Horse]
    }
}
```

### d) `Stack` (LIFO)

A stack is a Last-In, First-Out (LIFO) data structure. The last element added is the first one to be removed. The legacy `Stack` class is available, but it's recommended to use a `Deque` (like `ArrayDeque`) for stack operations.

-   **Operations**: `push` (add to top), `pop` (remove from top), `peek` (view top).
-   **When to use**: Reversing things, implementing undo functionality, parsing expressions.

```java
import java.util.ArrayDeque;
import java.util.Deque;

public class StackExample {
    public static void main(String[] args) {
        Deque<String> bookStack = new ArrayDeque<>();
        bookStack.push("Java Programming");
        bookStack.push("Data Structures");
        bookStack.push("Algorithms");

        System.out.println("Top of stack: " + bookStack.peek()); // Algorithms
        String topBook = bookStack.pop();
        System.out.println("Popped: " + topBook); // Algorithms
        System.out.println("Stack now: " + bookStack); // [Data Structures, Java Programming]
    }
}
```

### e) `Queue` (FIFO)

A queue is a First-In, First-Out (FIFO) data structure. The first element added is the first one to be removed. `LinkedList` and `ArrayDeque` are common implementations.

-   **Operations**: `add`/`offer` (add to end), `remove`/`poll` (remove from front), `element`/`peek` (view front).
-   **When to use**: Processing tasks in order, breadth-first search in graphs, managing requests.

```java
import java.util.LinkedList;
import java.util.Queue;

public class QueueExample {
    public static void main(String[] args) {
        Queue<String> customerLine = new LinkedList<>();
        customerLine.offer("Alice");
        customerLine.offer("Bob");
        customerLine.offer("Charlie");

        System.out.println("Queue: " + customerLine); // [Alice, Bob, Charlie]
        String nextCustomer = customerLine.poll();
        System.out.println("Now serving: " + nextCustomer); // Alice
        System.out.println("Queue now: " + customerLine); // [Bob, Charlie]
    }
}
```

---

## 3. Non-Linear Data Structures

These structures don't have a linear, sequential arrangement of elements.

### a) `HashSet`

A `HashSet` is an implementation of the `Set` interface that uses a hash table for storage.

-   **Characteristics**: Unordered, no duplicate elements, very fast insertion, deletion, and lookup (`O(1)` on average).
-   **When to use**: When you need to store a collection of unique items and don't care about their order. Great for checking for the existence of an item quickly.

```java
import java.util.HashSet;
import java.util.Set;

public class HashSetExample {
    public static void main(String[] args) {
        Set<String> uniqueNames = new HashSet<>();
        uniqueNames.add("John");
        uniqueNames.add("Jane");
        uniqueNames.add("John"); // This will be ignored

        System.out.println("Set contains 'Jane': " + uniqueNames.contains("Jane")); // true
        System.out.println("Set: " + uniqueNames); // [John, Jane] (order not guaranteed)
    }
}
```

### b) `TreeSet`

A `TreeSet` is an implementation of the `Set` interface that uses a tree structure (a Red-Black Tree) to store elements in a sorted order.

-   **Characteristics**: Sorted order, no duplicates, good performance for insertion, deletion, and lookup (`O(log n)`).
-   **When to use**: When you need a collection of unique items that are always kept in sorted order.

```java
import java.util.TreeSet;
import java.util.Set;

public class TreeSetExample {
    public static void main(String[] args) {
        Set<Integer> sortedNumbers = new TreeSet<>();
        sortedNumbers.add(50);
        sortedNumbers.add(10);
        sortedNumbers.add(90);

        System.out.println("TreeSet: " + sortedNumbers); // [10, 50, 90]
    }
}
```

### c) `HashMap`

A `HashMap` is an implementation of the `Map` interface that stores key-value pairs in a hash table.

-   **Characteristics**: Unordered, unique keys, very fast insertion, retrieval, and deletion by key (`O(1)` on average).
-   **When to use**: The default choice for a map. Use it when you need to associate unique keys with values and need fast access.

```java
import java.util.HashMap;
import java.util.Map;

public class HashMapExample {
    public static void main(String[] args) {
        Map<String, Integer> studentScores = new HashMap<>();
        studentScores.put("Alice", 95);
        studentScores.put("Bob", 82);
        studentScores.put("Charlie", 95);

        System.out.println("Bob's score: " + studentScores.get("Bob")); // 82
        System.out.println("Map: " + studentScores); // {Bob=82, Alice=95, Charlie=95}
    }
}
```

### d) `TreeMap`

A `TreeMap` is an implementation of the `Map` interface that stores key-value pairs in a sorted tree structure, ordered by the keys.

-   **Characteristics**: Sorted by key, unique keys, good performance for all operations (`O(log n)`).
-   **When to use**: When you need a map that keeps its entries sorted by key.

```java
import java.util.TreeMap;
import java.util.Map;

public class TreeMapExample {
    public static void main(String[] args) {
        Map<String, Integer> sortedScores = new TreeMap<>();
        sortedScores.put("Charlie", 95);
        sortedScores.put("Alice", 95);
        sortedScores.put("Bob", 82);

        System.out.println("TreeMap: " + sortedScores); // {Alice=95, Bob=82, Charlie=95}
    }
}
```
