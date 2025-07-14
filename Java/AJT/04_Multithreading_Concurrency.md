# Multithreading and Concurrency in Java

## 1. Introduction to Threads

### What is a Thread?
A thread is a lightweight sub-process, the smallest unit of processing. Threads enable concurrent execution within a single program.

### Creating Threads

#### Method 1: Extending Thread Class
```java
class MyThread extends Thread {
    private String threadName;
    
    public MyThread(String name) {
        this.threadName = name;
    }
    
    @Override
    public void run() {
        for (int i = 0; i < 5; i++) {
            System.out.println(threadName + " - Count: " + i);
            try {
                Thread.sleep(1000); // Sleep for 1 second
            } catch (InterruptedException e) {
                System.out.println(threadName + " interrupted");
                return;
            }
        }
        System.out.println(threadName + " finished");
    }
}

// Usage
public class ThreadExample1 {
    public static void main(String[] args) {
        MyThread thread1 = new MyThread("Thread-1");
        MyThread thread2 = new MyThread("Thread-2");
        
        thread1.start(); // Start the thread
        thread2.start();
        
        // Main thread continues
        System.out.println("Main thread finished");
    }
}
```

#### Method 2: Implementing Runnable Interface (Preferred)
```java
class MyTask implements Runnable {
    private String taskName;
    
    public MyTask(String name) {
        this.taskName = name;
    }
    
    @Override
    public void run() {
        for (int i = 0; i < 5; i++) {
            System.out.println(taskName + " - Count: " + i);
            try {
                Thread.sleep(500);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return;
            }
        }
    }
}

// Usage
public class ThreadExample2 {
    public static void main(String[] args) {
        Thread thread1 = new Thread(new MyTask("Task-1"));
        Thread thread2 = new Thread(new MyTask("Task-2"));
        
        thread1.start();
        thread2.start();
    }
}
```

#### Method 3: Using Lambda Expressions (Java 8+)
```java
public class ThreadExample3 {
    public static void main(String[] args) {
        // Lambda expression
        Thread thread1 = new Thread(() -> {
            for (int i = 0; i < 3; i++) {
                System.out.println("Lambda Thread - " + i);
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
        });
        
        // Method reference
        Thread thread2 = new Thread(ThreadExample3::printNumbers);
        
        thread1.start();
        thread2.start();
    }
    
    public static void printNumbers() {
        for (int i = 0; i < 3; i++) {
            System.out.println("Method Reference Thread - " + i);
            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
    }
}
```

## 2. Thread Lifecycle and States

### Thread States
```java
public class ThreadStates {
    public static void main(String[] args) throws InterruptedException {
        Thread thread = new Thread(() -> {
            try {
                Thread.sleep(2000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        });
        
        System.out.println("State after creation: " + thread.getState()); // NEW
        
        thread.start();
        System.out.println("State after start: " + thread.getState()); // RUNNABLE
        
        Thread.sleep(100);
        System.out.println("State during sleep: " + thread.getState()); // TIMED_WAITING
        
        thread.join(); // Wait for thread to complete
        System.out.println("State after completion: " + thread.getState()); // TERMINATED
    }
}
```

### Thread Methods
```java
public class ThreadMethods {
    public static void main(String[] args) throws InterruptedException {
        Thread workerThread = new Thread(() -> {
            for (int i = 0; i < 10; i++) {
                System.out.println("Worker: " + i);
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    System.out.println("Worker interrupted");
                    return;
                }
            }
        });
        
        // Thread properties
        workerThread.setName("WorkerThread");
        workerThread.setPriority(Thread.MAX_PRIORITY);
        workerThread.setDaemon(false); // Non-daemon thread
        
        System.out.println("Thread name: " + workerThread.getName());
        System.out.println("Thread priority: " + workerThread.getPriority());
        System.out.println("Is daemon: " + workerThread.isDaemon());
        System.out.println("Is alive: " + workerThread.isAlive());
        
        workerThread.start();
        
        // Main thread sleeps
        Thread.sleep(3000);
        
        // Interrupt the worker thread
        workerThread.interrupt();
        
        // Wait for worker thread to complete
        workerThread.join();
        
        System.out.println("Worker thread completed");
    }
}
```

## 3. Synchronization

### The Problem: Race Conditions
```java
class Counter {
    private int count = 0;
    
    public void increment() {
        count++; // Not atomic - can cause race conditions
    }
    
    public int getCount() {
        return count;
    }
}

public class RaceConditionExample {
    public static void main(String[] args) throws InterruptedException {
        Counter counter = new Counter();
        
        // Create multiple threads that increment the counter
        Thread[] threads = new Thread[10];
        for (int i = 0; i < 10; i++) {
            threads[i] = new Thread(() -> {
                for (int j = 0; j < 1000; j++) {
                    counter.increment();
                }
            });
            threads[i].start();
        }
        
        // Wait for all threads to complete
        for (Thread thread : threads) {
            thread.join();
        }
        
        System.out.println("Expected: 10000, Actual: " + counter.getCount());
        // Result is often less than 10000 due to race conditions
    }
}
```

### Solution 1: Synchronized Methods
```java
class SynchronizedCounter {
    private int count = 0;
    
    public synchronized void increment() {
        count++; // Now thread-safe
    }
    
    public synchronized int getCount() {
        return count;
    }
}
```

### Solution 2: Synchronized Blocks
```java
class SynchronizedBlockCounter {
    private int count = 0;
    private final Object lock = new Object();
    
    public void increment() {
        synchronized(lock) {
            count++;
        }
    }
    
    public int getCount() {
        synchronized(lock) {
            return count;
        }
    }
}
```

### Solution 3: Static Synchronization
```java
class StaticSynchronization {
    private static int staticCounter = 0;
    
    public static synchronized void incrementStatic() {
        staticCounter++;
    }
    
    // Equivalent to:
    public static void incrementStatic2() {
        synchronized(StaticSynchronization.class) {
            staticCounter++;
        }
    }
}
```

## 4. Inter-thread Communication

### wait(), notify(), and notifyAll()
```java
class ProducerConsumer {
    private int data;
    private boolean hasData = false;
    
    public synchronized void produce(int value) throws InterruptedException {
        while (hasData) {
            wait(); // Wait until consumer consumes the data
        }
        
        this.data = value;
        this.hasData = true;
        System.out.println("Produced: " + value);
        notify(); // Notify waiting consumer
    }
    
    public synchronized int consume() throws InterruptedException {
        while (!hasData) {
            wait(); // Wait until producer produces data
        }
        
        int value = this.data;
        this.hasData = false;
        System.out.println("Consumed: " + value);
        notify(); // Notify waiting producer
        return value;
    }
}

public class ProducerConsumerExample {
    public static void main(String[] args) {
        ProducerConsumer pc = new ProducerConsumer();
        
        // Producer thread
        Thread producer = new Thread(() -> {
            try {
                for (int i = 1; i <= 5; i++) {
                    pc.produce(i);
                    Thread.sleep(1000);
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        });
        
        // Consumer thread
        Thread consumer = new Thread(() -> {
            try {
                for (int i = 1; i <= 5; i++) {
                    pc.consume();
                    Thread.sleep(1500);
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        });
        
        producer.start();
        consumer.start();
    }
}
```

## 5. Advanced Synchronization with java.util.concurrent

### ReentrantLock
```java
import java.util.concurrent.locks.ReentrantLock;

class ReentrantLockExample {
    private final ReentrantLock lock = new ReentrantLock();
    private int count = 0;
    
    public void increment() {
        lock.lock();
        try {
            count++;
            System.out.println("Count: " + count + " by " + Thread.currentThread().getName());
        } finally {
            lock.unlock(); // Always unlock in finally block
        }
    }
    
    public boolean tryIncrement() {
        if (lock.tryLock()) {
            try {
                count++;
                return true;
            } finally {
                lock.unlock();
            }
        }
        return false;
    }
    
    public int getCount() {
        lock.lock();
        try {
            return count;
        } finally {
            lock.unlock();
        }
    }
}
```

### ReadWriteLock
```java
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

class ReadWriteLockExample {
    private final ReadWriteLock lock = new ReentrantReadWriteLock();
    private String data = "Initial Data";
    
    public String readData() {
        lock.readLock().lock();
        try {
            System.out.println("Reading: " + data + " by " + Thread.currentThread().getName());
            Thread.sleep(1000); // Simulate read operation
            return data;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return null;
        } finally {
            lock.readLock().unlock();
        }
    }
    
    public void writeData(String newData) {
        lock.writeLock().lock();
        try {
            System.out.println("Writing: " + newData + " by " + Thread.currentThread().getName());
            this.data = newData;
            Thread.sleep(2000); // Simulate write operation
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        } finally {
            lock.writeLock().unlock();
        }
    }
}
```

### Condition Variables
```java
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;

class BoundedBuffer<T> {
    private final Object[] buffer;
    private int head, tail, count;
    private final ReentrantLock lock = new ReentrantLock();
    private final Condition notFull = lock.newCondition();
    private final Condition notEmpty = lock.newCondition();
    
    public BoundedBuffer(int capacity) {
        buffer = new Object[capacity];
    }
    
    public void put(T item) throws InterruptedException {
        lock.lock();
        try {
            while (count == buffer.length) {
                notFull.await(); // Wait until buffer is not full
            }
            
            buffer[tail] = item;
            tail = (tail + 1) % buffer.length;
            count++;
            
            notEmpty.signal(); // Signal waiting consumers
        } finally {
            lock.unlock();
        }
    }
    
    @SuppressWarnings("unchecked")
    public T take() throws InterruptedException {
        lock.lock();
        try {
            while (count == 0) {
                notEmpty.await(); // Wait until buffer is not empty
            }
            
            T item = (T) buffer[head];
            buffer[head] = null;
            head = (head + 1) % buffer.length;
            count--;
            
            notFull.signal(); // Signal waiting producers
            return item;
        } finally {
            lock.unlock();
        }
    }
}
```

## 6. Atomic Variables

### AtomicInteger Example
```java
import java.util.concurrent.atomic.AtomicInteger;

class AtomicExample {
    private AtomicInteger atomicCounter = new AtomicInteger(0);
    
    public void increment() {
        atomicCounter.incrementAndGet();
    }
    
    public void add(int value) {
        atomicCounter.addAndGet(value);
    }
    
    public boolean compareAndSet(int expected, int update) {
        return atomicCounter.compareAndSet(expected, update);
    }
    
    public int getCount() {
        return atomicCounter.get();
    }
}

public class AtomicTest {
    public static void main(String[] args) throws InterruptedException {
        AtomicExample example = new AtomicExample();
        
        Thread[] threads = new Thread[10];
        for (int i = 0; i < 10; i++) {
            threads[i] = new Thread(() -> {
                for (int j = 0; j < 1000; j++) {
                    example.increment();
                }
            });
            threads[i].start();
        }
        
        for (Thread thread : threads) {
            thread.join();
        }
        
        System.out.println("Final count: " + example.getCount()); // Always 10000
    }
}
```

### AtomicReference Example
```java
import java.util.concurrent.atomic.AtomicReference;

class Node {
    String data;
    Node next;
    
    Node(String data) {
        this.data = data;
    }
}

class AtomicStack {
    private AtomicReference<Node> head = new AtomicReference<>();
    
    public void push(String data) {
        Node newNode = new Node(data);
        Node currentHead;
        
        do {
            currentHead = head.get();
            newNode.next = currentHead;
        } while (!head.compareAndSet(currentHead, newNode));
    }
    
    public String pop() {
        Node currentHead;
        Node newHead;
        
        do {
            currentHead = head.get();
            if (currentHead == null) {
                return null;
            }
            newHead = currentHead.next;
        } while (!head.compareAndSet(currentHead, newHead));
        
        return currentHead.data;
    }
}
```

## 7. Thread Pools and Executors

### ExecutorService
```java
import java.util.concurrent.*;

public class ExecutorExample {
    public static void main(String[] args) throws InterruptedException {
        // Fixed thread pool
        ExecutorService fixedPool = Executors.newFixedThreadPool(3);
        
        // Submit tasks
        for (int i = 0; i < 10; i++) {
            final int taskId = i;
            fixedPool.submit(() -> {
                System.out.println("Task " + taskId + " executed by " + 
                                 Thread.currentThread().getName());
                try {
                    Thread.sleep(2000);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            });
        }
        
        // Shutdown executor
        fixedPool.shutdown();
        
        // Wait for termination
        if (!fixedPool.awaitTermination(10, TimeUnit.SECONDS)) {
            fixedPool.shutdownNow();
        }
    }
}
```

### Different Types of Thread Pools
```java
public class ThreadPoolTypes {
    public static void main(String[] args) {
        // 1. Fixed Thread Pool
        ExecutorService fixedPool = Executors.newFixedThreadPool(4);
        
        // 2. Cached Thread Pool - creates threads as needed
        ExecutorService cachedPool = Executors.newCachedThreadPool();
        
        // 3. Single Thread Executor
        ExecutorService singleExecutor = Executors.newSingleThreadExecutor();
        
        // 4. Scheduled Thread Pool
        ScheduledExecutorService scheduledPool = Executors.newScheduledThreadPool(2);
        
        // Schedule task with delay
        scheduledPool.schedule(() -> 
            System.out.println("Delayed task executed"), 5, TimeUnit.SECONDS);
        
        // Schedule task at fixed rate
        scheduledPool.scheduleAtFixedRate(() -> 
            System.out.println("Periodic task"), 0, 2, TimeUnit.SECONDS);
        
        // Custom ThreadPoolExecutor
        ThreadPoolExecutor customPool = new ThreadPoolExecutor(
            2, 4, 60L, TimeUnit.SECONDS,
            new LinkedBlockingQueue<>(10),
            new ThreadFactory() {
                private int counter = 0;
                @Override
                public Thread newThread(Runnable r) {
                    return new Thread(r, "CustomThread-" + counter++);
                }
            },
            new RejectedExecutionHandler() {
                @Override
                public void rejectedExecution(Runnable r, ThreadPoolExecutor executor) {
                    System.out.println("Task rejected: " + r.toString());
                }
            }
        );
        
        // Shutdown all pools
        fixedPool.shutdown();
        cachedPool.shutdown();
        singleExecutor.shutdown();
        scheduledPool.shutdown();
        customPool.shutdown();
    }
}
```

### Future and Callable
```java
import java.util.concurrent.*;
import java.util.List;
import java.util.ArrayList;

public class FutureExample {
    public static void main(String[] args) throws InterruptedException, ExecutionException {
        ExecutorService executor = Executors.newFixedThreadPool(3);
        
        // Callable returns a result
        Callable<Integer> task = () -> {
            Thread.sleep(2000);
            return 42;
        };
        
        // Submit callable and get Future
        Future<Integer> future = executor.submit(task);
        
        // Do other work while task is running
        System.out.println("Task submitted, doing other work...");
        
        // Get result (blocks until task completes)
        Integer result = future.get();
        System.out.println("Task result: " + result);
        
        // Submit multiple tasks
        List<Future<String>> futures = new ArrayList<>();
        for (int i = 0; i < 5; i++) {
            final int taskId = i;
            Future<String> f = executor.submit(() -> {
                Thread.sleep(1000 + taskId * 500);
                return "Result from task " + taskId;
            });
            futures.add(f);
        }
        
        // Process results as they complete
        for (Future<String> f : futures) {
            try {
                String result2 = f.get(3, TimeUnit.SECONDS); // Timeout after 3 seconds
                System.out.println(result2);
            } catch (TimeoutException e) {
                System.out.println("Task timed out");
                f.cancel(true); // Cancel the task
            }
        }
        
        executor.shutdown();
    }
}
```

### CompletableFuture (Java 8+)
```java
import java.util.concurrent.CompletableFuture;

public class CompletableFutureExample {
    public static void main(String[] args) throws InterruptedException {
        // Simple async task
        CompletableFuture<String> future1 = CompletableFuture.supplyAsync(() -> {
            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            return "Hello";
        });
        
        CompletableFuture<String> future2 = CompletableFuture.supplyAsync(() -> {
            try {
                Thread.sleep(1500);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            return "World";
        });
        
        // Combine results
        CompletableFuture<String> combined = future1.thenCombine(future2, 
            (s1, s2) -> s1 + " " + s2);
        
        combined.thenAccept(System.out::println);
        
        // Chain operations
        CompletableFuture<Integer> chained = CompletableFuture
            .supplyAsync(() -> "42")
            .thenApply(Integer::parseInt)
            .thenApply(i -> i * 2);
        
        chained.thenAccept(result -> System.out.println("Chained result: " + result));
        
        // Handle exceptions
        CompletableFuture<String> withException = CompletableFuture
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
        
        withException.thenAccept(System.out::println);
        
        // Wait for all to complete
        Thread.sleep(3000);
    }
}
```

## 8. Concurrent Collections

### ConcurrentHashMap
```java
import java.util.concurrent.ConcurrentHashMap;

public class ConcurrentHashMapExample {
    public static void main(String[] args) {
        ConcurrentHashMap<String, Integer> map = new ConcurrentHashMap<>();
        
        // Thread-safe operations
        map.put("key1", 1);
        map.putIfAbsent("key2", 2);
        
        // Atomic operations
        map.compute("key1", (k, v) -> v == null ? 1 : v + 1);
        map.computeIfAbsent("key3", k -> 3);
        map.computeIfPresent("key1", (k, v) -> v * 2);
        
        // Parallel operations
        map.forEach(1, (k, v) -> System.out.println(k + "=" + v));
        
        Integer sum = map.reduceValues(1, Integer::sum);
        System.out.println("Sum: " + sum);
    }
}
```

### BlockingQueue
```java
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;

public class BlockingQueueExample {
    public static void main(String[] args) throws InterruptedException {
        BlockingQueue<String> queue = new LinkedBlockingQueue<>(10);
        
        // Producer
        Thread producer = new Thread(() -> {
            try {
                for (int i = 0; i < 5; i++) {
                    String item = "Item " + i;
                    queue.put(item); // Blocks if queue is full
                    System.out.println("Produced: " + item);
                    Thread.sleep(1000);
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        });
        
        // Consumer
        Thread consumer = new Thread(() -> {
            try {
                while (true) {
                    String item = queue.take(); // Blocks if queue is empty
                    System.out.println("Consumed: " + item);
                    Thread.sleep(2000);
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

## 9. Best Practices and Common Pitfalls

### Best Practices
```java
public class BestPractices {
    
    // 1. Always use try-finally with locks
    private final ReentrantLock lock = new ReentrantLock();
    
    public void goodLockingPractice() {
        lock.lock();
        try {
            // Critical section
        } finally {
            lock.unlock(); // Always unlock
        }
    }
    
    // 2. Handle InterruptedException properly
    public void goodInterruptHandling() {
        try {
            Thread.sleep(1000);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt(); // Restore interrupt status
            return; // or throw RuntimeException
        }
    }
    
    // 3. Use thread-safe collections
    private final Map<String, String> safeMap = new ConcurrentHashMap<>();
    
    // 4. Prefer immutable objects
    public static final class ImmutablePerson {
        private final String name;
        private final int age;
        
        public ImmutablePerson(String name, int age) {
            this.name = name;
            this.age = age;
        }
        
        public String getName() { return name; }
        public int getAge() { return age; }
    }
    
    // 5. Use volatile for simple flags
    private volatile boolean running = true;
    
    public void stopTask() {
        running = false;
    }
}
```

### Common Pitfalls
```java
public class CommonPitfalls {
    
    // PITFALL 1: Deadlock
    private final Object lock1 = new Object();
    private final Object lock2 = new Object();
    
    public void method1() {
        synchronized(lock1) {
            synchronized(lock2) {
                // Code
            }
        }
    }
    
    public void method2() {
        synchronized(lock2) { // Different order - can cause deadlock
            synchronized(lock1) {
                // Code
            }
        }
    }
    
    // PITFALL 2: Race condition in lazy initialization
    private ExpensiveObject instance; // Not thread-safe
    
    public ExpensiveObject getInstance() {
        if (instance == null) {
            instance = new ExpensiveObject(); // Race condition
        }
        return instance;
    }
    
    // CORRECT: Double-checked locking
    private volatile ExpensiveObject safeInstance;
    
    public ExpensiveObject getSafeInstance() {
        if (safeInstance == null) {
            synchronized(this) {
                if (safeInstance == null) {
                    safeInstance = new ExpensiveObject();
                }
            }
        }
        return safeInstance;
    }
    
    // PITFALL 3: Not shutting down executors
    public void badExecutorUsage() {
        ExecutorService executor = Executors.newFixedThreadPool(5);
        // Submit tasks...
        // executor.shutdown(); // MISSING - causes memory leak
    }
}

class ExpensiveObject {
    // Expensive to create
}
```

## 10. Performance Considerations

### Thread Pool Sizing
```java
public class ThreadPoolSizing {
    
    // CPU-intensive tasks
    public static int getCpuIntensivePoolSize() {
        return Runtime.getRuntime().availableProcessors();
    }
    
    // I/O-intensive tasks
    public static int getIoIntensivePoolSize() {
        return Runtime.getRuntime().availableProcessors() * 2;
    }
    
    // Custom calculation based on blocking coefficient
    public static int calculatePoolSize(double blockingCoefficient) {
        int numberOfCores = Runtime.getRuntime().availableProcessors();
        return (int) (numberOfCores / (1 - blockingCoefficient));
    }
}
```

### Memory Visibility and Happens-Before
```java
public class MemoryVisibility {
    private int counter = 0;
    private volatile boolean flag = false;
    
    // Thread 1
    public void writer() {
        counter = 42;       // 1
        flag = true;        // 2 - volatile write
    }
    
    // Thread 2
    public void reader() {
        if (flag) {         // 3 - volatile read
            // counter is guaranteed to be 42 here
            // due to happens-before relationship
            assert counter == 42;
        }
    }
}
```

## Summary

Java's multithreading and concurrency features provide:

- **Thread Creation**: Multiple ways to create and manage threads
- **Synchronization**: Tools to prevent race conditions and ensure thread safety
- **Advanced Concurrency**: Modern utilities for complex concurrent programming
- **Thread Pools**: Efficient management of thread resources
- **Atomic Operations**: Lock-free programming for better performance
- **Concurrent Collections**: Thread-safe data structures

Key principles:
- Minimize shared mutable state
- Use thread-safe collections when appropriate
- Prefer immutable objects
- Handle interruptions properly
- Avoid deadlocks with consistent lock ordering
- Choose appropriate synchronization mechanisms
- Size thread pools based on workload characteristics
