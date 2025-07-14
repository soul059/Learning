# Threading and Multithreading in Java

This guide provides a detailed look into the concepts of threading and multithreading in Java, which are essential for building responsive and high-performance applications.

---

## 1. What is a Thread?

A **thread** is the smallest unit of execution within a process. It's a lightweight subprocess that has its own path of execution. A Java application runs by default in a single thread—the `main` thread.

### Process vs. Thread

-   **Process**: A program in execution (e.g., a running Java application). Each process has its own memory space. Processes are heavyweight and communication between them is expensive.
-   **Thread**: A segment of a process. Multiple threads within the same process share the same memory space, which makes them lightweight and allows for efficient communication.

Every Java application has at least one thread: the `main` thread. The JVM creates this thread to execute the `main()` method.

---

## 2. What is Multithreading?

**Multithreading** is the process of executing multiple threads simultaneously. This allows a program to perform multiple operations concurrently, improving the utilization of the CPU and making applications more responsive.

### Advantages of Multithreading

1.  **Responsiveness**: It prevents the user interface from freezing. For example, a long-running task can execute on a separate thread without blocking the main UI thread.
2.  **CPU Utilization**: It allows the CPU to be used more efficiently. If one thread is waiting for a resource (e.g., I/O), another thread can run.
3.  **Resource Sharing**: Threads share the same memory space, making it easy to share data between them.
4.  **Parallelism**: On multi-core processors, multithreading can lead to true parallel execution, significantly speeding up computation-intensive tasks.

---

## 3. Creating Threads in Java

There are two primary ways to create a thread in Java:

### a) Extending the `Thread` Class

You can create a new class that extends `java.lang.Thread` and override its `run()` method.

```java
// MyThread.java
public class MyThread extends Thread {
    private String threadName;

    public MyThread(String name) {
        this.threadName = name;
    }

    @Override
    public void run() {
        // This is the entry point for the new thread.
        for (int i = 0; i < 5; i++) {
            System.out.println(threadName + " is running, count: " + i);
            try {
                // Pause for a moment
                Thread.sleep(500);
            } catch (InterruptedException e) {
                System.out.println(threadName + " was interrupted.");
            }
        }
        System.out.println(threadName + " has finished.");
    }
}

// Main.java
public class Main {
    public static void main(String[] args) {
        System.out.println("Main thread starting.");

        MyThread thread1 = new MyThread("Thread-1");
        thread1.start(); // This starts the execution of the thread's run() method

        MyThread thread2 = new MyThread("Thread-2");
        thread2.start();

        System.out.println("Main thread finished.");
    }
}
```
**Note:** Calling `run()` directly would execute the code in the current thread, not a new one. You must call `start()` to create a new thread.

### b) Implementing the `Runnable` Interface (Preferred Method)

You can implement the `java.lang.Runnable` interface and pass an instance of your class to a `Thread`'s constructor.

**Why is this preferred?**
-   Java does not support multiple inheritance. If your class already extends another class, you cannot extend `Thread`. Implementing an interface does not have this limitation.
-   It promotes good object-oriented design by separating the task (the `Runnable`) from the execution mechanism (the `Thread`).

```java
// MyRunnable.java
public class MyRunnable implements Runnable {
    private String taskName;

    public MyRunnable(String name) {
        this.taskName = name;
    }

    @Override
    public void run() {
        // The task to be executed
        for (int i = 0; i < 5; i++) {
            System.out.println(taskName + " is executing, count: " + i);
            try {
                Thread.sleep(500);
            } catch (InterruptedException e) {
                System.out.println(taskName + " was interrupted.");
            }
        }
        System.out.println(taskName + " has completed.");
    }
}

// Main.java
public class Main {
    public static void main(String[] args) {
        System.out.println("Main thread starting.");

        // Create a Runnable task
        MyRunnable task1 = new MyRunnable("Task-1");
        // Create a Thread and pass the task to it
        Thread thread1 = new Thread(task1);
        thread1.start();

        // Using a lambda expression for a simple task (Java 8+)
        Runnable task2 = () -> System.out.println("This is a simple task running in a thread.");
        Thread thread2 = new Thread(task2);
        thread2.start();

        System.out.println("Main thread finished.");
    }
}
```

---

## 4. The Thread Lifecycle

A thread goes through several states during its life:
1.  **New**: The thread has been created but has not yet started (i.e., `start()` has not been called).
2.  **Runnable**: The thread is ready to run and is waiting for the thread scheduler to allocate CPU time. This state includes the thread being actively running.
3.  **Blocked/Waiting**: The thread is temporarily inactive. It's not consuming CPU cycles. This can happen if it's waiting for I/O, waiting to acquire a lock, or has been put to sleep.
4.  **Timed Waiting**: A thread is in this state if it has called a method with a timeout (e.g., `Thread.sleep(1000)` or `wait(1000)`).
5.  **Terminated**: The thread has completed its execution (its `run()` method has finished) or has been otherwise terminated.

---

## 5. Thread Synchronization

When multiple threads share resources (like objects or variables), you need to manage their access to prevent problems like **race conditions** and **data corruption**. Synchronization ensures that only one thread can access a shared resource at a time.

### The `synchronized` Keyword

Java's primary mechanism for synchronization is the `synchronized` keyword. It can be applied to methods or blocks of code.

-   **Synchronized Method**: When a method is declared `synchronized`, a thread must acquire the intrinsic lock of the object before it can execute the method.

```java
public class Counter {
    private int count = 0;

    // Only one thread can execute this method at a time on a given instance of Counter
    public synchronized void increment() {
        count++;
    }

    public int getCount() {
        return count;
    }
}
```

-   **Synchronized Block**: This allows you to synchronize only a part of a method, which can be more efficient. You specify which object's lock to acquire.

```java
public class BankAccount {
    private double balance;
    private final Object lock = new Object(); // A dedicated lock object

    public void deposit(double amount) {
        // Some non-critical code here...

        synchronized (lock) { // Acquire the lock for this block only
            // Critical section: only one thread can be here at a time
            this.balance += amount;
        }

        // More non-critical code...
    }
}
```

### Deadlock

A common problem in multithreading where two or more threads are blocked forever, each waiting for the other to release a resource. This typically happens when multiple threads need the same set of locks but acquire them in a different order.

**Example Scenario:**
1.  Thread 1 locks Resource A and tries to lock Resource B.
2.  Thread 2 locks Resource B and tries to lock Resource A.

Both threads will wait indefinitely. Avoiding deadlock requires careful lock ordering and design.
