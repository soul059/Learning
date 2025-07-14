# A Guide to Design Patterns in Java

Design patterns are proven, reusable solutions to commonly occurring problems within a given context in software design. They are not finished designs that can be transformed directly into code; rather, they are templates for how to solve a problem that can be used in many different situations.

---

## Why Use Design Patterns?

-   **Common Platform**: They provide a standard terminology and understanding for developers.
-   **Proven Solutions**: They are well-tested solutions to common problems, which helps prevent subtle issues.
-   **Code Reusability**: They promote reusability that leads to more robust and maintainable code.
-   **Improved Readability**: Using a standard pattern makes the code's intent clearer to other developers.

Design patterns are typically categorized into three main groups:

1.  **Creational Patterns**: Deal with object creation mechanisms.
2.  **Structural Patterns**: Deal with object composition and relationships.
3.  **Behavioral Patterns**: Deal with communication between objects.

---

## 1. Creational Patterns

These patterns provide various object creation mechanisms, which increase flexibility and reuse of existing code.

### a) Singleton Pattern

**Intent**: Ensures that a class has only one instance and provides a global point of access to it.

**When to use**: When exactly one object is needed to coordinate actions across the system, such as for a logger, a database connection pool, or a configuration manager.

**Example (Thread-Safe):**
```java
public class DatabaseConnection {
    // The single instance, volatile to ensure visibility across threads.
    private static volatile DatabaseConnection instance;

    private DatabaseConnection() {
        // Private constructor to prevent instantiation from outside.
    }

    // Public method to provide access to the single instance.
    public static DatabaseConnection getInstance() {
        // Double-checked locking for thread safety and performance.
        if (instance == null) {
            synchronized (DatabaseConnection.class) {
                if (instance == null) {
                    instance = new DatabaseConnection();
                }
            }
        }
        return instance;
    }

    public void connect() {
        System.out.println("Connected to the database.");
    }
}
```

### b) Factory Method Pattern

**Intent**: Defines an interface for creating an object, but lets subclasses decide which class to instantiate.

**When to use**: When a class cannot anticipate the class of objects it must create, or when a class wants its subclasses to specify the objects it creates.

**Example:**
```java
// The Product interface
interface Document {
    void open();
}

// Concrete Products
class WordDocument implements Document {
    public void open() { System.out.println("Opening Word document."); }
}

class PdfDocument implements Document {
    public void open() { System.out.println("Opening PDF document."); }
}

// The Creator (Factory)
abstract class DocumentFactory {
    // The factory method
    public abstract Document createDocument();

    public void openDocument() {
        Document doc = createDocument();
        doc.open();
    }
}

// Concrete Creators
class WordDocumentFactory extends DocumentFactory {
    public Document createDocument() { return new WordDocument(); }
}

class PdfDocumentFactory extends DocumentFactory {
    public Document createDocument() { return new PdfDocument(); }
}
```

### c) Builder Pattern

**Intent**: Separates the construction of a complex object from its representation, allowing the same construction process to create different representations.

**When to use**: To avoid "telescoping constructors" (constructors with many parameters) and to create complex, immutable objects step-by-step.

**Example:**
```java
// The complex object to be built
class Computer {
    private final String cpu;
    private final String ram;
    private final String storage;
    // Optional parameters
    private final String graphicsCard;
    private final String os;

    private Computer(Builder builder) {
        this.cpu = builder.cpu;
        this.ram = builder.ram;
        this.storage = builder.storage;
        this.graphicsCard = builder.graphicsCard;
        this.os = builder.os;
    }

    // The static nested Builder class
    public static class Builder {
        private final String cpu; // required
        private final String ram;   // required
        private String storage = "256GB SSD"; // default value
        private String graphicsCard; // optional
        private String os;           // optional

        public Builder(String cpu, String ram) {
            this.cpu = cpu;
            this.ram = ram;
        }

        public Builder storage(String storage) {
            this.storage = storage;
            return this;
        }

        public Builder graphicsCard(String graphicsCard) {
            this.graphicsCard = graphicsCard;
            return this;
        }

        public Builder os(String os) {
            this.os = os;
            return this;
        }

        public Computer build() {
            return new Computer(this);
        }
    }
}

// Usage:
// Computer gamingPC = new Computer.Builder("Intel i9", "32GB")
//     .storage("2TB NVMe SSD")
//     .graphicsCard("NVIDIA RTX 4090")
//     .os("Windows 11")
//     .build();
```

---

## 2. Structural Patterns

These patterns explain how to assemble objects and classes into larger structures while keeping these structures flexible and efficient.

### a) Adapter Pattern

**Intent**: Allows objects with incompatible interfaces to collaborate.

**When to use**: When you want to use an existing class, but its interface does not match the one you need.

**Example:**
```java
// The target interface that the client expects
interface MediaPlayer {
    void play(String audioType, String fileName);
}

// The adaptee - an existing class with an incompatible interface
class AdvancedMediaPlayer {
    public void playVlc(String fileName) { System.out.println("Playing vlc file: " + fileName); }
    public void playMp4(String fileName) { System.out.println("Playing mp4 file: " + fileName); }
}

// The Adapter class
class MediaAdapter implements MediaPlayer {
    AdvancedMediaPlayer advancedPlayer = new AdvancedMediaPlayer();

    @Override
    public void play(String audioType, String fileName) {
        if (audioType.equalsIgnoreCase("vlc")) {
            advancedPlayer.playVlc(fileName);
        } else if (audioType.equalsIgnoreCase("mp4")) {
            advancedPlayer.playMp4(fileName);
        }
    }
}

// The client code only works with the target interface
class AudioPlayer implements MediaPlayer {
    MediaAdapter mediaAdapter;

    @Override
    public void play(String audioType, String fileName) {
        if (audioType.equalsIgnoreCase("mp3")) {
            System.out.println("Playing mp3 file: " + fileName);
        } else if (audioType.equalsIgnoreCase("vlc") || audioType.equalsIgnoreCase("mp4")) {
            mediaAdapter = new MediaAdapter();
            mediaAdapter.play(audioType, fileName);
        } else {
            System.out.println("Invalid media type: " + audioType);
        }
    }
}
```

### b) Decorator Pattern

**Intent**: Attaches additional responsibilities to an object dynamically. Decorators provide a flexible alternative to subclassing for extending functionality.

**When to use**: When you want to add responsibilities to individual objects dynamically and transparently, without affecting other objects.

**Example:**
```java
// The Component interface
interface Coffee {
    double getCost();
    String getDescription();
}

// Concrete Component
class SimpleCoffee implements Coffee {
    public double getCost() { return 2.0; }
    public String getDescription() { return "Simple coffee"; }
}

// The base Decorator class
abstract class CoffeeDecorator implements Coffee {
    protected final Coffee decoratedCoffee;

    public CoffeeDecorator(Coffee coffee) { this.decoratedCoffee = coffee; }
    public double getCost() { return decoratedCoffee.getCost(); }
    public String getDescription() { return decoratedCoffee.getDescription(); }
}

// Concrete Decorators
class WithMilk extends CoffeeDecorator {
    public WithMilk(Coffee coffee) { super(coffee); }
    public double getCost() { return super.getCost() + 0.5; }
    public String getDescription() { return super.getDescription() + ", with milk"; }
}

class WithSugar extends CoffeeDecorator {
    public WithSugar(Coffee coffee) { super(coffee); }
    public double getCost() { return super.getCost() + 0.2; }
    public String getDescription() { return super.getDescription() + ", with sugar"; }
}

// Usage:
// Coffee myCoffee = new SimpleCoffee();
// myCoffee = new WithMilk(myCoffee);
// myCoffee = new WithSugar(myCoffee);
// System.out.println(myCoffee.getCost()); // 2.7
// System.out.println(myCoffee.getDescription()); // Simple coffee, with milk, with sugar
```

---

## 3. Behavioral Patterns

These patterns are concerned with algorithms and the assignment of responsibilities between objects.

### a) Observer Pattern

**Intent**: Defines a one-to-many dependency between objects so that when one object (the subject) changes state, all its dependents (the observers) are notified and updated automatically.

**When to use**: When a change to one object requires changing others, and you don't know how many objects need to be changed.

**Example:**
```java
import java.util.ArrayList;
import java.util.List;

// The Observer interface
interface Observer {
    void update(String news);
}

// The Subject interface
interface Subject {
    void registerObserver(Observer observer);
    void removeObserver(Observer observer);
    void notifyObservers();
}

// Concrete Subject
class NewsAgency implements Subject {
    private List<Observer> observers = new ArrayList<>();
    private String news;

    public void setNews(String news) {
        this.news = news;
        notifyObservers();
    }

    public void registerObserver(Observer o) { observers.add(o); }
    public void removeObserver(Observer o) { observers.remove(o); }
    public void notifyObservers() {
        for (Observer o : observers) {
            o.update(news);
        }
    }
}

// Concrete Observer
class NewsChannel implements Observer {
    private String channelName;
    public NewsChannel(String name) { this.channelName = name; }
    public void update(String news) {
        System.out.println(channelName + " received news: " + news);
    }
}
```

### b) Strategy Pattern

**Intent**: Defines a family of algorithms, encapsulates each one, and makes them interchangeable. Strategy lets the algorithm vary independently from clients that use it.

**When to use**: When you have multiple variants of an algorithm and you want to switch between them at runtime.

**Example:**
```java
// The Strategy interface
interface PaymentStrategy {
    void pay(int amount);
}

// Concrete Strategies
class CreditCardStrategy implements PaymentStrategy {
    public void pay(int amount) { System.out.println("Paid " + amount + " with Credit Card."); }
}

class PayPalStrategy implements PaymentStrategy {
    public void pay(int amount) { System.out.println("Paid " + amount + " with PayPal."); }
}

// The Context class
class ShoppingCart {
    private PaymentStrategy paymentStrategy;

    public void setPaymentStrategy(PaymentStrategy strategy) {
        this.paymentStrategy = strategy;
    }

    public void checkout(int amount) {
        paymentStrategy.pay(amount);
    }
}

// Usage:
// ShoppingCart cart = new ShoppingCart();
// cart.setPaymentStrategy(new PayPalStrategy());
// cart.checkout(100); // Paid 100 with PayPal.
```
