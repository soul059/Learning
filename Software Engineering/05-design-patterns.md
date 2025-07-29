# Design Patterns

## Introduction

Design patterns are reusable solutions to commonly occurring problems in software design. They represent best practices and provide a shared vocabulary for developers. Design patterns are not code but rather templates that can be applied to solve design problems in different situations.

## History and Benefits

### Origin
- Introduced by Christopher Alexander in architecture (1977)
- Adapted to software by Gang of Four (GoF) in "Design Patterns: Elements of Reusable Object-Oriented Software" (1994)

### Benefits of Design Patterns

1. **Reusability**: Proven solutions that can be applied to similar problems
2. **Communication**: Common vocabulary for developers
3. **Best Practices**: Encapsulate design expertise and experience
4. **Maintainability**: Well-structured, flexible code
5. **Documentation**: Self-documenting code through recognizable patterns

## Classification of Design Patterns

Design patterns are classified into three main categories:

### 1. Creational Patterns
**Purpose**: Deal with object creation mechanisms
**Focus**: How objects are created, composed, and represented

### 2. Structural Patterns
**Purpose**: Deal with object composition and relationships
**Focus**: How classes and objects are composed to form larger structures

### 3. Behavioral Patterns
**Purpose**: Deal with communication between objects and the assignment of responsibilities
**Focus**: How objects interact and distribute responsibilities

## Creational Patterns

### 1. Singleton Pattern

**Intent**: Ensure a class has only one instance and provide a global point of access to it.

**Problem**: Sometimes you need exactly one instance of a class (e.g., database connection, logger, cache).

**Solution**:
```java
public class Singleton {
    private static volatile Singleton instance;
    
    private Singleton() {
        // Private constructor to prevent instantiation
    }
    
    public static Singleton getInstance() {
        if (instance == null) {
            synchronized (Singleton.class) {
                if (instance == null) {
                    instance = new Singleton();
                }
            }
        }
        return instance;
    }
}
```

**Thread-Safe Enum Implementation**:
```java
public enum Singleton {
    INSTANCE;
    
    public void doSomething() {
        // Business logic
    }
}
```

**Use Cases**:
- Database connections
- Logging services
- Configuration settings
- Cache implementations

**Pros**:
- Controlled access to sole instance
- Reduced memory footprint
- Global access point

**Cons**:
- Violates Single Responsibility Principle
- Difficult to unit test
- Hidden dependencies

### 2. Factory Method Pattern

**Intent**: Create objects without specifying their exact classes.

**Problem**: Need to create objects but the exact class isn't known until runtime.

**Solution**:
```java
// Product interface
interface Shape {
    void draw();
}

// Concrete products
class Rectangle implements Shape {
    public void draw() {
        System.out.println("Drawing Rectangle");
    }
}

class Circle implements Shape {
    public void draw() {
        System.out.println("Drawing Circle");
    }
}

// Creator
abstract class ShapeFactory {
    public abstract Shape createShape();
    
    public void renderShape() {
        Shape shape = createShape();
        shape.draw();
    }
}

// Concrete creators
class RectangleFactory extends ShapeFactory {
    public Shape createShape() {
        return new Rectangle();
    }
}

class CircleFactory extends ShapeFactory {
    public Shape createShape() {
        return new Circle();
    }
}
```

**Use Cases**:
- UI component creation
- Database driver selection
- Parser selection based on file type

### 3. Abstract Factory Pattern

**Intent**: Provide an interface for creating families of related objects.

**Example**:
```java
// Abstract factory
interface GUIFactory {
    Button createButton();
    Checkbox createCheckbox();
}

// Concrete factories
class WindowsFactory implements GUIFactory {
    public Button createButton() {
        return new WindowsButton();
    }
    
    public Checkbox createCheckbox() {
        return new WindowsCheckbox();
    }
}

class MacFactory implements GUIFactory {
    public Button createButton() {
        return new MacButton();
    }
    
    public Checkbox createCheckbox() {
        return new MacCheckbox();
    }
}
```

### 4. Builder Pattern

**Intent**: Construct complex objects step by step.

**Problem**: Creating objects with many optional parameters or complex construction logic.

**Solution**:
```java
class Pizza {
    private String dough;
    private String sauce;
    private String topping;
    
    private Pizza(PizzaBuilder builder) {
        this.dough = builder.dough;
        this.sauce = builder.sauce;
        this.topping = builder.topping;
    }
    
    public static class PizzaBuilder {
        private String dough;
        private String sauce;
        private String topping;
        
        public PizzaBuilder setDough(String dough) {
            this.dough = dough;
            return this;
        }
        
        public PizzaBuilder setSauce(String sauce) {
            this.sauce = sauce;
            return this;
        }
        
        public PizzaBuilder setTopping(String topping) {
            this.topping = topping;
            return this;
        }
        
        public Pizza build() {
            return new Pizza(this);
        }
    }
}

// Usage
Pizza pizza = new Pizza.PizzaBuilder()
    .setDough("thin")
    .setSauce("tomato")
    .setTopping("cheese")
    .build();
```

### 5. Prototype Pattern

**Intent**: Create objects by cloning existing instances.

**Solution**:
```java
interface Cloneable {
    Object clone();
}

class Shape implements Cloneable {
    private String type;
    
    public Shape(String type) {
        this.type = type;
    }
    
    public Object clone() {
        return new Shape(this.type);
    }
}
```

## Structural Patterns

### 1. Adapter Pattern

**Intent**: Allow incompatible interfaces to work together.

**Problem**: Need to use an existing class with an incompatible interface.

**Solution**:
```java
// Target interface
interface MediaPlayer {
    void play(String audioType, String fileName);
}

// Adaptee (existing class with incompatible interface)
class AdvancedMediaPlayer {
    void playVlc(String fileName) {
        System.out.println("Playing vlc file: " + fileName);
    }
    
    void playMp4(String fileName) {
        System.out.println("Playing mp4 file: " + fileName);
    }
}

// Adapter
class MediaAdapter implements MediaPlayer {
    private AdvancedMediaPlayer advancedPlayer;
    
    public MediaAdapter(String audioType) {
        if (audioType.equalsIgnoreCase("vlc") || audioType.equalsIgnoreCase("mp4")) {
            advancedPlayer = new AdvancedMediaPlayer();
        }
    }
    
    public void play(String audioType, String fileName) {
        if (audioType.equalsIgnoreCase("vlc")) {
            advancedPlayer.playVlc(fileName);
        } else if (audioType.equalsIgnoreCase("mp4")) {
            advancedPlayer.playMp4(fileName);
        }
    }
}

// Client
class AudioPlayer implements MediaPlayer {
    private MediaAdapter mediaAdapter;
    
    public void play(String audioType, String fileName) {
        if (audioType.equalsIgnoreCase("mp3")) {
            System.out.println("Playing mp3 file: " + fileName);
        } else {
            mediaAdapter = new MediaAdapter(audioType);
            mediaAdapter.play(audioType, fileName);
        }
    }
}
```

### 2. Decorator Pattern

**Intent**: Add new functionality to objects dynamically without altering their structure.

**Solution**:
```java
// Component interface
interface Coffee {
    double getCost();
    String getDescription();
}

// Concrete component
class SimpleCoffee implements Coffee {
    public double getCost() {
        return 1.0;
    }
    
    public String getDescription() {
        return "Simple coffee";
    }
}

// Base decorator
abstract class CoffeeDecorator implements Coffee {
    protected Coffee coffee;
    
    public CoffeeDecorator(Coffee coffee) {
        this.coffee = coffee;
    }
}

// Concrete decorators
class MilkDecorator extends CoffeeDecorator {
    public MilkDecorator(Coffee coffee) {
        super(coffee);
    }
    
    public double getCost() {
        return coffee.getCost() + 0.5;
    }
    
    public String getDescription() {
        return coffee.getDescription() + ", milk";
    }
}

class SugarDecorator extends CoffeeDecorator {
    public SugarDecorator(Coffee coffee) {
        super(coffee);
    }
    
    public double getCost() {
        return coffee.getCost() + 0.2;
    }
    
    public String getDescription() {
        return coffee.getDescription() + ", sugar";
    }
}

// Usage
Coffee coffee = new SimpleCoffee();
coffee = new MilkDecorator(coffee);
coffee = new SugarDecorator(coffee);
System.out.println(coffee.getDescription() + " costs " + coffee.getCost());
```

### 3. Facade Pattern

**Intent**: Provide a simplified interface to a complex subsystem.

**Solution**:
```java
// Complex subsystem classes
class CPU {
    public void freeze() { System.out.println("CPU freezing"); }
    public void jump(long position) { System.out.println("CPU jumping to " + position); }
    public void execute() { System.out.println("CPU executing"); }
}

class Memory {
    public void load(long position, byte[] data) {
        System.out.println("Loading data to memory at " + position);
    }
}

class HardDrive {
    public byte[] read(long lba, int size) {
        System.out.println("Reading from hard drive");
        return new byte[size];
    }
}

// Facade
class ComputerFacade {
    private CPU cpu;
    private Memory memory;
    private HardDrive hardDrive;
    
    public ComputerFacade() {
        this.cpu = new CPU();
        this.memory = new Memory();
        this.hardDrive = new HardDrive();
    }
    
    public void startComputer() {
        cpu.freeze();
        memory.load(0, hardDrive.read(0, 1024));
        cpu.jump(0);
        cpu.execute();
    }
}

// Usage
ComputerFacade computer = new ComputerFacade();
computer.startComputer();
```

### 4. Composite Pattern

**Intent**: Compose objects into tree structures to represent part-whole hierarchies.

**Solution**:
```java
// Component
interface FileSystemItem {
    void display(String indent);
}

// Leaf
class File implements FileSystemItem {
    private String name;
    
    public File(String name) {
        this.name = name;
    }
    
    public void display(String indent) {
        System.out.println(indent + "File: " + name);
    }
}

// Composite
class Directory implements FileSystemItem {
    private String name;
    private List<FileSystemItem> items = new ArrayList<>();
    
    public Directory(String name) {
        this.name = name;
    }
    
    public void add(FileSystemItem item) {
        items.add(item);
    }
    
    public void display(String indent) {
        System.out.println(indent + "Directory: " + name);
        for (FileSystemItem item : items) {
            item.display(indent + "  ");
        }
    }
}
```

### 5. Proxy Pattern

**Intent**: Provide a placeholder or surrogate for another object to control access to it.

**Types**:
- **Virtual Proxy**: Lazy loading of expensive objects
- **Protection Proxy**: Access control
- **Remote Proxy**: Represents objects in different address spaces
- **Cache Proxy**: Caching results

**Solution**:
```java
interface Image {
    void display();
}

class RealImage implements Image {
    private String filename;
    
    public RealImage(String filename) {
        this.filename = filename;
        loadFromDisk();
    }
    
    private void loadFromDisk() {
        System.out.println("Loading " + filename);
    }
    
    public void display() {
        System.out.println("Displaying " + filename);
    }
}

class ProxyImage implements Image {
    private RealImage realImage;
    private String filename;
    
    public ProxyImage(String filename) {
        this.filename = filename;
    }
    
    public void display() {
        if (realImage == null) {
            realImage = new RealImage(filename);
        }
        realImage.display();
    }
}
```

## Behavioral Patterns

### 1. Observer Pattern

**Intent**: Define a one-to-many dependency between objects so that when one object changes state, all dependents are notified.

**Solution**:
```java
import java.util.*;

// Subject interface
interface Subject {
    void addObserver(Observer observer);
    void removeObserver(Observer observer);
    void notifyObservers();
}

// Observer interface
interface Observer {
    void update(String message);
}

// Concrete subject
class NewsAgency implements Subject {
    private List<Observer> observers = new ArrayList<>();
    private String news;
    
    public void addObserver(Observer observer) {
        observers.add(observer);
    }
    
    public void removeObserver(Observer observer) {
        observers.remove(observer);
    }
    
    public void notifyObservers() {
        for (Observer observer : observers) {
            observer.update(news);
        }
    }
    
    public void setNews(String news) {
        this.news = news;
        notifyObservers();
    }
}

// Concrete observer
class NewsChannel implements Observer {
    private String name;
    
    public NewsChannel(String name) {
        this.name = name;
    }
    
    public void update(String news) {
        System.out.println(name + " received news: " + news);
    }
}
```

### 2. Strategy Pattern

**Intent**: Define a family of algorithms, encapsulate each one, and make them interchangeable.

**Solution**:
```java
// Strategy interface
interface PaymentStrategy {
    void pay(double amount);
}

// Concrete strategies
class CreditCardPayment implements PaymentStrategy {
    private String cardNumber;
    
    public CreditCardPayment(String cardNumber) {
        this.cardNumber = cardNumber;
    }
    
    public void pay(double amount) {
        System.out.println("Paid $" + amount + " using credit card " + cardNumber);
    }
}

class PayPalPayment implements PaymentStrategy {
    private String email;
    
    public PayPalPayment(String email) {
        this.email = email;
    }
    
    public void pay(double amount) {
        System.out.println("Paid $" + amount + " using PayPal account " + email);
    }
}

// Context
class ShoppingCart {
    private PaymentStrategy paymentStrategy;
    
    public void setPaymentStrategy(PaymentStrategy paymentStrategy) {
        this.paymentStrategy = paymentStrategy;
    }
    
    public void checkout(double amount) {
        paymentStrategy.pay(amount);
    }
}

// Usage
ShoppingCart cart = new ShoppingCart();
cart.setPaymentStrategy(new CreditCardPayment("1234-5678-9012-3456"));
cart.checkout(100.0);

cart.setPaymentStrategy(new PayPalPayment("user@example.com"));
cart.checkout(50.0);
```

### 3. Command Pattern

**Intent**: Encapsulate a request as an object, allowing you to parameterize clients with different requests.

**Solution**:
```java
// Command interface
interface Command {
    void execute();
    void undo();
}

// Receiver
class Light {
    public void turnOn() {
        System.out.println("Light is ON");
    }
    
    public void turnOff() {
        System.out.println("Light is OFF");
    }
}

// Concrete commands
class TurnOnCommand implements Command {
    private Light light;
    
    public TurnOnCommand(Light light) {
        this.light = light;
    }
    
    public void execute() {
        light.turnOn();
    }
    
    public void undo() {
        light.turnOff();
    }
}

class TurnOffCommand implements Command {
    private Light light;
    
    public TurnOffCommand(Light light) {
        this.light = light;
    }
    
    public void execute() {
        light.turnOff();
    }
    
    public void undo() {
        light.turnOn();
    }
}

// Invoker
class RemoteControl {
    private Command command;
    private Command lastCommand;
    
    public void setCommand(Command command) {
        this.command = command;
    }
    
    public void pressButton() {
        command.execute();
        lastCommand = command;
    }
    
    public void pressUndo() {
        if (lastCommand != null) {
            lastCommand.undo();
        }
    }
}
```

### 4. State Pattern

**Intent**: Allow an object to alter its behavior when its internal state changes.

**Solution**:
```java
// State interface
interface State {
    void handle(Context context);
}

// Concrete states
class StartState implements State {
    public void handle(Context context) {
        System.out.println("Starting the machine");
        context.setState(new RunningState());
    }
}

class RunningState implements State {
    public void handle(Context context) {
        System.out.println("Machine is running");
        context.setState(new StopState());
    }
}

class StopState implements State {
    public void handle(Context context) {
        System.out.println("Stopping the machine");
        context.setState(new StartState());
    }
}

// Context
class Context {
    private State state;
    
    public Context() {
        state = new StartState();
    }
    
    public void setState(State state) {
        this.state = state;
    }
    
    public void request() {
        state.handle(this);
    }
}
```

### 5. Template Method Pattern

**Intent**: Define the skeleton of an algorithm in a base class, letting subclasses override specific steps.

**Solution**:
```java
abstract class DataProcessor {
    // Template method
    public final void process() {
        readData();
        processData();
        writeData();
    }
    
    protected abstract void readData();
    protected abstract void processData();
    protected abstract void writeData();
}

class CSVDataProcessor extends DataProcessor {
    protected void readData() {
        System.out.println("Reading data from CSV file");
    }
    
    protected void processData() {
        System.out.println("Processing CSV data");
    }
    
    protected void writeData() {
        System.out.println("Writing processed data to CSV file");
    }
}

class XMLDataProcessor extends DataProcessor {
    protected void readData() {
        System.out.println("Reading data from XML file");
    }
    
    protected void processData() {
        System.out.println("Processing XML data");
    }
    
    protected void writeData() {
        System.out.println("Writing processed data to XML file");
    }
}
```

## Advanced Patterns

### 1. Model-View-Controller (MVC)

**Intent**: Separate application logic into three interconnected components.

**Components**:
- **Model**: Data and business logic
- **View**: User interface
- **Controller**: Handles user input and coordinates Model and View

### 2. Model-View-Presenter (MVP)

**Intent**: Similar to MVC but the presenter handles all the UI logic.

### 3. Model-View-ViewModel (MVVM)

**Intent**: Separate development of graphical user interface from business logic.

### 4. Dependency Injection

**Intent**: Achieve Inversion of Control between classes and their dependencies.

```java
// Without dependency injection
class UserService {
    private UserRepository userRepository = new DatabaseUserRepository();
    
    public User getUser(Long id) {
        return userRepository.findById(id);
    }
}

// With dependency injection
class UserService {
    private UserRepository userRepository;
    
    public UserService(UserRepository userRepository) {
        this.userRepository = userRepository;
    }
    
    public User getUser(Long id) {
        return userRepository.findById(id);
    }
}
```

## Anti-Patterns

Anti-patterns are common solutions that appear beneficial but are actually ineffective or counterproductive.

### Common Anti-Patterns

1. **God Object**: Class that does too much
2. **Spaghetti Code**: Code with poor structure
3. **Copy and Paste Programming**: Code duplication
4. **Magic Numbers**: Hard-coded numeric values
5. **Golden Hammer**: Over-reliance on familiar technology
6. **Premature Optimization**: Optimizing before identifying bottlenecks

## When to Use Design Patterns

### Appropriate Use Cases

1. **Recurring Problems**: When you encounter the same design problem repeatedly
2. **Complex Systems**: When system complexity justifies the overhead
3. **Team Development**: When patterns improve communication and understanding
4. **Framework Development**: When building reusable components

### When NOT to Use

1. **Simple Problems**: Don't over-engineer simple solutions
2. **Performance Critical**: When patterns add unnecessary overhead
3. **Unclear Requirements**: When the problem isn't well understood
4. **Learning Phase**: Don't force patterns when learning fundamentals

## Best Practices

### Pattern Selection Guidelines

1. **Understand the Problem**: Identify the core problem before choosing a pattern
2. **Consider Alternatives**: Evaluate multiple patterns and approaches
3. **Start Simple**: Use simpler solutions before complex patterns
4. **Focus on Intent**: Understand why the pattern exists
5. **Adapt to Context**: Modify patterns to fit your specific needs

### Implementation Best Practices

1. **Follow SOLID Principles**: Ensure patterns support good design principles
2. **Use Meaningful Names**: Choose clear, descriptive names for pattern elements
3. **Document Intent**: Explain why you chose a particular pattern
4. **Test Thoroughly**: Ensure pattern implementation works correctly
5. **Refactor When Needed**: Improve pattern implementation over time

## Modern Pattern Implementations

### Functional Programming Patterns

Many traditional OOP patterns can be simplified with functional programming:

```java
// Strategy pattern with lambdas
public class Calculator {
    public int calculate(int a, int b, BinaryOperator<Integer> operation) {
        return operation.apply(a, b);
    }
}

// Usage
Calculator calc = new Calculator();
int sum = calc.calculate(5, 3, (a, b) -> a + b);
int product = calc.calculate(5, 3, (a, b) -> a * b);
```

### Reactive Patterns

Patterns for handling asynchronous data streams:

```java
// Observer pattern with reactive streams
Observable<String> observable = Observable.create(emitter -> {
    emitter.onNext("Hello");
    emitter.onNext("World");
    emitter.onComplete();
});

observable.subscribe(System.out::println);
```

## Summary

Design patterns are powerful tools for creating maintainable, flexible software. Key takeaways:

1. **Learn Common Patterns**: Master the fundamental GoF patterns
2. **Understand Intent**: Focus on why patterns exist, not just how to implement them
3. **Use Appropriately**: Apply patterns when they solve real problems
4. **Adapt to Context**: Modify patterns to fit your specific needs
5. **Stay Current**: Learn modern interpretations and implementations
6. **Avoid Over-Engineering**: Don't force patterns where they don't belong
7. **Document Decisions**: Explain pattern choices for future maintainers

Patterns are guides, not rules. Use them wisely to create better software architecture and design.
