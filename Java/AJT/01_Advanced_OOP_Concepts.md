# Advanced Object-Oriented Programming Concepts in Java

## 1. Abstract Classes and Methods

### Abstract Classes
An abstract class is a class that cannot be instantiated directly and may contain abstract methods that must be implemented by subclasses.

```java
abstract class Shape {
    protected String color;
    
    // Concrete method
    public void setColor(String color) {
        this.color = color;
    }
    
    // Abstract method - must be implemented by subclasses
    public abstract double calculateArea();
    public abstract void draw();
}

class Circle extends Shape {
    private double radius;
    
    public Circle(double radius) {
        this.radius = radius;
    }
    
    @Override
    public double calculateArea() {
        return Math.PI * radius * radius;
    }
    
    @Override
    public void draw() {
        System.out.println("Drawing a " + color + " circle with radius " + radius);
    }
}
```

### Key Points:
- Abstract classes can have both abstract and concrete methods
- Cannot be instantiated using `new` keyword
- Can have constructors, instance variables, and static methods
- Use `abstract` keyword for both class and methods

## 2. Interfaces and Default Methods

### Traditional Interfaces
```java
interface Drawable {
    void draw();
    double getArea();
}

interface Colorable {
    void setColor(String color);
    String getColor();
}

// Multiple interface implementation
class Rectangle implements Drawable, Colorable {
    private double width, height;
    private String color;
    
    @Override
    public void draw() {
        System.out.println("Drawing rectangle");
    }
    
    @Override
    public double getArea() {
        return width * height;
    }
    
    @Override
    public void setColor(String color) {
        this.color = color;
    }
    
    @Override
    public String getColor() {
        return color;
    }
}
```

### Default Methods (Java 8+)
```java
interface Vehicle {
    void start();
    void stop();
    
    // Default method
    default void honk() {
        System.out.println("Beep beep!");
    }
    
    // Static method
    static void checkEngine() {
        System.out.println("Engine check complete");
    }
}

class Car implements Vehicle {
    @Override
    public void start() {
        System.out.println("Car started");
    }
    
    @Override
    public void stop() {
        System.out.println("Car stopped");
    }
    
    // Can override default method if needed
    @Override
    public void honk() {
        System.out.println("Car horn: Honk!");
    }
}
```

## 3. Inner Classes

### Member Inner Class
```java
class OuterClass {
    private int outerField = 10;
    
    class InnerClass {
        public void display() {
            System.out.println("Outer field: " + outerField);
        }
    }
    
    public void createInner() {
        InnerClass inner = new InnerClass();
        inner.display();
    }
}

// Usage
OuterClass outer = new OuterClass();
OuterClass.InnerClass inner = outer.new InnerClass();
```

### Static Nested Class
```java
class OuterClass {
    private static int staticField = 20;
    private int instanceField = 30;
    
    static class StaticNestedClass {
        public void display() {
            System.out.println("Static field: " + staticField);
            // Cannot access instance field directly
            // System.out.println(instanceField); // Error
        }
    }
}

// Usage
OuterClass.StaticNestedClass nested = new OuterClass.StaticNestedClass();
```

### Local Inner Class
```java
class OuterClass {
    public void method() {
        final int localVar = 40;
        
        class LocalInnerClass {
            public void display() {
                System.out.println("Local variable: " + localVar);
            }
        }
        
        LocalInnerClass local = new LocalInnerClass();
        local.display();
    }
}
```

### Anonymous Inner Class
```java
interface Greeting {
    void greet();
}

class Example {
    public void createAnonymous() {
        // Anonymous class implementing interface
        Greeting greeting = new Greeting() {
            @Override
            public void greet() {
                System.out.println("Hello from anonymous class!");
            }
        };
        
        greeting.greet();
        
        // Anonymous class extending class
        Thread thread = new Thread() {
            @Override
            public void run() {
                System.out.println("Running in anonymous thread");
            }
        };
        
        thread.start();
    }
}
```

## 4. Enums

### Basic Enum
```java
enum Day {
    MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY
}

// Usage
Day today = Day.MONDAY;
switch (today) {
    case MONDAY:
        System.out.println("Start of work week");
        break;
    case FRIDAY:
        System.out.println("TGIF!");
        break;
    default:
        System.out.println("Regular day");
}
```

### Enum with Fields and Methods
```java
enum Planet {
    MERCURY(3.303e+23, 2.4397e6),
    VENUS(4.869e+24, 6.0518e6),
    EARTH(5.976e+24, 6.37814e6),
    MARS(6.421e+23, 3.3972e6);
    
    private final double mass;   // in kilograms
    private final double radius; // in meters
    
    // Constructor
    Planet(double mass, double radius) {
        this.mass = mass;
        this.radius = radius;
    }
    
    // Methods
    public double getMass() { return mass; }
    public double getRadius() { return radius; }
    
    public double surfaceGravity() {
        return 6.67300E-11 * mass / (radius * radius);
    }
}

// Usage
double earthWeight = 175;
double mass = earthWeight / Planet.EARTH.surfaceGravity();
for (Planet p : Planet.values()) {
    System.out.printf("Your weight on %s is %f%n", p, p.surfaceGravity() * mass);
}
```

## 5. Polymorphism

### Runtime Polymorphism (Method Overriding)
```java
class Animal {
    public void makeSound() {
        System.out.println("Animal makes a sound");
    }
    
    public void eat() {
        System.out.println("Animal eats");
    }
}

class Dog extends Animal {
    @Override
    public void makeSound() {
        System.out.println("Dog barks");
    }
    
    public void wagTail() {
        System.out.println("Dog wags tail");
    }
}

class Cat extends Animal {
    @Override
    public void makeSound() {
        System.out.println("Cat meows");
    }
}

// Polymorphic behavior
Animal[] animals = {new Dog(), new Cat(), new Animal()};
for (Animal animal : animals) {
    animal.makeSound(); // Calls overridden method
}
```

### Compile-time Polymorphism (Method Overloading)
```java
class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
    
    public double add(double a, double b) {
        return a + b;
    }
    
    public int add(int a, int b, int c) {
        return a + b + c;
    }
    
    public String add(String a, String b) {
        return a + b;
    }
}
```

## 6. Type Casting and instanceof

### Upcasting and Downcasting
```java
class Animal {
    public void eat() {
        System.out.println("Animal eats");
    }
}

class Dog extends Animal {
    public void bark() {
        System.out.println("Dog barks");
    }
}

// Upcasting (implicit)
Animal animal = new Dog();
animal.eat(); // Works
// animal.bark(); // Error - Animal doesn't have bark method

// Downcasting (explicit)
if (animal instanceof Dog) {
    Dog dog = (Dog) animal;
    dog.bark(); // Now works
}

// Safe casting with instanceof
public void handleAnimal(Animal animal) {
    if (animal instanceof Dog) {
        Dog dog = (Dog) animal;
        dog.bark();
    } else if (animal instanceof Cat) {
        Cat cat = (Cat) animal;
        cat.meow();
    }
}
```

## 7. Object Class Methods

### equals() and hashCode()
```java
class Person {
    private String name;
    private int age;
    
    public Person(String name, int age) {
        this.name = name;
        this.age = age;
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        
        Person person = (Person) obj;
        return age == person.age && 
               Objects.equals(name, person.name);
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(name, age);
    }
    
    @Override
    public String toString() {
        return "Person{name='" + name + "', age=" + age + '}';
    }
    
    @Override
    protected Object clone() throws CloneNotSupportedException {
        return super.clone();
    }
}
```

## 8. Best Practices

### 1. Favor Composition over Inheritance
```java
// Instead of inheritance
class Car extends Engine {
    // Car IS-A Engine (incorrect relationship)
}

// Use composition
class Car {
    private Engine engine; // Car HAS-A Engine (correct relationship)
    
    public Car(Engine engine) {
        this.engine = engine;
    }
    
    public void start() {
        engine.start();
    }
}
```

### 2. Program to Interfaces
```java
// Good
List<String> list = new ArrayList<>();
Map<String, Integer> map = new HashMap<>();

// Less flexible
ArrayList<String> list = new ArrayList<>();
HashMap<String, Integer> map = new HashMap<>();
```

### 3. Use Builder Pattern for Complex Objects
```java
class Computer {
    private String CPU;
    private String RAM;
    private String storage;
    private String GPU;
    
    private Computer(Builder builder) {
        this.CPU = builder.CPU;
        this.RAM = builder.RAM;
        this.storage = builder.storage;
        this.GPU = builder.GPU;
    }
    
    public static class Builder {
        private String CPU;
        private String RAM;
        private String storage;
        private String GPU;
        
        public Builder setCPU(String CPU) {
            this.CPU = CPU;
            return this;
        }
        
        public Builder setRAM(String RAM) {
            this.RAM = RAM;
            return this;
        }
        
        public Builder setStorage(String storage) {
            this.storage = storage;
            return this;
        }
        
        public Builder setGPU(String GPU) {
            this.GPU = GPU;
            return this;
        }
        
        public Computer build() {
            return new Computer(this);
        }
    }
}

// Usage
Computer computer = new Computer.Builder()
    .setCPU("Intel i7")
    .setRAM("16GB")
    .setStorage("1TB SSD")
    .setGPU("RTX 3080")
    .build();
```

## Summary

Advanced OOP concepts in Java provide powerful tools for creating flexible, maintainable, and reusable code:

- **Abstract classes** define partial implementations and contracts
- **Interfaces** define contracts and enable multiple inheritance of type
- **Inner classes** provide encapsulation and organization
- **Enums** create type-safe constants with behavior
- **Polymorphism** enables flexible and extensible designs
- **Proper inheritance** and composition create clean relationships
- **Object methods** provide standard behaviors for all objects

These concepts form the foundation for advanced Java programming and design patterns.
