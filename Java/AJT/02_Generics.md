# Generics in Java

## 1. Introduction to Generics

Generics enable types (classes and interfaces) to be parameters when defining classes, interfaces, and methods. They provide compile-time type safety and eliminate the need for casting.

### Before Generics (Raw Types)
```java
// Before Java 5 - Type unsafe
List list = new ArrayList();
list.add("Hello");
list.add(123); // No compile-time error
String str = (String) list.get(1); // Runtime ClassCastException
```

### With Generics
```java
// Type safe with generics
List<String> list = new ArrayList<String>();
list.add("Hello");
// list.add(123); // Compile-time error
String str = list.get(0); // No casting needed
```

## 2. Generic Classes

### Basic Generic Class
```java
public class Box<T> {
    private T content;
    
    public void set(T content) {
        this.content = content;
    }
    
    public T get() {
        return content;
    }
    
    public boolean isEmpty() {
        return content == null;
    }
}

// Usage
Box<String> stringBox = new Box<>();
stringBox.set("Hello");
String value = stringBox.get();

Box<Integer> intBox = new Box<>();
intBox.set(42);
Integer number = intBox.get();
```

### Multiple Type Parameters
```java
public class Pair<T, U> {
    private T first;
    private U second;
    
    public Pair(T first, U second) {
        this.first = first;
        this.second = second;
    }
    
    public T getFirst() { return first; }
    public U getSecond() { return second; }
    
    public void setFirst(T first) { this.first = first; }
    public void setSecond(U second) { this.second = second; }
    
    @Override
    public String toString() {
        return "(" + first + ", " + second + ")";
    }
}

// Usage
Pair<String, Integer> nameAge = new Pair<>("John", 25);
Pair<Double, Boolean> scorePass = new Pair<>(85.5, true);
```

### Generic Class with Bounded Type Parameters
```java
public class NumberBox<T extends Number> {
    private T number;
    
    public NumberBox(T number) {
        this.number = number;
    }
    
    public double getDoubleValue() {
        return number.doubleValue(); // Can call Number methods
    }
    
    public boolean isPositive() {
        return number.doubleValue() > 0;
    }
}

// Usage
NumberBox<Integer> intBox = new NumberBox<>(42);
NumberBox<Double> doubleBox = new NumberBox<>(3.14);
// NumberBox<String> stringBox = new NumberBox<>("Hello"); // Compile error
```

## 3. Generic Methods

### Basic Generic Methods
```java
public class Utility {
    // Generic method
    public static <T> void swap(T[] array, int i, int j) {
        T temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
    
    // Generic method with return type
    public static <T> T getMiddle(T[] array) {
        return array[array.length / 2];
    }
    
    // Multiple type parameters
    public static <T, U> boolean isEqual(T obj1, U obj2) {
        return obj1 != null && obj1.equals(obj2);
    }
}

// Usage
String[] names = {"Alice", "Bob", "Charlie"};
Utility.swap(names, 0, 2);
String middle = Utility.getMiddle(names);

Integer[] numbers = {1, 2, 3, 4, 5};
Utility.swap(numbers, 1, 3);
Integer midNumber = Utility.getMiddle(numbers);
```

### Generic Methods with Bounded Type Parameters
```java
public class MathUtils {
    // Bounded type parameter
    public static <T extends Comparable<T>> T max(T a, T b) {
        return a.compareTo(b) > 0 ? a : b;
    }
    
    // Multiple bounds
    public static <T extends Number & Comparable<T>> T clamp(T value, T min, T max) {
        if (value.compareTo(min) < 0) return min;
        if (value.compareTo(max) > 0) return max;
        return value;
    }
}

// Usage
String maxString = MathUtils.max("apple", "banana");
Integer maxInt = MathUtils.max(10, 20);
Double clampedValue = MathUtils.clamp(15.5, 10.0, 20.0);
```

## 4. Wildcards

### Upper Bounded Wildcards (? extends)
```java
// Can read from but not write to (except null)
List<? extends Number> numbers = new ArrayList<Integer>();
// numbers.add(10); // Compile error - can't add
Number num = numbers.get(0); // OK - can read as Number

public static double sum(List<? extends Number> numbers) {
    double total = 0.0;
    for (Number num : numbers) {
        total += num.doubleValue();
    }
    return total;
}

// Usage
List<Integer> ints = Arrays.asList(1, 2, 3);
List<Double> doubles = Arrays.asList(1.1, 2.2, 3.3);
double intSum = sum(ints);
double doubleSum = sum(doubles);
```

### Lower Bounded Wildcards (? super)
```java
// Can write but reading returns Object
List<? super Integer> numbers = new ArrayList<Number>();
numbers.add(10); // OK - can add Integer
numbers.add(20); // OK - can add Integer
// Integer num = numbers.get(0); // Compile error - returns Object
Object obj = numbers.get(0); // OK - returns Object

public static void addNumbers(List<? super Integer> list) {
    list.add(1);
    list.add(2);
    list.add(3);
}

// Usage
List<Integer> ints = new ArrayList<>();
List<Number> numbers = new ArrayList<>();
List<Object> objects = new ArrayList<>();

addNumbers(ints);    // OK
addNumbers(numbers); // OK
addNumbers(objects); // OK
```

### Unbounded Wildcards (?)
```java
public static void printList(List<?> list) {
    for (Object item : list) {
        System.out.println(item);
    }
}

public static int getSize(List<?> list) {
    return list.size();
}

// Usage - works with any type
printList(Arrays.asList("a", "b", "c"));
printList(Arrays.asList(1, 2, 3));
printList(Arrays.asList(true, false));
```

## 5. Type Erasure

### Understanding Type Erasure
```java
// At compile time
List<String> stringList = new ArrayList<String>();
List<Integer> intList = new ArrayList<Integer>();

// At runtime (after type erasure)
List stringList = new ArrayList();
List intList = new ArrayList();

// Both have the same class
System.out.println(stringList.getClass() == intList.getClass()); // true
```

### Bridge Methods
```java
class Node<T> {
    public T data;
    
    public void setData(T data) {
        this.data = data;
    }
}

class MyNode extends Node<String> {
    @Override
    public void setData(String data) {
        super.setData(data);
    }
    
    // Compiler generates bridge method:
    // public void setData(Object data) {
    //     setData((String) data);
    // }
}
```

## 6. Generic Interfaces

### Basic Generic Interface
```java
public interface Repository<T, ID> {
    void save(T entity);
    T findById(ID id);
    List<T> findAll();
    void delete(ID id);
    boolean exists(ID id);
}

// Implementation
public class UserRepository implements Repository<User, Long> {
    private Map<Long, User> users = new HashMap<>();
    
    @Override
    public void save(User user) {
        users.put(user.getId(), user);
    }
    
    @Override
    public User findById(Long id) {
        return users.get(id);
    }
    
    @Override
    public List<User> findAll() {
        return new ArrayList<>(users.values());
    }
    
    @Override
    public void delete(Long id) {
        users.remove(id);
    }
    
    @Override
    public boolean exists(Long id) {
        return users.containsKey(id);
    }
}
```

### Functional Interface with Generics
```java
@FunctionalInterface
public interface Transformer<T, R> {
    R transform(T input);
    
    // Default method
    default <V> Transformer<T, V> andThen(Transformer<R, V> after) {
        return input -> after.transform(transform(input));
    }
}

// Usage
Transformer<String, Integer> stringLength = String::length;
Transformer<Integer, String> intToString = Object::toString;

// Chaining
Transformer<String, String> lengthToString = stringLength.andThen(intToString);
String result = lengthToString.transform("Hello"); // "5"
```

## 7. Advanced Generic Patterns

### Generic Factory Pattern
```java
public interface Factory<T> {
    T create();
}

public class DatabaseConnectionFactory implements Factory<Connection> {
    @Override
    public Connection create() {
        // Create and return database connection
        return DriverManager.getConnection("jdbc:...");
    }
}

public class FactoryManager {
    private Map<Class<?>, Factory<?>> factories = new HashMap<>();
    
    public <T> void registerFactory(Class<T> type, Factory<T> factory) {
        factories.put(type, factory);
    }
    
    @SuppressWarnings("unchecked")
    public <T> T create(Class<T> type) {
        Factory<T> factory = (Factory<T>) factories.get(type);
        return factory != null ? factory.create() : null;
    }
}
```

### Generic Builder Pattern
```java
public abstract class Builder<T extends Builder<T>> {
    protected abstract T self();
    
    public abstract Object build();
}

public class PersonBuilder extends Builder<PersonBuilder> {
    private String name;
    private int age;
    private String email;
    
    @Override
    protected PersonBuilder self() {
        return this;
    }
    
    public PersonBuilder setName(String name) {
        this.name = name;
        return self();
    }
    
    public PersonBuilder setAge(int age) {
        this.age = age;
        return self();
    }
    
    public PersonBuilder setEmail(String email) {
        this.email = email;
        return self();
    }
    
    @Override
    public Person build() {
        return new Person(name, age, email);
    }
}
```

## 8. Best Practices and Common Pitfalls

### Best Practices
```java
// 1. Use bounded wildcards for API flexibility
public static void copy(List<? extends T> src, List<? super T> dest) {
    for (T item : src) {
        dest.add(item);
    }
}

// 2. Use generic methods when type is used only in method
public static <T> List<T> emptyList() {
    return new ArrayList<T>();
}

// 3. Prefer generic types over raw types
List<String> list = new ArrayList<>(); // Good
// List list = new ArrayList(); // Avoid

// 4. Use meaningful type parameter names
public class Cache<K, V> {} // Good - Key, Value
public class Pair<F, S> {}  // Good - First, Second
public class Box<T> {}      // Good - Type
```

### Common Pitfalls
```java
// 1. Cannot create instances of type parameters
public class GenericClass<T> {
    // T instance = new T(); // Error
    
    // Use factory method or Class<T> parameter instead
    public T createInstance(Class<T> clazz) throws Exception {
        return clazz.getDeclaredConstructor().newInstance();
    }
}

// 2. Cannot create arrays of generic types
// List<String>[] arrays = new List<String>[10]; // Error
List<String>[] arrays = new List[10]; // OK but unchecked warning

// 3. Cannot use primitives as type arguments
// List<int> numbers = new ArrayList<>(); // Error
List<Integer> numbers = new ArrayList<>(); // OK

// 4. Static context doesn't have access to class type parameters
public class GenericClass<T> {
    // static T staticField; // Error
    // static T staticMethod() { return null; } // Error
    
    static <U> U genericStaticMethod(U param) { // OK
        return param;
    }
}
```

## 9. Real-World Examples

### Generic Data Access Object (DAO)
```java
public abstract class GenericDAO<T, ID> {
    protected Class<T> entityClass;
    
    @SuppressWarnings("unchecked")
    public GenericDAO() {
        Type genericSuperclass = getClass().getGenericSuperclass();
        ParameterizedType parameterizedType = (ParameterizedType) genericSuperclass;
        entityClass = (Class<T>) parameterizedType.getActualTypeArguments()[0];
    }
    
    public abstract void save(T entity);
    public abstract T findById(ID id);
    public abstract List<T> findAll();
    public abstract void update(T entity);
    public abstract void delete(ID id);
    
    protected Class<T> getEntityClass() {
        return entityClass;
    }
}

public class UserDAO extends GenericDAO<User, Long> {
    @Override
    public void save(User user) {
        // Implementation specific to User
    }
    
    @Override
    public User findById(Long id) {
        // Implementation specific to User
        return null;
    }
    
    // Other method implementations...
}
```

### Generic Event System
```java
public interface EventListener<T extends Event> {
    void onEvent(T event);
}

public abstract class Event {
    private final long timestamp;
    
    public Event() {
        this.timestamp = System.currentTimeMillis();
    }
    
    public long getTimestamp() {
        return timestamp;
    }
}

public class UserEvent extends Event {
    private final String userId;
    private final String action;
    
    public UserEvent(String userId, String action) {
        this.userId = userId;
        this.action = action;
    }
    
    // Getters...
}

public class EventBus {
    private Map<Class<?>, List<EventListener<?>>> listeners = new HashMap<>();
    
    @SuppressWarnings("unchecked")
    public <T extends Event> void register(Class<T> eventType, EventListener<T> listener) {
        listeners.computeIfAbsent(eventType, k -> new ArrayList<>()).add(listener);
    }
    
    @SuppressWarnings("unchecked")
    public <T extends Event> void publish(T event) {
        List<EventListener<?>> eventListeners = listeners.get(event.getClass());
        if (eventListeners != null) {
            for (EventListener<?> listener : eventListeners) {
                ((EventListener<T>) listener).onEvent(event);
            }
        }
    }
}

// Usage
EventBus eventBus = new EventBus();
eventBus.register(UserEvent.class, event -> 
    System.out.println("User " + event.getUserId() + " performed " + event.getAction())
);

eventBus.publish(new UserEvent("user123", "login"));
```

## Summary

Generics provide:
- **Type Safety**: Compile-time checking prevents ClassCastException
- **Elimination of Casting**: No need for explicit type casting
- **Code Reusability**: Write once, use with different types
- **Performance**: No boxing/unboxing for primitives when using generics
- **Cleaner Code**: Intent is clearer and code is more readable

Key concepts:
- Generic classes and methods
- Bounded type parameters
- Wildcards (extends, super, unbounded)
- Type erasure and its implications
- Best practices for generic design
