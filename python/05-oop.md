# Object-Oriented Programming in Python

## Table of Contents
1. [Classes and Objects](#classes-and-objects)
2. [Attributes and Methods](#attributes-and-methods)
3. [Inheritance](#inheritance)
4. [Polymorphism](#polymorphism)
5. [Encapsulation](#encapsulation)
6. [Special Methods (Magic Methods)](#special-methods-magic-methods)
7. [Advanced OOP Concepts](#advanced-oop-concepts)
8. [Design Patterns](#design-patterns)
9. [Metaclasses](#metaclasses)
10. [Abstract Base Classes](#abstract-base-classes)
11. [Dataclasses and Type Hints](#dataclasses-and-type-hints)

## Classes and Objects

### 1. Basic Class Definition
```python
# Basic class definition
class Person:
    """A class representing a person."""
    
    # Class variable (shared by all instances)
    species = "Homo sapiens"
    
    def __init__(self, name, age):
        """Initialize a Person instance."""
        self.name = name  # Instance variable
        self.age = age    # Instance variable
    
    def introduce(self):
        """Introduce the person."""
        return f"Hi, I'm {self.name} and I'm {self.age} years old."

# Creating objects (instances)
person1 = Person("Alice", 25)
person2 = Person("Bob", 30)

print(person1.introduce())  # Hi, I'm Alice and I'm 25 years old.
print(person2.introduce())  # Hi, I'm Bob and I'm 30 years old.

# Accessing class variables
print(Person.species)       # Homo sapiens
print(person1.species)      # Homo sapiens
```

### 2. Class vs Instance Variables
```python
class BankAccount:
    # Class variable
    bank_name = "Python Bank"
    interest_rate = 0.05
    
    def __init__(self, account_holder, initial_balance=0):
        # Instance variables
        self.account_holder = account_holder
        self.balance = initial_balance
        self.account_number = self._generate_account_number()
    
    def _generate_account_number(self):
        """Generate a unique account number."""
        import random
        return f"ACC{random.randint(100000, 999999)}"
    
    def deposit(self, amount):
        """Deposit money into the account."""
        self.balance += amount
        return f"Deposited ${amount}. New balance: ${self.balance}"
    
    def withdraw(self, amount):
        """Withdraw money from the account."""
        if amount <= self.balance:
            self.balance -= amount
            return f"Withdrew ${amount}. New balance: ${self.balance}"
        else:
            return "Insufficient funds"

# Usage
account1 = BankAccount("Alice", 1000)
account2 = BankAccount("Bob", 500)

print(account1.deposit(200))    # Deposited $200. New balance: $1200
print(account2.withdraw(100))   # Withdrew $100. New balance: $400

# Modifying class variables
BankAccount.interest_rate = 0.06  # Affects all instances
print(f"New interest rate: {BankAccount.interest_rate}")
```

### 3. Instance, Class, and Static Methods
```python
import datetime

class Person:
    population = 0  # Class variable to track population
    
    def __init__(self, name, birth_year):
        self.name = name
        self.birth_year = birth_year
        Person.population += 1
    
    # Instance method (has access to self)
    def get_age(self):
        """Calculate current age."""
        current_year = datetime.datetime.now().year
        return current_year - self.birth_year
    
    # Class method (has access to cls, not self)
    @classmethod
    def get_population(cls):
        """Get current population count."""
        return cls.population
    
    @classmethod
    def from_age(cls, name, age):
        """Alternative constructor: create person from age."""
        current_year = datetime.datetime.now().year
        birth_year = current_year - age
        return cls(name, birth_year)
    
    # Static method (no access to self or cls)
    @staticmethod
    def is_adult(age):
        """Check if person is adult (18+)."""
        return age >= 18
    
    def __str__(self):
        return f"Person(name='{self.name}', age={self.get_age()})"

# Usage
person1 = Person("Alice", 1995)
person2 = Person.from_age("Bob", 25)  # Using class method

print(person1.get_age())              # Instance method
print(Person.get_population())        # Class method
print(Person.is_adult(person1.get_age()))  # Static method

print(person1)  # Person(name='Alice', age=30)
```

## Attributes and Methods

### 1. Property Decorators
```python
class Circle:
    def __init__(self, radius):
        self._radius = radius
    
    @property
    def radius(self):
        """Get the radius."""
        return self._radius
    
    @radius.setter
    def radius(self, value):
        """Set the radius with validation."""
        if value < 0:
            raise ValueError("Radius cannot be negative")
        self._radius = value
    
    @radius.deleter
    def radius(self):
        """Delete the radius."""
        print("Deleting radius")
        del self._radius
    
    @property
    def area(self):
        """Calculate area (read-only property)."""
        return 3.14159 * self._radius ** 2
    
    @property
    def diameter(self):
        """Calculate diameter (read-only property)."""
        return 2 * self._radius

# Usage
circle = Circle(5)
print(f"Radius: {circle.radius}")    # 5
print(f"Area: {circle.area}")        # 78.53975
print(f"Diameter: {circle.diameter}") # 10

circle.radius = 3                    # Using setter
print(f"New area: {circle.area}")    # 28.27431

# circle.area = 100  # Would raise AttributeError (read-only)
del circle.radius    # Using deleter
```

### 2. Descriptors
```python
class ValidatedAttribute:
    def __init__(self, min_value=None, max_value=None):
        self.min_value = min_value
        self.max_value = max_value
    
    def __set_name__(self, owner, name):
        self.name = f'_{name}'
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, self.name)
    
    def __set__(self, obj, value):
        if self.min_value is not None and value < self.min_value:
            raise ValueError(f"{self.name[1:]} must be >= {self.min_value}")
        if self.max_value is not None and value > self.max_value:
            raise ValueError(f"{self.name[1:]} must be <= {self.max_value}")
        setattr(obj, self.name, value)

class Student:
    grade = ValidatedAttribute(min_value=0, max_value=100)
    age = ValidatedAttribute(min_value=0, max_value=150)
    
    def __init__(self, name, age, grade):
        self.name = name
        self.age = age      # Uses descriptor validation
        self.grade = grade  # Uses descriptor validation

# Usage
student = Student("Alice", 20, 85)
print(f"Grade: {student.grade}")

student.grade = 95  # OK
# student.grade = 105  # Would raise ValueError
# student.age = -5     # Would raise ValueError
```

### 3. Dynamic Attributes
```python
class DynamicClass:
    def __init__(self):
        self.data = {}
    
    def __getattr__(self, name):
        """Called when attribute is not found in normal way."""
        if name in self.data:
            return self.data[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def __setattr__(self, name, value):
        """Called when setting any attribute."""
        if name == 'data':
            super().__setattr__(name, value)
        else:
            if not hasattr(self, 'data'):
                super().__setattr__('data', {})
            self.data[name] = value
    
    def __delattr__(self, name):
        """Called when deleting an attribute."""
        if name in self.data:
            del self.data[name]
        else:
            super().__delattr__(name)

# Usage
obj = DynamicClass()
obj.name = "Alice"          # Stored in data dict
obj.age = 25               # Stored in data dict
print(obj.name)            # Alice
print(obj.age)             # 25

del obj.age
# print(obj.age)           # Would raise AttributeError
```

## Inheritance

### 1. Single Inheritance
```python
# Base class (parent)
class Animal:
    def __init__(self, name, species):
        self.name = name
        self.species = species
    
    def make_sound(self):
        return "Some generic animal sound"
    
    def info(self):
        return f"{self.name} is a {self.species}"

# Derived class (child)
class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name, "Canine")  # Call parent constructor
        self.breed = breed
    
    def make_sound(self):  # Override parent method
        return "Woof!"
    
    def fetch(self):  # New method specific to Dog
        return f"{self.name} is fetching the ball!"

class Cat(Animal):
    def __init__(self, name, indoor=True):
        super().__init__(name, "Feline")
        self.indoor = indoor
    
    def make_sound(self):
        return "Meow!"
    
    def climb(self):
        return f"{self.name} is climbing!"

# Usage
dog = Dog("Buddy", "Golden Retriever")
cat = Cat("Whiskers", indoor=True)

print(dog.info())        # Buddy is a Canine
print(dog.make_sound())  # Woof!
print(dog.fetch())       # Buddy is fetching the ball!

print(cat.info())        # Whiskers is a Feline
print(cat.make_sound())  # Meow!

# Check inheritance
print(isinstance(dog, Animal))  # True
print(isinstance(dog, Dog))     # True
print(issubclass(Dog, Animal))  # True
```

### 2. Multiple Inheritance
```python
class Flyable:
    def fly(self):
        return f"{self.name} is flying!"

class Swimmable:
    def swim(self):
        return f"{self.name} is swimming!"

class Duck(Animal, Flyable, Swimmable):
    def __init__(self, name):
        super().__init__(name, "Waterfowl")
    
    def make_sound(self):
        return "Quack!"

class Penguin(Animal, Swimmable):
    def __init__(self, name):
        super().__init__(name, "Bird")
    
    def make_sound(self):
        return "Squawk!"

# Usage
duck = Duck("Donald")
penguin = Penguin("Pingu")

print(duck.fly())        # Donald is flying!
print(duck.swim())       # Donald is swimming!
print(duck.make_sound()) # Quack!

print(penguin.swim())    # Pingu is swimming!
# print(penguin.fly())   # Would raise AttributeError

# Method Resolution Order (MRO)
print(Duck.__mro__)
# (<class '__main__.Duck'>, <class '__main__.Animal'>, 
#  <class '__main__.Flyable'>, <class '__main__.Swimmable'>, <class 'object'>)
```

### 3. Abstract Classes
```python
from abc import ABC, abstractmethod

class Shape(ABC):
    """Abstract base class for shapes."""
    
    def __init__(self, name):
        self.name = name
    
    @abstractmethod
    def area(self):
        """Calculate area - must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def perimeter(self):
        """Calculate perimeter - must be implemented by subclasses."""
        pass
    
    def description(self):  # Concrete method
        return f"This is a {self.name}"

class Rectangle(Shape):
    def __init__(self, width, height):
        super().__init__("Rectangle")
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height
    
    def perimeter(self):
        return 2 * (self.width + self.height)

class Circle(Shape):
    def __init__(self, radius):
        super().__init__("Circle")
        self.radius = radius
    
    def area(self):
        return 3.14159 * self.radius ** 2
    
    def perimeter(self):
        return 2 * 3.14159 * self.radius

# Usage
# shape = Shape("Generic")  # Would raise TypeError - can't instantiate abstract class

rectangle = Rectangle(5, 3)
circle = Circle(4)

print(f"Rectangle area: {rectangle.area()}")      # 15
print(f"Circle area: {circle.area():.2f}")        # 50.27
print(rectangle.description())                     # This is a Rectangle
```

## Polymorphism

### 1. Method Overriding
```python
class Vehicle:
    def __init__(self, brand, model):
        self.brand = brand
        self.model = model
    
    def start(self):
        return f"{self.brand} {self.model} is starting..."
    
    def stop(self):
        return f"{self.brand} {self.model} is stopping..."

class Car(Vehicle):
    def start(self):
        return f"Car {self.brand} {self.model} engine is starting with a key..."

class Motorcycle(Vehicle):
    def start(self):
        return f"Motorcycle {self.brand} {self.model} engine is kick-starting..."

class ElectricCar(Vehicle):
    def start(self):
        return f"Electric car {self.brand} {self.model} is silently starting..."

# Polymorphic behavior
vehicles = [
    Car("Toyota", "Camry"),
    Motorcycle("Harley", "Davidson"),
    ElectricCar("Tesla", "Model 3")
]

for vehicle in vehicles:
    print(vehicle.start())  # Each calls its own version of start()
```

### 2. Duck Typing
```python
class Dog:
    def make_sound(self):
        return "Woof!"
    
    def move(self):
        return "Running on four legs"

class Bird:
    def make_sound(self):
        return "Tweet!"
    
    def move(self):
        return "Flying with wings"

class Robot:
    def make_sound(self):
        return "Beep!"
    
    def move(self):
        return "Rolling on wheels"

def animal_actions(entity):
    """If it walks like a duck and quacks like a duck, it's a duck."""
    print(f"Sound: {entity.make_sound()}")
    print(f"Movement: {entity.move()}")

# Duck typing in action
entities = [Dog(), Bird(), Robot()]

for entity in entities:
    animal_actions(entity)  # Works with any object that has these methods
```

### 3. Operator Overloading
```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        """Overload + operator."""
        return Vector(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        """Overload - operator."""
        return Vector(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        """Overload * operator for scalar multiplication."""
        return Vector(self.x * scalar, self.y * scalar)
    
    def __eq__(self, other):
        """Overload == operator."""
        return self.x == other.x and self.y == other.y
    
    def __str__(self):
        return f"Vector({self.x}, {self.y})"
    
    def __repr__(self):
        return f"Vector({self.x}, {self.y})"

# Usage
v1 = Vector(3, 4)
v2 = Vector(1, 2)

v3 = v1 + v2    # Vector(4, 6)
v4 = v1 - v2    # Vector(2, 2)
v5 = v1 * 2     # Vector(6, 8)

print(v3)       # Vector(4, 6)
print(v1 == v2) # False
```

## Encapsulation

### 1. Private and Protected Members
```python
class BankAccount:
    def __init__(self, account_number, initial_balance):
        self.account_number = account_number    # Public
        self._balance = initial_balance         # Protected (convention)
        self.__pin = 1234                      # Private (name mangling)
    
    def get_balance(self):
        """Public method to access protected data."""
        return self._balance
    
    def _validate_transaction(self, amount):
        """Protected method - internal use."""
        return amount > 0 and amount <= self._balance
    
    def __encrypt_data(self, data):
        """Private method - heavily name mangled."""
        return f"encrypted_{data}"
    
    def withdraw(self, amount, pin):
        """Public method with validation."""
        if pin != self.__pin:
            return "Invalid PIN"
        
        if self._validate_transaction(amount):
            self._balance -= amount
            return f"Withdrew ${amount}. Balance: ${self._balance}"
        else:
            return "Invalid transaction"
    
    def change_pin(self, old_pin, new_pin):
        """Change PIN with validation."""
        if old_pin == self.__pin:
            self.__pin = new_pin
            return "PIN changed successfully"
        return "Invalid old PIN"

# Usage
account = BankAccount("123456", 1000)

print(account.account_number)     # 123456 (public)
print(account.get_balance())      # 1000 (accessing protected via public method)

# Direct access to protected (works but not recommended)
print(account._balance)           # 1000

# Private members are name-mangled
# print(account.__pin)            # Would raise AttributeError
print(account._BankAccount__pin)  # 1234 (name-mangled access)

account.withdraw(100, 1234)       # Valid withdrawal
account.withdraw(100, 1111)       # Invalid PIN
```

### 2. Properties for Encapsulation
```python
class Temperature:
    def __init__(self, celsius=0):
        self._celsius = celsius
    
    @property
    def celsius(self):
        return self._celsius
    
    @celsius.setter
    def celsius(self, value):
        if value < -273.15:
            raise ValueError("Temperature cannot be below absolute zero")
        self._celsius = value
    
    @property
    def fahrenheit(self):
        return (self._celsius * 9/5) + 32
    
    @fahrenheit.setter
    def fahrenheit(self, value):
        self.celsius = (value - 32) * 5/9
    
    @property
    def kelvin(self):
        return self._celsius + 273.15
    
    @kelvin.setter
    def kelvin(self, value):
        self.celsius = value - 273.15

# Usage
temp = Temperature(25)
print(f"Celsius: {temp.celsius}")      # 25
print(f"Fahrenheit: {temp.fahrenheit}") # 77.0
print(f"Kelvin: {temp.kelvin}")         # 298.15

temp.fahrenheit = 100
print(f"Celsius: {temp.celsius}")       # 37.77777777777778

# temp.celsius = -300  # Would raise ValueError
```

## Special Methods (Magic Methods)

### 1. Object Representation
```python
class Book:
    def __init__(self, title, author, pages):
        self.title = title
        self.author = author
        self.pages = pages
    
    def __str__(self):
        """String representation for end users."""
        return f"'{self.title}' by {self.author}"
    
    def __repr__(self):
        """String representation for developers."""
        return f"Book('{self.title}', '{self.author}', {self.pages})"
    
    def __len__(self):
        """Length of the book (number of pages)."""
        return self.pages
    
    def __bool__(self):
        """Boolean representation (True if book has pages)."""
        return self.pages > 0

# Usage
book = Book("1984", "George Orwell", 328)

print(str(book))    # '1984' by George Orwell
print(repr(book))   # Book('1984', 'George Orwell', 328)
print(len(book))    # 328
print(bool(book))   # True

empty_book = Book("", "", 0)
print(bool(empty_book))  # False
```

### 2. Comparison Methods
```python
class Student:
    def __init__(self, name, grade):
        self.name = name
        self.grade = grade
    
    def __eq__(self, other):
        """Equal to."""
        return self.grade == other.grade
    
    def __lt__(self, other):
        """Less than."""
        return self.grade < other.grade
    
    def __le__(self, other):
        """Less than or equal to."""
        return self.grade <= other.grade
    
    def __gt__(self, other):
        """Greater than."""
        return self.grade > other.grade
    
    def __ge__(self, other):
        """Greater than or equal to."""
        return self.grade >= other.grade
    
    def __ne__(self, other):
        """Not equal to."""
        return self.grade != other.grade
    
    def __str__(self):
        return f"{self.name} (Grade: {self.grade})"

# Usage
student1 = Student("Alice", 85)
student2 = Student("Bob", 92)
student3 = Student("Charlie", 85)

print(student1 == student3)  # True
print(student1 < student2)   # True
print(student2 > student1)   # True

# Sorting becomes possible
students = [student1, student2, student3]
sorted_students = sorted(students)
for student in sorted_students:
    print(student)
```

### 3. Container Methods
```python
class Playlist:
    def __init__(self, name):
        self.name = name
        self.songs = []
    
    def __len__(self):
        """Return number of songs."""
        return len(self.songs)
    
    def __getitem__(self, index):
        """Get song by index."""
        return self.songs[index]
    
    def __setitem__(self, index, song):
        """Set song at index."""
        self.songs[index] = song
    
    def __delitem__(self, index):
        """Delete song at index."""
        del self.songs[index]
    
    def __contains__(self, song):
        """Check if song is in playlist."""
        return song in self.songs
    
    def __iter__(self):
        """Make playlist iterable."""
        return iter(self.songs)
    
    def append(self, song):
        """Add song to playlist."""
        self.songs.append(song)

# Usage
playlist = Playlist("My Favorites")
playlist.append("Song 1")
playlist.append("Song 2")
playlist.append("Song 3")

print(len(playlist))              # 3
print(playlist[0])                # Song 1
print("Song 2" in playlist)       # True

# Iteration
for song in playlist:
    print(f"Playing: {song}")

# Slicing works too
print(playlist[1:3])              # ['Song 2', 'Song 3']
```

## Advanced OOP Concepts

### 1. Metaclasses
```python
class SingletonMeta(type):
    """Metaclass that creates singleton instances."""
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Database(metaclass=SingletonMeta):
    """Singleton database connection."""
    
    def __init__(self):
        self.connection = "Connected to database"

# Usage
db1 = Database()
db2 = Database()

print(db1 is db2)  # True - same instance
print(id(db1))     # Same ID
print(id(db2))     # Same ID
```

### 2. Context Managers
```python
class FileManager:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = None
    
    def __enter__(self):
        """Enter the context."""
        print(f"Opening file {self.filename}")
        self.file = open(self.filename, self.mode)
        return self.file
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context."""
        print(f"Closing file {self.filename}")
        if self.file:
            self.file.close()
        
        if exc_type:
            print(f"Exception occurred: {exc_val}")
        
        return False  # Don't suppress exceptions

# Usage
with FileManager("test.txt", "w") as f:
    f.write("Hello, World!")
# File is automatically closed
```

### 3. Composition vs Inheritance
```python
# Composition example
class Engine:
    def __init__(self, horsepower):
        self.horsepower = horsepower
    
    def start(self):
        return f"Engine with {self.horsepower} HP is starting"

class Transmission:
    def __init__(self, type_):
        self.type = type_
    
    def shift(self, gear):
        return f"{self.type} transmission shifting to gear {gear}"

class Car:
    """Car uses composition - HAS-A relationship."""
    
    def __init__(self, make, model, engine, transmission):
        self.make = make
        self.model = model
        self.engine = engine        # Composition
        self.transmission = transmission  # Composition
    
    def start(self):
        return f"{self.make} {self.model}: {self.engine.start()}"
    
    def drive(self):
        return f"Driving: {self.transmission.shift(1)}"

# Usage
engine = Engine(200)
transmission = Transmission("Manual")
car = Car("Toyota", "Camry", engine, transmission)

print(car.start())  # Toyota Camry: Engine with 200 HP is starting
print(car.drive())  # Driving: Manual transmission shifting to gear 1
```

### 4. Mixins
```python
class JSONMixin:
    """Mixin to add JSON serialization capability."""
    
    def to_json(self):
        """Convert object to JSON string."""
        import json
        return json.dumps(self.__dict__)
    
    @classmethod
    def from_json(cls, json_str):
        """Create object from JSON string."""
        import json
        data = json.loads(json_str)
        return cls(**data)

class TimestampMixin:
    """Mixin to add timestamp functionality."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from datetime import datetime
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    def touch(self):
        """Update the timestamp."""
        from datetime import datetime
        self.updated_at = datetime.now()

class User(JSONMixin, TimestampMixin):
    """User class with JSON and timestamp capabilities."""
    
    def __init__(self, name, email):
        self.name = name
        self.email = email
        super().__init__()  # Initialize mixins

# Usage
user = User("Alice", "alice@example.com")
print(user.created_at)

json_str = user.to_json()
print(json_str)

# Note: from_json won't work perfectly here due to datetime objects
# This is a simplified example
```

## Best Practices

1. **Follow SOLID Principles**:
```python
# Single Responsibility Principle
class EmailSender:
    def send_email(self, message, recipient):
        # Only responsible for sending emails
        pass

class UserManager:
    def create_user(self, user_data):
        # Only responsible for user management
        pass

# Open/Closed Principle
class PaymentProcessor:
    def process(self, payment_method):
        return payment_method.process()

class CreditCardPayment:
    def process(self):
        return "Processing credit card payment"

class PayPalPayment:
    def process(self):
        return "Processing PayPal payment"
```

2. **Use composition over inheritance when appropriate**:
```python
# Instead of deep inheritance hierarchies
class Vehicle:
    def __init__(self, engine, wheels):
        self.engine = engine
        self.wheels = wheels
```

3. **Keep interfaces simple and focused**:
```python
from abc import ABC, abstractmethod

class Drawable(ABC):
    @abstractmethod
    def draw(self):
        pass

class Movable(ABC):
    @abstractmethod
    def move(self, x, y):
        pass

# Classes implement only what they need
class Circle(Drawable):
    def draw(self):
        return "Drawing a circle"

class MovableCircle(Circle, Movable):
    def move(self, x, y):
        return f"Moving circle to ({x}, {y})"
```

4. **Use meaningful names and documentation**:
```python
class ShoppingCart:
    """Manages items in a shopping cart."""
    
    def add_item(self, product, quantity=1):
        """Add a product to the cart."""
        pass
    
    def calculate_total(self):
        """Calculate the total price of all items."""
        pass
    
    def apply_discount(self, discount_code):
        """Apply a discount code to the cart."""
        pass
```

## Design Patterns

### 1. Singleton Pattern

```python
class Singleton:
    """Singleton pattern implementation"""
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.value = 0
            self._initialized = True

# Test singleton
s1 = Singleton()
s2 = Singleton()
print(s1 is s2)  # True

# Better singleton using decorator
def singleton(cls):
    """Singleton decorator"""
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

@singleton
class DatabaseConnection:
    def __init__(self):
        self.connection = "Connected to database"

db1 = DatabaseConnection()
db2 = DatabaseConnection()
print(db1 is db2)  # True
```

### 2. Factory Pattern

```python
from abc import ABC, abstractmethod

class Animal(ABC):
    @abstractmethod
    def make_sound(self):
        pass

class Dog(Animal):
    def make_sound(self):
        return "Woof!"

class Cat(Animal):
    def make_sound(self):
        return "Meow!"

class AnimalFactory:
    """Factory for creating animals"""
    
    @staticmethod
    def create_animal(animal_type):
        if animal_type.lower() == "dog":
            return Dog()
        elif animal_type.lower() == "cat":
            return Cat()
        else:
            raise ValueError(f"Unknown animal type: {animal_type}")

# Usage
factory = AnimalFactory()
dog = factory.create_animal("dog")
cat = factory.create_animal("cat")

print(dog.make_sound())  # Woof!
print(cat.make_sound())  # Meow!
```

### 3. Observer Pattern

```python
class Subject:
    """Subject in observer pattern"""
    
    def __init__(self):
        self._observers = []
        self._state = None
    
    def attach(self, observer):
        """Attach an observer"""
        if observer not in self._observers:
            self._observers.append(observer)
    
    def detach(self, observer):
        """Detach an observer"""
        if observer in self._observers:
            self._observers.remove(observer)
    
    def notify(self):
        """Notify all observers"""
        for observer in self._observers:
            observer.update(self)
    
    def set_state(self, state):
        """Set state and notify observers"""
        self._state = state
        self.notify()
    
    def get_state(self):
        return self._state

class Observer(ABC):
    @abstractmethod
    def update(self, subject):
        pass

class ConcreteObserver(Observer):
    def __init__(self, name):
        self.name = name
    
    def update(self, subject):
        print(f"{self.name} received update: {subject.get_state()}")

# Usage
subject = Subject()
observer1 = ConcreteObserver("Observer 1")
observer2 = ConcreteObserver("Observer 2")

subject.attach(observer1)
subject.attach(observer2)

subject.set_state("New State")
# Output:
# Observer 1 received update: New State
# Observer 2 received update: New State
```

### 4. Strategy Pattern

```python
class SortStrategy(ABC):
    @abstractmethod
    def sort(self, data):
        pass

class BubbleSort(SortStrategy):
    def sort(self, data):
        """Bubble sort implementation"""
        arr = data.copy()
        n = len(arr)
        for i in range(n):
            for j in range(0, n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
        return arr

class QuickSort(SortStrategy):
    def sort(self, data):
        """Quick sort implementation"""
        if len(data) <= 1:
            return data
        pivot = data[len(data) // 2]
        left = [x for x in data if x < pivot]
        middle = [x for x in data if x == pivot]
        right = [x for x in data if x > pivot]
        return self.sort(left) + middle + self.sort(right)

class SortContext:
    def __init__(self, strategy: SortStrategy):
        self._strategy = strategy
    
    def set_strategy(self, strategy: SortStrategy):
        self._strategy = strategy
    
    def sort_data(self, data):
        return self._strategy.sort(data)

# Usage
data = [64, 34, 25, 12, 22, 11, 90]

context = SortContext(BubbleSort())
result1 = context.sort_data(data)
print(f"Bubble sort: {result1}")

context.set_strategy(QuickSort())
result2 = context.sort_data(data)
print(f"Quick sort: {result2}")
```

## Metaclasses

### Understanding Metaclasses

```python
# Basic metaclass example
class MyMetaclass(type):
    """Custom metaclass"""
    
    def __new__(mcs, name, bases, namespace):
        # Modify class creation
        namespace['class_id'] = f"ID_{name}"
        return super().__new__(mcs, name, bases, namespace)
    
    def __init__(cls, name, bases, namespace):
        # Initialize the class
        super().__init__(name, bases, namespace)
        print(f"Creating class: {name}")

class MyClass(metaclass=MyMetaclass):
    def __init__(self, value):
        self.value = value

# Class has the added attribute
print(MyClass.class_id)  # ID_MyClass

# Metaclass for automatic property creation
class AutoPropertyMeta(type):
    """Metaclass that creates properties automatically"""
    
    def __new__(mcs, name, bases, namespace):
        # Find attributes that should become properties
        for key, value in list(namespace.items()):
            if key.startswith('_') and not key.startswith('__'):
                property_name = key[1:]  # Remove leading underscore
                
                def make_property(attr_name):
                    def getter(self):
                        return getattr(self, attr_name)
                    
                    def setter(self, value):
                        setattr(self, attr_name, value)
                    
                    return property(getter, setter)
                
                namespace[property_name] = make_property(key)
        
        return super().__new__(mcs, name, bases, namespace)

class Person(metaclass=AutoPropertyMeta):
    def __init__(self, name, age):
        self._name = name
        self._age = age

# Properties are automatically created
person = Person("Alice", 25)
print(person.name)  # Alice
person.age = 26
print(person.age)   # 26
```

### Metaclass Use Cases

```python
# Singleton metaclass
class SingletonMeta(type):
    """Metaclass for creating singleton classes"""
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class DatabaseConnection(metaclass=SingletonMeta):
    def __init__(self):
        self.connection = "Connected"

db1 = DatabaseConnection()
db2 = DatabaseConnection()
print(db1 is db2)  # True

# Registry metaclass
class RegistryMeta(type):
    """Metaclass that maintains a registry of classes"""
    registry = {}
    
    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        mcs.registry[name] = cls
        return cls

class BasePlugin(metaclass=RegistryMeta):
    pass

class EmailPlugin(BasePlugin):
    def send(self, message):
        return f"Sending email: {message}"

class SMSPlugin(BasePlugin):
    def send(self, message):
        return f"Sending SMS: {message}"

# Access registered classes
print(RegistryMeta.registry)  # Contains EmailPlugin and SMSPlugin
```

## Abstract Base Classes

### Using ABC Module

```python
from abc import ABC, abstractmethod, abstractproperty

class Shape(ABC):
    """Abstract base class for shapes"""
    
    @abstractmethod
    def area(self):
        """Calculate the area of the shape"""
        pass
    
    @abstractmethod
    def perimeter(self):
        """Calculate the perimeter of the shape"""
        pass
    
    @property
    @abstractmethod
    def name(self):
        """Name of the shape"""
        pass
    
    # Concrete method
    def description(self):
        return f"This is a {self.name} with area {self.area()}"

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height
    
    def perimeter(self):
        return 2 * (self.width + self.height)
    
    @property
    def name(self):
        return "Rectangle"

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    
    def area(self):
        import math
        return math.pi * self.radius ** 2
    
    def perimeter(self):
        import math
        return 2 * math.pi * self.radius
    
    @property
    def name(self):
        return "Circle"

# Usage
rectangle = Rectangle(5, 3)
circle = Circle(3)

print(rectangle.description())
print(circle.description())

# Cannot instantiate abstract class
# shape = Shape()  # TypeError: Can't instantiate abstract class

# Abstract class with template method
class DataProcessor(ABC):
    """Template method pattern with ABC"""
    
    def process(self, data):
        """Template method"""
        cleaned_data = self.clean_data(data)
        processed_data = self.process_data(cleaned_data)
        return self.format_output(processed_data)
    
    @abstractmethod
    def clean_data(self, data):
        pass
    
    @abstractmethod
    def process_data(self, data):
        pass
    
    def format_output(self, data):
        """Default implementation"""
        return str(data)

class NumberProcessor(DataProcessor):
    def clean_data(self, data):
        return [x for x in data if isinstance(x, (int, float))]
    
    def process_data(self, data):
        return sum(data)

processor = NumberProcessor()
result = processor.process([1, 2, "hello", 3.5, None, 4])
print(result)  # 10.5
```

## Dataclasses and Type Hints

### Dataclasses

```python
from dataclasses import dataclass, field
from typing import List, Optional, ClassVar
from datetime import datetime

@dataclass
class Person:
    """Person dataclass with type hints"""
    name: str
    age: int
    email: Optional[str] = None
    active: bool = True
    
    def __post_init__(self):
        """Called after __init__"""
        if self.age < 0:
            raise ValueError("Age cannot be negative")

# Usage
person = Person("Alice", 25, "alice@example.com")
print(person)  # Person(name='Alice', age=25, email='alice@example.com', active=True)

# Dataclass with default factory
@dataclass
class ShoppingCart:
    items: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    
    def add_item(self, item: str):
        self.items.append(item)

cart1 = ShoppingCart()
cart2 = ShoppingCart()
cart1.add_item("Apple")

print(cart1.items)  # ['Apple']
print(cart2.items)  # [] (separate lists)

# Dataclass with advanced features
@dataclass(frozen=True, order=True)  # Immutable and orderable
class Point:
    x: float
    y: float
    
    def distance_from_origin(self) -> float:
        return (self.x ** 2 + self.y ** 2) ** 0.5

points = [Point(3, 4), Point(1, 1), Point(0, 0)]
points.sort()  # Can sort due to order=True
print(points)

# point = Point(1, 2)
# point.x = 5  # AttributeError: can't set attribute (frozen=True)

# Dataclass with class variables
@dataclass
class BankAccount:
    owner: str
    balance: float = 0.0
    
    # Class variable
    bank_name: ClassVar[str] = "Python Bank"
    interest_rate: ClassVar[float] = 0.05
    
    def apply_interest(self):
        self.balance *= (1 + self.interest_rate)

account = BankAccount("Alice", 1000)
account.apply_interest()
print(account.balance)  # 1050.0
```

### Advanced Type Hints

```python
from typing import (
    Union, Optional, List, Dict, Tuple, Set,
    Callable, Generic, TypeVar, Protocol
)

# Union types
def process_id(user_id: Union[int, str]) -> str:
    return str(user_id)

# Optional (shorthand for Union[T, None])
def greet(name: Optional[str] = None) -> str:
    if name is None:
        return "Hello, World!"
    return f"Hello, {name}!"

# Generic types
T = TypeVar('T')

class Stack(Generic[T]):
    def __init__(self):
        self._items: List[T] = []
    
    def push(self, item: T) -> None:
        self._items.append(item)
    
    def pop(self) -> T:
        return self._items.pop()

# Usage with type hints
string_stack: Stack[str] = Stack()
string_stack.push("hello")

int_stack: Stack[int] = Stack()
int_stack.push(42)

# Protocols (structural typing)
class Drawable(Protocol):
    def draw(self) -> None: ...

class Circle:
    def draw(self) -> None:
        print("Drawing circle")

class Rectangle:
    def draw(self) -> None:
        print("Drawing rectangle")

def render_shape(shape: Drawable) -> None:
    shape.draw()

# Both Circle and Rectangle satisfy the Drawable protocol
render_shape(Circle())
render_shape(Rectangle())

# Complex type annotations
def process_data(
    data: Dict[str, List[Tuple[int, str]]],
    processor: Callable[[int], bool],
    default: Optional[str] = None
) -> Set[str]:
    """Process complex nested data structure"""
    result = set()
    for key, items in data.items():
        for number, text in items:
            if processor(number):
                result.add(text)
    return result

# Custom type aliases
UserId = int
UserName = str
UserData = Dict[UserId, UserName]

def get_user(user_id: UserId, users: UserData) -> Optional[UserName]:
    return users.get(user_id)

print("\nAdvanced OOP concepts completed!")
```
