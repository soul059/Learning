# JavaScript Intermediate Concepts

## Table of Contents
1. [ES6+ Features](#es6-features)
2. [Array Methods](#array-methods)
3. [Object-Oriented Programming](#object-oriented-programming)
4. [DOM Manipulation](#dom-manipulation)
5. [Events](#events)
6. [Error Handling](#error-handling)
7. [Regular Expressions](#regular-expressions)
8. [JSON](#json)
9. [Local Storage](#local-storage)
10. [Modules](#modules)

## ES6+ Features

### Template Literals:
```javascript
const name = "John";
const age = 30;

// Multi-line strings with variables
const message = `
  Hello, my name is ${name}.
  I am ${age} years old.
  Next year I'll be ${age + 1}.
`;

// Tagged template literals
function highlight(strings, ...values) {
  return strings.reduce((result, string, i) => {
    return result + string + (values[i] ? `<mark>${values[i]}</mark>` : '');
  }, '');
}

const highlighted = highlight`Hello ${name}, you are ${age} years old!`;
```

### Destructuring:

#### Array Destructuring:
```javascript
const numbers = [1, 2, 3, 4, 5];

// Basic destructuring
const [first, second] = numbers;
console.log(first, second); // 1, 2

// Skip elements
const [a, , c] = numbers;
console.log(a, c); // 1, 3

// Rest operator
const [head, ...tail] = numbers;
console.log(head); // 1
console.log(tail); // [2, 3, 4, 5]

// Default values
const [x = 0, y = 0] = [1];
console.log(x, y); // 1, 0
```

#### Object Destructuring:
```javascript
const person = {
  name: "John",
  age: 30,
  city: "New York",
  country: "USA"
};

// Basic destructuring
const { name, age } = person;

// Renaming variables
const { name: personName, age: personAge } = person;

// Default values
const { name, occupation = "Unknown" } = person;

// Nested destructuring
const user = {
  id: 1,
  profile: {
    name: "John",
    avatar: "avatar.jpg"
  }
};

const { profile: { name: userName, avatar } } = user;

// Function parameter destructuring
function greet({ name, age }) {
  return `Hello ${name}, you are ${age} years old`;
}

greet(person);
```

### Spread Operator:
```javascript
// Arrays
const arr1 = [1, 2, 3];
const arr2 = [4, 5, 6];
const combined = [...arr1, ...arr2]; // [1, 2, 3, 4, 5, 6]

// Objects
const obj1 = { a: 1, b: 2 };
const obj2 = { c: 3, d: 4 };
const merged = { ...obj1, ...obj2 }; // { a: 1, b: 2, c: 3, d: 4 }

// Function arguments
function sum(a, b, c) {
  return a + b + c;
}

const numbers = [1, 2, 3];
console.log(sum(...numbers)); // 6

// Copying arrays/objects
const originalArray = [1, 2, 3];
const copiedArray = [...originalArray];

const originalObject = { name: "John" };
const copiedObject = { ...originalObject };
```

### Enhanced Object Literals:
```javascript
const name = "John";
const age = 30;

// Property shorthand
const person = { name, age }; // Same as { name: name, age: age }

// Method shorthand
const calculator = {
  add(a, b) {
    return a + b;
  },
  multiply(a, b) {
    return a * b;
  }
};

// Computed property names
const property = "dynamicKey";
const obj = {
  [property]: "value",
  [`${property}2`]: "value2"
};
```

### Classes:
```javascript
class Person {
  constructor(name, age) {
    this.name = name;
    this.age = age;
  }
  
  // Instance method
  greet() {
    return `Hello, I'm ${this.name}`;
  }
  
  // Static method
  static species() {
    return "Homo sapiens";
  }
  
  // Getter
  get info() {
    return `${this.name} (${this.age})`;
  }
  
  // Setter
  set newAge(age) {
    if (age > 0) {
      this.age = age;
    }
  }
}

// Inheritance
class Student extends Person {
  constructor(name, age, grade) {
    super(name, age); // Call parent constructor
    this.grade = grade;
  }
  
  study() {
    return `${this.name} is studying`;
  }
  
  // Override parent method
  greet() {
    return `${super.greet()}, I'm a student`;
  }
}

const john = new Person("John", 30);
const alice = new Student("Alice", 20, "A");

console.log(john.greet()); // "Hello, I'm John"
console.log(alice.study()); // "Alice is studying"
console.log(Person.species()); // "Homo sapiens"
```

### Symbols:
```javascript
// Creating symbols
const sym1 = Symbol();
const sym2 = Symbol('description');
const sym3 = Symbol('description');

console.log(sym2 === sym3); // false (symbols are unique)

// Using symbols as object keys
const id = Symbol('id');
const user = {
  name: "John",
  [id]: 123
};

console.log(user[id]); // 123

// Well-known symbols
const obj = {
  [Symbol.iterator]: function* () {
    yield 1;
    yield 2;
    yield 3;
  }
};

for (const value of obj) {
  console.log(value); // 1, 2, 3
}
```

### Maps and Sets:

#### Maps:
```javascript
// Creating a Map
const map = new Map();

// Setting values
map.set('name', 'John');
map.set('age', 30);
map.set(1, 'number key');

// Getting values
console.log(map.get('name')); // "John"
console.log(map.has('age')); // true
console.log(map.size); // 3

// Iterating over Map
for (const [key, value] of map) {
  console.log(key, value);
}

// Map with initial values
const fruits = new Map([
  ['apple', 5],
  ['banana', 3],
  ['orange', 8]
]);

// Map methods
console.log([...map.keys()]); // ['name', 'age', 1]
console.log([...map.values()]); // ['John', 30, 'number key']
console.log([...map.entries()]); // [['name', 'John'], ...]
```

#### Sets:
```javascript
// Creating a Set
const set = new Set();

// Adding values
set.add(1);
set.add(2);
set.add(2); // Duplicates are ignored
set.add('hello');

console.log(set.size); // 3

// Checking for values
console.log(set.has(1)); // true
console.log(set.has(3)); // false

// Iterating over Set
for (const value of set) {
  console.log(value);
}

// Set with initial values
const numbers = new Set([1, 2, 3, 3, 4, 4, 5]);
console.log(numbers); // Set(5) {1, 2, 3, 4, 5}

// Converting Set to Array
const uniqueArray = [...numbers];

// Set operations
const setA = new Set([1, 2, 3]);
const setB = new Set([3, 4, 5]);

// Union
const union = new Set([...setA, ...setB]);

// Intersection
const intersection = new Set([...setA].filter(x => setB.has(x)));

// Difference
const difference = new Set([...setA].filter(x => !setB.has(x)));
```

## Array Methods

### Transformation Methods:

#### map():
```javascript
const numbers = [1, 2, 3, 4, 5];

// Transform each element
const doubled = numbers.map(num => num * 2);
console.log(doubled); // [2, 4, 6, 8, 10]

// With index and array parameters
const withIndex = numbers.map((num, index, arr) => ({
  value: num,
  index: index,
  isLast: index === arr.length - 1
}));
```

#### filter():
```javascript
const numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

// Filter even numbers
const evens = numbers.filter(num => num % 2 === 0);
console.log(evens); // [2, 4, 6, 8, 10]

// Filter objects
const people = [
  { name: "John", age: 30 },
  { name: "Alice", age: 25 },
  { name: "Bob", age: 35 }
];

const adults = people.filter(person => person.age >= 30);
```

#### reduce():
```javascript
const numbers = [1, 2, 3, 4, 5];

// Sum all numbers
const sum = numbers.reduce((acc, num) => acc + num, 0);
console.log(sum); // 15

// Find maximum
const max = numbers.reduce((acc, num) => Math.max(acc, num), -Infinity);

// Group by property
const people = [
  { name: "John", department: "IT" },
  { name: "Alice", department: "HR" },
  { name: "Bob", department: "IT" }
];

const grouped = people.reduce((acc, person) => {
  const dept = person.department;
  if (!acc[dept]) {
    acc[dept] = [];
  }
  acc[dept].push(person);
  return acc;
}, {});

// Count occurrences
const fruits = ['apple', 'banana', 'apple', 'orange', 'banana', 'apple'];
const count = fruits.reduce((acc, fruit) => {
  acc[fruit] = (acc[fruit] || 0) + 1;
  return acc;
}, {});
```

### Search Methods:

#### find() and findIndex():
```javascript
const users = [
  { id: 1, name: "John", active: true },
  { id: 2, name: "Alice", active: false },
  { id: 3, name: "Bob", active: true }
];

// Find first matching element
const activeUser = users.find(user => user.active);
console.log(activeUser); // { id: 1, name: "John", active: true }

// Find index of first matching element
const inactiveIndex = users.findIndex(user => !user.active);
console.log(inactiveIndex); // 1
```

#### some() and every():
```javascript
const numbers = [1, 2, 3, 4, 5];

// Check if any element meets condition
const hasEven = numbers.some(num => num % 2 === 0);
console.log(hasEven); // true

// Check if all elements meet condition
const allPositive = numbers.every(num => num > 0);
console.log(allPositive); // true

const allEven = numbers.every(num => num % 2 === 0);
console.log(allEven); // false
```

### Utility Methods:

#### forEach():
```javascript
const fruits = ['apple', 'banana', 'orange'];

fruits.forEach((fruit, index) => {
  console.log(`${index}: ${fruit}`);
});

// Note: forEach doesn't return a new array
// Use map() if you need to transform and return
```

#### sort():
```javascript
const numbers = [3, 1, 4, 1, 5, 9, 2, 6];

// Sort numbers (convert to string by default)
numbers.sort(); // [1, 1, 2, 3, 4, 5, 6, 9] - but actually ["1", "1", "2", "3", "4", "5", "6", "9"]

// Sort numbers numerically
numbers.sort((a, b) => a - b); // [1, 1, 2, 3, 4, 5, 6, 9]

// Sort in descending order
numbers.sort((a, b) => b - a); // [9, 6, 5, 4, 3, 2, 1, 1]

// Sort objects
const people = [
  { name: "John", age: 30 },
  { name: "Alice", age: 25 },
  { name: "Bob", age: 35 }
];

people.sort((a, b) => a.age - b.age); // Sort by age ascending
people.sort((a, b) => a.name.localeCompare(b.name)); // Sort by name
```

#### Method Chaining:
```javascript
const numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

const result = numbers
  .filter(num => num % 2 === 0) // Get even numbers
  .map(num => num * num) // Square them
  .reduce((sum, num) => sum + num, 0); // Sum them up

console.log(result); // 220 (4 + 16 + 36 + 64 + 100)
```

## Object-Oriented Programming

### Constructor Functions (ES5 Style):
```javascript
function Person(name, age) {
  this.name = name;
  this.age = age;
}

// Adding methods to prototype
Person.prototype.greet = function() {
  return `Hello, I'm ${this.name}`;
};

Person.prototype.getAge = function() {
  return this.age;
};

// Creating instances
const john = new Person("John", 30);
const alice = new Person("Alice", 25);

console.log(john.greet()); // "Hello, I'm John"
```

### Prototypal Inheritance:
```javascript
// Parent constructor
function Animal(name) {
  this.name = name;
}

Animal.prototype.speak = function() {
  return `${this.name} makes a sound`;
};

// Child constructor
function Dog(name, breed) {
  Animal.call(this, name); // Call parent constructor
  this.breed = breed;
}

// Set up inheritance
Dog.prototype = Object.create(Animal.prototype);
Dog.prototype.constructor = Dog;

// Add child-specific methods
Dog.prototype.bark = function() {
  return `${this.name} barks`;
};

// Override parent method
Dog.prototype.speak = function() {
  return `${this.name} barks loudly`;
};

const dog = new Dog("Rex", "German Shepherd");
console.log(dog.speak()); // "Rex barks loudly"
console.log(dog.bark()); // "Rex barks"
```

### Modern Class Syntax:
```javascript
class Vehicle {
  constructor(make, model, year) {
    this.make = make;
    this.model = model;
    this.year = year;
    this._mileage = 0; // Private convention with underscore
  }
  
  // Instance method
  start() {
    return `${this.make} ${this.model} is starting`;
  }
  
  // Getter
  get age() {
    return new Date().getFullYear() - this.year;
  }
  
  // Setter
  set mileage(miles) {
    if (miles >= 0) {
      this._mileage = miles;
    }
  }
  
  get mileage() {
    return this._mileage;
  }
  
  // Static method
  static compare(vehicle1, vehicle2) {
    return vehicle1.year - vehicle2.year;
  }
}

class Car extends Vehicle {
  constructor(make, model, year, doors) {
    super(make, model, year);
    this.doors = doors;
  }
  
  // Override parent method
  start() {
    return `${super.start()} with ${this.doors} doors`;
  }
  
  // Child-specific method
  honk() {
    return "Beep beep!";
  }
}

const car1 = new Car("Toyota", "Camry", 2020, 4);
const car2 = new Car("Honda", "Civic", 2019, 4);

console.log(car1.start()); // "Toyota Camry is starting with 4 doors"
console.log(car1.age); // 5 (assuming current year is 2025)
console.log(Vehicle.compare(car1, car2)); // 1
```

### Private Fields and Methods (ES2022):
```javascript
class BankAccount {
  #balance = 0; // Private field
  #accountNumber; // Private field
  
  constructor(accountNumber, initialBalance = 0) {
    this.#accountNumber = accountNumber;
    this.#balance = initialBalance;
  }
  
  // Private method
  #validateAmount(amount) {
    return amount > 0 && typeof amount === 'number';
  }
  
  // Public methods
  deposit(amount) {
    if (this.#validateAmount(amount)) {
      this.#balance += amount;
      return this.#balance;
    }
    throw new Error('Invalid amount');
  }
  
  withdraw(amount) {
    if (this.#validateAmount(amount) && amount <= this.#balance) {
      this.#balance -= amount;
      return this.#balance;
    }
    throw new Error('Invalid amount or insufficient funds');
  }
  
  get balance() {
    return this.#balance;
  }
  
  get accountInfo() {
    return `Account ${this.#accountNumber}: $${this.#balance}`;
  }
}

const account = new BankAccount("12345", 1000);
console.log(account.deposit(500)); // 1500
console.log(account.balance); // 1500
// console.log(account.#balance); // SyntaxError: Private field '#balance' must be declared in an enclosing class
```

### Mixins:
```javascript
// Mixin for adding flying capability
const Flyable = {
  fly() {
    return `${this.name} is flying`;
  },
  
  land() {
    return `${this.name} has landed`;
  }
};

// Mixin for adding swimming capability
const Swimmable = {
  swim() {
    return `${this.name} is swimming`;
  },
  
  dive() {
    return `${this.name} is diving`;
  }
};

class Bird {
  constructor(name) {
    this.name = name;
  }
}

class Duck extends Bird {
  constructor(name) {
    super(name);
  }
}

// Add mixins to Duck prototype
Object.assign(Duck.prototype, Flyable, Swimmable);

const duck = new Duck("Daffy");
console.log(duck.fly()); // "Daffy is flying"
console.log(duck.swim()); // "Daffy is swimming"
```

## DOM Manipulation

### Selecting Elements:
```javascript
// Select by ID
const header = document.getElementById('header');

// Select by class name
const buttons = document.getElementsByClassName('btn');
const firstButton = document.getElementsByClassName('btn')[0];

// Select by tag name
const paragraphs = document.getElementsByTagName('p');

// Query selectors (more flexible)
const firstBtn = document.querySelector('.btn'); // First element with class 'btn'
const allBtns = document.querySelectorAll('.btn'); // All elements with class 'btn'
const complexSelect = document.querySelector('div.container > p:first-child');
```

### Modifying Content:
```javascript
const element = document.getElementById('myElement');

// Text content (safer, escapes HTML)
element.textContent = 'New text content';

// HTML content (can include HTML tags)
element.innerHTML = '<strong>Bold text</strong>';

// Attributes
element.setAttribute('class', 'new-class');
element.setAttribute('data-id', '123');
const classValue = element.getAttribute('class');

// Properties
element.id = 'newId';
element.className = 'class1 class2';
element.value = 'new value'; // For form elements

// Style
element.style.color = 'red';
element.style.backgroundColor = 'yellow';
element.style.fontSize = '16px';

// CSS classes
element.classList.add('new-class');
element.classList.remove('old-class');
element.classList.toggle('active');
element.classList.contains('visible'); // Returns boolean
```

### Creating and Modifying Elements:
```javascript
// Create new element
const newDiv = document.createElement('div');
newDiv.textContent = 'Hello World';
newDiv.className = 'my-div';

// Create text node
const textNode = document.createTextNode('This is text');

// Append to parent
const container = document.getElementById('container');
container.appendChild(newDiv);

// Insert before specific element
const existingElement = document.getElementById('existing');
container.insertBefore(newDiv, existingElement);

// More modern insertion methods
container.append(newDiv); // Adds to end
container.prepend(newDiv); // Adds to beginning
existingElement.before(newDiv); // Inserts before element
existingElement.after(newDiv); // Inserts after element

// Replace element
const oldElement = document.getElementById('old');
const newElement = document.createElement('span');
newElement.textContent = 'Replacement';
oldElement.replaceWith(newElement);

// Remove element
const elementToRemove = document.getElementById('remove-me');
elementToRemove.remove();

// Clone element
const original = document.getElementById('original');
const clone = original.cloneNode(true); // true for deep clone
```

### Traversing the DOM:
```javascript
const element = document.getElementById('myElement');

// Parent relationships
const parent = element.parentNode;
const parentElement = element.parentElement;

// Child relationships
const children = element.children; // HTMLCollection of child elements
const childNodes = element.childNodes; // NodeList including text nodes
const firstChild = element.firstElementChild;
const lastChild = element.lastElementChild;

// Sibling relationships
const nextSibling = element.nextElementSibling;
const previousSibling = element.previousElementSibling;

// Finding elements
const descendant = element.querySelector('.descendant-class');
const allDescendants = element.querySelectorAll('p');

// Checking relationships
const isChild = parent.contains(element); // true
const closestAncestor = element.closest('.ancestor-class');
```

### Form Handling:
```javascript
const form = document.getElementById('myForm');
const nameInput = document.getElementById('name');
const emailInput = document.getElementById('email');
const submitBtn = document.getElementById('submit');

// Get form values
const nameValue = nameInput.value;
const emailValue = emailInput.value;

// Set form values
nameInput.value = 'John Doe';
emailInput.value = 'john@example.com';

// Form validation
function validateForm() {
  if (nameInput.value.trim() === '') {
    alert('Name is required');
    return false;
  }
  
  if (!emailInput.value.includes('@')) {
    alert('Valid email is required');
    return false;
  }
  
  return true;
}

// Form data handling
const formData = new FormData(form);
const data = Object.fromEntries(formData.entries());

// Serialize form to URL-encoded string
const params = new URLSearchParams(formData);
const serialized = params.toString();
```

## Events

### Event Listeners:
```javascript
const button = document.getElementById('myButton');

// Add event listener
button.addEventListener('click', function(event) {
  console.log('Button clicked!');
  console.log('Event type:', event.type);
  console.log('Target element:', event.target);
});

// Arrow function event listener
button.addEventListener('click', (event) => {
  console.log('Clicked with arrow function');
});

// Named function
function handleClick(event) {
  console.log('Named function handler');
}

button.addEventListener('click', handleClick);

// Remove event listener
button.removeEventListener('click', handleClick);
```

### Event Object:
```javascript
document.addEventListener('click', function(event) {
  console.log('Event type:', event.type); // 'click'
  console.log('Target:', event.target); // Element that triggered event
  console.log('Current target:', event.currentTarget); // Element with event listener
  console.log('Mouse position:', event.clientX, event.clientY);
  console.log('Timestamp:', event.timeStamp);
  
  // Prevent default behavior
  event.preventDefault();
  
  // Stop event from bubbling up
  event.stopPropagation();
  
  // Stop other listeners on same element
  event.stopImmediatePropagation();
});
```

### Common Events:

#### Mouse Events:
```javascript
const element = document.getElementById('myElement');

element.addEventListener('click', (e) => console.log('Clicked'));
element.addEventListener('dblclick', (e) => console.log('Double clicked'));
element.addEventListener('mousedown', (e) => console.log('Mouse down'));
element.addEventListener('mouseup', (e) => console.log('Mouse up'));
element.addEventListener('mouseover', (e) => console.log('Mouse over'));
element.addEventListener('mouseout', (e) => console.log('Mouse out'));
element.addEventListener('mouseenter', (e) => console.log('Mouse enter'));
element.addEventListener('mouseleave', (e) => console.log('Mouse leave'));
element.addEventListener('mousemove', (e) => {
  console.log(`Mouse position: ${e.clientX}, ${e.clientY}`);
});
```

#### Keyboard Events:
```javascript
const input = document.getElementById('textInput');

input.addEventListener('keydown', (e) => {
  console.log('Key down:', e.key);
  console.log('Key code:', e.keyCode);
  console.log('Ctrl pressed:', e.ctrlKey);
  console.log('Shift pressed:', e.shiftKey);
  console.log('Alt pressed:', e.altKey);
});

input.addEventListener('keyup', (e) => {
  console.log('Key up:', e.key);
});

input.addEventListener('keypress', (e) => {
  console.log('Key press:', e.key);
});

// Specific key handling
document.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    console.log('Enter key pressed');
  }
  
  if (e.ctrlKey && e.key === 's') {
    e.preventDefault(); // Prevent browser save dialog
    console.log('Ctrl+S pressed');
  }
});
```

#### Form Events:
```javascript
const form = document.getElementById('myForm');
const input = document.getElementById('myInput');

form.addEventListener('submit', (e) => {
  e.preventDefault(); // Prevent form submission
  console.log('Form submitted');
  
  // Handle form submission manually
  const formData = new FormData(form);
  console.log('Form data:', Object.fromEntries(formData));
});

input.addEventListener('focus', (e) => {
  console.log('Input focused');
  e.target.style.backgroundColor = 'lightblue';
});

input.addEventListener('blur', (e) => {
  console.log('Input lost focus');
  e.target.style.backgroundColor = '';
});

input.addEventListener('input', (e) => {
  console.log('Input value changed:', e.target.value);
});

input.addEventListener('change', (e) => {
  console.log('Input change event:', e.target.value);
});
```

### Event Delegation:
```javascript
// Instead of adding event listeners to each item
const container = document.getElementById('container');

container.addEventListener('click', (e) => {
  // Check if clicked element has specific class
  if (e.target.classList.contains('delete-btn')) {
    const item = e.target.closest('.item');
    item.remove();
  }
  
  if (e.target.classList.contains('edit-btn')) {
    const item = e.target.closest('.item');
    const text = item.querySelector('.text');
    const newText = prompt('Edit text:', text.textContent);
    if (newText !== null) {
      text.textContent = newText;
    }
  }
});

// This works for dynamically added elements too
function addNewItem(text) {
  const item = document.createElement('div');
  item.className = 'item';
  item.innerHTML = `
    <span class="text">${text}</span>
    <button class="edit-btn">Edit</button>
    <button class="delete-btn">Delete</button>
  `;
  container.appendChild(item);
}
```

### Custom Events:
```javascript
// Create custom event
const customEvent = new CustomEvent('myCustomEvent', {
  detail: {
    message: 'Hello from custom event',
    timestamp: Date.now()
  },
  bubbles: true,
  cancelable: true
});

// Listen for custom event
document.addEventListener('myCustomEvent', (e) => {
  console.log('Custom event received:', e.detail);
});

// Dispatch custom event
document.dispatchEvent(customEvent);

// More complex custom event
class EventEmitter {
  constructor() {
    this.events = {};
  }
  
  on(event, callback) {
    if (!this.events[event]) {
      this.events[event] = [];
    }
    this.events[event].push(callback);
  }
  
  emit(event, data) {
    if (this.events[event]) {
      this.events[event].forEach(callback => callback(data));
    }
  }
  
  off(event, callback) {
    if (this.events[event]) {
      this.events[event] = this.events[event].filter(cb => cb !== callback);
    }
  }
}

const emitter = new EventEmitter();
emitter.on('userLogin', (user) => console.log('User logged in:', user));
emitter.emit('userLogin', { name: 'John', id: 123 });
```

## Error Handling

### try...catch...finally:
```javascript
// Basic try-catch
try {
  let result = riskyOperation();
  console.log(result);
} catch (error) {
  console.error('An error occurred:', error.message);
} finally {
  console.log('This always executes');
}

// Catching specific error types
try {
  JSON.parse('invalid json');
} catch (error) {
  if (error instanceof SyntaxError) {
    console.log('JSON syntax error');
  } else if (error instanceof ReferenceError) {
    console.log('Reference error');
  } else {
    console.log('Unknown error:', error);
  }
}

// Nested try-catch
try {
  try {
    throw new Error('Inner error');
  } catch (innerError) {
    console.log('Caught inner error:', innerError.message);
    throw new Error('Outer error');
  }
} catch (outerError) {
  console.log('Caught outer error:', outerError.message);
}
```

### Creating Custom Errors:
```javascript
// Custom error class
class ValidationError extends Error {
  constructor(message, field) {
    super(message);
    this.name = 'ValidationError';
    this.field = field;
  }
}

class NetworkError extends Error {
  constructor(message, statusCode) {
    super(message);
    this.name = 'NetworkError';
    this.statusCode = statusCode;
  }
}

// Using custom errors
function validateUser(user) {
  if (!user.name) {
    throw new ValidationError('Name is required', 'name');
  }
  
  if (!user.email || !user.email.includes('@')) {
    throw new ValidationError('Valid email is required', 'email');
  }
}

try {
  validateUser({ name: '', email: 'invalid' });
} catch (error) {
  if (error instanceof ValidationError) {
    console.log(`Validation error in ${error.field}: ${error.message}`);
  } else {
    console.log('Unexpected error:', error);
  }
}
```

### Error Handling Best Practices:
```javascript
// Function that might throw errors
function safeDivide(a, b) {
  if (typeof a !== 'number' || typeof b !== 'number') {
    throw new TypeError('Both arguments must be numbers');
  }
  
  if (b === 0) {
    throw new Error('Cannot divide by zero');
  }
  
  return a / b;
}

// Error handling with specific messages
function handleOperation(operation, ...args) {
  try {
    const result = operation(...args);
    return { success: true, result };
  } catch (error) {
    return {
      success: false,
      error: {
        message: error.message,
        type: error.constructor.name,
        stack: error.stack
      }
    };
  }
}

const result = handleOperation(safeDivide, 10, 2);
if (result.success) {
  console.log('Result:', result.result);
} else {
  console.error('Operation failed:', result.error);
}

// Global error handling
window.addEventListener('error', (event) => {
  console.error('Global error:', event.error);
  // Send error to logging service
});

// Promise rejection handling
window.addEventListener('unhandledrejection', (event) => {
  console.error('Unhandled promise rejection:', event.reason);
  event.preventDefault(); // Prevent browser default handling
});
```

## Regular Expressions

### Creating Regular Expressions:
```javascript
// Literal syntax
const regex1 = /pattern/flags;
const regex2 = /hello/i; // Case insensitive

// Constructor syntax
const regex3 = new RegExp('pattern', 'flags');
const regex4 = new RegExp('hello', 'i');

// Dynamic patterns
const word = 'hello';
const dynamicRegex = new RegExp(word, 'gi');
```

### Common Flags:
```javascript
const text = 'Hello World Hello';

// g - global (find all matches)
console.log(text.match(/hello/g)); // null (case sensitive)
console.log(text.match(/hello/gi)); // ['Hello', 'Hello']

// i - case insensitive
console.log(text.match(/hello/i)); // ['Hello']

// m - multiline (^ and $ match line boundaries)
const multilineText = 'first line\nsecond line';
console.log(multilineText.match(/^second/m)); // ['second']

// s - dotall (. matches newlines)
console.log('hello\nworld'.match(/hello.world/s)); // ['hello\nworld']
```

### Basic Patterns:
```javascript
// Literal characters
/hello/.test('hello world'); // true

// Character classes
/[abc]/.test('apple'); // true (matches 'a')
/[a-z]/.test('Hello'); // true (matches lowercase letters)
/[A-Z]/.test('hello'); // false
/[0-9]/.test('abc123'); // true (matches digits)

// Negated character classes
/[^0-9]/.test('123'); // false (no non-digits)
/[^0-9]/.test('abc'); // true (has non-digits)

// Predefined character classes
/\d/.test('123'); // true (digits: [0-9])
/\w/.test('hello'); // true (word characters: [a-zA-Z0-9_])
/\s/.test('hello world'); // true (whitespace)

// Negated predefined classes
/\D/.test('abc'); // true (non-digits)
/\W/.test('hello!'); // true (non-word characters)
/\S/.test('hello'); // true (non-whitespace)

// Dot (any character except newline)
/./.test('a'); // true
/./.test('\n'); // false (unless 's' flag is used)
```

### Quantifiers:
```javascript
// Exact count
/a{3}/.test('aaa'); // true
/a{3}/.test('aa'); // false

// Range
/a{2,4}/.test('aa'); // true
/a{2,4}/.test('aaaaa'); // true (matches first 4)

// Minimum
/a{2,}/.test('aa'); // true
/a{2,}/.test('a'); // false

// Common quantifiers
/a*/.test(''); // true (0 or more)
/a+/.test('a'); // true (1 or more)
/a?/.test(''); // true (0 or 1)

// Greedy vs non-greedy
const html = '<div>content</div>';
console.log(html.match(/<.*>/)); // ['<div>content</div>'] (greedy)
console.log(html.match(/<.*?>/)); // ['<div>'] (non-greedy)
```

### Anchors and Boundaries:
```javascript
// Start and end of string
/^hello/.test('hello world'); // true
/world$/.test('hello world'); // true
/^hello world$/.test('hello world'); // true (exact match)

// Word boundaries
/\bword\b/.test('a word here'); // true
/\bword\b/.test('sword'); // false

// Examples
const email = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;
console.log(email.test('user@example.com')); // true

const phone = /^\(\d{3}\) \d{3}-\d{4}$/;
console.log(phone.test('(123) 456-7890')); // true
```

### Groups and Capturing:
```javascript
// Grouping
const regex = /(hello|hi) (world|there)/;
console.log(regex.test('hello world')); // true
console.log(regex.test('hi there')); // true

// Capturing groups
const nameRegex = /(\w+) (\w+)/;
const match = 'John Doe'.match(nameRegex);
console.log(match[0]); // 'John Doe' (full match)
console.log(match[1]); // 'John' (first group)
console.log(match[2]); // 'Doe' (second group)

// Named capturing groups
const namedRegex = /(?<first>\w+) (?<last>\w+)/;
const namedMatch = 'John Doe'.match(namedRegex);
console.log(namedMatch.groups.first); // 'John'
console.log(namedMatch.groups.last); // 'Doe'

// Non-capturing groups
const nonCapturing = /(?:hello|hi) world/;
console.log('hello world'.match(nonCapturing)[0]); // 'hello world'
// No additional groups captured
```

### String Methods with RegExp:
```javascript
const text = 'The quick brown fox jumps over the lazy dog';

// match() - returns array of matches
console.log(text.match(/\b\w{4}\b/g)); // ['quick', 'brown', 'jumps', 'over', 'lazy']

// search() - returns index of first match
console.log(text.search(/fox/)); // 16

// replace() - replaces matches
console.log(text.replace(/\b\w{3}\b/g, 'XXX')); // 'XXX quick brown XXX jumps over XXX lazy XXX'

// Using replacement function
console.log(text.replace(/\b\w+\b/g, (match) => match.toUpperCase()));

// split() - splits string by regex
console.log('one,two;three:four'.split(/[,:;]/)); // ['one', 'two', 'three', 'four']
```

### Practical Examples:
```javascript
// Email validation
const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
function isValidEmail(email) {
  return emailRegex.test(email);
}

// Password validation (at least 8 chars, 1 uppercase, 1 lowercase, 1 digit)
const passwordRegex = /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)[a-zA-Z\d@$!%*?&]{8,}$/;

// URL validation
const urlRegex = /^https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)$/;

// Extract all hashtags from text
function extractHashtags(text) {
  const hashtagRegex = /#\w+/g;
  return text.match(hashtagRegex) || [];
}

// Format phone number
function formatPhoneNumber(phone) {
  const cleaned = phone.replace(/\D/g, '');
  const match = cleaned.match(/^(\d{3})(\d{3})(\d{4})$/);
  return match ? `(${match[1]}) ${match[2]}-${match[3]}` : null;
}

console.log(formatPhoneNumber('1234567890')); // '(123) 456-7890'
```

## JSON

### JSON Basics:
```javascript
// JSON is a string format for data exchange
const jsonString = '{"name": "John", "age": 30, "city": "New York"}';

// Parse JSON string to JavaScript object
const obj = JSON.parse(jsonString);
console.log(obj.name); // "John"

// Convert JavaScript object to JSON string
const person = { name: "Alice", age: 25, hobbies: ["reading", "coding"] };
const json = JSON.stringify(person);
console.log(json); // '{"name":"Alice","age":25,"hobbies":["reading","coding"]}'
```

### JSON.stringify() Options:
```javascript
const data = {
  name: "John",
  age: 30,
  password: "secret123",
  friends: ["Alice", "Bob"],
  address: {
    street: "123 Main St",
    city: "New York"
  }
};

// Basic stringification
console.log(JSON.stringify(data));

// With replacer function (filter/transform properties)
const filtered = JSON.stringify(data, (key, value) => {
  if (key === 'password') return undefined; // Exclude password
  if (typeof value === 'string') return value.toUpperCase();
  return value;
});

// With replacer array (only include specific properties)
const limited = JSON.stringify(data, ['name', 'age']);

// With space parameter for formatting
const formatted = JSON.stringify(data, null, 2); // 2 spaces indentation
console.log(formatted);

// Custom toJSON method
const customObject = {
  name: "John",
  age: 30,
  toJSON() {
    return {
      displayName: this.name,
      isAdult: this.age >= 18
    };
  }
};

console.log(JSON.stringify(customObject)); // '{"displayName":"John","isAdult":true}'
```

### JSON.parse() with Reviver:
```javascript
const jsonString = '{"name":"John","birthDate":"2023-01-15","score":"95"}';

// Parse with reviver function
const obj = JSON.parse(jsonString, (key, value) => {
  // Convert date strings to Date objects
  if (key === 'birthDate') return new Date(value);
  
  // Convert numeric strings to numbers
  if (key === 'score') return parseInt(value, 10);
  
  return value;
});

console.log(obj.birthDate instanceof Date); // true
console.log(typeof obj.score); // "number"
```

### Error Handling with JSON:
```javascript
function safeJsonParse(jsonString, defaultValue = null) {
  try {
    return JSON.parse(jsonString);
  } catch (error) {
    console.error('JSON parse error:', error.message);
    return defaultValue;
  }
}

function safeJsonStringify(obj, defaultValue = '{}') {
  try {
    return JSON.stringify(obj);
  } catch (error) {
    console.error('JSON stringify error:', error.message);
    return defaultValue;
  }
}

// Handle circular references
const circularObj = { name: "John" };
circularObj.self = circularObj; // Creates circular reference

// This would throw an error:
// JSON.stringify(circularObj); // TypeError: Converting circular structure to JSON

// Solution with replacer
const safeStringify = JSON.stringify(circularObj, (key, value) => {
  if (key === 'self') return '[Circular Reference]';
  return value;
});
```

### Working with Complex Data:
```javascript
// Deep clone using JSON (with limitations)
function deepClone(obj) {
  return JSON.parse(JSON.stringify(obj));
}

// Note: This method has limitations:
// - Functions are lost
// - Dates become strings
// - undefined values are lost
// - Symbols are lost

const original = {
  name: "John",
  date: new Date(),
  fn: () => console.log("Hello"),
  undef: undefined
};

const cloned = deepClone(original);
console.log(cloned); // { name: "John", date: "2023-..." }

// Better deep clone for complex objects
function betterDeepClone(obj) {
  if (obj === null || typeof obj !== 'object') return obj;
  if (obj instanceof Date) return new Date(obj);
  if (obj instanceof Array) return obj.map(item => betterDeepClone(item));
  
  const cloned = {};
  for (const key in obj) {
    if (obj.hasOwnProperty(key)) {
      cloned[key] = betterDeepClone(obj[key]);
    }
  }
  return cloned;
}
```

## Local Storage

### Basic Storage Operations:
```javascript
// Store data (only strings)
localStorage.setItem('username', 'john_doe');
localStorage.setItem('isLoggedIn', 'true');

// Retrieve data
const username = localStorage.getItem('username');
const isLoggedIn = localStorage.getItem('isLoggedIn');

console.log(username); // "john_doe"
console.log(typeof isLoggedIn); // "string" (not boolean!)

// Remove specific item
localStorage.removeItem('username');

// Clear all localStorage
localStorage.clear();

// Check if item exists
if (localStorage.getItem('username') !== null) {
  console.log('Username exists');
}
```

### Storing Complex Data:
```javascript
// Store objects and arrays (convert to JSON)
const user = {
  id: 123,
  name: 'John Doe',
  preferences: {
    theme: 'dark',
    language: 'en'
  },
  lastLogin: new Date()
};

// Store object
localStorage.setItem('user', JSON.stringify(user));

// Retrieve and parse object
const storedUser = JSON.parse(localStorage.getItem('user'));
console.log(storedUser.name); // "John Doe"

// Note: Dates become strings when stored in JSON
console.log(typeof storedUser.lastLogin); // "string"
```

### Storage Helper Functions:
```javascript
// Generic storage utilities
const storage = {
  set(key, value) {
    try {
      localStorage.setItem(key, JSON.stringify(value));
      return true;
    } catch (error) {
      console.error('Error storing data:', error);
      return false;
    }
  },
  
  get(key, defaultValue = null) {
    try {
      const item = localStorage.getItem(key);
      return item ? JSON.parse(item) : defaultValue;
    } catch (error) {
      console.error('Error retrieving data:', error);
      return defaultValue;
    }
  },
  
  remove(key) {
    localStorage.removeItem(key);
  },
  
  clear() {
    localStorage.clear();
  },
  
  has(key) {
    return localStorage.getItem(key) !== null;
  },
  
  size() {
    return localStorage.length;
  },
  
  keys() {
    return Object.keys(localStorage);
  }
};

// Usage
storage.set('userPreferences', { theme: 'dark', notifications: true });
const prefs = storage.get('userPreferences', { theme: 'light' });
```

### Session Storage:
```javascript
// sessionStorage works the same as localStorage
// but data is only available for the session (tab)

sessionStorage.setItem('temporaryData', 'This will be gone when tab closes');
const tempData = sessionStorage.getItem('temporaryData');

// All localStorage methods work with sessionStorage
const sessionUtils = {
  set: (key, value) => sessionStorage.setItem(key, JSON.stringify(value)),
  get: (key, defaultValue = null) => {
    const item = sessionStorage.getItem(key);
    return item ? JSON.parse(item) : defaultValue;
  }
};
```

### Storage Events:
```javascript
// Listen for storage changes (from other tabs/windows)
window.addEventListener('storage', (event) => {
  console.log('Storage changed:');
  console.log('Key:', event.key);
  console.log('Old value:', event.oldValue);
  console.log('New value:', event.newValue);
  console.log('Storage area:', event.storageArea);
  console.log('URL:', event.url);
});

// This event only fires when storage is changed from another tab
// Changes in the same tab don't trigger the event
```

### Practical Examples:
```javascript
// User preferences manager
class UserPreferences {
  constructor() {
    this.key = 'userPrefs';
    this.defaults = {
      theme: 'light',
      language: 'en',
      notifications: true,
      autoSave: false
    };
  }
  
  load() {
    const stored = localStorage.getItem(this.key);
    return stored ? { ...this.defaults, ...JSON.parse(stored) } : this.defaults;
  }
  
  save(preferences) {
    localStorage.setItem(this.key, JSON.stringify(preferences));
  }
  
  get(key) {
    const prefs = this.load();
    return prefs[key];
  }
  
  set(key, value) {
    const prefs = this.load();
    prefs[key] = value;
    this.save(prefs);
  }
  
  reset() {
    localStorage.removeItem(this.key);
  }
}

const userPrefs = new UserPreferences();

// Usage
userPrefs.set('theme', 'dark');
console.log(userPrefs.get('theme')); // "dark"

// Shopping cart manager
class ShoppingCart {
  constructor() {
    this.key = 'shoppingCart';
  }
  
  getItems() {
    return storage.get(this.key, []);
  }
  
  addItem(item) {
    const items = this.getItems();
    const existingIndex = items.findIndex(i => i.id === item.id);
    
    if (existingIndex > -1) {
      items[existingIndex].quantity += item.quantity || 1;
    } else {
      items.push({ ...item, quantity: item.quantity || 1 });
    }
    
    storage.set(this.key, items);
  }
  
  removeItem(itemId) {
    const items = this.getItems().filter(item => item.id !== itemId);
    storage.set(this.key, items);
  }
  
  clear() {
    storage.remove(this.key);
  }
  
  getTotal() {
    return this.getItems().reduce((total, item) => total + (item.price * item.quantity), 0);
  }
}

// Form data persistence
function autoSaveForm(formId) {
  const form = document.getElementById(formId);
  const storageKey = `form_${formId}`;
  
  // Load saved data
  const savedData = storage.get(storageKey, {});
  Object.keys(savedData).forEach(name => {
    const field = form.querySelector(`[name="${name}"]`);
    if (field) field.value = savedData[name];
  });
  
  // Save on input
  form.addEventListener('input', (e) => {
    const formData = new FormData(form);
    const data = Object.fromEntries(formData.entries());
    storage.set(storageKey, data);
  });
  
  // Clear on submit
  form.addEventListener('submit', () => {
    storage.remove(storageKey);
  });
}
```

## Modules

### ES6 Modules (import/export):

#### Named Exports:
```javascript
// math.js
export function add(a, b) {
  return a + b;
}

export function subtract(a, b) {
  return a - b;
}

export const PI = 3.14159;

// Alternative syntax
function multiply(a, b) {
  return a * b;
}

function divide(a, b) {
  return a / b;
}

export { multiply, divide };

// Exporting with different names
function power(base, exponent) {
  return Math.pow(base, exponent);
}

export { power as pow };
```

#### Default Exports:
```javascript
// calculator.js
class Calculator {
  add(a, b) { return a + b; }
  subtract(a, b) { return a - b; }
}

export default Calculator;

// Or inline default export
export default class Calculator {
  add(a, b) { return a + b; }
  subtract(a, b) { return a - b; }
}

// Default export for functions
export default function fibonacci(n) {
  if (n <= 1) return n;
  return fibonacci(n - 1) + fibonacci(n - 2);
}
```

#### Importing:
```javascript
// main.js

// Named imports
import { add, subtract, PI } from './math.js';
console.log(add(5, 3)); // 8

// Import with different names
import { pow as power } from './math.js';
console.log(power(2, 3)); // 8

// Import all named exports
import * as math from './math.js';
console.log(math.add(5, 3)); // 8
console.log(math.PI); // 3.14159

// Default import
import Calculator from './calculator.js';
const calc = new Calculator();

// Mixed imports
import Calculator, { add, subtract } from './calculator.js';

// Dynamic imports (returns a Promise)
async function loadMath() {
  const math = await import('./math.js');
  console.log(math.add(5, 3));
}

// Conditional dynamic import
if (condition) {
  import('./heavy-module.js').then(module => {
    module.heavyFunction();
  });
}
```

### Module Patterns:

#### Revealing Module Pattern:
```javascript
const MyModule = (function() {
  // Private variables and functions
  let privateVar = 0;
  
  function privateFunction() {
    console.log('This is private');
  }
  
  function increment() {
    privateVar++;
    privateFunction();
  }
  
  function getCount() {
    return privateVar;
  }
  
  // Public API
  return {
    increment,
    getCount,
    reset() {
      privateVar = 0;
    }
  };
})();

MyModule.increment();
console.log(MyModule.getCount()); // 1
```

#### Module with Dependencies:
```javascript
// userService.js
import { apiCall } from './api.js';
import { validateUser } from './validation.js';

class UserService {
  async createUser(userData) {
    if (!validateUser(userData)) {
      throw new Error('Invalid user data');
    }
    
    return await apiCall('/users', {
      method: 'POST',
      body: JSON.stringify(userData)
    });
  }
  
  async getUser(id) {
    return await apiCall(`/users/${id}`);
  }
}

export default new UserService(); // Export singleton instance

// Or export the class for multiple instances
export { UserService };
```

#### Barrel Exports:
```javascript
// components/index.js (barrel file)
export { default as Header } from './Header.js';
export { default as Footer } from './Footer.js';
export { default as Sidebar } from './Sidebar.js';
export { Button, Input } from './Forms.js';

// Now you can import multiple components from one file
import { Header, Footer, Button } from './components/index.js';
```

### CommonJS (Node.js):
```javascript
// math.js (CommonJS)
function add(a, b) {
  return a + b;
}

function subtract(a, b) {
  return a - b;
}

module.exports = {
  add,
  subtract
};

// Or export individual functions
exports.multiply = (a, b) => a * b;
exports.divide = (a, b) => a / b;

// main.js
const { add, subtract } = require('./math');
const math = require('./math');

console.log(add(5, 3)); // 8
console.log(math.subtract(10, 4)); // 6
```

### Module Best Practices:
```javascript
// 1. One main export per module
// user.js
export default class User {
  constructor(name) {
    this.name = name;
  }
}

// 2. Use named exports for utilities
// utils.js
export const formatDate = (date) => {
  return date.toLocaleDateString();
};

export const capitalize = (str) => {
  return str.charAt(0).toUpperCase() + str.slice(1);
};

// 3. Keep modules focused and cohesive
// Don't mix unrelated functionality

// 4. Avoid circular dependencies
// If A imports B, B shouldn't import A

// 5. Use index files for clean imports
// components/
//   ├── Header.js
//   ├── Footer.js
//   └── index.js

// index.js
export { default as Header } from './Header.js';
export { default as Footer } from './Footer.js';

// Clean import
import { Header, Footer } from './components';
```

---

This covers the intermediate concepts of JavaScript including ES6+ features, advanced array methods, OOP concepts, DOM manipulation, events, error handling, regular expressions, JSON, local storage, and modules. These concepts build upon the basics and prepare you for advanced JavaScript development.
