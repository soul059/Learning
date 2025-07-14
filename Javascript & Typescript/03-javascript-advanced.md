# JavaScript Advanced Concepts

## Table of Contents
1. [Asynchronous JavaScript](#asynchronous-javascript)
2. [Closures and Lexical Scope](#closures-and-lexical-scope)
3. [Prototypes and Inheritance](#prototypes-and-inheritance)
4. [Advanced Functions](#advanced-functions)
5. [Memory Management](#memory-management)
6. [Performance Optimization](#performance-optimization)
7. [Design Patterns](#design-patterns)
8. [Metaprogramming](#metaprogramming)
9. [Web APIs](#web-apis)
10. [Testing](#testing)

## Asynchronous JavaScript

### Callbacks:
```javascript
// Basic callback pattern
function fetchData(callback) {
  setTimeout(() => {
    const data = { id: 1, name: "John" };
    callback(null, data);
  }, 1000);
}

fetchData((error, data) => {
  if (error) {
    console.error("Error:", error);
  } else {
    console.log("Data:", data);
  }
});

// Callback hell example
function getUserData(userId, callback) {
  fetchUser(userId, (err, user) => {
    if (err) return callback(err);
    
    fetchUserPosts(user.id, (err, posts) => {
      if (err) return callback(err);
      
      fetchPostComments(posts[0].id, (err, comments) => {
        if (err) return callback(err);
        
        callback(null, { user, posts, comments });
      });
    });
  });
}

// Error-first callback convention
function readFile(filename, callback) {
  // Simulate file reading
  setTimeout(() => {
    if (filename === 'nonexistent.txt') {
      callback(new Error('File not found'));
    } else {
      callback(null, 'File content');
    }
  }, 100);
}
```

### Promises:
```javascript
// Creating a Promise
const myPromise = new Promise((resolve, reject) => {
  const success = Math.random() > 0.5;
  
  setTimeout(() => {
    if (success) {
      resolve("Operation successful!");
    } else {
      reject(new Error("Operation failed!"));
    }
  }, 1000);
});

// Consuming Promises
myPromise
  .then(result => {
    console.log("Success:", result);
    return result.toUpperCase();
  })
  .then(upperResult => {
    console.log("Upper:", upperResult);
  })
  .catch(error => {
    console.error("Error:", error.message);
  })
  .finally(() => {
    console.log("Cleanup operations");
  });

// Promise chaining
function fetchUser(id) {
  return new Promise((resolve) => {
    setTimeout(() => resolve({ id, name: `User ${id}` }), 500);
  });
}

function fetchUserPosts(userId) {
  return new Promise((resolve) => {
    setTimeout(() => resolve([
      { id: 1, title: "Post 1", userId },
      { id: 2, title: "Post 2", userId }
    ]), 300);
  });
}

fetchUser(1)
  .then(user => {
    console.log("User:", user);
    return fetchUserPosts(user.id);
  })
  .then(posts => {
    console.log("Posts:", posts);
  })
  .catch(console.error);

// Promise.all - Wait for all promises
const promises = [
  fetchUser(1),
  fetchUser(2),
  fetchUser(3)
];

Promise.all(promises)
  .then(users => {
    console.log("All users:", users);
  })
  .catch(error => {
    console.error("One or more failed:", error);
  });

// Promise.allSettled - Wait for all, regardless of outcome
Promise.allSettled(promises)
  .then(results => {
    results.forEach((result, index) => {
      if (result.status === 'fulfilled') {
        console.log(`User ${index + 1}:`, result.value);
      } else {
        console.error(`User ${index + 1} failed:`, result.reason);
      }
    });
  });

// Promise.race - First to resolve/reject wins
Promise.race(promises)
  .then(firstUser => {
    console.log("First user:", firstUser);
  });

// Promise.any - First to resolve wins (ignores rejections)
Promise.any(promises)
  .then(firstSuccess => {
    console.log("First successful:", firstSuccess);
  })
  .catch(aggregateError => {
    console.error("All failed:", aggregateError.errors);
  });
```

### Async/Await:
```javascript
// Basic async/await
async function fetchUserData(userId) {
  try {
    const user = await fetchUser(userId);
    const posts = await fetchUserPosts(user.id);
    const comments = await fetchPostComments(posts[0].id);
    
    return { user, posts, comments };
  } catch (error) {
    console.error("Error fetching user data:", error);
    throw error;
  }
}

// Using async/await
async function main() {
  try {
    const userData = await fetchUserData(1);
    console.log("Complete user data:", userData);
  } catch (error) {
    console.error("Failed to get user data:", error);
  }
}

main();

// Parallel execution with async/await
async function fetchMultipleUsers() {
  try {
    // Sequential (slower)
    const user1 = await fetchUser(1);
    const user2 = await fetchUser(2);
    const user3 = await fetchUser(3);
    
    // Parallel (faster)
    const [pUser1, pUser2, pUser3] = await Promise.all([
      fetchUser(1),
      fetchUser(2),
      fetchUser(3)
    ]);
    
    return { sequential: [user1, user2, user3], parallel: [pUser1, pUser2, pUser3] };
  } catch (error) {
    console.error("Error:", error);
  }
}

// Error handling patterns
async function robustDataFetch(userId) {
  let attempts = 0;
  const maxAttempts = 3;
  
  while (attempts < maxAttempts) {
    try {
      const data = await fetchUserData(userId);
      return data;
    } catch (error) {
      attempts++;
      console.log(`Attempt ${attempts} failed:`, error.message);
      
      if (attempts >= maxAttempts) {
        throw new Error(`Failed after ${maxAttempts} attempts: ${error.message}`);
      }
      
      // Wait before retrying
      await new Promise(resolve => setTimeout(resolve, 1000 * attempts));
    }
  }
}

// Async iteration
async function* asyncGenerator() {
  for (let i = 1; i <= 5; i++) {
    await new Promise(resolve => setTimeout(resolve, 1000));
    yield i;
  }
}

async function processAsyncIterable() {
  for await (const value of asyncGenerator()) {
    console.log("Generated:", value);
  }
}
```

### Event Loop and Microtasks:
```javascript
// Understanding the event loop
console.log("1 - Start");

setTimeout(() => console.log("2 - setTimeout"), 0);

Promise.resolve().then(() => console.log("3 - Promise"));

console.log("4 - End");

// Output: 1, 4, 3, 2
// Microtasks (Promises) have higher priority than macrotasks (setTimeout)

// Microtask vs Macrotask demonstration
console.log("Script start");

setTimeout(() => console.log("setTimeout 1"), 0);
setTimeout(() => console.log("setTimeout 2"), 0);

Promise.resolve()
  .then(() => console.log("Promise 1"))
  .then(() => console.log("Promise 2"));

console.log("Script end");

// Output: Script start, Script end, Promise 1, Promise 2, setTimeout 1, setTimeout 2

// Advanced event loop example
function demonstrateEventLoop() {
  console.log("=== Event Loop Demo ===");
  
  // Immediate
  console.log("1 - Synchronous");
  
  // Macrotask
  setTimeout(() => console.log("2 - setTimeout 0ms"), 0);
  
  // Microtask
  Promise.resolve().then(() => {
    console.log("3 - Promise.resolve");
    return Promise.resolve();
  }).then(() => {
    console.log("4 - Chained Promise");
  });
  
  // Immediate
  console.log("5 - More synchronous");
  
  // Microtask
  queueMicrotask(() => console.log("6 - queueMicrotask"));
  
  // Another macrotask
  setTimeout(() => console.log("7 - setTimeout 0ms #2"), 0);
}

demonstrateEventLoop();
```

### Advanced Promise Patterns:
```javascript
// Custom Promise implementation (simplified)
class MyPromise {
  constructor(executor) {
    this.state = 'pending';
    this.value = undefined;
    this.handlers = [];
    
    const resolve = (value) => {
      if (this.state === 'pending') {
        this.state = 'fulfilled';
        this.value = value;
        this.handlers.forEach(handler => handler.onFulfilled(value));
      }
    };
    
    const reject = (reason) => {
      if (this.state === 'pending') {
        this.state = 'rejected';
        this.value = reason;
        this.handlers.forEach(handler => handler.onRejected(reason));
      }
    };
    
    try {
      executor(resolve, reject);
    } catch (error) {
      reject(error);
    }
  }
  
  then(onFulfilled, onRejected) {
    return new MyPromise((resolve, reject) => {
      const handle = () => {
        if (this.state === 'fulfilled') {
          if (onFulfilled) {
            try {
              resolve(onFulfilled(this.value));
            } catch (error) {
              reject(error);
            }
          } else {
            resolve(this.value);
          }
        } else if (this.state === 'rejected') {
          if (onRejected) {
            try {
              resolve(onRejected(this.value));
            } catch (error) {
              reject(error);
            }
          } else {
            reject(this.value);
          }
        } else {
          this.handlers.push({
            onFulfilled: (value) => {
              if (onFulfilled) {
                try {
                  resolve(onFulfilled(value));
                } catch (error) {
                  reject(error);
                }
              } else {
                resolve(value);
              }
            },
            onRejected: (reason) => {
              if (onRejected) {
                try {
                  resolve(onRejected(reason));
                } catch (error) {
                  reject(error);
                }
              } else {
                reject(reason);
              }
            }
          });
        }
      };
      
      handle();
    });
  }
}

// Promise utilities
const promiseUtils = {
  // Delay utility
  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  },
  
  // Timeout wrapper
  timeout(promise, ms) {
    return Promise.race([
      promise,
      new Promise((_, reject) => 
        setTimeout(() => reject(new Error('Timeout')), ms)
      )
    ]);
  },
  
  // Retry with exponential backoff
  async retry(fn, maxAttempts = 3, baseDelay = 1000) {
    let lastError;
    
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      try {
        return await fn();
      } catch (error) {
        lastError = error;
        
        if (attempt === maxAttempts) {
          throw lastError;
        }
        
        const delay = baseDelay * Math.pow(2, attempt - 1);
        await this.delay(delay);
      }
    }
  },
  
  // Batch processing
  async batchProcess(items, processor, batchSize = 5) {
    const results = [];
    
    for (let i = 0; i < items.length; i += batchSize) {
      const batch = items.slice(i, i + batchSize);
      const batchResults = await Promise.all(
        batch.map(item => processor(item))
      );
      results.push(...batchResults);
    }
    
    return results;
  }
};

// Usage examples
async function demonstrateUtils() {
  // Delay
  console.log("Starting delay...");
  await promiseUtils.delay(2000);
  console.log("Delay completed");
  
  // Timeout
  try {
    await promiseUtils.timeout(
      promiseUtils.delay(3000),
      2000
    );
  } catch (error) {
    console.log("Operation timed out");
  }
  
  // Retry
  let attempts = 0;
  const unreliableOperation = async () => {
    attempts++;
    if (attempts < 3) {
      throw new Error(`Attempt ${attempts} failed`);
    }
    return "Success!";
  };
  
  try {
    const result = await promiseUtils.retry(unreliableOperation);
    console.log("Retry result:", result);
  } catch (error) {
    console.error("All retry attempts failed");
  }
}
```

## Closures and Lexical Scope

### Understanding Closures:
```javascript
// Basic closure
function outerFunction(x) {
  // Outer variable
  const outerVariable = x;
  
  // Inner function has access to outer variables
  function innerFunction(y) {
    return outerVariable + y;
  }
  
  return innerFunction;
}

const addFive = outerFunction(5);
console.log(addFive(3)); // 8

// The inner function "closes over" the outer variable
console.log(addFive.toString()); // Shows the function still references outerVariable

// Practical closure example: Counter
function createCounter() {
  let count = 0;
  
  return {
    increment() {
      count++;
      return count;
    },
    decrement() {
      count--;
      return count;
    },
    getValue() {
      return count;
    }
  };
}

const counter1 = createCounter();
const counter2 = createCounter();

console.log(counter1.increment()); // 1
console.log(counter1.increment()); // 2
console.log(counter2.increment()); // 1 (separate closure)
console.log(counter1.getValue()); // 2
```

### Lexical Scope:
```javascript
// Lexical scope demonstration
const globalVar = "I'm global";

function outerScope() {
  const outerVar = "I'm outer";
  
  function middleScope() {
    const middleVar = "I'm middle";
    
    function innerScope() {
      const innerVar = "I'm inner";
      
      // Inner scope has access to all outer scopes
      console.log(innerVar);  // ✓ Works
      console.log(middleVar); // ✓ Works
      console.log(outerVar);  // ✓ Works
      console.log(globalVar); // ✓ Works
    }
    
    innerScope();
    // console.log(innerVar); // ✗ ReferenceError
  }
  
  middleScope();
}

outerScope();

// Scope chain example
function a() {
  const varA = 'A';
  
  function b() {
    const varB = 'B';
    
    function c() {
      const varC = 'C';
      console.log(varA, varB, varC); // A B C
    }
    
    c();
  }
  
  b();
}

a();
```

### Common Closure Patterns:

#### Module Pattern:
```javascript
const Module = (function() {
  // Private variables and functions
  let privateVar = 0;
  const privateArray = [];
  
  function privateFunction() {
    console.log('This is private');
  }
  
  // Public API
  return {
    publicMethod() {
      privateVar++;
      privateFunction();
      return privateVar;
    },
    
    addToArray(item) {
      privateArray.push(item);
    },
    
    getArray() {
      return [...privateArray]; // Return copy to maintain privacy
    },
    
    reset() {
      privateVar = 0;
      privateArray.length = 0;
    }
  };
})();

Module.publicMethod(); // 1
Module.addToArray('item1');
console.log(Module.getArray()); // ['item1']
// Module.privateVar; // undefined (not accessible)
```

#### Factory Function:
```javascript
function createPerson(name, age) {
  // Private variables
  let _name = name;
  let _age = age;
  let _secrets = [];
  
  // Private methods
  function validateAge(newAge) {
    return newAge >= 0 && newAge <= 150;
  }
  
  // Public interface
  return {
    // Getters
    getName() {
      return _name;
    },
    
    getAge() {
      return _age;
    },
    
    // Setters with validation
    setName(newName) {
      if (typeof newName === 'string' && newName.length > 0) {
        _name = newName;
      }
    },
    
    setAge(newAge) {
      if (validateAge(newAge)) {
        _age = newAge;
      }
    },
    
    // Methods
    introduce() {
      return `Hi, I'm ${_name} and I'm ${_age} years old`;
    },
    
    addSecret(secret) {
      _secrets.push(secret);
    },
    
    getSecretCount() {
      return _secrets.length;
    }
  };
}

const person = createPerson("Alice", 30);
console.log(person.introduce()); // "Hi, I'm Alice and I'm 30 years old"
person.setAge(31);
console.log(person.getAge()); // 31
// person._age; // undefined (private)
```

#### Memoization with Closures:
```javascript
function memoize(fn) {
  const cache = new Map();
  
  return function(...args) {
    const key = JSON.stringify(args);
    
    if (cache.has(key)) {
      console.log('Cache hit for:', key);
      return cache.get(key);
    }
    
    console.log('Computing for:', key);
    const result = fn.apply(this, args);
    cache.set(key, result);
    return result;
  };
}

// Expensive function to memoize
function fibonacci(n) {
  if (n <= 1) return n;
  return fibonacci(n - 1) + fibonacci(n - 2);
}

const memoizedFib = memoize(fibonacci);

console.log(memoizedFib(10)); // Computes
console.log(memoizedFib(10)); // Cache hit
console.log(memoizedFib(11)); // Computes (but uses cached value for fib(10))
```

### Closure Gotchas:

#### Loop Closure Problem:
```javascript
// Problem: All functions reference the same variable
console.log("=== Problem ===");
var functions = [];

for (var i = 0; i < 3; i++) {
  functions[i] = function() {
    return i; // References the same 'i'
  };
}

functions.forEach((fn, index) => {
  console.log(`Function ${index} returns:`, fn()); // All return 3
});

// Solution 1: IIFE (Immediately Invoked Function Expression)
console.log("=== Solution 1: IIFE ===");
var functions1 = [];

for (var i = 0; i < 3; i++) {
  functions1[i] = (function(j) {
    return function() {
      return j; // Each function has its own 'j'
    };
  })(i);
}

functions1.forEach((fn, index) => {
  console.log(`Function ${index} returns:`, fn()); // 0, 1, 2
});

// Solution 2: let (block scope)
console.log("=== Solution 2: let ===");
var functions2 = [];

for (let i = 0; i < 3; i++) {
  functions2[i] = function() {
    return i; // Each iteration has its own 'i'
  };
}

functions2.forEach((fn, index) => {
  console.log(`Function ${index} returns:`, fn()); // 0, 1, 2
});

// Solution 3: bind
console.log("=== Solution 3: bind ===");
var functions3 = [];

for (var i = 0; i < 3; i++) {
  functions3[i] = function(j) {
    return j;
  }.bind(null, i);
}

functions3.forEach((fn, index) => {
  console.log(`Function ${index} returns:`, fn()); // 0, 1, 2
});
```

#### Memory Leaks with Closures:
```javascript
// Potential memory leak
function attachListeners() {
  const veryLargeArray = new Array(1000000).fill('data');
  
  document.getElementById('button').addEventListener('click', function() {
    // This closure keeps veryLargeArray in memory
    console.log('Button clicked');
  });
}

// Better approach
function attachListenersBetter() {
  const veryLargeArray = new Array(1000000).fill('data');
  
  // Extract needed data
  const neededData = veryLargeArray.length;
  
  document.getElementById('button').addEventListener('click', function() {
    // Only neededData is kept in memory
    console.log('Array length was:', neededData);
  });
  
  // veryLargeArray can be garbage collected
}

// Cleanup pattern
function createTimerWithCleanup() {
  let intervalId;
  let count = 0;
  
  const obj = {
    start() {
      intervalId = setInterval(() => {
        count++;
        console.log('Count:', count);
      }, 1000);
    },
    
    stop() {
      if (intervalId) {
        clearInterval(intervalId);
        intervalId = null;
      }
    },
    
    getCount() {
      return count;
    }
  };
  
  // Cleanup method
  obj.destroy = function() {
    this.stop();
    count = null;
  };
  
  return obj;
}
```

### Advanced Closure Techniques:

#### Partial Application:
```javascript
function partial(fn, ...presetArgs) {
  return function(...laterArgs) {
    return fn(...presetArgs, ...laterArgs);
  };
}

function multiply(a, b, c) {
  return a * b * c;
}

const multiplyByTwo = partial(multiply, 2);
const multiplyByTwoAndThree = partial(multiply, 2, 3);

console.log(multiplyByTwo(3, 4)); // 2 * 3 * 4 = 24
console.log(multiplyByTwoAndThree(5)); // 2 * 3 * 5 = 30

// Currying with closures
function curry(fn) {
  return function curried(...args) {
    if (args.length >= fn.length) {
      return fn.apply(this, args);
    } else {
      return function(...args2) {
        return curried.apply(this, args.concat(args2));
      };
    }
  };
}

const curriedMultiply = curry(multiply);
console.log(curriedMultiply(2)(3)(4)); // 24
console.log(curriedMultiply(2, 3)(4)); // 24
console.log(curriedMultiply(2)(3, 4)); // 24
```

#### Function Composition:
```javascript
function compose(...functions) {
  return function(value) {
    return functions.reduceRight((acc, fn) => fn(acc), value);
  };
}

function pipe(...functions) {
  return function(value) {
    return functions.reduce((acc, fn) => fn(acc), value);
  };
}

// Example functions
const add1 = x => x + 1;
const multiply2 = x => x * 2;
const subtract3 = x => x - 3;

// Composition (right to left)
const composed = compose(subtract3, multiply2, add1);
console.log(composed(5)); // ((5 + 1) * 2) - 3 = 9

// Pipe (left to right)
const piped = pipe(add1, multiply2, subtract3);
console.log(piped(5)); // ((5 + 1) * 2) - 3 = 9
```

## Prototypes and Inheritance

### Understanding Prototypes:
```javascript
// Every function has a prototype property
function Person(name) {
  this.name = name;
}

console.log(Person.prototype); // {}
console.log(typeof Person.prototype); // "object"

// Adding methods to prototype
Person.prototype.greet = function() {
  return `Hello, I'm ${this.name}`;
};

Person.prototype.species = 'Homo sapiens';

// Creating instances
const john = new Person('John');
const jane = new Person('Jane');

console.log(john.greet()); // "Hello, I'm John"
console.log(jane.greet()); // "Hello, I'm Jane"

// Both instances share the same method
console.log(john.greet === jane.greet); // true

// Prototype chain
console.log(john.__proto__ === Person.prototype); // true
console.log(Person.prototype.__proto__ === Object.prototype); // true
console.log(Object.prototype.__proto__ === null); // true
```

### Prototype Chain:
```javascript
// Understanding the prototype chain
function Animal(name) {
  this.name = name;
}

Animal.prototype.eat = function() {
  return `${this.name} is eating`;
};

function Dog(name, breed) {
  Animal.call(this, name); // Call parent constructor
  this.breed = breed;
}

// Set up inheritance
Dog.prototype = Object.create(Animal.prototype);
Dog.prototype.constructor = Dog;

Dog.prototype.bark = function() {
  return `${this.name} is barking`;
};

const dog = new Dog('Rex', 'German Shepherd');

console.log(dog.eat()); // "Rex is eating" (inherited)
console.log(dog.bark()); // "Rex is barking" (own method)

// Prototype chain traversal
console.log(dog.hasOwnProperty('name')); // true
console.log(dog.hasOwnProperty('eat')); // false (inherited)

// Walking the prototype chain
let current = dog;
let level = 0;

while (current) {
  console.log(`Level ${level}:`, current.constructor.name);
  current = Object.getPrototypeOf(current);
  level++;
}
// Level 0: Dog
// Level 1: Animal
// Level 2: Object
// Level 3: null
```

### Modern Inheritance Patterns:
```javascript
// Class-based inheritance (ES6+)
class Vehicle {
  constructor(make, model) {
    this.make = make;
    this.model = model;
  }
  
  start() {
    return `${this.make} ${this.model} is starting`;
  }
  
  static compare(v1, v2) {
    return v1.make === v2.make;
  }
}

class Car extends Vehicle {
  constructor(make, model, doors) {
    super(make, model);
    this.doors = doors;
  }
  
  start() {
    return `${super.start()} with ${this.doors} doors`;
  }
  
  honk() {
    return 'Beep beep!';
  }
}

const car = new Car('Toyota', 'Camry', 4);
console.log(car.start()); // "Toyota Camry is starting with 4 doors"

// Mixin pattern
const Flyable = {
  fly() {
    return `${this.name} is flying`;
  },
  land() {
    return `${this.name} has landed`;
  }
};

const Swimmable = {
  swim() {
    return `${this.name} is swimming`;
  },
  dive() {
    return `${this.name} is diving`;
  }
};

class Duck {
  constructor(name) {
    this.name = name;
  }
}

// Apply mixins
Object.assign(Duck.prototype, Flyable, Swimmable);

const duck = new Duck('Donald');
console.log(duck.fly()); // "Donald is flying"
console.log(duck.swim()); // "Donald is swimming"
```

### Advanced Prototype Manipulation:
```javascript
// Object.create() for prototype inheritance
const animalMethods = {
  eat() {
    return `${this.name} is eating`;
  },
  sleep() {
    return `${this.name} is sleeping`;
  }
};

function createAnimal(name, species) {
  const animal = Object.create(animalMethods);
  animal.name = name;
  animal.species = species;
  return animal;
}

const lion = createAnimal('Simba', 'Lion');
console.log(lion.eat()); // "Simba is eating"

// Prototype descriptor
const personPrototype = {
  greet() {
    return `Hello, I'm ${this.name}`;
  }
};

// Create object with specific prototype
const person = Object.create(personPrototype, {
  name: {
    value: 'Alice',
    writable: true,
    enumerable: true,
    configurable: true
  },
  age: {
    value: 30,
    writable: true,
    enumerable: false, // Won't show in for...in
    configurable: true
  }
});

console.log(person.greet()); // "Hello, I'm Alice"
console.log(Object.keys(person)); // ['name'] (age is not enumerable)

// Prototype pollution protection
function secureObjectCreate(proto, properties) {
  // Prevent prototype pollution
  if (proto === null || typeof proto !== 'object') {
    throw new Error('Prototype must be an object or null');
  }
  
  const dangerous = ['constructor', '__proto__', 'prototype'];
  if (properties) {
    for (const key of Object.keys(properties)) {
      if (dangerous.includes(key)) {
        throw new Error(`Property ${key} is not allowed`);
      }
    }
  }
  
  return Object.create(proto, properties);
}
```

### Prototype-based Design Patterns:
```javascript
// Prototype pattern for object creation
const CarPrototype = {
  init(make, model, year) {
    this.make = make;
    this.model = model;
    this.year = year;
    return this;
  },
  
  start() {
    return `${this.make} ${this.model} (${this.year}) is starting`;
  },
  
  clone() {
    return Object.create(CarPrototype).init(this.make, this.model, this.year);
  }
};

const originalCar = Object.create(CarPrototype).init('Honda', 'Civic', 2020);
const clonedCar = originalCar.clone();

console.log(originalCar.start()); // "Honda Civic (2020) is starting"
console.log(clonedCar.start()); // "Honda Civic (2020) is starting"

// Decorator pattern with prototypes
function addFeature(object, feature) {
  const decorated = Object.create(Object.getPrototypeOf(object));
  
  // Copy own properties
  Object.keys(object).forEach(key => {
    decorated[key] = object[key];
  });
  
  // Add feature
  Object.assign(decorated, feature);
  
  return decorated;
}

const basicCar = { make: 'Toyota', model: 'Corolla' };

const gpsFeature = {
  navigate(destination) {
    return `Navigating to ${destination}`;
  }
};

const airConditioningFeature = {
  setTemperature(temp) {
    return `Setting temperature to ${temp}°C`;
  }
};

const luxuryCar = addFeature(
  addFeature(basicCar, gpsFeature),
  airConditioningFeature
);

console.log(luxuryCar.navigate('Airport')); // "Navigating to Airport"
console.log(luxuryCar.setTemperature(22)); // "Setting temperature to 22°C"
```

## Advanced Functions

### Function Properties and Methods:
```javascript
// Function properties
function myFunction() {
  return 'Hello World';
}

console.log(myFunction.name); // "myFunction"
console.log(myFunction.length); // 0 (number of parameters)

function withParams(a, b, c = 'default') {
  return a + b + c;
}

console.log(withParams.length); // 2 (default parameters don't count)

// Custom function properties
myFunction.customProperty = 'I am a custom property';
myFunction.callCount = 0;

function trackingFunction() {
  trackingFunction.callCount++;
  return `Called ${trackingFunction.callCount} times`;
}

console.log(trackingFunction()); // "Called 1 times"
console.log(trackingFunction()); // "Called 2 times"
```

### call, apply, and bind:
```javascript
const person1 = { name: 'John', age: 30 };
const person2 = { name: 'Jane', age: 25 };

function introduce(greeting, punctuation) {
  return `${greeting}, I'm ${this.name} and I'm ${this.age} years old${punctuation}`;
}

// call() - invoke with specific 'this' and individual arguments
console.log(introduce.call(person1, 'Hello', '!')); 
// "Hello, I'm John and I'm 30 years old!"

// apply() - invoke with specific 'this' and array of arguments
console.log(introduce.apply(person2, ['Hi', '.'])); 
// "Hi, I'm Jane and I'm 25 years old."

// bind() - create new function with bound 'this'
const boundIntroduce = introduce.bind(person1);
console.log(boundIntroduce('Hey', '!!!')); 
// "Hey, I'm John and I'm 30 years old!!!"

// Partial application with bind
const greetJohn = introduce.bind(person1, 'Greetings');
console.log(greetJohn('.')); 
// "Greetings, I'm John and I'm 30 years old."

// Practical example: borrowing methods
const numbers = [1, 5, 3, 9, 2];
const max = Math.max.apply(null, numbers);
console.log(max); // 9

// Modern alternative with spread
const maxModern = Math.max(...numbers);
console.log(maxModern); // 9
```

### Higher-Order Functions:
```javascript
// Functions that take other functions as arguments
function withLogging(fn) {
  return function(...args) {
    console.log(`Calling function with args:`, args);
    const result = fn.apply(this, args);
    console.log(`Function returned:`, result);
    return result;
  };
}

function add(a, b) {
  return a + b;
}

const loggedAdd = withLogging(add);
loggedAdd(2, 3); // Logs the call and result

// Function composition
function compose(...functions) {
  return function(value) {
    return functions.reduceRight((acc, fn) => fn(acc), value);
  };
}

const addOne = x => x + 1;
const double = x => x * 2;
const square = x => x * x;

const composedFunction = compose(square, double, addOne);
console.log(composedFunction(3)); // ((3 + 1) * 2)² = 64

// Function pipeline
const pipe = (...functions) => (value) => 
  functions.reduce((acc, fn) => fn(acc), value);

const pipeline = pipe(addOne, double, square);
console.log(pipeline(3)); // ((3 + 1) * 2)² = 64

// Advanced HOF: retry function
function retry(fn, maxAttempts = 3, delay = 1000) {
  return async function(...args) {
    let lastError;
    
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      try {
        return await fn.apply(this, args);
      } catch (error) {
        lastError = error;
        console.log(`Attempt ${attempt} failed:`, error.message);
        
        if (attempt < maxAttempts) {
          await new Promise(resolve => setTimeout(resolve, delay));
        }
      }
    }
    
    throw lastError;
  };
}

// Usage
const unreliableAPI = async () => {
  if (Math.random() < 0.7) {
    throw new Error('Network error');
  }
  return 'Success!';
};

const reliableAPI = retry(unreliableAPI, 3, 500);
```

### Function Factories:
```javascript
// Creating specialized functions
function createValidator(validationRules) {
  return function(data) {
    const errors = [];
    
    for (const [field, rules] of Object.entries(validationRules)) {
      const value = data[field];
      
      for (const rule of rules) {
        if (!rule.test(value)) {
          errors.push({
            field,
            message: rule.message,
            value
          });
        }
      }
    }
    
    return {
      isValid: errors.length === 0,
      errors
    };
  };
}

// Validation rules
const userValidationRules = {
  email: [
    {
      test: value => typeof value === 'string' && value.includes('@'),
      message: 'Email must be a valid email address'
    }
  ],
  age: [
    {
      test: value => typeof value === 'number' && value >= 0 && value <= 150,
      message: 'Age must be between 0 and 150'
    }
  ],
  name: [
    {
      test: value => typeof value === 'string' && value.length >= 2,
      message: 'Name must be at least 2 characters long'
    }
  ]
};

const validateUser = createValidator(userValidationRules);

const userData = { email: 'invalid', age: -5, name: 'A' };
const validation = validateUser(userData);
console.log(validation);

// State machine factory
function createStateMachine(states, initialState) {
  let currentState = initialState;
  const history = [initialState];
  
  return {
    getCurrentState() {
      return currentState;
    },
    
    transition(event) {
      const stateConfig = states[currentState];
      
      if (stateConfig && stateConfig.on && stateConfig.on[event]) {
        const newState = stateConfig.on[event];
        history.push(newState);
        currentState = newState;
        
        // Execute enter action if defined
        if (states[newState].enter) {
          states[newState].enter();
        }
        
        return true;
      }
      
      return false; // Invalid transition
    },
    
    getHistory() {
      return [...history];
    },
    
    canTransition(event) {
      const stateConfig = states[currentState];
      return !!(stateConfig && stateConfig.on && stateConfig.on[event]);
    }
  };
}

// Traffic light state machine
const trafficLightStates = {
  red: {
    on: { timer: 'green' },
    enter: () => console.log('🔴 Stop!')
  },
  green: {
    on: { timer: 'yellow' },
    enter: () => console.log('🟢 Go!')
  },
  yellow: {
    on: { timer: 'red' },
    enter: () => console.log('🟡 Caution!')
  }
};

const trafficLight = createStateMachine(trafficLightStates, 'red');
trafficLight.transition('timer'); // 🟢 Go!
trafficLight.transition('timer'); // 🟡 Caution!
trafficLight.transition('timer'); // 🔴 Stop!
```

### Function Decorators:
```javascript
// Decorator functions
function time(fn) {
  return function(...args) {
    console.time(fn.name);
    const result = fn.apply(this, args);
    console.timeEnd(fn.name);
    return result;
  };
}

function cache(fn) {
  const cacheMap = new Map();
  
  return function(...args) {
    const key = JSON.stringify(args);
    
    if (cacheMap.has(key)) {
      console.log('Cache hit');
      return cacheMap.get(key);
    }
    
    const result = fn.apply(this, args);
    cacheMap.set(key, result);
    return result;
  };
}

function debounce(fn, delay) {
  let timeoutId;
  
  return function(...args) {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => fn.apply(this, args), delay);
  };
}

function throttle(fn, limit) {
  let inThrottle;
  
  return function(...args) {
    if (!inThrottle) {
      fn.apply(this, args);
      inThrottle = true;
      setTimeout(() => inThrottle = false, limit);
    }
  };
}

// Decorator composition
function applyDecorators(fn, ...decorators) {
  return decorators.reduce((decorated, decorator) => decorator(decorated), fn);
}

// Example function
function fibonacci(n) {
  if (n <= 1) return n;
  return fibonacci(n - 1) + fibonacci(n - 2);
}

// Apply multiple decorators
const optimizedFib = applyDecorators(fibonacci, cache, time);

console.log(optimizedFib(40)); // First call - slow
console.log(optimizedFib(40)); // Second call - fast (cached)

// Class method decorators (using proposal syntax conceptually)
function methodDecorator(target, propertyKey, descriptor) {
  const originalMethod = descriptor.value;
  
  descriptor.value = function(...args) {
    console.log(`Calling ${propertyKey} with args:`, args);
    return originalMethod.apply(this, args);
  };
  
  return descriptor;
}

// Manual application (since decorators aren't fully supported yet)
class Calculator {
  add(a, b) {
    return a + b;
  }
}

// Apply decorator manually
const descriptor = Object.getOwnPropertyDescriptor(Calculator.prototype, 'add');
const decorated = methodDecorator(Calculator.prototype, 'add', descriptor);
Object.defineProperty(Calculator.prototype, 'add', decorated);

const calc = new Calculator();
calc.add(2, 3); // Logs: "Calling add with args: [2, 3]"
```

This covers the first major section of advanced JavaScript concepts. The content includes asynchronous programming, closures, prototypes, and advanced function techniques. Due to length limits, I'll continue with the remaining advanced topics in separate files.

---

**Next sections to cover in separate files:**
- Memory Management & Performance Optimization
- Design Patterns
- Metaprogramming
- Web APIs
- Testing
