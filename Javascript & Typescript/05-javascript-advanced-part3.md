# JavaScript Advanced Concepts - Part 3

## Table of Contents
1. [Metaprogramming](#metaprogramming)
2. [Web APIs](#web-apis)
3. [Testing](#testing)
4. [Modern JavaScript Features](#modern-javascript-features)

## Metaprogramming

### Proxies:
```javascript
// Basic Proxy usage
const target = {
  name: 'John',
  age: 30
};

const handler = {
  get(target, property, receiver) {
    console.log(`Getting property: ${property}`);
    
    if (property in target) {
      return target[property];
    } else {
      console.log(`Property ${property} does not exist`);
      return undefined;
    }
  },
  
  set(target, property, value, receiver) {
    console.log(`Setting property: ${property} = ${value}`);
    
    if (property === 'age' && typeof value !== 'number') {
      throw new TypeError('Age must be a number');
    }
    
    target[property] = value;
    return true;
  },
  
  has(target, property) {
    console.log(`Checking if property exists: ${property}`);
    return property in target;
  },
  
  deleteProperty(target, property) {
    console.log(`Deleting property: ${property}`);
    
    if (property === 'name') {
      throw new Error('Cannot delete name property');
    }
    
    delete target[property];
    return true;
  },
  
  ownKeys(target) {
    console.log('Getting all keys');
    return Object.keys(target);
  }
};

const proxy = new Proxy(target, handler);

// Usage examples
console.log(proxy.name);        // Getting property: name
proxy.age = 31;                 // Setting property: age = 31
console.log('age' in proxy);    // Checking if property exists: age
console.log(Object.keys(proxy)); // Getting all keys

// Advanced Proxy patterns
class ObservableObject {
  constructor(target, onChange) {
    this.target = target;
    this.onChange = onChange;
    
    return new Proxy(target, {
      set: (target, property, value, receiver) => {
        const oldValue = target[property];
        target[property] = value;
        
        this.onChange({
          type: 'set',
          property,
          value,
          oldValue,
          target: receiver
        });
        
        return true;
      },
      
      deleteProperty: (target, property) => {
        const oldValue = target[property];
        delete target[property];
        
        this.onChange({
          type: 'delete',
          property,
          oldValue,
          target
        });
        
        return true;
      }
    });
  }
}

const observableUser = new ObservableObject(
  { name: 'Alice', age: 25 },
  (change) => console.log('Change detected:', change)
);

observableUser.name = 'Bob';     // Change detected: { type: 'set', ... }
delete observableUser.age;       // Change detected: { type: 'delete', ... }

// Validation Proxy
class ValidatedObject {
  constructor(target, validators = {}) {
    this.validators = validators;
    
    return new Proxy(target, {
      set: (target, property, value) => {
        if (this.validators[property]) {
          const isValid = this.validators[property](value);
          if (!isValid) {
            throw new Error(`Invalid value for ${property}: ${value}`);
          }
        }
        
        target[property] = value;
        return true;
      }
    });
  }
}

const user = new ValidatedObject({}, {
  email: (value) => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value),
  age: (value) => typeof value === 'number' && value >= 0 && value <= 150
});

user.email = 'test@example.com'; // Valid
user.age = 25;                   // Valid
// user.email = 'invalid';       // Would throw error

// Function proxy for logging
function createLoggedFunction(fn, name) {
  return new Proxy(fn, {
    apply(target, thisArg, argumentsList) {
      console.log(`Calling function ${name} with args:`, argumentsList);
      const result = target.apply(thisArg, argumentsList);
      console.log(`Function ${name} returned:`, result);
      return result;
    }
  });
}

const add = createLoggedFunction((a, b) => a + b, 'add');
add(5, 3); // Logs function call and result

// Array-like object with Proxy
class CustomArray {
  constructor(...items) {
    this.items = items;
    this.length = items.length;
    
    return new Proxy(this, {
      get(target, property) {
        if (property in target) {
          return target[property];
        }
        
        const index = Number(property);
        if (Number.isInteger(index) && index >= 0) {
          return target.items[index];
        }
        
        return undefined;
      },
      
      set(target, property, value) {
        const index = Number(property);
        if (Number.isInteger(index) && index >= 0) {
          target.items[index] = value;
          target.length = Math.max(target.length, index + 1);
          return true;
        }
        
        target[property] = value;
        return true;
      },
      
      has(target, property) {
        const index = Number(property);
        if (Number.isInteger(index) && index >= 0) {
          return index < target.length;
        }
        return property in target;
      }
    });
  }
  
  push(...items) {
    this.items.push(...items);
    this.length = this.items.length;
    return this.length;
  }
  
  pop() {
    const result = this.items.pop();
    this.length = this.items.length;
    return result;
  }
}

const customArray = new CustomArray(1, 2, 3);
console.log(customArray[0]);    // 1
customArray[3] = 4;
console.log(customArray.length); // 4
```

### Reflect API:
```javascript
// Reflect provides methods for interceptable JavaScript operations
const obj = { name: 'John', age: 30 };

// Reflect.get() - like obj[prop]
console.log(Reflect.get(obj, 'name')); // 'John'

// Reflect.set() - like obj[prop] = value
Reflect.set(obj, 'city', 'New York');
console.log(obj); // { name: 'John', age: 30, city: 'New York' }

// Reflect.has() - like 'prop' in obj
console.log(Reflect.has(obj, 'name')); // true

// Reflect.deleteProperty() - like delete obj[prop]
Reflect.deleteProperty(obj, 'age');
console.log(obj); // { name: 'John', city: 'New York' }

// Reflect.ownKeys() - like Object.keys(obj)
console.log(Reflect.ownKeys(obj)); // ['name', 'city']

// Using Reflect with Proxy for better forwarding
const enhancedHandler = {
  get(target, property, receiver) {
    console.log(`Accessing property: ${property}`);
    return Reflect.get(target, property, receiver);
  },
  
  set(target, property, value, receiver) {
    console.log(`Setting property: ${property} = ${value}`);
    return Reflect.set(target, property, value, receiver);
  },
  
  defineProperty(target, property, descriptor) {
    console.log(`Defining property: ${property}`);
    return Reflect.defineProperty(target, property, descriptor);
  },
  
  getPrototypeOf(target) {
    console.log('Getting prototype');
    return Reflect.getPrototypeOf(target);
  }
};

const enhancedProxy = new Proxy({}, enhancedHandler);

// Reflect.construct() - like new Constructor()
class Person {
  constructor(name, age) {
    this.name = name;
    this.age = age;
  }
}

const person = Reflect.construct(Person, ['Alice', 25]);
console.log(person); // Person { name: 'Alice', age: 25 }

// Reflect.apply() - like func.apply()
function greet(greeting, name) {
  return `${greeting}, ${name}!`;
}

const result = Reflect.apply(greet, null, ['Hello', 'World']);
console.log(result); // 'Hello, World!'

// Meta-programming with getters/setters
class SmartObject {
  constructor() {
    this._data = {};
    
    return new Proxy(this, {
      get(target, property) {
        if (property.startsWith('get')) {
          const propName = property.slice(3).toLowerCase();
          return () => target._data[propName];
        }
        
        if (property.startsWith('set')) {
          const propName = property.slice(3).toLowerCase();
          return (value) => {
            target._data[propName] = value;
            return target;
          };
        }
        
        return Reflect.get(target, property);
      }
    });
  }
}

const smart = new SmartObject();
smart.setName('John');
smart.setAge(30);
console.log(smart.getName()); // 'John'
console.log(smart.getAge());  // 30
```

### Symbols and Well-Known Symbols:
```javascript
// Creating symbols
const sym1 = Symbol();
const sym2 = Symbol('description');
const sym3 = Symbol('description');

console.log(sym2 === sym3); // false - symbols are always unique

// Symbol registry
const globalSym1 = Symbol.for('mySymbol');
const globalSym2 = Symbol.for('mySymbol');
console.log(globalSym1 === globalSym2); // true

console.log(Symbol.keyFor(globalSym1)); // 'mySymbol'

// Using symbols as object keys
const obj = {};
const nameSymbol = Symbol('name');
const ageSymbol = Symbol('age');

obj[nameSymbol] = 'John';
obj[ageSymbol] = 30;
obj.regularProp = 'visible';

console.log(Object.keys(obj));           // ['regularProp']
console.log(Object.getOwnPropertySymbols(obj)); // [Symbol(name), Symbol(age)]

// Well-known symbols for metaprogramming
class CustomIterable {
  constructor(data) {
    this.data = data;
  }
  
  // Symbol.iterator makes object iterable
  [Symbol.iterator]() {
    let index = 0;
    const data = this.data;
    
    return {
      next() {
        if (index < data.length) {
          return { value: data[index++], done: false };
        } else {
          return { done: true };
        }
      }
    };
  }
  
  // Symbol.toStringTag for Object.prototype.toString
  [Symbol.toStringTag] = 'CustomIterable';
  
  // Symbol.hasInstance for instanceof operator
  static [Symbol.hasInstance](instance) {
    return instance.data && Array.isArray(instance.data);
  }
}

const iterable = new CustomIterable([1, 2, 3, 4, 5]);

// Use with for...of
for (const value of iterable) {
  console.log(value); // 1, 2, 3, 4, 5
}

console.log(Object.prototype.toString.call(iterable)); // [object CustomIterable]
console.log(iterable instanceof CustomIterable); // true

// Symbol.toPrimitive for type conversion
class Money {
  constructor(amount, currency) {
    this.amount = amount;
    this.currency = currency;
  }
  
  [Symbol.toPrimitive](hint) {
    switch (hint) {
      case 'number':
        return this.amount;
      case 'string':
        return `${this.amount} ${this.currency}`;
      default:
        return `${this.amount} ${this.currency}`;
    }
  }
}

const money = new Money(100, 'USD');
console.log(+money);        // 100 (number conversion)
console.log(`${money}`);    // '100 USD' (string conversion)
console.log(money + '');    // '100 USD' (default conversion)

// Private fields with symbols
const _private = Symbol('private');

class BankAccount {
  constructor(initialBalance) {
    this[_private] = {
      balance: initialBalance,
      transactions: []
    };
  }
  
  deposit(amount) {
    this[_private].balance += amount;
    this[_private].transactions.push({ type: 'deposit', amount });
  }
  
  withdraw(amount) {
    if (this[_private].balance >= amount) {
      this[_private].balance -= amount;
      this[_private].transactions.push({ type: 'withdraw', amount });
      return true;
    }
    return false;
  }
  
  getBalance() {
    return this[_private].balance;
  }
}

const account = new BankAccount(1000);
account.deposit(500);
console.log(account.getBalance()); // 1500
// account[_private] is not accessible from outside without the symbol
```

## Web APIs

### Fetch API:
```javascript
// Basic fetch usage
async function fetchBasicExample() {
  try {
    const response = await fetch('https://jsonplaceholder.typicode.com/posts/1');
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    console.log('Post data:', data);
    
    // Response metadata
    console.log('Status:', response.status);
    console.log('Headers:', response.headers);
    console.log('URL:', response.url);
    
  } catch (error) {
    console.error('Fetch error:', error);
  }
}

// Advanced fetch with options
async function fetchAdvanced() {
  const options = {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': 'Bearer your-token-here',
      'X-Custom-Header': 'custom-value'
    },
    body: JSON.stringify({
      title: 'New Post',
      body: 'This is the post content',
      userId: 1
    }),
    mode: 'cors',           // cors, no-cors, same-origin
    credentials: 'include', // include, same-origin, omit
    cache: 'no-cache',      // default, no-cache, reload, force-cache, only-if-cached
    redirect: 'follow',     // follow, error, manual
    referrerPolicy: 'no-referrer'
  };
  
  try {
    const response = await fetch('https://jsonplaceholder.typicode.com/posts', options);
    const data = await response.json();
    console.log('Created post:', data);
  } catch (error) {
    console.error('Error creating post:', error);
  }
}

// HTTP client class
class HTTPClient {
  constructor(baseURL = '', defaultOptions = {}) {
    this.baseURL = baseURL;
    this.defaultOptions = {
      headers: {
        'Content-Type': 'application/json'
      },
      ...defaultOptions
    };
    this.interceptors = {
      request: [],
      response: []
    };
  }
  
  addRequestInterceptor(interceptor) {
    this.interceptors.request.push(interceptor);
  }
  
  addResponseInterceptor(interceptor) {
    this.interceptors.response.push(interceptor);
  }
  
  async request(url, options = {}) {
    // Apply request interceptors
    let config = {
      ...this.defaultOptions,
      ...options,
      headers: {
        ...this.defaultOptions.headers,
        ...options.headers
      }
    };
    
    for (const interceptor of this.interceptors.request) {
      config = await interceptor(config);
    }
    
    const fullURL = url.startsWith('http') ? url : `${this.baseURL}${url}`;
    
    try {
      let response = await fetch(fullURL, config);
      
      // Apply response interceptors
      for (const interceptor of this.interceptors.response) {
        response = await interceptor(response);
      }
      
      return response;
    } catch (error) {
      throw error;
    }
  }
  
  async get(url, options = {}) {
    return this.request(url, { ...options, method: 'GET' });
  }
  
  async post(url, data, options = {}) {
    return this.request(url, {
      ...options,
      method: 'POST',
      body: JSON.stringify(data)
    });
  }
  
  async put(url, data, options = {}) {
    return this.request(url, {
      ...options,
      method: 'PUT',
      body: JSON.stringify(data)
    });
  }
  
  async delete(url, options = {}) {
    return this.request(url, { ...options, method: 'DELETE' });
  }
}
```

### WebSockets:
```javascript
// WebSocket client class
class WebSocketClient {
  constructor(url, protocols = []) {
    this.url = url;
    this.protocols = protocols;
    this.ws = null;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.reconnectInterval = 1000;
    this.messageQueue = [];
    this.eventHandlers = new Map();
  }
  
  connect() {
    try {
      this.ws = new WebSocket(this.url, this.protocols);
      
      this.ws.onopen = (event) => {
        console.log('WebSocket connected');
        this.reconnectAttempts = 0;
        
        // Send queued messages
        while (this.messageQueue.length > 0) {
          const message = this.messageQueue.shift();
          this.ws.send(message);
        }
        
        this.emit('open', event);
      };
      
      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          this.emit('message', data);
          
          // Handle specific message types
          if (data.type) {
            this.emit(`message:${data.type}`, data);
          }
        } catch (error) {
          console.error('Error parsing message:', error);
          this.emit('message', event.data);
        }
      };
      
      this.ws.onclose = (event) => {
        console.log('WebSocket closed:', event.code, event.reason);
        this.emit('close', event);
        
        // Attempt reconnection
        if (!event.wasClean && this.reconnectAttempts < this.maxReconnectAttempts) {
          this.reconnect();
        }
      };
      
      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        this.emit('error', error);
      };
      
    } catch (error) {
      console.error('Failed to create WebSocket:', error);
      this.emit('error', error);
    }
  }
  
  reconnect() {
    this.reconnectAttempts++;
    console.log(`Reconnecting... attempt ${this.reconnectAttempts}`);
    
    setTimeout(() => {
      this.connect();
    }, this.reconnectInterval * this.reconnectAttempts);
  }
  
  send(data) {
    const message = typeof data === 'string' ? data : JSON.stringify(data);
    
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(message);
    } else {
      // Queue message for when connection is established
      this.messageQueue.push(message);
    }
  }
  
  close(code = 1000, reason = '') {
    if (this.ws) {
      this.ws.close(code, reason);
    }
  }
  
  on(event, handler) {
    if (!this.eventHandlers.has(event)) {
      this.eventHandlers.set(event, []);
    }
    this.eventHandlers.get(event).push(handler);
  }
  
  off(event, handler) {
    if (this.eventHandlers.has(event)) {
      const handlers = this.eventHandlers.get(event);
      const index = handlers.indexOf(handler);
      if (index > -1) {
        handlers.splice(index, 1);
      }
    }
  }
  
  emit(event, data) {
    if (this.eventHandlers.has(event)) {
      this.eventHandlers.get(event).forEach(handler => {
        try {
          handler(data);
        } catch (error) {
          console.error('Error in event handler:', error);
        }
      });
    }
  }
  
  getState() {
    if (!this.ws) return 'CLOSED';
    
    switch (this.ws.readyState) {
      case WebSocket.CONNECTING: return 'CONNECTING';
      case WebSocket.OPEN: return 'OPEN';
      case WebSocket.CLOSING: return 'CLOSING';
      case WebSocket.CLOSED: return 'CLOSED';
      default: return 'UNKNOWN';
    }
  }
}

// Real-time chat example
class ChatClient extends WebSocketClient {
  constructor(url, username) {
    super(url);
    this.username = username;
    this.rooms = new Set();
  }
  
  joinRoom(roomId) {
    this.send({
      type: 'join_room',
      roomId,
      username: this.username
    });
    this.rooms.add(roomId);
  }
  
  leaveRoom(roomId) {
    this.send({
      type: 'leave_room',
      roomId,
      username: this.username
    });
    this.rooms.delete(roomId);
  }
  
  sendMessage(roomId, message) {
    this.send({
      type: 'chat_message',
      roomId,
      username: this.username,
      message,
      timestamp: Date.now()
    });
  }
}
```

## Testing

### Unit Testing Fundamentals:
```javascript
// Simple test framework
class TestFramework {
  constructor() {
    this.tests = [];
    this.hooks = {
      beforeAll: [],
      beforeEach: [],
      afterEach: [],
      afterAll: []
    };
  }
  
  describe(description, callback) {
    console.log(`\n--- ${description} ---`);
    callback();
  }
  
  it(description, testFunction) {
    this.tests.push({ description, testFunction });
  }
  
  beforeAll(hook) {
    this.hooks.beforeAll.push(hook);
  }
  
  beforeEach(hook) {
    this.hooks.beforeEach.push(hook);
  }
  
  afterEach(hook) {
    this.hooks.afterEach.push(hook);
  }
  
  afterAll(hook) {
    this.hooks.afterAll.push(hook);
  }
  
  async run() {
    let passed = 0;
    let failed = 0;
    
    // Run beforeAll hooks
    for (const hook of this.hooks.beforeAll) {
      await hook();
    }
    
    for (const test of this.tests) {
      try {
        // Run beforeEach hooks
        for (const hook of this.hooks.beforeEach) {
          await hook();
        }
        
        await test.testFunction();
        console.log(`✓ ${test.description}`);
        passed++;
        
        // Run afterEach hooks
        for (const hook of this.hooks.afterEach) {
          await hook();
        }
      } catch (error) {
        console.log(`✗ ${test.description}`);
        console.log(`  Error: ${error.message}`);
        failed++;
      }
    }
    
    // Run afterAll hooks
    for (const hook of this.hooks.afterAll) {
      await hook();
    }
    
    console.log(`\nResults: ${passed} passed, ${failed} failed`);
  }
}

// Assertion library
class Assertions {
  static assertEqual(actual, expected, message = '') {
    if (actual !== expected) {
      throw new Error(
        `${message}\nExpected: ${expected}\nActual: ${actual}`
      );
    }
  }
  
  static assertDeepEqual(actual, expected, message = '') {
    if (JSON.stringify(actual) !== JSON.stringify(expected)) {
      throw new Error(
        `${message}\nExpected: ${JSON.stringify(expected)}\nActual: ${JSON.stringify(actual)}`
      );
    }
  }
  
  static assertTrue(value, message = '') {
    if (!value) {
      throw new Error(`${message}\nExpected: true\nActual: ${value}`);
    }
  }
  
  static assertFalse(value, message = '') {
    if (value) {
      throw new Error(`${message}\nExpected: false\nActual: ${value}`);
    }
  }
  
  static assertThrows(fn, errorType = Error, message = '') {
    try {
      fn();
      throw new Error(`${message}\nExpected function to throw ${errorType.name}`);
    } catch (error) {
      if (!(error instanceof errorType)) {
        throw new Error(
          `${message}\nExpected: ${errorType.name}\nActual: ${error.constructor.name}`
        );
      }
    }
  }
  
  static async assertRejects(promise, errorType = Error, message = '') {
    try {
      await promise;
      throw new Error(`${message}\nExpected promise to reject with ${errorType.name}`);
    } catch (error) {
      if (!(error instanceof errorType)) {
        throw new Error(
          `${message}\nExpected: ${errorType.name}\nActual: ${error.constructor.name}`
        );
      }
    }
  }
}

// Example usage
const test = new TestFramework();

// Test a Calculator class
class Calculator {
  add(a, b) {
    return a + b;
  }
  
  divide(a, b) {
    if (b === 0) {
      throw new Error('Division by zero');
    }
    return a / b;
  }
  
  async asyncAdd(a, b) {
    return new Promise(resolve => {
      setTimeout(() => resolve(a + b), 10);
    });
  }
}

test.describe('Calculator Tests', () => {
  let calculator;
  
  test.beforeEach(() => {
    calculator = new Calculator();
  });
  
  test.it('should add two numbers correctly', () => {
    const result = calculator.add(2, 3);
    Assertions.assertEqual(result, 5, 'Addition failed');
  });
  
  test.it('should throw error when dividing by zero', () => {
    Assertions.assertThrows(
      () => calculator.divide(10, 0),
      Error,
      'Division by zero should throw error'
    );
  });
  
  test.it('should handle async operations', async () => {
    const result = await calculator.asyncAdd(5, 3);
    Assertions.assertEqual(result, 8, 'Async addition failed');
  });
});

// Run tests
test.run();
```

### Test Mocking and Spies:
```javascript
// Mock function implementation
class MockFunction {
  constructor(implementation = () => {}) {
    this.implementation = implementation;
    this.calls = [];
    this.returnValues = [];
  }
  
  (...args) {
    this.calls.push({
      args: [...args],
      timestamp: Date.now()
    });
    
    const result = this.implementation(...args);
    this.returnValues.push(result);
    return result;
  }
  
  mockReturnValue(value) {
    this.implementation = () => value;
    return this;
  }
  
  mockReturnValueOnce(value) {
    const currentImplementation = this.implementation;
    let called = false;
    
    this.implementation = (...args) => {
      if (!called) {
        called = true;
        return value;
      }
      return currentImplementation(...args);
    };
    
    return this;
  }
  
  mockImplementation(fn) {
    this.implementation = fn;
    return this;
  }
  
  mockResolvedValue(value) {
    this.implementation = () => Promise.resolve(value);
    return this;
  }
  
  mockRejectedValue(error) {
    this.implementation = () => Promise.reject(error);
    return this;
  }
  
  // Assertions
  toHaveBeenCalled() {
    return this.calls.length > 0;
  }
  
  toHaveBeenCalledTimes(times) {
    return this.calls.length === times;
  }
  
  toHaveBeenCalledWith(...args) {
    return this.calls.some(call => 
      call.args.length === args.length &&
      call.args.every((arg, index) => arg === args[index])
    );
  }
  
  toHaveBeenLastCalledWith(...args) {
    if (this.calls.length === 0) return false;
    const lastCall = this.calls[this.calls.length - 1];
    return lastCall.args.length === args.length &&
           lastCall.args.every((arg, index) => arg === args[index]);
  }
  
  reset() {
    this.calls = [];
    this.returnValues = [];
    this.implementation = () => {};
  }
}

// Spy implementation
class Spy {
  constructor(object, methodName) {
    this.object = object;
    this.methodName = methodName;
    this.originalMethod = object[methodName];
    this.mockFn = new MockFunction(this.originalMethod);
    
    // Replace the original method
    object[methodName] = (...args) => this.mockFn(...args);
  }
  
  restore() {
    this.object[this.methodName] = this.originalMethod;
  }
  
  // Delegate to mock function
  mockReturnValue(value) {
    this.mockFn.mockReturnValue(value);
    return this;
  }
  
  mockImplementation(fn) {
    this.mockFn.mockImplementation(fn);
    return this;
  }
  
  toHaveBeenCalled() {
    return this.mockFn.toHaveBeenCalled();
  }
  
  toHaveBeenCalledTimes(times) {
    return this.mockFn.toHaveBeenCalledTimes(times);
  }
  
  toHaveBeenCalledWith(...args) {
    return this.mockFn.toHaveBeenCalledWith(...args);
  }
}

// Test utilities
const TestUtils = {
  createMock: (implementation) => new MockFunction(implementation),
  
  spy: (object, methodName) => new Spy(object, methodName),
  
  mockModule: (module, mocks) => {
    const originalMethods = {};
    
    for (const [methodName, mockImplementation] of Object.entries(mocks)) {
      originalMethods[methodName] = module[methodName];
      module[methodName] = mockImplementation;
    }
    
    return {
      restore: () => {
        for (const [methodName, originalMethod] of Object.entries(originalMethods)) {
          module[methodName] = originalMethod;
        }
      }
    };
  },
  
  // Timer mocking
  mockTimers: () => {
    const originalSetTimeout = global.setTimeout;
    const originalSetInterval = global.setInterval;
    const originalClearTimeout = global.clearTimeout;
    const originalClearInterval = global.clearInterval;
    
    let timers = [];
    let currentTime = 0;
    
    global.setTimeout = (callback, delay) => {
      const id = timers.length;
      timers.push({
        id,
        callback,
        time: currentTime + delay,
        type: 'timeout'
      });
      return id;
    };
    
    global.setInterval = (callback, delay) => {
      const id = timers.length;
      timers.push({
        id,
        callback,
        time: currentTime + delay,
        interval: delay,
        type: 'interval'
      });
      return id;
    };
    
    return {
      advanceTimersByTime: (time) => {
        currentTime += time;
        
        const readyTimers = timers.filter(timer => timer.time <= currentTime);
        
        for (const timer of readyTimers) {
          timer.callback();
          
          if (timer.type === 'interval') {
            timer.time = currentTime + timer.interval;
          } else {
            timers = timers.filter(t => t.id !== timer.id);
          }
        }
      },
      
      restore: () => {
        global.setTimeout = originalSetTimeout;
        global.setInterval = originalSetInterval;
        global.clearTimeout = originalClearTimeout;
        global.clearInterval = originalClearInterval;
      }
    };
  }
};

// Example usage
test.describe('UserService Tests', () => {
  let userService;
  let apiClient;
  let mockFetch;
  
  test.beforeEach(() => {
    // Mock API client
    apiClient = {
      get: TestUtils.createMock(),
      post: TestUtils.createMock()
    };
    
    userService = new UserService(apiClient);
    
    // Mock global fetch
    mockFetch = TestUtils.createMock();
    global.fetch = mockFetch;
  });
  
  test.it('should fetch user data', async () => {
    const userData = { id: 1, name: 'John' };
    apiClient.get.mockResolvedValue(userData);
    
    const result = await userService.getUser(1);
    
    Assertions.assertTrue(apiClient.get.toHaveBeenCalledWith('/users/1'));
    Assertions.assertDeepEqual(result, userData);
  });
  
  test.it('should handle API errors', async () => {
    const error = new Error('API Error');
    apiClient.get.mockRejectedValue(error);
    
    await Assertions.assertRejects(
      userService.getUser(1),
      Error,
      'Should propagate API errors'
    );
  });
});

// Example service to test
class UserService {
  constructor(apiClient) {
    this.apiClient = apiClient;
  }
  
  async getUser(id) {
    return await this.apiClient.get(`/users/${id}`);
  }
  
  async createUser(userData) {
    return await this.apiClient.post('/users', userData);
  }
}
```

## Modern JavaScript Features

### ES2020+ Features:
```javascript
// Optional Chaining (?.)
const user = {
  profile: {
    social: {
      twitter: '@johndoe'
    }
  }
};

// Before optional chaining
const twitter1 = user && user.profile && user.profile.social && user.profile.social.twitter;

// With optional chaining
const twitter2 = user?.profile?.social?.twitter;
const instagram = user?.profile?.social?.instagram; // undefined (no error)

// Method calls
const result = user?.profile?.getName?.(); // Won't error if getName doesn't exist

// Array access
const firstPost = user?.posts?.[0];

// Nullish Coalescing (??)
const username = user?.name ?? 'Anonymous';
const port = process.env.PORT ?? 3000;

// Different from ||
const config = {
  debug: false,
  timeout: 0
};

const debugMode = config.debug || true;     // true (incorrect)
const debugMode2 = config.debug ?? true;   // false (correct)
const timeout = config.timeout || 5000;    // 5000 (incorrect)
const timeout2 = config.timeout ?? 5000;   // 0 (correct)

// Logical Assignment Operators
let x = null;
x ??= 'default value';  // x = x ?? 'default value'
console.log(x); // 'default value'

let y = false;
y ||= true;  // y = y || true
console.log(y); // true

let z = true;
z &&= false;  // z = z && false
console.log(z); // false

// Dynamic Imports
async function loadModule() {
  try {
    const module = await import('./math-utils.js');
    const result = module.add(5, 3);
    console.log(result);
  } catch (error) {
    console.error('Failed to load module:', error);
  }
}

// Conditional dynamic imports
async function loadChartLibrary(type) {
  let chartModule;
  
  switch (type) {
    case 'line':
      chartModule = await import('./line-chart.js');
      break;
    case 'bar':
      chartModule = await import('./bar-chart.js');
      break;
    default:
      chartModule = await import('./default-chart.js');
  }
  
  return chartModule.default;
}

// Promise.allSettled()
const promises = [
  fetch('/api/users'),
  fetch('/api/posts'),
  fetch('/api/comments')
];

const results = await Promise.allSettled(promises);

results.forEach((result, index) => {
  if (result.status === 'fulfilled') {
    console.log(`Promise ${index} succeeded:`, result.value);
  } else {
    console.log(`Promise ${index} failed:`, result.reason);
  }
});

// String.prototype.matchAll()
const text = 'The year 2023 and 2024 are important';
const yearRegex = /(\d{4})/g;

// Old way
const matches1 = [];
let match;
while ((match = yearRegex.exec(text)) !== null) {
  matches1.push(match);
}

// New way
const matches2 = [...text.matchAll(yearRegex)];
console.log(matches2); // [['2023', '2023'], ['2024', '2024']]

// GlobalThis
// Works in all environments (browser, Node.js, Web Workers)
const global = globalThis;

// BigInt for large integers
const bigNumber = 9007199254740991n; // or BigInt(9007199254740991)
const anotherBig = BigInt('12345678901234567890');

console.log(bigNumber + 1n); // 9007199254740992n
console.log(typeof bigNumber); // 'bigint'

// Cannot mix BigInt with regular numbers
// console.log(bigNumber + 1); // TypeError
console.log(bigNumber + BigInt(1)); // OK

// WeakRefs and FinalizationRegistry
let target = { name: 'example' };
const weakRef = new WeakRef(target);

// Later, check if object is still alive
if (weakRef.deref()) {
  console.log('Object is still alive:', weakRef.deref().name);
} else {
  console.log('Object has been garbage collected');
}

// FinalizationRegistry for cleanup
const registry = new FinalizationRegistry((heldValue) => {
  console.log('Cleanup:', heldValue);
});

registry.register(target, 'some-identifier');

// Private fields and methods
class User {
  #id;
  #name;
  #email;
  
  constructor(id, name, email) {
    this.#id = id;
    this.#name = name;
    this.#email = email;
  }
  
  #validateEmail(email) {
    return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
  }
  
  #hashPassword(password) {
    // Private method implementation
    return password + '_hashed';
  }
  
  updateEmail(newEmail) {
    if (this.#validateEmail(newEmail)) {
      this.#email = newEmail;
    } else {
      throw new Error('Invalid email format');
    }
  }
  
  getId() {
    return this.#id;
  }
  
  // Static private members
  static #instances = new Set();
  
  static #validateId(id) {
    return typeof id === 'number' && id > 0;
  }
  
  static create(id, name, email) {
    if (!this.#validateId(id)) {
      throw new Error('Invalid ID');
    }
    
    const user = new User(id, name, email);
    this.#instances.add(user);
    return user;
  }
}

const user = new User(1, 'John', 'john@example.com');
// console.log(user.#id); // SyntaxError: Private field '#id' must be declared in an enclosing class
```

### Top-Level Await:
```javascript
// In modules, you can now use await at the top level
// This only works in ES modules

// module.js
const data = await fetch('https://api.example.com/data').then(r => r.json());
console.log('Data loaded:', data);

// Dynamic module loading with top-level await
const config = await import('./config.js');
const database = await import('./database.js');

await database.connect(config.dbUrl);

export { data, config, database };

// Conditional imports
const isProduction = process.env.NODE_ENV === 'production';
const logger = isProduction 
  ? await import('./production-logger.js')
  : await import('./development-logger.js');

export default logger.default;

// Error handling with top-level await
let userData;
try {
  userData = await fetch('/api/user').then(r => r.json());
} catch (error) {
  console.error('Failed to load user data:', error);
  userData = { name: 'Guest', role: 'visitor' };
}

export { userData };

// Sequential vs parallel loading
// Sequential (slower)
const moduleA = await import('./module-a.js');
const moduleB = await import('./module-b.js');
const moduleC = await import('./module-c.js');

// Parallel (faster)
const [moduleA2, moduleB2, moduleC2] = await Promise.all([
  import('./module-a.js'),
  import('./module-b.js'),
  import('./module-c.js')
]);
```

### Advanced Module Patterns:
```javascript
// Re-exports
// utils/index.js
export { default as DateUtils } from './date-utils.js';
export { default as StringUtils } from './string-utils.js';
export { default as ArrayUtils } from './array-utils.js';
export * from './validators.js';

// Namespace imports
// main.js
import * as Utils from './utils/index.js';

const formatted = Utils.DateUtils.format(new Date());
const validated = Utils.validateEmail('test@example.com');

// Default exports with named exports
// logger.js
class Logger {
  constructor(level = 'info') {
    this.level = level;
  }
  
  log(message) {
    console.log(`[${this.level.toUpperCase()}] ${message}`);
  }
}

export const levels = ['debug', 'info', 'warn', 'error'];
export const createLogger = (level) => new Logger(level);
export default Logger;

// Import with both default and named
import Logger, { levels, createLogger } from './logger.js';

// Conditional exports in package.json
{
  "exports": {
    ".": {
      "import": "./esm/index.js",
      "require": "./cjs/index.js",
      "types": "./types/index.d.ts"
    },
    "./utils": {
      "import": "./esm/utils.js",
      "require": "./cjs/utils.js"
    },
    "./package.json": "./package.json"
  }
}

// Module federation (advanced pattern)
const ModuleFederation = {
  modules: new Map(),
  
  async register(name, moduleFactory) {
    this.modules.set(name, moduleFactory);
  },
  
  async load(name) {
    if (!this.modules.has(name)) {
      throw new Error(`Module ${name} not found`);
    }
    
    const moduleFactory = this.modules.get(name);
    
    if (typeof moduleFactory === 'function') {
      return await moduleFactory();
    }
    
    return moduleFactory;
  },
  
  async loadRemote(url, name) {
    try {
      const module = await import(url);
      this.modules.set(name, module);
      return module;
    } catch (error) {
      console.error(`Failed to load remote module ${name} from ${url}:`, error);
      throw error;
    }
  }
};

// Usage
await ModuleFederation.register('charts', () => import('./charts.js'));
await ModuleFederation.loadRemote('https://cdn.example.com/modules/widgets.js', 'widgets');

const charts = await ModuleFederation.load('charts');
const widgets = await ModuleFederation.load('widgets');
```

This comprehensive guide covers all major JavaScript concepts from basic to advanced, including:

- **Fundamentals**: Variables, data types, functions, control flow
- **ES6+ Features**: Classes, modules, destructuring, promises
- **Advanced Concepts**: Closures, prototypes, async programming, metaprogramming
- **Performance**: Memory management, optimization techniques
- **Design Patterns**: Creational, structural, and behavioral patterns
- **Modern APIs**: Fetch, WebSockets, Service Workers
- **Testing**: Unit tests, mocking, assertions
- **Latest Features**: Optional chaining, nullish coalescing, private fields, top-level await

Each section includes practical examples and real-world use cases to help you understand and apply these concepts effectively.
