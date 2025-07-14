# JavaScript Advanced Concepts - Part 2

## Table of Contents
1. [Memory Management](#memory-management)
2. [Performance Optimization](#performance-optimization)
3. [Design Patterns](#design-patterns)
4. [Metaprogramming](#metaprogramming)

## Memory Management

### Understanding Memory in JavaScript:
```javascript
// Stack vs Heap memory
// Primitives are stored in stack
let num = 42;           // Stack
let str = "hello";      // Stack (string literals)
let bool = true;        // Stack

// Objects are stored in heap, references in stack
let obj = { name: "John" };  // Reference in stack, object in heap
let arr = [1, 2, 3];         // Reference in stack, array in heap

// Understanding references
let obj1 = { value: 10 };
let obj2 = obj1;  // Same reference, not a copy

obj2.value = 20;
console.log(obj1.value); // 20 (both point to same object)

// Creating actual copies
let obj3 = { ...obj1 };          // Shallow copy
let obj4 = JSON.parse(JSON.stringify(obj1)); // Deep copy (limited)
```

### Garbage Collection:
```javascript
// Objects become eligible for GC when no references exist
function createObjects() {
  let obj1 = { data: new Array(1000000).fill('data') };
  let obj2 = { ref: obj1 };
  
  // obj1 and obj2 will be GC'd when function exits
  // (unless returned or stored elsewhere)
  
  return obj2; // obj2 is returned, so it won't be GC'd
}

let result = createObjects();
// obj1 is still referenced by obj2.ref, so it won't be GC'd
result = null; // Now both objects can be GC'd

// Circular references (handled by modern GC)
function createCircularReference() {
  let objA = {};
  let objB = {};
  
  objA.ref = objB;
  objB.ref = objA;
  
  // In old browsers, this would cause memory leaks
  // Modern GC can handle this
}

// WeakMap and WeakSet for weak references
const weakMap = new WeakMap();
const weakSet = new WeakSet();

let key = { id: 1 };
weakMap.set(key, "some value");
weakSet.add(key);

// When key is no longer referenced elsewhere, 
// it can be GC'd along with its WeakMap entry
key = null;
```

### Memory Leaks and Prevention:
```javascript
// Common memory leak patterns and fixes

// 1. Event listeners not removed
class ComponentWithLeak {
  constructor() {
    this.data = new Array(1000000).fill('data');
    
    // Memory leak: event listener keeps component alive
    document.addEventListener('click', this.handleClick.bind(this));
  }
  
  handleClick() {
    console.log('Component clicked');
  }
}

class ComponentFixed {
  constructor() {
    this.data = new Array(1000000).fill('data');
    this.boundHandleClick = this.handleClick.bind(this);
    
    document.addEventListener('click', this.boundHandleClick);
  }
  
  handleClick() {
    console.log('Component clicked');
  }
  
  destroy() {
    // Remove event listener to prevent memory leak
    document.removeEventListener('click', this.boundHandleClick);
    this.data = null;
  }
}

// 2. Timers not cleared
class TimerLeak {
  constructor() {
    this.data = new Array(1000000).fill('data');
    
    // Memory leak: timer keeps object alive
    setInterval(() => {
      console.log('Timer tick');
    }, 1000);
  }
}

class TimerFixed {
  constructor() {
    this.data = new Array(1000000).fill('data');
    
    this.timerId = setInterval(() => {
      console.log('Timer tick');
    }, 1000);
  }
  
  destroy() {
    clearInterval(this.timerId);
    this.data = null;
  }
}

// 3. Closures holding large objects
function createClosureLeak() {
  const largeData = new Array(1000000).fill('data');
  
  // This closure keeps largeData in memory
  return function() {
    console.log('Closure called');
    // Even though we don't use largeData, it's still referenced
  };
}

function createClosureFixed() {
  const largeData = new Array(1000000).fill('data');
  const neededValue = largeData.length; // Extract only what's needed
  
  return function() {
    console.log('Closure called, length was:', neededValue);
    // largeData can now be GC'd
  };
}

// 4. Detached DOM nodes
function createDetachedNodes() {
  const container = document.createElement('div');
  const children = [];
  
  // Create many child nodes
  for (let i = 0; i < 1000; i++) {
    const child = document.createElement('div');
    child.textContent = `Child ${i}`;
    container.appendChild(child);
    children.push(child); // Keeping references to children
  }
  
  // Remove container from DOM
  document.body.appendChild(container);
  document.body.removeChild(container);
  
  // children array still holds references to detached DOM nodes
  return children;
}

// Memory monitoring utilities
const memoryMonitor = {
  measureMemoryUsage(label, fn) {
    if (performance.memory) {
      const before = performance.memory.usedJSHeapSize;
      const result = fn();
      const after = performance.memory.usedJSHeapSize;
      
      console.log(`${label}: ${((after - before) / 1024 / 1024).toFixed(2)} MB`);
      return result;
    } else {
      console.log('Memory measurement not available');
      return fn();
    }
  },
  
  logMemoryInfo() {
    if (performance.memory) {
      const memory = performance.memory;
      console.log('Memory Info:', {
        used: `${(memory.usedJSHeapSize / 1024 / 1024).toFixed(2)} MB`,
        total: `${(memory.totalJSHeapSize / 1024 / 1024).toFixed(2)} MB`,
        limit: `${(memory.jsHeapSizeLimit / 1024 / 1024).toFixed(2)} MB`
      });
    }
  },
  
  forceGC() {
    // Only works in Node.js with --expose-gc flag
    if (global.gc) {
      global.gc();
      console.log('Garbage collection forced');
    } else {
      console.log('GC not available');
    }
  }
};
```

### Object Pooling:
```javascript
// Object pool to reduce GC pressure
class ObjectPool {
  constructor(createFn, resetFn, initialSize = 10) {
    this.createFn = createFn;
    this.resetFn = resetFn;
    this.pool = [];
    
    // Pre-populate pool
    for (let i = 0; i < initialSize; i++) {
      this.pool.push(this.createFn());
    }
  }
  
  acquire() {
    return this.pool.length > 0 ? this.pool.pop() : this.createFn();
  }
  
  release(obj) {
    this.resetFn(obj);
    this.pool.push(obj);
  }
  
  size() {
    return this.pool.length;
  }
}

// Example: Vector pool for game development
class Vector2D {
  constructor(x = 0, y = 0) {
    this.x = x;
    this.y = y;
  }
  
  set(x, y) {
    this.x = x;
    this.y = y;
    return this;
  }
  
  add(other) {
    this.x += other.x;
    this.y += other.y;
    return this;
  }
}

const vectorPool = new ObjectPool(
  () => new Vector2D(),
  (vector) => vector.set(0, 0),
  50
);

// Usage
function calculatePhysics() {
  const velocity = vectorPool.acquire().set(10, 5);
  const acceleration = vectorPool.acquire().set(0, -9.8);
  
  velocity.add(acceleration);
  
  // Do calculations...
  
  // Return objects to pool
  vectorPool.release(velocity);
  vectorPool.release(acceleration);
}

// Array pool for temporary arrays
class ArrayPool {
  constructor() {
    this.pools = new Map(); // Size -> Array of arrays
  }
  
  acquire(size) {
    if (!this.pools.has(size)) {
      this.pools.set(size, []);
    }
    
    const pool = this.pools.get(size);
    
    if (pool.length > 0) {
      return pool.pop();
    } else {
      return new Array(size);
    }
  }
  
  release(arr) {
    const size = arr.length;
    arr.fill(null); // Clear references
    
    if (!this.pools.has(size)) {
      this.pools.set(size, []);
    }
    
    this.pools.get(size).push(arr);
  }
}

const arrayPool = new ArrayPool();

function processData(dataSize) {
  const tempArray = arrayPool.acquire(dataSize);
  
  // Use tempArray...
  
  arrayPool.release(tempArray);
}
```

## Performance Optimization

### DOM Performance:
```javascript
// Efficient DOM manipulation
const domOptimizations = {
  // Batch DOM updates
  batchDOMUpdates(container, items) {
    // Bad: Multiple reflows
    items.forEach(item => {
      const div = document.createElement('div');
      div.textContent = item;
      container.appendChild(div); // Reflow on each append
    });
    
    // Good: Single reflow
    const fragment = document.createDocumentFragment();
    items.forEach(item => {
      const div = document.createElement('div');
      div.textContent = item;
      fragment.appendChild(div);
    });
    container.appendChild(fragment); // Single reflow
  },
  
  // Use efficient selectors
  efficientSelectors() {
    // Slow: Complex selector
    document.querySelectorAll('div.container > ul.list > li.item:nth-child(odd)');
    
    // Fast: ID selector
    document.getElementById('specific-element');
    
    // Fast: Class selector
    document.getElementsByClassName('item');
    
    // Cache selectors
    const container = document.getElementById('container');
    const items = container.getElementsByClassName('item');
  },
  
  // Minimize layout thrashing
  avoidLayoutThrashing(elements) {
    // Bad: Reading and writing properties alternately
    elements.forEach(el => {
      el.style.height = el.offsetHeight + 10 + 'px'; // Read then write
    });
    
    // Good: Batch reads and writes
    const heights = elements.map(el => el.offsetHeight); // Batch reads
    elements.forEach((el, i) => {
      el.style.height = heights[i] + 10 + 'px'; // Batch writes
    });
  },
  
  // Virtual scrolling for large lists
  createVirtualList(container, items, itemHeight, visibleCount) {
    const totalHeight = items.length * itemHeight;
    const viewport = document.createElement('div');
    viewport.style.height = `${visibleCount * itemHeight}px`;
    viewport.style.overflow = 'auto';
    
    const content = document.createElement('div');
    content.style.height = `${totalHeight}px`;
    content.style.position = 'relative';
    
    let startIndex = 0;
    
    function renderVisibleItems() {
      content.innerHTML = '';
      const endIndex = Math.min(startIndex + visibleCount, items.length);
      
      for (let i = startIndex; i < endIndex; i++) {
        const item = document.createElement('div');
        item.style.position = 'absolute';
        item.style.top = `${i * itemHeight}px`;
        item.style.height = `${itemHeight}px`;
        item.textContent = items[i];
        content.appendChild(item);
      }
    }
    
    viewport.addEventListener('scroll', () => {
      const newStartIndex = Math.floor(viewport.scrollTop / itemHeight);
      if (newStartIndex !== startIndex) {
        startIndex = newStartIndex;
        renderVisibleItems();
      }
    });
    
    viewport.appendChild(content);
    container.appendChild(viewport);
    renderVisibleItems();
  }
};
```

### Algorithm Optimization:
```javascript
// Optimization techniques
const algorithmOptimizations = {
  // Memoization
  memoize(fn) {
    const cache = new Map();
    
    return function(...args) {
      const key = JSON.stringify(args);
      
      if (cache.has(key)) {
        return cache.get(key);
      }
      
      const result = fn.apply(this, args);
      cache.set(key, result);
      return result;
    };
  },
  
  // Fibonacci with memoization
  fibonacciMemo: null,
  
  initFibonacci() {
    this.fibonacciMemo = this.memoize(function(n) {
      if (n <= 1) return n;
      return this.fibonacciMemo(n - 1) + this.fibonacciMemo(n - 2);
    });
  },
  
  // Optimized array operations
  arrayOptimizations: {
    // Fast array clearing
    clearArray(arr) {
      arr.length = 0; // Faster than arr.splice(0)
    },
    
    // Fast array copying
    copyArray(arr) {
      return arr.slice(); // Faster than [...arr] for large arrays
    },
    
    // Efficient array search
    binarySearch(sortedArray, target) {
      let left = 0;
      let right = sortedArray.length - 1;
      
      while (left <= right) {
        const mid = Math.floor((left + right) / 2);
        
        if (sortedArray[mid] === target) {
          return mid;
        } else if (sortedArray[mid] < target) {
          left = mid + 1;
        } else {
          right = mid - 1;
        }
      }
      
      return -1;
    },
    
    // Chunked processing for large arrays
    processInChunks(array, processor, chunkSize = 1000) {
      return new Promise((resolve) => {
        let index = 0;
        const results = [];
        
        function processChunk() {
          const chunk = array.slice(index, index + chunkSize);
          const chunkResults = chunk.map(processor);
          results.push(...chunkResults);
          
          index += chunkSize;
          
          if (index < array.length) {
            // Use setTimeout to prevent blocking
            setTimeout(processChunk, 0);
          } else {
            resolve(results);
          }
        }
        
        processChunk();
      });
    }
  },
  
  // Debouncing and throttling
  debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
      const later = () => {
        clearTimeout(timeout);
        func(...args);
      };
      clearTimeout(timeout);
      timeout = setTimeout(later, wait);
    };
  },
  
  throttle(func, limit) {
    let inThrottle;
    return function(...args) {
      if (!inThrottle) {
        func.apply(this, args);
        inThrottle = true;
        setTimeout(() => inThrottle = false, limit);
      }
    };
  },
  
  // Lazy evaluation
  createLazySequence(generator) {
    return {
      map(fn) {
        const self = this;
        return createLazySequence(function* () {
          for (const item of self) {
            yield fn(item);
          }
        });
      },
      
      filter(predicate) {
        const self = this;
        return createLazySequence(function* () {
          for (const item of self) {
            if (predicate(item)) {
              yield item;
            }
          }
        });
      },
      
      take(count) {
        const self = this;
        return createLazySequence(function* () {
          let taken = 0;
          for (const item of self) {
            if (taken >= count) break;
            yield item;
            taken++;
          }
        });
      },
      
      toArray() {
        return Array.from(this);
      },
      
      [Symbol.iterator]: generator
    };
  }
};

// Usage examples
algorithmOptimizations.initFibonacci();
console.log(algorithmOptimizations.fibonacciMemo(40)); // Fast due to memoization

// Lazy sequence example
const lazyNumbers = algorithmOptimizations.createLazySequence(function* () {
  let i = 0;
  while (true) {
    yield i++;
  }
});

const result = lazyNumbers
  .filter(x => x % 2 === 0)
  .map(x => x * x)
  .take(5)
  .toArray();

console.log(result); // [0, 4, 16, 36, 64]
```

### Performance Monitoring:
```javascript
// Performance measurement utilities
const performanceUtils = {
  // High-resolution timing
  measureTime(label, fn) {
    const start = performance.now();
    const result = fn();
    const end = performance.now();
    
    console.log(`${label}: ${(end - start).toFixed(2)}ms`);
    return result;
  },
  
  // Async function timing
  async measureAsyncTime(label, asyncFn) {
    const start = performance.now();
    const result = await asyncFn();
    const end = performance.now();
    
    console.log(`${label}: ${(end - start).toFixed(2)}ms`);
    return result;
  },
  
  // Memory usage tracking
  trackMemoryUsage(label) {
    if (performance.memory) {
      console.log(`${label} Memory:`, {
        used: `${(performance.memory.usedJSHeapSize / 1024 / 1024).toFixed(2)} MB`,
        total: `${(performance.memory.totalJSHeapSize / 1024 / 1024).toFixed(2)} MB`
      });
    }
  },
  
  // Performance observer
  observePerformance() {
    if ('PerformanceObserver' in window) {
      const observer = new PerformanceObserver((list) => {
        list.getEntries().forEach((entry) => {
          console.log(`${entry.name}: ${entry.duration.toFixed(2)}ms`);
        });
      });
      
      observer.observe({ entryTypes: ['measure', 'navigation', 'paint'] });
    }
  },
  
  // Benchmark comparison
  benchmark(functions, iterations = 1000) {
    const results = {};
    
    for (const [name, fn] of Object.entries(functions)) {
      const start = performance.now();
      
      for (let i = 0; i < iterations; i++) {
        fn();
      }
      
      const end = performance.now();
      results[name] = {
        totalTime: end - start,
        avgTime: (end - start) / iterations,
        opsPerSecond: iterations / ((end - start) / 1000)
      };
    }
    
    // Sort by performance
    const sorted = Object.entries(results)
      .sort(([, a], [, b]) => a.totalTime - b.totalTime);
    
    console.log('Benchmark Results:');
    sorted.forEach(([name, stats], index) => {
      console.log(`${index + 1}. ${name}:`);
      console.log(`   Total: ${stats.totalTime.toFixed(2)}ms`);
      console.log(`   Avg: ${stats.avgTime.toFixed(4)}ms`);
      console.log(`   Ops/sec: ${stats.opsPerSecond.toFixed(0)}`);
    });
    
    return results;
  },
  
  // Frame rate monitoring
  monitorFPS(callback, duration = 5000) {
    let frameCount = 0;
    let startTime = performance.now();
    let lastTime = startTime;
    
    function countFrame() {
      frameCount++;
      const currentTime = performance.now();
      
      if (currentTime - startTime >= duration) {
        const fps = (frameCount * 1000) / (currentTime - startTime);
        callback(fps);
        return;
      }
      
      requestAnimationFrame(countFrame);
    }
    
    requestAnimationFrame(countFrame);
  }
};

// Usage example
performanceUtils.benchmark({
  forLoop: () => {
    let sum = 0;
    for (let i = 0; i < 1000; i++) {
      sum += i;
    }
    return sum;
  },
  
  reduce: () => {
    return Array.from({ length: 1000 }, (_, i) => i)
      .reduce((sum, i) => sum + i, 0);
  },
  
  whileLoop: () => {
    let sum = 0;
    let i = 0;
    while (i < 1000) {
      sum += i;
      i++;
    }
    return sum;
  }
});
```

## Design Patterns

### Creational Patterns:

#### Singleton Pattern:
```javascript
// Singleton with ES6 classes
class Singleton {
  constructor() {
    if (Singleton.instance) {
      return Singleton.instance;
    }
    
    this.data = {};
    this.timestamp = Date.now();
    Singleton.instance = this;
  }
  
  getData() {
    return this.data;
  }
  
  setData(key, value) {
    this.data[key] = value;
  }
  
  static getInstance() {
    if (!Singleton.instance) {
      Singleton.instance = new Singleton();
    }
    return Singleton.instance;
  }
}

// Usage
const instance1 = new Singleton();
const instance2 = new Singleton();
const instance3 = Singleton.getInstance();

console.log(instance1 === instance2); // true
console.log(instance1 === instance3); // true

// Module-based Singleton
const ConfigManager = (function() {
  let instance;
  
  function createInstance() {
    return {
      config: {},
      
      get(key) {
        return this.config[key];
      },
      
      set(key, value) {
        this.config[key] = value;
      },
      
      getAll() {
        return { ...this.config };
      }
    };
  }
  
  return {
    getInstance() {
      if (!instance) {
        instance = createInstance();
      }
      return instance;
    }
  };
})();

// Lazy Singleton
class LazyDatabase {
  static getInstance() {
    if (!LazyDatabase._instance) {
      LazyDatabase._instance = new LazyDatabase();
    }
    return LazyDatabase._instance;
  }
  
  constructor() {
    if (LazyDatabase._instance) {
      throw new Error('Use LazyDatabase.getInstance()');
    }
    
    this.connection = null;
    this.connected = false;
  }
  
  connect() {
    if (!this.connected) {
      console.log('Connecting to database...');
      this.connection = { id: Math.random() };
      this.connected = true;
    }
    return this.connection;
  }
  
  query(sql) {
    if (!this.connected) {
      this.connect();
    }
    console.log(`Executing: ${sql}`);
    return `Result for: ${sql}`;
  }
}
```

#### Factory Pattern:
```javascript
// Simple Factory
class VehicleFactory {
  static createVehicle(type, options) {
    switch (type.toLowerCase()) {
      case 'car':
        return new Car(options);
      case 'truck':
        return new Truck(options);
      case 'motorcycle':
        return new Motorcycle(options);
      default:
        throw new Error(`Unknown vehicle type: ${type}`);
    }
  }
}

class Vehicle {
  constructor(options) {
    this.make = options.make;
    this.model = options.model;
    this.year = options.year;
  }
  
  start() {
    return `${this.make} ${this.model} is starting`;
  }
}

class Car extends Vehicle {
  constructor(options) {
    super(options);
    this.doors = options.doors || 4;
    this.type = 'car';
  }
}

class Truck extends Vehicle {
  constructor(options) {
    super(options);
    this.cargoCapacity = options.cargoCapacity || 1000;
    this.type = 'truck';
  }
}

class Motorcycle extends Vehicle {
  constructor(options) {
    super(options);
    this.engineSize = options.engineSize || 500;
    this.type = 'motorcycle';
  }
}

// Usage
const car = VehicleFactory.createVehicle('car', {
  make: 'Toyota',
  model: 'Camry',
  year: 2023,
  doors: 4
});

// Abstract Factory
class UIFactory {
  createButton() {
    throw new Error('createButton() must be implemented');
  }
  
  createInput() {
    throw new Error('createInput() must be implemented');
  }
}

class WindowsUIFactory extends UIFactory {
  createButton() {
    return new WindowsButton();
  }
  
  createInput() {
    return new WindowsInput();
  }
}

class MacUIFactory extends UIFactory {
  createButton() {
    return new MacButton();
  }
  
  createInput() {
    return new MacInput();
  }
}

class UIElement {
  render() {
    throw new Error('render() must be implemented');
  }
}

class WindowsButton extends UIElement {
  render() {
    return '<button class="windows-btn">Windows Button</button>';
  }
}

class MacButton extends UIElement {
  render() {
    return '<button class="mac-btn">Mac Button</button>';
  }
}

class WindowsInput extends UIElement {
  render() {
    return '<input class="windows-input" type="text">';
  }
}

class MacInput extends UIElement {
  render() {
    return '<input class="mac-input" type="text">';
  }
}

// Factory method for UI creation
function createUIFactory(platform) {
  switch (platform) {
    case 'windows':
      return new WindowsUIFactory();
    case 'mac':
      return new MacUIFactory();
    default:
      throw new Error(`Unsupported platform: ${platform}`);
  }
}
```

#### Builder Pattern:
```javascript
// Complex object construction
class QueryBuilder {
  constructor() {
    this.query = {
      select: [],
      from: '',
      joins: [],
      where: [],
      groupBy: [],
      having: [],
      orderBy: [],
      limit: null
    };
  }
  
  select(fields) {
    if (Array.isArray(fields)) {
      this.query.select.push(...fields);
    } else {
      this.query.select.push(fields);
    }
    return this;
  }
  
  from(table) {
    this.query.from = table;
    return this;
  }
  
  join(table, condition) {
    this.query.joins.push({ type: 'JOIN', table, condition });
    return this;
  }
  
  leftJoin(table, condition) {
    this.query.joins.push({ type: 'LEFT JOIN', table, condition });
    return this;
  }
  
  where(condition) {
    this.query.where.push(condition);
    return this;
  }
  
  groupBy(fields) {
    if (Array.isArray(fields)) {
      this.query.groupBy.push(...fields);
    } else {
      this.query.groupBy.push(fields);
    }
    return this;
  }
  
  orderBy(field, direction = 'ASC') {
    this.query.orderBy.push({ field, direction });
    return this;
  }
  
  limit(count) {
    this.query.limit = count;
    return this;
  }
  
  build() {
    let sql = '';
    
    // SELECT
    sql += `SELECT ${this.query.select.join(', ')}`;
    
    // FROM
    sql += ` FROM ${this.query.from}`;
    
    // JOINS
    this.query.joins.forEach(join => {
      sql += ` ${join.type} ${join.table} ON ${join.condition}`;
    });
    
    // WHERE
    if (this.query.where.length > 0) {
      sql += ` WHERE ${this.query.where.join(' AND ')}`;
    }
    
    // GROUP BY
    if (this.query.groupBy.length > 0) {
      sql += ` GROUP BY ${this.query.groupBy.join(', ')}`;
    }
    
    // ORDER BY
    if (this.query.orderBy.length > 0) {
      const orderClauses = this.query.orderBy.map(
        order => `${order.field} ${order.direction}`
      );
      sql += ` ORDER BY ${orderClauses.join(', ')}`;
    }
    
    // LIMIT
    if (this.query.limit !== null) {
      sql += ` LIMIT ${this.query.limit}`;
    }
    
    return sql;
  }
}

// Usage
const query = new QueryBuilder()
  .select(['users.name', 'posts.title'])
  .from('users')
  .leftJoin('posts', 'users.id = posts.user_id')
  .where('users.active = 1')
  .where('posts.published = 1')
  .orderBy('posts.created_at', 'DESC')
  .limit(10)
  .build();

console.log(query);

// Director pattern with Builder
class ComputerBuilder {
  constructor() {
    this.computer = {};
  }
  
  setCPU(cpu) {
    this.computer.cpu = cpu;
    return this;
  }
  
  setRAM(ram) {
    this.computer.ram = ram;
    return this;
  }
  
  setStorage(storage) {
    this.computer.storage = storage;
    return this;
  }
  
  setGPU(gpu) {
    this.computer.gpu = gpu;
    return this;
  }
  
  build() {
    return { ...this.computer };
  }
}

class ComputerDirector {
  static buildGamingComputer(builder) {
    return builder
      .setCPU('Intel i9-11900K')
      .setRAM('32GB DDR4')
      .setStorage('1TB NVMe SSD')
      .setGPU('RTX 3080')
      .build();
  }
  
  static buildOfficeComputer(builder) {
    return builder
      .setCPU('Intel i5-11400')
      .setRAM('16GB DDR4')
      .setStorage('512GB SSD')
      .setGPU('Integrated Graphics')
      .build();
  }
}

const gamingPC = ComputerDirector.buildGamingComputer(new ComputerBuilder());
const officePC = ComputerDirector.buildOfficeComputer(new ComputerBuilder());
```

### Structural Patterns:

#### Adapter Pattern:
```javascript
// Adapting incompatible interfaces
class OldPrinter {
  print(text) {
    console.log(`Old Printer: ${text}`);
  }
}

class NewPrinter {
  printDocument(document) {
    console.log(`New Printer: ${document.content}`);
  }
}

// Adapter to make OldPrinter work with new interface
class PrinterAdapter {
  constructor(oldPrinter) {
    this.oldPrinter = oldPrinter;
  }
  
  printDocument(document) {
    this.oldPrinter.print(document.content);
  }
}

// Client code expects new interface
function printWithNewInterface(printer, document) {
  printer.printDocument(document);
}

const oldPrinter = new OldPrinter();
const adapter = new PrinterAdapter(oldPrinter);
const newPrinter = new NewPrinter();

const document = { content: 'Hello World', pages: 1 };

printWithNewInterface(adapter, document);   // Works with adapter
printWithNewInterface(newPrinter, document); // Works directly

// API Adapter example
class WeatherAPIv1 {
  getTemperature(city) {
    // Old API returns temperature in Fahrenheit
    return {
      city: city,
      temp: 72,
      unit: 'F'
    };
  }
}

class WeatherAPIv2 {
  getCurrentWeather(location) {
    // New API returns temperature in Celsius
    return {
      location: location,
      temperature: 22,
      unit: 'C',
      humidity: 60,
      pressure: 1013
    };
  }
}

class WeatherAdapter {
  constructor(apiVersion) {
    this.api = apiVersion === 1 ? new WeatherAPIv1() : new WeatherAPIv2();
    this.version = apiVersion;
  }
  
  getWeatherData(location) {
    if (this.version === 1) {
      const data = this.api.getTemperature(location);
      // Convert to common format
      return {
        location: data.city,
        temperature: Math.round((data.temp - 32) * 5/9), // F to C
        unit: 'C',
        humidity: null,
        pressure: null
      };
    } else {
      return this.api.getCurrentWeather(location);
    }
  }
}
```

#### Decorator Pattern:
```javascript
// Component interface
class Coffee {
  cost() {
    return 5;
  }
  
  description() {
    return 'Simple coffee';
  }
}

// Decorator base class
class CoffeeDecorator {
  constructor(coffee) {
    this.coffee = coffee;
  }
  
  cost() {
    return this.coffee.cost();
  }
  
  description() {
    return this.coffee.description();
  }
}

// Concrete decorators
class MilkDecorator extends CoffeeDecorator {
  cost() {
    return this.coffee.cost() + 1;
  }
  
  description() {
    return this.coffee.description() + ', milk';
  }
}

class SugarDecorator extends CoffeeDecorator {
  cost() {
    return this.coffee.cost() + 0.5;
  }
  
  description() {
    return this.coffee.description() + ', sugar';
  }
}

class WhippedCreamDecorator extends CoffeeDecorator {
  cost() {
    return this.coffee.cost() + 2;
  }
  
  description() {
    return this.coffee.description() + ', whipped cream';
  }
}

// Usage
let coffee = new Coffee();
console.log(`${coffee.description()}: $${coffee.cost()}`);

coffee = new MilkDecorator(coffee);
console.log(`${coffee.description()}: $${coffee.cost()}`);

coffee = new SugarDecorator(coffee);
console.log(`${coffee.description()}: $${coffee.cost()}`);

coffee = new WhippedCreamDecorator(coffee);
console.log(`${coffee.description()}: $${coffee.cost()}`);

// Function decorators
function timeExecution(target, propertyKey, descriptor) {
  const originalMethod = descriptor.value;
  
  descriptor.value = function(...args) {
    const start = performance.now();
    const result = originalMethod.apply(this, args);
    const end = performance.now();
    
    console.log(`${propertyKey} took ${(end - start).toFixed(2)}ms`);
    return result;
  };
  
  return descriptor;
}

function cache(target, propertyKey, descriptor) {
  const originalMethod = descriptor.value;
  const cacheMap = new Map();
  
  descriptor.value = function(...args) {
    const key = JSON.stringify(args);
    
    if (cacheMap.has(key)) {
      console.log(`Cache hit for ${propertyKey}`);
      return cacheMap.get(key);
    }
    
    const result = originalMethod.apply(this, args);
    cacheMap.set(key, result);
    return result;
  };
  
  return descriptor;
}

class Calculator {
  @timeExecution
  @cache
  fibonacci(n) {
    if (n <= 1) return n;
    return this.fibonacci(n - 1) + this.fibonacci(n - 2);
  }
}
```

#### Facade Pattern:
```javascript
// Complex subsystem
class CPU {
  freeze() { console.log('CPU: Freezing...'); }
  jump(position) { console.log(`CPU: Jumping to ${position}`); }
  execute() { console.log('CPU: Executing...'); }
}

class Memory {
  load(position, data) {
    console.log(`Memory: Loading data at ${position}`);
  }
}

class HardDrive {
  read(lba, size) {
    console.log(`HardDrive: Reading ${size} bytes from ${lba}`);
    return 'boot data';
  }
}

// Facade
class ComputerFacade {
  constructor() {
    this.cpu = new CPU();
    this.memory = new Memory();
    this.hardDrive = new HardDrive();
  }
  
  start() {
    console.log('Computer starting...');
    this.cpu.freeze();
    this.memory.load(0, this.hardDrive.read(0, 1024));
    this.cpu.jump(0);
    this.cpu.execute();
    console.log('Computer started!');
  }
}

// Simple interface for complex operation
const computer = new ComputerFacade();
computer.start(); // One simple call handles complex boot process

// API Facade example
class DatabaseFacade {
  constructor() {
    this.connection = null;
    this.queryBuilder = null;
    this.cache = new Map();
  }
  
  async findUser(id) {
    const cacheKey = `user:${id}`;
    
    if (this.cache.has(cacheKey)) {
      return this.cache.get(cacheKey);
    }
    
    // Complex database operations hidden behind simple interface
    await this.connect();
    const query = this.buildUserQuery(id);
    const result = await this.executeQuery(query);
    const user = this.transformResult(result);
    
    this.cache.set(cacheKey, user);
    return user;
  }
  
  async connect() {
    if (!this.connection) {
      console.log('Connecting to database...');
      // Complex connection logic
    }
  }
  
  buildUserQuery(id) {
    return `SELECT * FROM users WHERE id = ${id}`;
  }
  
  async executeQuery(query) {
    console.log(`Executing: ${query}`);
    return { id: 1, name: 'John', email: 'john@example.com' };
  }
  
  transformResult(result) {
    return {
      ...result,
      fullName: result.name,
      emailDomain: result.email.split('@')[1]
    };
  }
}
```

### Behavioral Patterns:

#### Observer Pattern:
```javascript
// Subject (Observable)
class EventEmitter {
  constructor() {
    this.events = {};
  }
  
  on(event, callback) {
    if (!this.events[event]) {
      this.events[event] = [];
    }
    this.events[event].push(callback);
    
    // Return unsubscribe function
    return () => {
      this.events[event] = this.events[event].filter(cb => cb !== callback);
    };
  }
  
  off(event, callback) {
    if (this.events[event]) {
      this.events[event] = this.events[event].filter(cb => cb !== callback);
    }
  }
  
  emit(event, data) {
    if (this.events[event]) {
      this.events[event].forEach(callback => callback(data));
    }
  }
  
  once(event, callback) {
    const unsubscribe = this.on(event, (data) => {
      callback(data);
      unsubscribe();
    });
  }
}

// Usage
const emitter = new EventEmitter();

const unsubscribe1 = emitter.on('userLogin', (user) => {
  console.log(`User ${user.name} logged in`);
});

const unsubscribe2 = emitter.on('userLogin', (user) => {
  console.log(`Welcome ${user.name}!`);
});

emitter.emit('userLogin', { name: 'John', id: 1 });

// Model-View pattern with Observer
class Model extends EventEmitter {
  constructor() {
    super();
    this.data = {};
  }
  
  set(key, value) {
    const oldValue = this.data[key];
    this.data[key] = value;
    
    this.emit('change', { key, value, oldValue });
    this.emit(`change:${key}`, { value, oldValue });
  }
  
  get(key) {
    return this.data[key];
  }
}

class View {
  constructor(model) {
    this.model = model;
    this.element = document.createElement('div');
    
    // Listen to model changes
    this.model.on('change', this.render.bind(this));
  }
  
  render(changeData) {
    console.log('View updated:', changeData);
    this.element.innerHTML = JSON.stringify(this.model.data, null, 2);
  }
}

const model = new Model();
const view = new View(model);

model.set('name', 'John');
model.set('age', 30);
```

#### Strategy Pattern:
```javascript
// Strategy interface
class SortStrategy {
  sort(data) {
    throw new Error('sort() must be implemented');
  }
}

// Concrete strategies
class BubbleSortStrategy extends SortStrategy {
  sort(data) {
    console.log('Using Bubble Sort');
    const arr = [...data];
    
    for (let i = 0; i < arr.length; i++) {
      for (let j = 0; j < arr.length - i - 1; j++) {
        if (arr[j] > arr[j + 1]) {
          [arr[j], arr[j + 1]] = [arr[j + 1], arr[j]];
        }
      }
    }
    
    return arr;
  }
}

class QuickSortStrategy extends SortStrategy {
  sort(data) {
    console.log('Using Quick Sort');
    return this.quickSort([...data]);
  }
  
  quickSort(arr) {
    if (arr.length <= 1) return arr;
    
    const pivot = arr[Math.floor(arr.length / 2)];
    const left = arr.filter(x => x < pivot);
    const middle = arr.filter(x => x === pivot);
    const right = arr.filter(x => x > pivot);
    
    return [...this.quickSort(left), ...middle, ...this.quickSort(right)];
  }
}

class MergeSortStrategy extends SortStrategy {
  sort(data) {
    console.log('Using Merge Sort');
    return this.mergeSort([...data]);
  }
  
  mergeSort(arr) {
    if (arr.length <= 1) return arr;
    
    const mid = Math.floor(arr.length / 2);
    const left = this.mergeSort(arr.slice(0, mid));
    const right = this.mergeSort(arr.slice(mid));
    
    return this.merge(left, right);
  }
  
  merge(left, right) {
    const result = [];
    let leftIndex = 0;
    let rightIndex = 0;
    
    while (leftIndex < left.length && rightIndex < right.length) {
      if (left[leftIndex] < right[rightIndex]) {
        result.push(left[leftIndex]);
        leftIndex++;
      } else {
        result.push(right[rightIndex]);
        rightIndex++;
      }
    }
    
    return result.concat(left.slice(leftIndex)).concat(right.slice(rightIndex));
  }
}

// Context
class Sorter {
  constructor(strategy) {
    this.strategy = strategy;
  }
  
  setStrategy(strategy) {
    this.strategy = strategy;
  }
  
  sort(data) {
    return this.strategy.sort(data);
  }
}

// Usage
const data = [64, 34, 25, 12, 22, 11, 90];

const sorter = new Sorter(new BubbleSortStrategy());
console.log(sorter.sort(data));

sorter.setStrategy(new QuickSortStrategy());
console.log(sorter.sort(data));

sorter.setStrategy(new MergeSortStrategy());
console.log(sorter.sort(data));

// Validation strategy example
class ValidationStrategy {
  validate(value) {
    throw new Error('validate() must be implemented');
  }
}

class EmailValidationStrategy extends ValidationStrategy {
  validate(email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return {
      isValid: emailRegex.test(email),
      message: emailRegex.test(email) ? '' : 'Invalid email format'
    };
  }
}

class PasswordValidationStrategy extends ValidationStrategy {
  validate(password) {
    const isValid = password.length >= 8 && /[A-Z]/.test(password) && /[0-9]/.test(password);
    return {
      isValid,
      message: isValid ? '' : 'Password must be at least 8 chars with uppercase and number'
    };
  }
}

class Validator {
  constructor() {
    this.strategies = new Map();
  }
  
  addStrategy(field, strategy) {
    this.strategies.set(field, strategy);
  }
  
  validate(data) {
    const results = {};
    
    for (const [field, strategy] of this.strategies) {
      if (data.hasOwnProperty(field)) {
        results[field] = strategy.validate(data[field]);
      }
    }
    
    return results;
  }
}

const validator = new Validator();
validator.addStrategy('email', new EmailValidationStrategy());
validator.addStrategy('password', new PasswordValidationStrategy());

const formData = {
  email: 'user@example.com',
  password: 'weak'
};

console.log(validator.validate(formData));
```

This completes the second part of advanced JavaScript concepts, covering memory management, performance optimization, and design patterns. The content provides practical examples and real-world applications of these advanced concepts.

---

**Remaining sections for the next file:**
- Metaprogramming (Proxies, Reflect, Symbols)
- Web APIs (Fetch, WebSockets, Service Workers, etc.)
- Testing (Unit tests, Integration tests, Mocking)
- Advanced Browser APIs and Modern JavaScript Features
