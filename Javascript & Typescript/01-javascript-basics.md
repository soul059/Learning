# JavaScript Basics

## Table of Contents
1. [Introduction to JavaScript](#introduction-to-javascript)
2. [Variables and Data Types](#variables-and-data-types)
3. [Operators](#operators)
4. [Control Flow](#control-flow)
5. [Functions](#functions)
6. [Arrays](#arrays)
7. [Objects](#objects)
8. [Strings](#strings)
9. [Type Conversion](#type-conversion)
10. [Scope and Hoisting](#scope-and-hoisting)

## Introduction to JavaScript

JavaScript is a high-level, interpreted programming language that is one of the core technologies of the World Wide Web. It's used for:
- Client-side web development (frontend)
- Server-side development (Node.js)
- Mobile app development
- Desktop applications

### Key Features:
- **Dynamic typing**: Variables can hold different types of values
- **Interpreted language**: No compilation step required
- **First-class functions**: Functions are treated as values
- **Event-driven programming**: Responds to user interactions

### Adding JavaScript to HTML:
```javascript
// Inline JavaScript
<script>
  console.log("Hello, World!");
</script>

// External JavaScript file
<script src="script.js"></script>
```

## Variables and Data Types

### Variable Declarations:
```javascript
// var (function-scoped, can be redeclared)
var name = "John";

// let (block-scoped, cannot be redeclared)
let age = 25;

// const (block-scoped, cannot be reassigned)
const PI = 3.14159;
```

### Primitive Data Types:

#### 1. Number
```javascript
let integer = 42;
let float = 3.14;
let scientific = 2.5e6; // 2,500,000
let infinity = Infinity;
let notANumber = NaN;

// Number methods
console.log(Number.isInteger(42)); // true
console.log(Number.parseFloat("3.14")); // 3.14
```

#### 2. String
```javascript
let singleQuotes = 'Hello';
let doubleQuotes = "World";
let templateLiteral = `Hello, ${name}!`;

// String methods
console.log("Hello".length); // 5
console.log("Hello".toUpperCase()); // "HELLO"
console.log("Hello".charAt(0)); // "H"
```

#### 3. Boolean
```javascript
let isTrue = true;
let isFalse = false;

// Boolean conversion
console.log(Boolean(1)); // true
console.log(Boolean(0)); // false
console.log(Boolean("")); // false
```

#### 4. Undefined
```javascript
let undefinedVar;
console.log(undefinedVar); // undefined
```

#### 5. Null
```javascript
let nullVar = null;
console.log(nullVar); // null
```

#### 6. Symbol (ES6)
```javascript
let sym1 = Symbol('description');
let sym2 = Symbol('description');
console.log(sym1 === sym2); // false (symbols are unique)
```

#### 7. BigInt (ES2020)
```javascript
let bigNumber = 123456789012345678901234567890n;
let anotherBig = BigInt("123456789012345678901234567890");
```

### Non-Primitive Data Types:
```javascript
// Object
let person = {
  name: "John",
  age: 30
};

// Array
let numbers = [1, 2, 3, 4, 5];

// Function
function greet() {
  return "Hello!";
}
```

## Operators

### Arithmetic Operators:
```javascript
let a = 10, b = 3;

console.log(a + b); // 13 (addition)
console.log(a - b); // 7 (subtraction)
console.log(a * b); // 30 (multiplication)
console.log(a / b); // 3.333... (division)
console.log(a % b); // 1 (modulus/remainder)
console.log(a ** b); // 1000 (exponentiation)

// Increment/Decrement
let x = 5;
console.log(++x); // 6 (pre-increment)
console.log(x++); // 6 (post-increment)
console.log(x); // 7
```

### Assignment Operators:
```javascript
let x = 10;
x += 5; // x = x + 5 (15)
x -= 3; // x = x - 3 (12)
x *= 2; // x = x * 2 (24)
x /= 4; // x = x / 4 (6)
x %= 4; // x = x % 4 (2)
```

### Comparison Operators:
```javascript
console.log(5 == "5"); // true (loose equality)
console.log(5 === "5"); // false (strict equality)
console.log(5 != "5"); // false (loose inequality)
console.log(5 !== "5"); // true (strict inequality)
console.log(5 > 3); // true
console.log(5 < 3); // false
console.log(5 >= 5); // true
console.log(5 <= 4); // false
```

### Logical Operators:
```javascript
let a = true, b = false;

console.log(a && b); // false (AND)
console.log(a || b); // true (OR)
console.log(!a); // false (NOT)

// Short-circuit evaluation
console.log(false && someFunction()); // someFunction() not called
console.log(true || someFunction()); // someFunction() not called
```

## Control Flow

### Conditional Statements:

#### if...else
```javascript
let age = 18;

if (age >= 18) {
  console.log("You are an adult");
} else if (age >= 13) {
  console.log("You are a teenager");
} else {
  console.log("You are a child");
}
```

#### Ternary Operator
```javascript
let status = age >= 18 ? "adult" : "minor";
console.log(status);
```

#### switch Statement
```javascript
let day = "Monday";

switch (day) {
  case "Monday":
    console.log("Start of work week");
    break;
  case "Friday":
    console.log("TGIF!");
    break;
  case "Saturday":
  case "Sunday":
    console.log("Weekend!");
    break;
  default:
    console.log("Regular day");
}
```

### Loops:

#### for Loop
```javascript
// Basic for loop
for (let i = 0; i < 5; i++) {
  console.log(i);
}

// for...in (iterates over object properties)
let person = { name: "John", age: 30 };
for (let key in person) {
  console.log(key + ": " + person[key]);
}

// for...of (iterates over iterable values)
let arr = [1, 2, 3, 4, 5];
for (let value of arr) {
  console.log(value);
}
```

#### while Loop
```javascript
let i = 0;
while (i < 5) {
  console.log(i);
  i++;
}
```

#### do...while Loop
```javascript
let j = 0;
do {
  console.log(j);
  j++;
} while (j < 5);
```

### Loop Control:
```javascript
for (let i = 0; i < 10; i++) {
  if (i === 3) continue; // Skip iteration
  if (i === 7) break; // Exit loop
  console.log(i);
}
```

## Functions

### Function Declaration:
```javascript
function greet(name) {
  return `Hello, ${name}!`;
}

console.log(greet("John")); // "Hello, John!"
```

### Function Expression:
```javascript
let greet = function(name) {
  return `Hello, ${name}!`;
};
```

### Arrow Functions (ES6):
```javascript
// Basic arrow function
let greet = (name) => {
  return `Hello, ${name}!`;
};

// Shortened syntax
let greet2 = name => `Hello, ${name}!`;

// Multiple parameters
let add = (a, b) => a + b;

// No parameters
let sayHello = () => "Hello!";
```

### Function Parameters:

#### Default Parameters:
```javascript
function greet(name = "World") {
  return `Hello, ${name}!`;
}

console.log(greet()); // "Hello, World!"
console.log(greet("John")); // "Hello, John!"
```

#### Rest Parameters:
```javascript
function sum(...numbers) {
  return numbers.reduce((total, num) => total + num, 0);
}

console.log(sum(1, 2, 3, 4)); // 10
```

### Return Statement:
```javascript
function multiply(a, b) {
  return a * b; // Function ends here
  console.log("This won't execute");
}

// Function without return statement returns undefined
function noReturn() {
  console.log("No return value");
}

console.log(noReturn()); // undefined
```

## Arrays

### Creating Arrays:
```javascript
// Array literal
let fruits = ["apple", "banana", "orange"];

// Array constructor
let numbers = new Array(1, 2, 3, 4, 5);

// Empty array with specific length
let empty = new Array(5); // [undefined, undefined, undefined, undefined, undefined]
```

### Array Properties and Methods:

#### Length Property:
```javascript
let arr = [1, 2, 3, 4, 5];
console.log(arr.length); // 5

// Changing length
arr.length = 3;
console.log(arr); // [1, 2, 3]
```

#### Adding/Removing Elements:
```javascript
let fruits = ["apple", "banana"];

// Add to end
fruits.push("orange"); // ["apple", "banana", "orange"]

// Remove from end
let last = fruits.pop(); // "orange"

// Add to beginning
fruits.unshift("grape"); // ["grape", "apple", "banana"]

// Remove from beginning
let first = fruits.shift(); // "grape"
```

#### Array Methods:
```javascript
let numbers = [1, 2, 3, 4, 5];

// indexOf and includes
console.log(numbers.indexOf(3)); // 2
console.log(numbers.includes(4)); // true

// slice (doesn't modify original)
let sliced = numbers.slice(1, 4); // [2, 3, 4]

// splice (modifies original)
numbers.splice(2, 1, "new"); // [1, 2, "new", 4, 5]

// join
console.log(numbers.join("-")); // "1-2-new-4-5"

// reverse
numbers.reverse(); // [5, 4, "new", 2, 1]

// sort
let names = ["Charlie", "Alice", "Bob"];
names.sort(); // ["Alice", "Bob", "Charlie"]
```

### Multidimensional Arrays:
```javascript
let matrix = [
  [1, 2, 3],
  [4, 5, 6],
  [7, 8, 9]
];

console.log(matrix[1][2]); // 6
```

## Objects

### Creating Objects:

#### Object Literal:
```javascript
let person = {
  name: "John",
  age: 30,
  city: "New York",
  isEmployed: true
};
```

#### Object Constructor:
```javascript
let person = new Object();
person.name = "John";
person.age = 30;
```

### Accessing Object Properties:

#### Dot Notation:
```javascript
console.log(person.name); // "John"
person.age = 31;
```

#### Bracket Notation:
```javascript
console.log(person["name"]); // "John"
person["age"] = 31;

// Useful for dynamic property names
let property = "name";
console.log(person[property]); // "John"
```

### Object Methods:
```javascript
let person = {
  name: "John",
  age: 30,
  greet: function() {
    return `Hello, I'm ${this.name}`;
  },
  // ES6 method shorthand
  sayGoodbye() {
    return `Goodbye from ${this.name}`;
  }
};

console.log(person.greet()); // "Hello, I'm John"
```

### Object Properties:

#### Adding/Deleting Properties:
```javascript
let obj = { a: 1, b: 2 };

// Add property
obj.c = 3;
obj["d"] = 4;

// Delete property
delete obj.b;
console.log(obj); // { a: 1, c: 3, d: 4 }
```

#### Checking Property Existence:
```javascript
let obj = { name: "John", age: 30 };

console.log("name" in obj); // true
console.log(obj.hasOwnProperty("age")); // true
console.log(obj.city !== undefined); // false
```

### Object Iteration:
```javascript
let person = { name: "John", age: 30, city: "NYC" };

// for...in loop
for (let key in person) {
  console.log(key + ": " + person[key]);
}

// Object methods
console.log(Object.keys(person)); // ["name", "age", "city"]
console.log(Object.values(person)); // ["John", 30, "NYC"]
console.log(Object.entries(person)); // [["name", "John"], ["age", 30], ["city", "NYC"]]
```

## Strings

### String Creation:
```javascript
let str1 = "Hello World";
let str2 = 'Hello World';
let str3 = `Hello World`;

// String constructor
let str4 = new String("Hello World");
```

### Template Literals (ES6):
```javascript
let name = "John";
let age = 30;

let message = `Hello, my name is ${name} and I am ${age} years old.`;
console.log(message);

// Multiline strings
let multiline = `
  This is a
  multiline
  string
`;
```

### String Properties and Methods:

#### Length:
```javascript
let str = "Hello World";
console.log(str.length); // 11
```

#### Character Access:
```javascript
let str = "Hello";
console.log(str[0]); // "H"
console.log(str.charAt(1)); // "e"
console.log(str.charCodeAt(0)); // 72 (ASCII code)
```

#### String Search:
```javascript
let str = "Hello World";

console.log(str.indexOf("o")); // 4 (first occurrence)
console.log(str.lastIndexOf("o")); // 7 (last occurrence)
console.log(str.includes("World")); // true
console.log(str.startsWith("Hello")); // true
console.log(str.endsWith("World")); // true
```

#### String Extraction:
```javascript
let str = "Hello World";

console.log(str.slice(0, 5)); // "Hello"
console.log(str.substring(6, 11)); // "World"
console.log(str.substr(6, 5)); // "World" (deprecated)
```

#### String Modification:
```javascript
let str = "Hello World";

console.log(str.toLowerCase()); // "hello world"
console.log(str.toUpperCase()); // "HELLO WORLD"
console.log(str.replace("World", "JavaScript")); // "Hello JavaScript"
console.log(str.trim()); // Removes whitespace from both ends

// Split and join
let words = str.split(" "); // ["Hello", "World"]
let joined = words.join("-"); // "Hello-World"
```

### Regular Expressions with Strings:
```javascript
let str = "Hello World 123";
let pattern = /\d+/; // Matches digits

console.log(str.match(pattern)); // ["123"]
console.log(str.search(pattern)); // 12 (index of match)
console.log(str.replace(/\d+/g, "***")); // "Hello World ***"
```

## Type Conversion

### Implicit Type Conversion (Coercion):
```javascript
// String conversion
console.log("5" + 3); // "53"
console.log("5" - 3); // 2
console.log("5" * 3); // 15

// Boolean conversion
console.log(Boolean(1)); // true
console.log(Boolean(0)); // false
console.log(Boolean("")); // false
console.log(Boolean("0")); // true

// Null and undefined
console.log(null + 5); // 5
console.log(undefined + 5); // NaN
```

### Explicit Type Conversion:

#### To String:
```javascript
let num = 123;
console.log(String(num)); // "123"
console.log(num.toString()); // "123"
console.log(num + ""); // "123"
```

#### To Number:
```javascript
let str = "123";
console.log(Number(str)); // 123
console.log(parseInt(str)); // 123
console.log(parseFloat("123.45")); // 123.45
console.log(+str); // 123
```

#### To Boolean:
```javascript
console.log(Boolean(1)); // true
console.log(Boolean(0)); // false
console.log(!!1); // true (double negation)
```

### Falsy and Truthy Values:

#### Falsy Values:
```javascript
// These values are considered false in boolean context
console.log(Boolean(false)); // false
console.log(Boolean(0)); // false
console.log(Boolean(-0)); // false
console.log(Boolean(0n)); // false
console.log(Boolean("")); // false
console.log(Boolean(null)); // false
console.log(Boolean(undefined)); // false
console.log(Boolean(NaN)); // false
```

#### Truthy Values:
```javascript
// Everything else is truthy
console.log(Boolean(true)); // true
console.log(Boolean(1)); // true
console.log(Boolean("0")); // true
console.log(Boolean("false")); // true
console.log(Boolean([])); // true
console.log(Boolean({})); // true
```

## Scope and Hoisting

### Scope Types:

#### Global Scope:
```javascript
var globalVar = "I'm global";
let globalLet = "I'm also global";
const globalConst = "I'm global too";

function showGlobal() {
  console.log(globalVar); // Accessible
}
```

#### Function Scope:
```javascript
function myFunction() {
  var functionScoped = "I'm function scoped";
  let blockScoped = "I'm in function";
  
  if (true) {
    var stillFunctionScoped = "Still function scoped";
    let blockOnly = "Block scoped only";
  }
  
  console.log(stillFunctionScoped); // Accessible
  // console.log(blockOnly); // ReferenceError
}
```

#### Block Scope:
```javascript
if (true) {
  var functionScoped = "Function scoped";
  let blockScoped = "Block scoped";
  const alsoBlockScoped = "Also block scoped";
}

console.log(functionScoped); // Accessible
// console.log(blockScoped); // ReferenceError
// console.log(alsoBlockScoped); // ReferenceError
```

### Hoisting:

#### Variable Hoisting:
```javascript
// This code:
console.log(x); // undefined (not ReferenceError)
var x = 5;

// Is interpreted as:
var x;
console.log(x); // undefined
x = 5;

// let and const are hoisted but not initialized
// console.log(y); // ReferenceError: Cannot access 'y' before initialization
let y = 10;
```

#### Function Hoisting:
```javascript
// Function declarations are fully hoisted
sayHello(); // "Hello!" - works before declaration

function sayHello() {
  console.log("Hello!");
}

// Function expressions are not hoisted
// sayGoodbye(); // TypeError: sayGoodbye is not a function
var sayGoodbye = function() {
  console.log("Goodbye!");
};
```

### Best Practices:
1. Use `const` by default, `let` when you need to reassign
2. Avoid `var` in modern JavaScript
3. Declare variables at the top of their scope
4. Use meaningful variable names
5. Initialize variables when declaring them

```javascript
// Good practice
const PI = 3.14159;
let radius = 5;
let area = PI * radius * radius;

// Avoid
var x = 1;
var y = 2;
var z = x + y;
```

---

This covers the fundamental concepts of JavaScript. Next, we'll explore intermediate concepts including ES6+ features, DOM manipulation, and more advanced programming patterns.
