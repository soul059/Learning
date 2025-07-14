# TypeScript Fundamentals

## Table of Contents
1. [Introduction to TypeScript](#introduction-to-typescript)
2. [Type System](#type-system)
3. [Basic Types](#basic-types)
4. [Type Annotations](#type-annotations)
5. [Type Inference](#type-inference)
6. [Union and Intersection Types](#union-and-intersection-types)
7. [Literal Types](#literal-types)
8. [Type Aliases](#type-aliases)
9. [Interfaces](#interfaces)
10. [Optional and Readonly Properties](#optional-and-readonly-properties)

## Introduction to TypeScript

### What is TypeScript?
TypeScript is a strongly typed programming language that builds on JavaScript by adding static type definitions. It's developed by Microsoft and compiles to plain JavaScript.

### Key Benefits:
- **Static Type Checking**: Catch errors at compile time
- **Enhanced IDE Support**: Better autocomplete, refactoring, and navigation
- **Self-Documenting Code**: Types serve as documentation
- **Better Refactoring**: Safer large-scale code changes
- **JavaScript Compatibility**: All valid JavaScript is valid TypeScript

### TypeScript vs JavaScript:
```typescript
// JavaScript
function greet(name) {
  return "Hello, " + name;
}

greet("Alice");    // ✓ Works
greet(123);        // ✓ Works but might not be intended
greet();           // ✓ Works but returns "Hello, undefined"

// TypeScript
function greet(name: string): string {
  return "Hello, " + name;
}

greet("Alice");    // ✓ Works
greet(123);        // ✗ Error: Argument of type 'number' is not assignable to parameter of type 'string'
greet();           // ✗ Error: Expected 1 arguments, but got 0
```

### Setting Up TypeScript:
```bash
# Install TypeScript globally
npm install -g typescript

# Initialize TypeScript project
tsc --init

# Compile TypeScript files
tsc filename.ts

# Watch mode (auto-compile on changes)
tsc filename.ts --watch
```

### TypeScript Configuration (tsconfig.json):
```json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "commonjs",
    "lib": ["ES2020", "DOM"],
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "noImplicitAny": true,
    "noImplicitReturns": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist"]
}
```

## Type System

### Static vs Dynamic Typing:
```typescript
// Static typing (TypeScript)
let message: string = "Hello";
message = 123; // ✗ Error: Type 'number' is not assignable to type 'string'

// Dynamic typing (JavaScript)
let message = "Hello";
message = 123; // ✓ Works fine
```

### Type Safety Benefits:
```typescript
// Without types (JavaScript-like)
function calculateArea(shape) {
  if (shape.type === "rectangle") {
    return shape.width * shape.height;
  } else if (shape.type === "circle") {
    return Math.PI * shape.radius * shape.radius;
  }
}

// With types (TypeScript)
interface Rectangle {
  type: "rectangle";
  width: number;
  height: number;
}

interface Circle {
  type: "circle";
  radius: number;
}

type Shape = Rectangle | Circle;

function calculateArea(shape: Shape): number {
  if (shape.type === "rectangle") {
    // TypeScript knows shape is Rectangle here
    return shape.width * shape.height;
  } else {
    // TypeScript knows shape is Circle here
    return Math.PI * shape.radius * shape.radius;
  }
}

const rect: Rectangle = {
  type: "rectangle",
  width: 10,
  height: 20
};

const area = calculateArea(rect); // TypeScript ensures correct usage
```

## Basic Types

### Primitive Types:
```typescript
// Boolean
let isDone: boolean = false;
let isActive: boolean = true;

// Number (all numbers are floating point)
let decimal: number = 6;
let hex: number = 0xf00d;
let binary: number = 0b1010;
let octal: number = 0o744;
let big: bigint = 100n;

// String
let color: string = "blue";
let fullName: string = `Bob Bobbington`;
let age: number = 37;
let sentence: string = `Hello, my name is ${fullName}. I'll be ${age + 1} years old next month.`;

// Null and Undefined
let u: undefined = undefined;
let n: null = null;

// In strict mode, null and undefined are only assignable to themselves and void
let someValue: string | null = null; // Union type to allow null
```

### Array Types:
```typescript
// Array of numbers
let list: number[] = [1, 2, 3];
let list2: Array<number> = [1, 2, 3]; // Generic array type

// Array of strings
let fruits: string[] = ["apple", "banana", "orange"];

// Mixed array using union types
let mixed: (string | number)[] = ["hello", 42, "world"];

// Array of objects
interface Person {
  name: string;
  age: number;
}

let people: Person[] = [
  { name: "Alice", age: 30 },
  { name: "Bob", age: 25 }
];

// Readonly arrays
let readonlyNumbers: readonly number[] = [1, 2, 3];
// readonlyNumbers.push(4); // ✗ Error: Property 'push' does not exist on type 'readonly number[]'
```

### Tuple Types:
```typescript
// Basic tuple
let x: [string, number];
x = ["hello", 10]; // ✓ OK
// x = [10, "hello"]; // ✗ Error: Type 'number' is not assignable to type 'string'

// Accessing tuple elements
console.log(x[0].substring(1)); // ✓ OK - string methods available
console.log(x[1].toFixed(2));   // ✓ OK - number methods available

// Optional tuple elements
let optionalTuple: [string, number?];
optionalTuple = ["hello"]; // ✓ OK
optionalTuple = ["hello", 42]; // ✓ OK

// Rest elements in tuples
let restTuple: [string, ...number[]];
restTuple = ["hello"]; // ✓ OK
restTuple = ["hello", 1, 2, 3]; // ✓ OK

// Named tuple elements (TypeScript 4.0+)
let namedTuple: [name: string, age: number, active?: boolean];
namedTuple = ["Alice", 30]; // ✓ OK
namedTuple = ["Bob", 25, true]; // ✓ OK

// Readonly tuples
let readonlyTuple: readonly [string, number] = ["hello", 42];
// readonlyTuple[0] = "hi"; // ✗ Error: Cannot assign to '0' because it is a read-only property
```

### Enum Types:
```typescript
// Numeric enums (default)
enum Direction {
  Up,    // 0
  Down,  // 1
  Left,  // 2
  Right  // 3
}

let dir: Direction = Direction.Up;
console.log(dir); // 0
console.log(Direction[0]); // "Up" (reverse mapping)

// Explicit numeric values
enum Status {
  Pending = 1,
  Approved = 2,
  Rejected = 3
}

// String enums
enum Color {
  Red = "red",
  Green = "green",
  Blue = "blue"
}

let color: Color = Color.Red;
console.log(color); // "red"

// Heterogeneous enums (not recommended)
enum Mixed {
  No = 0,
  Yes = "YES"
}

// Computed and constant members
enum FileAccess {
  // Constant members
  None,
  Read = 1 << 1,
  Write = 1 << 2,
  ReadWrite = Read | Write,
  
  // Computed member
  G = "123".length
}

// Const enums (inlined at compile time)
const enum HttpStatus {
  OK = 200,
  NotFound = 404,
  InternalServerError = 500
}

let status = HttpStatus.OK; // Becomes: let status = 200;
```

### Any Type:
```typescript
// Any type disables type checking
let notSure: any = 4;
notSure = "maybe a string instead";
notSure = false; // ✓ OK, definitely a boolean

// Any propagates through object access
let anyObj: any = { x: 0 };
anyObj.foo.bar.baz; // ✓ No error (but dangerous!)
anyObj.trim(); // ✓ No error
anyObj(); // ✓ No error
new anyObj(); // ✓ No error

// Array of any
let list: any[] = [1, true, "free"];
list[1] = 100;

// Avoid any when possible - use unknown instead
let userInput: unknown;
let userName: string;

// Must type-check before using unknown
if (typeof userInput === "string") {
  userName = userInput; // ✓ OK after type guard
}
```

### Void, Never, and Object Types:
```typescript
// Void - absence of any type (usually for functions that don't return)
function warnUser(): void {
  console.log("This is my warning message");
}

let unusable: void = undefined; // Only undefined assignable to void

// Never - represents values that never occur
function error(message: string): never {
  throw new Error(message);
}

function infiniteLoop(): never {
  while (true) {
    // Never returns
  }
}

// Object type
let obj: object = { name: "Alice" };
// obj.name; // ✗ Error: Property 'name' does not exist on type 'object'

// Better to use specific object types
let person: { name: string; age: number } = {
  name: "Alice",
  age: 30
};
```

## Type Annotations

### Variable Annotations:
```typescript
// Explicit type annotations
let name: string = "Alice";
let age: number = 30;
let isActive: boolean = true;

// Type annotation without initialization
let height: number;
height = 180; // Must assign before use in strict mode

// Multiple variable declarations
let x: number, y: number, z: number;

// Constant with type annotation
const PI: number = 3.14159;
```

### Function Annotations:
```typescript
// Parameter and return type annotations
function add(x: number, y: number): number {
  return x + y;
}

// Arrow function annotations
const multiply = (x: number, y: number): number => x * y;

// Optional parameters
function greet(name: string, greeting?: string): string {
  return `${greeting || "Hello"}, ${name}!`;
}

// Default parameters
function createUser(name: string, role: string = "user"): void {
  console.log(`Created ${role}: ${name}`);
}

// Rest parameters
function sum(message: string, ...numbers: number[]): number {
  const total = numbers.reduce((acc, num) => acc + num, 0);
  console.log(`${message}: ${total}`);
  return total;
}

// Function overloads
function format(value: string): string;
function format(value: number): string;
function format(value: boolean): string;
function format(value: string | number | boolean): string {
  return String(value);
}

// Using function types
type MathOperation = (x: number, y: number) => number;

const subtract: MathOperation = (x, y) => x - y;
const divide: MathOperation = (x, y) => x / y;
```

### Object Annotations:
```typescript
// Object type annotations
let person: {
  name: string;
  age: number;
  email?: string; // Optional property
  readonly id: number; // Readonly property
} = {
  name: "Alice",
  age: 30,
  id: 123
};

// person.id = 456; // ✗ Error: Cannot assign to 'id' because it is a read-only property

// Nested object types
let company: {
  name: string;
  address: {
    street: string;
    city: string;
    zipCode: string;
  };
  employees: {
    name: string;
    department: string;
  }[];
} = {
  name: "Tech Corp",
  address: {
    street: "123 Main St",
    city: "Anytown",
    zipCode: "12345"
  },
  employees: [
    { name: "Alice", department: "Engineering" },
    { name: "Bob", department: "Marketing" }
  ]
};

// Index signatures for dynamic properties
let scores: {
  [subject: string]: number;
} = {
  math: 95,
  english: 87,
  science: 92
};

scores.history = 88; // ✓ OK
// scores.math = "A"; // ✗ Error: Type 'string' is not assignable to type 'number'
```

## Type Inference

### Basic Type Inference:
```typescript
// TypeScript infers types automatically
let message = "Hello"; // TypeScript infers: string
let count = 42; // TypeScript infers: number
let isComplete = true; // TypeScript infers: boolean

// message = 123; // ✗ Error: Type 'number' is not assignable to type 'string'

// Array inference
let numbers = [1, 2, 3]; // TypeScript infers: number[]
let mixed = [1, "hello", true]; // TypeScript infers: (string | number | boolean)[]

// Object inference
let user = {
  name: "Alice",
  age: 30,
  isActive: true
}; // TypeScript infers the shape

// Function return type inference
function double(x: number) {
  return x * 2; // TypeScript infers return type: number
}

// Contextual typing
window.onmousedown = function(mouseEvent) {
  // mouseEvent is inferred as MouseEvent
  console.log(mouseEvent.clientX);
};

// Array method inference
const users = [
  { name: "Alice", age: 30 },
  { name: "Bob", age: 25 }
];

const names = users.map(user => user.name); // TypeScript infers: string[]
const adults = users.filter(user => user.age >= 18); // TypeScript infers: { name: string; age: number }[]
```

### Best Common Type:
```typescript
// TypeScript finds the best common type
let mixed = [0, 1, null]; // TypeScript infers: (number | null)[]

class Animal {
  name: string;
  constructor(name: string) {
    this.name = name;
  }
}

class Dog extends Animal {
  breed: string;
  constructor(name: string, breed: string) {
    super(name);
    this.breed = breed;
  }
}

class Cat extends Animal {
  indoor: boolean;
  constructor(name: string, indoor: boolean) {
    super(name);
    this.indoor = indoor;
  }
}

// TypeScript infers: Animal[]
let animals = [new Dog("Rex", "Labrador"), new Cat("Whiskers", true)];
```

### When to Use Explicit Types:
```typescript
// When inference isn't sufficient or clear
let userId: string | number; // Union type needs to be explicit
userId = "user123";
userId = 12345;

// For function parameters (always need explicit types)
function processUser(id: string | number) {
  // ...
}

// When you want to enforce a specific type
let config: { [key: string]: any } = {}; // Instead of let config = {};

// For complex return types
function getApiResponse(): Promise<{ data: any[]; status: string }> {
  return fetch('/api/data').then(response => response.json());
}

// For better error messages
interface UserPreferences {
  theme: "light" | "dark";
  language: string;
  notifications: boolean;
}

let preferences: UserPreferences = {
  theme: "light",
  language: "en",
  notifications: true
};
```

## Union and Intersection Types

### Union Types:
```typescript
// Union types (OR relationship)
let id: string | number;
id = "abc123"; // ✓ OK
id = 12345; // ✓ OK
// id = true; // ✗ Error

// Function with union parameters
function formatId(id: string | number): string {
  // Type narrowing required
  if (typeof id === "string") {
    return id.toUpperCase();
  } else {
    return id.toString();
  }
}

// Union with object types
interface Bird {
  type: "bird";
  flyingSpeed: number;
}

interface Horse {
  type: "horse";
  runningSpeed: number;
}

type Animal = Bird | Horse;

function moveAnimal(animal: Animal) {
  // Discriminated union pattern
  switch (animal.type) {
    case "bird":
      console.log(`Flying at ${animal.flyingSpeed} mph`);
      break;
    case "horse":
      console.log(`Running at ${animal.runningSpeed} mph`);
      break;
  }
}

// Union with arrays
let values: (string | number)[] = ["hello", 42, "world", 123];

// Union with functions
type StringProcessor = (str: string) => string;
type NumberProcessor = (num: number) => number;
type Processor = StringProcessor | NumberProcessor;
```

### Intersection Types:
```typescript
// Intersection types (AND relationship)
interface Name {
  name: string;
}

interface Age {
  age: number;
}

type Person = Name & Age;

let person: Person = {
  name: "Alice", // Must have name (from Name)
  age: 30 // Must have age (from Age)
};

// Intersection with methods
interface CanFly {
  fly(): void;
}

interface CanSwim {
  swim(): void;
}

type Amphibious = CanFly & CanSwim;

class Duck implements Amphibious {
  fly() {
    console.log("Flying through the air");
  }
  
  swim() {
    console.log("Swimming in water");
  }
}

// Intersection with conflicting types
interface X {
  a: string;
  b: string;
}

interface Y {
  a: number;
  c: string;
}

type XY = X & Y; // { a: never; b: string; c: string; }
// Property 'a' becomes 'never' because string & number = never

// Practical intersection example
interface Timestamped {
  timestamp: Date;
}

interface Tagged {
  tag: string;
}

function addMetadata<T>(obj: T): T & Timestamped & Tagged {
  return {
    ...obj,
    timestamp: new Date(),
    tag: "processed"
  };
}

const user = { name: "Alice", age: 30 };
const userWithMetadata = addMetadata(user);
// Type: { name: string; age: number; } & Timestamped & Tagged
```

### Type Guards:
```typescript
// typeof type guards
function processValue(value: string | number) {
  if (typeof value === "string") {
    // TypeScript knows value is string here
    return value.toUpperCase();
  } else {
    // TypeScript knows value is number here
    return value.toFixed(2);
  }
}

// instanceof type guards
class Car {
  drive() {
    console.log("Driving a car");
  }
}

class Boat {
  sail() {
    console.log("Sailing a boat");
  }
}

function operate(vehicle: Car | Boat) {
  if (vehicle instanceof Car) {
    vehicle.drive(); // TypeScript knows it's a Car
  } else {
    vehicle.sail(); // TypeScript knows it's a Boat
  }
}

// in operator type guard
interface Fish {
  swim(): void;
}

interface Bird {
  fly(): void;
}

function move(animal: Fish | Bird) {
  if ("swim" in animal) {
    animal.swim(); // TypeScript knows it's Fish
  } else {
    animal.fly(); // TypeScript knows it's Bird
  }
}

// Custom type guards
function isString(value: any): value is string {
  return typeof value === "string";
}

function example(value: any) {
  if (isString(value)) {
    // TypeScript knows value is string here
    console.log(value.toUpperCase());
  }
}

// Assertion functions
function assertIsNumber(value: any): asserts value is number {
  if (typeof value !== "number") {
    throw new Error("Expected number");
  }
}

function processNumber(value: any) {
  assertIsNumber(value);
  // TypeScript knows value is number after assertion
  return value.toFixed(2);
}
```

## Literal Types

### String Literal Types:
```typescript
// String literal types
type Theme = "light" | "dark" | "auto";
type Size = "small" | "medium" | "large";

function setTheme(theme: Theme) {
  // theme can only be "light", "dark", or "auto"
  console.log(`Setting theme to ${theme}`);
}

setTheme("light"); // ✓ OK
setTheme("dark"); // ✓ OK
// setTheme("blue"); // ✗ Error: Argument of type '"blue"' is not assignable to parameter of type 'Theme'

// Template literal types (TypeScript 4.1+)
type EventName<T extends string> = `on${Capitalize<T>}`;
type ButtonEvent = EventName<"click">; // "onClick"
type InputEvent = EventName<"change">; // "onChange"

type HTTPMethod = "GET" | "POST" | "PUT" | "DELETE";
type Endpoint = `/api/${string}`;

function makeRequest(method: HTTPMethod, endpoint: Endpoint) {
  console.log(`${method} ${endpoint}`);
}

makeRequest("GET", "/api/users"); // ✓ OK
// makeRequest("GET", "/users"); // ✗ Error: must start with "/api/"
```

### Numeric Literal Types:
```typescript
// Numeric literal types
type DiceRoll = 1 | 2 | 3 | 4 | 5 | 6;
type HttpStatusCode = 200 | 404 | 500;

function rollDice(): DiceRoll {
  return (Math.floor(Math.random() * 6) + 1) as DiceRoll;
}

function handleResponse(status: HttpStatusCode) {
  switch (status) {
    case 200:
      console.log("Success");
      break;
    case 404:
      console.log("Not found");
      break;
    case 500:
      console.log("Server error");
      break;
  }
}

// Boolean literal types
type Confirmation = true; // Only true is allowed
let confirmed: Confirmation = true;
// let denied: Confirmation = false; // ✗ Error
```

### Literal Type Widening and Narrowing:
```typescript
// Literal type widening
let x = "hello"; // Type: string (widened)
const y = "hello"; // Type: "hello" (literal)

// Preventing widening with const assertions
let z = "hello" as const; // Type: "hello"
let config = {
  theme: "dark",
  size: "large"
} as const; // Type: { readonly theme: "dark"; readonly size: "large"; }

// Arrays with const assertions
let colors = ["red", "green", "blue"] as const;
// Type: readonly ["red", "green", "blue"]

// Object with const assertion
const settings = {
  mode: "development",
  port: 3000,
  features: ["auth", "logging"]
} as const;
// All properties become readonly and literal types
```

## Type Aliases

### Basic Type Aliases:
```typescript
// Simple type aliases
type ID = string | number;
type UserID = string;
type ProductID = number;

// Using type aliases
let userId: UserID = "user123";
let productId: ProductID = 456;

// Function type aliases
type EventHandler = (event: Event) => void;
type AsyncFunction<T> = () => Promise<T>;
type Predicate<T> = (item: T) => boolean;

const clickHandler: EventHandler = (event) => {
  console.log("Clicked!");
};

const fetchUser: AsyncFunction<User> = async () => {
  // Implementation
  return { name: "Alice", age: 30 };
};

// Object type aliases
type Point = {
  x: number;
  y: number;
};

type Rectangle = {
  topLeft: Point;
  bottomRight: Point;
};

type Circle = {
  center: Point;
  radius: number;
};

// Union type aliases
type Shape = Rectangle | Circle;
type Status = "pending" | "approved" | "rejected";
type Theme = "light" | "dark" | "auto";
```

### Generic Type Aliases:
```typescript
// Generic type aliases
type Container<T> = {
  value: T;
  getValue(): T;
  setValue(value: T): void;
};

type ApiResponse<T> = {
  data: T;
  status: number;
  message: string;
};

type KeyValuePair<K, V> = {
  key: K;
  value: V;
};

// Using generic type aliases
let stringContainer: Container<string> = {
  value: "hello",
  getValue() { return this.value; },
  setValue(value) { this.value = value; }
};

let userResponse: ApiResponse<User> = {
  data: { name: "Alice", age: 30 },
  status: 200,
  message: "Success"
};

// Conditional type aliases
type NonNullable<T> = T extends null | undefined ? never : T;
type ArrayElement<T> = T extends readonly (infer E)[] ? E : never;

type StringArray = string[];
type StringElement = ArrayElement<StringArray>; // string

// Mapped type aliases
type Partial<T> = {
  [P in keyof T]?: T[P];
};

type Required<T> = {
  [P in keyof T]-?: T[P];
};

type Readonly<T> = {
  readonly [P in keyof T]: T[P];
};
```

### Advanced Type Aliases:
```typescript
// Recursive type aliases
type Json = string | number | boolean | null | Json[] | { [key: string]: Json };

type TreeNode<T> = {
  value: T;
  children: TreeNode<T>[];
};

// Template literal type aliases
type CSSProperty = `--${string}`;
type EventName<T extends string> = `on${Capitalize<T>}`;
type HttpsUrl = `https://${string}`;

// Utility type aliases
type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

type DeepReadonly<T> = {
  readonly [P in keyof T]: T[P] extends object ? DeepReadonly<T[P]> : T[P];
};

// Function composition types
type Compose<F, G> = F extends (arg: any) => infer A
  ? G extends (arg: A) => infer B
    ? (arg: Parameters<F>[0]) => B
    : never
  : never;

// Example usage
type User = {
  name: string;
  age: number;
  email: string;
};

type PartialUser = Partial<User>; // All properties optional
type RequiredUser = Required<PartialUser>; // All properties required again
type ReadonlyUser = Readonly<User>; // All properties readonly
```

## Interfaces

### Basic Interfaces:
```typescript
// Basic interface definition
interface User {
  name: string;
  age: number;
  email: string;
}

// Implementing interfaces
let user: User = {
  name: "Alice",
  age: 30,
  email: "alice@example.com"
};

// Interface with methods
interface Calculator {
  add(x: number, y: number): number;
  subtract(x: number, y: number): number;
}

class BasicCalculator implements Calculator {
  add(x: number, y: number): number {
    return x + y;
  }
  
  subtract(x: number, y: number): number {
    return x - y;
  }
}

// Interface for function types
interface SearchFunction {
  (source: string, substring: string): boolean;
}

let mySearch: SearchFunction = function(source, substring) {
  return source.indexOf(substring) > -1;
};
```

### Interface Extension:
```typescript
// Extending interfaces
interface Animal {
  name: string;
  age: number;
}

interface Dog extends Animal {
  breed: string;
  bark(): void;
}

interface Cat extends Animal {
  indoor: boolean;
  meow(): void;
}

// Multiple inheritance
interface Flyable {
  fly(): void;
  altitude: number;
}

interface Swimmable {
  swim(): void;
  depth: number;
}

interface Duck extends Animal, Flyable, Swimmable {
  quack(): void;
}

class MallardDuck implements Duck {
  name: string;
  age: number;
  altitude: number;
  depth: number;
  
  constructor(name: string, age: number) {
    this.name = name;
    this.age = age;
    this.altitude = 0;
    this.depth = 0;
  }
  
  fly(): void {
    this.altitude = 100;
    console.log("Flying");
  }
  
  swim(): void {
    this.depth = 5;
    console.log("Swimming");
  }
  
  quack(): void {
    console.log("Quack!");
  }
}
```

### Interface Merging:
```typescript
// Declaration merging - interfaces with same name are merged
interface Window {
  title: string;
}

interface Window {
  ts: number;
}

// Merged interface has both properties
let window: Window = {
  title: "My App",
  ts: Date.now()
};

// Useful for extending third-party libraries
interface Array<T> {
  shuffle(): T[];
}

Array.prototype.shuffle = function() {
  // Implementation
  return this.sort(() => Math.random() - 0.5);
};

// Now all arrays have shuffle method
let numbers = [1, 2, 3, 4, 5];
numbers.shuffle();
```

### Generic Interfaces:
```typescript
// Generic interfaces
interface Repository<T> {
  save(entity: T): void;
  findById(id: string): T | undefined;
  findAll(): T[];
  delete(id: string): boolean;
}

interface User {
  id: string;
  name: string;
  email: string;
}

class UserRepository implements Repository<User> {
  private users: User[] = [];
  
  save(user: User): void {
    this.users.push(user);
  }
  
  findById(id: string): User | undefined {
    return this.users.find(user => user.id === id);
  }
  
  findAll(): User[] {
    return [...this.users];
  }
  
  delete(id: string): boolean {
    const index = this.users.findIndex(user => user.id === id);
    if (index > -1) {
      this.users.splice(index, 1);
      return true;
    }
    return false;
  }
}

// Generic interface with constraints
interface Comparable<T> {
  compareTo(other: T): number;
}

interface Sortable<T extends Comparable<T>> {
  sort(): T[];
}

// Generic interface with multiple type parameters
interface KeyValueStore<K, V> {
  set(key: K, value: V): void;
  get(key: K): V | undefined;
  has(key: K): boolean;
  delete(key: K): boolean;
  keys(): K[];
  values(): V[];
}
```

### Index Signatures:
```typescript
// String index signature
interface StringDictionary {
  [key: string]: string;
}

let dict: StringDictionary = {
  name: "Alice",
  city: "New York",
  country: "USA"
};

// Number index signature
interface NumberArray {
  [index: number]: string;
}

let arr: NumberArray = ["first", "second", "third"];

// Mixed index signatures
interface MixedDictionary {
  [key: string]: string | number;
  [index: number]: string; // Must be assignable to string index type
}

// Index signature with known properties
interface Config {
  apiUrl: string;
  timeout: number;
  [key: string]: any; // Allow additional properties
}

let config: Config = {
  apiUrl: "https://api.example.com",
  timeout: 5000,
  debug: true, // Additional property allowed
  retries: 3   // Additional property allowed
};

// Readonly index signature
interface ReadonlyDictionary {
  readonly [key: string]: string;
}

let readonlyDict: ReadonlyDictionary = {
  name: "Alice",
  city: "New York"
};

// readonlyDict.name = "Bob"; // ✗ Error: Index signature only permits reading
```

## Optional and Readonly Properties

### Optional Properties:
```typescript
// Optional properties with ?
interface User {
  name: string;
  age: number;
  email?: string; // Optional
  phone?: string; // Optional
}

// Valid objects
let user1: User = {
  name: "Alice",
  age: 30
}; // email and phone are optional

let user2: User = {
  name: "Bob", 
  age: 25,
  email: "bob@example.com"
}; // phone is still optional

// Function with optional parameters
function createUser(name: string, age: number, email?: string): User {
  const user: User = { name, age };
  if (email) {
    user.email = email;
  }
  return user;
}

// Optional properties in function types
interface EventListener {
  (event: Event): void;
  passive?: boolean;
  once?: boolean;
}

// Optional method signatures
interface ApiClient {
  get(url: string): Promise<any>;
  post(url: string, data: any): Promise<any>;
  put?(url: string, data: any): Promise<any>; // Optional method
  delete?(url: string): Promise<any>; // Optional method
}
```

### Readonly Properties:
```typescript
// Readonly properties
interface Point {
  readonly x: number;
  readonly y: number;
}

let point: Point = { x: 10, y: 20 };
// point.x = 30; // ✗ Error: Cannot assign to 'x' because it is a read-only property

// Readonly with arrays
interface ReadonlyArray<T> {
  readonly [index: number]: T;
  readonly length: number;
  // Other methods that don't mutate the array
  forEach(callbackfn: (value: T, index: number, array: readonly T[]) => void): void;
  map<U>(callbackfn: (value: T, index: number, array: readonly T[]) => U): U[];
}

let numbers: readonly number[] = [1, 2, 3];
// numbers.push(4); // ✗ Error: Property 'push' does not exist on type 'readonly number[]'

// ReadonlyArray utility type
let mutableArray: number[] = [1, 2, 3];
let readonlyArray: ReadonlyArray<number> = mutableArray; // OK
// readonlyArray[0] = 4; // ✗ Error

// Const assertions create readonly types
let config = {
  apiUrl: "https://api.example.com",
  timeout: 5000,
  features: ["auth", "logging"]
} as const;
// Type: {
//   readonly apiUrl: "https://api.example.com";
//   readonly timeout: 5000;
//   readonly features: readonly ["auth", "logging"];
// }

// Readonly with generics
interface Container<T> {
  readonly value: T;
  readonly items: readonly T[];
}

// Partial readonly (some properties readonly)
interface PartiallyReadonly {
  readonly id: string;
  name: string; // Mutable
  readonly createdAt: Date;
  updatedAt: Date; // Mutable
}
```

### Combining Optional and Readonly:
```typescript
// Optional and readonly together
interface UserProfile {
  readonly id: string;
  name: string;
  readonly email?: string; // Optional and readonly
  bio?: string; // Optional and mutable
  readonly preferences?: { // Optional, readonly object
    theme: "light" | "dark";
    notifications: boolean;
  };
}

// Utility types for optional/readonly combinations
type OptionalReadonly<T, K extends keyof T> = Omit<T, K> & {
  readonly [P in K]?: T[P];
};

interface BaseUser {
  id: string;
  name: string;
  email: string;
  age: number;
}

type UserUpdate = OptionalReadonly<BaseUser, "id">; 
// { readonly id?: string; name: string; email: string; age: number; }

// Making all properties optional and readonly
type DeepReadonlyPartial<T> = {
  readonly [P in keyof T]?: T[P] extends object 
    ? DeepReadonlyPartial<T[P]> 
    : T[P];
};

type ReadonlyPartialUser = DeepReadonlyPartial<UserProfile>;
```

This covers the fundamental TypeScript concepts that are unique to TypeScript and not covered in JavaScript. The type system, annotations, inference, and interface concepts form the foundation for more advanced TypeScript features.
