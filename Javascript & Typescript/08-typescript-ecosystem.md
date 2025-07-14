# TypeScript Advanced Features & Ecosystem

## Table of Contents
1. [Decorators](#decorators)
2. [Namespaces](#namespaces)
3. [Declaration Merging](#declaration-merging)
4. [Module Resolution](#module-resolution)
5. [Declaration Files](#declaration-files)
6. [Compiler API](#compiler-api)
7. [Advanced Configuration](#advanced-configuration)
8. [Performance Optimization](#performance-optimization)
9. [Type-only Imports/Exports](#type-only-imports-exports)
10. [Experimental Features](#experimental-features)

## Decorators

### Class Decorators:
```typescript
// Enable decorators in tsconfig.json:
// "experimentalDecorators": true,
// "emitDecoratorMetadata": true

// Simple class decorator
function sealed(constructor: Function) {
  Object.seal(constructor);
  Object.seal(constructor.prototype);
}

@sealed
class BugReport {
  type = "report";
  title: string;

  constructor(t: string) {
    this.title = t;
  }
}

// Decorator factory
function classLogger<T extends { new (...args: any[]): {} }>(constructor: T) {
  return class extends constructor {
    constructor(...args: any[]) {
      console.log(`Creating instance of ${constructor.name}`);
      super(...args);
    }
  };
}

@classLogger
class User {
  name: string;
  constructor(name: string) {
    this.name = name;
  }
}

// Advanced class decorator with options
function Entity(tableName: string) {
  return function <T extends { new (...args: any[]): {} }>(constructor: T) {
    return class extends constructor {
      tableName = tableName;
      
      save() {
        console.log(`Saving to ${tableName} table`);
      }
      
      static getTableName() {
        return tableName;
      }
    };
  };
}

@Entity("users")
class Customer {
  id: number;
  name: string;
  
  constructor(id: number, name: string) {
    this.id = id;
    this.name = name;
  }
}

const customer = new Customer(1, "Alice");
customer.save(); // "Saving to users table"
```

### Method Decorators:
```typescript
// Method decorator
function enumerable(value: boolean) {
  return function (target: any, propertyKey: string, descriptor: PropertyDescriptor) {
    descriptor.enumerable = value;
  };
}

function log(target: any, propertyKey: string, descriptor: PropertyDescriptor) {
  const originalMethod = descriptor.value;

  descriptor.value = function (...args: any[]) {
    console.log(`Calling ${propertyKey} with args:`, args);
    const result = originalMethod.apply(this, args);
    console.log(`Result:`, result);
    return result;
  };

  return descriptor;
}

// Timing decorator
function timing(target: any, propertyKey: string, descriptor: PropertyDescriptor) {
  const originalMethod = descriptor.value;

  descriptor.value = async function (...args: any[]) {
    const start = performance.now();
    const result = await originalMethod.apply(this, args);
    const end = performance.now();
    console.log(`${propertyKey} took ${end - start} milliseconds`);
    return result;
  };

  return descriptor;
}

// Retry decorator
function retry(attempts: number) {
  return function (target: any, propertyKey: string, descriptor: PropertyDescriptor) {
    const originalMethod = descriptor.value;

    descriptor.value = async function (...args: any[]) {
      let lastError: any;
      
      for (let i = 0; i < attempts; i++) {
        try {
          return await originalMethod.apply(this, args);
        } catch (error) {
          lastError = error;
          console.log(`Attempt ${i + 1} failed:`, error);
        }
      }
      
      throw lastError;
    };

    return descriptor;
  };
}

class Calculator {
  @enumerable(false)
  @log
  add(a: number, b: number): number {
    return a + b;
  }

  @timing
  async complexCalculation(n: number): Promise<number> {
    // Simulate complex calculation
    await new Promise(resolve => setTimeout(resolve, 100));
    return n * n;
  }

  @retry(3)
  async unreliableOperation(): Promise<string> {
    if (Math.random() < 0.7) {
      throw new Error("Random failure");
    }
    return "Success!";
  }
}
```

### Property Decorators:
```typescript
// Property decorator
function format(formatString: string) {
  return function (target: any, propertyKey: string) {
    let value = target[propertyKey];

    const getter = () => value;
    const setter = (newVal: string) => {
      value = formatString.replace('{}', newVal);
    };

    Object.defineProperty(target, propertyKey, {
      get: getter,
      set: setter,
      enumerable: true,
      configurable: true
    });
  };
}

// Validation decorator
function required(target: any, propertyKey: string) {
  const privateKey = `_${propertyKey}`;
  
  Object.defineProperty(target, propertyKey, {
    get() {
      return this[privateKey];
    },
    set(value: any) {
      if (value === null || value === undefined || value === '') {
        throw new Error(`${propertyKey} is required`);
      }
      this[privateKey] = value;
    },
    enumerable: true,
    configurable: true
  });
}

// Type conversion decorator
function toNumber(target: any, propertyKey: string) {
  const privateKey = `_${propertyKey}`;
  
  Object.defineProperty(target, propertyKey, {
    get() {
      return this[privateKey];
    },
    set(value: any) {
      this[privateKey] = Number(value);
    },
    enumerable: true,
    configurable: true
  });
}

class Person {
  @format("Hello, {}!")
  greeting: string;

  @required
  name: string;

  @toNumber
  age: number;

  constructor(name: string, age: any) {
    this.name = name;
    this.age = age;
  }
}

const person = new Person("Alice", "30");
person.greeting = "Alice";
console.log(person.greeting); // "Hello, Alice!"
console.log(person.age); // 30 (number)
```

### Parameter Decorators:
```typescript
// Parameter decorator for logging
function logParameter(target: any, propertyKey: string, parameterIndex: number) {
  const existingLoggedParameters: number[] = Reflect.getOwnMetadata("logged_parameters", target, propertyKey) || [];
  existingLoggedParameters.push(parameterIndex);
  Reflect.defineMetadata("logged_parameters", existingLoggedParameters, target, propertyKey);
}

// Parameter validation decorator
function validate(validationRule: (value: any) => boolean) {
  return function (target: any, propertyKey: string, parameterIndex: number) {
    const existingValidators: any[] = Reflect.getOwnMetadata("parameter_validators", target, propertyKey) || [];
    existingValidators[parameterIndex] = validationRule;
    Reflect.defineMetadata("parameter_validators", existingValidators, target, propertyKey);
  };
}

// Method decorator that uses parameter metadata
function validateParameters(target: any, propertyKey: string, descriptor: PropertyDescriptor) {
  const originalMethod = descriptor.value;
  
  descriptor.value = function (...args: any[]) {
    const validators: any[] = Reflect.getOwnMetadata("parameter_validators", target, propertyKey) || [];
    
    validators.forEach((validator, index) => {
      if (validator && !validator(args[index])) {
        throw new Error(`Parameter ${index} validation failed`);
      }
    });
    
    const loggedParams: number[] = Reflect.getOwnMetadata("logged_parameters", target, propertyKey) || [];
    loggedParams.forEach(index => {
      console.log(`Parameter ${index}:`, args[index]);
    });
    
    return originalMethod.apply(this, args);
  };
  
  return descriptor;
}

class MathService {
  @validateParameters
  divide(
    @logParameter
    @validate((value: number) => typeof value === 'number' && !isNaN(value))
    a: number,
    
    @logParameter
    @validate((value: number) => typeof value === 'number' && value !== 0)
    b: number
  ): number {
    return a / b;
  }
}

const mathService = new MathService();
// mathService.divide(10, 0); // Error: Parameter 1 validation failed
mathService.divide(10, 2); // Logs parameters and returns 5
```

### Decorator Metadata:
```typescript
// Install reflect-metadata: npm install reflect-metadata
import "reflect-metadata";

// Metadata decorator
function metadata(key: string, value: any) {
  return function (target: any, propertyKey?: string) {
    if (propertyKey) {
      Reflect.defineMetadata(key, value, target, propertyKey);
    } else {
      Reflect.defineMetadata(key, value, target);
    }
  };
}

// Route decorator for web framework
function route(path: string, method: string = 'GET') {
  return function (target: any, propertyKey: string, descriptor: PropertyDescriptor) {
    Reflect.defineMetadata('route', { path, method }, target, propertyKey);
  };
}

// Controller decorator
function controller(basePath: string) {
  return function (target: any) {
    Reflect.defineMetadata('basePath', basePath, target);
  };
}

@controller('/api/users')
class UserController {
  @route('/', 'GET')
  @metadata('description', 'Get all users')
  getAllUsers() {
    return { users: [] };
  }

  @route('/:id', 'GET')
  @metadata('description', 'Get user by ID')
  getUserById(id: string) {
    return { user: { id } };
  }

  @route('/', 'POST')
  @metadata('description', 'Create new user')
  createUser(userData: any) {
    return { user: userData };
  }
}

// Route discovery function
function getRoutes(controllerClass: any) {
  const basePath = Reflect.getMetadata('basePath', controllerClass) || '';
  const methods = Object.getOwnPropertyNames(controllerClass.prototype);
  
  return methods
    .filter(method => method !== 'constructor')
    .map(method => {
      const routeMetadata = Reflect.getMetadata('route', controllerClass.prototype, method);
      const description = Reflect.getMetadata('description', controllerClass.prototype, method);
      
      if (routeMetadata) {
        return {
          path: basePath + routeMetadata.path,
          method: routeMetadata.method,
          handler: method,
          description
        };
      }
    })
    .filter(Boolean);
}

const routes = getRoutes(UserController);
console.log(routes);
```

## Namespaces

### Basic Namespaces:
```typescript
// Basic namespace
namespace Geometry {
  export interface Point {
    x: number;
    y: number;
  }

  export class Vector {
    constructor(public x: number, public y: number) {}

    add(other: Vector): Vector {
      return new Vector(this.x + other.x, this.y + other.y);
    }

    magnitude(): number {
      return Math.sqrt(this.x * this.x + this.y * this.y);
    }
  }

  export function distance(a: Point, b: Point): number {
    const dx = a.x - b.x;
    const dy = a.y - b.y;
    return Math.sqrt(dx * dx + dy * dy);
  }

  // Internal function (not exported)
  function helper() {
    return "internal";
  }
}

// Usage
const point1: Geometry.Point = { x: 0, y: 0 };
const point2: Geometry.Point = { x: 3, y: 4 };
const vector = new Geometry.Vector(1, 2);
const dist = Geometry.distance(point1, point2);
```

### Nested Namespaces:
```typescript
namespace MyCompany {
  export namespace Utils {
    export namespace String {
      export function capitalize(str: string): string {
        return str.charAt(0).toUpperCase() + str.slice(1);
      }

      export function reverse(str: string): string {
        return str.split('').reverse().join('');
      }
    }

    export namespace Array {
      export function chunk<T>(array: T[], size: number): T[][] {
        const chunks: T[][] = [];
        for (let i = 0; i < array.length; i += size) {
          chunks.push(array.slice(i, i + size));
        }
        return chunks;
      }

      export function flatten<T>(arrays: T[][]): T[] {
        return arrays.reduce((acc, arr) => acc.concat(arr), []);
      }
    }
  }

  export namespace Models {
    export interface User {
      id: string;
      name: string;
      email: string;
    }

    export interface Product {
      id: string;
      name: string;
      price: number;
    }

    export class UserService {
      private users: User[] = [];

      addUser(user: User): void {
        this.users.push(user);
      }

      findUser(id: string): User | undefined {
        return this.users.find(u => u.id === id);
      }
    }
  }
}

// Usage
const capitalized = MyCompany.Utils.String.capitalize("hello");
const chunks = MyCompany.Utils.Array.chunk([1, 2, 3, 4, 5], 2);
const userService = new MyCompany.Models.UserService();
```

### Namespace Aliases:
```typescript
// Long namespace path
namespace VeryLongCompanyName {
  export namespace VeryLongProjectName {
    export namespace VeryLongModuleName {
      export class VeryLongClassName {
        doSomething() {
          return "result";
        }
      }

      export function veryLongFunctionName() {
        return "result";
      }
    }
  }
}

// Create aliases for convenience
import VLM = VeryLongCompanyName.VeryLongProjectName.VeryLongModuleName;
import VLC = VeryLongCompanyName.VeryLongProjectName.VeryLongModuleName.VeryLongClassName;

// Usage with aliases
const instance = new VLC();
const result = VLM.veryLongFunctionName();
```

### Namespaces vs Modules:
```typescript
// Namespace approach (single file compilation)
namespace Database {
  export interface Connection {
    host: string;
    port: number;
    database: string;
  }

  export class MySQLConnection implements Connection {
    constructor(
      public host: string,
      public port: number,
      public database: string
    ) {}

    connect(): Promise<void> {
      return Promise.resolve();
    }
  }

  export class PostgreSQLConnection implements Connection {
    constructor(
      public host: string,
      public port: number,
      public database: string
    ) {}

    connect(): Promise<void> {
      return Promise.resolve();
    }
  }
}

// Module approach (preferred for modern TypeScript)
// database.ts
export interface Connection {
  host: string;
  port: number;
  database: string;
}

export class MySQLConnection implements Connection {
  constructor(
    public host: string,
    public port: number,
    public database: string
  ) {}

  connect(): Promise<void> {
    return Promise.resolve();
  }
}

// mysql.ts
export class PostgreSQLConnection implements Connection {
  constructor(
    public host: string,
    public port: number,
    public database: string
  ) {}

  connect(): Promise<void> {
    return Promise.resolve();
  }
}

// Usage
// import { MySQLConnection, PostgreSQLConnection } from './database';
```

## Declaration Merging

### Interface Merging:
```typescript
// First declaration
interface User {
  name: string;
  age: number;
}

// Second declaration - gets merged
interface User {
  email: string;
}

// Third declaration - also merged
interface User {
  isActive: boolean;
}

// Merged interface has all properties
const user: User = {
  name: "Alice",
  age: 30,
  email: "alice@example.com",
  isActive: true
};

// Method overloading through interface merging
interface Calculator {
  add(a: number, b: number): number;
}

interface Calculator {
  add(a: string, b: string): string;
}

interface Calculator {
  add(a: number[], b: number[]): number[];
}

// Implementation must handle all overloads
class CalculatorImpl implements Calculator {
  add(a: any, b: any): any {
    if (typeof a === "number" && typeof b === "number") {
      return a + b;
    }
    if (typeof a === "string" && typeof b === "string") {
      return a + b;
    }
    if (Array.isArray(a) && Array.isArray(b)) {
      return [...a, ...b];
    }
    throw new Error("Invalid arguments");
  }
}
```

### Namespace Merging:
```typescript
// Namespace and interface merging
namespace Animals {
  export interface Dog {
    breed: string;
  }
}

interface Animals.Dog {
  name: string;
}

// Now Animals.Dog has both breed and name
const dog: Animals.Dog = {
  breed: "Labrador",
  name: "Rex"
};

// Namespace and class merging
class Album {
  label: Album.AlbumLabel;
}

namespace Album {
  export class AlbumLabel {
    constructor(public name: string) {}
  }
}

// Usage
const album = new Album();
album.label = new Album.AlbumLabel("Universal");

// Namespace and function merging
function buildLabel(name: string): string {
  return buildLabel.prefix + name + buildLabel.suffix;
}

namespace buildLabel {
  export let suffix = "";
  export let prefix = "Hello, ";
}

console.log(buildLabel("Alice")); // "Hello, Alice"
```

### Module Augmentation:
```typescript
// Extending global objects
declare global {
  interface Array<T> {
    shuffle(): T[];
    last(): T | undefined;
  }

  interface String {
    capitalize(): string;
    truncate(length: number): string;
  }

  interface Number {
    clamp(min: number, max: number): number;
  }
}

// Implementing the extensions
Array.prototype.shuffle = function () {
  const array = [...this];
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
  return array;
};

Array.prototype.last = function () {
  return this[this.length - 1];
};

String.prototype.capitalize = function () {
  return this.charAt(0).toUpperCase() + this.slice(1);
};

String.prototype.truncate = function (length: number) {
  return this.length > length ? this.slice(0, length) + "..." : this.toString();
};

Number.prototype.clamp = function (min: number, max: number) {
  return Math.min(Math.max(this.valueOf(), min), max);
};

// Usage
const numbers = [1, 2, 3, 4, 5];
console.log(numbers.shuffle()); // [3, 1, 5, 2, 4] (random order)
console.log(numbers.last()); // 5

console.log("hello".capitalize()); // "Hello"
console.log("very long string".truncate(10)); // "very long ..."

console.log((15).clamp(0, 10)); // 10

// Augmenting external modules
declare module "lodash" {
  interface LoDashStatic {
    customMethod(value: any): any;
  }
}

// If you were to implement it:
// import * as _ from "lodash";
// _.customMethod = function(value: any) { return value; };
```

### Declaration File Merging:
```typescript
// types/global.d.ts
declare global {
  interface Window {
    myGlobalFunction(): void;
    APP_CONFIG: {
      apiUrl: string;
      version: string;
    };
  }
}

// types/express.d.ts
declare namespace Express {
  interface Request {
    user?: {
      id: string;
      name: string;
    };
  }

  interface Response {
    success(data?: any): Response;
    error(message: string, code?: number): Response;
  }
}

// types/node.d.ts
declare module NodeJS {
  interface ProcessEnv {
    NODE_ENV: 'development' | 'production' | 'test';
    DATABASE_URL: string;
    JWT_SECRET: string;
  }
}

// Usage in your application
window.myGlobalFunction(); // Available globally
const config = window.APP_CONFIG; // Typed config

// In Express middleware
app.use((req, res, next) => {
  req.user = { id: "123", name: "Alice" }; // Typed user
  next();
});

app.get('/api/users', (req, res) => {
  res.success({ users: [] }); // Custom method available
});

// Environment variables are typed
const dbUrl = process.env.DATABASE_URL; // string
const nodeEnv = process.env.NODE_ENV; // 'development' | 'production' | 'test'
```

## Module Resolution

### Module Resolution Strategies:
```typescript
// tsconfig.json module resolution settings
{
  "compilerOptions": {
    "moduleResolution": "node",  // or "classic"
    "baseUrl": "./src",
    "paths": {
      "@/*": ["*"],
      "@components/*": ["components/*"],
      "@utils/*": ["utils/*"],
      "@services/*": ["services/*"],
      "@types/*": ["types/*"]
    }
  }
}

// Path mapping examples
import { Button } from "@components/Button";      // → src/components/Button
import { ApiService } from "@services/api";      // → src/services/api
import { User } from "@types/user";               // → src/types/user
import { helpers } from "@/utils/helpers";       // → src/utils/helpers
```

### Triple-Slash Directives:
```typescript
// Reference types
/// <reference types="node" />
/// <reference types="jest" />

// Reference paths
/// <reference path="./types/global.d.ts" />
/// <reference path="../declarations/api.d.ts" />

// AMD module dependencies
/// <amd-module name="MyModule" />

// Custom lib references
/// <reference lib="es2020" />
/// <reference lib="dom" />

// Example usage in a declaration file
/// <reference types="node" />

declare global {
  namespace NodeJS {
    interface Global {
      __DEV__: boolean;
      __TEST__: boolean;
    }
  }
}

export {};
```

### Conditional Type Loading:
```typescript
// Dynamic imports based on environment
async function loadAnalytics() {
  if (process.env.NODE_ENV === 'production') {
    const { Analytics } = await import('./analytics/production');
    return Analytics;
  } else {
    const { MockAnalytics } = await import('./analytics/mock');
    return MockAnalytics;
  }
}

// Platform-specific modules
async function loadFileSystem() {
  if (typeof window !== 'undefined') {
    // Browser environment
    const { BrowserFileSystem } = await import('./filesystem/browser');
    return BrowserFileSystem;
  } else {
    // Node.js environment
    const { NodeFileSystem } = await import('./filesystem/node');
    return NodeFileSystem;
  }
}

// Feature detection loading
async function loadPolyfills() {
  const polyfills = [];

  if (!Array.prototype.includes) {
    polyfills.push(import('./polyfills/array-includes'));
  }

  if (!Object.entries) {
    polyfills.push(import('./polyfills/object-entries'));
  }

  await Promise.all(polyfills);
}

// Conditional module exports
export const config = process.env.NODE_ENV === 'production'
  ? require('./config/production')
  : require('./config/development');

// Type-only conditional imports
type Analytics = typeof import('./analytics/production')['Analytics'] |
                typeof import('./analytics/mock')['MockAnalytics'];
```

## Declaration Files

### Writing Declaration Files:
```typescript
// third-party-lib.d.ts
declare module "third-party-lib" {
  // Function declarations
  export function initialize(config: Config): void;
  export function process(data: string): ProcessResult;

  // Class declarations
  export class DataProcessor {
    constructor(options?: ProcessorOptions);
    process(input: string): string;
    getStatus(): ProcessorStatus;
  }

  // Interface declarations
  export interface Config {
    apiKey: string;
    timeout?: number;
    retries?: number;
  }

  export interface ProcessResult {
    success: boolean;
    data?: any;
    error?: string;
  }

  export interface ProcessorOptions {
    mode: 'fast' | 'accurate';
    concurrency?: number;
  }

  export type ProcessorStatus = 'idle' | 'processing' | 'error';

  // Constants
  export const VERSION: string;
  export const DEFAULT_TIMEOUT: number;

  // Namespace for nested functionality
  export namespace Utils {
    function validate(input: any): boolean;
    function sanitize(input: string): string;
  }
}

// Global library declaration
declare global {
  interface Window {
    ThirdPartyLib: {
      init(config: any): void;
      process(data: string): any;
    };
  }

  const THIRD_PARTY_VERSION: string;
}

// UMD module declaration
declare const ThirdPartyLib: {
  initialize(config: any): void;
  process(data: string): any;
};

export = ThirdPartyLib;
export as namespace ThirdPartyLib;
```

### Advanced Declaration Patterns:
```typescript
// Generic library declaration
declare module "generic-collection" {
  export class Collection<T> {
    constructor(items?: T[]);
    add(item: T): void;
    remove(item: T): boolean;
    find(predicate: (item: T) => boolean): T | undefined;
    filter(predicate: (item: T) => boolean): T[];
    map<U>(mapper: (item: T) => U): Collection<U>;
    toArray(): T[];
  }

  export function createCollection<T>(items?: T[]): Collection<T>;
}

// Plugin system declaration
declare module "plugin-system" {
  export interface Plugin<T = any> {
    name: string;
    version: string;
    activate(context: T): void;
    deactivate(): void;
  }

  export class PluginManager<T> {
    constructor(context: T);
    register(plugin: Plugin<T>): void;
    unregister(pluginName: string): void;
    getPlugin(name: string): Plugin<T> | undefined;
    activateAll(): void;
    deactivateAll(): void;
  }

  export function createPlugin<T>(
    definition: Omit<Plugin<T>, 'activate' | 'deactivate'> & {
      activate?: (context: T) => void;
      deactivate?: () => void;
    }
  ): Plugin<T>;
}

// Event emitter declaration
declare module "event-emitter" {
  export type EventHandler<T = any> = (event: T) => void;

  export interface EventMap {
    [eventName: string]: any;
  }

  export class EventEmitter<T extends EventMap = EventMap> {
    on<K extends keyof T>(event: K, handler: EventHandler<T[K]>): void;
    off<K extends keyof T>(event: K, handler: EventHandler<T[K]>): void;
    emit<K extends keyof T>(event: K, data: T[K]): void;
    once<K extends keyof T>(event: K, handler: EventHandler<T[K]>): void;
    removeAllListeners<K extends keyof T>(event?: K): void;
  }

  export function createEventEmitter<T extends EventMap>(): EventEmitter<T>;
}

// Usage examples
interface MyEvents {
  'user:login': { userId: string; timestamp: Date };
  'user:logout': { userId: string };
  'data:updated': { type: string; data: any };
}

const emitter = createEventEmitter<MyEvents>();
emitter.on('user:login', (event) => {
  // event is typed as { userId: string; timestamp: Date }
  console.log(`User ${event.userId} logged in at ${event.timestamp}`);
});
```

### Ambient Module Declarations:
```typescript
// For CSS modules
declare module "*.css" {
  const styles: { [className: string]: string };
  export default styles;
}

declare module "*.scss" {
  const styles: { [className: string]: string };
  export default styles;
}

declare module "*.module.css" {
  const styles: { [className: string]: string };
  export default styles;
}

// For asset imports
declare module "*.png" {
  const src: string;
  export default src;
}

declare module "*.jpg" {
  const src: string;
  export default src;
}

declare module "*.svg" {
  const ReactComponent: React.ComponentType<React.SVGProps<SVGSVGElement>>;
  const src: string;
  
  export { ReactComponent };
  export default src;
}

// For JSON imports
declare module "*.json" {
  const value: any;
  export default value;
}

// For markdown imports
declare module "*.md" {
  const content: string;
  export default content;
}

// For web workers
declare module "*.worker.ts" {
  class WebpackWorker extends Worker {
    constructor();
  }
  export default WebpackWorker;
}

// Usage
import styles from "./component.module.css";
import logo from "./assets/logo.png";
import config from "./config.json";
import readme from "./README.md";
import Worker from "./data.worker.ts";

const worker = new Worker();
const className = styles.container;
```

## Type-only Imports/Exports

### Type-only Import/Export Syntax:
```typescript
// types.ts
export interface User {
  id: string;
  name: string;
  email: string;
}

export interface Product {
  id: string;
  name: string;
  price: number;
}

export class UserService {
  getUser(id: string): Promise<User> {
    return Promise.resolve({ id, name: "Alice", email: "alice@example.com" });
  }
}

export const API_URL = "https://api.example.com";

// main.ts - Type-only imports
import type { User, Product } from "./types";
import { UserService, API_URL } from "./types";

// This creates a type alias, not a runtime import
type UserProfile = User & {
  profile: {
    avatar: string;
    bio: string;
  };
};

// Regular import for runtime usage
const userService = new UserService();
console.log(API_URL);

// Mixed imports
import { type User as UserType, UserService as UserServiceClass } from "./types";

// Type-only exports
export type { User, Product } from "./types";
export { UserService } from "./types";

// Re-exporting types only
export type * from "./types";

// Conditional type-only exports
export type { User } from "./types";
export type { Product } from "./types";
```

### Benefits of Type-only Imports:
```typescript
// Before: Regular imports
import { User, ApiResponse, UserService } from "./api";

// This includes types AND runtime code in the bundle
const service = new UserService();
let user: User;
let response: ApiResponse<User>;

// After: Type-only imports where appropriate
import type { User, ApiResponse } from "./api";
import { UserService } from "./api";

// Only UserService is included in the runtime bundle
// User and ApiResponse are only used for type checking
const service = new UserService();
let user: User;
let response: ApiResponse<User>;

// Compiler optimization
// TypeScript can eliminate type-only imports from the output
```

### Advanced Type-only Patterns:
```typescript
// Dynamic type-only imports
async function processData<T>() {
  const { DataProcessor } = await import("./processor");
  type ProcessorType = typeof DataProcessor;
  
  // Type is available for type checking
  const processor: ProcessorType = new DataProcessor();
  return processor.process();
}

// Conditional type imports
type DatabaseConfig = typeof import("./config/database")["default"];
type ApiConfig = typeof import("./config/api")["default"];

// Type-only namespace imports
import type * as Types from "./types";

function processUser(user: Types.User): Types.ProcessResult {
  return { success: true, data: user };
}

// Type-only default imports
import type DefaultUser from "./user";
import type { User as NamedUser } from "./user";

// Type assertions with type-only imports
import type { ApiResponse } from "./api";

const response = await fetch("/api/users");
const data = await response.json() as ApiResponse<User[]>;

// Generic constraints with type-only imports
import type { Serializable } from "./serialization";

function serialize<T extends Serializable>(data: T): string {
  return JSON.stringify(data);
}
```

## Advanced Configuration

### Strict Configuration:
```json
// tsconfig.json - Ultra-strict configuration
{
  "compilerOptions": {
    // Type Checking
    "strict": true,                           // Enable all strict type checking options
    "noImplicitAny": true,                   // Error on expressions with implied 'any' type
    "strictNullChecks": true,                // Enable strict null checks
    "strictFunctionTypes": true,             // Enable strict checking of function types
    "strictBindCallApply": true,             // Enable strict 'bind', 'call', and 'apply' methods
    "strictPropertyInitialization": true,    // Enable strict checking of property initialization
    "noImplicitReturns": true,              // Error when not all code paths return a value
    "noFallthroughCasesInSwitch": true,     // Error on fallthrough cases in switch statement
    "noUncheckedIndexedAccess": true,       // Add 'undefined' to index signature results
    "noImplicitOverride": true,             // Ensure overriding members are marked with 'override'
    "noPropertyAccessFromIndexSignature": true, // Require undeclared properties to be accessed using element access
    
    // Additional Checks
    "noUnusedLocals": true,                 // Error on unused local variables
    "noUnusedParameters": true,             // Error on unused parameters
    "exactOptionalPropertyTypes": true,     // Interpret optional property types as written
    "allowUnreachableCode": false,          // Error on unreachable code
    "allowUnusedLabels": false,             // Error on unused labels
    
    // Module Resolution
    "moduleResolution": "node",
    "esModuleInterop": true,
    "allowSyntheticDefaultImports": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    
    // Emit
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true,
    "removeComments": false,
    "emitDecoratorMetadata": true,
    "experimentalDecorators": true,
    
    // Output
    "target": "ES2020",
    "module": "ESNext",
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "outDir": "./dist",
    "rootDir": "./src",
    
    // Path Mapping
    "baseUrl": "./src",
    "paths": {
      "@/*": ["*"],
      "@components/*": ["components/*"],
      "@utils/*": ["utils/*"],
      "@types/*": ["types/*"]
    }
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist", "**/*.test.ts", "**/*.spec.ts"]
}
```

### Project References:
```json
// Root tsconfig.json
{
  "files": [],
  "references": [
    { "path": "./packages/core" },
    { "path": "./packages/ui" },
    { "path": "./packages/utils" },
    { "path": "./apps/web" },
    { "path": "./apps/api" }
  ]
}

// packages/core/tsconfig.json
{
  "extends": "../../tsconfig.base.json",
  "compilerOptions": {
    "composite": true,
    "declaration": true,
    "outDir": "./dist",
    "rootDir": "./src"
  },
  "include": ["src/**/*"],
  "exclude": ["**/*.test.ts"]
}

// packages/ui/tsconfig.json
{
  "extends": "../../tsconfig.base.json",
  "compilerOptions": {
    "composite": true,
    "declaration": true,
    "outDir": "./dist",
    "rootDir": "./src",
    "jsx": "react-jsx"
  },
  "include": ["src/**/*"],
  "references": [
    { "path": "../core" }
  ]
}

// apps/web/tsconfig.json
{
  "extends": "../../tsconfig.base.json",
  "compilerOptions": {
    "noEmit": true,
    "jsx": "react-jsx"
  },
  "include": ["src/**/*"],
  "references": [
    { "path": "../../packages/core" },
    { "path": "../../packages/ui" },
    { "path": "../../packages/utils" }
  ]
}
```

### Build Scripts:
```json
// package.json
{
  "scripts": {
    "build": "tsc --build",
    "build:clean": "tsc --build --clean",
    "build:watch": "tsc --build --watch",
    "build:force": "tsc --build --force",
    "type-check": "tsc --noEmit",
    "type-check:watch": "tsc --noEmit --watch"
  }
}
```

This completes the comprehensive TypeScript documentation covering all advanced features and ecosystem concepts that go beyond basic JavaScript. These features enable powerful type-safe development with excellent tooling support and compile-time guarantees.
