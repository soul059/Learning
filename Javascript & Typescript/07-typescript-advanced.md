# TypeScript Advanced Concepts

## Table of Contents
1. [Generics](#generics)
2. [Advanced Types](#advanced-types)
3. [Utility Types](#utility-types)
4. [Mapped Types](#mapped-types)
5. [Conditional Types](#conditional-types)
6. [Template Literal Types](#template-literal-types)
7. [Type Manipulation](#type-manipulation)
8. [Decorators](#decorators)
9. [Namespaces](#namespaces)
10. [Declaration Merging](#declaration-merging)

## Generics

### Basic Generics:
```typescript
// Generic functions
function identity<T>(arg: T): T {
  return arg;
}

// Usage
let output1 = identity<string>("hello"); // Type: string
let output2 = identity<number>(42); // Type: number
let output3 = identity("world"); // Type inference: string

// Generic with multiple type parameters
function pair<T, U>(first: T, second: U): [T, U] {
  return [first, second];
}

let result = pair<string, number>("hello", 42); // Type: [string, number]
let inferred = pair("world", true); // Type: [string, boolean]

// Generic array function
function getFirst<T>(arr: T[]): T | undefined {
  return arr[0];
}

let firstString = getFirst(["a", "b", "c"]); // Type: string | undefined
let firstNumber = getFirst([1, 2, 3]); // Type: number | undefined
```

### Generic Constraints:
```typescript
// Basic constraint
interface Lengthwise {
  length: number;
}

function loggingIdentity<T extends Lengthwise>(arg: T): T {
  console.log(arg.length); // Now we know it has a .length property
  return arg;
}

// loggingIdentity(3); // ✗ Error: number doesn't have length
loggingIdentity("hello"); // ✓ OK: string has length
loggingIdentity([1, 2, 3]); // ✓ OK: array has length
loggingIdentity({ length: 10, value: 3 }); // ✓ OK: object has length

// Constraint with keyof
function getProperty<T, K extends keyof T>(obj: T, key: K): T[K] {
  return obj[key];
}

let person = { name: "Alice", age: 30, city: "New York" };
let name = getProperty(person, "name"); // Type: string
let age = getProperty(person, "age"); // Type: number
// let invalid = getProperty(person, "salary"); // ✗ Error: "salary" doesn't exist

// Multiple constraints
interface Named {
  name: string;
}

interface Aged {
  age: number;
}

function greetPerson<T extends Named & Aged>(person: T): string {
  return `Hello ${person.name}, you are ${person.age} years old`;
}

// Conditional constraints
type ApiResponse<T> = T extends string
  ? { message: T }
  : T extends number
  ? { code: T }
  : { data: T };

type StringResponse = ApiResponse<string>; // { message: string }
type NumberResponse = ApiResponse<number>; // { code: number }
type ObjectResponse = ApiResponse<object>; // { data: object }
```

### Generic Classes:
```typescript
// Generic class
class Container<T> {
  private items: T[] = [];
  
  add(item: T): void {
    this.items.push(item);
  }
  
  get(index: number): T | undefined {
    return this.items[index];
  }
  
  getAll(): T[] {
    return [...this.items];
  }
  
  find(predicate: (item: T) => boolean): T | undefined {
    return this.items.find(predicate);
  }
  
  filter(predicate: (item: T) => boolean): T[] {
    return this.items.filter(predicate);
  }
}

// Usage
let stringContainer = new Container<string>();
stringContainer.add("hello");
stringContainer.add("world");

let numberContainer = new Container<number>();
numberContainer.add(1);
numberContainer.add(2);

// Generic class with constraints
interface Comparable<T> {
  compareTo(other: T): number;
}

class SortedList<T extends Comparable<T>> {
  private items: T[] = [];
  
  add(item: T): void {
    this.items.push(item);
    this.items.sort((a, b) => a.compareTo(b));
  }
  
  get(index: number): T | undefined {
    return this.items[index];
  }
}

// Implementation
class Person implements Comparable<Person> {
  constructor(public name: string, public age: number) {}
  
  compareTo(other: Person): number {
    return this.age - other.age;
  }
}

let people = new SortedList<Person>();
people.add(new Person("Alice", 30));
people.add(new Person("Bob", 25));
// List is automatically sorted by age
```

### Generic Interfaces:
```typescript
// Generic interface
interface Repository<T, K> {
  save(entity: T): void;
  findById(id: K): T | undefined;
  findAll(): T[];
  update(id: K, entity: Partial<T>): boolean;
  delete(id: K): boolean;
}

// Implementing generic interface
class UserRepository implements Repository<User, string> {
  private users: Map<string, User> = new Map();
  
  save(user: User): void {
    this.users.set(user.id, user);
  }
  
  findById(id: string): User | undefined {
    return this.users.get(id);
  }
  
  findAll(): User[] {
    return Array.from(this.users.values());
  }
  
  update(id: string, userUpdate: Partial<User>): boolean {
    const user = this.users.get(id);
    if (user) {
      Object.assign(user, userUpdate);
      return true;
    }
    return false;
  }
  
  delete(id: string): boolean {
    return this.users.delete(id);
  }
}

// Generic interface with default type parameters
interface ApiClient<T = any, E = Error> {
  get<R = T>(url: string): Promise<R>;
  post<R = T>(url: string, data: T): Promise<R>;
  handleError(error: E): void;
}

// Factory pattern with generics
interface Factory<T> {
  create(...args: any[]): T;
}

class UserFactory implements Factory<User> {
  create(name: string, email: string): User {
    return {
      id: Math.random().toString(36),
      name,
      email,
      createdAt: new Date()
    };
  }
}
```

### Advanced Generic Patterns:
```typescript
// Generic type with conditional return
type ApiResult<T, E = Error> = {
  success: true;
  data: T;
} | {
  success: false;
  error: E;
};

function handleApiResult<T>(result: ApiResult<T>): T {
  if (result.success) {
    return result.data;
  } else {
    throw result.error;
  }
}

// Generic builder pattern
class QueryBuilder<T> {
  private conditions: string[] = [];
  
  where(condition: string): QueryBuilder<T> {
    this.conditions.push(condition);
    return this;
  }
  
  and(condition: string): QueryBuilder<T> {
    this.conditions.push(`AND ${condition}`);
    return this;
  }
  
  or(condition: string): QueryBuilder<T> {
    this.conditions.push(`OR ${condition}`);
    return this;
  }
  
  build(): string {
    return this.conditions.join(' ');
  }
}

// Usage
let query = new QueryBuilder<User>()
  .where("age > 18")
  .and("status = 'active'")
  .or("role = 'admin'")
  .build();

// Generic middleware pattern
type Middleware<T, R> = (input: T) => R | Promise<R>;

class Pipeline<T> {
  private middlewares: Middleware<any, any>[] = [];
  
  use<R>(middleware: Middleware<T, R>): Pipeline<R> {
    this.middlewares.push(middleware);
    return this as any;
  }
  
  async execute(input: T): Promise<any> {
    let result = input;
    for (const middleware of this.middlewares) {
      result = await middleware(result);
    }
    return result;
  }
}

// Usage
const pipeline = new Pipeline<string>()
  .use((str: string) => str.trim())
  .use((str: string) => str.toUpperCase())
  .use((str: string) => str.split(''));

pipeline.execute("  hello world  "); // ["H", "E", "L", "L", "O", " ", "W", "O", "R", "L", "D"]
```

## Advanced Types

### Index Types:
```typescript
// keyof operator
interface Person {
  name: string;
  age: number;
  email: string;
}

type PersonKeys = keyof Person; // "name" | "age" | "email"

// Using keyof with functions
function getProperty<T, K extends keyof T>(obj: T, key: K): T[K] {
  return obj[key];
}

const person: Person = { name: "Alice", age: 30, email: "alice@example.com" };
const name = getProperty(person, "name"); // Type: string
const age = getProperty(person, "age"); // Type: number

// Indexed access types
type PersonName = Person["name"]; // string
type PersonAge = Person["age"]; // number
type PersonFields = Person[keyof Person]; // string | number

// Array element types
type StringArray = string[];
type ArrayElement = StringArray[number]; // string

// Tuple element types
type Tuple = [string, number, boolean];
type FirstElement = Tuple[0]; // string
type SecondElement = Tuple[1]; // number
type TupleElement = Tuple[number]; // string | number | boolean
```

### Mapped Types:
```typescript
// Basic mapped type
type Readonly<T> = {
  readonly [P in keyof T]: T[P];
};

type Optional<T> = {
  [P in keyof T]?: T[P];
};

type Nullable<T> = {
  [P in keyof T]: T[P] | null;
};

// Using mapped types
interface User {
  id: string;
  name: string;
  email: string;
  age: number;
}

type ReadonlyUser = Readonly<User>;
type OptionalUser = Optional<User>;
type NullableUser = Nullable<User>;

// Advanced mapped types with key remapping
type Getters<T> = {
  [K in keyof T as `get${Capitalize<string & K>}`]: () => T[K];
};

type UserGetters = Getters<User>;
// {
//   getId: () => string;
//   getName: () => string;
//   getEmail: () => string;
//   getAge: () => number;
// }

// Conditional mapped types
type NonFunctionPropertyNames<T> = {
  [K in keyof T]: T[K] extends Function ? never : K;
}[keyof T];

type NonFunctionProperties<T> = Pick<T, NonFunctionPropertyNames<T>>;

class Example {
  name: string = "";
  age: number = 0;
  greet() { return "hello"; }
  getName() { return this.name; }
}

type ExampleData = NonFunctionProperties<Example>; // { name: string; age: number; }

// Template literal key mapping
type EventHandlers<T> = {
  [K in keyof T as `on${Capitalize<string & K>}Change`]: (value: T[K]) => void;
};

type UserEventHandlers = EventHandlers<User>;
// {
//   onIdChange: (value: string) => void;
//   onNameChange: (value: string) => void;
//   onEmailChange: (value: string) => void;
//   onAgeChange: (value: number) => void;
// }
```

### Conditional Types:
```typescript
// Basic conditional types
type IsString<T> = T extends string ? true : false;

type Test1 = IsString<string>; // true
type Test2 = IsString<number>; // false

// Nested conditional types
type TypeName<T> = T extends string
  ? "string"
  : T extends number
  ? "number"
  : T extends boolean
  ? "boolean"
  : T extends undefined
  ? "undefined"
  : T extends Function
  ? "function"
  : "object";

type T1 = TypeName<string>; // "string"
type T2 = TypeName<() => void>; // "function"
type T3 = TypeName<string[]>; // "object"

// Inferring types with conditional types
type ReturnType<T> = T extends (...args: any[]) => infer R ? R : any;

type Func1 = () => string;
type Func2 = () => number;
type Return1 = ReturnType<Func1>; // string
type Return2 = ReturnType<Func2>; // number

// Array element inference
type ArrayElementType<T> = T extends (infer U)[] ? U : never;

type StringArrayElement = ArrayElementType<string[]>; // string
type NumberArrayElement = ArrayElementType<number[]>; // number

// Promise value inference
type PromiseValue<T> = T extends Promise<infer U> ? U : T;

type AsyncString = PromiseValue<Promise<string>>; // string
type SyncString = PromiseValue<string>; // string

// Distributive conditional types
type ToArray<T> = T extends any ? T[] : never;

type DistributedArray = ToArray<string | number>; // string[] | number[]

// Non-distributive conditional types
type ToArrayNonDistributive<T> = [T] extends [any] ? T[] : never;

type NonDistributedArray = ToArrayNonDistributive<string | number>; // (string | number)[]
```

### Recursive Types:
```typescript
// Recursive type definitions
type Json = string | number | boolean | null | Json[] | { [key: string]: Json };

type DeepReadonly<T> = {
  readonly [P in keyof T]: T[P] extends object ? DeepReadonly<T[P]> : T[P];
};

type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

// Tree structure
type TreeNode<T> = {
  value: T;
  children: TreeNode<T>[];
};

// Flatten array types
type FlatArray<T> = T extends readonly (infer U)[]
  ? U extends readonly any[]
    ? FlatArray<U>
    : U
  : T;

type NestedArray = number[][][];
type Flattened = FlatArray<NestedArray>; // number

// Path string type
type PathString<T, K extends keyof T = keyof T> = K extends string
  ? T[K] extends object
    ? `${K}.${PathString<T[K]>}` | K
    : K
  : never;

interface NestedObject {
  user: {
    profile: {
      name: string;
      settings: {
        theme: string;
      };
    };
  };
  posts: Array<{ title: string }>;
}

type Paths = PathString<NestedObject>;
// "user" | "posts" | "user.profile" | "user.profile.name" | "user.profile.settings" | "user.profile.settings.theme"
```

## Utility Types

### Built-in Utility Types:
```typescript
interface User {
  id: string;
  name: string;
  email: string;
  age: number;
  isActive: boolean;
}

// Partial<T> - makes all properties optional
type PartialUser = Partial<User>;
// {
//   id?: string;
//   name?: string;
//   email?: string;
//   age?: number;
//   isActive?: boolean;
// }

// Required<T> - makes all properties required
type RequiredUser = Required<PartialUser>;
// Same as User with all properties required

// Readonly<T> - makes all properties readonly
type ReadonlyUser = Readonly<User>;
// {
//   readonly id: string;
//   readonly name: string;
//   readonly email: string;
//   readonly age: number;
//   readonly isActive: boolean;
// }

// Pick<T, K> - creates a type with only specified properties
type UserSummary = Pick<User, "id" | "name" | "email">;
// {
//   id: string;
//   name: string;
//   email: string;
// }

// Omit<T, K> - creates a type without specified properties
type UserWithoutId = Omit<User, "id">;
// {
//   name: string;
//   email: string;
//   age: number;
//   isActive: boolean;
// }

// Record<K, T> - creates an object type with specified keys and values
type UserRoles = Record<string, "admin" | "user" | "guest">;
// { [key: string]: "admin" | "user" | "guest" }

type StatusFlags = Record<"pending" | "completed" | "failed", boolean>;
// {
//   pending: boolean;
//   completed: boolean;
//   failed: boolean;
// }

// Exclude<T, U> - excludes types that are assignable to U
type StringOrNumber = string | number | boolean;
type OnlyStringOrNumber = Exclude<StringOrNumber, boolean>; // string | number

// Extract<T, U> - extracts types that are assignable to U
type OnlyString = Extract<StringOrNumber, string>; // string

// NonNullable<T> - excludes null and undefined
type NonNullableString = NonNullable<string | null | undefined>; // string
```

### Function Utility Types:
```typescript
// Parameters<T> - extracts function parameter types as tuple
function createUser(name: string, age: number, email?: string): User {
  return { id: "", name, age, email: email || "", isActive: true };
}

type CreateUserParams = Parameters<typeof createUser>; // [string, number, string?]

// ReturnType<T> - extracts function return type
type CreateUserReturn = ReturnType<typeof createUser>; // User

// ConstructorParameters<T> - extracts constructor parameter types
class UserService {
  constructor(private apiUrl: string, private timeout: number) {}
}

type UserServiceParams = ConstructorParameters<typeof UserService>; // [string, number]

// InstanceType<T> - extracts instance type of constructor
type UserServiceInstance = InstanceType<typeof UserService>; // UserService

// ThisParameterType<T> - extracts 'this' parameter type
function greetUser(this: User, greeting: string): string {
  return `${greeting}, ${this.name}!`;
}

type GreetThisType = ThisParameterType<typeof greetUser>; // User

// OmitThisParameter<T> - removes 'this' parameter
type GreetWithoutThis = OmitThisParameter<typeof greetUser>; // (greeting: string) => string
```

### Advanced Utility Types:
```typescript
// Custom utility types
type DeepRequired<T> = {
  [P in keyof T]-?: T[P] extends object ? DeepRequired<T[P]> : T[P];
};

type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

type Mutable<T> = {
  -readonly [P in keyof T]: T[P];
};

type Optional<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>;

type RequiredKeys<T> = {
  [K in keyof T]-?: {} extends Pick<T, K> ? never : K;
}[keyof T];

type OptionalKeys<T> = {
  [K in keyof T]-?: {} extends Pick<T, K> ? K : never;
}[keyof T];

// Key manipulation utilities
type CamelCase<S extends string> = S extends `${infer P1}_${infer P2}${infer P3}`
  ? `${P1}${Uppercase<P2>}${CamelCase<P3>}`
  : S;

type CamelCaseKeys<T> = {
  [K in keyof T as CamelCase<string & K>]: T[K];
};

interface SnakeCaseUser {
  user_id: string;
  first_name: string;
  last_name: string;
  email_address: string;
}

type CamelCaseUser = CamelCaseKeys<SnakeCaseUser>;
// {
//   userId: string;
//   firstName: string;
//   lastName: string;
//   emailAddress: string;
// }

// Value type utilities
type ValueOf<T> = T[keyof T];

type UserValues = ValueOf<User>; // string | number | boolean

// Function composition utility
type Compose<F extends (...args: any[]) => any, G extends (...args: any[]) => any> = 
  F extends (arg: infer A) => infer B
    ? G extends (arg: B) => infer C
      ? (arg: A) => C
      : never
    : never;

// Promise utilities
type Awaited<T> = T extends PromiseLike<infer U> ? Awaited<U> : T;

type AwaitedUser = Awaited<Promise<User>>; // User
type AwaitedNested = Awaited<Promise<Promise<string>>>; // string
```

## Mapped Types

### Basic Mapped Type Patterns:
```typescript
// Property transformation
type Stringify<T> = {
  [K in keyof T]: string;
};

type Nullify<T> = {
  [K in keyof T]: T[K] | null;
};

type Functionize<T> = {
  [K in keyof T]: () => T[K];
};

interface Example {
  name: string;
  age: number;
  active: boolean;
}

type StringifiedExample = Stringify<Example>;
// { name: string; age: string; active: string; }

type NullifiedExample = Nullify<Example>;
// { name: string | null; age: number | null; active: boolean | null; }

type FunctionizedExample = Functionize<Example>;
// { name: () => string; age: () => number; active: () => boolean; }
```

### Key Transformation:
```typescript
// Adding prefixes/suffixes to keys
type AddPrefix<T, P extends string> = {
  [K in keyof T as `${P}${string & K}`]: T[K];
};

type AddSuffix<T, S extends string> = {
  [K in keyof T as `${string & K}${S}`]: T[K];
};

type PrefixedUser = AddPrefix<User, "get">;
// { getId: string; getName: string; getEmail: string; getAge: number; getIsActive: boolean; }

type SuffixedUser = AddSuffix<User, "Changed">;
// { idChanged: string; nameChanged: string; emailChanged: string; ageChanged: number; isActiveChanged: boolean; }

// Key case transformation
type UppercaseKeys<T> = {
  [K in keyof T as Uppercase<string & K>]: T[K];
};

type LowercaseKeys<T> = {
  [K in keyof T as Lowercase<string & K>]: T[K];
};

type CapitalizedKeys<T> = {
  [K in keyof T as Capitalize<string & K>]: T[K];
};

// Conditional key transformation
type OnlyStringKeys<T> = {
  [K in keyof T as T[K] extends string ? K : never]: T[K];
};

type OnlyNumberKeys<T> = {
  [K in keyof T as T[K] extends number ? K : never]: T[K];
};

type UserStringFields = OnlyStringKeys<User>;
// { id: string; name: string; email: string; }

type UserNumberFields = OnlyNumberKeys<User>;
// { age: number; }
```

### Advanced Mapping Patterns:
```typescript
// Recursive mapped types
type DeepMutable<T> = {
  -readonly [P in keyof T]: T[P] extends object ? DeepMutable<T[P]> : T[P];
};

type DeepReadonly<T> = {
  readonly [P in keyof T]: T[P] extends object ? DeepReadonly<T[P]> : T[P];
};

// Proxy object pattern
type Proxify<T> = {
  [P in keyof T]: { get(): T[P]; set(v: T[P]): void; };
};

// Event handler mapping
type EventHandlerMap<T> = {
  [K in keyof T as `on${Capitalize<string & K>}`]: (value: T[K]) => void;
};

type UserEventHandlers = EventHandlerMap<User>;
// {
//   onId: (value: string) => void;
//   onName: (value: string) => void;
//   onEmail: (value: string) => void;
//   onAge: (value: number) => void;
//   onIsActive: (value: boolean) => void;
// }

// Validation mapping
type ValidationMap<T> = {
  [K in keyof T]: (value: T[K]) => boolean;
};

const userValidation: ValidationMap<User> = {
  id: (id) => id.length > 0,
  name: (name) => name.length >= 2,
  email: (email) => email.includes("@"),
  age: (age) => age >= 0 && age <= 150,
  isActive: (active) => typeof active === "boolean"
};

// Diff mapping
type Diff<T, U> = {
  [K in keyof T]: K extends keyof U 
    ? T[K] extends U[K] 
      ? never 
      : T[K]
    : T[K];
};

type UserDiff = Diff<User, { id: string; name: string; }>;
// { id: never; name: never; email: string; age: number; isActive: boolean; }
```

### Mapped Type Utilities:
```typescript
// Builder pattern with mapped types
type Builder<T> = {
  [K in keyof T]: (value: T[K]) => Builder<T>;
} & {
  build(): T;
};

function createBuilder<T>(): Builder<T> {
  const data = {} as any;
  
  const builder = {} as Builder<T>;
  
  // This would need proper implementation
  builder.build = () => data as T;
  
  return builder;
}

// Lens pattern for deep property access
type Lens<T, K extends keyof T> = {
  get: (obj: T) => T[K];
  set: (obj: T, value: T[K]) => T;
};

function createLens<T, K extends keyof T>(key: K): Lens<T, K> {
  return {
    get: (obj) => obj[key],
    set: (obj, value) => ({ ...obj, [key]: value })
  };
}

// State management with mapped types
type StateActions<T> = {
  [K in keyof T as `set${Capitalize<string & K>}`]: (value: T[K]) => void;
} & {
  [K in keyof T as `reset${Capitalize<string & K>}`]: () => void;
};

type UserStateActions = StateActions<User>;
// {
//   setId: (value: string) => void;
//   setName: (value: string) => void;
//   setEmail: (value: string) => void;
//   setAge: (value: number) => void;
//   setIsActive: (value: boolean) => void;
//   resetId: () => void;
//   resetName: () => void;
//   resetEmail: () => void;
//   resetAge: () => void;
//   resetIsActive: () => void;
// }
```

## Conditional Types

### Advanced Conditional Patterns:
```typescript
// Multiple condition chains
type ComplexConditional<T> = T extends string
  ? T extends `${infer Start}${"_"}${infer End}`
    ? `${Start}${Capitalize<End>}`
    : T
  : T extends number
  ? T extends infer N
    ? N extends number
      ? `${N}`
      : never
    : never
  : T extends boolean
  ? T extends true
    ? "yes"
    : "no"
  : "unknown";

type Test1 = ComplexConditional<"hello_world">; // "helloWorld"
type Test2 = ComplexConditional<42>; // "42"
type Test3 = ComplexConditional<true>; // "yes"
type Test4 = ComplexConditional<false>; // "no"

// Distributive conditional types
type ToArray<T> = T extends any ? T[] : never;
type StringOrNumberArray = ToArray<string | number>; // string[] | number[]

// Non-distributive (using tuple trick)
type ToArrayNonDistributive<T> = [T] extends [any] ? T[] : never;
type UnionArray = ToArrayNonDistributive<string | number>; // (string | number)[]

// Filter types using conditionals
type Filter<T, U> = T extends U ? T : never;
type OnlyStrings = Filter<string | number | boolean, string>; // string

// Exclude specific types
type ExcludeFunction<T> = T extends Function ? never : T;
type NoFunctions = ExcludeFunction<string | number | (() => void)>; // string | number

// Extract specific patterns
type ExtractArrays<T> = T extends (infer U)[] ? U[] : never;
type ArrayTypes = ExtractArrays<string[] | number | boolean[]>; // string[] | boolean[]
```

### Inference with Conditional Types:
```typescript
// Function parameter inference
type GetParameters<T> = T extends (...args: infer P) => any ? P : never;

type FuncParams = GetParameters<(a: string, b: number) => void>; // [string, number]

// Return type inference
type GetReturnType<T> = T extends (...args: any[]) => infer R ? R : never;

type FuncReturn = GetReturnType<() => Promise<string>>; // Promise<string>

// Array element inference
type GetArrayElement<T> = T extends (infer U)[] ? U : never;

type ElementType = GetArrayElement<string[]>; // string

// Promise value inference
type UnwrapPromise<T> = T extends Promise<infer U> ? U : T;

type PromiseString = UnwrapPromise<Promise<string>>; // string
type RegularString = UnwrapPromise<string>; // string

// Recursive unwrapping
type DeepUnwrapPromise<T> = T extends Promise<infer U> 
  ? DeepUnwrapPromise<U> 
  : T;

type NestedPromise = DeepUnwrapPromise<Promise<Promise<string>>>; // string

// Object property inference
type GetProperty<T, K> = K extends keyof T ? T[K] : never;

type UserName = GetProperty<User, "name">; // string
type UserInvalid = GetProperty<User, "invalid">; // never

// Tuple element inference
type Head<T> = T extends readonly [infer H, ...any[]] ? H : never;
type Tail<T> = T extends readonly [any, ...infer T] ? T : never;

type FirstElement = Head<[string, number, boolean]>; // string
type RestElements = Tail<[string, number, boolean]>; // [number, boolean]

// String pattern inference
type ExtractAfterUnderscore<T> = T extends `${string}_${infer After}` ? After : never;

type AfterUnderscore = ExtractAfterUnderscore<"hello_world">; // "world"
```

### Practical Conditional Type Applications:
```typescript
// API response type based on method
type ApiResponse<Method extends string> = Method extends "GET"
  ? { data: any }
  : Method extends "POST" | "PUT"
  ? { data: any; id: string }
  : Method extends "DELETE"
  ? { success: boolean }
  : never;

type GetResponse = ApiResponse<"GET">; // { data: any }
type PostResponse = ApiResponse<"POST">; // { data: any; id: string }
type DeleteResponse = ApiResponse<"DELETE">; // { success: boolean }

// Route parameter extraction
type ExtractRouteParams<T> = T extends `${string}:${infer Param}/${infer Rest}`
  ? { [K in Param]: string } & ExtractRouteParams<Rest>
  : T extends `${string}:${infer Param}`
  ? { [K in Param]: string }
  : {};

type UserRoute = ExtractRouteParams<"/users/:id/posts/:postId">;
// { id: string; postId: string }

// Database query type
type QueryResult<T, Operation> = Operation extends "findOne"
  ? T | null
  : Operation extends "findMany"
  ? T[]
  : Operation extends "create"
  ? T
  : Operation extends "update"
  ? T | null
  : Operation extends "delete"
  ? boolean
  : never;

type FindOneUser = QueryResult<User, "findOne">; // User | null
type FindManyUsers = QueryResult<User, "findMany">; // User[]
type CreateUser = QueryResult<User, "create">; // User

// Form validation based on field type
type ValidationRule<T> = T extends string
  ? { minLength?: number; maxLength?: number; pattern?: RegExp }
  : T extends number
  ? { min?: number; max?: number; step?: number }
  : T extends boolean
  ? { required?: boolean }
  : T extends Date
  ? { minDate?: Date; maxDate?: Date }
  : never;

type UserValidationRules = {
  [K in keyof User]: ValidationRule<User[K]>;
};
// {
//   id: { minLength?: number; maxLength?: number; pattern?: RegExp };
//   name: { minLength?: number; maxLength?: number; pattern?: RegExp };
//   email: { minLength?: number; maxLength?: number; pattern?: RegExp };
//   age: { min?: number; max?: number; step?: number };
//   isActive: { required?: boolean };
// }
```

## Template Literal Types

### Basic Template Literals:
```typescript
// Simple template literal types
type Greeting = `Hello ${string}`;
type NumberString = `${number}`;
type BooleanString = `${boolean}`;

// Usage
let greeting1: Greeting = "Hello World"; // ✓ OK
let greeting2: Greeting = "Hello TypeScript"; // ✓ OK
// let greeting3: Greeting = "Hi World"; // ✗ Error

let numStr: NumberString = "123"; // ✓ OK
let boolStr: BooleanString = "true"; // ✓ OK

// Combining with union types
type Theme = "light" | "dark";
type Size = "small" | "medium" | "large";
type ClassName = `theme-${Theme}` | `size-${Size}`;

// Results in: "theme-light" | "theme-dark" | "size-small" | "size-medium" | "size-large"

// With literal types
type HttpMethod = "GET" | "POST" | "PUT" | "DELETE";
type ApiEndpoint = `/api/${string}`;
type ApiCall = `${HttpMethod} ${ApiEndpoint}`;

let apiCall: ApiCall = "GET /api/users"; // ✓ OK
let apiCall2: ApiCall = "POST /api/users/123"; // ✓ OK
```

### Pattern Matching with Template Literals:
```typescript
// Extract parts from template literals
type ExtractRouteParams<T extends string> = 
  T extends `${string}:${infer Param}/${infer Rest}`
    ? { [K in Param]: string } & ExtractRouteParams<Rest>
    : T extends `${string}:${infer Param}`
    ? { [K in Param]: string }
    : {};

type UserPostRoute = ExtractRouteParams<"/users/:userId/posts/:postId">;
// { userId: string; postId: string }

// Version string parsing
type ParseVersion<T extends string> = 
  T extends `${infer Major}.${infer Minor}.${infer Patch}`
    ? { major: Major; minor: Minor; patch: Patch }
    : never;

type Version = ParseVersion<"1.2.3">; // { major: "1"; minor: "2"; patch: "3" }

// CSS property extraction
type ExtractCSSVar<T> = T extends `--${infer VarName}` ? VarName : never;

type CSSVariable = ExtractCSSVar<"--primary-color">; // "primary-color"

// File extension extraction
type GetFileExtension<T extends string> = 
  T extends `${string}.${infer Ext}` ? Ext : never;

type JSExtension = GetFileExtension<"app.js">; // "js"
type TSExtension = GetFileExtension<"component.tsx">; // "tsx"
```

### String Manipulation Functions:
```typescript
// Built-in string manipulation types
type UppercaseGreeting = Uppercase<"hello world">; // "HELLO WORLD"
type LowercaseGreeting = Lowercase<"HELLO WORLD">; // "hello world"
type CapitalizedGreeting = Capitalize<"hello world">; // "Hello world"
type UncapitalizedGreeting = Uncapitalize<"Hello World">; // "hello World"

// Custom string transformations
type PascalCase<S extends string> = 
  S extends `${infer First}${infer Rest}`
    ? `${Uppercase<First>}${Rest}`
    : S;

type CamelCase<S extends string> = 
  S extends `${infer P1}_${infer P2}${infer P3}`
    ? `${P1}${Uppercase<P2>}${CamelCase<P3>}`
    : S;

type KebabToCamel<S extends string> = 
  S extends `${infer P1}-${infer P2}${infer P3}`
    ? `${P1}${Uppercase<P2>}${KebabToCamel<P3>}`
    : S;

type PascalCased = PascalCase<"hello">; // "Hello"
type CamelCased = CamelCase<"hello_world_test">; // "helloWorldTest"
type KebabCased = KebabToCamel<"hello-world-test">; // "helloWorldTest"

// Join string arrays
type Join<T extends readonly string[], D extends string = ","> =
  T extends readonly [infer F, ...infer R]
    ? F extends string
      ? R extends readonly string[]
        ? R["length"] extends 0
          ? F
          : `${F}${D}${Join<R, D>}`
        : never
      : never
    : "";

type Joined = Join<["a", "b", "c"], "-">; // "a-b-c"

// Split strings
type Split<S extends string, D extends string> =
  S extends `${infer T}${D}${infer U}` 
    ? [T, ...Split<U, D>] 
    : [S];

type SplitResult = Split<"a-b-c", "-">; // ["a", "b", "c"]
```

### Practical Template Literal Applications:
```typescript
// SQL query builder types
type SQLTable = "users" | "posts" | "comments";
type SQLOperation = "SELECT" | "INSERT" | "UPDATE" | "DELETE";
type SQLQuery<Op extends SQLOperation, Table extends SQLTable> = 
  `${Op} * FROM ${Table}`;

type UserQuery = SQLQuery<"SELECT", "users">; // "SELECT * FROM users"

// CSS class builder
type CSSModifier = "hover" | "focus" | "active" | "disabled";
type CSSState<Base extends string, Modifier extends CSSModifier> = 
  `${Base}:${Modifier}`;

type ButtonStates = CSSState<"button", CSSModifier>;
// "button:hover" | "button:focus" | "button:active" | "button:disabled"

// Event names
type DOMEventName<T extends string> = `on${Capitalize<T>}`;
type ReactEventName<T extends string> = `on${Capitalize<T>}`;

type ClickEvent = DOMEventName<"click">; // "onClick"
type ChangeEvent = ReactEventName<"change">; // "onChange"

// Environment variable names
type EnvPrefix = "NEXT_PUBLIC" | "REACT_APP";
type EnvVar<Prefix extends EnvPrefix, Name extends string> = 
  `${Prefix}_${Uppercase<Name>}`;

type NextPublicVar = EnvVar<"NEXT_PUBLIC", "api_url">; // "NEXT_PUBLIC_API_URL"

// Configuration keys
type ConfigSection = "database" | "cache" | "auth";
type ConfigKey<Section extends ConfigSection, Key extends string> = 
  `${Section}.${Key}`;

type DatabaseConfig = ConfigKey<"database", "host" | "port" | "name">;
// "database.host" | "database.port" | "database.name"

// Type-safe string builders
type UrlBuilder<Base extends string> = {
  path<P extends string>(path: P): UrlBuilder<`${Base}/${P}`>;
  query<K extends string, V extends string>(key: K, value: V): UrlBuilder<`${Base}?${K}=${V}`>;
  build(): Base;
};

// REST API endpoint builder
type HttpVerb = "GET" | "POST" | "PUT" | "DELETE" | "PATCH";
type Endpoint<Method extends HttpVerb, Path extends string> = {
  method: Method;
  path: Path;
  url: `${Method} ${Path}`;
};

type GetUsers = Endpoint<"GET", "/users">; 
// { method: "GET"; path: "/users"; url: "GET /users" }

type CreateUser = Endpoint<"POST", "/users">;
// { method: "POST"; path: "/users"; url: "POST /users" }
```

This comprehensive guide covers the advanced TypeScript concepts that go beyond basic JavaScript. These features provide powerful type-level programming capabilities and help create more robust, self-documenting code with excellent IDE support and compile-time error detection.
