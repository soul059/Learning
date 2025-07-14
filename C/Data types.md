---
tags:
  - Coding
  - dataType
---
In programming, **data types** define the type of data that a variable can hold. They specify how the data is stored in memory, the operations that can be performed on it, and the type of values it can take. Data types are essential as they help the compiler or interpreter understand what kind of data is being dealt with.

Here’s a **detailed explanation of data types** in the context of most programming languages, particularly **C/C++**:

---

## **1. Fundamental Data Types**
These are the basic building blocks used to define variables.

| **Type** | **Description**                                      | **Example**            |
| -------- | ---------------------------------------------------- | ---------------------- |
| `int`    | Stores integers (whole numbers)                      | `int age = 25;`        |
| `float`  | Stores decimal numbers (single precision)            | `float price = 99.99;` |
| `double` | Stores decimal numbers (double precision)            | `double pi = 3.14159;` |
| `char`   | Stores single characters                             | `char grade = 'A';`    |
| `bool`   | Stores boolean values (`true` or `false`) (C++ only) | `bool isAlive = true;` |

### Details:
1. **Integer (`int`)**:
   - Size: Typically 4 bytes (depends on the system).
   - Range: `-2,147,483,648` to `2,147,483,647` (on a 4-byte system).
   - Example: `int x = 100;`

2. **Floating-Point (`float`)**:
   - Stores decimal numbers with less precision compared to `double`.
   - Example: `float temp = 98.6;`

3. **Double (`double`)**:
   - Used for high-precision decimal values.
   - Example: `double pi = 3.14159265359;`

4. **Character (`char`)**:
   - Stores a single character using ASCII encoding (1 byte).
   - Example: `char letter = 'A';`

5. **Boolean (`bool`)**:
   - Represents logical values (`true` or `false`) and is only available in C++.
   - Example: `bool isRaining = false;`

---

## **2. Derived Data Types**
These are derived from fundamental types to create more complex data structures.

### **Array**
- A collection of elements of the same type, stored in contiguous memory.
- **Example**:
  ```c
  int arr[5] = {1, 2, 3, 4, 5};
  ```
  Access elements via index: `arr[0]`.

### **[[Pointers]]**
- A variable that stores the memory address of another variable.
- **Example**:
  ```c
  int x = 10;
  int *ptr = &x;  // Pointer stores the address of x
  ```

### **[[Structures]] (`struct`)**
- Groups different data types into a single entity.
- **Example**:
  ```c
  struct Person {
      char name[50];
      int age;
      float height;
  };
  Person p1 = {"Keval", 25, 5.9};
  ```

### **[[Union]]**
- Similar to `struct`, but all members share the same memory location.
- **Example**:
  ```c
  union Data {
      int i;
      float f;
  };
  Data d;
  d.i = 10;  // Allocates memory for `i` only.
  ```

### **Enumeration (`enum`)**
- Defines a set of named integer constants.
- **Example**:
  ```c
  enum Color { RED, GREEN, BLUE };
  Color c = GREEN;
  ```

---

## **3. Abstract or User-defined Data Types [[Object Oriented Programing (OOP)]]**
These are programmer-defined types, used to organize and manage large codebases.

### **Class**
- Blueprint for creating objects; encapsulates data (attributes) and functions (methods).
- **Example in C++**:
  ```cpp
  class Car {
      public:
          string brand;
          void start() {
              cout << brand << " is starting!" << endl;
          }
  };
  ```

### **Typedef or `using`**
- Provides an alias for existing data types to improve code readability.
- **Example**:
  ```c
  typedef unsigned int uint;
  uint age = 30;
  ```

---

## **4. Modifiers for Data Types**
Modifiers alter the size or range of data types.

| **Modifier** | **Description**                             | **Example**       |
| ------------ | ------------------------------------------- | ----------------- |
| `signed`     | Allows both positive and negative values    | `signed int x;`   |
| `unsigned`   | Allows only positive values (extends range) | `unsigned int x;` |
| `short`      | Reduces the size of an integer              | `short int x;`    |
| `long`       | Increases the size of an integer or double  | `long int x;`     |

### **Example**:
```c
unsigned int x = 500;  // 0 to 4,294,967,295 (on a 4-byte system)
long double pi = 3.14159;
```

---

## **5. Void Data Type**
- Represents "no type" and is used in:
  - Functions that do not return a value (`void`).
  - Generic pointers (`void*`).

**Example**:
```c
void greet() {
    printf("Hello, World!");
}
```

---

## **6. Memory Size of Data Types (on a 64-bit system)**
| **Data Type** | **Size**       |
|---------------|----------------|
| `char`        | 1 byte         |
| `int`         | 4 bytes        |
| `float`       | 4 bytes        |
| `double`      | 8 bytes        |
| `bool`        | 1 byte (C++)   |
| `long int`    | 8 bytes        |
| `long double` | 16 bytes       |
| `void`        | No size        |

---

### **Choosing the Right Data Type**
1. **Use `int` for whole numbers** unless you need very large or very small numbers.
2. **Use `float` or `double` for fractional numbers**, with `double` preferred for precision.
3. **Use `char` for characters or small-sized data**.
4. For **larger data requirements**, use `long` or `long long`.

Let me know if you'd like a deeper dive into any specific data type or related concept! 😊