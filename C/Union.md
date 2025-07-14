---
tags:
  - Coding
  - dataType/userDefine
---
In programming, particularly in languages like **C and C++**, a **union** is a user-defined data type that allows you to store different types of data in the same memory location. It is similar to a ,[[Structures | struct]] but the key difference is that **all members of a union share the same memory space**, so only one member can store a value at any given time.

---

### **How Unions Work**
- A union allocates **one shared memory block** for all its members, and the size of the union is determined by the size of its largest member.
- You can define a union using the `union` keyword.

---

### **Syntax for Union Declaration**
```c
union UnionName {
    data_type member1;
    data_type member2;
    // More members
};
```

---

### **Example in C**
```c
#include <stdio.h>

// Define a union
union Data {
    int i;
    float f;
    char str[20];
};

int main() {
    union Data data;

    // Assign values to the union members
    data.i = 10;
    printf("data.i: %d\n", data.i);

    data.f = 220.5;
    printf("data.f: %.2f\n", data.f);

    // Assign a value to the string member
    // This will overwrite previous values because of shared memory
    sprintf(data.str, "Hello, World!");
    printf("data.str: %s\n", data.str);

    return 0;
}
```

**Output** (approximate, depending on memory behavior):
```
data.i: 10
data.f: 220.50
data.str: Hello, World!
```

**Note:** When `data.str` is assigned a value, it overwrites the previous values of `data.i` and `data.f`, as all members share the same memory space.

---

### **Difference Between Struct and Union**
| **Aspect**        | **Struct**                                   | **Union**                                   |
| ----------------- | -------------------------------------------- | ------------------------------------------- |
| Memory Allocation | Allocates memory for all members.            | Shares memory among members.                |
| Value Storage     | All members can store values simultaneously. | Only one member can hold a value at a time. |
| Size              | Size is the sum of all members.              | Size is the largest member.                 |

---

### **Use Cases for Unions**
- **Memory Optimization**: Useful when working with embedded systems or memory-constrained environments.
- **Variant Data Types**: Storing different types of data (e.g., integers, floats, or strings) in the same location based on context.
- **Low-Level Programming**: Handling hardware registers or working with raw memory.
