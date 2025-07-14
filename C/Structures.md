---
tags:
  - Coding
  - dataType/userDefine
---
In programming, a **structure** is a user-defined data type that allows you to group together related variables under one entity. It is particularly useful for organizing complex data. Structures are common in languages like C, C++, and other similar languages.

---

### **Why Use Structures?**
Structures help you group variables of different types (e.g., integers, floats, strings) into a single, cohesive unit. For example, if you're representing a "Person," you can group their name, age, and address in a structure instead of using multiple separate variables.

---

### **Defining and Using Structures in C**
Here’s how you can define and work with structures in C:

#### **1. Define a Structure**
Use the `struct` keyword:
```c
struct Person {
    char name[50];
    int age;
    char address[100];
};
```

This creates a blueprint for a "Person" with three fields: `name`, `age`, and `address`.

---

#### **2. Create and Access a Structure**
Once the structure is defined, you can create variables of that structure and access the fields using the dot `.` operator:
```c
#include <stdio.h>

struct Person {
    char name[50];
    int age;
    char address[100];
};

int main() {
    struct Person person1;

    // Assign values
    sprintf(person1.name, "Keval");
    person1.age = 25;
    sprintf(person1.address, "Jam Jodhpur, Gujarat, India");

    // Access and print values
    printf("Name: %s\n", person1.name);
    printf("Age: %d\n", person1.age);
    printf("Address: %s\n", person1.address);

    return 0;
}
```

**Output:**
```
Name: Keval
Age: 25
Address: Jam Jodhpur, Gujarat, India
```

---

### **Advantages of Structures**
1. **Group Related Data**: Makes complex data easier to manage.
2. **Modular Code**: Structures make your code more organized and readable.
3. **Custom Types**: You can create your own data types that reflect real-world entities.

---

### **Advanced Features**
- **Arrays of Structures**: You can create arrays of structures for handling multiple entities (like multiple people).
- **Pointers to Structures**: Efficiently pass structures to functions or handle large datasets.
