---
tags:
  - Coding
  - operator
---

### **1. Arithmetic Operators**
| **Operator** | **Operation**       | **Example** | **Result** |
| ------------ | ------------------- | ----------- | ---------- |
| `+`          | Addition            | `10 + 5`    | `15`       |
| `-`          | Subtraction         | `10 - 5`    | `5`        |
| `*`          | Multiplication      | `10 * 5`    | `50`       |
| `/`          | Division            | `10 / 5`    | `2`        |
| `%`          | Modulus (remainder) | `10 % 3`    | `1`        |

---

### **2. Relational (Comparison) Operators**
| **Operator** | **Operation**            | **Example** | **Result** |
| ------------ | ------------------------ | ----------- | ---------- |
| `==`         | Equal to                 | `5 == 5`    | `true`     |
| `!=`         | Not equal to             | `5 != 3`    | `true`     |
| `<`          | Less than                | `3 < 5`     | `true`     |
| `>`          | Greater than             | `5 > 3`     | `true`     |
| `<=`         | Less than or equal to    | `3 <= 5`    | `true`     |
| `>=`         | Greater than or equal to | `5 >= 5`    | `true`     |

---

### **3. Logical Operators**
| **Operator** | **Operation** | **Example**     | **Result** |       |     |        |        |
| ------------ | ------------- | --------------- | ---------- | ----- | --- | ------ | ------ |
| `&&`         | Logical AND   | `true && false` | `false`    |       |     |        |        |
| `            |               | `               | Logical OR | `true |     | false` | `true` |
| `!`          | Logical NOT   | `!true`         | `false`    |       |     |        |        |

---

### **4. Bitwise Operators**
| **Operator** | **Operation** | **Example** | **Result**        |     |     |
| ------------ | ------------- | ----------- | ----------------- | --- | --- |
| `&`          | Bitwise AND   | `5 & 3`     | `1`               |     |     |
| `            | `             | Bitwise OR  | `5                | 3`  | `7` |
| `^`          | Bitwise XOR   | `5 ^ 3`     | `6`               |     |     |
| `~`          | Bitwise NOT   | `~5`        | `-6` (complement) |     |     |
| `<<`         | Left shift    | `5 << 1`    | `10`              |     |     |
| `>>`         | Right shift   | `5 >> 1`    | `2`               |     |     |

---

### **5. Assignment Operators**
| **Operator** | **Operation**       | **Example** | **Equivalent To** |
| ------------ | ------------------- | ----------- | ----------------- |
| `=`          | Assignment          | `a = 5`     | -                 |
| `+=`         | Add and assign      | `a += 5`    | `a = a + 5`       |
| `-=`         | Subtract and assign | `a -= 5`    | `a = a - 5`       |
| `*=`         | Multiply and assign | `a *= 5`    | `a = a * 5`       |
| `/=`         | Divide and assign   | `a /= 5`    | `a = a / 5`       |
| `%=`         | Modulus and assign  | `a %= 5`    | `a = a % 5`       |

---

### **6. Increment/Decrement Operators**
| **Operator** | **Operation**        | **Example**       | **Result**           |
|--------------|----------------------|-------------------|----------------------|
| `++`         | Increment by 1       | `a++` (postfix)   | Returns `a`, then `a+1` |
| `--`         | Decrement by 1       | `a--` (postfix)   | Returns `a`, then `a-1` |
| `++`         | Increment by 1       | `++a` (prefix)    | Returns `a+1` immediately |
| `--`         | Decrement by 1       | `--a` (prefix)    | Returns `a-1` immediately |

---

### **7. Conditional (Ternary) Operator**
| **Operator** | **Operation**           | **Example**            | **Result**           |
|--------------|-------------------------|------------------------|----------------------|
| `? :`        | Conditional (if-else)   | `(a > b) ? x : y`      | Returns `x` if `a > b`, otherwise `y` |

---

### **8. Special Operators**
| **Operator** | **Operation**           | **Example**            | **Description**      |
|--------------|-------------------------|------------------------|----------------------|
| `sizeof`     | Get size of a type      | `sizeof(int)`          | Returns size of `int` |
| `&`          | Address-of operator     | `&a`                   | Gets memory address of `a` |
| `*`          | Dereference operator    | `*ptr`                 | Access value at pointer `ptr` |
| `->`         | Access via pointer      | `ptr->member`          | Access structure member via pointer |
| `.`          | Access structure member | `obj.member`           | Access member of structure |

---

### **9. Comma Operator**
| **Operator** | **Operation**           | **Example**            | **Result**           |
|--------------|-------------------------|------------------------|----------------------|
| `,`          | Evaluate multiple expressions | `a = (b, c, d)`  | Evaluates `b`, `c`, `d` and assigns `d` to `a` |

---

### **10. Type Casting Operator**
| **Operator** | **Operation**           | **Example**            | **Description**      |
|--------------|-------------------------|------------------------|----------------------|
| `(type)`     | Type casting            | `(float)a`             | Converts `a` to `float` type |

---
 **operator precedence table** in C, which determines the order in which operators are evaluated in expressions. Operators with higher precedence are evaluated before those with lower precedence, and operators with the same precedence are evaluated according to their associativity.

---

### **Operator Precedence Table**

| **Precedence** | **Operators**                          | **Associativity**             |
|----------------|----------------------------------------|-------------------------------|
| **1**          | `()` (parentheses), `[]` (array subscript), `->` (structure pointer), `.` (structure member) | Left-to-right                |
| **2**          | `++` (post-increment), `--` (post-decrement) | Left-to-right                |
| **3**          | `++` (pre-increment), `--` (pre-decrement), `+` (unary plus), `-` (unary minus), `~` (bitwise NOT), `!` (logical NOT), `*` (dereference), `&` (address-of), `(type)` (type cast), `sizeof` | Right-to-left                |
| **4**          | `*` (multiplication), `/` (division), `%` (modulus) | Left-to-right                |
| **5**          | `+` (addition), `-` (subtraction)      | Left-to-right                |
| **6**          | `<<` (left shift), `>>` (right shift)  | Left-to-right                |
| **7**          | `<`, `<=`, `>`, `>=` (relational operators) | Left-to-right                |
| **8**          | `==`, `!=` (equality and inequality)   | Left-to-right                |
| **9**          | `&` (bitwise AND)                     | Left-to-right                |
| **10**         | `^` (bitwise XOR)                     | Left-to-right                |
| **11**         | `|` (bitwise OR)                      | Left-to-right                |
| **12**         | `&&` (logical AND)                    | Left-to-right                |
| **13**         | `||` (logical OR)                     | Left-to-right                |
| **14**         | `?:` (ternary conditional operator)    | Right-to-left                |
| **15**         | `=` (assignment), `+=`, `-=`, `*=`, `/=`, `%=`, `<<=`, `>>=`, `&=`, `|=`, `^=` | Right-to-left                |
| **16**         | `,` (comma operator)                  | Left-to-right                |

---

### **How to Use Precedence and Associativity**
- Operators with higher precedence (lower number in the table) are evaluated before those with lower precedence.
  Example:
  ```c
  int result = 10 + 5 * 3; // Multiplication (*) happens before addition (+)
  ```
  **Result:** `10 + (5 * 3) = 25`

- Associativity determines the direction in which operators of the same precedence are evaluated:
  - **Left-to-right**: Evaluation starts from the leftmost operator (e.g., addition/subtraction).
  - **Right-to-left**: Evaluation starts from the rightmost operator (e.g., unary operators, assignment).

---
