---
tags:
  - Coding
  - Loops
---
In programming, a **loop** is a powerful concept used to repeat a block of code multiple times. This is particularly useful when you want to perform repetitive tasks without writing the same code over and over again. There are different types of loops, such as **for loops**, **while loops**, and **do-while loops**, each serving slightly different purposes.

### Let's explore them:

#### **1. For Loop**
A **for loop** is used when you know in advance how many times the code should run. It iterates over a sequence, such as a range of numbers.

**Example in Python:**
```python
for i in range(5):
    print(f"This is iteration {i}")
```

**Example in C:**
```c
for (int i = 0; i < 5; i++) {
    printf("This is iteration %d\n", i);
}
```
This will print the text 5 times, with `i` representing the current iteration number.

---

#### **2. While Loop**
A **while loop** runs as long as a specified condition is true. It’s useful when you don’t know how many times you need to repeat the code, but you know the condition for continuation.

**Example in Python:**
```python
count = 0
while count < 5:
    print(f"This is iteration {count}")
    count += 1
```

**Example in C:**
```c
int count = 0;
while (count < 5) {
    printf("This is iteration %d\n", count);
    count++;
}
```
This loop keeps running while `count` is less than 5.

---

#### **3. Do-While Loop**
The **do-while loop** is similar to the `while` loop, but it ensures the code runs at least once before checking the condition.

**Example in C:**
```c
int count = 0;
do {
    printf("This is iteration %d\n", count);
    count++;
} while (count < 5);
```
This will execute the code once, then check the condition, and repeat until the condition becomes false.

---

### Summary:
- Use **for loops** when the number of iterations is predefined.
- Use **while loops** when the condition determines the repetition.
- Use **do-while loops** when you need the code to run at least once before checking the condition.
