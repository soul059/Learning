---
tags:
  - Coding
  - recursion
---
**Recursion** is a programming concept where a function calls itself in order to solve a problem. It's often used to solve problems that can be broken down into smaller, similar sub-problems. Recursion continues until it reaches a base case, which stops the function from calling itself indefinitely.

---

### Key Components of Recursion:
1. **Base Case**: The condition at which the recursion stops. Without a base case, the recursion would continue forever and cause an error.
2. **Recursive Case**: The part of the function where it calls itself to solve a smaller sub-problem.

---

### Example of Recursion in Python (Factorial Calculation):
The factorial of a number \( n \) (denoted as \( n! \)) is the product of all positive integers less than or equal to \( n \).

For example:
\[
5! = 5 \times 4 \times 3 \times 2 \times 1 = 120
\]

Here's how recursion can compute the factorial:
```python
def factorial(n):
    if n == 1:  # Base case
        return 1
    else:  # Recursive case
        return n * factorial(n - 1)

# Example usage:
result = factorial(5)
print(result)  # Output: 120
```

---

### How It Works:
1. `factorial(5)` calls `factorial(4)`.
2. `factorial(4)` calls `factorial(3)`, and so on, until it reaches `factorial(1)`.
3. When `factorial(1)` is called, it hits the **base case** and returns `1`.
4. The returned values are then multiplied together in reverse order:
   - `factorial(2) = 2 * 1`
   - `factorial(3) = 3 * 2`
   - `factorial(4) = 4 * 6`
   - `factorial(5) = 5 * 24`

---

### Example in C (Fibonacci Sequence):
The Fibonacci sequence is another common problem where recursion is used. The sequence starts with 0 and 1, and each subsequent number is the sum of the previous two.

```c
#include <stdio.h>

int fibonacci(int n) {
    if (n == 0) return 0;  // Base case
    if (n == 1) return 1;  // Base case
    return fibonacci(n - 1) + fibonacci(n - 2);  // Recursive case
}

int main() {
    int n = 6;  // Find the 6th Fibonacci number
    printf("Fibonacci number: %d\n", fibonacci(n));
    return 0;
}
```

---

### Important Notes:
- **Recursion vs. Iteration**: Recursion can make code more intuitive for problems like tree traversal or divide-and-conquer algorithms. However, it can also use more memory than an iterative solution because of function calls stored on the stack.
- **Risk of Infinite Recursion**: Always ensure a base case exists to prevent endless recursion.

Does this clear up how recursion works, or would you like to explore a specific example more deeply? 😊