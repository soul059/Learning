# 04. Control Structures

## 📋 Overview
Control structures determine the flow of execution in a program. They allow you to make decisions, repeat operations, and organize code execution based on conditions.

## 🔀 Decision Making Statements

### 1. **if Statement**

#### Basic if Statement
```cpp
#include <iostream>
using namespace std;

int main() {
    int age;
    cout << "Enter your age: ";
    cin >> age;
    
    if (age >= 18) {
        cout << "You are eligible to vote!" << endl;
    }
    
    return 0;
}
```

#### if-else Statement
```cpp
int main() {
    int number;
    cout << "Enter a number: ";
    cin >> number;
    
    if (number > 0) {
        cout << "The number is positive." << endl;
    } else {
        cout << "The number is non-positive." << endl;
    }
    
    return 0;
}
```

#### if-else if-else Ladder
```cpp
int main() {
    int score;
    cout << "Enter your score: ";
    cin >> score;
    
    if (score >= 90) {
        cout << "Grade: A" << endl;
    } else if (score >= 80) {
        cout << "Grade: B" << endl;
    } else if (score >= 70) {
        cout << "Grade: C" << endl;
    } else if (score >= 60) {
        cout << "Grade: D" << endl;
    } else {
        cout << "Grade: F" << endl;
    }
    
    return 0;
}
```

#### Nested if Statements
```cpp
int main() {
    int age;
    bool hasLicense;
    
    cout << "Enter age: ";
    cin >> age;
    cout << "Do you have a license? (1 for yes, 0 for no): ";
    cin >> hasLicense;
    
    if (age >= 18) {
        if (hasLicense) {
            cout << "You can drive!" << endl;
        } else {
            cout << "You need to get a license first." << endl;
        }
    } else {
        cout << "You are too young to drive." << endl;
    }
    
    return 0;
}
```

### 2. **switch Statement**

#### Basic switch Statement
```cpp
#include <iostream>
using namespace std;

int main() {
    char grade;
    cout << "Enter your grade (A, B, C, D, F): ";
    cin >> grade;
    
    switch (grade) {
        case 'A':
        case 'a':
            cout << "Excellent! You scored 90-100%" << endl;
            break;
        case 'B':
        case 'b':
            cout << "Good! You scored 80-89%" << endl;
            break;
        case 'C':
        case 'c':
            cout << "Average! You scored 70-79%" << endl;
            break;
        case 'D':
        case 'd':
            cout << "Below Average! You scored 60-69%" << endl;
            break;
        case 'F':
        case 'f':
            cout << "Failed! You scored below 60%" << endl;
            break;
        default:
            cout << "Invalid grade entered!" << endl;
            break;
    }
    
    return 0;
}
```

#### switch with Fall-through
```cpp
int main() {
    int day;
    cout << "Enter day number (1-7): ";
    cin >> day;
    
    switch (day) {
        case 1:
            cout << "Monday" << endl;
            break;
        case 2:
            cout << "Tuesday" << endl;
            break;
        case 3:
            cout << "Wednesday" << endl;
            break;
        case 4:
            cout << "Thursday" << endl;
            break;
        case 5:
            cout << "Friday" << endl;
            break;
        case 6:
        case 7:
            cout << "Weekend!" << endl;
            break;
        default:
            cout << "Invalid day!" << endl;
    }
    
    return 0;
}
```

#### Menu-Driven Program with switch
```cpp
#include <iostream>
using namespace std;

int main() {
    int choice;
    double num1, num2, result;
    
    do {
        cout << "\n=== Calculator Menu ===" << endl;
        cout << "1. Addition" << endl;
        cout << "2. Subtraction" << endl;
        cout << "3. Multiplication" << endl;
        cout << "4. Division" << endl;
        cout << "5. Exit" << endl;
        cout << "Enter your choice: ";
        cin >> choice;
        
        switch (choice) {
            case 1:
                cout << "Enter two numbers: ";
                cin >> num1 >> num2;
                result = num1 + num2;
                cout << "Result: " << result << endl;
                break;
                
            case 2:
                cout << "Enter two numbers: ";
                cin >> num1 >> num2;
                result = num1 - num2;
                cout << "Result: " << result << endl;
                break;
                
            case 3:
                cout << "Enter two numbers: ";
                cin >> num1 >> num2;
                result = num1 * num2;
                cout << "Result: " << result << endl;
                break;
                
            case 4:
                cout << "Enter two numbers: ";
                cin >> num1 >> num2;
                if (num2 != 0) {
                    result = num1 / num2;
                    cout << "Result: " << result << endl;
                } else {
                    cout << "Error: Division by zero!" << endl;
                }
                break;
                
            case 5:
                cout << "Goodbye!" << endl;
                break;
                
            default:
                cout << "Invalid choice! Please try again." << endl;
        }
    } while (choice != 5);
    
    return 0;
}
```

## 🔄 Looping Statements

### 1. **for Loop**

#### Basic for Loop
```cpp
#include <iostream>
using namespace std;

int main() {
    // Print numbers 1 to 10
    for (int i = 1; i <= 10; i++) {
        cout << i << " ";
    }
    cout << endl;
    
    // Print even numbers from 2 to 20
    for (int i = 2; i <= 20; i += 2) {
        cout << i << " ";
    }
    cout << endl;
    
    // Countdown
    for (int i = 10; i >= 1; i--) {
        cout << i << " ";
    }
    cout << "Blast off!" << endl;
    
    return 0;
}
```

#### for Loop Variations
```cpp
int main() {
    // Multiple initialization and increment
    for (int i = 0, j = 10; i < j; i++, j--) {
        cout << "i = " << i << ", j = " << j << endl;
    }
    
    // Empty sections
    int k = 0;
    for (; k < 5; ) {  // Initialization and increment outside
        cout << k << " ";
        k++;
    }
    cout << endl;
    
    // Infinite loop (be careful!)
    // for (;;) {
    //     cout << "This runs forever!" << endl;
    //     break;  // Need break to exit
    // }
    
    return 0;
}
```

#### Nested for Loops
```cpp
int main() {
    // Multiplication table
    cout << "Multiplication Table:" << endl;
    for (int i = 1; i <= 5; i++) {
        for (int j = 1; j <= 5; j++) {
            cout << i * j << "\t";
        }
        cout << endl;
    }
    
    // Pattern printing
    cout << "\nStar Pattern:" << endl;
    for (int i = 1; i <= 5; i++) {
        for (int j = 1; j <= i; j++) {
            cout << "* ";
        }
        cout << endl;
    }
    
    return 0;
}
```

### 2. **Range-based for Loop (C++11)**

```cpp
#include <iostream>
#include <vector>
#include <string>
using namespace std;

int main() {
    // With arrays
    int arr[] = {1, 2, 3, 4, 5};
    cout << "Array elements: ";
    for (int element : arr) {
        cout << element << " ";
    }
    cout << endl;
    
    // With vectors
    vector<string> names = {"Alice", "Bob", "Charlie", "Diana"};
    cout << "Names: ";
    for (const string& name : names) {
        cout << name << " ";
    }
    cout << endl;
    
    // Modifying elements (use reference)
    vector<int> numbers = {1, 2, 3, 4, 5};
    for (int& num : numbers) {
        num *= 2;  // Double each element
    }
    
    cout << "Doubled numbers: ";
    for (const auto& num : numbers) {  // auto keyword
        cout << num << " ";
    }
    cout << endl;
    
    return 0;
}
```

### 3. **while Loop**

#### Basic while Loop
```cpp
#include <iostream>
using namespace std;

int main() {
    // Count from 1 to 5
    int i = 1;
    while (i <= 5) {
        cout << i << " ";
        i++;
    }
    cout << endl;
    
    // Input validation
    int number;
    cout << "Enter a positive number: ";
    cin >> number;
    
    while (number <= 0) {
        cout << "Invalid! Enter a positive number: ";
        cin >> number;
    }
    cout << "Thank you! You entered: " << number << endl;
    
    return 0;
}
```

#### while Loop Examples
```cpp
int main() {
    // Find factorial
    int n, factorial = 1;
    cout << "Enter a number: ";
    cin >> n;
    
    int temp = n;
    while (temp > 0) {
        factorial *= temp;
        temp--;
    }
    cout << "Factorial of " << n << " is " << factorial << endl;
    
    // Number guessing game
    int secret = 42, guess;
    cout << "Guess the number (1-100): ";
    cin >> guess;
    
    while (guess != secret) {
        if (guess < secret) {
            cout << "Too low! Try again: ";
        } else {
            cout << "Too high! Try again: ";
        }
        cin >> guess;
    }
    cout << "Congratulations! You guessed it!" << endl;
    
    return 0;
}
```

### 4. **do-while Loop**

```cpp
#include <iostream>
using namespace std;

int main() {
    // Basic do-while
    int i = 1;
    do {
        cout << i << " ";
        i++;
    } while (i <= 5);
    cout << endl;
    
    // Menu system (executes at least once)
    int choice;
    do {
        cout << "\n=== Menu ===" << endl;
        cout << "1. Option 1" << endl;
        cout << "2. Option 2" << endl;
        cout << "3. Exit" << endl;
        cout << "Enter choice: ";
        cin >> choice;
        
        switch (choice) {
            case 1:
                cout << "You selected Option 1" << endl;
                break;
            case 2:
                cout << "You selected Option 2" << endl;
                break;
            case 3:
                cout << "Exiting..." << endl;
                break;
            default:
                cout << "Invalid choice!" << endl;
        }
    } while (choice != 3);
    
    return 0;
}
```

## 🚦 Jump Statements

### 1. **break Statement**

```cpp
#include <iostream>
using namespace std;

int main() {
    // break in for loop
    cout << "Numbers 1 to 10, but stop at 6:" << endl;
    for (int i = 1; i <= 10; i++) {
        if (i == 6) {
            break;  // Exit the loop when i equals 6
        }
        cout << i << " ";
    }
    cout << endl;
    
    // break in nested loops (only breaks inner loop)
    cout << "Nested loop with break:" << endl;
    for (int i = 1; i <= 3; i++) {
        cout << "Outer loop: " << i << endl;
        for (int j = 1; j <= 5; j++) {
            if (j == 3) {
                break;  // Only breaks inner loop
            }
            cout << "  Inner loop: " << j << endl;
        }
    }
    
    return 0;
}
```

### 2. **continue Statement**

```cpp
int main() {
    // continue in for loop
    cout << "Odd numbers from 1 to 10:" << endl;
    for (int i = 1; i <= 10; i++) {
        if (i % 2 == 0) {
            continue;  // Skip even numbers
        }
        cout << i << " ";
    }
    cout << endl;
    
    // continue in while loop
    cout << "Numbers 1 to 10, skipping 5:" << endl;
    int i = 0;
    while (i < 10) {
        i++;
        if (i == 5) {
            continue;  // Skip printing 5
        }
        cout << i << " ";
    }
    cout << endl;
    
    return 0;
}
```

### 3. **goto Statement** (Generally Avoided)

```cpp
#include <iostream>
using namespace std;

int main() {
    int choice;
    
start:  // Label
    cout << "Enter 1 to continue, 2 to restart, 3 to exit: ";
    cin >> choice;
    
    switch (choice) {
        case 1:
            cout << "Continuing..." << endl;
            break;
        case 2:
            cout << "Restarting..." << endl;
            goto start;  // Jump to label
        case 3:
            cout << "Exiting..." << endl;
            goto end;
        default:
            cout << "Invalid choice!" << endl;
            goto start;
    }
    
end:  // Label
    cout << "Program ended." << endl;
    return 0;
}
```

**Note:** `goto` is generally discouraged as it can make code hard to read and maintain. Use structured control statements instead.

## 🎯 Practical Examples

### 1. **Prime Number Checker**
```cpp
#include <iostream>
#include <cmath>
using namespace std;

int main() {
    int num;
    cout << "Enter a number: ";
    cin >> num;
    
    if (num <= 1) {
        cout << num << " is not a prime number." << endl;
        return 0;
    }
    
    bool isPrime = true;
    for (int i = 2; i <= sqrt(num); i++) {
        if (num % i == 0) {
            isPrime = false;
            break;
        }
    }
    
    if (isPrime) {
        cout << num << " is a prime number." << endl;
    } else {
        cout << num << " is not a prime number." << endl;
    }
    
    return 0;
}
```

### 2. **Fibonacci Series**
```cpp
int main() {
    int n;
    cout << "Enter number of terms: ";
    cin >> n;
    
    if (n <= 0) {
        cout << "Invalid input!" << endl;
        return 1;
    }
    
    cout << "Fibonacci series: ";
    
    if (n >= 1) cout << "0 ";
    if (n >= 2) cout << "1 ";
    
    int prev1 = 0, prev2 = 1;
    for (int i = 3; i <= n; i++) {
        int current = prev1 + prev2;
        cout << current << " ";
        prev1 = prev2;
        prev2 = current;
    }
    cout << endl;
    
    return 0;
}
```

### 3. **Simple ATM System**
```cpp
#include <iostream>
#include <iomanip>
using namespace std;

int main() {
    double balance = 1000.0;  // Initial balance
    int choice;
    double amount;
    
    cout << "=== Welcome to ATM ===" << endl;
    
    do {
        cout << "\n1. Check Balance" << endl;
        cout << "2. Deposit" << endl;
        cout << "3. Withdraw" << endl;
        cout << "4. Exit" << endl;
        cout << "Enter your choice: ";
        cin >> choice;
        
        switch (choice) {
            case 1:
                cout << fixed << setprecision(2);
                cout << "Current balance: $" << balance << endl;
                break;
                
            case 2:
                cout << "Enter deposit amount: $";
                cin >> amount;
                if (amount > 0) {
                    balance += amount;
                    cout << "Deposit successful!" << endl;
                    cout << "New balance: $" << balance << endl;
                } else {
                    cout << "Invalid amount!" << endl;
                }
                break;
                
            case 3:
                cout << "Enter withdrawal amount: $";
                cin >> amount;
                if (amount > 0 && amount <= balance) {
                    balance -= amount;
                    cout << "Withdrawal successful!" << endl;
                    cout << "New balance: $" << balance << endl;
                } else if (amount > balance) {
                    cout << "Insufficient funds!" << endl;
                } else {
                    cout << "Invalid amount!" << endl;
                }
                break;
                
            case 4:
                cout << "Thank you for using ATM!" << endl;
                break;
                
            default:
                cout << "Invalid choice! Please try again." << endl;
        }
    } while (choice != 4);
    
    return 0;
}
```

### 4. **Pattern Printing Programs**
```cpp
int main() {
    int n = 5;
    
    // Right triangle
    cout << "Right Triangle:" << endl;
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= i; j++) {
            cout << "* ";
        }
        cout << endl;
    }
    
    // Inverted triangle
    cout << "\nInverted Triangle:" << endl;
    for (int i = n; i >= 1; i--) {
        for (int j = 1; j <= i; j++) {
            cout << "* ";
        }
        cout << endl;
    }
    
    // Pyramid
    cout << "\nPyramid:" << endl;
    for (int i = 1; i <= n; i++) {
        // Print spaces
        for (int j = 1; j <= n - i; j++) {
            cout << " ";
        }
        // Print stars
        for (int j = 1; j <= 2 * i - 1; j++) {
            cout << "*";
        }
        cout << endl;
    }
    
    // Number pattern
    cout << "\nNumber Pattern:" << endl;
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= i; j++) {
            cout << j << " ";
        }
        cout << endl;
    }
    
    return 0;
}
```

## 💡 Best Practices

### 1. **Choose the Right Loop**
```cpp
// Use for loop when you know the number of iterations
for (int i = 0; i < 10; i++) {
    // Process 10 items
}

// Use while loop for condition-based repetition
while (userWantsToContinue) {
    // Process until user decides to stop
}

// Use do-while when you need at least one execution
do {
    // Show menu at least once
} while (choice != exitOption);

// Use range-based for with containers
for (const auto& item : container) {
    // Process each item
}
```

### 2. **Avoid Deep Nesting**
```cpp
// Bad: Deep nesting
if (condition1) {
    if (condition2) {
        if (condition3) {
            // Do something
        }
    }
}

// Better: Early returns or combined conditions
if (!condition1) return;
if (!condition2) return;
if (!condition3) return;
// Do something

// Or
if (condition1 && condition2 && condition3) {
    // Do something
}
```

### 3. **Use Meaningful Variable Names**
```cpp
// Bad
for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
        // What do i and j represent?
    }
}

// Better
for (int row = 0; row < numRows; row++) {
    for (int col = 0; col < numCols; col++) {
        // Clear what row and col represent
    }
}
```

### 4. **Initialize Variables**
```cpp
// Always initialize loop variables
int count = 0;
while (count < 10) {
    // ...
    count++;
}

// Initialize variables used in conditions
bool found = false;
int index = 0;
while (!found && index < arraySize) {
    // ...
}
```

## 🔗 Related Topics
- [Operators](./03-operators.md)
- [Functions](./05-functions.md)
- [Arrays & Strings](./06-arrays-strings.md)

---
*Previous: [Operators](./03-operators.md) | Next: [Functions](./05-functions.md)*
