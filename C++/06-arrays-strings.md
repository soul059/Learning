# 06. Arrays & Strings

## 📋 Overview
Arrays are collections of elements of the same data type stored in contiguous memory locations. Strings are arrays of characters used to represent text. Understanding arrays and strings is fundamental for data manipulation in C++.

## 📊 Arrays

### 1. **Array Declaration and Initialization**

#### Basic Array Declaration
```cpp
#include <iostream>
using namespace std;

int main() {
    // Declaration
    int numbers[5];           // Array of 5 integers (uninitialized)
    
    // Declaration with initialization
    int scores[5] = {85, 92, 78, 96, 88};
    
    // Partial initialization (rest are zero)
    int values[5] = {1, 2};   // {1, 2, 0, 0, 0}
    
    // Size inferred from initializer
    int data[] = {10, 20, 30, 40, 50};  // Size is 5
    
    // Zero initialization
    int zeros[5] = {};        // All elements are 0
    
    return 0;
}
```

#### Different Types of Arrays
```cpp
char letters[26];                    // Character array
double temperatures[7];              // Double array
bool flags[10] = {false};           // Boolean array
string names[3] = {"Alice", "Bob", "Charlie"};  // String array

// Multi-dimensional arrays
int matrix[3][4];                   // 3x4 matrix
int cube[2][3][4];                  // 3D array
```

### 2. **Array Access and Manipulation**
```cpp
#include <iostream>
using namespace std;

int main() {
    int numbers[5] = {10, 20, 30, 40, 50};
    
    // Accessing elements (0-based indexing)
    cout << "First element: " << numbers[0] << endl;    // 10
    cout << "Last element: " << numbers[4] << endl;     // 50
    
    // Modifying elements
    numbers[1] = 25;
    numbers[3] = 45;
    
    // Printing all elements
    cout << "Array elements: ";
    for (int i = 0; i < 5; i++) {
        cout << numbers[i] << " ";
    }
    cout << endl;
    
    // Using range-based for loop (C++11)
    cout << "Using range-based for: ";
    for (int num : numbers) {
        cout << num << " ";
    }
    cout << endl;
    
    return 0;
}
```

### 3. **Array Size and Bounds**
```cpp
#include <iostream>
using namespace std;

int main() {
    int arr[10] = {1, 2, 3, 4, 5};
    
    // Calculate array size
    int arraySize = sizeof(arr) / sizeof(arr[0]);
    cout << "Array size: " << arraySize << endl;
    
    // Safe array access
    int index;
    cout << "Enter index (0-9): ";
    cin >> index;
    
    if (index >= 0 && index < arraySize) {
        cout << "Element at index " << index << ": " << arr[index] << endl;
    } else {
        cout << "Index out of bounds!" << endl;
    }
    
    return 0;
}
```

### 4. **Multi-dimensional Arrays**
```cpp
#include <iostream>
using namespace std;

int main() {
    // 2D array declaration and initialization
    int matrix[3][4] = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12}
    };
    
    // Alternative initialization
    int grid[2][3] = {{1, 2, 3}, {4, 5, 6}};
    
    // Accessing 2D array elements
    cout << "Element at [1][2]: " << matrix[1][2] << endl;  // 7
    
    // Printing 2D array
    cout << "Matrix:" << endl;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            cout << matrix[i][j] << "\t";
        }
        cout << endl;
    }
    
    // 3D array example
    int cube[2][2][2] = {
        {{1, 2}, {3, 4}},
        {{5, 6}, {7, 8}}
    };
    
    cout << "3D array element [1][0][1]: " << cube[1][0][1] << endl;  // 6
    
    return 0;
}
```

### 5. **Array Functions**
```cpp
#include <iostream>
#include <algorithm>  // For sort, reverse, etc.
using namespace std;

// Function to print array
void printArray(int arr[], int size) {
    for (int i = 0; i < size; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;
}

// Function to find maximum element
int findMax(int arr[], int size) {
    int max = arr[0];
    for (int i = 1; i < size; i++) {
        if (arr[i] > max) {
            max = arr[i];
        }
    }
    return max;
}

// Function to sum array elements
int sumArray(int arr[], int size) {
    int sum = 0;
    for (int i = 0; i < size; i++) {
        sum += arr[i];
    }
    return sum;
}

// Function to search element
int linearSearch(int arr[], int size, int target) {
    for (int i = 0; i < size; i++) {
        if (arr[i] == target) {
            return i;  // Return index
        }
    }
    return -1;  // Not found
}

int main() {
    int numbers[] = {64, 34, 25, 12, 22, 11, 90};
    int size = sizeof(numbers) / sizeof(numbers[0]);
    
    cout << "Original array: ";
    printArray(numbers, size);
    
    cout << "Maximum element: " << findMax(numbers, size) << endl;
    cout << "Sum of elements: " << sumArray(numbers, size) << endl;
    
    int target = 25;
    int index = linearSearch(numbers, size, target);
    if (index != -1) {
        cout << target << " found at index " << index << endl;
    } else {
        cout << target << " not found" << endl;
    }
    
    // Sort array
    sort(numbers, numbers + size);
    cout << "Sorted array: ";
    printArray(numbers, size);
    
    return 0;
}
```

## 🔤 Strings

### 1. **C-Style Strings (Character Arrays)**
```cpp
#include <iostream>
#include <cstring>  // For string functions
using namespace std;

int main() {
    // Declaration and initialization
    char str1[20] = "Hello";           // Null-terminated
    char str2[] = "World";             // Size inferred
    char str3[10];                     // Uninitialized
    
    // Character by character initialization
    char str4[6] = {'H', 'e', 'l', 'l', 'o', '\0'};
    
    // Input/Output
    cout << "Enter a string: ";
    cin >> str3;  // Reads until whitespace
    cout << "You entered: " << str3 << endl;
    
    // String functions
    cout << "Length of str1: " << strlen(str1) << endl;
    
    // String concatenation
    strcat(str1, " ");
    strcat(str1, str2);
    cout << "Concatenated: " << str1 << endl;
    
    // String comparison
    if (strcmp(str1, "Hello World") == 0) {
        cout << "Strings are equal" << endl;
    }
    
    // String copy
    char copy[20];
    strcpy(copy, str1);
    cout << "Copied string: " << copy << endl;
    
    return 0;
}
```

### 2. **C++ String Class**
```cpp
#include <iostream>
#include <string>
using namespace std;

int main() {
    // Declaration and initialization
    string str1;                    // Empty string
    string str2 = "Hello";          // Direct initialization
    string str3("World");           // Constructor
    string str4(5, 'A');           // "AAAAA"
    string str5 = str2;            // Copy constructor
    
    // Input/Output
    cout << "Enter your name: ";
    getline(cin, str1);  // Reads entire line including spaces
    cout << "Hello, " << str1 << "!" << endl;
    
    // String operations
    cout << "str2: " << str2 << endl;
    cout << "Length: " << str2.length() << endl;
    cout << "Size: " << str2.size() << endl;
    cout << "Empty? " << (str2.empty() ? "Yes" : "No") << endl;
    
    // String concatenation
    string greeting = str2 + " " + str3;
    cout << "Greeting: " << greeting << endl;
    
    // Append
    str2 += " C++";
    cout << "After append: " << str2 << endl;
    
    return 0;
}
```

### 3. **String Access and Modification**
```cpp
#include <iostream>
#include <string>
using namespace std;

int main() {
    string text = "Programming";
    
    // Character access
    cout << "First character: " << text[0] << endl;           // 'P'
    cout << "Last character: " << text[text.length()-1] << endl; // 'g'
    
    // Safe character access
    cout << "Character at index 5: " << text.at(5) << endl;  // 'a'
    
    // Modify characters
    text[0] = 'p';  // Change to lowercase
    cout << "Modified: " << text << endl;
    
    // Iterating through string
    cout << "Characters: ";
    for (char c : text) {
        cout << c << " ";
    }
    cout << endl;
    
    // Using iterators
    cout << "Using iterators: ";
    for (auto it = text.begin(); it != text.end(); ++it) {
        cout << *it << " ";
    }
    cout << endl;
    
    return 0;
}
```

### 4. **String Methods and Operations**
```cpp
#include <iostream>
#include <string>
#include <algorithm>
using namespace std;

int main() {
    string text = "Hello, World! Welcome to C++ Programming.";
    
    // Substring
    string sub = text.substr(7, 5);  // Starting at index 7, length 5
    cout << "Substring: " << sub << endl;  // "World"
    
    // Find operations
    size_t pos = text.find("World");
    if (pos != string::npos) {
        cout << "'World' found at position: " << pos << endl;
    }
    
    // Find and replace
    string original = "I love Java";
    size_t java_pos = original.find("Java");
    if (java_pos != string::npos) {
        original.replace(java_pos, 4, "C++");
    }
    cout << "After replace: " << original << endl;
    
    // Insert and erase
    string sentence = "I programming";
    sentence.insert(2, "love ");
    cout << "After insert: " << sentence << endl;
    
    sentence.erase(2, 5);  // Remove "love "
    cout << "After erase: " << sentence << endl;
    
    // Case conversion
    string word = "Hello World";
    transform(word.begin(), word.end(), word.begin(), ::tolower);
    cout << "Lowercase: " << word << endl;
    
    transform(word.begin(), word.end(), word.begin(), ::toupper);
    cout << "Uppercase: " << word << endl;
    
    return 0;
}
```

### 5. **String Comparison and Searching**
```cpp
#include <iostream>
#include <string>
using namespace std;

int main() {
    string str1 = "apple";
    string str2 = "banana";
    string str3 = "apple";
    
    // Comparison operators
    cout << "str1 == str3: " << (str1 == str3) << endl;  // true
    cout << "str1 < str2: " << (str1 < str2) << endl;    // true (lexicographic)
    cout << "str2 > str1: " << (str2 > str1) << endl;    // true
    
    // Compare method
    int result = str1.compare(str2);
    if (result < 0) {
        cout << "str1 comes before str2 alphabetically" << endl;
    } else if (result > 0) {
        cout << "str1 comes after str2 alphabetically" << endl;
    } else {
        cout << "str1 and str2 are equal" << endl;
    }
    
    // Search operations
    string text = "The quick brown fox jumps over the lazy dog";
    
    // Count occurrences of a character
    int count = 0;
    for (char c : text) {
        if (c == 'o') count++;
    }
    cout << "Number of 'o' characters: " << count << endl;
    
    // Find all occurrences of a substring
    string search = "the";
    size_t pos = 0;
    cout << "Positions of 'the': ";
    while ((pos = text.find(search, pos)) != string::npos) {
        cout << pos << " ";
        pos++;
    }
    cout << endl;
    
    return 0;
}
```

## 🔧 String Utility Functions

### Custom String Functions
```cpp
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
using namespace std;

// Function to reverse a string
string reverseString(string str) {
    int start = 0;
    int end = str.length() - 1;
    
    while (start < end) {
        swap(str[start], str[end]);
        start++;
        end--;
    }
    return str;
}

// Function to check if string is palindrome
bool isPalindrome(string str) {
    string reversed = reverseString(str);
    return str == reversed;
}

// Function to count words
int countWords(const string& str) {
    istringstream iss(str);
    string word;
    int count = 0;
    
    while (iss >> word) {
        count++;
    }
    return count;
}

// Function to split string by delimiter
vector<string> splitString(const string& str, char delimiter) {
    vector<string> tokens;
    istringstream iss(str);
    string token;
    
    while (getline(iss, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

// Function to trim whitespace
string trim(const string& str) {
    size_t start = str.find_first_not_of(" \t\n\r");
    if (start == string::npos) return "";
    
    size_t end = str.find_last_not_of(" \t\n\r");
    return str.substr(start, end - start + 1);
}

// Function to remove all occurrences of a character
string removeChar(string str, char ch) {
    str.erase(remove(str.begin(), str.end(), ch), str.end());
    return str;
}

int main() {
    string text = "A man a plan a canal Panama";
    
    cout << "Original: " << text << endl;
    cout << "Reversed: " << reverseString(text) << endl;
    cout << "Is palindrome: " << (isPalindrome("racecar") ? "Yes" : "No") << endl;
    cout << "Word count: " << countWords(text) << endl;
    
    // Split example
    string csv = "apple,banana,orange,grape";
    vector<string> fruits = splitString(csv, ',');
    cout << "Fruits: ";
    for (const string& fruit : fruits) {
        cout << fruit << " ";
    }
    cout << endl;
    
    // Trim example
    string padded = "   Hello World   ";
    cout << "Trimmed: '" << trim(padded) << "'" << endl;
    
    // Remove character example
    string sentence = "Hello, World!";
    cout << "Without commas: " << removeChar(sentence, ',') << endl;
    
    return 0;
}
```

## 🎯 Practical Examples

### 1. **Student Grade System**
```cpp
#include <iostream>
#include <string>
#include <iomanip>
using namespace std;

int main() {
    const int NUM_STUDENTS = 3;
    const int NUM_SUBJECTS = 4;
    
    string students[NUM_STUDENTS] = {"Alice", "Bob", "Charlie"};
    string subjects[NUM_SUBJECTS] = {"Math", "Physics", "Chemistry", "Biology"};
    int grades[NUM_STUDENTS][NUM_SUBJECTS];
    
    // Input grades
    for (int i = 0; i < NUM_STUDENTS; i++) {
        cout << "\nEnter grades for " << students[i] << ":" << endl;
        for (int j = 0; j < NUM_SUBJECTS; j++) {
            cout << subjects[j] << ": ";
            cin >> grades[i][j];
        }
    }
    
    // Display grade report
    cout << "\n" << setw(10) << "Student";
    for (int j = 0; j < NUM_SUBJECTS; j++) {
        cout << setw(12) << subjects[j];
    }
    cout << setw(10) << "Average" << endl;
    
    cout << string(70, '-') << endl;
    
    for (int i = 0; i < NUM_STUDENTS; i++) {
        cout << setw(10) << students[i];
        
        int total = 0;
        for (int j = 0; j < NUM_SUBJECTS; j++) {
            cout << setw(12) << grades[i][j];
            total += grades[i][j];
        }
        
        double average = (double)total / NUM_SUBJECTS;
        cout << setw(10) << fixed << setprecision(1) << average << endl;
    }
    
    return 0;
}
```

### 2. **Text Analysis Program**
```cpp
#include <iostream>
#include <string>
#include <algorithm>
#include <cctype>
using namespace std;

int main() {
    string text;
    cout << "Enter a sentence: ";
    getline(cin, text);
    
    // Count statistics
    int characters = text.length();
    int letters = 0, digits = 0, spaces = 0, others = 0;
    int vowels = 0, consonants = 0;
    
    for (char c : text) {
        if (isalpha(c)) {
            letters++;
            char lower = tolower(c);
            if (lower == 'a' || lower == 'e' || lower == 'i' || 
                lower == 'o' || lower == 'u') {
                vowels++;
            } else {
                consonants++;
            }
        } else if (isdigit(c)) {
            digits++;
        } else if (isspace(c)) {
            spaces++;
        } else {
            others++;
        }
    }
    
    // Count words
    int words = 1;
    for (char c : text) {
        if (isspace(c)) {
            words++;
        }
    }
    if (text.empty() || text[0] == ' ') {
        words = 0;
    }
    
    // Display analysis
    cout << "\n=== Text Analysis ===" << endl;
    cout << "Text: \"" << text << "\"" << endl;
    cout << "Total characters: " << characters << endl;
    cout << "Letters: " << letters << endl;
    cout << "Vowels: " << vowels << endl;
    cout << "Consonants: " << consonants << endl;
    cout << "Digits: " << digits << endl;
    cout << "Spaces: " << spaces << endl;
    cout << "Other characters: " << others << endl;
    cout << "Words: " << words << endl;
    
    return 0;
}
```

### 3. **Simple Caesar Cipher**
```cpp
#include <iostream>
#include <string>
using namespace std;

string caesarEncrypt(string text, int shift) {
    string result = "";
    
    for (char c : text) {
        if (isalpha(c)) {
            char base = isupper(c) ? 'A' : 'a';
            result += char((c - base + shift) % 26 + base);
        } else {
            result += c;  // Keep non-alphabetic characters unchanged
        }
    }
    
    return result;
}

string caesarDecrypt(string text, int shift) {
    return caesarEncrypt(text, 26 - shift);
}

int main() {
    string message;
    int shift;
    
    cout << "Enter message: ";
    getline(cin, message);
    
    cout << "Enter shift value (1-25): ";
    cin >> shift;
    
    string encrypted = caesarEncrypt(message, shift);
    string decrypted = caesarDecrypt(encrypted, shift);
    
    cout << "\nOriginal: " << message << endl;
    cout << "Encrypted: " << encrypted << endl;
    cout << "Decrypted: " << decrypted << endl;
    
    return 0;
}
```

## 💡 Best Practices

### 1. **Array Safety**
```cpp
// Always check bounds
bool safeArrayAccess(int arr[], int size, int index) {
    return (index >= 0 && index < size);
}

// Use const for read-only arrays
void printArray(const int arr[], int size) {
    for (int i = 0; i < size; i++) {
        cout << arr[i] << " ";
    }
}

// Prefer std::array for fixed-size arrays (C++11)
#include <array>
array<int, 5> numbers = {1, 2, 3, 4, 5};
```

### 2. **String Efficiency**
```cpp
// Use const reference for string parameters
void processString(const string& str) {
    // Avoids copying the string
}

// Reserve capacity for large strings
string buildLargeString() {
    string result;
    result.reserve(1000);  // Avoid multiple reallocations
    
    for (int i = 0; i < 100; ++i) {
        result += "Some text ";
    }
    
    return result;
}

// Use string_view for read-only string operations (C++17)
#include <string_view>
void analyzeText(string_view text) {
    // No copying, just a view of the string
}
```

### 3. **Memory Management**
```cpp
// Prefer modern alternatives
#include <vector>
#include <array>

// Instead of dynamic arrays
// int* arr = new int[size];  // Manual memory management

// Use vectors
vector<int> dynamicArray(size);  // Automatic memory management

// Use arrays for fixed size
array<int, 10> fixedArray;  // Stack allocation, bounds checking
```

## 🔗 Related Topics
- [Functions](./05-functions.md)
- [Pointers & References](./07-pointers-references.md)
- [STL (Standard Template Library)](./15-stl.md)
- [Memory Management](./16-memory-management.md)

---
*Previous: [Functions](./05-functions.md) | Next: [Pointers & References](./07-pointers-references.md)*
