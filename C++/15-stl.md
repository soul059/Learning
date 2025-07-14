# 15. STL (Standard Template Library)

## 📋 Overview
The Standard Template Library (STL) is a collection of template classes and functions that provide common data structures and algorithms. It consists of containers, iterators, algorithms, and function objects.

## 📦 Containers

### 1. **Sequence Containers**

#### vector - Dynamic Array
```cpp
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int main() {
    // Declaration and initialization
    vector<int> vec1;                           // Empty vector
    vector<int> vec2(5);                        // 5 elements, default initialized to 0
    vector<int> vec3(5, 10);                    // 5 elements, all set to 10
    vector<int> vec4 = {1, 2, 3, 4, 5};        // Initializer list
    vector<int> vec5(vec4);                     // Copy constructor
    
    // Adding elements
    vec1.push_back(10);
    vec1.push_back(20);
    vec1.push_back(30);
    
    // Accessing elements
    cout << "First element: " << vec1[0] << endl;          // No bounds checking
    cout << "Second element: " << vec1.at(1) << endl;      // With bounds checking
    cout << "Last element: " << vec1.back() << endl;
    cout << "First element: " << vec1.front() << endl;
    
    // Size and capacity
    cout << "Size: " << vec1.size() << endl;
    cout << "Capacity: " << vec1.capacity() << endl;
    cout << "Empty: " << (vec1.empty() ? "Yes" : "No") << endl;
    
    // Iterating
    cout << "Elements: ";
    for (int x : vec1) {
        cout << x << " ";
    }
    cout << endl;
    
    // Using iterators
    cout << "Using iterators: ";
    for (auto it = vec1.begin(); it != vec1.end(); ++it) {
        cout << *it << " ";
    }
    cout << endl;
    
    // Modifying
    vec1.insert(vec1.begin() + 1, 15);  // Insert 15 at position 1
    vec1.erase(vec1.begin() + 2);       // Remove element at position 2
    vec1.pop_back();                    // Remove last element
    
    // Resize and reserve
    vec1.resize(10);        // Resize to 10 elements
    vec1.reserve(20);       // Reserve capacity for 20 elements
    
    // Clear all elements
    vec1.clear();
    cout << "After clear, size: " << vec1.size() << endl;
    
    return 0;
}
```

#### list - Doubly Linked List
```cpp
#include <iostream>
#include <list>
using namespace std;

int main() {
    list<int> lst = {3, 1, 4, 1, 5, 9, 2, 6};
    
    // Adding elements
    lst.push_back(7);
    lst.push_front(0);
    
    // Display list
    cout << "List: ";
    for (int x : lst) {
        cout << x << " ";
    }
    cout << endl;
    
    // Insert and erase
    auto it = lst.begin();
    advance(it, 3);  // Move iterator to 4th position
    lst.insert(it, 99);
    
    // Remove elements
    lst.remove(1);      // Remove all occurrences of 1
    lst.pop_front();    // Remove first element
    lst.pop_back();     // Remove last element
    
    // Sort and unique
    lst.sort();
    lst.unique();       // Remove consecutive duplicates
    
    cout << "After operations: ";
    for (int x : lst) {
        cout << x << " ";
    }
    cout << endl;
    
    return 0;
}
```

#### deque - Double-ended Queue
```cpp
#include <iostream>
#include <deque>
using namespace std;

int main() {
    deque<int> dq;
    
    // Add elements
    dq.push_back(10);
    dq.push_back(20);
    dq.push_front(5);
    dq.push_front(1);
    
    cout << "Deque: ";
    for (int x : dq) {
        cout << x << " ";  // Output: 1 5 10 20
    }
    cout << endl;
    
    // Access elements
    cout << "Front: " << dq.front() << endl;  // 1
    cout << "Back: " << dq.back() << endl;    // 20
    cout << "At index 2: " << dq[2] << endl; // 10
    
    // Remove elements
    dq.pop_front();
    dq.pop_back();
    
    cout << "After removing front and back: ";
    for (int x : dq) {
        cout << x << " ";  // Output: 5 10
    }
    cout << endl;
    
    return 0;
}
```

### 2. **Associative Containers**

#### set - Unique Sorted Elements
```cpp
#include <iostream>
#include <set>
using namespace std;

int main() {
    set<int> s = {3, 1, 4, 1, 5, 9, 2, 6, 5};  // Duplicates automatically removed
    
    // Display set (automatically sorted)
    cout << "Set: ";
    for (int x : s) {
        cout << x << " ";  // Output: 1 2 3 4 5 6 9
    }
    cout << endl;
    
    // Insert elements
    s.insert(8);
    s.insert(1);  // Duplicate, won't be added
    
    // Find elements
    if (s.find(5) != s.end()) {
        cout << "Found 5 in set" << endl;
    }
    
    // Count elements (0 or 1 for set)
    cout << "Count of 5: " << s.count(5) << endl;
    
    // Remove elements
    s.erase(3);                    // Remove by value
    s.erase(s.find(9));           // Remove by iterator
    
    // Range operations
    auto lower = s.lower_bound(4);  // First element >= 4
    auto upper = s.upper_bound(6);  // First element > 6
    
    cout << "Range [4, 6]: ";
    for (auto it = lower; it != upper; ++it) {
        cout << *it << " ";
    }
    cout << endl;
    
    return 0;
}
```

#### map - Key-Value Pairs
```cpp
#include <iostream>
#include <map>
#include <string>
using namespace std;

int main() {
    map<string, int> ages;
    
    // Insert elements
    ages["Alice"] = 25;
    ages["Bob"] = 30;
    ages["Charlie"] = 22;
    ages.insert({"Diana", 28});
    ages.insert(make_pair("Eve", 35));
    
    // Access elements
    cout << "Alice's age: " << ages["Alice"] << endl;
    cout << "Bob's age: " << ages.at("Bob") << endl;  // at() throws exception if key not found
    
    // Iterate through map
    cout << "All ages:" << endl;
    for (const auto& pair : ages) {
        cout << pair.first << ": " << pair.second << endl;
    }
    
    // Find element
    auto it = ages.find("Charlie");
    if (it != ages.end()) {
        cout << "Found Charlie, age: " << it->second << endl;
    }
    
    // Check if key exists
    if (ages.count("Frank") == 0) {
        cout << "Frank not found" << endl;
    }
    
    // Remove elements
    ages.erase("Bob");
    
    // Size and empty check
    cout << "Map size: " << ages.size() << endl;
    cout << "Empty: " << (ages.empty() ? "Yes" : "No") << endl;
    
    return 0;
}
```

#### multiset and multimap
```cpp
#include <iostream>
#include <set>
#include <map>
using namespace std;

int main() {
    // multiset - allows duplicates
    multiset<int> ms = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3};
    
    cout << "Multiset: ";
    for (int x : ms) {
        cout << x << " ";  // Duplicates preserved, sorted
    }
    cout << endl;
    
    cout << "Count of 1: " << ms.count(1) << endl;  // 2
    cout << "Count of 5: " << ms.count(5) << endl;  // 2
    
    // multimap - allows duplicate keys
    multimap<string, int> scores;
    scores.insert({"Alice", 85});
    scores.insert({"Alice", 92});
    scores.insert({"Bob", 78});
    scores.insert({"Alice", 88});
    
    cout << "All scores:" << endl;
    for (const auto& pair : scores) {
        cout << pair.first << ": " << pair.second << endl;
    }
    
    // Find all values for a key
    auto range = scores.equal_range("Alice");
    cout << "Alice's scores: ";
    for (auto it = range.first; it != range.second; ++it) {
        cout << it->second << " ";
    }
    cout << endl;
    
    return 0;
}
```

### 3. **Unordered Containers (C++11)**

#### unordered_set and unordered_map
```cpp
#include <iostream>
#include <unordered_set>
#include <unordered_map>
#include <string>
using namespace std;

int main() {
    // unordered_set - hash table implementation
    unordered_set<string> words = {"apple", "banana", "cherry", "date"};
    
    // Fast insertion and lookup (average O(1))
    words.insert("elderberry");
    
    if (words.find("banana") != words.end()) {
        cout << "Found banana" << endl;
    }
    
    cout << "Words: ";
    for (const string& word : words) {
        cout << word << " ";  // Order is not guaranteed
    }
    cout << endl;
    
    // unordered_map - hash table for key-value pairs
    unordered_map<string, int> inventory;
    inventory["apples"] = 50;
    inventory["bananas"] = 30;
    inventory["oranges"] = 25;
    
    // Fast access
    cout << "Apples in inventory: " << inventory["apples"] << endl;
    
    // Performance information
    cout << "Bucket count: " << inventory.bucket_count() << endl;
    cout << "Load factor: " << inventory.load_factor() << endl;
    
    return 0;
}
```

### 4. **Container Adaptors**

#### stack, queue, priority_queue
```cpp
#include <iostream>
#include <stack>
#include <queue>
using namespace std;

int main() {
    // Stack (LIFO - Last In, First Out)
    stack<int> stk;
    stk.push(10);
    stk.push(20);
    stk.push(30);
    
    cout << "Stack elements (top to bottom): ";
    while (!stk.empty()) {
        cout << stk.top() << " ";
        stk.pop();
    }
    cout << endl;
    
    // Queue (FIFO - First In, First Out)
    queue<int> q;
    q.push(10);
    q.push(20);
    q.push(30);
    
    cout << "Queue elements (front to back): ";
    while (!q.empty()) {
        cout << q.front() << " ";
        q.pop();
    }
    cout << endl;
    
    // Priority Queue (max-heap by default)
    priority_queue<int> pq;
    pq.push(30);
    pq.push(10);
    pq.push(50);
    pq.push(20);
    
    cout << "Priority queue (highest priority first): ";
    while (!pq.empty()) {
        cout << pq.top() << " ";
        pq.pop();
    }
    cout << endl;
    
    // Min-heap priority queue
    priority_queue<int, vector<int>, greater<int>> min_pq;
    min_pq.push(30);
    min_pq.push(10);
    min_pq.push(50);
    min_pq.push(20);
    
    cout << "Min priority queue (lowest priority first): ";
    while (!min_pq.empty()) {
        cout << min_pq.top() << " ";
        min_pq.pop();
    }
    cout << endl;
    
    return 0;
}
```

## 🔄 Iterators

### Iterator Types and Usage
```cpp
#include <iostream>
#include <vector>
#include <list>
#include <iterator>
using namespace std;

int main() {
    vector<int> vec = {1, 2, 3, 4, 5};
    
    // Different types of iterators
    vector<int>::iterator it;                    // Mutable iterator
    vector<int>::const_iterator cit;             // Immutable iterator
    vector<int>::reverse_iterator rit;           // Reverse iterator
    vector<int>::const_reverse_iterator crit;    // Const reverse iterator
    
    // Forward iteration
    cout << "Forward: ";
    for (it = vec.begin(); it != vec.end(); ++it) {
        cout << *it << " ";
    }
    cout << endl;
    
    // Reverse iteration
    cout << "Reverse: ";
    for (rit = vec.rbegin(); rit != vec.rend(); ++rit) {
        cout << *it << " ";
    }
    cout << endl;
    
    // Auto keyword (C++11)
    cout << "Using auto: ";
    for (auto it = vec.begin(); it != vec.end(); ++it) {
        cout << *it << " ";
    }
    cout << endl;
    
    // Iterator arithmetic (random access iterators)
    auto it2 = vec.begin();
    it2 += 2;  // Move to 3rd element
    cout << "Element at position 2: " << *it2 << endl;
    
    cout << "Distance from begin to it2: " << distance(vec.begin(), it2) << endl;
    
    // Iterator categories with list (bidirectional iterator)
    list<int> lst = {1, 2, 3, 4, 5};
    auto lit = lst.begin();
    ++lit;  // Forward
    --lit;  // Backward
    // lit += 2;  // Error: list iterators don't support random access
    
    advance(lit, 2);  // Safe way to move iterator by n positions
    cout << "List element after advancing by 2: " << *lit << endl;
    
    return 0;
}
```

## 🔧 Algorithms

### 1. **Non-modifying Algorithms**
```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
using namespace std;

int main() {
    vector<int> vec = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5};
    
    // Finding elements
    auto it = find(vec.begin(), vec.end(), 5);
    if (it != vec.end()) {
        cout << "Found 5 at position: " << distance(vec.begin(), it) << endl;
    }
    
    // Count elements
    int count_5 = count(vec.begin(), vec.end(), 5);
    cout << "Number of 5s: " << count_5 << endl;
    
    // Count with condition
    int even_count = count_if(vec.begin(), vec.end(), [](int x) { return x % 2 == 0; });
    cout << "Even numbers: " << even_count << endl;
    
    // Find with condition
    auto even_it = find_if(vec.begin(), vec.end(), [](int x) { return x % 2 == 0; });
    if (even_it != vec.end()) {
        cout << "First even number: " << *even_it << endl;
    }
    
    // Check conditions
    bool all_positive = all_of(vec.begin(), vec.end(), [](int x) { return x > 0; });
    bool any_negative = any_of(vec.begin(), vec.end(), [](int x) { return x < 0; });
    bool none_zero = none_of(vec.begin(), vec.end(), [](int x) { return x == 0; });
    
    cout << "All positive: " << (all_positive ? "Yes" : "No") << endl;
    cout << "Any negative: " << (any_negative ? "Yes" : "No") << endl;
    cout << "None zero: " << (none_zero ? "Yes" : "No") << endl;
    
    // Min/max elements
    auto min_it = min_element(vec.begin(), vec.end());
    auto max_it = max_element(vec.begin(), vec.end());
    cout << "Min element: " << *min_it << endl;
    cout << "Max element: " << *max_it << endl;
    
    // Accumulate (sum)
    int sum = accumulate(vec.begin(), vec.end(), 0);
    cout << "Sum: " << sum << endl;
    
    return 0;
}
```

### 2. **Modifying Algorithms**
```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
using namespace std;

int main() {
    vector<int> vec = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5};
    vector<int> dest(vec.size());
    
    // Copy
    copy(vec.begin(), vec.end(), dest.begin());
    
    // Copy with condition
    vector<int> evens;
    copy_if(vec.begin(), vec.end(), back_inserter(evens), 
            [](int x) { return x % 2 == 0; });
    
    cout << "Even numbers: ";
    for (int x : evens) {
        cout << x << " ";
    }
    cout << endl;
    
    // Transform (modify each element)
    vector<int> squares(vec.size());
    transform(vec.begin(), vec.end(), squares.begin(), 
              [](int x) { return x * x; });
    
    cout << "Squares: ";
    for (int x : squares) {
        cout << x << " ";
    }
    cout << endl;
    
    // Replace
    vector<int> vec2 = vec;
    replace(vec2.begin(), vec2.end(), 5, 99);  // Replace all 5s with 99
    
    cout << "After replace: ";
    for (int x : vec2) {
        cout << x << " ";
    }
    cout << endl;
    
    // Fill
    vector<int> vec3(10);
    fill(vec3.begin(), vec3.end(), 42);
    
    // Generate
    vector<int> vec4(10);
    int counter = 0;
    generate(vec4.begin(), vec4.end(), [&counter]() { return ++counter; });
    
    cout << "Generated: ";
    for (int x : vec4) {
        cout << x << " ";
    }
    cout << endl;
    
    // Remove (doesn't actually remove, returns new end)
    vector<int> vec5 = {1, 2, 3, 2, 4, 2, 5};
    auto new_end = remove(vec5.begin(), vec5.end(), 2);
    vec5.erase(new_end, vec5.end());  // Actually remove
    
    cout << "After removing 2s: ";
    for (int x : vec5) {
        cout << x << " ";
    }
    cout << endl;
    
    return 0;
}
```

### 3. **Sorting and Searching**
```cpp
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int main() {
    vector<int> vec = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5};
    
    // Sort
    sort(vec.begin(), vec.end());
    cout << "Sorted: ";
    for (int x : vec) {
        cout << x << " ";
    }
    cout << endl;
    
    // Sort with custom comparator
    vector<int> vec2 = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5};
    sort(vec2.begin(), vec2.end(), greater<int>());  // Descending order
    
    cout << "Sorted descending: ";
    for (int x : vec2) {
        cout << x << " ";
    }
    cout << endl;
    
    // Partial sort
    vector<int> vec3 = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5};
    partial_sort(vec3.begin(), vec3.begin() + 3, vec3.end());  // Sort first 3 elements
    
    cout << "Partial sort (first 3): ";
    for (int x : vec3) {
        cout << x << " ";
    }
    cout << endl;
    
    // Binary search (on sorted container)
    bool found = binary_search(vec.begin(), vec.end(), 5);
    cout << "Binary search for 5: " << (found ? "Found" : "Not found") << endl;
    
    // Lower and upper bound
    auto lower = lower_bound(vec.begin(), vec.end(), 5);
    auto upper = upper_bound(vec.begin(), vec.end(), 5);
    
    cout << "Range of 5s: [" << distance(vec.begin(), lower) 
         << ", " << distance(vec.begin(), upper) << ")" << endl;
    
    // Equal range
    auto range = equal_range(vec.begin(), vec.end(), 5);
    cout << "Count of 5s: " << distance(range.first, range.second) << endl;
    
    return 0;
}
```

## 🎯 Practical Examples

### 1. **Word Frequency Counter**
```cpp
#include <iostream>
#include <map>
#include <string>
#include <sstream>
#include <algorithm>
using namespace std;

int main() {
    string text = "the quick brown fox jumps over the lazy dog the fox is quick";
    
    // Convert to lowercase and count words
    transform(text.begin(), text.end(), text.begin(), ::tolower);
    
    map<string, int> wordCount;
    istringstream iss(text);
    string word;
    
    while (iss >> word) {
        wordCount[word]++;
    }
    
    cout << "Word frequencies:" << endl;
    for (const auto& pair : wordCount) {
        cout << pair.first << ": " << pair.second << endl;
    }
    
    // Find most frequent word
    auto maxWord = max_element(wordCount.begin(), wordCount.end(),
        [](const pair<string, int>& a, const pair<string, int>& b) {
            return a.second < b.second;
        });
    
    cout << "\nMost frequent word: " << maxWord->first 
         << " (appears " << maxWord->second << " times)" << endl;
    
    return 0;
}
```

### 2. **Student Grade Management**
```cpp
#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <numeric>
#include <string>
using namespace std;

struct Student {
    string name;
    vector<double> grades;
    
    double getAverage() const {
        if (grades.empty()) return 0.0;
        return accumulate(grades.begin(), grades.end(), 0.0) / grades.size();
    }
};

int main() {
    vector<Student> students = {
        {"Alice", {95, 87, 92, 88, 91}},
        {"Bob", {78, 85, 81, 79, 83}},
        {"Charlie", {92, 88, 94, 90, 96}},
        {"Diana", {85, 89, 87, 91, 88}}
    };
    
    // Display all students
    cout << "All students:" << endl;
    for (const auto& student : students) {
        cout << student.name << ": Average = " << student.getAverage() << endl;
    }
    
    // Sort by average grade (descending)
    sort(students.begin(), students.end(), 
         [](const Student& a, const Student& b) {
             return a.getAverage() > b.getAverage();
         });
    
    cout << "\nStudents sorted by average (highest first):" << endl;
    for (const auto& student : students) {
        cout << student.name << ": " << student.getAverage() << endl;
    }
    
    // Find student with highest single grade
    auto maxGradeStudent = max_element(students.begin(), students.end(),
        [](const Student& a, const Student& b) {
            auto maxA = max_element(a.grades.begin(), a.grades.end());
            auto maxB = max_element(b.grades.begin(), b.grades.end());
            return *maxA < *maxB;
        });
    
    auto maxGrade = max_element(maxGradeStudent->grades.begin(), 
                               maxGradeStudent->grades.end());
    
    cout << "\nHighest single grade: " << *maxGrade 
         << " by " << maxGradeStudent->name << endl;
    
    // Class statistics
    vector<double> allGrades;
    for (const auto& student : students) {
        allGrades.insert(allGrades.end(), student.grades.begin(), student.grades.end());
    }
    
    double classAverage = accumulate(allGrades.begin(), allGrades.end(), 0.0) / allGrades.size();
    auto minGrade = min_element(allGrades.begin(), allGrades.end());
    auto maxGradeOverall = max_element(allGrades.begin(), allGrades.end());
    
    cout << "\nClass Statistics:" << endl;
    cout << "Average: " << classAverage << endl;
    cout << "Lowest grade: " << *minGrade << endl;
    cout << "Highest grade: " << *maxGradeOverall << endl;
    
    return 0;
}
```

### 3. **Simple Text Editor with STL**
```cpp
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <sstream>
using namespace std;

class SimpleTextEditor {
private:
    vector<string> lines;
    vector<vector<string>> history;  // For undo functionality

public:
    void addLine(const string& line) {
        saveState();
        lines.push_back(line);
    }
    
    void insertLine(int position, const string& line) {
        saveState();
        if (position >= 0 && position <= lines.size()) {
            lines.insert(lines.begin() + position, line);
        }
    }
    
    void deleteLine(int position) {
        saveState();
        if (position >= 0 && position < lines.size()) {
            lines.erase(lines.begin() + position);
        }
    }
    
    void replaceLine(int position, const string& newLine) {
        saveState();
        if (position >= 0 && position < lines.size()) {
            lines[position] = newLine;
        }
    }
    
    void findAndReplace(const string& find, const string& replace) {
        saveState();
        for (auto& line : lines) {
            size_t pos = 0;
            while ((pos = line.find(find, pos)) != string::npos) {
                line.replace(pos, find.length(), replace);
                pos += replace.length();
            }
        }
    }
    
    vector<int> searchLines(const string& searchTerm) const {
        vector<int> results;
        for (int i = 0; i < lines.size(); ++i) {
            if (lines[i].find(searchTerm) != string::npos) {
                results.push_back(i);
            }
        }
        return results;
    }
    
    void sortLines() {
        saveState();
        sort(lines.begin(), lines.end());
    }
    
    void undo() {
        if (!history.empty()) {
            lines = history.back();
            history.pop_back();
        }
    }
    
    void display() const {
        cout << "\n--- Document ---" << endl;
        for (int i = 0; i < lines.size(); ++i) {
            cout << i << ": " << lines[i] << endl;
        }
        cout << "--- End ---" << endl;
    }
    
    int getLineCount() const { return lines.size(); }
    
    int getWordCount() const {
        int count = 0;
        for (const auto& line : lines) {
            istringstream iss(line);
            string word;
            while (iss >> word) {
                count++;
            }
        }
        return count;
    }

private:
    void saveState() {
        history.push_back(lines);
        if (history.size() > 10) {  // Keep only last 10 states
            history.erase(history.begin());
        }
    }
};

int main() {
    SimpleTextEditor editor;
    
    // Add some content
    editor.addLine("Hello World");
    editor.addLine("This is line 2");
    editor.addLine("Another line here");
    editor.addLine("Hello again");
    
    editor.display();
    
    // Search for lines containing "Hello"
    auto results = editor.searchLines("Hello");
    cout << "\nLines containing 'Hello': ";
    for (int lineNum : results) {
        cout << lineNum << " ";
    }
    cout << endl;
    
    // Replace all occurrences of "Hello" with "Hi"
    editor.findAndReplace("Hello", "Hi");
    editor.display();
    
    // Insert a line at position 1
    editor.insertLine(1, "Inserted line");
    editor.display();
    
    // Sort all lines
    editor.sortLines();
    editor.display();
    
    // Undo last operation
    editor.undo();
    cout << "\nAfter undo:";
    editor.display();
    
    // Statistics
    cout << "\nDocument statistics:" << endl;
    cout << "Lines: " << editor.getLineCount() << endl;
    cout << "Words: " << editor.getWordCount() << endl;
    
    return 0;
}
```

## 💡 Best Practices

### 1. **Container Selection**
```cpp
// Use vector for:
// - Random access needed
// - Frequent insertion/deletion at end
// - Cache-friendly iteration

// Use list for:
// - Frequent insertion/deletion in middle
// - No random access needed

// Use deque for:
// - Insertion/deletion at both ends
// - Random access needed

// Use set/map for:
// - Sorted data needed
// - Logarithmic search time

// Use unordered_set/unordered_map for:
// - Fast average lookup time
// - Order doesn't matter
```

### 2. **Algorithm Usage**
```cpp
// Prefer algorithms over manual loops
vector<int> vec = {1, 2, 3, 4, 5};

// Instead of:
int sum = 0;
for (int x : vec) {
    sum += x;
}

// Use:
int sum = accumulate(vec.begin(), vec.end(), 0);

// Use lambda functions with algorithms
auto isEven = [](int x) { return x % 2 == 0; };
int evenCount = count_if(vec.begin(), vec.end(), isEven);
```

### 3. **Iterator Safety**
```cpp
vector<int> vec = {1, 2, 3, 4, 5};

// Be careful with iterator invalidation
for (auto it = vec.begin(); it != vec.end(); ) {
    if (*it % 2 == 0) {
        it = vec.erase(it);  // erase returns next valid iterator
    } else {
        ++it;
    }
}

// Prefer range-based for when possible
for (const auto& element : vec) {
    cout << element << " ";
}
```

## 🔗 Related Topics
- [Templates](./12-templates.md)
- [Memory Management](./16-memory-management.md)
- [Advanced Topics](./17-advanced-topics.md)

---
*Previous: [File I/O](./14-file-io.md) | Next: [Memory Management](./16-memory-management.md)*
