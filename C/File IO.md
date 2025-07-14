---
tags:
  - Coding
  - FileIO
---
**File I/O (Input/Output)** in programming refers to the process of reading from or writing to files stored on a computer. It enables programs to save data for later use or process data from external files. In **C++**, File I/O is handled using the `<fstream>` library, which provides three main classes:

1. **ofstream**: For writing to files (output).
2. **ifstream**: For reading from files (input).
3. **fstream**: For both reading from and writing to files.

---

### **Steps for File I/O in C++**

1. Include the `<fstream>` library.
2. Create an object of `ofstream`, `ifstream`, or `fstream`.
3. Open the file using `.open()` or directly during initialization.
4. Perform read/write operations using the file stream object.
5. Close the file using `.close()` to free system resources.

---

### **Example 1: Writing to a File**
```cpp
#include <iostream>
#include <fstream>
using namespace std;

int main() {
    ofstream outFile("example.txt"); // Create and open a file for writing

    if (outFile.is_open()) {
        outFile << "Hello, Keval!" << endl;
        outFile << "Welcome to File I/O in C++." << endl;
        outFile.close(); // Close the file
        cout << "Data written to file successfully." << endl;
    } else {
        cout << "Unable to open file for writing." << endl;
    }

    return 0;
}
```

This creates a file named `example.txt` and writes two lines into it.

---

### **Example 2: Reading from a File**
```cpp
#include <iostream>
#include <fstream>
#include <string>
using namespace std;

int main() {
    ifstream inFile("example.txt"); // Open the file for reading
    string line;

    if (inFile.is_open()) {
        while (getline(inFile, line)) { // Read line by line
            cout << line << endl; // Print each line to the console
        }
        inFile.close(); // Close the file
    } else {
        cout << "Unable to open file for reading." << endl;
    }

    return 0;
}
```

This reads the contents of `example.txt` and prints it to the console.

---

### **Example 3: Reading and Writing with `fstream`**
```cpp
#include <iostream>
#include <fstream>
using namespace std;

int main() {
    fstream file("example.txt", ios::in | ios::out | ios::app); // Open file for read, write, and append

    if (file.is_open()) {
        // Write to the file
        file << "This is an appended line." << endl;

        // Move the file pointer back to the beginning to read from the file
        file.seekg(0, ios::beg);

        // Read and display the file's content
        string line;
        while (getline(file, line)) {
            cout << line << endl;
        }

        file.close(); // Close the file
    } else {
        cout << "Unable to open file." << endl;
    }

    return 0;
}
```

This appends a line to `example.txt` and then reads the entire file.

---

### **File Modes in C++**
When opening a file, you can specify the mode using flags like:
- `ios::in`: Open for reading.
- `ios::out`: Open for writing.
- `ios::app`: Append to the file (write at the end).
- `ios::trunc`: Truncate the file (delete existing content).
- `ios::binary`: Open file in binary mode.

---

### **Error Handling**
Always check if the file was successfully opened using `.is_open()` to prevent runtime errors.

---

### **Applications of File I/O**
- **Storing User Data**: Save user preferences or progress in a game.
- **Configuration Files**: Load settings for an application.
- **Data Processing**: Process large datasets stored in files.
