---
tags:
  - CodingLanguage/C
  - FileIO
---

In **C**, file input/output (I/O) is used to work with files to either read data from them or write data to them. The **<stdio.h>** header file provides a set of functions to perform file I/O operations. Here's a breakdown of the key concepts and usage:

---

### **Steps for File I/O in C**
1. **Open a File**: Use `fopen()` to open a file.
2. **Perform File Operations**: Use functions like `fprintf()`, `fscanf()`, `fwrite()`, `fread()`, etc.
3. **Close the File**: Use `fclose()` to close the file after operations.

---

### **File Opening Modes in `fopen()`**
When opening a file, you can specify modes like:
- `"r"`: Open for reading. File must exist.
- `"w"`: Open for writing. Creates a new file or overwrites an existing file.
- `"a"`: Open for appending. Data is written at the end of the file.
- `"r+"`: Open for both reading and writing.
- `"w+"`: Open for reading and writing (overwrites the file).
- `"a+"`: Open for reading and appending.

---

### **Example: Writing to a File**
```c
#include <stdio.h>

int main() {
    FILE *file = fopen("example.txt", "w"); // Open file in write mode

    if (file == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    fprintf(file, "Hello, Keval!\n"); // Write a line to the file
    fprintf(file, "Welcome to File I/O in C.\n");

    fclose(file); // Close the file
    printf("Data written to file successfully.\n");

    return 0;
}
```

This creates or overwrites a file named `example.txt` and writes two lines into it.

---

### **Example: Reading from a File**
```c
#include <stdio.h>

int main() {
    FILE *file = fopen("example.txt", "r"); // Open file in read mode

    if (file == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    char line[100];
    while (fgets(line, sizeof(line), file) != NULL) { // Read line by line
        printf("%s", line); // Print each line
    }

    fclose(file); // Close the file
    return 0;
}
```

This reads the content of `example.txt` line by line and prints it to the console.

---

### **Example: Appending to a File**
```c
#include <stdio.h>

int main() {
    FILE *file = fopen("example.txt", "a"); // Open file in append mode

    if (file == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    fprintf(file, "This is an appended line.\n"); // Append a line to the file

    fclose(file); // Close the file
    printf("Data appended successfully.\n");

    return 0;
}
```

This adds a new line at the end of the file without overwriting existing content.

---

### **Important File I/O Functions**
| **Function**         | **Description**                   |
| -------------------- | --------------------------------- |
| `fopen()`            | Opens a file.                     |
| `fclose()`           | Closes an opened file.            |
| `fprintf()`          | Writes formatted data to a file.  |
| `fscanf()`           | Reads formatted data from a file. |
| `fgets()`            | Reads a line of text from a file. |
| `fputc()`/`fgetc()`  | Writes/reads a single character.  |
| `fwrite()`/`fread()` | Writes/reads binary data.         |

---

### **Error Handling**
Always check the return value of `fopen()` to ensure the file was opened successfully. If the file cannot be opened, it returns `NULL`.

---

### **Applications**
- Storing user data (e.g., configuration files, logs).
- Processing and analyzing data from files (e.g., CSV files).
- Building database-like systems.
