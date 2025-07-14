# Guide: Building a Simple Shell in C++

This guide will walk you through the process of creating a basic, cross-platform shell application in C++, similar to this project. We'll cover the core logic, handling configuration with JSON, and adding color to the output.

This guide is intended for those new to C++ or those looking to understand the inner workings of this project.

## Prerequisites

*   A C++ compiler (like G++, Clang, or MSVC). We'll use `g++` in our examples.
*   Basic knowledge of C++ syntax.
*   Git for fetching the required libraries.

## Core Concepts

A shell, at its heart, is a simple loop that does four things:

1.  **Read**: Read a command from the user.
2.  **Parse**: Break the command into a program and its arguments.
3.  **Execute**: Run the parsed command.
4.  **Loop**: Repeat the process.

Our `yoo.cpp` implements this with a few key functions:

*   `yoo_loop()`: The main loop that keeps the shell running.
*   `handle_input()`: Reads the user's command from standard input.
*   `validate_input()`: Parses the input and checks if it's a valid, known command.
*   `input_exicute()`: Executes the command using the operating system's `system()` call.

## Dependencies

Our shell uses two excellent, modern C++ libraries to make life easier. Both are header-only, which means you just need to download their source code and include them in your project—no complex building required.

### 1. `{fmt}` for Text Formatting and Colors

The `{fmt}` library is a fast, safe, and modern alternative to `printf` and `iostreams`.

*   **Why use it?** It provides a simple and powerful way to format strings and, importantly for us, to print colored text to the terminal.
*   **How to get it:**
    ```bash
    git clone https://github.com/fmtlib/fmt.git
    ```
    You just need to copy the `include/fmt` directory into your project.

*   **How to use it:** To print colored text, you include `fmt/color.h` and use `fmt::print` with a foreground color style.

    ```cpp
    #include "fmt/color.h"

    // Print "Hello" in red
    fmt::print(fmt::fg(fmt::color::red), "Hello\n");
    ```

### 2. `nlohmann/json` for Configuration

This library provides an intuitive way to work with JSON in C++.

*   **Why use it?** It allows us to easily load our shell's configuration (like colors and commands) from an external `config.json` file, making the shell customizable without recompiling.
*   **How to get it:** Download the single header file `json.hpp` from the [official repository](https://github.com/nlohmann/json) and place it in your project's include path.

*   **How to use it:**

    ```cpp
    #include %3Cfstream%3E
    #include "json.hpp"

    // for convenience
    using json = nlohmann::json;

    std::ifstream f("config.json");
    json data = json::parse(f);

    // Access data like a standard C++ container
    std::string shell_name = data["shell"]["print"]["name"];
    ```

## Building the Shell: Step-by-Step

Let's demonstrate the core logic with simplified examples.

### Step 1: The Main Loop

Every shell needs an infinite loop to keep accepting commands. The loop only breaks when the user types an exit command.

```cpp
// In main.cpp
void yoo_loop() {
    while(true) {
        // 1. Print the prompt
        fmt::print("MyShell> ");

        // 2. Read input
        std::string input = handle_input();

        // 3. Validate and execute
        if (validate_and_execute(input)) {
            // continue
        } 

        // 4. Handle exit
        if (input == "exit") {
            break;
        }
    }
}
```

### Step 2: Executing Commands

The simplest way to execute a command is with the `system()` function from `<cstdlib>`. It passes the command string directly to the host operating system's command processor.

```cpp
#include <cstdlib>

void execute_command(const std::string& command) {
    int return_code = system(command.c_str());
    if (return_code != 0) {
        fmt::print(fmt::fg(fmt::color::red), "Command failed!\n");
    }
}
```

### Step 3: Using JSON for Configuration

Let's read a shell name from `config.json`.

**`config.json`:**
```json
{
    "shell_name": "MyAwesomeShell"
}
```

**`main.cpp`:**
```cpp
#include <fstream>
#include "json.hpp"
#include "fmt/core.h"

int main() {
    std::ifstream f("config.json");
    nlohmann::json data = nlohmann::json::parse(f);

    std::string name = data["shell_name"];
    fmt::print("Welcome to {}\n", name);

    // ... start the shell loop ...
    return 0;
}
```

### Step 4: Using `{fmt}` for Colors

Now, let's make our prompt magenta using a hex color from our config file, just like in `yoo.cpp`.

**`config.json`:**
```json
{
    "prompt_color": "#FF00FF"
}
```

**`main.cpp`:**
```cpp
#include "fmt/color.h"

// Helper function to convert hex string to an RGB color object
inline fmt::rgb hex_to_rgb(const std::string& hex) {
    // (Implementation is the same as hex_to_rgb_my in config.h)
    // ... logic to parse hex string ...
    unsigned int r, g, b;
    std::stringstream ss;
    ss << std::hex << hex.substr(1);
    unsigned int hex_value;
    ss >> hex_value;
    r = (hex_value >> 16) & 0xFF;
    g = (hex_value >> 8) & 0xFF;
    b = hex_value & 0xFF;
    return fmt::rgb(r, g, b);
}

void yoo_loop() {
    // Load color from config...
    std::string prompt_hex_color = "#FF00FF"; // from config

    while(true) {
        fmt::print(fmt::fg(hex_to_rgb(prompt_hex_color)), "MyShell> ");
        // ... handle input ...
    }
}
```

## Conclusion

By combining a simple read-execute loop with the power of the `{fmt}` and `nlohmann/json` libraries, you can build a powerful and customizable shell. This project, `yoo-shell`, expands on these concepts to create a more feature-rich, cross-platform experience by using preprocessor directives (`#ifdef _WIN64`) to provide different command lists for different operating systems.

Feel free to explore the code in `yoo.cpp` and `config.cpp` to see these concepts implemented in a complete project.>)