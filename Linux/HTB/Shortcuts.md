# Useful Linux Terminal Shortcuts

Mastering keyboard shortcuts can significantly speed up and simplify your work in the Linux terminal, reducing the need for mouse usage and saving typing time. Here are some essential shortcuts:

## Auto-Complete

* **`[TAB]`**: Initiates auto-completion. Suggests and completes commands, file names, directory names, and options based on your input. Pressing `[TAB]` twice often shows all possible completions.

## Cursor Movement

* **`[CTRL] + A`**: Moves the cursor to the **beginning** of the current line.
* **`[CTRL] + E`**: Moves the cursor to the **end** of the current line.
* **`[CTRL] + [←]` / `[→]`**: Jumps the cursor to the beginning of the **previous** (`[←]`) or **current** (`[→]`) word.
* **`[ALT] + B` / `F`**: Jumps backward (`B`) or forward (`F`) one word.

## Erase Text

* **`[CTRL] + U`**: Erases everything from the current cursor position to the **beginning** of the line.
* **`[Ctrl] + K`**: Erases everything from the current cursor position to the **end** of the line.
* **`[Ctrl] + W`**: Erases the word immediately preceding the cursor position.

## Paste Erased Contents

* **`[Ctrl] + Y`**: Pastes the text or word that was most recently erased using `[CTRL] + U`, `[Ctrl] + K`, or `[Ctrl] + W`.

## End Task / Process

* **`[CTRL] + C`**: Sends the `SIGINT` signal to the current foreground process, typically ending or interrupting it without confirmation. Useful for stopping running commands or scripts.

## End-of-File (EOF)

* **`[CTRL] + D`**: Signals the end of input (End-of-File or End-of-Transmission) to a program reading from standard input. Can also be used to exit a shell session if no process is reading input.

## Clear Terminal

* **`[CTRL] + L`**: Clears the terminal screen, providing a clean workspace. This is equivalent to typing the `clear` command.

## Background a Process

* **`[CTRL] + Z`**: Sends the `SIGTSTP` signal to the current foreground process, suspending its execution and moving it to the background.

## Search Through Command History

* **`[CTRL] + R`**: Initiates an interactive search through your command history. Type characters, and the shell will show the most recent command matching your input. Press `[Ctrl] + R` again to cycle through older matches.
* **`[↑]` / `[↓]`**: Scrolls through previously executed commands one by one (up for previous, down for next).

## Switch Between Applications

* **`[ALT] + [TAB]`**: Switches between currently open graphical applications on your desktop environment.

## Zoom (in some terminal emulators)

* **`[CTRL] + [+]`**: Zooms in on the terminal text size.
* **`[CTRL] + [-]`**: Zooms out on the terminal text size.

Incorporating these shortcuts into your workflow will significantly boost your efficiency and comfort level when working with the Linux command line. Practice using them regularly until they become second nature.