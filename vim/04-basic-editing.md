# 04 - Basic Editing

Master the fundamental editing operations in Vim. Learn to insert, delete, copy, and paste text efficiently using Vim's powerful command system.

## 🎯 The Vim Editing Philosophy

### **Operator + Motion = Action**
Vim's power comes from combining operators (actions) with motions (movement):
```
d + w = delete word
c + $ = change to end of line
y + 3j = copy current line and 3 lines below
```

## ✏️ Insert Operations

### **Entering Insert Mode**
```
i    # Insert before cursor
I    # Insert at beginning of line
a    # Append after cursor
A    # Append at end of line
o    # Open new line below and insert
O    # Open new line above and insert
s    # Substitute character (delete char and insert)
S    # Substitute line (delete line and insert)
```

### **Insert Mode Tips**
- Use `Ctrl+w` to delete word backward
- Use `Ctrl+u` to delete to beginning of line
- Use `Ctrl+h` for backspace
- Use `Ctrl+t` to indent, `Ctrl+d` to unindent

### **Leaving Insert Mode Efficiently**
```
Esc        # Return to normal mode
Ctrl+[     # Alternative to Esc
Ctrl+c     # Quick exit (may skip some autocmds)
```

## ❌ Delete Operations

### **Character Deletion**
```
x     # Delete character under cursor
X     # Delete character before cursor
rx    # Replace character under cursor with 'x'
```

### **Word Deletion**
```
dw    # Delete word forward
db    # Delete word backward
dW    # Delete WORD forward
diw   # Delete inner word (whole word regardless of cursor position)
daw   # Delete a word (including surrounding whitespace)
```

### **Line Deletion**
```
dd    # Delete entire line
D     # Delete from cursor to end of line
d$    # Delete from cursor to end of line (same as D)
d^    # Delete from cursor to beginning of line
d0    # Delete from cursor to beginning of line
```

### **Advanced Deletion**
```
dt{char}  # Delete till character
df{char}  # Delete up to and including character
d/{pattern} # Delete until search pattern
d}        # Delete to end of paragraph
dG        # Delete to end of file
dgg       # Delete to beginning of file
```

## 📋 Copy and Paste (Yank and Put)

### **Copying (Yanking)**
```
yy    # Yank (copy) entire line
Y     # Yank entire line (same as yy)
yw    # Yank word
y$    # Yank to end of line
y^    # Yank to beginning of line
yiw   # Yank inner word
yaw   # Yank a word (with whitespace)
```

### **Pasting (Putting)**
```
p     # Put after cursor/below line
P     # Put before cursor/above line
```

### **Advanced Copying**
```
y5j   # Copy current line and 5 lines below
y}    # Copy to end of paragraph
yG    # Copy to end of file
ygg   # Copy to beginning of file
```

## 🔄 Change Operations

Change operations delete text and immediately enter insert mode:

```
cw    # Change word
c$    # Change to end of line
C     # Change to end of line (same as c$)
cc    # Change entire line
ciw   # Change inner word
caw   # Change a word
ct{char} # Change till character
```

## 🔁 Undo and Redo

```
u       # Undo last change
Ctrl+r  # Redo (undo the undo)
U       # Undo all changes on current line
.       # Repeat last change
```

### **Advanced Undo**
```
:earlier 10m  # Go to state 10 minutes ago
:later 5m     # Go to state 5 minutes later
:undolist     # Show undo tree
```

## 🔢 Using Counts with Operators

Multiply the effect of any operation:

```
3dd   # Delete 3 lines
5x    # Delete 5 characters
2dw   # Delete 2 words
4yy   # Copy 4 lines
10p   # Paste 10 times
```

## 🎯 Text Objects

Text objects define what to operate on:

### **Inner Objects (i)**
```
iw    # inner word
is    # inner sentence
ip    # inner paragraph
i(    # inner parentheses
i[    # inner brackets
i{    # inner braces
i"    # inner quotes
i'    # inner single quotes
```

### **Around Objects (a)**
```
aw    # a word (includes whitespace)
as    # a sentence
ap    # a paragraph
a(    # around parentheses
a[    # around brackets
a{    # around braces
a"    # around quotes
a'    # around single quotes
```

### **Examples**
```
diw   # Delete inner word
ci(   # Change inside parentheses
ya"   # Yank around quotes
dap   # Delete around paragraph
```

## 🎮 Practice Exercises

### Exercise 1: Basic Insert and Delete
```
1. Open a file with some text
2. Practice each insert command (i, I, a, A, o, O)
3. Practice character deletion (x, X)
4. Practice word deletion (dw, db, diw)
5. Practice line deletion (dd, D)
```

### Exercise 2: Copy and Paste
```
Text: "The quick brown fox jumps over the lazy dog"

1. Copy the word "quick" using 'yiw'
2. Move to end of line and paste with 'p'
3. Copy entire line with 'yy'
4. Paste above current line with 'P'
5. Copy 3 words with '3yw'
```

### Exercise 3: Change Operations
```
1. Change a word with 'cw'
2. Change to end of line with 'c$'
3. Change inside quotes with 'ci"'
4. Change around parentheses with 'ca('
```

### Exercise 4: Text Objects Practice
```
Text: "function calculate(price, discount) { return price * discount; }"

1. Delete inside parentheses: 'di('
2. Change inside braces: 'ci{'
3. Yank around function name: position cursor on 'calculate' and 'yaw'
4. Delete the entire function: 'daw' or 'da}'
```

## 🏆 Advanced Editing Techniques

### **Joining Lines**
```
J     # Join current line with next line
gJ    # Join lines without adding space
5J    # Join current line with next 4 lines
```

### **Changing Case**
```
~     # Switch case of character under cursor
guu   # Make line lowercase
gUU   # Make line uppercase
guw   # Make word lowercase
gUw   # Make word uppercase
```

### **Increment/Decrement Numbers**
```
Ctrl+a  # Increment number under cursor
Ctrl+x  # Decrement number under cursor
5Ctrl+a # Add 5 to number under cursor
```

### **Indentation**
```
>>    # Indent line
<<    # Unindent line
5>>   # Indent 5 lines
>}    # Indent to end of paragraph
=     # Auto-indent (use with motions)
gg=G  # Auto-indent entire file
```

## 🔧 Configuration for Better Editing

### **Better Backspace**
```vim
set backspace=indent,eol,start
```

### **Auto-indentation**
```vim
set autoindent
set smartindent
```

### **Tab Settings**
```vim
set expandtab      # Use spaces instead of tabs
set tabstop=4      # Tab width
set shiftwidth=4   # Indent width
set softtabstop=4  # Soft tab width
```

### **Show Whitespace**
```vim
set list
set listchars=tab:▸\ ,eol:¬,space:·
```

## 🚫 Common Mistakes

### **Staying in Insert Mode Too Long**
- **Problem**: Using arrow keys in insert mode
- **Solution**: Exit to normal mode, navigate, then re-enter insert

### **Not Using Text Objects**
- **Problem**: Manually selecting text boundaries
- **Solution**: Use `iw`, `aw`, `i(`, etc. for precise selection

### **Forgetting the Dot Command**
- **Problem**: Repeating complex operations manually
- **Solution**: Use `.` to repeat last change

### **Not Using Counts**
- **Problem**: Repeating single operations
- **Solution**: Use counts like `3dd`, `5x`, `2dw`

## 🎯 Quick Reference

### **Essential Operators**
```
d  # Delete
c  # Change
y  # Yank (copy)
p  # Put (paste)
r  # Replace
~  # Change case
>  # Indent
<  # Unindent
=  # Auto-indent
```

### **Essential Motions**
```
w/b/e  # Word movements
0/^/$  # Line movements
f/t    # Character search
{/}    # Paragraph movements
```

### **Essential Text Objects**
```
iw/aw  # Word
i(/a(  # Parentheses
i"/a"  # Quotes
ip/ap  # Paragraph
```

## 🎯 Next Steps

Master these editing fundamentals, then explore:
1. [Text Objects](./05-text-objects.md) for more precise editing
2. [Search and Replace](./06-search-replace.md) for bulk operations
3. Practice combining operators with motions daily
4. Learn to use the dot command for efficiency

---

**Key Principle**: Think in terms of "what" you want to do (operator) and "where" you want to do it (motion/text object). This is the path to Vim mastery! 🎯
