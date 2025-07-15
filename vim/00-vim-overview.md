# 🚀 Vim Learning Path - Core Concepts

This file provides a foundational understanding of Vim's core concepts and serves as a bridge to the comprehensive learning materials.

## 📚 Learning Resources

### **Primary Learning Path**
Follow the structured learning path in this repository:
1. [01 - Getting Started](./01-getting-started.md)
2. [02 - Vim Modes](./02-vim-modes.md) 
3. [03 - Basic Navigation](./03-basic-navigation.md)
4. [04 - Basic Editing](./04-basic-editing.md)
5. [Continue through all modules...](./README.md)

### **External Resources**
- [Vim Hero](https://www.vim-hero.com) - Interactive vim learning
- [Vimtutor](command:vimtutor) - Built-in interactive tutorial
- [Practical Vim Book](https://pragprog.com/titles/dnvim2/practical-vim-second-edition/)

## 🎭 Understanding Vim Modes

### **Normal Mode** (Default)
**Purpose**: Navigation and text manipulation
- Moving the cursor efficiently
- Deleting, copying, and pasting text
- Executing commands and entering other modes
- **Key Point**: This is where you spend most of your time

### **Insert Mode** 
**Purpose**: Text input and editing
- Similar to traditional text editors
- Type characters that appear on screen
- **Key Point**: Enter only to add text, exit immediately after

### **Visual Mode**
**Purpose**: Text selection and manipulation
- Select text for operations
- Three variants: character, line, and block selection

### **Command-line Mode**
**Purpose**: Execute Ex commands
- File operations (save, quit, etc.)
- Search and replace
- Configuration changes

## 🧭 Movement Fundamentals

### **Why hjkl Instead of Arrow Keys?**
- **Efficiency**: Hands stay on home row
- **Speed**: No hand movement required  
- **Philosophy**: Minimize movement, maximize productivity

### **Basic Movement Pattern**
```
h ← j ↓ k ↑ l →
```

### **Word Movement**
- `w` → move to **next word**
- `e` → move to **end** of current word  
- `b` → move **back** to previous word

**What Defines a Word?**
In Vim, a word consists of letters, digits, underscores, OR a sequence of non-blank characters separated by whitespace.

### **Advanced Movement**
- `gg` → Go to first line of file
- `G` → Go to last line of file
- `0` → Beginning of line
- `$` → End of line
- `f{char}` → Find character in line
- `%` → Jump between matching brackets `( ) [ ] { }`

### **Sentence and Paragraph Movement**
- `(` , `)` → Move back and forward by sentence
- `{` , `}` → Move by paragraph

## ✏️ Insert Mode Basics

### **Entering Insert Mode**
- `i` → Insert before cursor
- `a` → Insert after cursor  
- `I` → Insert at beginning of line
- `A` → Insert at end of line
- `o` → Open new line below
- `O` → Open new line above

### **Exiting Insert Mode**
- `Esc` → Return to normal mode
- Consider mapping Caps Lock to Escape for efficiency

## 🎯 Basic Editing Operations

### **Character Operations**
- `x` → Delete character under cursor
- `r{char}` → Replace character under cursor
- `s` → Substitute character (delete and enter insert mode)

### **Word Operations**
- `dw` → Delete word
- `cw` → Change word
- `yiw` → Copy inner word

### **Line Operations**
- `dd` → Delete entire line
- `cc` → Change entire line
- `yy` → Copy entire line

## 🎯 Text Objects (Power Feature)

Text objects represent logical text structures:

### **Inner vs Around**
- `i` → Inner (content only)
- `a` → Around (content + delimiters/whitespace)

### **Common Text Objects**
- `iw`/`aw` → Inner/around word
- `i"`/`a"` → Inner/around quotes
- `i(`/`a(` → Inner/around parentheses
- `i{`/`a{` → Inner/around braces

### **Examples**
- `ci"` → Change inside quotes (cursor anywhere in quotes)
- `da(` → Delete around parentheses  
- `yiw` → Copy inner word

## 🔄 Undo and Repeat

- `u` → Undo last change
- `Ctrl+r` → Redo
- `.` → Repeat last change (most powerful command!)

## 🎮 Practice Workflow

1. **Start with vimtutor**: Built-in interactive tutorial
2. **Use hjkl daily**: Force good habits by disabling arrow keys
3. **Master text objects**: They're incredibly powerful
4. **Practice operator+motion**: Think in combinations
5. **Learn one new thing weekly**: Gradual improvement

## 📖 Next Steps

Once comfortable with these basics:
1. Learn search and replace patterns
2. Master buffer and window management  
3. Explore macros for automation
4. Customize your vimrc configuration
5. Install useful plugins

---

**Remember**: Vim has a steep learning curve, but the productivity gains are enormous. Start with basics and build gradually! 🚀

For the complete learning experience, follow the structured modules in this repository starting with [Getting Started](./01-getting-started.md).
