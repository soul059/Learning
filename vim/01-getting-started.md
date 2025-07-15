# 01 - Getting Started with Vim

Welcome to Vim! This guide will help you take your first steps with the most powerful text editor ever created.

## 🔧 Installation

### Linux/macOS
```bash
# Ubuntu/Debian
sudo apt install vim

# macOS with Homebrew
brew install vim

# CentOS/RHEL
sudo yum install vim
```

### Windows
- Download from [vim.org](https://www.vim.org/download.php)
- Or use Windows Subsystem for Linux (WSL)
- Git Bash includes vim

## 🎯 Why Learn Vim?

### **Advantages**
- **Speed**: Once mastered, editing becomes incredibly fast
- **Ubiquity**: Available on virtually every system
- **No Mouse Dependency**: Pure keyboard-driven workflow
- **Powerful**: Handles any text editing task
- **Customizable**: Highly configurable and extensible

### **Philosophy**
- **Modal Editing**: Different modes for different tasks
- **Composability**: Commands combine logically
- **Efficiency**: Minimize keystrokes and hand movement

## 🚀 Your First Vim Session

### Starting Vim
```bash
vim filename.txt    # Open a file
vim                 # Start with empty buffer
vimtutor           # Interactive tutorial (highly recommended!)
```

### Essential First Commands
```
:help              # Access help system
:q                 # Quit
:q!                # Quit without saving
:w                 # Save (write)
:wq                # Save and quit
ZZ                 # Save and quit (shortcut)
```

## 🗝️ The Golden Rules

### **Rule 1: Stay on Home Row**
- Your hands should rest on `asdf` (left) and `jkl;` (right)
- Avoid arrow keys - use `hjkl` instead

### **Rule 2: Escape is Your Friend**
- Press `Esc` to return to Normal mode
- Consider mapping Caps Lock to Escape

### **Rule 3: Practice Little and Often**
- 15 minutes daily beats 2 hours weekly
- Use Vim for small tasks first

## 🎮 First Steps Practice

### Exercise 1: Basic Movement
1. Open a file: `vim practice.txt`
2. Press `i` to enter Insert mode
3. Type some text
4. Press `Esc` to return to Normal mode
5. Use `h`, `j`, `k`, `l` to move around
6. Save with `:w` and quit with `:q`

### Exercise 2: Simple Editing
1. Open your file again
2. Move to a word and press `x` to delete a character
3. Press `u` to undo
4. Press `Ctrl+r` to redo
5. Practice until comfortable

## 🔧 Essential Setup

### Basic `.vimrc` Configuration
Create `~/.vimrc` (Linux/macOS) or `_vimrc` (Windows):

```vim
" Basic settings
set number              " Show line numbers
set relativenumber      " Show relative line numbers
set hlsearch           " Highlight search results
set incsearch          " Incremental search
set ignorecase         " Case insensitive search
set smartcase          " Case sensitive if uppercase used
set autoindent         " Auto indent new lines
set expandtab          " Use spaces instead of tabs
set tabstop=4          " Tab width
set shiftwidth=4       " Indent width
set softtabstop=4      " Soft tab width

" Better backspace behavior
set backspace=indent,eol,start

" Show matching brackets
set showmatch

" Enable syntax highlighting
syntax on

" Enable file type detection
filetype on
filetype plugin on
filetype indent on
```

## 🎯 Next Steps

Once you're comfortable with basic navigation:
1. Complete the built-in `vimtutor`
2. Move on to [Vim Modes](./02-vim-modes.md)
3. Practice daily with real files
4. Join the Vim community online

## 🆘 Getting Help

### In Vim
```
:help              # General help
:help :command     # Help for specific command
:help mode         # Help for specific mode
:helpgrep pattern  # Search help files
```

### External Resources
- `man vim` - System manual
- [Vim Wiki](https://vim.fandom.com/wiki/Vim_Tips_Wiki)
- [r/vim](https://reddit.com/r/vim) - Reddit community

## 💡 Pro Tips for Beginners

1. **Start Small**: Use Vim for simple tasks first
2. **Learn Gradually**: Master basics before advanced features
3. **Use Vimtutor**: Complete it multiple times
4. **Map Caps Lock**: Makes Escape easier to reach
5. **Be Patient**: The learning curve is steep but worth it

---

**Remember**: Everyone struggles with Vim initially. Persistence is key! 🔑
