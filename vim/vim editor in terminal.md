# 🖥️ Vim Editor in Terminal - Command Reference

Essential commands and configurations for using Vim effectively in terminal environments.

## 📁 Configuration Files

### **vimrc Location**
```bash
# Linux/macOS
~/.vimrc              # Main configuration file

# Windows  
$HOME/_vimrc          # Main configuration file
```

## 💻 Essential Terminal Commands

### **Starting Vim**
```bash
vim                   # Start with empty buffer
vim filename.txt      # Open specific file
vim +10 file.txt      # Open file at line 10
vim +/pattern file    # Open file and search for pattern
vim -R file.txt       # Open in read-only mode
vimtutor             # Interactive tutorial
```

### **Exit Commands**
```vim
:q                   # Quit (fails if unsaved changes)
:q!                  # Quit without saving (abandon changes)
:wq                  # Write and quit
:x                   # Write and quit (only if changes made)
ZZ                   # Write and quit (normal mode)
:qa!                 # Abandon all changes and exit all windows
```

**Important**: Use `!` to force commands when there are unsaved changes.

## 🔍 Search and Replace in Terminal

### **Basic Search**
```vim
/{pattern}           # Search forward
?{pattern}           # Search backward
n                    # Next occurrence
N                    # Previous occurrence
*                    # Search word under cursor
```

### **Search and Replace**
```vim
:s/old/new/          # Replace first occurrence in line
:s/old/new/g         # Replace all in line
:%s/old/new/g        # Replace all in file
:%s/old/new/gc       # Replace with confirmation
```

**Advanced Replace Examples:**
```vim
# Replace in specific lines
:1,10s/old/new/g     # Lines 1-10
:.,+5s/old/new/g     # Current line plus next 5

# Case insensitive replace
:%s/old/new/gi

# Use last search pattern
:%s//new/g           # Replaces last searched pattern
```

### **Clear Search Highlighting**
```vim
:nohlsearch          # Clear highlighting
:noh                 # Short form

# Add to .vimrc for easy access:
nnoremap <Esc><Esc> :nohlsearch<CR>
```

## 📂 File Operations

### **File Management**
```vim
:w                   # Save current file
:w filename          # Save as filename
:w! filename         # Force save as filename
:e filename          # Edit new file
:e!                  # Reload current file (discard changes)
:r filename          # Read file into current buffer
:r !command          # Read command output into buffer
```

### **Buffer Management**
```vim
:ls                  # List all buffers
:b filename          # Switch to buffer
:b#                  # Switch to previous buffer
:bn                  # Next buffer
:bp                  # Previous buffer
:bd                  # Delete current buffer
```

## 🔧 Configuration Commands

### **Settings (Set Command)**
```vim
:set number          # Show line numbers
:set nonumber        # Hide line numbers
:set relativenumber  # Show relative line numbers
:set hlsearch        # Highlight search results
:set ignorecase      # Case insensitive search
:set smartcase       # Case sensitive if uppercase used
:set autoindent      # Auto indent new lines
:set expandtab       # Use spaces instead of tabs
:set tabstop=4       # Tab width
:set shiftwidth=4    # Indent width
```

### **Viewing Settings**
```vim
:set                 # Show all modified settings
:set all             # Show all settings
:set number?         # Show specific setting value
```

## 🧹 Text Manipulation

### **Sorting**
```vim
:sort                # Sort lines alphabetically
:sort!               # Reverse sort
:sort u              # Sort and remove duplicates
:sort n              # Numeric sort
```

### **Line Operations**
```vim
:g/pattern/d         # Delete all lines matching pattern
:g!/pattern/d        # Delete lines NOT matching pattern
:v/pattern/d         # Same as above (v = inverse of g)
:%d                  # Delete all lines
```

### **Global Commands**
```vim
:g/TODO/p            # Print all lines containing "TODO"
:g/function/nu       # Show line numbers for lines with "function"
:g/^$/d              # Delete all empty lines
```

## 📋 Register Operations

### **Viewing Registers**
```vim
:reg                 # Show all registers
:reg a               # Show specific register
:reg abc             # Show multiple registers
```

### **Using Registers**
```vim
"ay                  # Yank to register 'a'
"ap                  # Paste from register 'a'
"+y                  # Yank to system clipboard
"+p                  # Paste from system clipboard
```

## 🔧 Terminal Integration

### **Running Shell Commands**
```vim
:!command            # Execute shell command
:!ls                 # List directory contents
:!date               # Show current date
:r !ls               # Read ls output into buffer
```

### **Terminal-Specific Features**
```vim
# Better mouse support
:set mouse=a

# Clipboard integration
:set clipboard=unnamedplus    # Linux
:set clipboard=unnamed        # macOS

# Color support
:set termguicolors           # True color support
```

## 🎨 Terminal Appearance

### **Basic Colors**
```vim
:syntax on           # Enable syntax highlighting
:colorscheme desert  # Set color scheme
:set background=dark # Dark background
```

### **Line Display**
```vim
:set number relativenumber   # Show both line numbers
:set cursorline             # Highlight current line
:set ruler                  # Show cursor position
:set laststatus=2           # Always show status line
```

## 🚀 Performance Tips

### **For Large Files**
```vim
:set lazyredraw      # Don't redraw during macros
:syntax off          # Disable syntax highlighting
:set nowrap          # Don't wrap lines
```

### **Memory Management**
```vim
:set undolevels=100  # Limit undo levels
:set maxmempattern=2000  # Limit regex memory
```

## 🔍 Help System

### **Getting Help**
```vim
:help                # General help
:help :command       # Help for specific command
:help function       # Help for function
:help 'option'       # Help for option
:helpgrep pattern    # Search help files
```

### **Navigation in Help**
```vim
Ctrl+]               # Follow link
Ctrl+o               # Go back
:q                   # Quit help
```

## 💡 Pro Tips for Terminal Use

### **Essential .vimrc for Terminal**
```vim
" Basic settings for terminal use
set nocompatible
set number relativenumber
set ignorecase smartcase
set incsearch hlsearch
set autoindent
set expandtab tabstop=4 shiftwidth=4
set backspace=indent,eol,start
set mouse=a
set clipboard=unnamedplus
set laststatus=2

" Quick escape
inoremap jj <Esc>

" Clear search
nnoremap <Esc><Esc> :nohlsearch<CR>

" Quick save
nnoremap <Leader>w :w<CR>
```

### **Useful Aliases**
```bash
# Add to your shell configuration
alias vi='vim'
alias view='vim -R'  # Read-only mode
```

### **Terminal Multiplexer Integration**
Works great with tmux or screen:
```bash
# Copy mode integration
set -g mode-keys vi  # In tmux.conf
```

---

**Remember**: Vim in terminal is incredibly powerful. These commands form the foundation of efficient text editing in any Unix-like environment! 🚀

clipboard item stored in **+** register 

-> pasting
		`"<Register name>p`
-> coping 
		`"<Rregister name>yy`