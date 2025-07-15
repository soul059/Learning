# 🚀 Advanced Vim Concepts and Techniques

Explore advanced Vim features that will transform your editing efficiency. Master these concepts to unlock Vim's full potential.

## 📚 Learning Resources

### **Essential Reading**
1. [Modes of Vim](https://www.freecodecamp.org/news/vim-editor-modes-explained/)
2. [Advanced Vim Techniques](https://learnvim.irian.to/basics/moving_in_file)
3. [Vim Motion Tutorial Part 1](https://www.youtube.com/watch?v=lWTzqPfy1gE)
4. [Vim Motion Tutorial Part 2](https://www.youtube.com/watch?v=nBjEzQlJLHE)

## ⚙️ The Anatomy of Vim Commands

### **Command Structure**
```
[count] operator [count] motion
```

**Components:**
- **Count**: Number of times to repeat
- **Operator**: Action to perform (d, c, y, etc.)
- **Motion**: Where to apply the action

### **Examples**
```vim
d12j            # Delete 12 lines below cursor
y5k             # Yank 5 lines above cursor  
c3w             # Change 3 words forward
5dd             # Delete 5 lines
2ci"            # Change inside quotes (repeat twice)
```

### **Inner vs Around Objects**
Replace count with `i` (inner) or `a` (around):

```vim
diw             # Delete inner word (cursor anywhere in word)
daw             # Delete around word (includes whitespace)
ci"             # Change inside quotes
ca"             # Change around quotes (includes quotes)
```

**Key Insight**: Use inner/around for precision regardless of cursor position.

## 🎭 Advanced Mode Mastery

### **Normal Mode Excellence**
**Purpose**: Command and navigation hub
- Access via `Esc` or `Ctrl+[`
- Where you spend 80% of your time
- Every key is a command

**Advanced Techniques:**
```vim
u               # Undo
Ctrl+r          # Redo
.               # Repeat last change (most powerful!)
```

### **Insert Mode Efficiency**
**Purpose**: Text input only
- Enter for specific tasks, exit immediately
- Minimize time spent here

**Advanced Insert Commands:**
```vim
Ctrl+w          # Delete word backward
Ctrl+u          # Delete to line beginning
Ctrl+t          # Indent line
Ctrl+d          # Unindent line
Ctrl+n          # Auto-complete next
Ctrl+p          # Auto-complete previous
```

### **Visual Mode Mastery**

#### **Character-wise Visual** (`v`)
```vim
v               # Start character selection
{motion}        # Extend selection
{operator}      # Apply operation
```

#### **Line-wise Visual** (`V`)  
```vim
V               # Start line selection
j/k             # Extend by lines
d               # Delete selected lines
```

#### **Block Visual** (`Ctrl+v`)
```vim
Ctrl+v          # Start block selection
j/k             # Extend vertically
l/h             # Extend horizontally
I               # Insert at beginning of all lines
A               # Append at end of all lines
```

**Block Visual Use Cases:**
- Adding comments to multiple lines
- Editing columnar data
- Creating ASCII art
- Formatting tables

### **Command Mode Power**

**Ex Commands Structure:**
```vim
:[range]command[options]
```

**Advanced Range Examples:**
```vim
:1,5d           # Delete lines 1-5
:.,$s/old/new/g # Replace from current line to end
:'<,'>sort      # Sort visual selection
:/start/,/end/d # Delete from "start" to "end"
```

## 🎯 Advanced Text Objects

### **Custom Text Objects with Plugins**
```vim
" With vim-textobj-user plugin
ie              # Entire buffer
al              # Around line (with newline)
il              # Inner line (without newline)
ii              # Current indentation level
ai              # Around indentation level
```

### **Advanced Object Combinations**
```vim
" Nested operations
ci"             # Change inside quotes
ci'             # Change inside single quotes  
ci`             # Change inside backticks
ci{             # Change inside braces
ci[             # Change inside brackets
ci(             # Change inside parentheses
cit             # Change inside HTML/XML tag
```

### **Multi-level Objects**
```vim
" Working with nested structures
ca{             # Change around braces (includes braces)
da{             # Delete around braces
ya{             # Yank around braces
=a{             # Auto-indent around braces
```

## 🔄 Advanced Repetition and Automation

### **The Dot Command (`.`)**
**Most Important Command**: Repeats last change

**Effective Workflow:**
1. Make a change: `cw new_word<Esc>`
2. Move to next location: `n` (or any motion)
3. Repeat: `.`
4. Continue: `n.n.n.`

### **Macro Mastery**
```vim
qa              # Start recording macro in register 'a'
{commands}      # Perform your actions
q               # Stop recording
@a              # Execute macro 'a'
@@              # Repeat last macro
5@a             # Execute macro 5 times
```

**Advanced Macro Techniques:**
```vim
:let @a='...'   # Edit macro as text
:put a          # Paste macro content to see/edit
qA              # Append to existing macro 'a'
```

## 🔍 Advanced Search and Navigation

### **Smart Search Patterns**
```vim
/\<word\>       # Exact word boundaries
/word\c         # Case insensitive
/word\C         # Case sensitive
/^word          # Word at line beginning
/word$          # Word at line end
```

### **Advanced Substitution**
```vim
:%s/\(.*\):\(.*\)/\2:\1/g    # Swap around colon
:%s/\w\+/\U&/g               # Uppercase all words
:%s/\<\w/\u&/g               # Capitalize first letter
```

### **Global Commands**
```vim
:g/pattern/command           # Execute command on matching lines
:g/TODO/d                    # Delete all TODO lines
:g/function/t$               # Copy function lines to end
:g/import/m0                 # Move imports to top
```

## 📋 Advanced Register Usage

### **Named Registers**
```vim
"ay             # Yank to register 'a'
"Ay             # Append to register 'a'  
"ap             # Paste from register 'a'
```

### **Special Registers**
```vim
""              # Unnamed register (default)
"0              # Last yank register
"1-"9           # Delete history registers
"+              # System clipboard
"*              # Selection clipboard (X11)
"/              # Last search pattern
":              # Last command
".              # Last inserted text
"%              # Current filename
"#              # Alternate filename
```

## 🪟 Advanced Window Management

### **Window Splitting**
```vim
:sp filename    # Horizontal split with file
:vsp filename   # Vertical split with file
Ctrl+w s        # Split current window
Ctrl+w v        # Vertical split current window
```

### **Window Resizing**
```vim
Ctrl+w +        # Increase height
Ctrl+w -        # Decrease height  
Ctrl+w >        # Increase width
Ctrl+w <        # Decrease width
Ctrl+w =        # Equal sizes
Ctrl+w _        # Maximize height
Ctrl+w |        # Maximize width
```

### **Window Movement**
```vim
Ctrl+w H        # Move window to far left
Ctrl+w J        # Move window to bottom
Ctrl+w K        # Move window to top
Ctrl+w L        # Move window to far right
```

## 🎨 Advanced Customization

### **Custom Key Mappings**
```vim
" Normal mode mappings
nnoremap <leader>w :w<CR>
nnoremap <leader>q :q<CR>

" Insert mode mappings  
inoremap jj <Esc>
inoremap <C-l> <Right>

" Visual mode mappings
vnoremap < <gv
vnoremap > >gv

" Command mode mappings
cnoremap <C-a> <Home>
cnoremap <C-e> <End>
```

### **Advanced Functions**
```vim
function! ToggleNumber()
  if(&relativenumber == 1)
    set norelativenumber
    set number
  else
    set relativenumber
  endif
endfunc

nnoremap <leader>n :call ToggleNumber()<CR>
```

## 🚀 Workflow Optimization

### **Efficient Editing Patterns**
1. **Navigate first, then edit**: Use motions to position precisely
2. **Think in text objects**: `ciw`, `ci"`, `ca(`
3. **Use counts**: `3dd`, `5w`, `2ci"`
4. **Master the dot command**: Make one good change, repeat it
5. **Create macros for complex repetitive tasks**

### **Advanced Productivity Tips**
```vim
" Quick file switching
:b <Tab>        # Buffer completion
Ctrl+^          # Switch to previous buffer

" Fast navigation
gd              # Go to definition
*               # Search word under cursor
#               # Search word backward
``              # Jump to last position
```

## 🎯 Next Level Skills

Once you master these advanced concepts:
1. **Learn regular expressions** thoroughly
2. **Master plugin ecosystem** (NERDTree, fzf, etc.)
3. **Create custom text objects** with vim-textobj-user
4. **Build complex macros** for specific workflows
5. **Integrate with external tools** (git, make, etc.)

---

**Key Principle**: Advanced Vim mastery comes from combining simple commands in powerful ways. Master the fundamentals, then layer on complexity gradually! 🚀
