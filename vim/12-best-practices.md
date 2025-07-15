# 12 - Best Practices and Productivity Tips

Master these best practices to achieve maximum productivity with Vim. Learn workflow optimizations, common pitfalls to avoid, and pro tips for daily Vim usage.

## 🎯 Core Principles

### **1. Think in Motions, Not Positions**
```
# Instead of:
llllllllll    # Moving character by character

# Do this:
w             # Move by word
5w            # Move 5 words
f{char}       # Find character
/{pattern}    # Search for pattern
```

### **2. Minimize Hand Movement**
- Stay on home row (`asdf` and `jkl;`)
- Use `hjkl` instead of arrow keys
- Map Caps Lock to Escape
- Learn to touch type if you haven't

### **3. Master the Dot Command**
```
.    # Repeat last change
```
The dot command is your best friend for repetitive tasks.

### **4. Embrace Modal Editing**
- Spend most time in Normal mode
- Use Insert mode only for adding text
- Exit Insert mode immediately after typing

## 🚫 Why Not Use Arrow Keys?

### **The Problem**
- Forces hand movement away from home row
- Slower than `hjkl`
- Breaks Vim's efficiency philosophy

### **The Solution**
Disable arrow keys to force good habits:
```vim
" In your .vimrc
noremap <Up> <NOP>
noremap <Down> <NOP>
noremap <Left> <NOP>
noremap <Right> <NOP>
inoremap <Up> <NOP>
inoremap <Down> <NOP>
inoremap <Left> <NOP>
inoremap <Right> <NOP>
```

### **Finger Positioning**
```
Left hand:  a s d f
Right hand: j k l ;
```
Your right index finger handles both `h` and `j`.

## ⚡ Speed Optimization Techniques

### **1. Word Movement Over Character Movement**
```
# Slow:
hjkl repeatedly

# Fast:
w, b, e       # Word movements
f{char}       # Find character
t{char}       # Till character
```

### **2. Use Counts Effectively**
```
5j            # Move down 5 lines
3w            # Move forward 3 words
2dd           # Delete 2 lines
4yy           # Copy 4 lines
```

### **3. Master Text Objects**
```
diw           # Delete inner word
ci"           # Change inside quotes
ya(           # Yank around parentheses
dap           # Delete around paragraph
```

### **4. Efficient Line Operations**
```
0             # Start of line
^             # First non-blank character
$             # End of line
A             # Append at end of line
I             # Insert at beginning of line
```

## 🎮 When to Use Different Operators

### **Replace vs Substitute**
```
r{char}       # Replace single character (stay in normal mode)
s             # Substitute character (enter insert mode)
```

**Use `r` when**: Quick single character fix
**Use `s` when**: Replacing with multiple characters

### **Find vs Search**
```
f{char}       # Find character in current line
/{pattern}    # Search across entire file
```

**Use `f/F/t/T` when**: Navigating within a line
**Use `/` and `?` when**: Finding text across multiple lines

### **Word Movement Variants**
```
w/W           # Next word/WORD
e/E           # End of word/WORD
b/B           # Back word/WORD
```

**Use lowercase**: For programming (respects punctuation)
**Use uppercase**: For prose (space-separated only)

## 🔧 Essential Vim Configuration

### **Minimal `.vimrc` for Productivity**
```vim
" Basic settings
set number relativenumber    " Line numbers
set ignorecase smartcase    " Smart search
set incsearch hlsearch      " Better search
set autoindent              " Auto indent
set expandtab               " Spaces instead of tabs
set tabstop=4 shiftwidth=4  " Tab settings
set backspace=indent,eol,start  " Better backspace

" Disable arrow keys
noremap <Up> <NOP>
noremap <Down> <NOP>
noremap <Left> <NOP>
noremap <Right> <NOP>

" Better search clearing
nnoremap <Esc><Esc> :nohlsearch<CR>

" Quick save
nnoremap <C-s> :w<CR>
inoremap <C-s> <Esc>:w<CR>

" Quick quit
nnoremap <leader>q :q<CR>

" Set leader key
let mapleader = " "
```

### **Essential Mappings**
```vim
" Quick escape alternatives
inoremap jj <Esc>
inoremap jk <Esc>

" Better window navigation
nnoremap <C-h> <C-w>h
nnoremap <C-j> <C-w>j
nnoremap <C-k> <C-w>k
nnoremap <C-l> <C-w>l

" Quick indentation
vnoremap < <gv
vnoremap > >gv

" Move lines up/down
nnoremap <leader>j :m .+1<CR>==
nnoremap <leader>k :m .-2<CR>==
```

## 📋 Effective Copy/Paste Workflow

### **Understanding Registers**
```vim
:reg          # View all registers
"ay           # Yank to register 'a'
"ap           # Paste from register 'a'
"+y           # Yank to system clipboard
"+p           # Paste from system clipboard
```

### **Smart Copy/Paste**
```vim
# Copy line without newline
0y$

# Copy word under cursor
yiw

# Copy inside quotes/brackets
yi"   yi'   yi(   yi[   yi{

# Paste and indent properly
]p    # Paste and adjust indentation
[p    # Paste above and adjust indentation
```

## 🔍 Search and Replace Best Practices

### **Efficient Search Patterns**
```vim
/\<word\>     # Exact word match
/word\c       # Case insensitive search
/word\C       # Case sensitive search
```

### **Smart Replace Workflow**
```vim
# 1. Search first
/old_text

# 2. Replace with confirmation
:%s//new_text/gc

# 3. Or replace all
:%s//new_text/g
```

### **Useful Search Shortcuts**
```vim
*             # Search word under cursor forward
#             # Search word under cursor backward
gd            # Go to local definition
gD            # Go to global definition
```

## 🏃‍♂️ Movement Efficiency

### **Short Distance (0-5 characters)**
```
h, j, k, l    # Character movement
```

### **Medium Distance (5-20 characters)**
```
w, b, e       # Word movement
f, t          # Character search
0, ^, $       # Line boundaries
```

### **Long Distance (20+ characters)**
```
/{pattern}    # Search
*, #          # Search word under cursor
gg, G         # File boundaries
H, M, L       # Screen positions
```

### **Cross-File Navigation**
```
:e filename   # Edit file
:b partial    # Switch to buffer
Ctrl+^        # Switch to previous buffer
:ls           # List buffers
```

## 🎭 Mode Management

### **Staying in Normal Mode**
- Use Normal mode for navigation
- Enter Insert mode only to add text
- Exit Insert mode immediately after typing

### **Efficient Mode Switching**
```
# Quick escapes
Esc           # Standard escape
Ctrl+[        # Alternative escape
Ctrl+c        # Quick escape

# Smart insert positions
i             # Insert here
a             # Insert after
I             # Insert at line start
A             # Insert at line end
o             # New line below
O             # New line above
```

## 🔄 Repetition and Automation

### **The Dot Command**
```
.             # Repeat last change
```

**Workflow Example:**
1. `cw` - Change word
2. Type new word
3. `Esc` - Return to normal mode
4. Move to next word
5. `.` - Repeat the change

### **Macro Basics**
```
qq            # Start recording macro 'q'
{commands}    # Perform operations
q             # Stop recording
@q            # Execute macro
5@q           # Execute macro 5 times
```

## 🚫 Common Anti-Patterns

### **1. Character-by-Character Movement**
```
# Don't do this:
hjkl hjkl hjkl

# Do this instead:
3w            # Move 3 words
f{char}       # Find character
/{pattern}    # Search
```

### **2. Staying in Insert Mode**
```
# Don't: Navigate in insert mode with arrows
# Do: Esc → navigate → insert again
```

### **3. Not Using Text Objects**
```
# Don't: Manual selection
# Do: diw, ci", ya(, etc.
```

### **4. Ignoring the Dot Command**
```
# Don't: Repeat operations manually
# Do: Use . to repeat last change
```

### **5. Mouse Usage**
```
# Don't: Use mouse for selection/navigation
# Do: Use keyboard-only workflow
```

## 📈 Productivity Metrics

### **Signs You're Improving**
- Rarely use arrow keys
- Automatically use text objects
- Think in operator+motion combinations
- Use counts naturally (3w, 5j, 2dd)
- Rarely enter Visual mode for simple operations

### **Practice Routine**
1. **Daily basics**: 10 minutes of hjkl navigation
2. **Weekly focus**: Pick one new feature to master
3. **Monthly review**: Audit your workflow for inefficiencies
4. **Challenge yourself**: Time common editing tasks

## 🎯 Advanced Workflow Tips

### **Buffer Management**
```vim
:ls           # List all buffers
:b#           # Switch to previous buffer
:bd           # Delete current buffer
:bn           # Next buffer
:bp           # Previous buffer
```

### **Window Management**
```vim
:sp           # Horizontal split
:vsp          # Vertical split
Ctrl+w +      # Increase height
Ctrl+w -      # Decrease height
Ctrl+w =      # Equal window sizes
```

### **Tab Management**
```vim
:tabnew       # New tab
:tabc         # Close tab
gt            # Next tab
gT            # Previous tab
```

## 💡 Pro Tips

### **1. Caps Lock Mapping**
Map Caps Lock to Escape for easier access:
- **macOS**: System Preferences → Keyboard → Modifier Keys
- **Windows**: Use software like [Uncap](https://github.com/susam/uncap)
- **Linux**: `setxkbmap -option caps:escape`

### **2. Learn One New Thing Weekly**
- Week 1: Master hjkl navigation
- Week 2: Learn text objects
- Week 3: Master search and replace
- Week 4: Learn macros
- Continue building...

### **3. Practice with Real Work**
Don't just do exercises - use Vim for actual work:
- Edit configuration files
- Write documentation
- Code your projects
- Take notes

### **4. Join the Community**
- Follow [r/vim](https://reddit.com/r/vim)
- Read Vim blogs and tips
- Share your discoveries
- Learn from others' workflows

## 🎯 Next Level

Once you've mastered these basics:
1. Learn about plugins and package managers
2. Explore advanced features like folding
3. Master regular expressions
4. Create custom mappings and functions
5. Contribute to the Vim ecosystem

---

**Remember**: Vim mastery is a journey, not a destination. Focus on steady progress and building muscle memory through consistent practice! 🚀
