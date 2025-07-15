# 🎓 Vim Quick Reference and Cheat Sheet

A comprehensive quick reference for all Vim commands, organized by category for easy lookup during editing sessions.

## 🚀 Modes Quick Reference

| Mode | Enter | Exit | Purpose |
|------|-------|------|---------|
| Normal | `Esc` | N/A | Navigation, commands |
| Insert | `i`, `a`, `o`, `I`, `A`, `O` | `Esc` | Text input |
| Visual | `v`, `V`, `Ctrl+v` | `Esc` | Text selection |
| Command | `:`, `/`, `?` | `Enter`, `Esc` | Ex commands, search |

## 🧭 Movement Commands

### **Character Movement**
```
h j k l    # Left, down, up, right
0          # Beginning of line
^          # First non-blank character
$          # End of line
g_         # Last non-blank character
```

### **Word Movement**
```
w          # Next word
W          # Next WORD (space-separated)
b          # Previous word
B          # Previous WORD
e          # End of word
E          # End of WORD
ge         # End of previous word
```

### **Line Movement**
```
gg         # First line
G          # Last line
5G         # Go to line 5
:5         # Go to line 5
H          # Top of screen
M          # Middle of screen
L          # Bottom of screen
```

### **Search Movement**
```
f{char}    # Find next char in line
F{char}    # Find previous char in line
t{char}    # Till next char
T{char}    # Till previous char
;          # Repeat last f/F/t/T
,          # Repeat last f/F/t/T (opposite)
*          # Search word under cursor forward
#          # Search word under cursor backward
```

### **Page Movement**
```
Ctrl+f     # Page forward
Ctrl+b     # Page backward
Ctrl+d     # Half page down
Ctrl+u     # Half page up
Ctrl+e     # Scroll down one line
Ctrl+y     # Scroll up one line
```

## ✏️ Editing Commands

### **Insert Mode**
```
i          # Insert before cursor
I          # Insert at beginning of line
a          # Append after cursor
A          # Append at end of line
o          # Open new line below
O          # Open new line above
s          # Substitute character
S          # Substitute line
```

### **Delete Commands**
```
x          # Delete character under cursor
X          # Delete character before cursor
dw         # Delete word
dd         # Delete line
D          # Delete to end of line
d$         # Delete to end of line
d0         # Delete to beginning of line
```

### **Change Commands**
```
cw         # Change word
cc         # Change line
C          # Change to end of line
c$         # Change to end of line
r{char}    # Replace character
R          # Replace mode
```

### **Copy and Paste**
```
yy         # Copy line
Y          # Copy line
yw         # Copy word
y$         # Copy to end of line
p          # Paste after cursor/line
P          # Paste before cursor/line
```

## 🎯 Text Objects

### **Inner Objects**
```
iw         # Inner word
is         # Inner sentence
ip         # Inner paragraph
i(         # Inner parentheses
i[         # Inner brackets
i{         # Inner braces
i"         # Inner quotes
i'         # Inner single quotes
it         # Inner tag (HTML/XML)
```

### **Around Objects**
```
aw         # Around word
as         # Around sentence
ap         # Around paragraph
a(         # Around parentheses
a[         # Around brackets
a{         # Around braces
a"         # Around quotes
a'         # Around single quotes
at         # Around tag
```

## 🔍 Search and Replace

### **Search**
```
/{pattern}     # Search forward
?{pattern}     # Search backward
n              # Next match
N              # Previous match
*              # Search word under cursor
#              # Search word backward
```

### **Replace**
```
:s/old/new/           # Replace first in line
:s/old/new/g          # Replace all in line
:%s/old/new/g         # Replace all in file
:%s/old/new/gc        # Replace with confirmation
:1,10s/old/new/g      # Replace in lines 1-10
```

### **Global Commands**
```
:g/pattern/d          # Delete lines matching pattern
:g/pattern/p          # Print lines matching pattern
:v/pattern/d          # Delete lines NOT matching
```

## 🔄 Undo and Repeat

```
u              # Undo
Ctrl+r         # Redo
U              # Undo all changes in line
.              # Repeat last change
```

## 📋 Registers and Macros

### **Registers**
```
"ay            # Yank to register 'a'
"ap            # Paste from register 'a'
"+y            # Yank to system clipboard
"+p            # Paste from system clipboard
:reg           # View all registers
```

### **Macros**
```
qq             # Start recording macro 'q'
q              # Stop recording
@q             # Execute macro 'q'
5@q            # Execute macro 5 times
@@             # Repeat last macro
```

## 📁 File Operations

### **File Commands**
```
:w             # Save
:w filename    # Save as
:q             # Quit
:q!            # Quit without saving
:wq            # Save and quit
:x             # Save and quit
ZZ             # Save and quit
:e filename    # Edit file
:r filename    # Read file into buffer
```

### **Buffer Management**
```
:ls            # List buffers
:b#            # Switch to previous buffer
:bn            # Next buffer
:bp            # Previous buffer
:bd            # Delete buffer
```

## 🪟 Windows and Tabs

### **Window Commands**
```
:sp            # Horizontal split
:vsp           # Vertical split
Ctrl+w h/j/k/l # Navigate windows
Ctrl+w w       # Next window
Ctrl+w c       # Close window
Ctrl+w o       # Only this window
Ctrl+w =       # Equal window sizes
```

### **Tab Commands**
```
:tabnew        # New tab
:tabc          # Close tab
gt             # Next tab
gT             # Previous tab
:tabs          # List tabs
```

## 📊 Visual Mode Commands

### **Visual Selection**
```
v              # Character-wise visual
V              # Line-wise visual
Ctrl+v         # Block visual
gv             # Reselect last visual
```

### **Visual Operations**
```
d              # Delete selection
c              # Change selection
y              # Copy selection
>              # Indent selection
<              # Unindent selection
=              # Auto-indent selection
```

## 🔧 Command Line Commands

### **Settings**
```
:set number            # Show line numbers
:set relativenumber    # Relative line numbers
:set hlsearch          # Highlight search
:set ignorecase        # Case insensitive search
:set autoindent        # Auto indent
:set expandtab         # Use spaces for tabs
```

### **Information**
```
:help              # Help system
:version           # Vim version info
:set               # Show all settings
:set option?       # Show specific setting
:echo $MYVIMRC     # Show vimrc location
```

## 🎮 Advanced Commands

### **Folding**
```
zf             # Create fold
zo             # Open fold
zc             # Close fold
za             # Toggle fold
zR             # Open all folds
zM             # Close all folds
```

### **Marks**
```
m{a-z}         # Set mark
'{mark}        # Jump to mark line
`{mark}        # Jump to mark position
''             # Jump to previous position
```

### **Jumps**
```
Ctrl+o         # Go to previous location
Ctrl+i         # Go to next location
:jumps         # Show jump list
```

## 🔢 Numbers and Counts

Most commands accept count prefixes:
```
5j             # Move down 5 lines
3w             # Move forward 3 words
2dd            # Delete 2 lines
4yy            # Copy 4 lines
10p            # Paste 10 times
```

## 🎯 Essential Key Combinations

### **Must-Know Combos**
```
diw            # Delete inner word
ciw            # Change inner word
yiw            # Copy inner word
ci"            # Change inside quotes
da(            # Delete around parentheses
ggVG           # Select all
```

### **Productivity Combos**
```
Ctrl+a         # Increment number
Ctrl+x         # Decrement number
J              # Join lines
~              # Toggle case
>>             # Indent line
<<             # Unindent line
```

## 🚀 Getting Help

```
:help          # General help
:help {topic}  # Specific help
:helpgrep      # Search help
:help index    # All commands
vimtutor       # Interactive tutorial
```

## 💡 Pro Tips

1. **Use counts with motions**: `5w`, `3dd`, `2ci"`
2. **Combine operators with text objects**: `diw`, `ci"`, `ya(`
3. **Master the dot command**: `.` repeats last change
4. **Use relative line numbers**: `:set relativenumber`
5. **Map Caps Lock to Escape** for easier access
6. **Stay in Normal mode** - only enter Insert when adding text
7. **Think in operator+motion** combinations
8. **Use search for navigation**: `/pattern` then `n`/`N`

---

**Keep this reference handy while learning Vim. With practice, these commands become muscle memory!** 🎯
