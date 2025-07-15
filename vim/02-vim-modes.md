# 02 - Vim Modes

Understanding Vim's modal nature is crucial for becoming proficient. Each mode serves a specific purpose and has its own set of commands.

## 🎭 The Four Main Modes

### 🔧 Normal Mode (Command Mode)
**Default mode** - Used for navigation and text manipulation

**Purpose:**
- Moving the cursor
- Deleting, copying, and pasting text
- Executing commands
- Entering other modes

**Key Characteristics:**
- Vim starts in this mode
- All keys are commands, not text input
- Most efficient mode for editing
- Access to all Vim's power

**How to Enter:**
- Press `Esc` from any other mode
- Press `Ctrl+[` (alternative to Esc)
- Press `Ctrl+c` (in most cases)

### ✏️ Insert Mode
**Text editing mode** - Similar to traditional text editors

**Purpose:**
- Inserting and editing text
- Typing characters that appear on screen

**Entering Insert Mode:**
```
i    # Insert before cursor
I    # Insert at beginning of line
a    # Insert after cursor (append)
A    # Insert at end of line
o    # Open new line below and insert
O    # Open new line above and insert
s    # Substitute character (delete char and insert)
S    # Substitute line (delete line and insert)
```

**Visual Indicator:**
- Status line shows `-- INSERT --`
- Cursor usually changes shape (depends on terminal)

### 👁️ Visual Mode
**Text selection mode** - For selecting and manipulating blocks of text

**Types of Visual Mode:**

#### Character-wise Visual (`v`)
```
v       # Start character selection
```
- Selects individual characters
- Similar to clicking and dragging with mouse

#### Line-wise Visual (`V`)
```
V       # Start line selection
```
- Selects entire lines
- Useful for operations on whole lines

#### Block Visual (`Ctrl+v`)
```
Ctrl+v  # Start block selection
```
- Selects rectangular blocks
- Perfect for columnar editing

**Common Operations in Visual Mode:**
```
d       # Delete selection
y       # Copy (yank) selection
c       # Change selection (delete and enter insert)
>       # Indent selection
<       # Unindent selection
```

### 💻 Command-line Mode
**Command execution mode** - For running Ex commands

**Entering Command-line Mode:**
```
:       # Start Ex command
/       # Start forward search
?       # Start backward search
!       # Execute shell command
```

**Common Commands:**
```
:w      # Write (save) file
:q      # Quit
:wq     # Write and quit
:q!     # Quit without saving
:help   # Access help system
:set    # Change settings
```

## 🔄 Mode Transitions

```
Normal Mode (Default)
    ↓ i,a,o,I,A,O,s,S
Insert Mode
    ↓ Esc
Normal Mode
    ↓ v,V,Ctrl+v
Visual Mode
    ↓ Esc
Normal Mode
    ↓ :,/,?,!
Command-line Mode
    ↓ Enter/Esc
Normal Mode
```

## 🎯 Practical Examples

### Example 1: Basic Text Editing
```
1. Start in Normal mode
2. Press 'i' → Insert mode
3. Type "Hello World"
4. Press Esc → Normal mode
5. Press ':w' → Save file
```

### Example 2: Select and Delete Lines
```
1. Normal mode
2. Press 'V' → Line-wise Visual mode
3. Press 'j' three times → Select 4 lines
4. Press 'd' → Delete selected lines
5. Back in Normal mode
```

### Example 3: Search and Replace
```
1. Normal mode
2. Press ':' → Command-line mode
3. Type 's/old/new/g' → Replace all 'old' with 'new'
4. Press Enter → Execute command
5. Back in Normal mode
```

## 🏆 Best Practices

### **Mode Awareness**
- Always know which mode you're in
- Check status line for mode indicators
- Practice mode transitions until automatic

### **Efficient Mode Usage**
- Spend most time in Normal mode
- Enter Insert mode only to add text
- Use Visual mode for precise selections
- Command-line mode for file operations

### **Common Beginner Mistakes**
1. **Staying in Insert mode too long**
   - Use Normal mode for navigation
   - Switch to Insert only when adding text

2. **Forgetting current mode**
   - Always check status line
   - Press Esc when in doubt

3. **Not using appropriate mode**
   - Use Visual mode for selections
   - Use Command-line mode for file operations

## 🎮 Practice Exercises

### Exercise 1: Mode Navigation
1. Open a file with some text
2. Practice entering each mode
3. Navigate between modes 20 times
4. Observe status line changes

### Exercise 2: Insert Mode Variants
```
i   # Insert here
a   # Insert after
I   # Insert at line start
A   # Insert at line end
o   # New line below
O   # New line above
```
Practice each one until comfortable.

### Exercise 3: Visual Selection
1. Select words with `v`
2. Select lines with `V`
3. Select blocks with `Ctrl+v`
4. Practice different operations (d, y, c)

## 🔧 Customization Tips

### Status Line Enhancement
Add to `.vimrc` for better mode visibility:
```vim
" Always show status line
set laststatus=2

" Custom status line showing mode
set statusline=%f\ %m%r%h%w\ [%{&ff}]\ [%Y]\ [%04l,%04v]\ [%p%%]
```

### Mode-specific Cursor
```vim
" Change cursor shape in different modes
let &t_SI = "\e[6 q"    " Insert mode - thin cursor
let &t_EI = "\e[2 q"    " Normal mode - block cursor
```

## 🆘 Troubleshooting

### **Stuck in Insert Mode?**
- Press `Esc` or `Ctrl+[`
- If unresponsive, try `Ctrl+c`

### **Can't Type?**
- You're in Normal mode
- Press `i` to enter Insert mode

### **Commands Not Working?**
- Check you're in Normal mode
- Press `Esc` first

### **Lost in Command-line Mode?**
- Press `Esc` to cancel
- Or press `Enter` if command is valid

## 🎯 Next Steps

Once you're comfortable with modes:
1. Master [Basic Navigation](./03-basic-navigation.md)
2. Practice mode transitions daily
3. Learn mode-specific shortcuts
4. Explore advanced mode features

---

**Key Takeaway**: Vim's power comes from its modal nature. Master the modes, and you'll master Vim! 🎭
