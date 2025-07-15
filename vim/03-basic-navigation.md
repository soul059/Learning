# 03 - Basic Navigation

Efficient navigation is the foundation of Vim mastery. Learn to move around text without ever touching the arrow keys or mouse.

## 🗝️ The Golden Rule: Home Row Keys

### **Why `hjkl` Instead of Arrow Keys?**
- **Efficiency**: Hands stay on home row
- **Speed**: No hand movement required
- **Vim Philosophy**: Minimize movement, maximize productivity

### **Finger Positioning**
```
Left hand:  a s d f
Right hand: j k l ;
```

## 🧭 Basic Movement Commands

### **Character Movement**
```
h    # Move left  (←)
j    # Move down  (↓)  
k    # Move up    (↑)
l    # Move right (→)
```

**Memory Aids:**
- `h` is leftmost, `l` is rightmost
- `j` looks like a down arrow
- `k` points up

### **Line Movement**
```
0    # Beginning of line (column 0)
^    # First non-blank character
$    # End of line
g_   # Last non-blank character
```

### **Word Movement**
```
w    # Next word (start)
W    # Next WORD (whitespace separated)
e    # End of current word
E    # End of current WORD
b    # Beginning of current/previous word
B    # Beginning of current/previous WORD
ge   # End of previous word
```

**Word vs WORD:**
- **word**: Letters, digits, underscores (hello_world)
- **WORD**: Non-whitespace characters (hello@world#123)

## 🎯 Advanced Movement

### **File Movement**
```
gg   # Go to first line
G    # Go to last line
5G   # Go to line 5
:5   # Go to line 5 (command mode)
5gg  # Go to line 5
```

### **Screen Movement**
```
H    # Top of screen (High)
M    # Middle of screen (Middle)
L    # Bottom of screen (Low)
```

### **Page Movement**
```
Ctrl+f  # Page forward (down)
Ctrl+b  # Page backward (up)
Ctrl+d  # Half page down
Ctrl+u  # Half page up
```

### **Paragraph Movement**
```
{    # Beginning of paragraph
}    # End of paragraph
```

### **Sentence Movement**
```
(    # Beginning of sentence
)    # End of sentence
```

## 🔍 Search-Based Navigation

### **Character Search in Line**
```
f{char}   # Find next occurrence of {char}
F{char}   # Find previous occurrence of {char}
t{char}   # Till next occurrence (cursor before)
T{char}   # Till previous occurrence (cursor after)
;         # Repeat last f/F/t/T
,         # Repeat last f/F/t/T in opposite direction
```

**Example:**
```
Line: "The quick brown fox jumps"
Cursor at 'q', press 'fo' → jumps to 'o' in "fox"
Press ';' → jumps to 'o' in "fox" again (if multiple)
```

### **Pattern Search**
```
/{pattern}   # Search forward for pattern
?{pattern}   # Search backward for pattern
n            # Next occurrence
N            # Previous occurrence
*            # Search forward for word under cursor
#            # Search backward for word under cursor
```

## 🔢 Using Counts

Most movement commands accept a count prefix:

```
5j     # Move down 5 lines
3w     # Move forward 3 words
10l    # Move right 10 characters
2}     # Move forward 2 paragraphs
4fx    # Find 4th occurrence of 'x'
```

## 🏃‍♂️ Efficient Movement Patterns

### **Short Distance (1-2 words)**
Use: `h`, `j`, `k`, `l`, `w`, `b`

### **Medium Distance (within screen)**
Use: `f/F/t/T`, `*/#`, `H/M/L`

### **Long Distance (across file)**
Use: `gg/G`, `/{pattern}`, `:line_number`

### **Precision Movement**
Use: `0`, `^`, `$`, `f/t`

## 🎮 Practice Exercises

### Exercise 1: Basic hjkl Movement
1. Open a large text file
2. Navigate only using `hjkl`
3. Avoid arrow keys completely
4. Practice for 10 minutes daily

### Exercise 2: Word Movement
```
Practice text: "The quick brown fox jumps over the lazy dog"

1. Start at 'T'
2. Use 'w' to move through each word
3. Use 'b' to move back
4. Use 'e' to move to word endings
```

### Exercise 3: Line Navigation
```
1. Press '0' to go to line start
2. Press '$' to go to line end
3. Press '^' to go to first non-blank
4. Repeat until automatic
```

### Exercise 4: Find Navigation
```
Practice line: "function calculateTotal(price, tax, discount)"

1. From start, press 'f(' to find opening parenthesis
2. Press 'f,' to find first comma
3. Press ';' to find next comma
4. Press 'F(' to go back to opening parenthesis
```

## 🏆 Advanced Navigation Tips

### **Relative Line Numbers**
Add to `.vimrc`:
```vim
set relativenumber
```
Then use: `5j` to jump 5 lines down easily.

### **Jump List**
```
Ctrl+o   # Go to previous location in jump list
Ctrl+i   # Go to next location in jump list
:jumps   # View jump list
```

### **Mark Navigation**
```
m{a-z}   # Set mark at current position
'{mark}  # Jump to line of mark
`{mark}  # Jump to exact position of mark
''       # Jump to previous position
``       # Jump to previous position (exact)
```

## 🚫 Common Mistakes

### **Using Arrow Keys**
- **Problem**: Inefficient hand movement
- **Solution**: Force yourself to use `hjkl`
- **Tip**: Disable arrow keys in `.vimrc`:
```vim
noremap <Up> <NOP>
noremap <Down> <NOP>
noremap <Left> <NOP>
noremap <Right> <NOP>
```

### **Moving One Character at a Time**
- **Problem**: Too granular movement
- **Solution**: Use word movement (`w`, `b`, `e`)

### **Not Using Counts**
- **Problem**: Repetitive single movements
- **Solution**: Learn to estimate distances (`5j`, `3w`)

### **Ignoring Search**
- **Problem**: Scrolling to find text
- **Solution**: Use `/` search for quick navigation

## 🔧 Configuration Tips

### **Better Search Highlighting**
```vim
set hlsearch      # Highlight search results
set incsearch     # Incremental search
set ignorecase    # Case insensitive
set smartcase     # Case sensitive if contains uppercase
```

### **Scroll Behavior**
```vim
set scrolloff=5   # Keep 5 lines visible around cursor
set sidescroll=1  # Horizontal scroll one character at a time
```

## 🎯 Next Steps

Master these navigation basics, then move on to:
1. [Basic Editing](./04-basic-editing.md)
2. Combine navigation with editing commands
3. Learn text objects for precise selection
4. Practice daily until movements become automatic

---

**Remember**: Speed comes from efficiency, not rushing. Master these movements and you'll edit text faster than ever before! 🚀
