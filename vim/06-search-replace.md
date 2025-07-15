# 06 - Search and Replace

Master Vim's powerful search and replace capabilities. Learn to find text quickly, perform complex substitutions, and use regular expressions effectively.

## 🔍 Basic Search

### **Forward and Backward Search**
```
/{pattern}    # Search forward for pattern
?{pattern}    # Search backward for pattern
n             # Next occurrence (same direction)
N             # Next occurrence (opposite direction)
```

### **Search Navigation**
```
*             # Search forward for word under cursor
#             # Search backward for word under cursor
g*            # Search forward for partial word under cursor
g#            # Search backward for partial word under cursor
```

### **Search History**
```
/{up-arrow}   # Browse search history
q/            # Open search command window
q?            # Open backward search command window
```

## ⚙️ Search Configuration

### **Essential Search Settings**
```vim
set ignorecase    # Case insensitive search
set smartcase     # Case sensitive if contains uppercase
set incsearch     # Incremental search (search as you type)
set hlsearch      # Highlight search results
```

### **Clear Search Highlighting**
```
:nohlsearch   # Clear highlighting (temporary)
:noh          # Short form

# Add to .vimrc for quick access:
nnoremap <Esc><Esc> :nohlsearch<CR>
```

## 🎯 Advanced Search Patterns

### **Exact Word Matching**
```
/\<word\>     # Search for exact word "word"
/\<vim\>      # Finds "vim" but not "vimrc"
```

### **Case Control**
```
/word\c       # Case insensitive search for this pattern
/word\C       # Case sensitive search for this pattern
```

### **Line Position**
```
/^pattern     # Pattern at beginning of line
/pattern$     # Pattern at end of line
/^$           # Empty lines
```

### **Character Classes**
```
/[0-9]        # Any digit
/[a-zA-Z]     # Any letter
/[aeiou]      # Any vowel
/[^0-9]       # Any non-digit
```

## 🔄 Basic Find and Replace

### **Substitute Command Syntax**
```
:[range]s/{pattern}/{replacement}/{flags}
```

### **Common Substitution Examples**
```
:s/old/new/           # Replace first occurrence in current line
:s/old/new/g          # Replace all occurrences in current line
:%s/old/new/g         # Replace all occurrences in entire file
:%s/old/new/gc        # Replace with confirmation
:1,10s/old/new/g      # Replace in lines 1-10
```

### **Useful Flags**
```
g     # Global (all occurrences in line)
c     # Confirm each replacement
i     # Case insensitive
I     # Case sensitive
n     # Count matches without replacing
```

## 🎮 Interactive Replace Workflow

### **The Confirmation Process**
When using `/gc` flag, Vim prompts for each replacement:
```
y     # Yes, replace this match
n     # No, skip this match
a     # All, replace this and all remaining matches
q     # Quit, don't replace any more
l     # Last, replace this match and quit
^E    # Scroll up
^Y    # Scroll down
```

### **Smart Replace Workflow**
```bash
# 1. Search first to see matches
/pattern

# 2. Review all occurrences with n/N

# 3. Replace with confirmation
:%s//replacement/gc

# 4. Or replace all if confident
:%s//replacement/g
```

## 🎯 Advanced Replace Techniques

### **Using Capture Groups**
```
:%s/\(.*\):\(.*\)/\2:\1/g    # Swap text around colon
:%s/\(\w\+\)/[\1]/g          # Wrap words in brackets
```

### **Special Replacement Characters**
```
&     # The whole matched pattern
\1    # First capture group
\2    # Second capture group
\r    # Newline
\t    # Tab
\n    # Null character (deletes)
```

### **Examples**
```
# Add quotes around words
:%s/\w\+/"&"/g

# Swap first and last name
:%s/\(\w\+\) \(\w\+\)/\2, \1/g

# Convert spaces to underscores
:%s/ /_/g

# Remove trailing whitespace
:%s/\s\+$//g
```

## 📍 Range Specifications

### **Line Ranges**
```
:5,10s/old/new/g      # Lines 5 to 10
:.s/old/new/g         # Current line only
:$s/old/new/g         # Last line only
:%s/old/new/g         # Entire file
:'<,'>s/old/new/g     # Visual selection
```

### **Pattern Ranges**
```
:/start/,/end/s/old/new/g     # From "start" to "end"
:/function/+5s/old/new/g      # From "function" plus 5 lines
```

### **Relative Ranges**
```
:.,+5s/old/new/g      # Current line plus next 5
:-2,+2s/old/new/g     # 2 lines before to 2 lines after
```

## 🔧 Practical Examples

### **Code Refactoring**
```
# Rename variable
:%s/\<oldVar\>/newVar/g

# Change function calls
:%s/oldFunction(/newFunction(/g

# Update import statements
:%s/from old_module/from new_module/g

# Convert single to double quotes
:%s/'/"/g
```

### **Text Formatting**
```
# Convert to uppercase
:%s/.*/\U&/g

# Convert to lowercase
:%s/.*/\L&/g

# Capitalize first letter of each word
:%s/\<\w/\u&/g

# Remove empty lines
:g/^$/d

# Join lines with commas
:%s/\n/, /g
```

### **Data Manipulation**
```
# Extract email addresses
:%s/.*\(\w\+@\w\+\.\w\+\).*/\1/g

# Format phone numbers
:%s/\(\d\{3}\)\(\d\{3}\)\(\d\{4}\)/(\1) \2-\3/g

# Convert CSV to pipe-separated
:%s/,/|/g
```

## 🔍 Global Commands

### **Global Command Syntax**
```
:g/{pattern}/{command}    # Execute command on lines matching pattern
:v/{pattern}/{command}    # Execute command on lines NOT matching pattern
```

### **Common Global Operations**
```
:g/TODO/d             # Delete all lines containing "TODO"
:g/^$/d               # Delete all empty lines
:g/pattern/p          # Print all lines matching pattern
:g/function/nu        # Show line numbers for lines with "function"
```

### **Complex Global Examples**
```
# Copy all function definitions to end of file
:g/^function/t$

# Move all import statements to top
:g/^import/m0

# Delete all commented lines
:g/^#/d

# Add semicolon to lines missing it
:g/[^;]$/s/$/;/
```

## 📊 Search Statistics

### **Count Matches**
```
:%s/pattern//gn       # Count occurrences without replacing
:g/pattern/          # Show all matching lines
```

### **Find Unique Lines**
```
:sort                 # Sort lines first
:g/^\(.*\)\n\1$/d     # Remove duplicate lines
```

## 🎯 Regular Expression Patterns

### **Anchors**
```
^     # Start of line
$     # End of line
\<    # Start of word
\>    # End of word
```

### **Quantifiers**
```
*     # Zero or more
\+    # One or more
\?    # Zero or one
\{n}  # Exactly n times
\{n,} # n or more times
\{n,m}# Between n and m times
```

### **Character Classes**
```
.     # Any character except newline
\w    # Word character [a-zA-Z0-9_]
\W    # Non-word character
\d    # Digit [0-9]
\D    # Non-digit
\s    # Whitespace
\S    # Non-whitespace
```

### **Groups and Alternatives**
```
\(pattern\)   # Capture group
\|            # Alternation (OR)
\[abc\]       # Character class
\[^abc\]      # Negated character class
```

## 🎮 Practice Exercises

### Exercise 1: Basic Search and Replace
```
Text: "The quick brown fox jumps over the lazy dog"
1. Search for "the" (case insensitive)
2. Replace all "the" with "a"
3. Replace only the first "the" in each line
```

### Exercise 2: Pattern Matching
```
Code: var firstName = "John"; var lastName = "Doe";
1. Change all variable declarations from "var" to "let"
2. Swap the values of firstName and lastName
3. Add semicolons to lines that don't have them
```

### Exercise 3: Global Operations
```
Text with mixed content including:
- Function definitions
- Comments
- Empty lines
- Import statements

1. Delete all empty lines
2. Move all import statements to the top
3. Add "// TODO: Document" before each function
```

## 💡 Pro Tips

### **Reuse Last Search**
```
:%s//replacement/g    # Uses last search pattern
```

### **Preview Before Replace**
```
# Use substitute with 'n' flag first
:%s/pattern/replacement/gn

# Then replace
:%s//replacement/g
```

### **Undo After Replace**
```
u         # Undo last substitution
:earlier 1m  # Go back 1 minute in time
```

### **Complex Patterns**
Break complex patterns into steps:
```
# Instead of one complex regex, do multiple simple ones
:%s/pattern1/temp/g
:%s/pattern2/temp2/g
:%s/temp/final/g
```

## 🚫 Common Mistakes

### **Escaping Issues**
```
# Wrong:
:%s/file.txt/newfile.txt/g

# Right (escape the dot):
:%s/file\.txt/newfile.txt/g
```

### **Greedy Matching**
```
# Matches too much:
:%s/".*"/"replacement"/g

# Use non-greedy:
:%s/".\{-}"/"replacement"/g
```

### **Case Sensitivity**
```
# Remember case settings affect search
:set ignorecase smartcase
```

## 🎯 Next Steps

Master search and replace, then explore:
1. [Buffers and Windows](./07-buffers-windows.md) for multi-file editing
2. [Macros and Registers](./09-macros-registers.md) for automation
3. Practice with real-world text manipulation tasks
4. Learn advanced regex patterns

---

**Key Principle**: Search and replace is one of Vim's most powerful features. Master the basics, then gradually add complexity as needed! 🎯
