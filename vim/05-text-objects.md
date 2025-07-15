# 05 - Text Objects

Text objects are one of Vim's most powerful features, allowing precise and intuitive text manipulation. Master them to edit with surgical precision.

## 🎯 What Are Text Objects?

Text objects define **what** to operate on, not **where** to move. They represent logical text structures like words, sentences, paragraphs, and delimited content.

### **The Power of Text Objects**
- Work regardless of cursor position within the object
- More intuitive than precise cursor positioning
- Combine with any operator (d, c, y, etc.)

## 🔧 Text Object Syntax

```
{operator}{a/i}{object}

operator: d (delete), c (change), y (yank), etc.
a/i:      a (around) or i (inner)
object:   w (word), s (sentence), p (paragraph), etc.
```

### **Inner vs Around**
- **Inner (`i`)**: Content only, excluding delimiters
- **Around (`a`)**: Content plus delimiters/whitespace

## 📝 Word Text Objects

### **Word (`w`)**
```
iw   # Inner word - just the word
aw   # Around word - word plus surrounding whitespace
```

**Example:**
```
Text: "The quick brown fox"
Cursor on "brown":
- diw → "The quick  fox" (deletes "brown")
- daw → "The quick fox" (deletes "brown" and space)
```

### **WORD (`W`)**
```
iW   # Inner WORD - non-whitespace sequence
aW   # Around WORD - WORD plus whitespace
```

**Example:**
```
Text: "email@domain.com and phone:123-456"
Cursor on "email@domain.com":
- diW → " and phone:123-456" (deletes entire email)
- daW → "and phone:123-456" (deletes email and space)
```

## 📄 Sentence and Paragraph Objects

### **Sentence (`s`)**
```
is   # Inner sentence - sentence content
as   # Around sentence - sentence with trailing space
```

**Example:**
```
Text: "Hello world. How are you? Fine thanks."
Cursor in "How are you?":
- dis → "Hello world. ? Fine thanks." (deletes "How are you")
- das → "Hello world.? Fine thanks." (deletes sentence and space)
```

### **Paragraph (`p`)**
```
ip   # Inner paragraph - paragraph content
ap   # Around paragraph - paragraph with blank lines
```

## 🔗 Delimited Text Objects

### **Parentheses (`(` or `)`)**
```
i(   # Inside parentheses
a(   # Around parentheses (including parens)
i)   # Same as i(
a)   # Same as a(
```

### **Brackets (`[` or `]`)**
```
i[   # Inside brackets
a[   # Around brackets (including brackets)
i]   # Same as i[
a]   # Same as a[
```

### **Braces (`{` or `}`)**
```
i{   # Inside braces
a{   # Around braces (including braces)
i}   # Same as i{
a}   # Same as a{
```

### **Angle Brackets (`<` or `>`)**
```
i<   # Inside angle brackets
a<   # Around angle brackets (including brackets)
i>   # Same as i<
a>   # Same as a<
```

## 📋 Quote Text Objects

### **Double Quotes (`"`)**
```
i"   # Inside double quotes
a"   # Around double quotes (including quotes)
```

### **Single Quotes (`'`)**
```
i'   # Inside single quotes
a'   # Around single quotes (including quotes)
```

### **Backticks (`` ` ``)**
```
i`   # Inside backticks
a`   # Around backticks (including backticks)
```

**Example:**
```
Text: 'Hello "world" and `code` here'
Cursor inside "world":
- di" → 'Hello "" and `code` here'
- da" → 'Hello  and `code` here'
```

## 🏷️ Tag Text Objects (HTML/XML)

### **Tag (`t`)**
```
it   # Inside tag - content between tags
at   # Around tag - including opening and closing tags
```

**Example:**
```
HTML: <div class="container"><p>Hello world</p></div>
Cursor inside <p> tag:
- dit → <div class="container"><p></p></div>
- dat → <div class="container"></div>
```

## 🎮 Practical Examples

### Example 1: Function Parameter Editing
```
Code: function calculate(price, tax, discount) {
Cursor anywhere in parentheses:
- di( → function calculate() {
- ci( → function calculate(|) { (cursor in insert mode)
- yi( → copies "price, tax, discount"
```

### Example 2: String Manipulation
```
Code: message = "Hello, world!"
Cursor anywhere in string:
- ci" → message = "|" (change content, cursor in insert mode)
- yi" → copies "Hello, world!"
- da" → message = 
```

### Example 3: Array/List Editing
```
Code: items = [apple, banana, cherry]
Cursor anywhere in brackets:
- di[ → items = []
- ci[ → items = [|] (cursor in insert mode)
- ya[ → copies entire array including brackets
```

### Example 4: HTML Tag Editing
```
HTML: <h1>Page Title</h1>
Cursor anywhere in content:
- dit → <h1></h1>
- cit → <h1>|</h1> (cursor in insert mode)
- dat → (entire tag deleted)
```

## 🏆 Advanced Text Object Patterns

### **Chaining Operations**
```
Text: "The quick brown fox jumps"
- ci"Hi there" → "Hi there" (change inside quotes)
- yi"p → copy and paste quoted content
```

### **Multiple Objects**
```
Function: func(arg1, arg2, arg3)
- di(i → starts changing inside parentheses
- 2diw → delete next 2 words
```

### **Text Objects with Counts**
```
Paragraphs: Several paragraphs of text...
- 2dap → delete current paragraph and next one
- 3yis → yank current sentence and next 2
```

## 🎯 Practice Exercises

### Exercise 1: Word Objects
```
Text: "The quick-brown fox jumps over lazy_dog"
1. Use diw on each word type
2. Use daw on each word type
3. Compare iw vs iW results
```

### Exercise 2: Delimiter Practice
```
Code: function test(a, b) { return [a + b, a * b]; }
1. Change inside parentheses: ci(
2. Delete inside brackets: di[
3. Change inside braces: ci{
4. Copy around entire function
```

### Exercise 3: Quote Manipulation
```
String: 'He said "Hello there" to me'
1. Change inside double quotes: ci"
2. Delete around single quotes: da'
3. Copy inside backticks (if any)
```

### Exercise 4: HTML Editing
```
HTML: <div><p>Hello <strong>world</strong></p></div>
1. Change inside <strong> tag: cit (cursor on "world")
2. Delete around <p> tag: dat (cursor anywhere in <p>)
3. Copy inside entire <div>: yit (cursor in <div>)
```

## 🔧 Custom Text Objects

You can create custom text objects with plugins:

### **Popular Text Object Plugins**
- `vim-textobj-user` - Framework for custom text objects
- `vim-textobj-entire` - Entire buffer (ae/ie)
- `vim-textobj-line` - Current line (al/il)
- `vim-textobj-indent` - Indentation level (ai/ii)

### **Example Custom Objects**
```
ie   # Entire buffer content
al   # Around line (with newline)
il   # Inner line (without newline)
ii   # Current indentation level
ai   # Around indentation level
```

## 🚫 Common Mistakes

### **Not Using Text Objects**
- **Problem**: Manual selection with visual mode
- **Solution**: Use text objects for faster, more accurate selection

### **Confusing Inner vs Around**
- **Problem**: Unexpected results with delimiters
- **Solution**: Practice both `i` and `a` variants

### **Wrong Cursor Position**
- **Problem**: Thinking cursor position matters
- **Solution**: Text objects work from anywhere within the object

### **Not Combining with Operators**
- **Problem**: Only using text objects with delete
- **Solution**: Try with change (c), yank (y), formatting (=), etc.

## 🏅 Pro Tips

### **Quick Word Operations**
```
ciw  # Change word (from anywhere in word)
daw  # Delete word with whitespace
yiw  # Copy just the word
```

### **Efficient Quote Editing**
```
ci"  # Change inside quotes
di'  # Delete inside single quotes
yi`  # Copy inside backticks
```

### **Smart Parentheses**
```
di(  # Clear function arguments
ci[  # Change array contents
ya{  # Copy entire code block
```

### **HTML/XML Efficiency**
```
dit  # Change tag content
dat  # Delete entire tag
yat  # Copy entire tag
```

## 🎯 Integration with Other Features

### **With Registers**
```
"ayi(  # Yank inside parentheses to register 'a'
"bp    # Paste from register 'b'
```

### **With Macros**
```
qq     # Start recording macro 'q'
ciw    # Change inner word
new_text<Esc>  # Type replacement
q      # Stop recording
@q     # Apply to other words
```

### **With Search**
```
/function<Enter>  # Find function
ciw              # Change the word
new_name<Esc>    # Replace with new name
n.               # Find next and repeat change
```

## 🎯 Next Steps

Master text objects, then explore:
1. [Search and Replace](./06-search-replace.md) for bulk text operations
2. [Macros and Registers](./09-macros-registers.md) for automation
3. Practice text objects daily until they become automatic
4. Explore custom text object plugins

---

**Key Insight**: Text objects make Vim editing intuitive. Instead of thinking about cursor positions, think about the logical structure of your text! 🎯
