---
tags:
  - "#CodeEditor/vim"
---

**Basic of vim form best website**
--> [learn basic from here](https://www.vim-hero.com)

![[all vim normal mode keys.png]]

# Introduction to Vim Modes
## Normal Mode

Used for:

- Moving the cursor
- Deleting characters, words, lines, etc
- Performing operations like Copy / Paste

## Insert Mode

Used for editing and inserting text.

You can think of insert mode as the default behavior in the editors you are already used to.

# Basic Movement

--> [[Best practices#Why not use the arrow keys? | Why not to use the arrow keys]]

While in normal mode you have access to many powerful motions to help you navigate through code.

Let's start by learning the basics. Use the `hjkl` keys as indicated below to move one character in any direction.

![[hjkl vim keys.png]]

# Moving With Words

--> [[vim imp#What defines a word? | What defines a word in vim]]

It is useful to navigate by word to travel medium distances, since all statements in a program are composed of words.

w --> move to the **next word**.

e --> move to the **end** of a word.

b --> move **back** a word.

# Advance movement

**gg** --> Go to the first line of the file
**G** --> Go to the last line of the file

**( , )** --> Move back and forward by sentence.
**[ , ]** --> simple skip some lines idk
**{ , }** --> skipping paragraph

But its very useful with combined edits [[Advance vim#**Anatomy of vim** | Anatomy of vim]]
1. `" Hello world "` you want to change string under `""` so you need to stay your cursor in `""` so use `ci" (it will cnage any thig under "")`
2. same for all `) , } , ]`

**%** --> It will jump form all brackets `( to ) , [ to ] , { to }`

# Insert Mode

So far you have learned how to move in **normal mode**. In order to edit text, you will need to enter **insert mode**.

**i** --> enter insert mode **before** your cursor.

**a** --> enter insert mode **after** your cursor.

**esc** --> go back to **normal mode**.

--> [[Best practices#Mapping Caps Lock to Escape | Mapping Caps Lock to Escape]]

# Inserting at Line Ends

In this lesson, you will learn how to enter **insert mode** at the beginning and end of a line.

I (Shift + i) --> enter insert mode at the **beginning of a line**show example.

A (Shift + a) --> enter insert mode at the **end of a line**.


# Opening New Lines

Use the **open** operator to add a new line above or below your cursor and then enter insert mode on that line.

o --> open a new line **below the cursor**show example.

O (Shift + o) --> open a new line **above the cursor**show example.

-->[[vim imp#Indentation Level | Indentation Level]]


# Making Small Edits

In this lesson, you will learn how to make small edits to your text using the s, x, and r operators.

x --> delete the character **under** the cursor.

s --> delete the character under your cursor and enter insert mode.

r --> replace the character under your cursor with the next character you type.

--> [[Best practices#When to use r vs s | When to use r vs s]]


# Moving to Line Ends

Use the following keys to move to the beginning or end of a line:

0 --> move to the **beginning** of a line.

$ --> move to the **end** of a line.

_ --> move to the **first word** in a line.


# Find Motion

Use the find motions to quickly jump to a specific character within a line.

**f** --> move forward to the **next occurrence** of {char} within the line.

**t** --> move forward to just **before next occurrence** of {char} within the line

**F (Shift + f)** --> move backward to the **previous occurrence** of {char} within the line.

**T (Shift + t)** --> move backward to the **after previous occurrence** of {char} within the line.

**;** --> **repeat** the last find motion.

**Note**: The find motions can only be used to navigate to characters within the current line. If no matching character is found in the current line, the cursor will not move.

--> [[Best practices#When to use f and F motions? | When to use f and F motions]]


# Searching and Replacing

 **/** --> Start a forward search. Type a keyword, then hit Enter to find it in the document.

 **?** --> Start a backward search.

**n** --> jumps to next occurrence 
**N ( shift + n )** --> jumps to previous occurrence

# Delete Operator

--> `d delete`

dw --> delete word

dd --> delete a line **it also copy that line**

D ( Shift + d ) --> delete from the cursor **to the end** of a line


# Copy Paste Lines

--> `y yank(copy)`
--> `p put(past)`

yy --> yank (copy) the **current line.

p --> put (paste) **below** the cursor.

P ( Shift + p ) -->put (paste) **above** the cursor.

Y (Shift +y) --> alternative mapping for yy.

--> [[vim editor in terminal#Register | Paste or Copy from Register]]


# Changing lines

--> `c change`

cc --> Remove line and enter insert mode

C ( shift + c ) --> change from cursor to the end of line


# Undo and Redo

u -->  Undo the last change.
Ctrl + r --> Redo the changes that were undone.

# Repeating Commands

 **.** --> Repeat the last command executed. For instance, if you deleted a word with **dw** , pressing **.** will repeat that action.

# Indentation

**>>** --> indent to right
**<<** --> indent to left

== --> used to auto indent 

For in [[Advance vim#**Visual Mode**|visual mode]] selected It can be just done by single `> , < , =` 

It also combined with all by just using 1 command ` > , < , = ` [[Advance vim#**Anatomy of vim** | Anatomy of vim]]
		I mean ` =G `  it will do auto indent entire file

# Marks

**m** --> its used to set mark particular cursor location 
	used with **extra{char}** for multiple marks
- ma
- mb

**'** --> It's used to go to marked location line at start
**\`** --> it's used to go to specific marked location 
	used with **extra{char}** for multiple location
- 'a
- \`b

--> [[Advance vim#**Anatomy of vim** | you can also use count]]


# Macro 

q --> To record macro
	used same as marks

@ --> it's used to execute the macro
	used same as marks

--> [[Advance vim#**Anatomy of vim** | you can also use count]]


# Folding
`z fold`

zf --> by selecting some code and it will create new fold

zo -->  unfold

zm --> do all fold
zr --> do all unfold

za --> toggle fold and unnfold 

# Join 

count + J ( shift + j ) --> it will join multiple lines to gether

# upper and lower case

gU (shift + u)--> convert word into upper
gu --> convert it into lower case

--> [[Advance vim#**Anatomy of vim**| It is used with command same as anatomy]]

to do with entire line repeat `u or U`


# Extra 

gf --> if have file that name that will go to that file
