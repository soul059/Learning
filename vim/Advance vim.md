---
tags:
  - CodeEditor/vim
---

1. [For modes of vim](https://www.freecodecamp.org/news/vim-editor-modes-explained/)
2. [For more advance stuff vim](https://learnvim.irian.to/basics/moving_in_file)
3. [YT video for motion part 1](https://www.youtube.com/watch?v=lWTzqPfy1gE)
4. [YT video for motion part 2](https://www.youtube.com/watch?v=nBjEzQlJLHE)

### **Anatomy of vim**
	command count motion

For example
1. d12j --> It will delete 12 lines below the cursor
2. y5k --> it will copy 5 lines up the cursor

--> [[Learning vim#Basic Movement| Motion]]
--> [[Learning vim#Delete Operator|| command]]

--> `i (inner)`  in place of count

1. let's say i have a word `Hello world` and my cursor is at `o in world` and I want to **delete** word `world` so if i use `dw ( it will delete an word from where my cursor is)` instead use `diw (it will delete word which my cursor is standing)`
2. For changing word from middle of word where my cursor is use `ciw`

### **Normal Mode**

By default, Vim starts in “normal” mode. Normal mode can be accessed from other modes by pressing `Esc` or `<C-[>`

To perform an undo, press `u` in normal mode. This undoes changes up to the last time you were in normal mode. If you want to redo (i.e., undo your undo) press `Ctrl+r` in normal mode.


### **Insert Mode**

This is the second most used mode, and will be the most familiar behavior to most people. Once in insert mode, typing inserts characters just like a regular text editor. You can enter it by using an insert command from normal mode.

--> [[Learning vim#Insert Mode| Insert Mode]]



### **Visual Mode**

Visual mode is used to make selections of text, similar to how clicking and dragging with a mouse behaves. Selecting text allows commands to apply only to the selection, such as copying, deleting, replacing, and so on.

To make a text selection:

- Press `v` to enter visual mode, this will also mark a starting selection point
- Move the cursor to the desired end selection point; vim will provide a visual highlight of the text selection

Visual mode also has the following variants:

- `V` to enter visual line mode, this will make text selections by line
- `<C-V>` to enter visual block mode, this will make text selections by blocks; moving the cursor will make rectangle selections of the text
		For visual block mode in windows ` ctrl + v ` is used for pest so use ` ctrl + q `

by pressing `I (shift + i)` you can enter in insert mode and by esc it will effect all selected

### **Command Mode**

Command mode has a wide variety of commands and can do things that normal mode can’t do as easily. To enter command mode type ’:’ from normal mode and then type your command which should appear at the bottom of the window. For example, to do a global find and replace type `:%s/foo/bar/g` to replace all ‘foo’ with ‘bar’

- `:` Enters command mode
- `%` Means across all lines
- `s` Means substitute
- `/foo` is regex to find things to replace
- `/bar/` is regex to replace things with
- `/g` means global, otherwise it would only execute once per line

Vim has a number of other methods that you can read about in the help documentation, `:h` or `:help`.


### **Replace Mode**

Replace mode allows you replace existing text by directly typing over it. Before entering this mode, get into normal mode and put your cursor on top of the first character that you want to replace. Then press ‘R’ (capital R) to enter replace mode. Now whatever you type will replace the existing text. The cursor automatically moves to the next character just like in insert mode. The only difference is that every character you type will replace the existing one.