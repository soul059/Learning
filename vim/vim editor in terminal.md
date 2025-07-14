---
tags:
  - CodeEditor/vim
---


For vim config file it's vimrc 
	`/.vimrc` for linux

### **Commands**

--> [[Advance vim#**Command Mode**| Command Mode]]

#### Exit
For exit form vim just go in command mode and write `quit or q`


:q! --> quit without saving
:qa! --> abandon all changes and exit 
:wq --> write the file and quit

--> [[vim imp#Use ! at start | Use ! at start]]

#### Sort

:sort u / :sort !


#### Set method
used to set configration

:set foldmethod indent

#### Deselecting Searched
:nohlsearch 

#### Searching and Replacing

`:s%/<search>/<replace>/g`

s -> for substituting
% ->  for entire file `avoid % to do in selected area`

#### Delete all occurrence

`:g/<string>/d` --> all lines match with string deleted
`:v/<string>/d` --> all lines does not match with string deleted

#### Register 
:reg --> name and content of register

clipboard item stored in **+** register 

-> pasting
		`"<Register name>p`
-> coping 
		`"<Rregister name>yy`