# 11 - Vim Configuration and Customization

Transform your Vim into a personalized, powerful editing environment. Learn to configure Vim, install plugins, and create custom workflows.

## 📁 Vim Configuration Files

### **Location of vimrc**
```bash
# Linux/macOS
~/.vimrc              # Main configuration file
~/.vim/               # Vim directory for plugins/settings

# Windows
$HOME/_vimrc          # Main configuration file
$HOME/vimfiles/       # Vim directory

# Check current locations in Vim
:echo $MYVIMRC        # Location of your vimrc
:version              # Shows all config file locations
```

### **Basic vimrc Structure**
```vim
" Comments start with a quote
" This is a comment

" Basic settings
set number
set relativenumber

" Key mappings
nnoremap <leader>q :q<CR>

" Autocommands
autocmd BufWritePre * :%s/\s\+$//e

" Functions
function! MyFunction()
  echo "Hello from Vim!"
endfunction
```

## ⚙️ Essential Settings

### **Display and Interface**
```vim
set number              " Show line numbers
set relativenumber      " Show relative line numbers
set cursorline          " Highlight current line
set showcmd             " Show command in status line
set showmode            " Show current mode
set ruler               " Show cursor position
set laststatus=2        " Always show status line
set wildmenu            " Enhanced command completion
set wildmode=list:longest  " Complete to longest common string
```

### **Search and Navigation**
```vim
set ignorecase          " Case insensitive search
set smartcase           " Case sensitive if uppercase present
set incsearch           " Incremental search
set hlsearch            " Highlight search results
set wrapscan            " Wrap around when searching
```

### **Indentation and Formatting**
```vim
set autoindent          " Copy indent from current line
set smartindent         " Smart autoindenting
set expandtab           " Use spaces instead of tabs
set tabstop=4           " Number of spaces per tab
set shiftwidth=4        " Number of spaces for indentation
set softtabstop=4       " Number of spaces per tab in insert mode
set smarttab            " Smart tab behavior
```

### **Editing Behavior**
```vim
set backspace=indent,eol,start  " Better backspace behavior
set mouse=a             " Enable mouse support
set clipboard=unnamedplus  " Use system clipboard
set undofile            " Persistent undo
set undodir=~/.vim/undo " Undo file directory
set backup              " Create backup files
set backupdir=~/.vim/backup  " Backup directory
```

### **Performance and Compatibility**
```vim
set lazyredraw          " Don't redraw during macros
set ttyfast             " Fast terminal connection
set updatetime=300      " Faster completion
set timeoutlen=500      " Shorter timeout for mappings
```

## 🗝️ Key Mappings

### **Leader Key**
```vim
let mapleader = " "     " Set space as leader key
let maplocalleader = "," " Set comma as local leader
```

### **Essential Mappings**
```vim
" Quick save and quit
nnoremap <leader>w :w<CR>
nnoremap <leader>q :q<CR>
nnoremap <leader>x :x<CR>

" Clear search highlighting
nnoremap <Esc><Esc> :nohlsearch<CR>

" Quick escape from insert mode
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
vnoremap <leader>j :m '>+1<CR>gv=gv
vnoremap <leader>k :m '<-2<CR>gv=gv
```

### **Buffer and Tab Management**
```vim
" Buffer navigation
nnoremap <leader>b :buffers<CR>
nnoremap <leader>n :bnext<CR>
nnoremap <leader>p :bprevious<CR>
nnoremap <leader>d :bdelete<CR>

" Tab navigation
nnoremap <leader>t :tabnew<CR>
nnoremap <leader>1 1gt
nnoremap <leader>2 2gt
nnoremap <leader>3 3gt
```

### **Useful Text Manipulations**
```vim
" Duplicate line
nnoremap <leader>y yyp

" Select all
nnoremap <leader>a ggVG

" Toggle case
nnoremap <leader>u ~

" Center search results
nnoremap n nzz
nnoremap N Nzz
```

## 🔌 Plugin Management

### **Popular Plugin Managers**

#### **vim-plug** (Recommended)
```vim
" Add to vimrc
call plug#begin('~/.vim/plugged')

" Plugins go here
Plug 'preservim/nerdtree'
Plug 'junegunn/fzf.vim'
Plug 'tpope/vim-surround'

call plug#end()
```

#### **Vundle**
```vim
set rtp+=~/.vim/bundle/Vundle.vim
call vundle#begin()

Plugin 'VundleVim/Vundle.vim'
Plugin 'preservim/nerdtree'

call vundle#end()
```

#### **Built-in Package Manager** (Vim 8+)
```bash
# Create directories
mkdir -p ~/.vim/pack/my-plugins/start

# Clone plugins
cd ~/.vim/pack/my-plugins/start
git clone https://github.com/preservim/nerdtree.git
```

### **Essential Plugins**

#### **File Management**
```vim
" NERDTree - File explorer
Plug 'preservim/nerdtree'
nnoremap <leader>e :NERDTreeToggle<CR>

" fzf - Fuzzy file finder
Plug 'junegunn/fzf', { 'do': { -> fzf#install() } }
Plug 'junegunn/fzf.vim'
nnoremap <leader>f :Files<CR>
nnoremap <leader>g :Rg<CR>
```

#### **Editing Enhancement**
```vim
" vim-surround - Manipulate surroundings
Plug 'tpope/vim-surround'

" vim-commentary - Easy commenting
Plug 'tpope/vim-commentary'

" auto-pairs - Auto-close brackets
Plug 'jiangmiao/auto-pairs'
```

#### **Visual Enhancement**
```vim
" Color schemes
Plug 'morhetz/gruvbox'
Plug 'dracula/vim', { 'as': 'dracula' }

" Status line
Plug 'vim-airline/vim-airline'
Plug 'vim-airline/vim-airline-themes'

" Git integration
Plug 'tpope/vim-fugitive'
Plug 'airblade/vim-gitgutter'
```

#### **Programming Support**
```vim
" Language support
Plug 'sheerun/vim-polyglot'

" Syntax checking
Plug 'dense-analysis/ale'

" Code completion
Plug 'neoclide/coc.nvim', {'branch': 'release'}
```

## 🎨 Color Schemes and Appearance

### **Setting Color Scheme**
```vim
colorscheme gruvbox     " Set color scheme
set background=dark     " Dark background

" Enable true colors (if terminal supports it)
if has('termguicolors')
  set termguicolors
endif

" Syntax highlighting
syntax enable
filetype plugin indent on
```

### **Custom Highlighting**
```vim
" Highlight trailing whitespace
highlight TrailingWhitespace ctermbg=red guibg=red
match TrailingWhitespace /\s\+$/

" Custom status line colors
hi StatusLine ctermbg=blue ctermfg=white
```

## 📝 Autocommands

### **File Type Specific Settings**
```vim
" Python files
autocmd FileType python setlocal expandtab shiftwidth=4 softtabstop=4

" JavaScript files
autocmd FileType javascript setlocal expandtab shiftwidth=2 softtabstop=2

" HTML files
autocmd FileType html setlocal expandtab shiftwidth=2 softtabstop=2
```

### **Useful Autocommands**
```vim
" Remove trailing whitespace on save
autocmd BufWritePre * :%s/\s\+$//e

" Jump to last cursor position
autocmd BufReadPost * if line("'\"") > 1 && line("'\"") <= line("$") | exe "normal! g'\"" | endif

" Auto-create directories when saving
autocmd BufWritePre * call mkdir(expand('<afile>:p:h'), 'p')

" Highlight on yank
autocmd TextYankPost * silent! lua vim.highlight.on_yank{higroup="IncSearch", timeout=700}
```

## 🔧 Custom Functions

### **Utility Functions**
```vim
" Toggle line numbers
function! ToggleNumbers()
  if &number
    set nonumber norelativenumber
  else
    set number relativenumber
  endif
endfunction
nnoremap <leader>tn :call ToggleNumbers()<CR>

" Clean up file
function! CleanUp()
  " Remove trailing whitespace
  %s/\s\+$//e
  " Remove empty lines at end
  %s/\n\+\%$//e
  " Convert tabs to spaces
  retab
endfunction
nnoremap <leader>c :call CleanUp()<CR>

" Create directory if it doesn't exist
function! MkdirIfNotExists(dir)
  if !isdirectory(a:dir)
    call mkdir(a:dir, 'p')
  endif
endfunction
```

### **Text Manipulation Functions**
```vim
" Reverse lines
function! ReverseLines() range
  let l:lines = getline(a:firstline, a:lastline)
  call reverse(l:lines)
  call setline(a:firstline, l:lines)
endfunction
command! -range ReverseLines <line1>,<line2>call ReverseLines()

" Sort words in line
function! SortWords()
  let line = getline('.')
  let words = split(line, ' ')
  call sort(words)
  call setline('.', join(words, ' '))
endfunction
nnoremap <leader>sw :call SortWords()<CR>
```

## 📋 Complete Example vimrc

```vim
" Minimal but powerful vimrc
set nocompatible

" Basic settings
set number relativenumber
set ignorecase smartcase
set incsearch hlsearch
set autoindent smartindent
set expandtab tabstop=4 shiftwidth=4 softtabstop=4
set backspace=indent,eol,start
set mouse=a
set clipboard=unnamedplus
set laststatus=2
set wildmenu wildmode=list:longest

" Key mappings
let mapleader = " "
nnoremap <leader>w :w<CR>
nnoremap <leader>q :q<CR>
nnoremap <Esc><Esc> :nohlsearch<CR>
inoremap jj <Esc>

" Window navigation
nnoremap <C-h> <C-w>h
nnoremap <C-j> <C-w>j
nnoremap <C-k> <C-w>k
nnoremap <C-l> <C-w>l

" Plugin management (vim-plug)
call plug#begin('~/.vim/plugged')
Plug 'preservim/nerdtree'
Plug 'junegunn/fzf.vim'
Plug 'tpope/vim-surround'
Plug 'tpope/vim-commentary'
Plug 'morhetz/gruvbox'
call plug#end()

" Color scheme
colorscheme gruvbox
set background=dark

" Plugin settings
nnoremap <leader>e :NERDTreeToggle<CR>
nnoremap <leader>f :Files<CR>

" File type settings
filetype plugin indent on
syntax enable

" Autocommands
autocmd BufWritePre * :%s/\s\+$//e
```

## 🎯 Configuration Best Practices

### **Organization**
```vim
" Organize vimrc sections
" 1. Basic settings
" 2. Key mappings
" 3. Plugin management
" 4. Plugin settings
" 5. Autocommands
" 6. Functions
```

### **Performance**
- Use lazy loading for plugins when possible
- Avoid heavy autocommands
- Profile startup time: `vim --startuptime startup.log`

### **Portability**
```vim
" Check for features before using
if has('clipboard')
  set clipboard=unnamedplus
endif

if has('persistent_undo')
  set undofile
  set undodir=~/.vim/undo
endif
```

### **Version Control**
```bash
# Keep your vimrc in version control
cd ~
git init
git add .vimrc
git commit -m "Initial vimrc"

# Backup important configs
cp .vimrc .vimrc.backup
```

## 🎯 Next Steps

1. Start with a minimal vimrc and add features gradually
2. Install a plugin manager and try essential plugins
3. Customize key mappings for your workflow
4. Learn to create simple functions and autocommands
5. Explore advanced features like snippets and language servers

---

**Remember**: Configuration is personal. Start simple and customize based on your actual needs, not what looks cool! 🎨
