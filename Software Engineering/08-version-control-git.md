# Version Control and Git

## Introduction

Version Control Systems (VCS) are tools that help manage changes to source code over time. They track modifications, allow multiple developers to collaborate, and provide the ability to revert to previous versions when needed.

## Why Version Control?

### Problems Without Version Control

1. **Lost Changes**: Accidental overwrites or deletions
2. **Collaboration Issues**: Multiple developers modifying same files
3. **No History**: Cannot see what changed, when, or why
4. **No Backup**: Single point of failure
5. **Branching Difficulties**: Hard to work on multiple features simultaneously
6. **Release Management**: Difficult to maintain different versions

### Benefits of Version Control

1. **Change Tracking**: Complete history of all changes
2. **Collaboration**: Multiple developers can work simultaneously
3. **Backup and Recovery**: Distributed copies provide backup
4. **Branching and Merging**: Work on features in isolation
5. **Blame/Annotation**: See who changed what and when
6. **Release Management**: Tag and manage different versions
7. **Rollback**: Revert to previous stable versions

## Types of Version Control Systems

### 1. Local Version Control Systems

**Example**: RCS (Revision Control System)
- Database stores all file changes locally
- Simple but no collaboration support
- Single point of failure

### 2. Centralized Version Control Systems

**Examples**: CVS, Subversion (SVN), Perforce

**Architecture**:
```
Client 1 ←→ Central Server ←→ Client 2
             (Repository)
```

**Advantages**:
- Central control and administration
- Simple to understand and set up
- Fine-grained access control

**Disadvantages**:
- Single point of failure
- Requires network connection
- Limited offline capabilities

### 3. Distributed Version Control Systems

**Examples**: Git, Mercurial, Bazaar

**Architecture**:
```
Client 1 (Full Repo) ←→ Remote Server ←→ Client 2 (Full Repo)
                        (Central Repo)
```

**Advantages**:
- No single point of failure
- Full offline capabilities
- Flexible workflows
- Fast operations (local)
- Better branching and merging

**Disadvantages**:
- More complex to understand
- Storage overhead (full history locally)
- Potential for complexity in workflows

## Git Fundamentals

### What is Git?

Git is a distributed version control system created by Linus Torvalds in 2005. It's designed to handle everything from small to very large projects with speed and efficiency.

### Key Concepts

#### 1. Repository (Repo)
**Definition**: Directory containing your project files and Git metadata (.git folder)

**Types**:
- **Local Repository**: On your local machine
- **Remote Repository**: On a server (GitHub, GitLab, etc.)
- **Bare Repository**: Contains only Git data, no working directory

#### 2. Working Directory
**Definition**: Current state of files you're working on

#### 3. Staging Area (Index)
**Definition**: Intermediate area where changes are prepared before committing

#### 4. Git Workflow
```
Working Directory → Staging Area → Local Repository → Remote Repository
     (edit)          (git add)      (git commit)     (git push)
```

### Git States

Files in Git can be in one of three states:

1. **Modified**: Changed but not committed
2. **Staged**: Marked for inclusion in next commit
3. **Committed**: Safely stored in local database

### Git Areas

1. **Working Tree**: Files you're currently editing
2. **Staging Area**: Files prepared for next commit
3. **Git Directory**: Where Git stores metadata and object database

## Basic Git Commands

### Repository Setup

#### Initialize a Repository
```bash
# Create new repository
git init

# Clone existing repository
git clone <repository-url>
git clone https://github.com/user/repo.git
```

#### Configuration
```bash
# Set user information (required for commits)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# View configuration
git config --list

# Set default editor
git config --global core.editor "code --wait"
```

### Basic Workflow Commands

#### Check Status
```bash
# Show status of working directory and staging area
git status

# Short status format
git status -s
```

#### Add Files to Staging
```bash
# Add specific file
git add filename.txt

# Add multiple files
git add file1.txt file2.txt

# Add all modified files
git add .

# Add all files (including new and deleted)
git add -A

# Interactive staging
git add -i
```

#### Commit Changes
```bash
# Commit with inline message
git commit -m "Add user authentication feature"

# Commit with detailed message (opens editor)
git commit

# Commit all modified files (skip staging)
git commit -am "Fix bug in user validation"

# Amend last commit
git commit --amend -m "Updated commit message"
```

#### View History
```bash
# Show commit history
git log

# Compact log format
git log --oneline

# Show graph of branches
git log --graph --oneline --decorate

# Show changes in commits
git log -p

# Show specific number of commits
git log -5

# Show commits by author
git log --author="John Doe"
```

#### Show Changes
```bash
# Show unstaged changes
git diff

# Show staged changes
git diff --staged

# Show changes between commits
git diff commit1 commit2

# Show changes in specific file
git diff filename.txt
```

### Working with Remote Repositories

#### Remote Management
```bash
# Show remote repositories
git remote -v

# Add remote repository
git remote add origin https://github.com/user/repo.git

# Remove remote
git remote remove origin

# Rename remote
git remote rename origin upstream
```

#### Synchronizing with Remote
```bash
# Fetch changes from remote (doesn't merge)
git fetch origin

# Pull changes from remote (fetch + merge)
git pull origin main

# Push changes to remote
git push origin main

# Push all branches
git push --all origin

# Push tags
git push --tags origin
```

## Branching and Merging

### What are Branches?

Branches are lightweight movable pointers to specific commits. They allow you to:
- Work on features in isolation
- Experiment without affecting main code
- Collaborate on different features simultaneously

### Branch Commands

#### Creating and Switching Branches
```bash
# Create new branch
git branch feature-login

# Switch to branch
git checkout feature-login

# Create and switch to new branch
git checkout -b feature-login

# Modern syntax (Git 2.23+)
git switch feature-login
git switch -c feature-login
```

#### Managing Branches
```bash
# List all branches
git branch

# List remote branches
git branch -r

# List all branches (local and remote)
git branch -a

# Delete branch
git branch -d feature-login

# Force delete branch (unmerged changes)
git branch -D feature-login

# Rename current branch
git branch -m new-branch-name
```

### Merging Strategies

#### 1. Fast-Forward Merge
When the target branch has not diverged from the source branch.

```bash
git checkout main
git merge feature-login
```

```
Before:     main → A → B
           feature →   C → D

After:      main → A → B → C → D
```

#### 2. Three-Way Merge
When both branches have diverged.

```bash
git checkout main
git merge feature-login
```

```
Before:     main → A → B → E
           feature →   C → D

After:      main → A → B → E → M
           feature →     C → D ↗
```

#### 3. Squash Merge
Combines all feature branch commits into a single commit.

```bash
git checkout main
git merge --squash feature-login
git commit -m "Add login feature"
```

### Merge Conflicts

#### What Causes Conflicts?
- Same file modified in both branches
- One branch modifies, another deletes
- Binary file conflicts

#### Resolving Conflicts
```bash
# After a merge conflict occurs
git status

# Edit conflicted files
# Files will contain conflict markers:
<<<<<<< HEAD
Current branch content
=======
Incoming branch content
>>>>>>> feature-branch

# After resolving conflicts
git add resolved-file.txt
git commit -m "Resolve merge conflict"
```

#### Conflict Resolution Tools
```bash
# Use merge tool
git mergetool

# Abort merge
git merge --abort

# Show conflicts
git diff --name-only --diff-filter=U
```

## Advanced Git Operations

### Rebasing

#### What is Rebasing?
Rebasing moves or combines commits to a new base commit, creating a linear history.

#### Interactive Rebase
```bash
# Rebase last 3 commits
git rebase -i HEAD~3

# Rebase onto another branch
git rebase main feature-branch
```

#### Rebase Options
- **pick**: Use commit as-is
- **reword**: Change commit message
- **edit**: Stop and modify commit
- **squash**: Combine with previous commit
- **drop**: Remove commit

#### Rebase vs Merge
| Aspect | Merge | Rebase |
|--------|-------|--------|
| History | Preserves original | Creates linear history |
| Commits | Keeps all commits | May modify commits |
| Conflicts | Resolve once | May resolve multiple times |
| Safety | Safer (non-destructive) | Can lose commits if done wrong |

### Stashing

#### Save Work Temporarily
```bash
# Stash current changes
git stash

# Stash with message
git stash save "Work in progress on feature X"

# Stash including untracked files
git stash -u

# List stashes
git stash list

# Apply most recent stash
git stash apply

# Apply and remove stash
git stash pop

# Apply specific stash
git stash apply stash@{2}

# Delete stash
git stash drop stash@{1}

# Clear all stashes
git stash clear
```

### Tags

#### Creating Tags
```bash
# Lightweight tag
git tag v1.0.0

# Annotated tag (recommended)
git tag -a v1.0.0 -m "Version 1.0.0 release"

# Tag specific commit
git tag -a v1.0.0 9fceb02 -m "Version 1.0.0"

# List tags
git tag

# Show tag information
git show v1.0.0

# Push tags to remote
git push origin v1.0.0
git push origin --tags
```

#### Semantic Versioning
Format: MAJOR.MINOR.PATCH
- **MAJOR**: Incompatible API changes
- **MINOR**: Add functionality (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

Examples: v1.0.0, v1.2.3, v2.0.0-beta.1

## Git Workflows

### 1. Centralized Workflow

**Structure**: Single main branch, all developers commit to it
**Use Case**: Small teams, simple projects

```bash
git clone <repo>
# Make changes
git add .
git commit -m "Add feature"
git pull origin main
git push origin main
```

### 2. Feature Branch Workflow

**Structure**: Main branch + feature branches
**Process**:
1. Create feature branch from main
2. Work on feature
3. Create pull request
4. Merge back to main

```bash
git checkout main
git pull origin main
git checkout -b feature-user-auth
# Work on feature
git push origin feature-user-auth
# Create pull request
```

### 3. Gitflow Workflow

**Branches**:
- **main**: Production-ready code
- **develop**: Integration branch
- **feature/***: Feature development
- **release/***: Release preparation
- **hotfix/***: Emergency fixes

**Process**:
```bash
# Start feature
git checkout develop
git checkout -b feature/new-feature

# Finish feature
git checkout develop
git merge feature/new-feature
git branch -d feature/new-feature

# Start release
git checkout develop
git checkout -b release/1.2.0

# Finish release
git checkout main
git merge release/1.2.0
git tag v1.2.0
git checkout develop
git merge release/1.2.0
```

### 4. GitHub Flow

**Structure**: Simple workflow with main branch and feature branches
**Process**:
1. Create feature branch
2. Make changes and commits
3. Open pull request
4. Discuss and review code
5. Merge and deploy

### 5. GitLab Flow

**Structure**: Combines feature branches with environment branches
**Branches**:
- **main**: Development
- **staging**: Staging environment
- **production**: Production environment

## Git Best Practices

### Commit Best Practices

#### 1. Commit Message Guidelines
```
type(scope): subject

body

footer
```

**Example**:
```
feat(auth): add user registration endpoint

Implement POST /api/auth/register endpoint that accepts
email and password and creates new user account.

Resolves: #123
```

**Types**:
- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code style changes
- **refactor**: Code refactoring
- **test**: Adding or modifying tests
- **chore**: Maintenance tasks

#### 2. Commit Frequency
- Commit early and often
- Each commit should be a logical unit
- Don't commit broken code
- Use staging area effectively

### Branch Best Practices

1. **Use descriptive branch names**:
   - `feature/user-authentication`
   - `bugfix/login-validation`
   - `hotfix/security-patch`

2. **Keep branches short-lived**
3. **Delete merged branches**
4. **Protect main branch** (require pull requests)
5. **Use branch naming conventions**

### Repository Best Practices

#### 1. .gitignore File
```gitignore
# Dependencies
node_modules/
vendor/

# Build outputs
dist/
build/
*.jar
*.war

# IDE files
.vscode/
.idea/
*.swp

# OS files
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Environment files
.env
.env.local

# Database
*.db
*.sqlite
```

#### 2. README.md
Include:
- Project description
- Installation instructions
- Usage examples
- Contributing guidelines
- License information

#### 3. Repository Structure
```
project/
├── .git/
├── .gitignore
├── README.md
├── LICENSE
├── src/
├── tests/
├── docs/
└── scripts/
```

## Advanced Git Features

### Git Hooks

#### What are Hooks?
Scripts that run automatically at certain points in Git workflow.

#### Types of Hooks

**Client-side hooks**:
- `pre-commit`: Before commit is finalized
- `prepare-commit-msg`: Before commit message editor
- `commit-msg`: Validate commit messages
- `post-commit`: After commit is completed

**Server-side hooks**:
- `pre-receive`: Before any references are updated
- `update`: Before individual reference is updated
- `post-receive`: After all references are updated

#### Example: Pre-commit Hook
```bash
#!/bin/sh
# .git/hooks/pre-commit

# Run tests before commit
npm test
if [ $? -ne 0 ]; then
    echo "Tests failed. Commit aborted."
    exit 1
fi

# Check code style
npm run lint
if [ $? -ne 0 ]; then
    echo "Linting failed. Commit aborted."
    exit 1
fi
```

### Git Submodules

#### What are Submodules?
Way to include one Git repository as a subdirectory of another.

```bash
# Add submodule
git submodule add https://github.com/user/library.git lib

# Clone repository with submodules
git clone --recursive <repo-url>

# Update submodules
git submodule update --init --recursive

# Pull changes in submodules
git submodule foreach git pull origin main
```

### Git Worktrees

#### Multiple Working Directories
```bash
# Create new worktree
git worktree add ../project-feature feature-branch

# List worktrees
git worktree list

# Remove worktree
git worktree remove ../project-feature
```

## Git Tools and Integration

### GUI Clients

**Cross-platform**:
- **SourceTree**: Free, feature-rich
- **GitKraken**: Visual Git client
- **GitHub Desktop**: Simple GitHub integration

**IDE Integration**:
- **VS Code**: Built-in Git support
- **IntelliJ IDEA**: Comprehensive Git tools
- **Sublime Merge**: Fast Git client

### Command Line Tools

**Git Aliases**:
```bash
git config --global alias.st status
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.unstage 'reset HEAD --'
git config --global alias.last 'log -1 HEAD'
git config --global alias.visual '!gitk'
```

**Git Bash Enhancements**:
- **Oh My Zsh**: Shell framework with Git plugins
- **Git Prompt**: Show Git status in prompt
- **Git Completion**: Tab completion for Git commands

### Hosting Services

#### GitHub
- **Features**: Issues, pull requests, actions, pages
- **Plans**: Free for public, paid for private
- **Best for**: Open source, collaborative projects

#### GitLab
- **Features**: Built-in CI/CD, issue tracking, wiki
- **Plans**: Free tier with generous limits
- **Best for**: DevOps integration, private projects

#### Bitbucket
- **Features**: Jira integration, pipelines
- **Plans**: Free for small teams
- **Best for**: Atlassian ecosystem integration

#### Azure DevOps
- **Features**: Integrated with Microsoft ecosystem
- **Plans**: Free for small teams
- **Best for**: Microsoft technology stack

## Troubleshooting Common Issues

### Undoing Changes

#### Unstaged Changes
```bash
# Discard changes in file
git checkout -- filename.txt

# Discard all unstaged changes
git checkout -- .

# Modern syntax
git restore filename.txt
git restore .
```

#### Staged Changes
```bash
# Unstage file
git reset HEAD filename.txt

# Unstage all files
git reset HEAD

# Modern syntax
git restore --staged filename.txt
```

#### Committed Changes
```bash
# Undo last commit (keep changes)
git reset --soft HEAD~1

# Undo last commit (discard changes)
git reset --hard HEAD~1

# Revert commit (create new commit)
git revert <commit-hash>
```

### Fixing Mistakes

#### Wrong Commit Message
```bash
# Amend last commit message
git commit --amend -m "Correct message"
```

#### Forgot to Add Files
```bash
# Add files and amend commit
git add forgotten-file.txt
git commit --amend --no-edit
```

#### Accidentally Committed to Wrong Branch
```bash
# Move commits to new branch
git branch new-branch
git reset --hard HEAD~3  # Remove 3 commits from current branch
git checkout new-branch
```

### Performance Issues

#### Large Repository
```bash
# Cleanup repository
git gc --aggressive

# Remove untracked files
git clean -fd

# Reduce repository size
git filter-branch --tree-filter 'rm -rf large-files' HEAD
```

## Security Best Practices

### Protecting Sensitive Information

1. **Never commit secrets**:
   - API keys
   - Passwords
   - Private keys
   - Database credentials

2. **Use environment variables**:
   ```bash
   # Instead of hardcoding
   const apiKey = process.env.API_KEY;
   ```

3. **Use .gitignore** for sensitive files
4. **Remove secrets from history**:
   ```bash
   git filter-branch --force --index-filter \
   'git rm --cached --ignore-unmatch secret-file.txt' \
   --prune-empty --tag-name-filter cat -- --all
   ```

### Access Control

1. **Use SSH keys** instead of passwords
2. **Enable two-factor authentication**
3. **Limit repository access**
4. **Regular access reviews**
5. **Use branch protection rules**

## Summary

Version Control and Git are essential tools for modern software development. Key takeaways:

1. **Understand Git Fundamentals**: Learn the three-stage workflow and basic commands
2. **Master Branching**: Use branches effectively for feature development
3. **Choose Appropriate Workflow**: Select workflow that fits your team and project
4. **Follow Best Practices**: Write good commit messages, use meaningful branch names
5. **Collaborate Effectively**: Use pull requests, code reviews, and proper merge strategies
6. **Security First**: Never commit secrets, use proper access controls
7. **Continuous Learning**: Git is powerful - keep learning advanced features

Git mastery comes with practice. Start with basic commands and gradually adopt more advanced features as your understanding grows.
