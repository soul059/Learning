# Process Management

## What is a Process?

A process is a running instance of a program in Linux. When you execute a command or run an application, the operating system creates a process to manage its execution.

## Process Characteristics

### Process ID (PID)
- Unique identifier for each process
- Assigned by the kernel when process is created
- Range: 1 to maximum (usually 32768)

### Parent Process ID (PPID)
- PID of the process that created this process
- All processes except init have a parent
- Forms a process tree structure

### Process States
- **Running (R)**: Currently executing or ready to run
- **Sleeping (S)**: Waiting for an event to complete
- **Uninterruptible Sleep (D)**: Waiting for I/O operations
- **Zombie (Z)**: Finished execution but still in process table
- **Stopped (T)**: Process execution suspended

## Process Types

### System Processes
- Started by the system during boot
- Run in kernel space
- Examples: kernel threads, device drivers

### User Processes
- Started by users or other processes
- Run in user space
- Examples: applications, shells, utilities

### Daemon Processes
- Background processes
- Provide services to other processes
- Usually started at boot time
- Examples: sshd, httpd, cron

## Process Creation

### fork() System Call
- Creates a copy of the current process
- Child process gets a new PID
- Both parent and child continue execution

### exec() System Call
- Replaces current process image with new program
- Same PID, but different program
- Often used after fork()

### Process Creation Flow
```
Parent Process
     |
   fork()
     |
    / \
   /   \
Parent  Child
  |      |
 wait() exec()
  |      |
  |   New Program
  |      |
  |    exit()
  |      |
   \    /
    \  /
  Parent continues
```

## Process Hierarchy

### Init Process (PID 1)
- First process started by kernel
- Parent of all other processes
- Responsible for system initialization
- Modern systems use systemd as init

### Process Tree
- All processes form a tree structure
- Root is init process
- Each process (except init) has exactly one parent
- Process can have multiple children

## Process Management Commands

### ps - Display Running Processes
```bash
# Show processes for current user
ps

# Show all processes
ps aux

# Show process tree
ps auxf

# Show processes in tree format
ps -ejH

# Show specific process
ps -p PID
```

### top - Real-time Process Viewer
```bash
# Show running processes dynamically
top

# Sort by CPU usage
top -o %CPU

# Sort by memory usage
top -o %MEM

# Show processes for specific user
top -u username
```

### htop - Enhanced Process Viewer
```bash
# Interactive process viewer (if installed)
htop
```

### pgrep - Find Process by Name
```bash
# Find process ID by name
pgrep firefox

# Find processes with pattern
pgrep -f "python script.py"

# Show process names too
pgrep -l firefox
```

### jobs - Show Active Jobs
```bash
# List current jobs
jobs

# List with process IDs
jobs -l
```

## Process Control

### Starting Processes

#### Foreground Process
```bash
# Run command in foreground
command

# Process blocks terminal until completion
firefox
```

#### Background Process
```bash
# Run command in background
command &

# Terminal remains available
firefox &
```

### Stopping Processes

#### Ctrl+C (SIGINT)
- Interrupt signal
- Polite request to terminate
- Process can catch and handle

#### Ctrl+Z (SIGTSTP)
- Suspend signal
- Pauses process execution
- Process can be resumed later

### kill Command - Send Signals
```bash
# Terminate process (SIGTERM)
kill PID

# Force kill process (SIGKILL)
kill -9 PID

# Send specific signal
kill -SIGNAL PID

# Kill process by name
killall process_name

# Kill all processes matching pattern
pkill pattern
```

### Signal Types
| Signal | Number | Description | Can be caught? |
|--------|--------|-------------|----------------|
| SIGTERM | 15 | Termination request | Yes |
| SIGKILL | 9 | Force kill | No |
| SIGINT | 2 | Interrupt (Ctrl+C) | Yes |
| SIGSTOP | 19 | Stop process | No |
| SIGCONT | 18 | Continue process | No |
| SIGHUP | 1 | Hangup | Yes |

## Job Control

### Background Jobs
```bash
# Start job in background
command &

# Move current job to background
Ctrl+Z
bg

# Move job to foreground
fg

# Move specific job to foreground
fg %job_number
```

### Job States
- **Running**: Currently executing
- **Stopped**: Paused execution
- **Done**: Completed successfully
- **Terminated**: Killed by signal

## Process Monitoring

### Process Information Files
```bash
# Process information directory
/proc/PID/

# Command line arguments
/proc/PID/cmdline

# Environment variables
/proc/PID/environ

# Memory maps
/proc/PID/maps

# Open files
/proc/PID/fd/
```

### System Load
```bash
# Show system load averages
uptime

# Load average explanation:
# - 1 minute average
# - 5 minute average  
# - 15 minute average

# Values:
# < 1.0: System not busy
# = 1.0: System fully utilized
# > 1.0: System overloaded
```

### Memory Usage
```bash
# Show memory usage
free -h

# Show swap usage
cat /proc/swaps

# Show memory info
cat /proc/meminfo
```

## Process Priorities

### Nice Values
- Range: -20 (highest) to 19 (lowest)
- Default: 0
- Lower number = higher priority
- Only root can set negative values

### Commands
```bash
# Start process with specific priority
nice -n 10 command

# Change priority of running process
renice 5 PID

# Change priority by process name
renice 5 -p $(pgrep process_name)
```

## Advanced Process Management

### nohup - Run Command Immune to Hangups
```bash
# Run command that continues after logout
nohup command &

# Output redirected to nohup.out
nohup command > output.log 2>&1 &
```

### screen/tmux - Terminal Multiplexers
```bash
# Start screen session
screen

# Detach from screen (Ctrl+A, D)
# Reattach to screen
screen -r

# List screen sessions
screen -ls
```

### systemd - Modern Process Manager
```bash
# Start service
systemctl start service_name

# Stop service
systemctl stop service_name

# Enable service at boot
systemctl enable service_name

# Check service status
systemctl status service_name
```

## Process Troubleshooting

### High CPU Usage
1. Use `top` or `htop` to identify processes
2. Check if process is stuck in loop
3. Consider killing or restarting process

### Memory Issues
1. Check memory usage with `free`
2. Identify memory-hungry processes
3. Kill unnecessary processes
4. Check for memory leaks

### Zombie Processes
1. Identify with `ps aux | grep defunct`
2. Usually indicates parent process bug
3. Kill parent process to clean zombies
4. Zombies don't consume resources except PID

### Too Many Processes
1. Check process limits with `ulimit -u`
2. Identify process sources
3. Kill unnecessary processes
4. Adjust system limits if needed

## Best Practices

1. **Monitor regularly**: Use tools like top, htop
2. **Clean termination**: Use SIGTERM before SIGKILL
3. **Background long tasks**: Use & or screen/tmux
4. **Set appropriate priorities**: Use nice for CPU-intensive tasks
5. **Check dependencies**: Understand process relationships
6. **Resource limits**: Monitor CPU and memory usage
7. **Log analysis**: Check system logs for process issues
