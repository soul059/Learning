# File System Structure

## Linux File System Hierarchy

The Linux file system follows the Filesystem Hierarchy Standard (FHS), which defines the directory structure and directory contents in Unix-like operating systems.

## Root Directory (/)

The root directory is the top-level directory in the Linux file system hierarchy. All other directories branch off from here.

```
/
├── bin/       # Essential user command binaries
├── boot/      # Boot loader files
├── dev/       # Device files
├── etc/       # System configuration files
├── home/      # User home directories
├── lib/       # Essential shared libraries
├── media/     # Mount points for removable media
├── mnt/       # Temporary mount points
├── opt/       # Optional application software packages
├── proc/      # Virtual filesystem for process information
├── root/      # Root user's home directory
├── run/       # Runtime data
├── sbin/      # Essential system binaries
├── srv/       # Data for services provided by system
├── sys/       # Virtual filesystem for system information
├── tmp/       # Temporary files
├── usr/       # Secondary hierarchy for user data
└── var/       # Variable data files
```

## Important Directories Explained

### /bin (Essential User Binaries)
- Contains essential command-line utilities
- Available to all users
- Examples: ls, cp, mv, rm, cat, ps

### /boot (Boot Files)
- Contains files needed for system boot
- Kernel images, bootloader configuration
- GRUB configuration files

### /dev (Device Files)
- Contains device files (special files)
- Hardware devices represented as files
- Examples: /dev/sda (hard disk), /dev/tty (terminals)

### /etc (System Configuration)
- System-wide configuration files
- No binaries should be stored here
- Examples:
  - `/etc/passwd` - User account information
  - `/etc/fstab` - File system mount information
  - `/etc/hosts` - Hostname to IP mappings

### /home (User Home Directories)
- Contains user personal directories
- Each user has a subdirectory here
- Examples: /home/username

### /lib (Essential Libraries)
- Essential shared libraries and kernel modules
- Libraries needed by binaries in /bin and /sbin
- Examples: libc.so, libm.so

### /media (Removable Media)
- Mount points for removable media
- USB drives, CDs, DVDs automatically mounted here
- Examples: /media/usb, /media/cdrom

### /mnt (Mount Points)
- Temporary mount points for mounting file systems
- System administrators can mount file systems here
- Examples: /mnt/backup, /mnt/network

### /opt (Optional Software)
- Optional application software packages
- Third-party software installations
- Examples: /opt/google, /opt/firefox

### /proc (Process Information)
- Virtual file system
- Contains information about running processes
- Examples:
  - `/proc/cpuinfo` - CPU information
  - `/proc/meminfo` - Memory information
  - `/proc/[pid]/` - Process-specific information

### /root (Root Home)
- Root user's home directory
- Separate from /home for security reasons
- Administrative files and scripts

### /run (Runtime Data)
- Runtime data for programs
- Replaces various /var/run directories
- Examples: /run/systemd, /run/user

### /sbin (System Binaries)
- Essential system administration binaries
- Usually require root privileges
- Examples: fdisk, fsck, init, ifconfig

### /srv (Service Data)
- Data for services provided by the system
- Web server files, FTP files
- Examples: /srv/www, /srv/ftp

### /sys (System Information)
- Virtual file system
- Information about devices, drivers, kernel features
- Interface to kernel data structures

### /tmp (Temporary Files)
- Temporary files created by programs
- Usually cleared on system restart
- World-writable directory

### /usr (User Hierarchy)
Secondary hierarchy containing:
- `/usr/bin` - Non-essential user binaries
- `/usr/lib` - Libraries for /usr/bin and /usr/sbin
- `/usr/local` - Local software installations
- `/usr/sbin` - Non-essential system binaries
- `/usr/share` - Shared data (documentation, icons)
- `/usr/src` - Source code

### /var (Variable Data)
Variable data files:
- `/var/log` - Log files
- `/var/mail` - User mailboxes
- `/var/spool` - Print queues, mail queues
- `/var/tmp` - Temporary files preserved between reboots
- `/var/www` - Web server files

## File Types in Linux

### Regular Files (-)
- Normal files containing data
- Text files, binaries, images, etc.

### Directories (d)
- Containers for other files and directories
- Special type of file in Linux

### Symbolic Links (l)
- Pointers to other files or directories
- Like shortcuts in Windows

### Character Devices (c)
- Devices that transfer data character by character
- Examples: keyboard, mouse, serial ports

### Block Devices (b)
- Devices that transfer data in blocks
- Examples: hard drives, USB drives

### Named Pipes (p)
- Special files for inter-process communication
- Also called FIFOs (First In, First Out)

### Sockets (s)
- Special files for network communication
- Inter-process communication endpoints

## File Permissions

Linux uses a permission system based on three types of users:
- **Owner (u)**: The user who owns the file
- **Group (g)**: Users in the file's group
- **Others (o)**: All other users

Each has three types of permissions:
- **Read (r)**: Permission to read the file
- **Write (w)**: Permission to modify the file
- **Execute (x)**: Permission to execute the file

Example: `-rwxr-xr--`
- First character: File type (- for regular file)
- Next 3: Owner permissions (rwx)
- Next 3: Group permissions (r-x)
- Last 3: Others permissions (r--)

## Mount Points

In Linux, storage devices must be mounted to be accessible:
- **Mounting**: Making a file system accessible
- **Mount Point**: Directory where the file system is attached
- Examples:
  - USB drive mounted at /media/usb
  - Network drive mounted at /mnt/network

## Important Configuration Files

| File | Purpose |
|------|---------|
| /etc/passwd | User account information |
| /etc/shadow | Encrypted password information |
| /etc/group | Group information |
| /etc/fstab | File system mount information |
| /etc/hosts | Hostname to IP mappings |
| /etc/hostname | System hostname |
| /etc/resolv.conf | DNS resolver configuration |
| /etc/crontab | System cron jobs |

## Navigating the File System

### Absolute Paths
- Start from root directory (/)
- Example: /home/user/documents/file.txt

### Relative Paths
- Relative to current directory
- Example: documents/file.txt (if in /home/user/)

### Special Directory References
- `.` - Current directory
- `..` - Parent directory
- `~` - User's home directory
- `/` - Root directory
