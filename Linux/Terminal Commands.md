# File and Directory Management
`ls`         --> Lists the contents of a directory (files and subdirectories).
`cd`         --> Changes the current working directory.
`pwd `       --> Prints the current working directory (the full path of the directory you are in).
`mkdir`      --> Creates a new directory.
`touch `     --> Creates a new empty file or updates the timestamp of an existing file.
`cp  `       --> Copies files or directories from one location to another.
`rm  `       --> Removes (deletes) files or directories. Use with caution, especially with options like -r.
`mv`         --> Moves or renames files or directories.
`rmdir `     --> Removes empty directories.

# File Viewing and Editing
`cat`        --> Concatenates and prints file contents to the standard output. Useful for viewing small files.
`tail `      --> Displays the last part (default 10 lines) of a file. Useful for monitoring logs.
`head`       --> Displays the first part (default 10 lines) of a file.
`less`       --> A pager that allows you to view file contents screen by screen, and scroll up or down.
`more `      --> A pager that allows you to view file contents screen by screen, only scrolling down.
`nano `      --> A simple, user-friendly text editor that runs in the terminal.
`vi   `      --> A powerful but less intuitive text editor that runs in the terminal.
`vim   `     --> An improved version of vi.

# Process Management
`ps`         --> Reports on current processes. Useful for seeing what programs are running.
`top`        --> Displays a dynamic real-time view of running processes and system resource usage.
`kill  `     --> Sends a signal to a process, typically to terminate it.
`pkill`      --> Kills processes based on name and other attributes.

# Networking Commands
`ifconfig`   --> Configures or displays network interface parameters. (Older command, `ip` is more current)
`ip `        --> A utility to show or manipulate routing, network devices, interfaces, and tunnels.
`ss  `       --> Utility to investigate sockets. Provides more detailed information than netstat.
`netstat`    --> Displays network connections, routing tables, interface statistics, etc.
`ssh`        --> Secure Shell: a cryptographic network protocol for operating network services securely over an unsecured network. Used for secure remote login.
`scp`        --> Secure Copy Protocol: a means of securely transferring computer files between a local host and a remote host or between two remote hosts.
`curl`       --> A command-line tool for transferring data with URLs. Supports various protocols like HTTP, HTTPS, FTP, etc.
`wget`       --> A free utility for non-interactive download of files from the web.
`ping`       --> Sends packets to a network host and listens for a response, used to test network connectivity.
`iptables `  --> A command-line firewall utility that uses policy chains to allow or deny traffic.
`ufw`        --> Uncomplicated Firewall: a user-friendly front-end for iptables.

# System Information and Management
`uname `     --> Prints system information, such as the kernel name, network node hostname, kernel release, etc.
`neofetch`   --> A command-line system information tool that displays information about your operating system, software, and hardware in an aesthetic and visually pleasing way.
`cal`        --> Displays a calendar.
`free`       --> Displays the amount of free and used physical and swap memory in the system.
`df`         --> Reports file system disk space usage.
`du `        --> Estimates file space usage.

# Search and Compression
`find  `     --> Searches for files and directories in a directory hierarchy.
`grep `      --> Searches for patterns in text and prints lines that match.
`tar`        --> Archives and extracts files. Can also be used for compression in conjunction with gzip or bzip2.
`gzip`       --> Compresses or decompresses files.
`unzip`      --> Lists, tests, or extracts files from a .zip archive.

# User and Group Management
`whoami`     --> Prints the effective username of the current user.
`useradd`    --> Creates a new user account.
`userdel`    --> Deletes a user account.
`groupadd`   --> Creates a new group.
`groupdel`   --> Deletes a group.
`sudo`       --> Executes a command as another user, by default the superuser (root).
`adduser`    --> A more user-friendly script for creating users than useradd.
`su`         --> Substitutes user identity, typically used to switch to the root user.
`exit`       --> Exits the current shell session or closes the terminal window.
`passwd`     --> Changes a user's password.

# Other Useful Commands
`echo`       --> Displays a line of text.
`shred`      --> Securely deletes a file by overwriting it multiple times.
`ln`         --> Creates links between files.
`clear`      --> Clears the terminal screen.
`apt`        --> A command-line tool for handling packages (installing, removing, updating, etc.) on Debian-based systems like Ubuntu and Kali Linux.
`finger`     --> Displays information about a user.
`man`        --> Displays the manual page for a command. Provides detailed information about commands.
`whatis`     --> Displays a one-line description of a command.
`cmp`        --> Compares two files byte by byte.
`diff`       --> Compares two files line by line and shows the differences.
`sort`       --> Sorts lines of text in files.
`uniq`       --> Reports or omits repeated lines.
`date`       --> Prints or sets the system date and time.
`wc`         --> Prints newline, word, and byte counts for files.
`history`    --> Displays the history of commands executed in the current shell session.
`reboot`     --> Restarts the system.
`shutdown`   --> Shuts down the system.