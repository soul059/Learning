# User and Group Management

## Overview

Linux is a multi-user operating system that supports multiple users working simultaneously. The system uses users and groups to control access to files, directories, and system resources.

## User Types

### Root User (Superuser)
- User ID (UID): 0
- Has unlimited access to the system
- Can perform any operation
- Home directory: /root
- Default shell: usually /bin/bash

### System Users
- UID range: 1-999 (varies by distribution)
- Used by system services and daemons
- Usually cannot log in interactively
- Examples: www-data, mysql, nobody

### Regular Users
- UID range: 1000+ (varies by distribution)
- Created for human users
- Home directory: /home/username
- Can be granted sudo privileges

## User Information Storage

### /etc/passwd
Contains basic user account information:
```
username:x:UID:GID:GECOS:home_directory:shell
```

Example:
```
john:x:1000:1000:John Doe:/home/john:/bin/bash
```

Fields:
1. **Username**: Login name
2. **Password**: 'x' indicates password in /etc/shadow
3. **UID**: User ID number
4. **GID**: Primary group ID
5. **GECOS**: User information (full name, etc.)
6. **Home Directory**: User's home directory path
7. **Shell**: Default shell for user

### /etc/shadow
Contains encrypted password information:
```
username:encrypted_password:last_changed:min_age:max_age:warning:inactive:expire:reserved
```

Example:
```
john:$6$rounds=4096$salt$hash:18500:0:99999:7:::
```

Fields:
1. **Username**: Login name
2. **Encrypted Password**: Hashed password
3. **Last Changed**: Days since password last changed
4. **Min Age**: Minimum days before password can be changed
5. **Max Age**: Maximum days password is valid
6. **Warning**: Days before expiration to warn user
7. **Inactive**: Days after expiration before account is disabled
8. **Expire**: Date when account expires
9. **Reserved**: For future use

### /etc/group
Contains group information:
```
group_name:x:GID:user_list
```

Example:
```
sudo:x:27:john,jane
```

Fields:
1. **Group Name**: Name of the group
2. **Password**: Usually 'x' (group passwords rarely used)
3. **GID**: Group ID number
4. **User List**: Comma-separated list of group members

## User Management Commands

### Creating Users

#### useradd - Add User Account
```bash
# Basic user creation
sudo useradd username

# Create user with home directory
sudo useradd -m username

# Create user with specific shell
sudo useradd -m -s /bin/bash username

# Create user with specific UID and GID
sudo useradd -u 1001 -g 1001 username

# Create user with additional groups
sudo useradd -m -G sudo,audio,video username

# Create user with comment
sudo useradd -m -c "John Doe" username

# Create user with expiration date
sudo useradd -m -e 2024-12-31 username
```

#### adduser - Interactive User Creation (Debian/Ubuntu)
```bash
# Interactive user creation with prompts
sudo adduser username
```

### Modifying Users

#### usermod - Modify User Account
```bash
# Change username
sudo usermod -l newname oldname

# Change user's home directory
sudo usermod -d /new/home/path username

# Change user's shell
sudo usermod -s /bin/zsh username

# Add user to additional groups
sudo usermod -a -G group1,group2 username

# Change user's primary group
sudo usermod -g newgroup username

# Lock user account
sudo usermod -L username

# Unlock user account
sudo usermod -U username

# Set account expiration
sudo usermod -e 2024-12-31 username
```

#### chfn - Change User Information
```bash
# Change user's full name and info
chfn username
```

### Deleting Users

#### userdel - Delete User Account
```bash
# Delete user (keep home directory)
sudo userdel username

# Delete user and home directory
sudo userdel -r username

# Force deletion even if user is logged in
sudo userdel -f username
```

### Password Management

#### passwd - Change Password
```bash
# Change your own password
passwd

# Change another user's password (root only)
sudo passwd username

# Lock user account
sudo passwd -l username

# Unlock user account
sudo passwd -u username

# Set password expiration
sudo passwd -x 90 username

# Force password change on next login
sudo passwd -e username

# Display password status
sudo passwd -S username
```

#### chage - Change Password Aging
```bash
# Set password expiration date
sudo chage -E 2024-12-31 username

# Set minimum days between password changes
sudo chage -m 7 username

# Set maximum days password is valid
sudo chage -M 90 username

# Set warning days before expiration
sudo chage -W 7 username

# Interactive password aging setup
sudo chage username

# Display password aging information
sudo chage -l username
```

## Group Management

### Creating Groups

#### groupadd - Add Group
```bash
# Create new group
sudo groupadd groupname

# Create group with specific GID
sudo groupadd -g 1001 groupname

# Create system group
sudo groupadd -r groupname
```

### Modifying Groups

#### groupmod - Modify Group
```bash
# Change group name
sudo groupmod -n newname oldname

# Change group GID
sudo groupmod -g 1002 groupname
```

#### gpasswd - Group Password Administration
```bash
# Add user to group
sudo gpasswd -a username groupname

# Remove user from group
sudo gpasswd -d username groupname

# Set group administrator
sudo gpasswd -A admin_user groupname

# Set group members
sudo gpasswd -M user1,user2,user3 groupname
```

### Deleting Groups

#### groupdel - Delete Group
```bash
# Delete group
sudo groupdel groupname
```

## User Information Commands

### id - Display User and Group IDs
```bash
# Show current user information
id

# Show specific user information
id username

# Show only UID
id -u username

# Show only GID
id -g username

# Show all groups
id -G username
```

### who - Show Logged in Users
```bash
# Show who is logged in
who

# Show with last boot time
who -b

# Show current user
whoami

# Show users and what they're doing
w
```

### last - Show Login History
```bash
# Show recent logins
last

# Show specific user's logins
last username

# Show last reboot times
last reboot
```

### users - List Current Users
```bash
# List currently logged in users
users
```

## Switching Users

### su - Switch User
```bash
# Switch to root user
su

# Switch to specific user
su username

# Switch with environment
su - username

# Execute command as another user
su -c "command" username
```

### sudo - Execute as Another User
```bash
# Execute command as root
sudo command

# Execute command as specific user
sudo -u username command

# Switch to root shell
sudo -i

# Edit sudoers file
sudo visudo

# List sudo privileges
sudo -l
```

## Sudo Configuration

### /etc/sudoers
Configuration file for sudo privileges:

```bash
# User privilege specification
root    ALL=(ALL:ALL) ALL
john    ALL=(ALL:ALL) ALL

# Group privilege specification
%sudo   ALL=(ALL:ALL) ALL
%admin  ALL=(ALL) NOPASSWD: ALL

# Allow user to run specific commands
jane    ALL=(ALL) /bin/systemctl, /bin/mount

# Allow group to run commands without password
%wheel  ALL=(ALL) NOPASSWD: ALL
```

### Adding Users to Sudo
```bash
# Add user to sudo group (Debian/Ubuntu)
sudo usermod -a -G sudo username

# Add user to wheel group (Red Hat/CentOS)
sudo usermod -a -G wheel username
```

## File Ownership and Permissions

### Ownership
Every file and directory has:
- **Owner**: User who owns the file
- **Group**: Group that owns the file

### Changing Ownership

#### chown - Change Owner
```bash
# Change file owner
sudo chown newowner file

# Change owner and group
sudo chown newowner:newgroup file

# Change owner recursively
sudo chown -R newowner directory/

# Change only group
sudo chown :newgroup file
```

#### chgrp - Change Group
```bash
# Change file group
sudo chgrp newgroup file

# Change group recursively
sudo chgrp -R newgroup directory/
```

## User Environment

### Login Process
1. User enters credentials
2. System validates against /etc/passwd and /etc/shadow
3. Sets up user environment
4. Starts user's shell
5. Executes login scripts

### User Shell Configuration
- **Global**: /etc/profile, /etc/bash.bashrc
- **User-specific**: ~/.bashrc, ~/.profile, ~/.bash_profile

### Environment Variables
```bash
# Display environment variables
env

# Display specific variable
echo $HOME
echo $USER
echo $PATH

# Set environment variable
export VARIABLE=value
```

## Security Best Practices

### Password Policies
1. **Strong passwords**: Use complex passwords
2. **Regular changes**: Implement password aging
3. **Account lockout**: Lock inactive accounts
4. **Two-factor authentication**: Where possible

### User Account Security
1. **Principle of least privilege**: Give minimum necessary access
2. **Regular audits**: Review user accounts regularly
3. **Remove unused accounts**: Delete old accounts
4. **Monitor logins**: Check login logs regularly

### Group Management
1. **Logical grouping**: Organize users by function
2. **Regular review**: Audit group memberships
3. **Temporary access**: Use groups for temporary permissions

## Common Administrative Tasks

### User Account Audit
```bash
# List all users
cut -d: -f1 /etc/passwd

# Find users with UID 0 (should only be root)
awk -F: '$3 == 0 { print $1 }' /etc/passwd

# Find users with empty passwords
sudo awk -F: '$2 == "" { print $1 }' /etc/shadow

# Check for duplicate UIDs
cut -d: -f3 /etc/passwd | sort | uniq -d
```

### Group Membership Check
```bash
# Show all groups
cut -d: -f1 /etc/group

# Show groups for user
groups username

# Show detailed group info
getent group groupname
```

### Account Cleanup
```bash
# Find home directories without users
find /home -maxdepth 1 -type d | while read dir; do
    user=$(basename "$dir")
    if ! id "$user" &>/dev/null; then
        echo "Orphaned directory: $dir"
    fi
done

# Find users without home directories
while IFS=: read -r user x uid gid gecos home shell; do
    if [[ $uid -ge 1000 && $uid -lt 65534 && ! -d "$home" ]]; then
        echo "User $user has no home directory"
    fi
done < /etc/passwd
```
