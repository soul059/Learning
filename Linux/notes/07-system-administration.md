# System Administration

## System Information

### Hardware Information
```bash
# CPU information
cat /proc/cpuinfo
lscpu

# Memory information
cat /proc/meminfo
free -h

# Disk information
lsblk
fdisk -l
df -h

# PCI devices
lspci

# USB devices
lsusb

# Hardware summary
hwinfo --short
```

### System Information
```bash
# System information
uname -a
hostnamectl

# OS release information
cat /etc/os-release
lsb_release -a

# Kernel version
uname -r

# System uptime
uptime

# Current date and time
date

# System load
cat /proc/loadavg
```

## Boot Process

### BIOS/UEFI Boot Sequence
1. **BIOS/UEFI**: Hardware initialization
2. **Boot Loader**: GRUB loads kernel
3. **Kernel**: Linux kernel initialization
4. **Init System**: systemd/SysV starts services
5. **Runlevel/Target**: System reaches operational state

### GRUB Configuration
```bash
# GRUB configuration file
/etc/default/grub

# Update GRUB after changes
sudo update-grub  # Debian/Ubuntu
sudo grub2-mkconfig -o /boot/grub2/grub.cfg  # Red Hat/CentOS

# GRUB menu entries
/boot/grub/grub.cfg
```

### Kernel Parameters
```bash
# View current kernel parameters
cat /proc/cmdline

# Temporary kernel parameter (at boot)
# Edit GRUB entry and add to linux line

# Permanent kernel parameter
# Edit /etc/default/grub
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash parameter=value"
```

## Init Systems

### systemd (Modern)

#### Service Management
```bash
# Start service
sudo systemctl start service_name

# Stop service
sudo systemctl stop service_name

# Restart service
sudo systemctl restart service_name

# Reload service configuration
sudo systemctl reload service_name

# Enable service at boot
sudo systemctl enable service_name

# Disable service at boot
sudo systemctl disable service_name

# Check service status
systemctl status service_name

# Check if service is active
systemctl is-active service_name

# Check if service is enabled
systemctl is-enabled service_name
```

#### Service Information
```bash
# List all services
systemctl list-units --type=service

# List failed services
systemctl --failed

# List enabled services
systemctl list-unit-files --state=enabled

# Show service dependencies
systemctl list-dependencies service_name
```

#### System Control
```bash
# Reboot system
sudo systemctl reboot

# Shutdown system
sudo systemctl poweroff

# Suspend system
sudo systemctl suspend

# Hibernate system
sudo systemctl hibernate

# Switch to emergency mode
sudo systemctl emergency

# Switch to rescue mode
sudo systemctl rescue
```

#### Targets (Runlevels)
```bash
# Show current target
systemctl get-default

# Set default target
sudo systemctl set-default multi-user.target

# Switch to target
sudo systemctl isolate graphical.target

# List available targets
systemctl list-units --type=target
```

### SysV Init (Legacy)

#### Service Management
```bash
# Start service
sudo service service_name start
sudo /etc/init.d/service_name start

# Stop service
sudo service service_name stop

# Restart service
sudo service service_name restart

# Check service status
sudo service service_name status

# List all services
service --status-all
```

#### Runlevels
```bash
# Check current runlevel
runlevel
who -r

# Change runlevel
sudo init 3

# Runlevel meanings:
# 0 - Halt
# 1 - Single user mode
# 2 - Multi-user without NFS
# 3 - Multi-user with networking
# 4 - Unused
# 5 - Graphical mode
# 6 - Reboot
```

## Log Management

### System Logs Location
```bash
# Main log directory
/var/log/

# Important log files
/var/log/messages      # General system messages
/var/log/syslog        # System log (Debian/Ubuntu)
/var/log/auth.log      # Authentication log
/var/log/secure        # Security log (Red Hat/CentOS)
/var/log/kern.log      # Kernel messages
/var/log/dmesg         # Boot messages
/var/log/mail.log      # Mail server log
/var/log/apache2/      # Apache web server logs
/var/log/nginx/        # Nginx web server logs
```

### journalctl (systemd logs)
```bash
# View all logs
journalctl

# View logs from current boot
journalctl -b

# View logs from previous boot
journalctl -b -1

# Follow logs in real-time
journalctl -f

# View logs for specific service
journalctl -u service_name

# View logs for specific time period
journalctl --since "2024-01-01 00:00:00"
journalctl --until "2024-01-02 00:00:00"

# View logs with priority
journalctl -p err    # Error and above
journalctl -p warning # Warning and above

# View kernel messages
journalctl -k

# Show disk usage
journalctl --disk-usage

# Clean old logs
sudo journalctl --vacuum-time=30d
sudo journalctl --vacuum-size=100M
```

### Log Rotation
```bash
# logrotate configuration
/etc/logrotate.conf
/etc/logrotate.d/

# Example logrotate configuration
/var/log/myapp/*.log {
    daily
    missingok
    rotate 52
    compress
    delaycompress
    notifempty
    create 644 user group
    postrotate
        systemctl reload myapp
    endscript
}

# Manual log rotation
sudo logrotate /etc/logrotate.conf

# Test logrotate configuration
sudo logrotate -d /etc/logrotate.conf
```

### rsyslog Configuration
```bash
# rsyslog configuration file
/etc/rsyslog.conf
/etc/rsyslog.d/

# Example rsyslog rule
mail.*                          /var/log/mail.log
auth,authpriv.*                 /var/log/auth.log
kern.*                          /var/log/kern.log
*.emerg                         :omusrmsg:*

# Restart rsyslog
sudo systemctl restart rsyslog
```

## Cron and Scheduled Tasks

### cron Overview
cron is a time-based job scheduler that allows you to run commands or scripts at specified intervals.

### Crontab Format
```
# ┌───────────── minute (0 - 59)
# │ ┌───────────── hour (0 - 23)
# │ │ ┌───────────── day of the month (1 - 31)
# │ │ │ ┌───────────── month (1 - 12)
# │ │ │ │ ┌───────────── day of the week (0 - 6) (Sunday to Saturday)
# │ │ │ │ │
# │ │ │ │ │
# * * * * * command
```

### Crontab Management
```bash
# Edit user crontab
crontab -e

# List user crontab
crontab -l

# Remove user crontab
crontab -r

# Edit another user's crontab (root only)
sudo crontab -u username -e

# List another user's crontab
sudo crontab -u username -l
```

### Crontab Examples
```bash
# Run every minute
* * * * * /path/to/script.sh

# Run every hour at minute 0
0 * * * * /path/to/script.sh

# Run every day at 2:30 AM
30 2 * * * /path/to/script.sh

# Run every Sunday at 3:00 AM
0 3 * * 0 /path/to/script.sh

# Run every weekday at 9:00 AM
0 9 * * 1-5 /path/to/script.sh

# Run every 15 minutes
*/15 * * * * /path/to/script.sh

# Run on the 1st of every month at midnight
0 0 1 * * /path/to/script.sh
```

### System Crontabs
```bash
# System-wide crontab
/etc/crontab

# System crontab directories
/etc/cron.d/
/etc/cron.daily/
/etc/cron.hourly/
/etc/cron.monthly/
/etc/cron.weekly/

# anacron (runs missed jobs)
/etc/anacrontab
```

### systemd Timers (Modern Alternative)
```bash
# Create timer unit file
/etc/systemd/system/mytask.timer

# Example timer
[Unit]
Description=Run mytask daily

[Timer]
OnCalendar=daily
Persistent=true

[Install]
WantedBy=timers.target

# Enable and start timer
sudo systemctl enable mytask.timer
sudo systemctl start mytask.timer

# List active timers
systemctl list-timers
```

## System Monitoring

### Resource Monitoring
```bash
# Real-time system monitor
top
htop  # Enhanced version

# I/O monitoring
iotop

# Network monitoring
iftop
nethogs

# Disk usage
du -sh /path/
ncdu /path/  # Interactive disk usage

# System activity reporter
sar -u 1 5   # CPU usage every 1 second, 5 times
sar -r 1 5   # Memory usage
sar -d 1 5   # Disk I/O
```

### Performance Analysis
```bash
# vmstat - Virtual memory statistics
vmstat 1 5

# iostat - I/O statistics
iostat -x 1 5

# mpstat - Multi-processor statistics
mpstat 1 5

# pidstat - Process statistics
pidstat 1 5

# nmon - System monitor
nmon
```

### System Information Files
```bash
# CPU information
/proc/cpuinfo

# Memory information
/proc/meminfo

# System uptime
/proc/uptime

# Load average
/proc/loadavg

# Process information
/proc/[pid]/

# File system information
/proc/filesystems

# Mount information
/proc/mounts

# Network statistics
/proc/net/dev
```

## System Backup and Recovery

### Backup Strategies
1. **Full Backup**: Complete system backup
2. **Incremental Backup**: Changes since last backup
3. **Differential Backup**: Changes since last full backup
4. **Snapshot**: Point-in-time copy

### Backup Tools

#### tar - Archive Tool
```bash
# Create archive
tar -czf backup.tar.gz /path/to/backup/

# Extract archive
tar -xzf backup.tar.gz

# List archive contents
tar -tzf backup.tar.gz

# Create incremental backup
tar -czf backup-incremental.tar.gz --newer-mtime='2024-01-01' /path/to/backup/
```

#### rsync - File Synchronization
```bash
# Local synchronization
rsync -av /source/ /destination/

# Remote synchronization
rsync -av /source/ user@remote:/destination/

# Backup with hard links (space efficient)
rsync -av --link-dest=/backup/previous /source/ /backup/current/

# Exclude files
rsync -av --exclude='*.tmp' /source/ /destination/
```

#### dd - Low-level Copy
```bash
# Create disk image
sudo dd if=/dev/sda of=/backup/disk.img bs=4M

# Restore disk image
sudo dd if=/backup/disk.img of=/dev/sda bs=4M

# Create compressed image
sudo dd if=/dev/sda bs=4M | gzip > /backup/disk.img.gz

# Monitor progress (with pv)
sudo dd if=/dev/sda bs=4M | pv | gzip > /backup/disk.img.gz
```

### Automated Backup Scripts
```bash
#!/bin/bash
# backup.sh - Simple backup script

BACKUP_DIR="/backup"
SOURCE_DIR="/home"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/backup_$DATE.tar.gz"

# Create backup
tar -czf "$BACKUP_FILE" "$SOURCE_DIR"

# Remove backups older than 30 days
find "$BACKUP_DIR" -name "backup_*.tar.gz" -mtime +30 -delete

echo "Backup completed: $BACKUP_FILE"
```

## Security Administration

### File Permissions and ACLs
```bash
# Standard permissions
chmod 755 file
chmod u+x file
chmod g-w file

# Access Control Lists (ACLs)
setfacl -m u:username:rwx file
setfacl -m g:groupname:rx file
getfacl file

# Default ACLs for directories
setfacl -d -m u:username:rwx directory/
```

### sudo Configuration
```bash
# Edit sudoers file
sudo visudo

# Example sudoers entries
username ALL=(ALL:ALL) ALL
%groupname ALL=(ALL) NOPASSWD: /usr/bin/systemctl

# Check sudo access
sudo -l

# sudo log file
/var/log/auth.log  # Debian/Ubuntu
/var/log/secure    # Red Hat/CentOS
```

### SSH Hardening
```bash
# SSH configuration file
/etc/ssh/sshd_config

# Security recommendations
PermitRootLogin no
PasswordAuthentication no
PubkeyAuthentication yes
Port 2222
AllowUsers username
Protocol 2

# Restart SSH service
sudo systemctl restart sshd
```

### System Updates
```bash
# Update package lists
sudo apt update         # Debian/Ubuntu
sudo dnf check-update   # Fedora

# Upgrade packages
sudo apt upgrade        # Debian/Ubuntu
sudo dnf upgrade        # Fedora

# Security updates only
sudo apt upgrade -s | grep -i security  # Show security updates
sudo unattended-upgrade                  # Install security updates
```

## Troubleshooting

### Common Issues

#### System Won't Boot
1. Check GRUB configuration
2. Boot from rescue media
3. Check file system integrity
4. Review kernel parameters

#### Performance Issues
1. Check CPU and memory usage
2. Analyze disk I/O
3. Review system logs
4. Check for resource limits

#### Network Problems
1. Check interface configuration
2. Test connectivity with ping
3. Verify routing table
4. Check firewall rules

#### Service Issues
1. Check service status
2. Review service logs
3. Verify configuration files
4. Check dependencies

### Rescue and Recovery
```bash
# Boot into rescue mode
# Add to kernel parameters: single or systemd.unit=rescue.target

# Mount file system read-only
mount -o remount,ro /

# Check file system
fsck /dev/sda1

# Reset root password
passwd root

# Chroot into system
chroot /mnt/sysimage
```

## Automation and Configuration Management

### Shell Scripting for Administration
```bash
#!/bin/bash
# system-info.sh - System information script

echo "=== System Information ==="
echo "Hostname: $(hostname)"
echo "OS: $(cat /etc/os-release | grep PRETTY_NAME | cut -d= -f2 | tr -d '"')"
echo "Kernel: $(uname -r)"
echo "Uptime: $(uptime -p)"
echo

echo "=== Resource Usage ==="
echo "CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d% -f1)%"
echo "Memory: $(free | grep Mem | awk '{printf "%.1f%%", $3/$2 * 100.0}')"
echo "Disk: $(df -h / | awk 'NR==2{printf "%s", $5}')"
```

### Configuration Management Tools
- **Ansible**: Agentless automation
- **Puppet**: Configuration management
- **Chef**: Infrastructure as code
- **SaltStack**: Remote execution and configuration

## Best Practices

### System Administration
1. **Document changes**: Keep change logs
2. **Test in staging**: Never test in production
3. **Regular backups**: Automate backup processes
4. **Monitor systems**: Implement monitoring solutions
5. **Security updates**: Keep systems patched
6. **Access control**: Implement least privilege
7. **Log analysis**: Regular log review
8. **Disaster recovery**: Have recovery plans
