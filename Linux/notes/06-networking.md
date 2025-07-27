# Networking in Linux

## Network Fundamentals

### OSI Model Layers
1. **Physical Layer**: Hardware transmission
2. **Data Link Layer**: Frame transmission (Ethernet)
3. **Network Layer**: Routing (IP)
4. **Transport Layer**: End-to-end delivery (TCP/UDP)
5. **Session Layer**: Session management
6. **Presentation Layer**: Data formatting
7. **Application Layer**: User applications

### TCP/IP Model
1. **Network Access Layer**: Physical + Data Link
2. **Internet Layer**: Network (IP)
3. **Transport Layer**: TCP/UDP
4. **Application Layer**: Application protocols

## Network Interfaces

### Interface Types
- **Ethernet**: Physical network adapters (eth0, enp3s0)
- **Wireless**: WiFi adapters (wlan0, wlp2s0)
- **Loopback**: Local interface (lo, 127.0.0.1)
- **Virtual**: Bridges, VPNs, containers

### Interface Naming
Modern systems use predictable network interface names:
- **enp3s0**: Ethernet, PCI bus 3, slot 0
- **wlp2s0**: Wireless, PCI bus 2, slot 0
- **enx**: Ethernet with MAC address

## Network Configuration Commands

### ip Command (Modern)

#### Interface Management
```bash
# Show all interfaces
ip link show

# Show specific interface
ip link show eth0

# Bring interface up
sudo ip link set eth0 up

# Bring interface down
sudo ip link set eth0 down

# Set interface MAC address
sudo ip link set eth0 address 00:11:22:33:44:55

# Set interface MTU
sudo ip link set eth0 mtu 1500
```

#### IP Address Management
```bash
# Show IP addresses
ip addr show

# Show specific interface addresses
ip addr show eth0

# Add IP address
sudo ip addr add 192.168.1.100/24 dev eth0

# Remove IP address
sudo ip addr del 192.168.1.100/24 dev eth0

# Flush all addresses from interface
sudo ip addr flush dev eth0
```

#### Routing
```bash
# Show routing table
ip route show

# Add default gateway
sudo ip route add default via 192.168.1.1

# Add specific route
sudo ip route add 10.0.0.0/8 via 192.168.1.1

# Delete route
sudo ip route del 10.0.0.0/8

# Show route to specific destination
ip route get 8.8.8.8
```

#### Neighbor Table (ARP)
```bash
# Show neighbor table
ip neigh show

# Add static ARP entry
sudo ip neigh add 192.168.1.1 lladdr 00:11:22:33:44:55 dev eth0

# Delete ARP entry
sudo ip neigh del 192.168.1.1 dev eth0

# Flush neighbor table
sudo ip neigh flush dev eth0
```

### Legacy Commands (ifconfig, route)

#### ifconfig
```bash
# Show all interfaces
ifconfig

# Show specific interface
ifconfig eth0

# Configure IP address
sudo ifconfig eth0 192.168.1.100 netmask 255.255.255.0

# Bring interface up/down
sudo ifconfig eth0 up
sudo ifconfig eth0 down

# Set MAC address
sudo ifconfig eth0 hw ether 00:11:22:33:44:55
```

#### route
```bash
# Show routing table
route -n

# Add default gateway
sudo route add default gw 192.168.1.1

# Add specific route
sudo route add -net 10.0.0.0/8 gw 192.168.1.1

# Delete route
sudo route del -net 10.0.0.0/8
```

## Network Configuration Files

### systemd-networkd

#### /etc/systemd/network/
```ini
# 01-eth0.network
[Match]
Name=eth0

[Network]
DHCP=yes
DNS=8.8.8.8
DNS=8.8.4.4

[DHCP]
UseDNS=false
```

### NetworkManager

#### /etc/NetworkManager/system-connections/
```ini
# Wired connection
[connection]
id=Wired connection 1
type=ethernet
interface-name=eth0

[ethernet]

[ipv4]
method=auto

[ipv6]
method=auto
```

### Traditional Configuration

#### /etc/network/interfaces (Debian/Ubuntu)
```bash
# Loopback interface
auto lo
iface lo inet loopback

# DHCP configuration
auto eth0
iface eth0 inet dhcp

# Static IP configuration
auto eth0
iface eth0 inet static
    address 192.168.1.100
    netmask 255.255.255.0
    gateway 192.168.1.1
    dns-nameservers 8.8.8.8 8.8.4.4
```

#### /etc/sysconfig/network-scripts/ (Red Hat/CentOS)
```bash
# ifcfg-eth0
TYPE=Ethernet
PROXY_METHOD=none
BROWSER_ONLY=no
BOOTPROTO=static
DEFROUTE=yes
IPV4_FAILURE_FATAL=no
IPV6INIT=yes
IPV6_AUTOCONF=yes
IPV6_DEFROUTE=yes
IPV6_FAILURE_FATAL=no
NAME=eth0
UUID=12345678-1234-1234-1234-123456789abc
DEVICE=eth0
ONBOOT=yes
IPADDR=192.168.1.100
PREFIX=24
GATEWAY=192.168.1.1
DNS1=8.8.8.8
DNS2=8.8.4.4
```

## DNS Configuration

### /etc/resolv.conf
```bash
# DNS servers
nameserver 8.8.8.8
nameserver 8.8.4.4
nameserver 2001:4860:4860::8888

# Search domains
search example.com local.domain

# Options
options timeout:2
options attempts:3
```

### /etc/hosts
```bash
# Local hostname resolution
127.0.0.1       localhost
127.0.1.1       hostname
192.168.1.10    server.local server
::1             localhost ip6-localhost ip6-loopback
```

### systemd-resolved
```bash
# Configuration file
/etc/systemd/resolved.conf

# Example configuration
[Resolve]
DNS=8.8.8.8 8.8.4.4
FallbackDNS=1.1.1.1 1.0.0.1
Domains=example.com
DNSSEC=yes
DNSOverTLS=yes
```

## Network Troubleshooting Tools

### ping - Test Connectivity
```bash
# Basic ping
ping google.com

# Ping with count
ping -c 4 google.com

# Ping with interval
ping -i 2 google.com

# Ping with packet size
ping -s 1000 google.com

# Ping IPv6
ping6 google.com
```

### traceroute - Trace Route
```bash
# Trace route to destination
traceroute google.com

# Use ICMP instead of UDP
traceroute -I google.com

# Set maximum hops
traceroute -m 15 google.com

# IPv6 traceroute
traceroute6 google.com
```

### netstat - Network Statistics
```bash
# Show all connections
netstat -a

# Show listening ports
netstat -l

# Show numeric addresses
netstat -n

# Show process information
netstat -p

# Show routing table
netstat -r

# Show interface statistics
netstat -i

# Common combinations
netstat -tlnp  # TCP listening ports with processes
netstat -ulnp  # UDP listening ports with processes
```

### ss - Socket Statistics (Modern)
```bash
# Show all sockets
ss -a

# Show listening sockets
ss -l

# Show TCP sockets
ss -t

# Show UDP sockets
ss -u

# Show processes
ss -p

# Show detailed information
ss -e

# Common combinations
ss -tlnp  # TCP listening ports with processes
ss -ulnp  # UDP listening ports with processes
```

### nmap - Network Mapper
```bash
# Scan single host
nmap 192.168.1.1

# Scan network range
nmap 192.168.1.0/24

# Scan specific ports
nmap -p 22,80,443 192.168.1.1

# Service detection
nmap -sV 192.168.1.1

# OS detection
nmap -O 192.168.1.1

# Aggressive scan
nmap -A 192.168.1.1
```

### tcpdump - Packet Capture
```bash
# Capture all traffic
sudo tcpdump

# Capture on specific interface
sudo tcpdump -i eth0

# Capture specific protocol
sudo tcpdump tcp

# Capture specific port
sudo tcpdump port 80

# Capture to file
sudo tcpdump -w capture.pcap

# Read from file
tcpdump -r capture.pcap

# Verbose output
sudo tcpdump -v

# Show packet contents
sudo tcpdump -X
```

### wget/curl - HTTP Tools
```bash
# Download file with wget
wget http://example.com/file.txt

# Download with curl
curl -O http://example.com/file.txt

# Test HTTP response
curl -I http://example.com

# Follow redirects
curl -L http://example.com

# POST data
curl -X POST -d "data=value" http://example.com/api

# With headers
curl -H "Content-Type: application/json" http://example.com
```

## Network Services

### SSH (Secure Shell)
```bash
# Connect to remote host
ssh user@hostname

# Connect with specific port
ssh -p 2222 user@hostname

# Copy files (SCP)
scp file.txt user@hostname:/path/

# Sync directories (rsync)
rsync -av /local/path/ user@hostname:/remote/path/

# SSH tunneling
ssh -L 8080:localhost:80 user@hostname
```

### FTP/SFTP
```bash
# Connect with FTP
ftp hostname

# Connect with SFTP
sftp user@hostname

# Transfer files
get remote_file
put local_file
```

### HTTP Services
```bash
# Start simple HTTP server (Python)
python3 -m http.server 8000

# Start simple HTTP server (Python 2)
python -m SimpleHTTPServer 8000

# Test web server
curl http://localhost:8000
```

## Firewall Configuration

### iptables (Traditional)
```bash
# List rules
sudo iptables -L

# Allow SSH
sudo iptables -A INPUT -p tcp --dport 22 -j ACCEPT

# Allow HTTP
sudo iptables -A INPUT -p tcp --dport 80 -j ACCEPT

# Allow HTTPS
sudo iptables -A INPUT -p tcp --dport 443 -j ACCEPT

# Block specific IP
sudo iptables -A INPUT -s 192.168.1.100 -j DROP

# Save rules (varies by distribution)
sudo iptables-save > /etc/iptables/rules.v4
```

### ufw (Uncomplicated Firewall)
```bash
# Enable firewall
sudo ufw enable

# Disable firewall
sudo ufw disable

# Allow service
sudo ufw allow ssh
sudo ufw allow http
sudo ufw allow https

# Allow specific port
sudo ufw allow 8080

# Allow from specific IP
sudo ufw allow from 192.168.1.100

# Deny service
sudo ufw deny telnet

# Show status
sudo ufw status
sudo ufw status verbose
```

### firewalld (Red Hat/CentOS)
```bash
# Check status
sudo firewall-cmd --state

# List zones
sudo firewall-cmd --get-zones

# Get default zone
sudo firewall-cmd --get-default-zone

# List services in zone
sudo firewall-cmd --list-services

# Add service
sudo firewall-cmd --add-service=http
sudo firewall-cmd --add-service=http --permanent

# Add port
sudo firewall-cmd --add-port=8080/tcp
sudo firewall-cmd --add-port=8080/tcp --permanent

# Reload configuration
sudo firewall-cmd --reload
```

## Network Monitoring

### iftop - Interface Traffic
```bash
# Monitor interface traffic
sudo iftop

# Monitor specific interface
sudo iftop -i eth0
```

### nethogs - Process Network Usage
```bash
# Show network usage by process
sudo nethogs

# Monitor specific interface
sudo nethogs eth0
```

### nload - Network Load
```bash
# Show network load
nload

# Monitor specific interface
nload eth0
```

### vnstat - Network Statistics
```bash
# Install vnstat
sudo apt install vnstat

# Show statistics
vnstat

# Show hourly statistics
vnstat -h

# Show daily statistics
vnstat -d

# Show monthly statistics
vnstat -m
```

## Wireless Networking

### iwconfig (Legacy)
```bash
# Show wireless interfaces
iwconfig

# Scan for networks
sudo iwlist wlan0 scan

# Connect to network
sudo iwconfig wlan0 essid "NetworkName"
sudo iwconfig wlan0 key s:password
```

### iw (Modern)
```bash
# Show wireless interfaces
iw dev

# Scan for networks
sudo iw wlan0 scan

# Show link information
iw wlan0 link

# Show station information
iw wlan0 station dump
```

### wpa_supplicant
```bash
# Configuration file: /etc/wpa_supplicant/wpa_supplicant.conf
network={
    ssid="NetworkName"
    psk="password"
}

# Connect using wpa_supplicant
sudo wpa_supplicant -D wext -i wlan0 -c /etc/wpa_supplicant/wpa_supplicant.conf
```

## Network Bonding/Teaming

### Bonding
```bash
# Load bonding module
sudo modprobe bonding

# Create bond interface
echo "+bond0" > /sys/class/net/bonding_masters

# Add slaves
echo "+eth0" > /sys/class/net/bond0/bonding/slaves
echo "+eth1" > /sys/class/net/bond0/bonding/slaves

# Set bonding mode
echo "802.3ad" > /sys/class/net/bond0/bonding/mode
```

## Network Security

### Port Security
1. **Close unused ports**: Disable unnecessary services
2. **Use non-standard ports**: Change default service ports
3. **Implement firewall rules**: Block unwanted traffic
4. **Monitor connections**: Regular network audits

### Encryption
1. **Use SSH**: Instead of telnet/rsh
2. **Use HTTPS**: Instead of HTTP
3. **VPN connections**: For remote access
4. **Certificate management**: Proper SSL/TLS setup

### Network Monitoring
1. **Log analysis**: Monitor access logs
2. **Intrusion detection**: Use tools like Snort
3. **Traffic analysis**: Monitor unusual patterns
4. **Regular audits**: Periodic security assessments
