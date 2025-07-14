# Detailed Notes: Reconnaissance and Information Gathering

## Passive Reconnaissance - Comprehensive Analysis

### Open Source Intelligence (OSINT) Methodology

#### Intelligence Cycle
1. **Planning and Direction**
   - Define intelligence requirements
   - Identify key questions to answer
   - Establish collection priorities
   - Set timeline and resources

2. **Collection**
   - Gather information from various sources
   - Use multiple collection methods
   - Document source reliability
   - Maintain operational security

3. **Processing**
   - Organize collected information
   - Translate foreign language content
   - Convert data to usable formats
   - Validate information accuracy

4. **Analysis and Production**
   - Analyze patterns and trends
   - Correlate information from multiple sources
   - Generate intelligence assessments
   - Create actionable intelligence products

5. **Dissemination**
   - Share intelligence with stakeholders
   - Protect sources and methods
   - Provide regular updates
   - Maintain need-to-know principles

#### Public Information Sources - Detailed Breakdown

**1. Domain and Network Information**

**WHOIS Databases:**
```bash
# Command examples for WHOIS queries
whois example.com
whois 192.168.1.1
nslookup example.com
dig example.com ANY
```

**Information Available:**
- Domain registration dates
- Registrar information
- Name server details
- Contact information (often redacted)
- Domain status and expiration
- Historical registration data

**WHOIS Privacy Considerations:**
- GDPR impact on WHOIS data
- Domain privacy services
- Proxy registration services
- Historical WHOIS databases

**DNS Records Analysis:**
```bash
# DNS enumeration examples
dig example.com A
dig example.com MX
dig example.com NS
dig example.com TXT
dig example.com SOA
```

**Record Types and Intelligence Value:**
- **A Records**: IPv4 addresses revealing infrastructure
- **AAAA Records**: IPv6 addresses
- **MX Records**: Mail server infrastructure
- **NS Records**: Name server information
- **TXT Records**: SPF, DKIM, verification codes
- **CNAME Records**: Subdomain aliases
- **SRV Records**: Service location information

**2. Search Engine Intelligence**

**Google Dorking Techniques:**
```
# Basic operators
site:example.com filetype:pdf
intitle:"index of" site:example.com
inurl:admin site:example.com
cache:example.com

# Advanced queries
site:example.com (inurl:login OR inurl:admin)
filetype:doc OR filetype:docx site:example.com
"confidential" OR "internal use only" site:example.com
```

**Bing and Alternative Search Engines:**
```
# Bing specific operators
ip:192.168.1.1
domain:example.com
url:example.com/admin
```

**Search Engine Categories:**
- General search engines (Google, Bing, DuckDuckGo)
- Academic search engines (Google Scholar, JSTOR)
- Code search engines (GitHub, SourceForge)
- Social media search engines
- Specialized databases

**3. Social Media Intelligence (SOCMINT)**

**Platform-Specific Collection:**

**LinkedIn:**
- Employee profiles and roles
- Company structure and hierarchy
- Technology skills and certifications
- Recent job postings and requirements
- Company connections and partnerships

**Facebook:**
- Personal information disclosure
- Check-ins and location data
- Social network mapping
- Event attendance
- Photo metadata (EXIF data)

**Twitter:**
- Real-time information sharing
- Geolocation data
- Sentiment analysis
- Hashtag analysis
- Social network analysis

**Instagram:**
- Visual intelligence gathering
- Location-based posts
- Story content analysis
- Follower network analysis

**GitHub:**
- Source code repositories
- Commit history analysis
- Developer information
- Technology stack identification
- Configuration files exposure

**Social Media Analysis Tools:**
- Maltego
- Social Links
- Pipl
- TinEye
- TweetDeck

**4. Company and Financial Intelligence**

**Corporate Databases:**
- SEC filings (10-K, 10-Q reports)
- Annual reports
- Press releases
- Patent databases
- Trademark registrations

**Financial Information:**
- Revenue and profit data
- Merger and acquisition activity
- Investment information
- Credit ratings
- Stock performance

**Business Intelligence:**
- Industry reports
- Market analysis
- Competitor information
- Supply chain relationships
- Customer information

**5. Technical Documentation**

**Job Postings Analysis:**
```
# Example technologies found in job postings
- Operating systems (Windows Server 2019, Red Hat Enterprise Linux)
- Databases (Oracle 12c, MySQL 8.0)
- Web technologies (Apache Tomcat, IIS)
- Security tools (McAfee ePO, Symantec)
- Network equipment (Cisco ASA, Juniper SRX)
```

**Patent and Research Papers:**
- Technical implementation details
- Innovation roadmaps
- Research partnerships
- Technology capabilities

### Passive Reconnaissance Tools - Detailed Analysis

#### theHarvester
**Capabilities:**
- Email address enumeration
- Subdomain discovery
- Host discovery
- Employee name gathering
- IP address identification

**Usage Examples:**
```bash
# Basic email harvesting
theHarvester -d example.com -l 500 -b google

# Multiple sources
theHarvester -d example.com -l 200 -b all

# DNS brute force
theHarvester -d example.com -c -b google
```

**Data Sources:**
- Google
- Bing
- DuckDuckGo
- Yahoo
- LinkedIn
- Twitter
- Shodan
- VirusTotal

#### Maltego
**Transformation Categories:**
- DNS transformations
- WHOIS transformations
- Social network transformations
- Geolocation transformations
- Company transformations

**Entity Types:**
- Person
- Company
- Domain
- IP Address
- Email Address
- Phone Number
- Location

**Transform Hubs:**
- Maltego Standard Transforms
- PassiveTotal
- Shodan
- VirusTotal
- Have I Been Pwned

#### Shodan
**Search Capabilities:**
- Internet-connected device discovery
- Service banner identification
- Geolocation mapping
- Historical data analysis
- Vulnerability identification

**Search Syntax:**
```
# Basic searches
port:22
country:US
org:"Example Corp"
product:Apache

# Advanced queries
port:443 ssl:"example.com"
http.title:"Admin Panel"
vuln:CVE-2017-0144
```

**Shodan Filters:**
- asn: Autonomous System Number
- city: City name
- country: Country code
- geo: Coordinates
- hostname: Hostname
- net: Network range
- org: Organization
- os: Operating system
- port: Port number
- product: Product name

#### Censys
**Search Categories:**
- IPv4 hosts
- Websites
- Certificates
- Domains

**Query Language:**
```
# Certificate search
parsed.subject.common_name: "*.example.com"

# HTTP search
protocols:"443/https"

# SSH search
services.service_name:SSH
```

### Information Analysis and Correlation

#### Data Validation Techniques
1. **Source Triangulation**
   - Compare information from multiple sources
   - Identify discrepancies and inconsistencies
   - Weight sources by reliability
   - Cross-reference timestamps

2. **Technical Validation**
   - Verify IP address ownership
   - Confirm DNS resolution
   - Test service accessibility
   - Validate certificate information

3. **Temporal Analysis**
   - Track information changes over time
   - Identify patterns and trends
   - Correlate events with timeline
   - Detect anomalies

#### Intelligence Correlation Methods

**Network Infrastructure Mapping:**
```
Example Organization Network Map:
- Primary Domain: example.com (192.168.1.100)
- Mail Server: mail.example.com (192.168.1.101)
- Web Server: www.example.com (192.168.1.102)
- FTP Server: ftp.example.com (192.168.1.103)
- VPN Gateway: vpn.example.com (192.168.1.104)
```

**Technology Stack Identification:**
```
Identified Technologies:
- Web Server: Apache 2.4.41
- Database: MySQL 8.0
- Framework: WordPress 5.8
- CDN: Cloudflare
- Analytics: Google Analytics
- SSL Certificate: Let's Encrypt
```

**Organizational Structure Mapping:**
```
Organizational Hierarchy:
- CEO: John Doe (LinkedIn, Twitter)
- CTO: Jane Smith (GitHub, LinkedIn)
- IT Manager: Bob Johnson (LinkedIn)
- Developers: 15 identified on LinkedIn
- System Administrators: 3 identified
```

## Active Reconnaissance - Comprehensive Analysis

### Network Discovery Techniques

#### Host Discovery Methods

**1. ICMP-Based Discovery**
```bash
# Ping sweep
nmap -sn 192.168.1.0/24

# ICMP timestamp requests
nmap -PP 192.168.1.1

# ICMP address mask requests
nmap -PM 192.168.1.1
```

**ICMP Types Used:**
- Type 8 (Echo Request): Standard ping
- Type 13 (Timestamp Request): System time information
- Type 17 (Address Mask Request): Subnet information

**2. TCP-Based Discovery**
```bash
# TCP SYN discovery
nmap -PS80,443,22 192.168.1.0/24

# TCP ACK discovery
nmap -PA80,443,22 192.168.1.0/24

# TCP Connect discovery
nmap -sT -p 80 192.168.1.0/24
```

**3. UDP-Based Discovery**
```bash
# UDP discovery
nmap -PU53,67,68,161 192.168.1.0/24

# Common UDP ports for discovery
# 53 (DNS), 67/68 (DHCP), 161 (SNMP)
```

**4. ARP-Based Discovery**
```bash
# ARP discovery (local network only)
nmap -PR 192.168.1.0/24

# Manual ARP scanning
arp-scan -l
arp-scan 192.168.1.0/24
```

#### Port Scanning Techniques

**1. TCP Scanning Methods**

**SYN Scan (Stealth Scan):**
```bash
# Basic SYN scan
nmap -sS 192.168.1.100

# SYN scan with version detection
nmap -sS -sV 192.168.1.100
```

**Advantages:**
- Fast and efficient
- Doesn't complete TCP handshake
- Less likely to be logged
- Default nmap scan type

**TCP Connect Scan:**
```bash
# Full TCP connection
nmap -sT 192.168.1.100
```

**Characteristics:**
- Completes full TCP handshake
- More likely to be detected and logged
- Works without raw socket privileges
- Slower than SYN scan

**FIN, NULL, and Xmas Scans:**
```bash
# FIN scan
nmap -sF 192.168.1.100

# NULL scan
nmap -sN 192.168.1.100

# Xmas scan
nmap -sX 192.168.1.100
```

**Evasion Characteristics:**
- May bypass simple firewalls
- Based on RFC 793 behavior
- Unreliable on Windows systems
- Useful for firewall testing

**2. UDP Scanning**
```bash
# UDP scan
nmap -sU 192.168.1.100

# UDP scan with version detection
nmap -sU -sV 192.168.1.100

# Fast UDP scan (top ports)
nmap -sU --top-ports 1000 192.168.1.100
```

**UDP Scanning Challenges:**
- No handshake mechanism
- ICMP responses may be rate-limited
- Takes significantly longer
- Many false positives/negatives

**3. Advanced Scanning Techniques**

**Timing Templates:**
```bash
# Paranoid (very slow, IDS evasion)
nmap -T0 192.168.1.100

# Sneaky (slow, IDS evasion)
nmap -T1 192.168.1.100

# Polite (slower, less bandwidth)
nmap -T2 192.168.1.100

# Normal (default timing)
nmap -T3 192.168.1.100

# Aggressive (faster, more aggressive)
nmap -T4 192.168.1.100

# Insane (very fast, may miss results)
nmap -T5 192.168.1.100
```

**Firewall Evasion:**
```bash
# Fragment packets
nmap -f 192.168.1.100

# Use decoy hosts
nmap -D RND:10 192.168.1.100

# Source port manipulation
nmap --source-port 53 192.168.1.100

# Idle scan (zombie host)
nmap -sI zombie_host 192.168.1.100
```

### Service Enumeration - Detailed Analysis

#### Banner Grabbing Techniques

**1. Manual Banner Grabbing**
```bash
# Telnet method
telnet 192.168.1.100 80
HEAD / HTTP/1.1
Host: example.com

# Netcat method
nc 192.168.1.100 22
nc 192.168.1.100 25

# OpenSSL for HTTPS
openssl s_client -connect 192.168.1.100:443
```

**2. Automated Banner Grabbing**
```bash
# Nmap banner grabbing
nmap -sV 192.168.1.100

# Nmap with scripts
nmap -sV --script=banner 192.168.1.100

# Aggressive service detection
nmap -sV -A 192.168.1.100
```

#### Service-Specific Enumeration

**1. HTTP/HTTPS Services**
```bash
# Web server enumeration
nmap -p 80,443 --script http-enum 192.168.1.100

# HTTP methods enumeration
nmap --script http-methods 192.168.1.100

# SSL/TLS information
nmap --script ssl-enum-ciphers 192.168.1.100
```

**Web Technology Identification:**
- Server headers (Apache, Nginx, IIS)
- Framework identification (WordPress, Drupal)
- Technology stacks (LAMP, MEAN, .NET)
- Version information

**2. SSH Services**
```bash
# SSH enumeration
nmap -p 22 --script ssh2-enum-algos 192.168.1.100

# SSH host key collection
nmap --script ssh-hostkey 192.168.1.100
```

**SSH Information Gathering:**
- SSH version and supported algorithms
- Host key fingerprints
- Authentication methods
- Banner information

**3. DNS Services**
```bash
# DNS enumeration
nmap -p 53 --script dns-zone-transfer 192.168.1.100

# DNS service detection
nmap -sU -p 53 --script dns-service-discovery 192.168.1.100
```

**DNS Enumeration Techniques:**
- Zone transfer attempts
- DNS version queries
- Recursive resolver testing
- DNS cache snooping

**4. SNMP Services**
```bash
# SNMP enumeration
nmap -sU -p 161 --script snmp-brute 192.168.1.100

# SNMP community string testing
onesixtyone -c community.txt 192.168.1.100
```

**SNMP Information:**
- System information (OID 1.3.6.1.2.1.1)
- Network interfaces
- Process information
- Software inventory

### Network Mapping and Documentation

#### Network Topology Discovery

**1. Route Tracing**
```bash
# Traceroute with TCP
nmap --traceroute 192.168.1.100

# Manual traceroute
tracert 192.168.1.100 (Windows)
traceroute 192.168.1.100 (Linux/Unix)

# MTR for continuous tracing
mtr 192.168.1.100
```

**2. Network Range Discovery**
```bash
# Network range scanning
nmap -sn 192.168.1.0/24

# Identify network segments
nmap --script targets-asn --script-args targets-asn.asn=15169
```

#### Documentation and Reporting

**1. Automated Documentation**
```bash
# Nmap output formats
nmap -oA scan_results 192.168.1.0/24

# XML output for parsing
nmap -oX results.xml 192.168.1.0/24

# Grepable output
nmap -oG results.gnmap 192.168.1.0/24
```

**2. Visual Network Mapping**
Tools for network visualization:
- Maltego
- Gephi
- yEd Graph Editor
- Lucidchart
- Draw.io

**3. Asset Inventory Creation**
```
Example Asset Inventory:
+---------------+----------------+----------+---------+------------+
| IP Address    | Hostname       | OS       | Services| Notes      |
+---------------+----------------+----------+---------+------------+
| 192.168.1.100 | web01.corp.com | Linux    | 22,80   | Web server |
| 192.168.1.101 | db01.corp.com  | Windows  | 1433    | SQL Server |
| 192.168.1.102 | dc01.corp.com  | Windows  | 53,389  | Domain Ctrl|
+---------------+----------------+----------+---------+------------+
```

### Legal and Ethical Considerations in Active Reconnaissance

#### Authorization Requirements
- Written permission for all scanning activities
- Clear scope definition (IP ranges, domains)
- Time window restrictions
- Emergency contact procedures

#### Impact Minimization
- Use appropriate timing templates
- Avoid denial of service conditions
- Monitor target system performance
- Implement scanning throttles

#### Detection Avoidance (When Authorized)
- Distributed scanning sources
- Timing randomization
- Traffic obfuscation
- Legitimate-looking traffic patterns

#### Documentation Requirements
- Complete scan logs
- Timestamp all activities
- Record all discovered information
- Maintain chain of custody
