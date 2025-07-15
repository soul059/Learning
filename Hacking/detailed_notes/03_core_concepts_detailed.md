# Detailed Notes: Core Concepts and Terminology

## Fundamental Security Principles - Deep Analysis

### CIA Triad - Comprehensive Breakdown

#### Confidentiality
**Definition**: Ensuring information is accessible only to authorized individuals, entities, or processes.

**Implementation Methods:**
1. **Access Controls**
   - Role-Based Access Control (RBAC)
   - Attribute-Based Access Control (ABAC)
   - Mandatory Access Control (MAC)
   - Discretionary Access Control (DAC)

2. **Encryption**
   - Data at rest encryption (AES-256, ChaCha20)
   - Data in transit encryption (TLS 1.3, VPN)
   - End-to-end encryption (Signal Protocol, PGP)
   - Homomorphic encryption (for processing encrypted data)

3. **Information Classification**
   - **Top Secret**: Unauthorized disclosure could cause exceptionally grave damage
   - **Secret**: Unauthorized disclosure could cause serious damage
   - **Confidential**: Unauthorized disclosure could cause damage
   - **Restricted**: Internal use only
   - **Public**: No restriction on disclosure

4. **Data Loss Prevention (DLP)**
   - Content inspection and filtering
   - Endpoint protection
   - Network monitoring
   - Cloud security gateways

**Common Threats to Confidentiality:**
- Eavesdropping and wiretapping
- Social engineering attacks
- Insider threats
- Shoulder surfing
- Dumpster diving
- Man-in-the-middle attacks

**Confidentiality Metrics:**
- Number of unauthorized access attempts
- Data classification compliance rates
- Encryption coverage percentage
- DLP policy violations

#### Integrity
**Definition**: Ensuring information and systems remain accurate, complete, and unaltered by unauthorized parties.

**Types of Integrity:**
1. **Data Integrity**
   - Accuracy of data
   - Completeness of data
   - Consistency across systems
   - Validity of data format

2. **System Integrity**
   - Operating system files unchanged
   - Application code unmodified
   - Configuration settings preserved
   - Hardware components functioning properly

**Implementation Methods:**
1. **Cryptographic Hashing**
   - SHA-256, SHA-3 for file integrity
   - Digital signatures for non-repudiation
   - HMAC for message authentication
   - Merkle trees for efficient verification

2. **Version Control**
   - Git for source code management
   - Database transaction logs
   - Configuration management tools
   - Backup and recovery systems

3. **Access Controls**
   - Write protection mechanisms
   - Approval workflows
   - Segregation of duties
   - Least privilege principles

4. **Monitoring and Auditing**
   - File integrity monitoring (FIM)
   - Database activity monitoring
   - System call auditing
   - Change detection systems

**Common Threats to Integrity:**
- Malware modifications
- Unauthorized changes by insiders
- Data corruption during transmission
- System tampering
- SQL injection attacks
- Buffer overflow exploits

**Integrity Verification Methods:**
- Checksums and hash functions
- Digital signatures
- Timestamping services
- Blockchain technology
- Consensus mechanisms

#### Availability
**Definition**: Ensuring information and systems are accessible and functional when needed by authorized users.

**Availability Measurements:**
- **Uptime**: 99.9% = 8.77 hours downtime/year
- **Downtime**: Planned vs. unplanned outages
- **Recovery Time Objective (RTO)**: Maximum acceptable downtime
- **Recovery Point Objective (RPO)**: Maximum acceptable data loss

**Implementation Methods:**
1. **Redundancy**
   - Hardware redundancy (RAID, clustering)
   - Network redundancy (multiple ISPs, load balancers)
   - Geographic redundancy (multiple data centers)
   - Staff redundancy (cross-training)

2. **Fault Tolerance**
   - Graceful degradation
   - Failover mechanisms
   - Hot standby systems
   - Circuit breakers

3. **Disaster Recovery**
   - Backup strategies (full, incremental, differential)
   - Offsite storage
   - Recovery procedures
   - Business continuity planning

4. **Performance Optimization**
   - Load balancing
   - Caching mechanisms
   - Database optimization
   - Network optimization

**Common Threats to Availability:**
- Denial of Service (DoS) attacks
- Distributed Denial of Service (DDoS) attacks
- Hardware failures
- Software bugs
- Natural disasters
- Power outages
- Network connectivity issues

**Availability Metrics:**
- Mean Time Between Failures (MTBF)
- Mean Time To Repair (MTTR)
- Service Level Agreement (SLA) compliance
- Customer satisfaction scores

### Extended Security Principles

#### Authentication
**Definition**: Verifying the identity of users, devices, or systems attempting to access resources.

**Authentication Factors:**
1. **Something You Know (Knowledge)**
   - Passwords
   - PINs
   - Security questions
   - Passphrases

2. **Something You Have (Possession)**
   - Smart cards
   - Hardware tokens
   - Mobile devices
   - USB keys

3. **Something You Are (Inherence)**
   - Fingerprints
   - Iris scans
   - Voice recognition
   - DNA analysis

4. **Something You Do (Behavior)**
   - Typing patterns
   - Gait analysis
   - Signature dynamics
   - Mouse movement patterns

5. **Somewhere You Are (Location)**
   - GPS coordinates
   - IP address ranges
   - Network location
   - Physical location

**Multi-Factor Authentication (MFA):**
- Two-Factor Authentication (2FA): Two different factors
- Three-Factor Authentication (3FA): Three different factors
- Risk-based authentication: Adaptive based on context
- Step-up authentication: Additional factors for sensitive operations

**Authentication Protocols:**
- Kerberos: Ticket-based authentication
- LDAP: Directory service authentication
- SAML: Web-based single sign-on
- OAuth 2.0: Authorization framework
- OpenID Connect: Identity layer on OAuth 2.0

#### Authorization
**Definition**: Granting appropriate access rights and permissions to authenticated users.

**Authorization Models:**
1. **Role-Based Access Control (RBAC)**
   - Users assigned to roles
   - Roles have specific permissions
   - Hierarchical role structures
   - Separation of duties enforcement

2. **Attribute-Based Access Control (ABAC)**
   - Policy-based access decisions
   - Multiple attributes considered
   - Dynamic access control
   - Fine-grained permissions

3. **Mandatory Access Control (MAC)**
   - System-enforced access control
   - Security labels and clearances
   - Government and military use
   - Bell-LaPadula and Biba models

4. **Discretionary Access Control (DAC)**
   - Owner-controlled access
   - File permission systems
   - Access control lists (ACLs)
   - Flexible but potentially insecure

**Authorization Principles:**
- Principle of least privilege
- Need-to-know basis
- Separation of duties
- Defense in depth

**Authorization Enforcement:**
- Access control matrices
- Capability-based systems
- Policy engines
- Middleware solutions

#### Non-Repudiation
**Definition**: Ensuring that actions or communications cannot be denied by the parties involved.

**Technical Implementation:**
1. **Digital Signatures**
   - RSA signatures
   - ECDSA (Elliptic Curve Digital Signature Algorithm)
   - DSA (Digital Signature Algorithm)
   - Hash functions for message integrity

2. **Timestamping**
   - Trusted timestamping authorities
   - RFC 3161 timestamp protocol
   - Blockchain-based timestamping
   - NTP synchronization

3. **Audit Logging**
   - Comprehensive activity logs
   - Tamper-evident logging
   - Log aggregation and correlation
   - Long-term log retention

4. **Cryptographic Proof Systems**
   - Zero-knowledge proofs
   - Merkle trees
   - Commitment schemes
   - Distributed ledgers

**Legal Considerations:**
- Electronic signature laws (ESIGN Act, eIDAS)
- Evidence admissibility requirements
- Chain of custody maintenance
- Expert witness testimony

#### Accountability
**Definition**: Tracking and recording user actions to establish responsibility for security-relevant events.

**Implementation Components:**
1. **Identity Management**
   - Unique user identifiers
   - Identity lifecycle management
   - Privileged account management
   - Service account governance

2. **Activity Monitoring**
   - User behavior analytics (UBA)
   - Database activity monitoring
   - Network traffic analysis
   - Application performance monitoring

3. **Audit Trail Generation**
   - System event logging
   - Application audit logs
   - Security event correlation
   - Compliance reporting

4. **Incident Investigation**
   - Digital forensics capabilities
   - Log analysis tools
   - Timeline reconstruction
   - Root cause analysis

**Accountability Frameworks:**
- NIST Cybersecurity Framework
- ISO 27001 controls
- SOX compliance requirements
- GDPR accountability principles

## Vulnerability Classification Systems

### OWASP Top 10 - Detailed Analysis

#### A01: Broken Access Control
**Description**: Restrictions on authenticated users are not properly enforced.

**Common Weaknesses:**
- Violation of principle of least privilege
- Bypassing access control checks
- Privilege escalation attacks
- CORS misconfiguration
- Force browsing to authenticated pages

**Examples:**
```
# URL manipulation example
Original: https://example.com/account?id=12345
Malicious: https://example.com/account?id=54321

# Parameter pollution
POST /transfer
amount=100&to=attacker&to=victim&amount=1000000
```

**Prevention:**
- Implement proper access control models
- Use deny-by-default policies
- Log access control failures
- Rate limit API access
- Implement proper session management

#### A02: Cryptographic Failures
**Description**: Failures related to cryptography often lead to sensitive data exposure.

**Common Issues:**
- Weak encryption algorithms (DES, RC4)
- Poor key management
- Insufficient entropy
- Hardcoded cryptographic keys
- Weak random number generation

**Examples:**
```python
# Weak encryption example
import hashlib
password_hash = hashlib.md5(password.encode()).hexdigest()  # Vulnerable

# Better approach
import bcrypt
password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
```

**Prevention:**
- Use strong, up-to-date encryption algorithms
- Implement proper key management
- Encrypt data in transit and at rest
- Use cryptographically secure random number generators
- Regular cryptographic algorithm reviews

#### A03: Injection
**Description**: Untrusted data is sent to an interpreter as part of a command or query.

**Types of Injection:**
1. **SQL Injection**
   - Classic SQL injection
   - Blind SQL injection
   - Time-based SQL injection
   - Union-based SQL injection

2. **NoSQL Injection**
   - MongoDB injection
   - CouchDB injection
   - JavaScript injection in NoSQL

3. **Command Injection**
   - OS command injection
   - LDAP injection
   - XPath injection

4. **Code Injection**
   - PHP code injection
   - Python code injection
   - JavaScript injection

**Examples:**
```sql
-- SQL Injection example
SELECT * FROM users WHERE id = '$user_id'
-- Malicious input: 1' OR '1'='1' --
-- Resulting query: SELECT * FROM users WHERE id = '1' OR '1'='1' --'
```

```python
# Command injection example
import os
filename = user_input  # Dangerous if not validated
os.system(f"cat {filename}")  # Vulnerable
# Malicious input: "file.txt; rm -rf /"
```

**Prevention:**
- Use parameterized queries
- Input validation and sanitization
- Escape special characters
- Use allow lists for input validation
- Implement least privilege for database accounts

### Common Vulnerability Scoring System (CVSS)

#### CVSS v3.1 Metrics

**Base Metrics:**
1. **Attack Vector (AV)**
   - Network (N): 0.85
   - Adjacent (A): 0.62
   - Local (L): 0.55
   - Physical (P): 0.2

2. **Attack Complexity (AC)**
   - Low (L): 0.77
   - High (H): 0.44

3. **Privileges Required (PR)**
   - None (N): 0.85
   - Low (L): 0.62/0.68
   - High (H): 0.27/0.5

4. **User Interaction (UI)**
   - None (N): 0.85
   - Required (R): 0.62

5. **Scope (S)**
   - Unchanged (U)
   - Changed (C)

6. **Impact Metrics (C/I/A)**
   - High (H): 0.56
   - Low (L): 0.22
   - None (N): 0

**Temporal Metrics:**
- Exploit Code Maturity
- Remediation Level
- Report Confidence

**Environmental Metrics:**
- Confidentiality Requirement
- Integrity Requirement
- Availability Requirement

#### CVSS Score Interpretation
- 0.0: None
- 0.1-3.9: Low
- 4.0-6.9: Medium
- 7.0-8.9: High
- 9.0-10.0: Critical

### Attack Methodologies - Comprehensive Analysis

#### Cyber Kill Chain - Detailed Breakdown

**Phase 1: Reconnaissance**
**Objectives:**
- Identify potential targets
- Gather intelligence about target organization
- Map network infrastructure
- Identify key personnel

**Techniques:**
- Open Source Intelligence (OSINT)
- Social media reconnaissance
- DNS enumeration
- WHOIS queries
- Google dorking
- Job posting analysis

**Tools:**
- theHarvester
- Maltego
- Shodan
- Censys
- Recon-ng

**Indicators:**
- Unusual DNS queries
- Suspicious social media activity
- Increased website traffic from unknown sources

**Phase 2: Weaponization**
**Objectives:**
- Create malicious payload
- Combine exploit with backdoor
- Ensure payload delivery capability

**Techniques:**
- Exploit development
- Payload generation
- Document weaponization
- Email template creation

**Tools:**
- Metasploit Framework
- Cobalt Strike
- Empire PowerShell
- Social Engineer Toolkit

**Artifacts:**
- Malicious documents
- Exploit code
- Command and control infrastructure

**Phase 3: Delivery**
**Objectives:**
- Transmit weapon to target
- Achieve initial system access
- Establish foothold in environment

**Techniques:**
- Spear phishing emails
- Watering hole attacks
- USB drops
- Supply chain compromise

**Vectors:**
- Email attachments
- Malicious links
- Removable media
- Third-party services

**Phase 4: Exploitation**
**Objectives:**
- Trigger payload execution
- Exploit application vulnerabilities
- Achieve code execution

**Common Exploits:**
- Buffer overflow attacks
- SQL injection
- Cross-site scripting
- Zero-day exploits
- Privilege escalation

**Techniques:**
- Memory corruption exploits
- Logic flaws exploitation
- Race condition attacks
- Return-oriented programming

**Phase 5: Installation**
**Objectives:**
- Install persistent backdoor
- Establish reliable access
- Survive system reboots

**Methods:**
- Registry modifications
- Service installation
- Scheduled task creation
- Startup folder placement
- DLL hijacking

**Persistence Techniques:**
- WMI event subscriptions
- Golden ticket attacks
- COM hijacking
- Image file execution options

**Phase 6: Command and Control (C2)**
**Objectives:**
- Establish communication channel
- Maintain persistent access
- Enable remote control

**Protocols:**
- HTTP/HTTPS
- DNS tunneling
- IRC channels
- Peer-to-peer networks

**Evasion Techniques:**
- Domain generation algorithms
- Fast flux networks
- Encrypted communications
- Traffic obfuscation

**Phase 7: Actions on Objectives**
**Objectives:**
- Achieve attack goals
- Collect target information
- Disrupt operations

**Common Actions:**
- Data exfiltration
- Lateral movement
- Privilege escalation
- System destruction
- Ransomware deployment

**Techniques:**
- Network reconnaissance
- Credential harvesting
- File system exploration
- Database access
- Email access

#### MITRE ATT&CK Framework

**Tactics (Why):**
1. Initial Access
2. Execution
3. Persistence
4. Privilege Escalation
5. Defense Evasion
6. Credential Access
7. Discovery
8. Lateral Movement
9. Collection
10. Command and Control
11. Exfiltration
12. Impact

**Techniques (How):**
- Specific methods within each tactic
- Over 200 documented techniques
- Real-world attack examples
- Mitigation strategies

**Procedures (Implementation):**
- Specific implementation of techniques
- Threat actor variations
- Tool-specific methods
- Environmental adaptations
