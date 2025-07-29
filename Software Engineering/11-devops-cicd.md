# DevOps and CI/CD

## Introduction

DevOps is a cultural and technical practice that combines software development (Dev) and IT operations (Ops) to shorten the development lifecycle and provide continuous delivery of high-quality software. CI/CD (Continuous Integration/Continuous Deployment) is a key component of DevOps practices.

## DevOps Overview

### What is DevOps?

DevOps is a set of practices, tools, and cultural philosophies that automate and integrate the processes between software development and IT operations teams. It emphasizes collaboration, communication, and integration between developers and operations professionals.

### DevOps Culture and Principles

#### Core Values
1. **Collaboration**: Breaking down silos between development and operations
2. **Communication**: Open and frequent communication between teams
3. **Integration**: Integrating processes and tools across the lifecycle
4. **Automation**: Automating repetitive tasks and processes
5. **Monitoring**: Continuous monitoring and feedback
6. **Learning**: Continuous learning and improvement

#### DevOps Principles
1. **Customer-Centric Action**: Focus on customer needs and feedback
2. **End-to-End Responsibility**: Teams own the entire lifecycle
3. **Continuous Improvement**: Constant learning and adaptation
4. **Automate Everything**: Reduce manual processes and errors
5. **Work as One Team**: Shared responsibility and accountability
6. **Monitor and Test Everything**: Comprehensive monitoring and testing

### Benefits of DevOps

#### Business Benefits
- **Faster Time to Market**: Quicker delivery of features and fixes
- **Improved Customer Satisfaction**: Better quality and faster response
- **Increased Revenue**: More frequent releases and faster innovation
- **Reduced Costs**: Efficient processes and reduced manual work

#### Technical Benefits
- **Improved Deployment Frequency**: More frequent, smaller deployments
- **Faster Recovery**: Quicker resolution of issues and incidents
- **Lower Change Failure Rate**: Better quality and fewer defects
- **Reduced Lead Time**: Faster development and deployment cycles

## Continuous Integration (CI)

### What is Continuous Integration?

Continuous Integration is the practice of frequently integrating code changes into a shared repository, where automated builds and tests are run to detect integration errors quickly.

### CI Principles

1. **Frequent Commits**: Developers commit code changes frequently (multiple times per day)
2. **Automated Builds**: Every commit triggers an automated build
3. **Automated Testing**: Comprehensive test suite runs with each build
4. **Fast Feedback**: Quick notification of build and test results
5. **Fix Broken Builds Immediately**: Broken builds get highest priority
6. **Keep Build Fast**: Builds should complete in under 10 minutes

### CI Process Flow

```
Developer → Commit Code → Version Control → CI Server → Build → Test → Feedback
    ↑                                                                      ↓
    └──────────────────── Fix Issues ←──────────────────────────────────────┘
```

### CI Best Practices

#### 1. Version Control Everything
```bash
# Project structure in version control
project/
├── src/                 # Source code
├── tests/              # Test files
├── scripts/            # Build and deployment scripts
├── docs/               # Documentation
├── Dockerfile          # Container configuration
├── docker-compose.yml  # Local development environment
├── .gitignore         # Git ignore rules
├── README.md          # Project documentation
└── .github/           # CI/CD workflows
    └── workflows/
        └── ci.yml     # GitHub Actions workflow
```

#### 2. Automate the Build
```yaml
# Example GitHub Actions CI workflow
name: Continuous Integration

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    
    strategy:
      matrix:
        node-version: [14.x, 16.x, 18.x]
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v3
    
    - name: Setup Node.js
      uses: actions/setup-node@v3
      with:
        node-version: ${{ matrix.node-version }}
        cache: 'npm'
    
    - name: Install dependencies
      run: npm ci
    
    - name: Run linting
      run: npm run lint
    
    - name: Run tests
      run: npm run test:coverage
    
    - name: Build application
      run: npm run build
    
    - name: Upload coverage reports
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage/lcov.info
        
    - name: Archive build artifacts
      uses: actions/upload-artifact@v3
      with:
        name: build-files
        path: dist/
```

#### 3. Test-Driven Development
```javascript
// Example test-first approach
// 1. Write failing test
describe('UserService', () => {
  it('should create user with valid data', async () => {
    const userData = {
      email: 'test@example.com',
      password: 'securePassword123'
    };
    
    const user = await userService.createUser(userData);
    
    expect(user).toBeDefined();
    expect(user.email).toBe(userData.email);
    expect(user.id).toBeDefined();
  });
});

// 2. Write minimal code to pass test
class UserService {
  async createUser(userData) {
    const user = {
      id: generateId(),
      email: userData.email,
      createdAt: new Date()
    };
    
    await this.userRepository.save(user);
    return user;
  }
}

// 3. Refactor and improve
```

#### 4. Build Pipeline Stages
```yaml
# Multi-stage pipeline example
stages:
  - name: Code Quality
    steps:
      - lint
      - security-scan
      - dependency-check
  
  - name: Testing
    steps:
      - unit-tests
      - integration-tests
      - contract-tests
  
  - name: Build
    steps:
      - compile
      - package
      - create-artifacts
  
  - name: Quality Gates
    steps:
      - code-coverage-check
      - performance-tests
      - security-tests
```

## Continuous Deployment (CD)

### What is Continuous Deployment?

Continuous Deployment extends CI by automatically deploying every change that passes the automated tests to production. Continuous Delivery is similar but requires manual approval for production deployment.

### CD vs Continuous Delivery

| Aspect | Continuous Delivery | Continuous Deployment |
|--------|-------------------|---------------------|
| Production Deploy | Manual approval required | Fully automated |
| Risk Level | Lower (human gate) | Higher (full automation) |
| Speed | Slower (approval delay) | Faster (immediate) |
| Best For | Critical systems | Non-critical systems |

### Deployment Strategies

#### 1. Blue-Green Deployment
```yaml
# Blue-Green deployment example
apiVersion: v1
kind: Service
metadata:
  name: app-service
spec:
  selector:
    app: myapp
    version: blue  # Switch to 'green' for deployment
  ports:
  - port: 80
    targetPort: 8080

---
# Blue environment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp-blue
spec:
  replicas: 3
  selector:
    matchLabels:
      app: myapp
      version: blue
  template:
    metadata:
      labels:
        app: myapp
        version: blue
    spec:
      containers:
      - name: app
        image: myapp:v1.0.0

---
# Green environment (new version)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp-green
spec:
  replicas: 3
  selector:
    matchLabels:
      app: myapp
      version: green
  template:
    metadata:
      labels:
        app: myapp
        version: green
    spec:
      containers:
      - name: app
        image: myapp:v1.1.0
```

#### 2. Canary Deployment
```yaml
# Canary deployment with Istio
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: myapp-vs
spec:
  http:
  - match:
    - headers:
        canary:
          exact: "true"
    route:
    - destination:
        host: myapp
        subset: v2
  - route:
    - destination:
        host: myapp
        subset: v1
      weight: 90
    - destination:
        host: myapp
        subset: v2
      weight: 10  # 10% traffic to new version
```

#### 3. Rolling Deployment
```yaml
# Rolling deployment configuration
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
spec:
  replicas: 10
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1    # Max 1 pod unavailable during update
      maxSurge: 1          # Max 1 extra pod during update
  template:
    spec:
      containers:
      - name: app
        image: myapp:v1.1.0
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
```

### CD Pipeline Example

```yaml
# Complete CD pipeline
name: CI/CD Pipeline

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run tests
        run: npm test
  
  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build Docker image
        run: |
          docker build -t myapp:${{ github.sha }} .
          docker tag myapp:${{ github.sha }} myapp:latest
      
      - name: Push to registry
        run: |
          echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
          docker push myapp:${{ github.sha }}
          docker push myapp:latest
  
  deploy-staging:
    needs: build
    runs-on: ubuntu-latest
    environment: staging
    steps:
      - name: Deploy to staging
        run: |
          kubectl set image deployment/myapp myapp=myapp:${{ github.sha }}
          kubectl rollout status deployment/myapp
  
  deploy-production:
    needs: deploy-staging
    runs-on: ubuntu-latest
    environment: production
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Deploy to production
        run: |
          kubectl set image deployment/myapp myapp=myapp:${{ github.sha }}
          kubectl rollout status deployment/myapp
```

## Infrastructure as Code (IaC)

### What is Infrastructure as Code?

Infrastructure as Code is the practice of managing and provisioning infrastructure through machine-readable definition files, rather than physical hardware configuration or interactive configuration tools.

### Benefits of IaC

1. **Version Control**: Infrastructure changes tracked in version control
2. **Reproducibility**: Consistent environment creation
3. **Automation**: Automated provisioning and management
4. **Documentation**: Infrastructure documented as code
5. **Cost Management**: Better resource optimization
6. **Disaster Recovery**: Quick environment recreation

### IaC Tools

#### 1. Terraform
```hcl
# Terraform example - AWS infrastructure
provider "aws" {
  region = "us-west-2"
}

resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = {
    Name = "main-vpc"
  }
}

resource "aws_subnet" "public" {
  count                   = 2
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.${count.index + 1}.0/24"
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true
  
  tags = {
    Name = "public-subnet-${count.index + 1}"
  }
}

resource "aws_eks_cluster" "main" {
  name     = "main-cluster"
  role_arn = aws_iam_role.cluster.arn
  version  = "1.21"

  vpc_config {
    subnet_ids = aws_subnet.public[*].id
  }

  depends_on = [
    aws_iam_role_policy_attachment.cluster_policy,
  ]
}
```

#### 2. AWS CloudFormation
```yaml
# CloudFormation template
AWSTemplateFormatVersion: '2010-09-09'
Description: 'Web application infrastructure'

Parameters:
  InstanceType:
    Type: String
    Default: t3.micro
    AllowedValues:
      - t3.micro
      - t3.small
      - t3.medium

Resources:
  VPC:
    Type: AWS::EC2::VPC
    Properties:
      CidrBlock: 10.0.0.0/16
      EnableDnsHostnames: true
      EnableDnsSupport: true
      Tags:
        - Key: Name
          Value: WebApp-VPC

  PublicSubnet:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      CidrBlock: 10.0.1.0/24
      AvailabilityZone: !Select [0, !GetAZs '']
      MapPublicIpOnLaunch: true

  EC2Instance:
    Type: AWS::EC2::Instance
    Properties:
      InstanceType: !Ref InstanceType
      ImageId: ami-0c02fb55956c7d316
      SubnetId: !Ref PublicSubnet
      SecurityGroupIds:
        - !Ref WebServerSecurityGroup

Outputs:
  InstanceId:
    Description: Instance ID
    Value: !Ref EC2Instance
```

#### 3. Kubernetes YAML
```yaml
# Kubernetes deployment and service
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-app
  labels:
    app: web-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: web-app
  template:
    metadata:
      labels:
        app: web-app
    spec:
      containers:
      - name: web-app
        image: nginx:1.21
        ports:
        - containerPort: 80
        resources:
          requests:
            memory: "64Mi"
            cpu: "250m"
          limits:
            memory: "128Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /
            port: 80
          initialDelaySeconds: 30
          periodSeconds: 10

---
apiVersion: v1
kind: Service
metadata:
  name: web-app-service
spec:
  selector:
    app: web-app
  ports:
  - protocol: TCP
    port: 80
    targetPort: 80
  type: LoadBalancer
```

## Containerization and Orchestration

### Docker

#### Dockerfile Best Practices
```dockerfile
# Multi-stage build for efficient images
FROM node:16-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

FROM node:16-alpine AS runtime

# Create non-root user
RUN addgroup -g 1001 -S nodejs
RUN adduser -S nextjs -u 1001

WORKDIR /app

# Copy application files
COPY --from=builder /app/node_modules ./node_modules
COPY --chown=nextjs:nodejs . .

# Switch to non-root user
USER nextjs

EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:3000/health || exit 1

CMD ["npm", "start"]
```

#### Docker Compose for Local Development
```yaml
# docker-compose.yml
version: '3.8'

services:
  web:
    build:
      context: .
      dockerfile: Dockerfile.dev
    ports:
      - "3000:3000"
    volumes:
      - .:/app
      - /app/node_modules
    environment:
      - NODE_ENV=development
    depends_on:
      - database
      - redis

  database:
    image: postgres:13-alpine
    environment:
      POSTGRES_DB: myapp
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:6-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

### Kubernetes

#### Deployment Patterns
```yaml
# Complete application deployment
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  database_url: "postgresql://postgres:password@db:5432/myapp"
  redis_url: "redis://redis:6379"

---
apiVersion: v1
kind: Secret
metadata:
  name: app-secrets
type: Opaque
data:
  api_key: <base64-encoded-key>
  jwt_secret: <base64-encoded-secret>

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: web-app
  template:
    metadata:
      labels:
        app: web-app
    spec:
      containers:
      - name: web-app
        image: myapp:latest
        ports:
        - containerPort: 3000
        env:
        - name: DATABASE_URL
          valueFrom:
            configMapKeyRef:
              name: app-config
              key: database_url
        - name: API_KEY
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: api_key
        livenessProbe:
          httpGet:
            path: /health
            port: 3000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 3000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: web-app-service
spec:
  selector:
    app: web-app
  ports:
  - port: 80
    targetPort: 3000
  type: ClusterIP

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: web-app-ingress
  annotations:
    kubernetes.io/ingress.class: "nginx"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - myapp.example.com
    secretName: myapp-tls
  rules:
  - host: myapp.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: web-app-service
            port:
              number: 80
```

## Monitoring and Observability

### The Three Pillars of Observability

#### 1. Metrics
```yaml
# Prometheus configuration
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'web-app'
    static_configs:
      - targets: ['web-app:3000']
    metrics_path: /metrics
    scrape_interval: 5s

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
```

```javascript
// Application metrics with Prometheus
const promClient = require('prom-client');

// Request duration histogram
const httpRequestDuration = new promClient.Histogram({
  name: 'http_request_duration_seconds',
  help: 'Duration of HTTP requests in seconds',
  labelNames: ['method', 'route', 'status_code'],
  buckets: [0.1, 0.3, 0.5, 0.7, 1, 3, 5, 7, 10]
});

// Request counter
const httpRequestTotal = new promClient.Counter({
  name: 'http_requests_total',
  help: 'Total number of HTTP requests',
  labelNames: ['method', 'route', 'status_code']
});

// Middleware to collect metrics
app.use((req, res, next) => {
  const startTime = Date.now();
  
  res.on('finish', () => {
    const duration = (Date.now() - startTime) / 1000;
    
    httpRequestDuration
      .labels(req.method, req.route?.path || req.path, res.statusCode)
      .observe(duration);
    
    httpRequestTotal
      .labels(req.method, req.route?.path || req.path, res.statusCode)
      .inc();
  });
  
  next();
});
```

#### 2. Logging
```javascript
// Structured logging with Winston
const winston = require('winston');

const logger = winston.createLogger({
  level: 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  defaultMeta: { service: 'web-app' },
  transports: [
    new winston.transports.File({ filename: 'error.log', level: 'error' }),
    new winston.transports.File({ filename: 'combined.log' }),
    new winston.transports.Console({
      format: winston.format.simple()
    })
  ]
});

// Usage
logger.info('User logged in', {
  userId: user.id,
  email: user.email,
  ip: req.ip,
  userAgent: req.get('User-Agent')
});

logger.error('Database connection failed', {
  error: error.message,
  stack: error.stack,
  database: 'postgresql'
});
```

#### 3. Tracing
```javascript
// Distributed tracing with OpenTelemetry
const { NodeSDK } = require('@opentelemetry/sdk-node');
const { getNodeAutoInstrumentations } = require('@opentelemetry/auto-instrumentations-node');
const { JaegerExporter } = require('@opentelemetry/exporter-jaeger');

const jaegerExporter = new JaegerExporter({
  endpoint: 'http://jaeger:14268/api/traces',
});

const sdk = new NodeSDK({
  traceExporter: jaegerExporter,
  instrumentations: [getNodeAutoInstrumentations()],
});

sdk.start();

// Custom spans
const opentelemetry = require('@opentelemetry/api');

async function processOrder(orderId) {
  const tracer = opentelemetry.trace.getTracer('order-service');
  
  return tracer.startActiveSpan('process-order', async (span) => {
    try {
      span.setAttributes({
        'order.id': orderId,
        'service.name': 'order-service'
      });
      
      // Process order logic
      const order = await getOrder(orderId);
      await validateOrder(order);
      await chargePayment(order);
      await fulfillOrder(order);
      
      span.setStatus({ code: opentelemetry.SpanStatusCode.OK });
      return order;
    } catch (error) {
      span.recordException(error);
      span.setStatus({
        code: opentelemetry.SpanStatusCode.ERROR,
        message: error.message
      });
      throw error;
    } finally {
      span.end();
    }
  });
}
```

### Alerting and Incident Response

#### Prometheus Alerting Rules
```yaml
# alerting-rules.yml
groups:
  - name: web-app-alerts
    rules:
      - alert: HighErrorRate
        expr: |
          (
            rate(http_requests_total{status_code=~"5.."}[5m]) /
            rate(http_requests_total[5m])
          ) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }} for the last 5 minutes"

      - alert: HighResponseTime
        expr: |
          histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 0.5
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High response time detected"
          description: "95th percentile response time is {{ $value }}s"

      - alert: PodCrashLooping
        expr: |
          rate(kube_pod_container_status_restarts_total[15m]) > 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Pod is crash looping"
          description: "Pod {{ $labels.pod }} in namespace {{ $labels.namespace }} is restarting frequently"
```

## DevOps Tools and Technologies

### CI/CD Platforms

#### GitHub Actions
```yaml
# Advanced GitHub Actions workflow
name: Full CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        node-version: [16.x, 18.x]
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: ${{ matrix.node-version }}
          cache: 'npm'
      
      - name: Install dependencies
        run: npm ci
      
      - name: Run linting
        run: npm run lint
      
      - name: Run unit tests
        run: npm run test:unit
      
      - name: Run integration tests
        run: npm run test:integration
        env:
          DATABASE_URL: postgresql://postgres:postgres@localhost:5432/testdb
    
    services:
      postgres:
        image: postgres:13
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Run Snyk to check for vulnerabilities
        uses: snyk/actions/node@master
        env:
          SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
      
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          scan-ref: '.'

  build:
    needs: [test, security]
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Docker Buildx
        uses: docker/setup-buildx-action@v2
      
      - name: Login to Container Registry
        uses: docker/login-action@v2
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v4
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=sha,prefix={{branch}}-
      
      - name: Build and push Docker image
        uses: docker/build-push-action@v4
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  deploy-staging:
    if: github.ref == 'refs/heads/develop'
    needs: build
    runs-on: ubuntu-latest
    environment: staging
    
    steps:
      - name: Deploy to staging
        run: |
          echo "Deploying to staging environment"
          # kubectl or helm deployment commands

  deploy-production:
    if: github.ref == 'refs/heads/main'
    needs: build
    runs-on: ubuntu-latest
    environment: production
    
    steps:
      - name: Deploy to production
        run: |
          echo "Deploying to production environment"
          # Blue-green or canary deployment
```

#### Jenkins Pipeline
```groovy
// Jenkinsfile
pipeline {
    agent any
    
    environment {
        DOCKER_REGISTRY = 'your-registry.com'
        DOCKER_REPO = 'myapp'
        KUBECONFIG = credentials('kubeconfig')
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        
        stage('Test') {
            parallel {
                stage('Unit Tests') {
                    steps {
                        sh 'npm install'
                        sh 'npm run test:unit'
                    }
                    post {
                        always {
                            publishTestResults testResultsPattern: 'test-results.xml'
                            publishCoverageGlobalResults parsers: [[$class: 'IstanbulCoverageParser']]
                        }
                    }
                }
                
                stage('Security Scan') {
                    steps {
                        sh 'npm audit'
                        sh 'docker run --rm -v $(pwd):/app -w /app aquasec/trivy fs .'
                    }
                }
            }
        }
        
        stage('Build') {
            steps {
                script {
                    def image = docker.build("${DOCKER_REGISTRY}/${DOCKER_REPO}:${BUILD_NUMBER}")
                    docker.withRegistry("https://${DOCKER_REGISTRY}", 'docker-registry-credentials') {
                        image.push()
                        image.push('latest')
                    }
                }
            }
        }
        
        stage('Deploy to Staging') {
            when {
                branch 'develop'
            }
            steps {
                sh """
                    helm upgrade --install myapp-staging ./helm-chart \
                        --set image.tag=${BUILD_NUMBER} \
                        --set environment=staging \
                        --namespace staging
                """
            }
        }
        
        stage('Integration Tests') {
            when {
                branch 'develop'
            }
            steps {
                sh 'npm run test:integration:staging'
            }
        }
        
        stage('Deploy to Production') {
            when {
                branch 'main'
            }
            steps {
                input message: 'Deploy to production?', ok: 'Deploy'
                sh """
                    helm upgrade --install myapp-prod ./helm-chart \
                        --set image.tag=${BUILD_NUMBER} \
                        --set environment=production \
                        --namespace production
                """
            }
        }
    }
    
    post {
        always {
            cleanWs()
        }
        
        failure {
            slackSend channel: '#deployments',
                     color: 'danger',
                     message: "Build failed: ${env.JOB_NAME} - ${env.BUILD_NUMBER}"
        }
        
        success {
            slackSend channel: '#deployments',
                     color: 'good',
                     message: "Build successful: ${env.JOB_NAME} - ${env.BUILD_NUMBER}"
        }
    }
}
```

### Configuration Management

#### Ansible Playbook
```yaml
# deploy-application.yml
---
- name: Deploy web application
  hosts: web_servers
  become: yes
  vars:
    app_name: myapp
    app_version: "{{ version | default('latest') }}"
    app_port: 3000
    
  tasks:
    - name: Update package cache
      apt:
        update_cache: yes
        cache_valid_time: 3600
    
    - name: Install Docker
      apt:
        name: docker.io
        state: present
    
    - name: Start Docker service
      systemd:
        name: docker
        state: started
        enabled: yes
    
    - name: Pull application image
      docker_image:
        name: "{{ docker_registry }}/{{ app_name }}:{{ app_version }}"
        source: pull
    
    - name: Stop existing container
      docker_container:
        name: "{{ app_name }}"
        state: stopped
      ignore_errors: yes
    
    - name: Remove existing container
      docker_container:
        name: "{{ app_name }}"
        state: absent
      ignore_errors: yes
    
    - name: Start new container
      docker_container:
        name: "{{ app_name }}"
        image: "{{ docker_registry }}/{{ app_name }}:{{ app_version }}"
        state: started
        restart_policy: always
        ports:
          - "{{ app_port }}:{{ app_port }}"
        env:
          NODE_ENV: production
          DATABASE_URL: "{{ database_url }}"
        healthcheck:
          test: ["CMD", "curl", "-f", "http://localhost:{{ app_port }}/health"]
          interval: 30s
          timeout: 10s
          retries: 3
    
    - name: Wait for application to be ready
      uri:
        url: "http://localhost:{{ app_port }}/health"
        status_code: 200
      retries: 10
      delay: 10
```

## Best Practices and Patterns

### DevOps Best Practices

#### 1. Cultural Practices
- **Shared Responsibility**: Development and operations teams share responsibility for application lifecycle
- **Collaboration**: Regular communication and collaboration between teams
- **Blameless Culture**: Focus on learning from failures rather than assigning blame
- **Continuous Learning**: Regular training and knowledge sharing

#### 2. Technical Practices
- **Everything as Code**: Infrastructure, configuration, and deployment pipelines as code
- **Immutable Infrastructure**: Replace infrastructure rather than modifying it
- **Microservices**: Small, independent services that can be deployed separately
- **API-First Design**: Design APIs before implementing services

#### 3. Process Practices
- **Small, Frequent Releases**: Deploy small changes frequently rather than large releases
- **Feature Flags**: Use feature toggles to control feature rollout
- **A/B Testing**: Test changes with subset of users before full rollout
- **Gradual Rollouts**: Use canary deployments and blue-green deployments

### Security in DevOps (DevSecOps)

#### Security in CI/CD Pipeline
```yaml
# Security-focused pipeline
name: Secure CI/CD

on: [push, pull_request]

jobs:
  security-checks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      # Dependency vulnerability scanning
      - name: Run Snyk vulnerability scan
        uses: snyk/actions/node@master
        env:
          SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
      
      # Static Application Security Testing (SAST)
      - name: Run CodeQL Analysis
        uses: github/codeql-action/analyze@v2
        with:
          languages: javascript
      
      # Container image scanning
      - name: Build Docker image
        run: docker build -t myapp:test .
      
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: 'myapp:test'
          format: 'sarif'
          output: 'trivy-results.sarif'
      
      # Infrastructure as Code scanning
      - name: Run Checkov
        uses: bridgecrewio/checkov-action@master
        with:
          directory: .
          framework: terraform
          
      # Secret scanning
      - name: Run GitLeaks
        uses: zricethezav/gitleaks-action@master
```

#### Security Best Practices
1. **Shift Security Left**: Integrate security early in development process
2. **Secrets Management**: Use proper secret management tools (HashiCorp Vault, AWS Secrets Manager)
3. **Least Privilege**: Grant minimum necessary permissions
4. **Regular Updates**: Keep dependencies and base images updated
5. **Security Scanning**: Automated security scanning in CI/CD pipeline

### Disaster Recovery and Business Continuity

#### Backup Strategies
```bash
#!/bin/bash
# Automated backup script

# Database backup
kubectl exec deployment/postgres -- pg_dump -U postgres myapp > backup-$(date +%Y%m%d-%H%M%S).sql

# Upload to S3
aws s3 cp backup-*.sql s3://myapp-backups/database/

# Application data backup
kubectl get pv -o yaml > pv-backup-$(date +%Y%m%d).yaml

# Configuration backup
kubectl get configmaps -o yaml > configmaps-backup-$(date +%Y%m%d).yaml
kubectl get secrets -o yaml > secrets-backup-$(date +%Y%m%d).yaml

# Clean up local files older than 7 days
find . -name "backup-*" -mtime +7 -delete
```

#### Disaster Recovery Testing
```yaml
# Disaster recovery test automation
apiVersion: batch/v1
kind: CronJob
metadata:
  name: dr-test
spec:
  schedule: "0 2 * * 0"  # Weekly on Sunday at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: dr-test
            image: dr-test:latest
            command:
            - /bin/bash
            - -c
            - |
              # Test database restoration
              kubectl exec deployment/postgres -- psql -U postgres -c "CREATE DATABASE drtest;"
              kubectl exec deployment/postgres -- pg_restore -U postgres -d drtest /backups/latest.sql
              
              # Test application deployment
              helm install drtest-app ./helm-chart --set environment=dr-test
              
              # Run smoke tests
              npm run test:smoke -- --url=http://drtest-app
              
              # Cleanup
              helm uninstall drtest-app
              kubectl exec deployment/postgres -- psql -U postgres -c "DROP DATABASE drtest;"
          restartPolicy: OnFailure
```

## Measuring DevOps Success

### Key Performance Indicators (KPIs)

#### DORA Metrics
1. **Deployment Frequency**: How often deployments occur
2. **Lead Time for Changes**: Time from commit to production
3. **Mean Time to Recovery (MTTR)**: Time to recover from failures
4. **Change Failure Rate**: Percentage of deployments causing failures

#### Additional Metrics
- **Build Success Rate**: Percentage of successful builds
- **Test Coverage**: Percentage of code covered by tests
- **Infrastructure Utilization**: Resource usage efficiency
- **Customer Satisfaction**: User feedback and satisfaction scores

### Monitoring Dashboard Example
```yaml
# Grafana dashboard configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: devops-dashboard
data:
  dashboard.json: |
    {
      "dashboard": {
        "title": "DevOps Metrics",
        "panels": [
          {
            "title": "Deployment Frequency",
            "type": "stat",
            "targets": [
              {
                "expr": "increase(deployments_total[7d])"
              }
            ]
          },
          {
            "title": "Lead Time",
            "type": "histogram",
            "targets": [
              {
                "expr": "histogram_quantile(0.5, lead_time_seconds_bucket)"
              }
            ]
          },
          {
            "title": "Error Rate",
            "type": "graph",
            "targets": [
              {
                "expr": "rate(http_requests_total{status=~\"5..\"}[5m])"
              }
            ]
          }
        ]
      }
    }
```

## Summary

DevOps and CI/CD are essential practices for modern software development. Key takeaways:

1. **Cultural Transformation**: DevOps is primarily about people and culture, not just tools
2. **Automation First**: Automate everything from testing to deployment
3. **Continuous Integration**: Integrate code changes frequently with automated testing
4. **Continuous Deployment**: Deploy small changes frequently to reduce risk
5. **Infrastructure as Code**: Manage infrastructure through version-controlled code
6. **Monitoring and Observability**: Implement comprehensive monitoring and logging
7. **Security Integration**: Build security into every stage of the pipeline
8. **Measure and Improve**: Use metrics to continuously improve processes

Success in DevOps requires commitment to both technical practices and cultural change. Start with simple automation and gradually build more sophisticated pipelines as your team's capabilities mature.
