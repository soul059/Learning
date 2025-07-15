# Authentication and Security in Python APIs

## Table of Contents
1. [Authentication Methods](#authentication-methods)
2. [JWT (JSON Web Tokens)](#jwt-json-web-tokens)
3. [OAuth 2.0](#oauth-20)
4. [API Security Best Practices](#api-security-best-practices)
5. [Rate Limiting](#rate-limiting)
6. [Input Validation and Sanitization](#input-validation-and-sanitization)
7. [HTTPS and SSL](#https-and-ssl)

## Authentication Methods

### 1. Basic Authentication
```python
import base64
from functools import wraps
from flask import Flask, request, jsonify

app = Flask(__name__)

# Simple user database (use proper database in production)
users = {
    "admin": "password123",
    "user1": "mypassword"
}

def check_auth(username, password):
    """Check if username/password combination is valid."""
    return username in users and users[username] == password

def authenticate():
    """Send 401 response for authentication failure."""
    return jsonify({
        'error': 'Authentication required',
        'message': 'Please provide valid credentials'
    }), 401

def requires_auth(f):
    """Decorator for requiring authentication."""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated

@app.route('/api/protected')
@requires_auth
def protected():
    return jsonify({'message': 'This is a protected endpoint'})

# Manual header parsing for Basic Auth
def parse_auth_header():
    """Parse Authorization header manually."""
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Basic '):
        return None, None
    
    try:
        # Decode base64 credentials
        credentials = base64.b64decode(auth_header[6:]).decode('utf-8')
        username, password = credentials.split(':', 1)
        return username, password
    except (ValueError, UnicodeDecodeError):
        return None, None

@app.route('/api/manual-auth')
def manual_auth():
    username, password = parse_auth_header()
    if not username or not check_auth(username, password):
        return authenticate()
    
    return jsonify({'message': f'Hello {username}'})
```

### 2. Token-Based Authentication
```python
import secrets
import datetime
from flask import Flask, request, jsonify
from functools import wraps

app = Flask(__name__)

# Token storage (use Redis or database in production)
active_tokens = {}
users = {"admin": "password123", "user1": "mypassword"}

def generate_token():
    """Generate a secure random token."""
    return secrets.token_urlsafe(32)

def create_token(username):
    """Create and store a new token for user."""
    token = generate_token()
    expiry = datetime.datetime.utcnow() + datetime.timedelta(hours=24)
    
    active_tokens[token] = {
        'username': username,
        'expires_at': expiry,
        'created_at': datetime.datetime.utcnow()
    }
    
    return token

def validate_token(token):
    """Validate token and return user info."""
    if token not in active_tokens:
        return None
    
    token_data = active_tokens[token]
    
    # Check if token is expired
    if datetime.datetime.utcnow() > token_data['expires_at']:
        del active_tokens[token]
        return None
    
    return token_data

def token_required(f):
    """Decorator for token-based authentication."""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        
        if not token:
            return jsonify({'error': 'Token is missing'}), 401
        
        if token.startswith('Bearer '):
            token = token[7:]  # Remove 'Bearer ' prefix
        
        token_data = validate_token(token)
        if not token_data:
            return jsonify({'error': 'Token is invalid or expired'}), 401
        
        # Add user info to request context
        request.current_user = token_data['username']
        return f(*args, **kwargs)
    
    return decorated

@app.route('/api/login', methods=['POST'])
def login():
    """Login endpoint to get token."""
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400
    
    if username not in users or users[username] != password:
        return jsonify({'error': 'Invalid credentials'}), 401
    
    token = create_token(username)
    
    return jsonify({
        'token': token,
        'expires_in': 86400,  # 24 hours in seconds
        'token_type': 'Bearer'
    })

@app.route('/api/logout', methods=['POST'])
@token_required
def logout():
    """Logout endpoint to invalidate token."""
    token = request.headers.get('Authorization')
    if token.startswith('Bearer '):
        token = token[7:]
    
    if token in active_tokens:
        del active_tokens[token]
    
    return jsonify({'message': 'Successfully logged out'})

@app.route('/api/profile')
@token_required
def profile():
    """Protected endpoint using token authentication."""
    return jsonify({
        'username': request.current_user,
        'message': 'This is your profile'
    })

# Token refresh endpoint
@app.route('/api/refresh', methods=['POST'])
@token_required
def refresh_token():
    """Refresh token endpoint."""
    old_token = request.headers.get('Authorization')[7:]  # Remove 'Bearer '
    username = request.current_user
    
    # Delete old token
    if old_token in active_tokens:
        del active_tokens[old_token]
    
    # Create new token
    new_token = create_token(username)
    
    return jsonify({
        'token': new_token,
        'expires_in': 86400,
        'token_type': 'Bearer'
    })
```

## JWT (JSON Web Tokens)

### 1. JWT Implementation with PyJWT
```python
import jwt
import datetime
from flask import Flask, request, jsonify
from functools import wraps

app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = 'your-secret-key-change-this'
app.config['JWT_ALGORITHM'] = 'HS256'
app.config['JWT_EXPIRATION_DELTA'] = datetime.timedelta(hours=24)

users = {
    "admin": {"password": "password123", "role": "admin"},
    "user1": {"password": "mypassword", "role": "user"}
}

def generate_jwt_token(username, role):
    """Generate JWT token for user."""
    payload = {
        'username': username,
        'role': role,
        'exp': datetime.datetime.utcnow() + app.config['JWT_EXPIRATION_DELTA'],
        'iat': datetime.datetime.utcnow(),
        'iss': 'your-api-name'  # Issuer
    }
    
    token = jwt.encode(
        payload,
        app.config['JWT_SECRET_KEY'],
        algorithm=app.config['JWT_ALGORITHM']
    )
    
    return token

def decode_jwt_token(token):
    """Decode and validate JWT token."""
    try:
        payload = jwt.decode(
            token,
            app.config['JWT_SECRET_KEY'],
            algorithms=[app.config['JWT_ALGORITHM']]
        )
        return payload
    except jwt.ExpiredSignatureError:
        return {'error': 'Token has expired'}
    except jwt.InvalidTokenError:
        return {'error': 'Invalid token'}

def jwt_required(roles=None):
    """Decorator for JWT authentication with optional role checking."""
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            token = request.headers.get('Authorization')
            
            if not token:
                return jsonify({'error': 'Token is missing'}), 401
            
            if token.startswith('Bearer '):
                token = token[7:]
            
            payload = decode_jwt_token(token)
            
            if 'error' in payload:
                return jsonify({'error': payload['error']}), 401
            
            # Check role if specified
            if roles and payload.get('role') not in roles:
                return jsonify({'error': 'Insufficient permissions'}), 403
            
            request.current_user = {
                'username': payload['username'],
                'role': payload['role']
            }
            
            return f(*args, **kwargs)
        return decorated
    return decorator

@app.route('/api/login', methods=['POST'])
def login():
    """Login endpoint with JWT."""
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400
    
    user = users.get(username)
    if not user or user['password'] != password:
        return jsonify({'error': 'Invalid credentials'}), 401
    
    token = generate_jwt_token(username, user['role'])
    
    return jsonify({
        'access_token': token,
        'token_type': 'Bearer',
        'expires_in': int(app.config['JWT_EXPIRATION_DELTA'].total_seconds()),
        'user': {
            'username': username,
            'role': user['role']
        }
    })

@app.route('/api/profile')
@jwt_required()
def profile():
    """Get user profile."""
    return jsonify({
        'user': request.current_user,
        'message': 'Profile data'
    })

@app.route('/api/admin')
@jwt_required(roles=['admin'])
def admin_only():
    """Admin-only endpoint."""
    return jsonify({
        'message': 'This is admin-only content',
        'user': request.current_user
    })

# JWT with refresh tokens
@app.route('/api/refresh', methods=['POST'])
def refresh():
    """Refresh JWT token."""
    refresh_token = request.json.get('refresh_token')
    
    if not refresh_token:
        return jsonify({'error': 'Refresh token required'}), 400
    
    # Validate refresh token (implement your logic)
    # For simplicity, we'll decode it as a regular JWT
    payload = decode_jwt_token(refresh_token)
    
    if 'error' in payload:
        return jsonify({'error': 'Invalid refresh token'}), 401
    
    # Generate new access token
    new_token = generate_jwt_token(payload['username'], payload['role'])
    
    return jsonify({
        'access_token': new_token,
        'token_type': 'Bearer',
        'expires_in': int(app.config['JWT_EXPIRATION_DELTA'].total_seconds())
    })
```

### 2. FastAPI JWT Implementation
```python
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import jwt
from datetime import datetime, timedelta
from passlib.context import CryptContext

app = FastAPI()

# Configuration
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# Models
class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int

class TokenData(BaseModel):
    username: str = None

class User(BaseModel):
    username: str
    email: str = None
    role: str = "user"

class UserInDB(User):
    hashed_password: str

# Fake database
fake_users_db = {
    "testuser": {
        "username": "testuser",
        "email": "test@example.com",
        "role": "user",
        "hashed_password": pwd_context.hash("testpass")
    },
    "admin": {
        "username": "admin",
        "email": "admin@example.com",
        "role": "admin",
        "hashed_password": pwd_context.hash("adminpass")
    }
}

def verify_password(plain_password, hashed_password):
    """Verify password against hash."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    """Hash password."""
    return pwd_context.hash(password)

def get_user(username: str):
    """Get user from database."""
    if username in fake_users_db:
        user_dict = fake_users_db[username]
        return UserInDB(**user_dict)

def authenticate_user(username: str, password: str):
    """Authenticate user credentials."""
    user = get_user(username)
    if not user or not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: timedelta = None):
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current user from JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except jwt.PyJWTError:
        raise credentials_exception
    
    user = get_user(username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    """Get current active user."""
    return current_user

def require_roles(allowed_roles: list):
    """Dependency to require specific roles."""
    def role_checker(current_user: User = Depends(get_current_active_user)):
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions"
            )
        return current_user
    return role_checker

@app.post("/api/token", response_model=Token)
async def login_for_access_token(username: str, password: str):
    """Login endpoint to get access token."""
    user = authenticate_user(username, password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "role": user.role},
        expires_delta=access_token_expires
    )
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60
    }

@app.get("/api/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    """Get current user profile."""
    return current_user

@app.get("/api/admin")
async def admin_endpoint(current_user: User = Depends(require_roles(["admin"]))):
    """Admin-only endpoint."""
    return {"message": "Hello admin!", "user": current_user.username}
```

## OAuth 2.0

### 1. OAuth 2.0 with Google
```python
from flask import Flask, request, redirect, session, jsonify, url_for
import requests
import secrets

app = Flask(__name__)
app.secret_key = 'your-secret-key'

# OAuth 2.0 Configuration
GOOGLE_CLIENT_ID = 'your-google-client-id'
GOOGLE_CLIENT_SECRET = 'your-google-client-secret'
GOOGLE_REDIRECT_URI = 'http://localhost:5000/api/auth/google/callback'

GOOGLE_AUTHORIZATION_URL = 'https://accounts.google.com/o/oauth2/auth'
GOOGLE_TOKEN_URL = 'https://oauth2.googleapis.com/token'
GOOGLE_USER_INFO_URL = 'https://www.googleapis.com/oauth2/v2/userinfo'

@app.route('/api/auth/google')
def google_auth():
    """Initiate Google OAuth flow."""
    # Generate state for CSRF protection
    state = secrets.token_urlsafe(32)
    session['oauth_state'] = state
    
    # Build authorization URL
    params = {
        'client_id': GOOGLE_CLIENT_ID,
        'redirect_uri': GOOGLE_REDIRECT_URI,
        'scope': 'openid email profile',
        'response_type': 'code',
        'state': state,
        'access_type': 'offline',  # For refresh tokens
        'prompt': 'consent'
    }
    
    auth_url = GOOGLE_AUTHORIZATION_URL + '?' + '&'.join([f"{k}={v}" for k, v in params.items()])
    return redirect(auth_url)

@app.route('/api/auth/google/callback')
def google_callback():
    """Handle Google OAuth callback."""
    # Verify state parameter
    if request.args.get('state') != session.get('oauth_state'):
        return jsonify({'error': 'Invalid state parameter'}), 400
    
    # Get authorization code
    code = request.args.get('code')
    if not code:
        return jsonify({'error': 'Authorization code not provided'}), 400
    
    # Exchange code for tokens
    token_data = {
        'client_id': GOOGLE_CLIENT_ID,
        'client_secret': GOOGLE_CLIENT_SECRET,
        'code': code,
        'grant_type': 'authorization_code',
        'redirect_uri': GOOGLE_REDIRECT_URI
    }
    
    token_response = requests.post(GOOGLE_TOKEN_URL, data=token_data)
    token_json = token_response.json()
    
    if 'access_token' not in token_json:
        return jsonify({'error': 'Failed to obtain access token'}), 400
    
    # Get user information
    headers = {'Authorization': f"Bearer {token_json['access_token']}"}
    user_response = requests.get(GOOGLE_USER_INFO_URL, headers=headers)
    user_info = user_response.json()
    
    # Store user session
    session['user'] = {
        'id': user_info['id'],
        'email': user_info['email'],
        'name': user_info['name'],
        'picture': user_info.get('picture')
    }
    
    # Generate your own JWT token for API access
    jwt_token = generate_jwt_token(user_info['email'], 'user')
    
    return jsonify({
        'message': 'Successfully authenticated',
        'user': session['user'],
        'access_token': jwt_token
    })

@app.route('/api/auth/logout')
def logout():
    """Logout user."""
    session.pop('user', None)
    session.pop('oauth_state', None)
    return jsonify({'message': 'Successfully logged out'})

# GitHub OAuth example
GITHUB_CLIENT_ID = 'your-github-client-id'
GITHUB_CLIENT_SECRET = 'your-github-client-secret'

@app.route('/api/auth/github')
def github_auth():
    """Initiate GitHub OAuth flow."""
    state = secrets.token_urlsafe(32)
    session['oauth_state'] = state
    
    auth_url = f"https://github.com/login/oauth/authorize?client_id={GITHUB_CLIENT_ID}&redirect_uri=http://localhost:5000/api/auth/github/callback&scope=user:email&state={state}"
    return redirect(auth_url)

@app.route('/api/auth/github/callback')
def github_callback():
    """Handle GitHub OAuth callback."""
    if request.args.get('state') != session.get('oauth_state'):
        return jsonify({'error': 'Invalid state parameter'}), 400
    
    code = request.args.get('code')
    if not code:
        return jsonify({'error': 'Authorization code not provided'}), 400
    
    # Exchange code for access token
    token_data = {
        'client_id': GITHUB_CLIENT_ID,
        'client_secret': GITHUB_CLIENT_SECRET,
        'code': code
    }
    
    headers = {'Accept': 'application/json'}
    token_response = requests.post('https://github.com/login/oauth/access_token', 
                                 data=token_data, headers=headers)
    token_json = token_response.json()
    
    # Get user information
    headers = {
        'Authorization': f"token {token_json['access_token']}",
        'Accept': 'application/json'
    }
    
    user_response = requests.get('https://api.github.com/user', headers=headers)
    user_info = user_response.json()
    
    session['user'] = {
        'id': user_info['id'],
        'username': user_info['login'],
        'name': user_info.get('name'),
        'email': user_info.get('email'),
        'avatar_url': user_info.get('avatar_url')
    }
    
    jwt_token = generate_jwt_token(user_info['login'], 'user')
    
    return jsonify({
        'message': 'Successfully authenticated with GitHub',
        'user': session['user'],
        'access_token': jwt_token
    })
```

### 2. OAuth 2.0 Server Implementation
```python
from flask import Flask, request, jsonify, redirect
import secrets
import base64
from datetime import datetime, timedelta

app = Flask(__name__)

# OAuth 2.0 Server Storage (use database in production)
clients = {
    'client_id_123': {
        'client_secret': 'client_secret_456',
        'redirect_uris': ['http://localhost:3000/callback'],
        'name': 'Test Client App'
    }
}

authorization_codes = {}
access_tokens = {}
refresh_tokens = {}

def generate_code():
    """Generate authorization code."""
    return secrets.token_urlsafe(32)

def generate_token():
    """Generate access/refresh token."""
    return secrets.token_urlsafe(64)

@app.route('/oauth/authorize')
def authorize():
    """OAuth 2.0 authorization endpoint."""
    # Extract parameters
    client_id = request.args.get('client_id')
    redirect_uri = request.args.get('redirect_uri')
    response_type = request.args.get('response_type')
    scope = request.args.get('scope', '')
    state = request.args.get('state', '')
    
    # Validate client
    if client_id not in clients:
        return jsonify({'error': 'invalid_client'}), 400
    
    client = clients[client_id]
    
    # Validate redirect URI
    if redirect_uri not in client['redirect_uris']:
        return jsonify({'error': 'invalid_redirect_uri'}), 400
    
    # Validate response type
    if response_type != 'code':
        error_url = f"{redirect_uri}?error=unsupported_response_type&state={state}"
        return redirect(error_url)
    
    # In real implementation, show user consent form
    # For demo, auto-approve
    
    # Generate authorization code
    code = generate_code()
    authorization_codes[code] = {
        'client_id': client_id,
        'redirect_uri': redirect_uri,
        'scope': scope,
        'user_id': 'user123',  # In real app, get from authenticated user
        'expires_at': datetime.utcnow() + timedelta(minutes=10)
    }
    
    # Redirect back to client with code
    callback_url = f"{redirect_uri}?code={code}&state={state}"
    return redirect(callback_url)

@app.route('/oauth/token', methods=['POST'])
def token():
    """OAuth 2.0 token endpoint."""
    grant_type = request.form.get('grant_type')
    
    # Client authentication
    auth_header = request.headers.get('Authorization')
    if auth_header and auth_header.startswith('Basic '):
        # Basic authentication
        credentials = base64.b64decode(auth_header[6:]).decode('utf-8')
        client_id, client_secret = credentials.split(':', 1)
    else:
        # Form parameters
        client_id = request.form.get('client_id')
        client_secret = request.form.get('client_secret')
    
    # Validate client
    if client_id not in clients or clients[client_id]['client_secret'] != client_secret:
        return jsonify({'error': 'invalid_client'}), 401
    
    if grant_type == 'authorization_code':
        return handle_authorization_code_grant()
    elif grant_type == 'refresh_token':
        return handle_refresh_token_grant()
    else:
        return jsonify({'error': 'unsupported_grant_type'}), 400

def handle_authorization_code_grant():
    """Handle authorization code grant."""
    code = request.form.get('code')
    redirect_uri = request.form.get('redirect_uri')
    
    # Validate authorization code
    if code not in authorization_codes:
        return jsonify({'error': 'invalid_grant'}), 400
    
    code_data = authorization_codes[code]
    
    # Check expiration
    if datetime.utcnow() > code_data['expires_at']:
        del authorization_codes[code]
        return jsonify({'error': 'invalid_grant'}), 400
    
    # Validate redirect URI
    if redirect_uri != code_data['redirect_uri']:
        return jsonify({'error': 'invalid_grant'}), 400
    
    # Generate tokens
    access_token = generate_token()
    refresh_token = generate_token()
    
    # Store tokens
    access_tokens[access_token] = {
        'client_id': code_data['client_id'],
        'user_id': code_data['user_id'],
        'scope': code_data['scope'],
        'expires_at': datetime.utcnow() + timedelta(hours=1)
    }
    
    refresh_tokens[refresh_token] = {
        'client_id': code_data['client_id'],
        'user_id': code_data['user_id'],
        'scope': code_data['scope']
    }
    
    # Clean up authorization code
    del authorization_codes[code]
    
    return jsonify({
        'access_token': access_token,
        'token_type': 'Bearer',
        'expires_in': 3600,
        'refresh_token': refresh_token,
        'scope': code_data['scope']
    })

def handle_refresh_token_grant():
    """Handle refresh token grant."""
    refresh_token = request.form.get('refresh_token')
    
    if refresh_token not in refresh_tokens:
        return jsonify({'error': 'invalid_grant'}), 400
    
    token_data = refresh_tokens[refresh_token]
    
    # Generate new access token
    access_token = generate_token()
    access_tokens[access_token] = {
        'client_id': token_data['client_id'],
        'user_id': token_data['user_id'],
        'scope': token_data['scope'],
        'expires_at': datetime.utcnow() + timedelta(hours=1)
    }
    
    return jsonify({
        'access_token': access_token,
        'token_type': 'Bearer',
        'expires_in': 3600,
        'scope': token_data['scope']
    })

@app.route('/api/user')
def get_user():
    """Protected resource endpoint."""
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({'error': 'invalid_token'}), 401
    
    token = auth_header[7:]
    
    if token not in access_tokens:
        return jsonify({'error': 'invalid_token'}), 401
    
    token_data = access_tokens[token]
    
    # Check expiration
    if datetime.utcnow() > token_data['expires_at']:
        del access_tokens[token]
        return jsonify({'error': 'invalid_token'}), 401
    
    # Return user data
    return jsonify({
        'user_id': token_data['user_id'],
        'scope': token_data['scope'],
        'client_id': token_data['client_id']
    })
```

## API Security Best Practices

### 1. Input Validation and Sanitization
```python
from marshmallow import Schema, fields, validate, ValidationError
from flask import Flask, request, jsonify
import re
import html

app = Flask(__name__)

# Validation schemas
class UserSchema(Schema):
    name = fields.Str(
        required=True,
        validate=[
            validate.Length(min=2, max=50),
            validate.Regexp(r'^[a-zA-Z\s]+$', error='Name can only contain letters and spaces')
        ]
    )
    email = fields.Email(required=True)
    age = fields.Int(
        required=True,
        validate=validate.Range(min=1, max=120)
    )
    password = fields.Str(
        required=True,
        validate=[
            validate.Length(min=8),
            validate.Regexp(
                r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]',
                error='Password must contain uppercase, lowercase, digit, and special character'
            )
        ]
    )

def sanitize_input(data):
    """Sanitize input data."""
    if isinstance(data, str):
        # Remove/escape HTML
        data = html.escape(data.strip())
        # Remove potential SQL injection characters (basic)
        data = re.sub(r'[\'";\\]', '', data)
    elif isinstance(data, dict):
        return {k: sanitize_input(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_input(item) for item in data]
    
    return data

def validate_and_sanitize(schema_class):
    """Decorator to validate and sanitize request data."""
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            schema = schema_class()
            
            try:
                # Validate data
                data = schema.load(request.json)
                # Sanitize data
                data = sanitize_input(data)
                # Add validated data to request
                request.validated_data = data
                return f(*args, **kwargs)
            except ValidationError as err:
                return jsonify({
                    'error': 'Validation failed',
                    'messages': err.messages
                }), 400
        
        return decorated
    return decorator

@app.route('/api/users', methods=['POST'])
@validate_and_sanitize(UserSchema)
def create_user():
    """Create user with validation."""
    data = request.validated_data
    # Process validated and sanitized data
    return jsonify({'message': 'User created', 'data': data})

# SQL Injection Prevention
from sqlalchemy import text

def safe_query_example(db_session, user_id):
    """Example of safe database query."""
    # BAD: Vulnerable to SQL injection
    # query = f"SELECT * FROM users WHERE id = {user_id}"
    
    # GOOD: Using parameterized queries
    query = text("SELECT * FROM users WHERE id = :user_id")
    result = db_session.execute(query, {'user_id': user_id})
    return result.fetchall()

# File Upload Security
import os
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = '/secure/uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Secure file upload endpoint."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check file size
    file.seek(0, os.SEEK_END)
    file_length = file.tell()
    file.seek(0)
    
    if file_length > MAX_FILE_SIZE:
        return jsonify({'error': 'File too large'}), 400
    
    if file and allowed_file(file.filename):
        # Secure filename
        filename = secure_filename(file.filename)
        
        # Add timestamp to prevent conflicts
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        filename = timestamp + filename
        
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        return jsonify({
            'message': 'File uploaded successfully',
            'filename': filename
        })
    
    return jsonify({'error': 'File type not allowed'}), 400
```

### 2. Rate Limiting
```python
from flask import Flask, request, jsonify
from functools import wraps
import time
from collections import defaultdict, deque

app = Flask(__name__)

# In-memory rate limiting (use Redis in production)
rate_limit_storage = defaultdict(deque)

class RateLimiter:
    def __init__(self, max_requests=100, window=3600):
        self.max_requests = max_requests
        self.window = window
    
    def is_allowed(self, key):
        """Check if request is allowed for given key."""
        now = time.time()
        requests = rate_limit_storage[key]
        
        # Remove old requests outside the window
        while requests and requests[0] <= now - self.window:
            requests.popleft()
        
        # Check if limit exceeded
        if len(requests) >= self.max_requests:
            return False
        
        # Add current request
        requests.append(now)
        return True
    
    def get_reset_time(self, key):
        """Get time when rate limit resets."""
        requests = rate_limit_storage[key]
        if requests:
            return int(requests[0] + self.window)
        return int(time.time() + self.window)

def rate_limit(max_requests=100, window=3600, key_func=None):
    """Rate limiting decorator."""
    limiter = RateLimiter(max_requests, window)
    
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            # Determine rate limit key
            if key_func:
                key = key_func()
            else:
                key = request.remote_addr
            
            if not limiter.is_allowed(key):
                reset_time = limiter.get_reset_time(key)
                return jsonify({
                    'error': 'Rate limit exceeded',
                    'message': f'Max {max_requests} requests per {window} seconds',
                    'reset_time': reset_time
                }), 429
            
            return f(*args, **kwargs)
        
        return decorated
    return decorator

# Different rate limits for different endpoints
@app.route('/api/login', methods=['POST'])
@rate_limit(max_requests=5, window=300)  # 5 requests per 5 minutes
def login():
    """Login with strict rate limiting."""
    return jsonify({'message': 'Login endpoint'})

@app.route('/api/data')
@rate_limit(max_requests=1000, window=3600)  # 1000 requests per hour
def get_data():
    """Data endpoint with higher rate limit."""
    return jsonify({'data': 'Some data'})

# User-specific rate limiting
def get_user_key():
    """Get rate limit key for authenticated user."""
    auth_header = request.headers.get('Authorization')
    if auth_header:
        # Extract user ID from token (simplified)
        return f"user_{auth_header[-10:]}"
    return request.remote_addr

@app.route('/api/user-data')
@rate_limit(max_requests=100, window=3600, key_func=get_user_key)
def get_user_data():
    """User-specific rate limiting."""
    return jsonify({'data': 'User data'})

# Redis-based rate limiting (production-ready)
import redis

redis_client = redis.Redis(host='localhost', port=6379, db=0)

class RedisRateLimiter:
    def __init__(self, redis_client, max_requests=100, window=3600):
        self.redis = redis_client
        self.max_requests = max_requests
        self.window = window
    
    def is_allowed(self, key):
        """Check if request is allowed using Redis."""
        pipe = self.redis.pipeline()
        now = time.time()
        window_start = now - self.window
        
        # Remove old entries
        pipe.zremrangebyscore(key, 0, window_start)
        
        # Count current requests
        pipe.zcard(key)
        
        # Add current request
        pipe.zadd(key, {str(now): now})
        
        # Set expiry
        pipe.expire(key, self.window)
        
        results = pipe.execute()
        request_count = results[1]
        
        return request_count < self.max_requests

redis_limiter = RedisRateLimiter(redis_client)

def redis_rate_limit(max_requests=100, window=3600):
    """Redis-based rate limiting decorator."""
    limiter = RedisRateLimiter(redis_client, max_requests, window)
    
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            key = f"rate_limit:{request.remote_addr}"
            
            if not limiter.is_allowed(key):
                return jsonify({
                    'error': 'Rate limit exceeded'
                }), 429
            
            return f(*args, **kwargs)
        
        return decorated
    return decorator
```

### 3. CORS and Security Headers
```python
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)

# Configure CORS
CORS(app, 
     origins=['http://localhost:3000', 'https://yourdomain.com'],
     methods=['GET', 'POST', 'PUT', 'DELETE'],
     allow_headers=['Content-Type', 'Authorization'],
     supports_credentials=True)

@app.after_request
def add_security_headers(response):
    """Add security headers to all responses."""
    # Prevent XSS attacks
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    
    # HTTPS enforcement (in production)
    if app.config.get('ENV') == 'production':
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    
    # Content Security Policy
    response.headers['Content-Security-Policy'] = "default-src 'self'"
    
    # Referrer Policy
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    
    return response

# Environment-specific configuration
class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY')
    
class ProductionConfig(Config):
    DEBUG = False
    TESTING = False
    
class DevelopmentConfig(Config):
    DEBUG = True
    TESTING = False

class TestingConfig(Config):
    DEBUG = False
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'

# Load configuration based on environment
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

app.config.from_object(config[os.environ.get('FLASK_ENV', 'default')])
```

This completes the authentication and security section. The documentation covers various authentication methods, JWT implementation, OAuth 2.0, and comprehensive security best practices including input validation, rate limiting, and security headers.
