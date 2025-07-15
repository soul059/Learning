# Python API Development - REST APIs

## Table of Contents
1. [Introduction to API Development](#introduction-to-api-development)
2. [Flask Framework](#flask-framework)
3. [FastAPI Framework](#fastapi-framework)
4. [Django REST Framework](#django-rest-framework)
5. [Database Integration](#database-integration)
6. [Authentication and Authorization](#authentication-and-authorization)
7. [API Documentation](#api-documentation)
8. [Testing APIs](#testing-apis)

## Introduction to API Development

### 1. REST API Principles
```python
"""
REST (Representational State Transfer) Principles:

1. Client-Server Architecture
2. Stateless Communication
3. Cacheable Responses
4. Uniform Interface
5. Layered System
6. Code on Demand (optional)

HTTP Methods:
- GET: Retrieve data
- POST: Create new resource
- PUT: Update entire resource
- PATCH: Partial update
- DELETE: Remove resource

HTTP Status Codes:
- 200: OK
- 201: Created
- 400: Bad Request
- 401: Unauthorized
- 403: Forbidden
- 404: Not Found
- 500: Internal Server Error
"""

# Example API Response Structure
api_response = {
    "status": "success",
    "message": "Data retrieved successfully",
    "data": {
        "id": 1,
        "name": "John Doe",
        "email": "john@example.com"
    },
    "timestamp": "2024-01-15T10:30:00Z"
}

error_response = {
    "status": "error",
    "message": "Validation failed",
    "errors": [
        {"field": "email", "message": "Invalid email format"},
        {"field": "age", "message": "Age must be a positive integer"}
    ],
    "timestamp": "2024-01-15T10:30:00Z"
}
```

### 2. API Design Best Practices
```python
"""
URL Design Patterns:

Good Examples:
- GET /api/v1/users                 # Get all users
- GET /api/v1/users/123             # Get user by ID
- POST /api/v1/users                # Create new user
- PUT /api/v1/users/123             # Update user
- DELETE /api/v1/users/123          # Delete user
- GET /api/v1/users/123/posts       # Get user's posts

Bad Examples:
- GET /api/getUsers
- POST /api/createUser
- GET /api/user_posts?user_id=123
"""

# Standard response helper
def create_response(status="success", message="", data=None, errors=None, status_code=200):
    """Create standardized API response."""
    response = {
        "status": status,
        "message": message,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if data is not None:
        response["data"] = data
    
    if errors is not None:
        response["errors"] = errors
    
    return response, status_code

# Pagination helper
def paginate_response(items, page, per_page, total):
    """Create paginated response."""
    return {
        "items": items,
        "pagination": {
            "page": page,
            "per_page": per_page,
            "total": total,
            "pages": (total + per_page - 1) // per_page
        }
    }
```

## Flask Framework

### 1. Basic Flask API
```python
from flask import Flask, request, jsonify
from datetime import datetime
import uuid

app = Flask(__name__)

# In-memory storage (use database in production)
users = []

# Helper function for responses
def create_response(status="success", message="", data=None, errors=None):
    response = {
        "status": status,
        "message": message,
        "timestamp": datetime.utcnow().isoformat()
    }
    if data is not None:
        response["data"] = data
    if errors is not None:
        response["errors"] = errors
    return response

@app.route('/api/v1/users', methods=['GET'])
def get_users():
    """Get all users with optional pagination."""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    
    start = (page - 1) * per_page
    end = start + per_page
    
    paginated_users = users[start:end]
    
    response = create_response(
        message="Users retrieved successfully",
        data={
            "users": paginated_users,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total": len(users),
                "pages": (len(users) + per_page - 1) // per_page
            }
        }
    )
    return jsonify(response)

@app.route('/api/v1/users/<user_id>', methods=['GET'])
def get_user(user_id):
    """Get user by ID."""
    user = next((u for u in users if u['id'] == user_id), None)
    
    if not user:
        response = create_response(
            status="error",
            message="User not found"
        )
        return jsonify(response), 404
    
    response = create_response(
        message="User retrieved successfully",
        data=user
    )
    return jsonify(response)

@app.route('/api/v1/users', methods=['POST'])
def create_user():
    """Create a new user."""
    data = request.get_json()
    
    # Validation
    required_fields = ['name', 'email']
    errors = []
    
    for field in required_fields:
        if field not in data or not data[field]:
            errors.append({"field": field, "message": f"{field} is required"})
    
    # Check if email already exists
    if data.get('email') and any(u['email'] == data['email'] for u in users):
        errors.append({"field": "email", "message": "Email already exists"})
    
    if errors:
        response = create_response(
            status="error",
            message="Validation failed",
            errors=errors
        )
        return jsonify(response), 400
    
    # Create user
    user = {
        "id": str(uuid.uuid4()),
        "name": data['name'],
        "email": data['email'],
        "created_at": datetime.utcnow().isoformat()
    }
    
    users.append(user)
    
    response = create_response(
        message="User created successfully",
        data=user
    )
    return jsonify(response), 201

@app.route('/api/v1/users/<user_id>', methods=['PUT'])
def update_user(user_id):
    """Update user by ID."""
    user = next((u for u in users if u['id'] == user_id), None)
    
    if not user:
        response = create_response(
            status="error",
            message="User not found"
        )
        return jsonify(response), 404
    
    data = request.get_json()
    
    # Update user
    if 'name' in data:
        user['name'] = data['name']
    if 'email' in data:
        user['email'] = data['email']
    
    user['updated_at'] = datetime.utcnow().isoformat()
    
    response = create_response(
        message="User updated successfully",
        data=user
    )
    return jsonify(response)

@app.route('/api/v1/users/<user_id>', methods=['DELETE'])
def delete_user(user_id):
    """Delete user by ID."""
    global users
    user = next((u for u in users if u['id'] == user_id), None)
    
    if not user:
        response = create_response(
            status="error",
            message="User not found"
        )
        return jsonify(response), 404
    
    users = [u for u in users if u['id'] != user_id]
    
    response = create_response(message="User deleted successfully")
    return jsonify(response)

# Error handlers
@app.errorhandler(404)
def not_found(error):
    response = create_response(
        status="error",
        message="Endpoint not found"
    )
    return jsonify(response), 404

@app.errorhandler(500)
def internal_error(error):
    response = create_response(
        status="error",
        message="Internal server error"
    )
    return jsonify(response), 500

if __name__ == '__main__':
    app.run(debug=True)
```

### 2. Flask with Blueprints and Middleware
```python
from flask import Flask, request, jsonify, g
from functools import wraps
import time
import logging

# Create Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Middleware for request logging
@app.before_request
def log_request_info():
    g.start_time = time.time()
    logger.info(f"Request: {request.method} {request.url}")

@app.after_request
def log_response_info(response):
    duration = time.time() - g.start_time
    logger.info(f"Response: {response.status_code} - {duration:.3f}s")
    return response

# Authentication decorator
def require_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({
                "status": "error",
                "message": "Missing or invalid authorization header"
            }), 401
        
        # In real application, validate the token
        token = auth_header.split(' ')[1]
        if token != 'valid-token':
            return jsonify({
                "status": "error",
                "message": "Invalid token"
            }), 401
        
        return f(*args, **kwargs)
    return decorated_function

# Rate limiting decorator
def rate_limit(max_requests=10, window=60):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Simple in-memory rate limiting (use Redis in production)
            client_ip = request.remote_addr
            # Implement rate limiting logic here
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# Blueprint for user routes
from flask import Blueprint
users_bp = Blueprint('users', __name__, url_prefix='/api/v1/users')

@users_bp.route('', methods=['GET'])
@rate_limit(max_requests=100)
def get_users():
    # Implementation here
    pass

@users_bp.route('', methods=['POST'])
@require_auth
@rate_limit(max_requests=5)
def create_user():
    # Implementation here
    pass

# Register blueprint
app.register_blueprint(users_bp)

# CORS middleware
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response
```

### 3. Flask with SQLAlchemy
```python
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from datetime import datetime
import uuid

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Models
class User(db.Model):
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    posts = db.relationship('Post', backref='author', lazy=True, cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'email': self.email,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

class Post(db.Model):
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    user_id = db.Column(db.String(36), db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'content': self.content,
            'user_id': self.user_id,
            'author': self.author.name if self.author else None,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

# API Routes
@app.route('/api/v1/users', methods=['GET'])
def get_users():
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    
    users = User.query.paginate(
        page=page, 
        per_page=per_page, 
        error_out=False
    )
    
    return jsonify({
        'status': 'success',
        'data': {
            'users': [user.to_dict() for user in users.items],
            'pagination': {
                'page': users.page,
                'pages': users.pages,
                'per_page': users.per_page,
                'total': users.total
            }
        }
    })

@app.route('/api/v1/users', methods=['POST'])
def create_user():
    data = request.get_json()
    
    # Validation
    if not data.get('name') or not data.get('email'):
        return jsonify({
            'status': 'error',
            'message': 'Name and email are required'
        }), 400
    
    # Check if email exists
    if User.query.filter_by(email=data['email']).first():
        return jsonify({
            'status': 'error',
            'message': 'Email already exists'
        }), 400
    
    # Create user
    user = User(name=data['name'], email=data['email'])
    db.session.add(user)
    db.session.commit()
    
    return jsonify({
        'status': 'success',
        'message': 'User created successfully',
        'data': user.to_dict()
    }), 201

@app.route('/api/v1/users/<user_id>/posts', methods=['GET'])
def get_user_posts(user_id):
    user = User.query.get_or_404(user_id)
    posts = Post.query.filter_by(user_id=user_id).all()
    
    return jsonify({
        'status': 'success',
        'data': {
            'user': user.to_dict(),
            'posts': [post.to_dict() for post in posts]
        }
    })

# Initialize database
@app.before_first_request
def create_tables():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True)
```

## FastAPI Framework

### 1. Basic FastAPI Application
```python
from fastapi import FastAPI, HTTPException, Depends, Query
from pydantic import BaseModel, EmailStr, validator
from typing import List, Optional
from datetime import datetime
import uuid

app = FastAPI(
    title="User Management API",
    description="A simple API for managing users",
    version="1.0.0"
)

# Pydantic models for request/response
class UserCreate(BaseModel):
    name: str
    email: EmailStr
    age: Optional[int] = None
    
    @validator('name')
    def name_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Name cannot be empty')
        return v
    
    @validator('age')
    def age_must_be_positive(cls, v):
        if v is not None and v <= 0:
            raise ValueError('Age must be positive')
        return v

class UserResponse(BaseModel):
    id: str
    name: str
    email: str
    age: Optional[int]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True

class UserUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    age: Optional[int] = None

# In-memory storage
users_db = []

# Helper functions
def get_user_by_id(user_id: str):
    user = next((u for u in users_db if u["id"] == user_id), None)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

def get_user_by_email(email: str):
    return next((u for u in users_db if u["email"] == email), None)

# API Routes
@app.get("/api/v1/users", response_model=List[UserResponse])
async def get_users(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(10, ge=1, le=100, description="Number of records to return")
):
    """Get all users with pagination."""
    return users_db[skip:skip + limit]

@app.get("/api/v1/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: str):
    """Get a user by ID."""
    return get_user_by_id(user_id)

@app.post("/api/v1/users", response_model=UserResponse, status_code=201)
async def create_user(user: UserCreate):
    """Create a new user."""
    # Check if email already exists
    if get_user_by_email(user.email):
        raise HTTPException(status_code=400, detail="Email already exists")
    
    # Create new user
    new_user = {
        "id": str(uuid.uuid4()),
        "name": user.name,
        "email": user.email,
        "age": user.age,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    
    users_db.append(new_user)
    return new_user

@app.put("/api/v1/users/{user_id}", response_model=UserResponse)
async def update_user(user_id: str, user_update: UserUpdate):
    """Update a user by ID."""
    user = get_user_by_id(user_id)
    
    # Check email uniqueness if email is being updated
    if user_update.email and user_update.email != user["email"]:
        if get_user_by_email(user_update.email):
            raise HTTPException(status_code=400, detail="Email already exists")
    
    # Update fields
    update_data = user_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        user[field] = value
    
    user["updated_at"] = datetime.utcnow()
    return user

@app.delete("/api/v1/users/{user_id}")
async def delete_user(user_id: str):
    """Delete a user by ID."""
    user = get_user_by_id(user_id)
    users_db.remove(user)
    return {"message": "User deleted successfully"}

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 2. FastAPI with Dependencies and Middleware
```python
from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import time
import logging

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "*.example.com"]
)

# Security
security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token."""
    token = credentials.credentials
    # In real application, verify JWT token
    if token != "valid-token":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    return {"user_id": "123", "username": "testuser"}

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url}")
    
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    logger.info(f"Response: {response.status_code} - {process_time:.3f}s")
    
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Dependency for pagination
def pagination_params(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(10, ge=1, le=100, description="Number of records to return")
):
    return {"skip": skip, "limit": limit}

# Protected route
@app.get("/api/v1/protected")
async def protected_route(current_user: dict = Depends(verify_token)):
    """Protected route that requires authentication."""
    return {"message": f"Hello {current_user['username']}", "user": current_user}

# Route with pagination dependency
@app.get("/api/v1/items")
async def get_items(pagination: dict = Depends(pagination_params)):
    """Get items with pagination."""
    # Simulate database query
    items = [{"id": i, "name": f"Item {i}"} for i in range(100)]
    start = pagination["skip"]
    end = start + pagination["limit"]
    
    return {
        "items": items[start:end],
        "pagination": {
            "skip": pagination["skip"],
            "limit": pagination["limit"],
            "total": len(items)
        }
    }
```

### 3. FastAPI with SQLAlchemy and Alembic
```python
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy import create_engine, Column, String, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from datetime import datetime
import uuid

# Database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database models
class User(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    posts = relationship("Post", back_populates="author", cascade="all, delete-orphan")

class Post(Base):
    __tablename__ = "posts"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    author = relationship("User", back_populates="posts")

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI()

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pydantic models
from pydantic import BaseModel
from typing import List, Optional

class PostCreate(BaseModel):
    title: str
    content: str

class PostResponse(BaseModel):
    id: str
    title: str
    content: str
    user_id: str
    author_name: str
    created_at: datetime
    
    class Config:
        orm_mode = True

class UserCreate(BaseModel):
    name: str
    email: str

class UserResponse(BaseModel):
    id: str
    name: str
    email: str
    created_at: datetime
    posts: List[PostResponse] = []
    
    class Config:
        orm_mode = True

# API Routes
@app.post("/api/v1/users", response_model=UserResponse)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    # Check if email exists
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create user
    db_user = User(**user.dict())
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.get("/api/v1/users/{user_id}", response_model=UserResponse)
def get_user(user_id: str, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user

@app.post("/api/v1/users/{user_id}/posts", response_model=PostResponse)
def create_post(user_id: str, post: PostCreate, db: Session = Depends(get_db)):
    # Check if user exists
    db_user = db.query(User).filter(User.id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Create post
    db_post = Post(**post.dict(), user_id=user_id)
    db.add(db_post)
    db.commit()
    db.refresh(db_post)
    
    # Add author name for response
    db_post.author_name = db_user.name
    return db_post

@app.get("/api/v1/posts", response_model=List[PostResponse])
def get_posts(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    posts = db.query(Post).offset(skip).limit(limit).all()
    
    # Add author name to each post
    for post in posts:
        post.author_name = post.author.name
    
    return posts
```

## Django REST Framework

### 1. Basic Django REST API
```python
# settings.py
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'corsheaders',
    'myapp',
]

REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.TokenAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticated',
    ],
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 10,
}

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.common.CommonMiddleware',
    # ... other middleware
]

CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

# models.py
from django.db import models
from django.contrib.auth.models import User
import uuid

class Post(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=200)
    content = models.TextField()
    author = models.ForeignKey(User, on_delete=models.CASCADE, related_name='posts')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return self.title

# serializers.py
from rest_framework import serializers
from django.contrib.auth.models import User
from .models import Post

class UserSerializer(serializers.ModelSerializer):
    posts_count = serializers.SerializerMethodField()
    
    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'first_name', 'last_name', 'posts_count']
        extra_kwargs = {'password': {'write_only': True}}
    
    def get_posts_count(self, obj):
        return obj.posts.count()
    
    def create(self, validated_data):
        password = validated_data.pop('password')
        user = User.objects.create_user(**validated_data)
        user.set_password(password)
        user.save()
        return user

class PostSerializer(serializers.ModelSerializer):
    author_name = serializers.CharField(source='author.username', read_only=True)
    
    class Meta:
        model = Post
        fields = ['id', 'title', 'content', 'author', 'author_name', 'created_at', 'updated_at']
        read_only_fields = ['author']
    
    def create(self, validated_data):
        validated_data['author'] = self.context['request'].user
        return super().create(validated_data)

# views.py
from rest_framework import generics, permissions, status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from django.contrib.auth.models import User
from .models import Post
from .serializers import UserSerializer, PostSerializer

class UserListCreateView(generics.ListCreateAPIView):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = [permissions.AllowAny]  # Allow registration

class UserDetailView(generics.RetrieveUpdateDestroyAPIView):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = [permissions.IsAuthenticated]

class PostListCreateView(generics.ListCreateAPIView):
    queryset = Post.objects.all()
    serializer_class = PostSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        queryset = super().get_queryset()
        author_id = self.request.query_params.get('author')
        if author_id:
            queryset = queryset.filter(author_id=author_id)
        return queryset

class PostDetailView(generics.RetrieveUpdateDestroyAPIView):
    queryset = Post.objects.all()
    serializer_class = PostSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def update(self, request, *args, **kwargs):
        post = self.get_object()
        if post.author != request.user:
            return Response(
                {"error": "You can only edit your own posts"}, 
                status=status.HTTP_403_FORBIDDEN
            )
        return super().update(request, *args, **kwargs)
    
    def destroy(self, request, *args, **kwargs):
        post = self.get_object()
        if post.author != request.user:
            return Response(
                {"error": "You can only delete your own posts"}, 
                status=status.HTTP_403_FORBIDDEN
            )
        return super().destroy(request, *args, **kwargs)

# Custom API views
@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated])
def user_profile(request):
    """Get current user's profile with posts."""
    user = request.user
    posts = user.posts.all()[:5]  # Latest 5 posts
    
    return Response({
        'user': UserSerializer(user).data,
        'recent_posts': PostSerializer(posts, many=True).data
    })

# urls.py
from django.urls import path, include
from . import views

urlpatterns = [
    path('api/v1/users/', views.UserListCreateView.as_view(), name='user-list'),
    path('api/v1/users/<int:pk>/', views.UserDetailView.as_view(), name='user-detail'),
    path('api/v1/posts/', views.PostListCreateView.as_view(), name='post-list'),
    path('api/v1/posts/<uuid:pk>/', views.PostDetailView.as_view(), name='post-detail'),
    path('api/v1/profile/', views.user_profile, name='user-profile'),
    path('api-auth/', include('rest_framework.urls')),
]
```

### 2. Django REST with Custom Permissions and Filters
```python
# permissions.py
from rest_framework import permissions

class IsOwnerOrReadOnly(permissions.BasePermission):
    """Custom permission to only allow owners to edit their objects."""
    
    def has_object_permission(self, request, view, obj):
        # Read permissions for any request
        if request.method in permissions.SAFE_METHODS:
            return True
        
        # Write permissions only to the owner
        return obj.author == request.user

class IsAdminOrReadOnly(permissions.BasePermission):
    """Custom permission to only allow admins to edit."""
    
    def has_permission(self, request, view):
        if request.method in permissions.SAFE_METHODS:
            return True
        return request.user.is_staff

# filters.py
import django_filters
from .models import Post

class PostFilter(django_filters.FilterSet):
    title = django_filters.CharFilter(lookup_expr='icontains')
    content = django_filters.CharFilter(lookup_expr='icontains')
    created_after = django_filters.DateTimeFilter(field_name='created_at', lookup_expr='gte')
    created_before = django_filters.DateTimeFilter(field_name='created_at', lookup_expr='lte')
    author_username = django_filters.CharFilter(field_name='author__username', lookup_expr='icontains')
    
    class Meta:
        model = Post
        fields = ['title', 'content', 'author', 'created_after', 'created_before']

# Advanced views.py
from rest_framework import generics, filters
from rest_framework.decorators import action
from rest_framework.response import Response
from django_filters.rest_framework import DjangoFilterBackend
from .permissions import IsOwnerOrReadOnly
from .filters import PostFilter

class PostViewSet(viewsets.ModelViewSet):
    queryset = Post.objects.all()
    serializer_class = PostSerializer
    permission_classes = [permissions.IsAuthenticated, IsOwnerOrReadOnly]
    filter_backends = [DjangoFilterBackend, filters.SearchBackend, filters.OrderingFilter]
    filterset_class = PostFilter
    search_fields = ['title', 'content']
    ordering_fields = ['created_at', 'updated_at', 'title']
    ordering = ['-created_at']
    
    def perform_create(self, serializer):
        serializer.save(author=self.request.user)
    
    @action(detail=False, methods=['get'])
    def my_posts(self, request):
        """Get current user's posts."""
        posts = self.queryset.filter(author=request.user)
        page = self.paginate_queryset(posts)
        if page is not None:
            serializer = self.get_serializer(page, many=True)
            return self.get_paginated_response(serializer.data)
        
        serializer = self.get_serializer(posts, many=True)
        return Response(serializer.data)
    
    @action(detail=True, methods=['post'])
    def like(self, request, pk=None):
        """Like a post."""
        post = self.get_object()
        # Implement like functionality
        return Response({'status': 'liked'})

# Pagination
from rest_framework.pagination import PageNumberPagination

class CustomPagination(PageNumberPagination):
    page_size = 10
    page_size_query_param = 'page_size'
    max_page_size = 100
    
    def get_paginated_response(self, data):
        return Response({
            'links': {
                'next': self.get_next_link(),
                'previous': self.get_previous_link()
            },
            'count': self.page.paginator.count,
            'total_pages': self.page.paginator.num_pages,
            'current_page': self.page.number,
            'results': data
        })
```

## Database Integration

### 1. SQLAlchemy ORM
```python
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

# Database setup
DATABASE_URL = "postgresql://user:password@localhost/dbname"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Models
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True)
    email = Column(String(100), unique=True, index=True)
    hashed_password = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    posts = relationship("Post", back_populates="author")

class Post(Base):
    __tablename__ = "posts"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(200), index=True)
    content = Column(Text)
    author_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    author = relationship("User", back_populates="posts")

# Database operations
class UserCRUD:
    def __init__(self, db: Session):
        self.db = db
    
    def create_user(self, user_data: dict):
        db_user = User(**user_data)
        self.db.add(db_user)
        self.db.commit()
        self.db.refresh(db_user)
        return db_user
    
    def get_user(self, user_id: int):
        return self.db.query(User).filter(User.id == user_id).first()
    
    def get_user_by_email(self, email: str):
        return self.db.query(User).filter(User.email == email).first()
    
    def get_users(self, skip: int = 0, limit: int = 100):
        return self.db.query(User).offset(skip).limit(limit).all()
    
    def update_user(self, user_id: int, user_data: dict):
        db_user = self.get_user(user_id)
        if db_user:
            for key, value in user_data.items():
                setattr(db_user, key, value)
            self.db.commit()
            self.db.refresh(db_user)
        return db_user
    
    def delete_user(self, user_id: int):
        db_user = self.get_user(user_id)
        if db_user:
            self.db.delete(db_user)
            self.db.commit()
        return db_user
```

### 2. Database Migrations with Alembic
```python
# alembic/env.py
from alembic import context
from sqlalchemy import engine_from_config, pool
from logging.config import fileConfig
from myapp.models import Base

# Alembic Config object
config = context.config

# Configure logging
fileConfig(config.config_file_name)

# Set target metadata
target_metadata = Base.metadata

def run_migrations_offline():
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()

# Migration commands:
# alembic init alembic
# alembic revision --autogenerate -m "Create users table"
# alembic upgrade head
# alembic downgrade -1
```

### 3. Database Connection Pool and Async Operations
```python
import asyncio
import asyncpg
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Async SQLAlchemy setup
DATABASE_URL = "postgresql+asyncpg://user:password@localhost/dbname"
engine = create_async_engine(DATABASE_URL, echo=True)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# Async database operations
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

async def create_user_async(db: AsyncSession, user_data: dict):
    db_user = User(**user_data)
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)
    return db_user

async def get_users_async(db: AsyncSession, skip: int = 0, limit: int = 100):
    result = await db.execute(
        select(User).offset(skip).limit(limit)
    )
    return result.scalars().all()

# Connection pooling with asyncpg
class DatabasePool:
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.pool = None
    
    async def create_pool(self):
        self.pool = await asyncpg.create_pool(
            self.database_url,
            min_size=10,
            max_size=20,
            command_timeout=60
        )
    
    async def close_pool(self):
        await self.pool.close()
    
    async def execute_query(self, query: str, *args):
        async with self.pool.acquire() as connection:
            return await connection.fetch(query, *args)
    
    async def execute_transaction(self, queries: list):
        async with self.pool.acquire() as connection:
            async with connection.transaction():
                results = []
                for query, args in queries:
                    result = await connection.fetch(query, *args)
                    results.append(result)
                return results

# Usage
db_pool = DatabasePool("postgresql://user:password@localhost/dbname")

async def main():
    await db_pool.create_pool()
    
    # Execute queries
    users = await db_pool.execute_query("SELECT * FROM users WHERE active = $1", True)
    
    await db_pool.close_pool()

asyncio.run(main())
```

## API Versioning and Management

### 1. API Versioning Strategies

```python
# URL Path Versioning
from flask import Flask, jsonify, request
from functools import wraps

app = Flask(__name__)

# Version 1 API
@app.route('/api/v1/users', methods=['GET'])
def get_users_v1():
    """Version 1 of users endpoint - returns basic info"""
    return jsonify({
        "version": "1.0",
        "users": [
            {"id": 1, "name": "John Doe"},
            {"id": 2, "name": "Jane Smith"}
        ]
    })

# Version 2 API
@app.route('/api/v2/users', methods=['GET'])
def get_users_v2():
    """Version 2 of users endpoint - returns detailed info"""
    return jsonify({
        "version": "2.0",
        "users": [
            {
                "id": 1, 
                "name": "John Doe", 
                "email": "john@example.com",
                "created_at": "2023-01-01"
            },
            {
                "id": 2, 
                "name": "Jane Smith", 
                "email": "jane@example.com",
                "created_at": "2023-01-02"
            }
        ],
        "pagination": {
            "page": 1,
            "total_pages": 1,
            "total_items": 2
        }
    })

# Header-based versioning
def version_required(version):
    """Decorator for header-based API versioning"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            api_version = request.headers.get('API-Version', '1.0')
            if api_version != version:
                return jsonify({
                    "error": "Unsupported API version",
                    "supported_versions": ["1.0", "2.0"]
                }), 400
            return f(*args, **kwargs)
        return decorated_function
    return decorator

@app.route('/api/users', methods=['GET'])
@version_required('2.0')
def get_users_header_versioned():
    """Header-versioned endpoint"""
    return jsonify({"message": "This is API version 2.0"})

# Content negotiation versioning
@app.route('/api/users', methods=['GET'])
def get_users_content_negotiated():
    """Content negotiation based versioning"""
    accept_header = request.headers.get('Accept', '')
    
    if 'application/vnd.api.v2+json' in accept_header:
        return jsonify({
            "version": "2.0",
            "data": "Version 2 response"
        })
    else:
        return jsonify({
            "version": "1.0", 
            "data": "Version 1 response"
        })
```

### 2. Rate Limiting

```python
import time
import redis
from functools import wraps
from flask import request, jsonify

# Redis-based rate limiter
class RedisRateLimiter:
    def __init__(self, redis_client):
        self.redis = redis_client
    
    def is_allowed(self, key, limit, window):
        """Check if request is within rate limit"""
        current_time = int(time.time())
        window_start = current_time - window
        
        # Remove old entries
        self.redis.zremrangebyscore(key, 0, window_start)
        
        # Count current requests
        current_requests = self.redis.zcard(key)
        
        if current_requests < limit:
            # Add current request
            self.redis.zadd(key, {current_time: current_time})
            self.redis.expire(key, window)
            return True
        
        return False

# In-memory rate limiter (for testing)
class InMemoryRateLimiter:
    def __init__(self):
        self.requests = {}
    
    def is_allowed(self, key, limit, window):
        current_time = time.time()
        
        if key not in self.requests:
            self.requests[key] = []
        
        # Remove old requests
        self.requests[key] = [
            req_time for req_time in self.requests[key] 
            if current_time - req_time < window
        ]
        
        if len(self.requests[key]) < limit:
            self.requests[key].append(current_time)
            return True
        
        return False

# Rate limiting decorator
rate_limiter = InMemoryRateLimiter()

def rate_limit(limit=100, window=3600):  # 100 requests per hour
    """Rate limiting decorator"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Use IP address as identifier
            client_ip = request.remote_addr
            key = f"rate_limit:{client_ip}:{f.__name__}"
            
            if not rate_limiter.is_allowed(key, limit, window):
                return jsonify({
                    "error": "Rate limit exceeded",
                    "message": f"Maximum {limit} requests per {window} seconds"
                }), 429
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

@app.route('/api/data')
@rate_limit(limit=10, window=60)  # 10 requests per minute
def get_data():
    return jsonify({"data": "Some data"})
```

### 3. API Middleware and Interceptors

```python
from datetime import datetime
import uuid
import logging

# Request/Response logging middleware
class APIMiddleware:
    def __init__(self, app):
        self.app = app
        self.logger = logging.getLogger('api_middleware')
    
    def __call__(self, environ, start_response):
        """WSGI middleware for request/response logging"""
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Log request
        self.logger.info(f"[{request_id}] {environ['REQUEST_METHOD']} {environ['PATH_INFO']}")
        
        def logging_start_response(status, headers):
            # Add request ID to headers
            headers.append(('X-Request-ID', request_id))
            
            # Log response
            duration = time.time() - start_time
            self.logger.info(
                f"[{request_id}] Response: {status} "
                f"Duration: {duration:.3f}s"
            )
            
            return start_response(status, headers)
        
        return self.app(environ, logging_start_response)

# Apply middleware
app.wsgi_app = APIMiddleware(app.wsgi_app)

# FastAPI middleware example
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import traceback

fastapi_app = FastAPI()

@fastapi_app.middleware("http")
async def error_handling_middleware(request: Request, call_next):
    """Global error handling middleware"""
    try:
        response = await call_next(request)
        return response
    except HTTPException:
        raise
    except Exception as e:
        # Log the error
        logging.error(f"Unhandled exception: {str(e)}")
        logging.error(traceback.format_exc())
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "message": "An unexpected error occurred"
            }
        )

@fastapi_app.middleware("http")
async def request_timing_middleware(request: Request, call_next):
    """Request timing middleware"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response
```

### 4. Advanced Response Patterns

```python
# Standardized API response wrapper
class APIResponse:
    """Standardized API response wrapper"""
    
    @staticmethod
    def success(data=None, message="Success", status_code=200):
        return {
            "success": True,
            "message": message,
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
            "status_code": status_code
        }, status_code
    
    @staticmethod
    def error(message="Error", errors=None, status_code=400):
        return {
            "success": False,
            "message": message,
            "errors": errors or [],
            "timestamp": datetime.utcnow().isoformat(),
            "status_code": status_code
        }, status_code
    
    @staticmethod
    def paginated(data, page, per_page, total, message="Success"):
        return {
            "success": True,
            "message": message,
            "data": data,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total": total,
                "pages": (total + per_page - 1) // per_page
            },
            "timestamp": datetime.utcnow().isoformat()
        }, 200

# Usage examples
@app.route('/api/users/<int:user_id>')
def get_user(user_id):
    try:
        # Simulate database lookup
        if user_id == 1:
            user_data = {"id": 1, "name": "John Doe", "email": "john@example.com"}
            return APIResponse.success(
                data=user_data, 
                message="User retrieved successfully"
            )
        else:
            return APIResponse.error(
                message="User not found",
                status_code=404
            )
    except Exception as e:
        return APIResponse.error(
            message="Internal server error",
            status_code=500
        )

@app.route('/api/users')
def list_users():
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    
    # Simulate paginated data
    all_users = [{"id": i, "name": f"User {i}"} for i in range(1, 101)]
    start = (page - 1) * per_page
    end = start + per_page
    page_users = all_users[start:end]
    
    return APIResponse.paginated(
        data=page_users,
        page=page,
        per_page=per_page,
        total=len(all_users),
        message="Users retrieved successfully"
    )

# HATEOAS (Hypermedia as the Engine of Application State)
class HATEOASResponse:
    """HATEOAS-compliant response builder"""
    
    @staticmethod
    def build_links(resource_id, resource_type, base_url="/api"):
        """Build hypermedia links for a resource"""
        return {
            "self": f"{base_url}/{resource_type}/{resource_id}",
            "edit": f"{base_url}/{resource_type}/{resource_id}",
            "delete": f"{base_url}/{resource_type}/{resource_id}",
            "collection": f"{base_url}/{resource_type}"
        }
    
    @staticmethod
    def user_response(user_data):
        """Build HATEOAS response for user resource"""
        return {
            **user_data,
            "_links": HATEOASResponse.build_links(
                user_data["id"], 
                "users"
            )
        }

@app.route('/api/hateoas/users/<int:user_id>')
def get_user_hateoas(user_id):
    user_data = {"id": user_id, "name": "John Doe", "email": "john@example.com"}
    return jsonify(HATEOASResponse.user_response(user_data))
```

### 5. API Caching Strategies

```python
import hashlib
from functools import wraps

# Simple in-memory cache
class SimpleCache:
    def __init__(self):
        self.cache = {}
    
    def get(self, key):
        return self.cache.get(key)
    
    def set(self, key, value, ttl=None):
        self.cache[key] = {
            'value': value,
            'expires': time.time() + ttl if ttl else None
        }
    
    def delete(self, key):
        self.cache.pop(key, None)
    
    def is_expired(self, key):
        if key not in self.cache:
            return True
        
        entry = self.cache[key]
        if entry['expires'] and time.time() > entry['expires']:
            self.delete(key)
            return True
        
        return False

cache = SimpleCache()

def cached_response(ttl=300):  # 5 minutes default
    """Decorator for caching API responses"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = f"{f.__name__}:{hashlib.md5(str(args + tuple(kwargs.items())).encode()).hexdigest()}"
            
            # Check cache
            if not cache.is_expired(cache_key):
                cached_data = cache.get(cache_key)
                if cached_data:
                    response = jsonify(cached_data['value'])
                    response.headers['X-Cache'] = 'HIT'
                    return response
            
            # Execute function and cache result
            result = f(*args, **kwargs)
            if hasattr(result, 'get_json'):
                cache.set(cache_key, result.get_json(), ttl)
            
            response = jsonify(result)
            response.headers['X-Cache'] = 'MISS'
            return response
        
        return decorated_function
    return decorator

@app.route('/api/cached/users')
@cached_response(ttl=600)  # Cache for 10 minutes
def get_cached_users():
    # Simulate expensive database operation
    time.sleep(1)
    return {"users": [{"id": 1, "name": "John"}, {"id": 2, "name": "Jane"}]}

# ETags for conditional requests
def generate_etag(data):
    """Generate ETag from data"""
    return hashlib.md5(str(data).encode()).hexdigest()

@app.route('/api/etag/users/<int:user_id>')
def get_user_with_etag(user_id):
    user_data = {"id": user_id, "name": "John Doe", "updated_at": "2023-01-01"}
    etag = generate_etag(user_data)
    
    # Check If-None-Match header
    if request.headers.get('If-None-Match') == etag:
        return '', 304  # Not Modified
    
    response = jsonify(user_data)
    response.headers['ETag'] = etag
    response.headers['Cache-Control'] = 'max-age=300'  # 5 minutes
    return response

print("\nAdvanced API development concepts completed!")
```
