# Testing and Documentation for Python APIs

## Table of Contents
1. [API Testing with pytest](#api-testing-with-pytest)
2. [Testing Flask APIs](#testing-flask-apis)
3. [Testing FastAPI](#testing-fastapi)
4. [API Documentation with OpenAPI/Swagger](#api-documentation-with-openapiswagger)
5. [Load Testing](#load-testing)
6. [Mocking and Test Doubles](#mocking-and-test-doubles)
7. [Integration Testing](#integration-testing)

## API Testing with pytest

### 1. Basic pytest Setup
```python
# conftest.py
import pytest
import os
import tempfile
from myapp import create_app, db
from myapp.models import User, Post

@pytest.fixture
def app():
    """Create application for testing."""
    db_fd, db_path = tempfile.mkstemp()
    
    app = create_app({
        'TESTING': True,
        'SQLALCHEMY_DATABASE_URI': f'sqlite:///{db_path}',
        'WTF_CSRF_ENABLED': False
    })
    
    with app.app_context():
        db.create_all()
        yield app
        db.drop_all()
    
    os.close(db_fd)
    os.unlink(db_path)

@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()

@pytest.fixture
def runner(app):
    """Create test CLI runner."""
    return app.test_cli_runner()

@pytest.fixture
def auth_headers():
    """Create authorization headers for testing."""
    # In real tests, you'd generate a valid token
    return {'Authorization': 'Bearer test-token'}

@pytest.fixture
def sample_user(app):
    """Create a sample user for testing."""
    with app.app_context():
        user = User(
            username='testuser',
            email='test@example.com',
            password='password123'
        )
        db.session.add(user)
        db.session.commit()
        return user

@pytest.fixture
def sample_post(app, sample_user):
    """Create a sample post for testing."""
    with app.app_context():
        post = Post(
            title='Test Post',
            content='This is a test post',
            author_id=sample_user.id
        )
        db.session.add(post)
        db.session.commit()
        return post

# test_api.py
import json
import pytest
from myapp.models import User, Post

class TestUserAPI:
    """Test user-related endpoints."""
    
    def test_get_users_empty(self, client):
        """Test getting users when none exist."""
        response = client.get('/api/v1/users')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['status'] == 'success'
        assert len(data['data']['users']) == 0
    
    def test_create_user_success(self, client):
        """Test successful user creation."""
        user_data = {
            'username': 'newuser',
            'email': 'newuser@example.com',
            'password': 'password123'
        }
        
        response = client.post(
            '/api/v1/users',
            data=json.dumps(user_data),
            content_type='application/json'
        )
        
        assert response.status_code == 201
        
        data = json.loads(response.data)
        assert data['status'] == 'success'
        assert data['data']['username'] == 'newuser'
        assert data['data']['email'] == 'newuser@example.com'
        assert 'password' not in data['data']  # Password should not be returned
    
    def test_create_user_validation_error(self, client):
        """Test user creation with validation errors."""
        user_data = {
            'username': '',  # Invalid: empty username
            'email': 'invalid-email',  # Invalid: bad email format
            'password': '123'  # Invalid: too short
        }
        
        response = client.post(
            '/api/v1/users',
            data=json.dumps(user_data),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        
        data = json.loads(response.data)
        assert data['status'] == 'error'
        assert 'errors' in data
    
    def test_create_user_duplicate_email(self, client, sample_user):
        """Test user creation with duplicate email."""
        user_data = {
            'username': 'anotheruser',
            'email': sample_user.email,  # Duplicate email
            'password': 'password123'
        }
        
        response = client.post(
            '/api/v1/users',
            data=json.dumps(user_data),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        
        data = json.loads(response.data)
        assert data['status'] == 'error'
        assert 'email already exists' in data['message'].lower()
    
    def test_get_user_by_id(self, client, sample_user):
        """Test getting user by ID."""
        response = client.get(f'/api/v1/users/{sample_user.id}')
        
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['status'] == 'success'
        assert data['data']['id'] == sample_user.id
        assert data['data']['username'] == sample_user.username
    
    def test_get_user_not_found(self, client):
        """Test getting non-existent user."""
        response = client.get('/api/v1/users/99999')
        
        assert response.status_code == 404
        
        data = json.loads(response.data)
        assert data['status'] == 'error'
    
    def test_update_user(self, client, sample_user, auth_headers):
        """Test updating user."""
        update_data = {
            'username': 'updateduser',
            'email': 'updated@example.com'
        }
        
        response = client.put(
            f'/api/v1/users/{sample_user.id}',
            data=json.dumps(update_data),
            content_type='application/json',
            headers=auth_headers
        )
        
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['status'] == 'success'
        assert data['data']['username'] == 'updateduser'
    
    def test_delete_user(self, client, sample_user, auth_headers):
        """Test deleting user."""
        response = client.delete(
            f'/api/v1/users/{sample_user.id}',
            headers=auth_headers
        )
        
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['status'] == 'success'
        
        # Verify user is deleted
        response = client.get(f'/api/v1/users/{sample_user.id}')
        assert response.status_code == 404

class TestPostAPI:
    """Test post-related endpoints."""
    
    def test_create_post(self, client, sample_user, auth_headers):
        """Test creating a post."""
        post_data = {
            'title': 'New Post',
            'content': 'This is a new post content',
            'author_id': sample_user.id
        }
        
        response = client.post(
            '/api/v1/posts',
            data=json.dumps(post_data),
            content_type='application/json',
            headers=auth_headers
        )
        
        assert response.status_code == 201
        
        data = json.loads(response.data)
        assert data['status'] == 'success'
        assert data['data']['title'] == 'New Post'
    
    def test_get_posts_with_pagination(self, client, sample_post):
        """Test getting posts with pagination."""
        response = client.get('/api/v1/posts?page=1&per_page=10')
        
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['status'] == 'success'
        assert 'pagination' in data['data']
        assert len(data['data']['posts']) >= 1

class TestAuthentication:
    """Test authentication endpoints."""
    
    def test_login_success(self, client, sample_user):
        """Test successful login."""
        login_data = {
            'username': sample_user.username,
            'password': 'password123'
        }
        
        response = client.post(
            '/api/v1/auth/login',
            data=json.dumps(login_data),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['status'] == 'success'
        assert 'access_token' in data['data']
    
    def test_login_invalid_credentials(self, client, sample_user):
        """Test login with invalid credentials."""
        login_data = {
            'username': sample_user.username,
            'password': 'wrongpassword'
        }
        
        response = client.post(
            '/api/v1/auth/login',
            data=json.dumps(login_data),
            content_type='application/json'
        )
        
        assert response.status_code == 401
        
        data = json.loads(response.data)
        assert data['status'] == 'error'
    
    def test_protected_endpoint_without_token(self, client):
        """Test accessing protected endpoint without token."""
        response = client.get('/api/v1/protected')
        
        assert response.status_code == 401
    
    def test_protected_endpoint_with_token(self, client, auth_headers):
        """Test accessing protected endpoint with valid token."""
        response = client.get('/api/v1/protected', headers=auth_headers)
        
        assert response.status_code == 200

# Parametrized tests
@pytest.mark.parametrize("email,expected_status", [
    ("valid@example.com", 201),
    ("invalid.email", 400),
    ("", 400),
    ("@example.com", 400),
    ("valid@", 400)
])
def test_user_creation_email_validation(client, email, expected_status):
    """Test user creation with various email formats."""
    user_data = {
        'username': 'testuser',
        'email': email,
        'password': 'password123'
    }
    
    response = client.post(
        '/api/v1/users',
        data=json.dumps(user_data),
        content_type='application/json'
    )
    
    assert response.status_code == expected_status

# Custom markers
@pytest.mark.slow
def test_bulk_user_creation(client):
    """Test creating many users (marked as slow test)."""
    users = []
    for i in range(100):
        user_data = {
            'username': f'user{i}',
            'email': f'user{i}@example.com',
            'password': 'password123'
        }
        users.append(user_data)
    
    for user_data in users:
        response = client.post(
            '/api/v1/users',
            data=json.dumps(user_data),
            content_type='application/json'
        )
        assert response.status_code == 201

# Test utilities
def assert_valid_user_response(data):
    """Helper function to validate user response structure."""
    assert 'id' in data
    assert 'username' in data
    assert 'email' in data
    assert 'created_at' in data
    assert 'password' not in data  # Ensure password is not exposed

def assert_error_response(data, expected_message=None):
    """Helper function to validate error response structure."""
    assert data['status'] == 'error'
    assert 'message' in data
    if expected_message:
        assert expected_message in data['message']
```

## Testing Flask APIs

### 1. Flask-specific Testing
```python
# test_flask_api.py
import pytest
import json
from unittest.mock import patch, MagicMock
from flask import url_for
from myapp import create_app, db
from myapp.models import User

class TestFlaskAPI:
    """Flask-specific API tests."""
    
    @pytest.fixture(autouse=True)
    def setup(self, app):
        """Setup for each test."""
        self.app = app
        self.client = app.test_client()
        self.ctx = app.app_context()
        self.ctx.push()
    
    def teardown(self):
        """Cleanup after each test."""
        self.ctx.pop()
    
    def test_application_context(self):
        """Test that application context is working."""
        assert self.app.config['TESTING'] is True
    
    def test_url_generation(self):
        """Test URL generation."""
        with self.app.test_request_context():
            assert url_for('api.get_users') == '/api/v1/users'
            assert url_for('api.get_user', user_id=1) == '/api/v1/users/1'
    
    def test_request_context(self):
        """Test request context in tests."""
        with self.client:
            response = self.client.get('/api/v1/users')
            # Access request object
            from flask import request
            assert request.endpoint == 'api.get_users'
    
    def test_session_handling(self):
        """Test session handling in Flask."""
        with self.client.session_transaction() as sess:
            sess['user_id'] = 123
        
        response = self.client.get('/api/v1/profile')
        # Test session-based functionality
    
    def test_custom_headers(self):
        """Test custom headers in requests."""
        headers = {
            'X-API-Version': '1.0',
            'X-Request-ID': 'test-request-123'
        }
        
        response = self.client.get('/api/v1/users', headers=headers)
        assert response.status_code == 200
    
    def test_file_upload(self):
        """Test file upload endpoint."""
        from io import BytesIO
        
        data = {
            'file': (BytesIO(b'test file content'), 'test.txt')
        }
        
        response = self.client.post(
            '/api/v1/upload',
            data=data,
            content_type='multipart/form-data'
        )
        
        assert response.status_code == 200
    
    def test_json_error_handling(self):
        """Test JSON error handling."""
        response = self.client.post(
            '/api/v1/users',
            data='invalid json',
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data

class TestFlaskMiddleware:
    """Test Flask middleware and hooks."""
    
    def test_before_request_hook(self, client, mocker):
        """Test before_request hook."""
        mock_before_request = mocker.patch('myapp.api.before_request_handler')
        
        client.get('/api/v1/users')
        
        mock_before_request.assert_called_once()
    
    def test_after_request_hook(self, client, mocker):
        """Test after_request hook."""
        mock_after_request = mocker.patch('myapp.api.after_request_handler')
        
        client.get('/api/v1/users')
        
        mock_after_request.assert_called_once()
    
    def test_error_handler(self, client):
        """Test custom error handlers."""
        # Trigger a 404 error
        response = client.get('/api/v1/nonexistent')
        
        assert response.status_code == 404
        data = json.loads(response.data)
        assert data['status'] == 'error'
        assert 'not found' in data['message'].lower()

# Database testing utilities
class TestDatabase:
    """Test database operations."""
    
    def test_database_connection(self, app):
        """Test database connection."""
        with app.app_context():
            # Test that we can connect to the database
            result = db.engine.execute('SELECT 1')
            assert result.fetchone()[0] == 1
    
    def test_model_creation(self, app):
        """Test model creation and querying."""
        with app.app_context():
            user = User(username='testuser', email='test@example.com')
            db.session.add(user)
            db.session.commit()
            
            # Query the user
            found_user = User.query.filter_by(username='testuser').first()
            assert found_user is not None
            assert found_user.email == 'test@example.com'
    
    def test_database_rollback(self, app):
        """Test database rollback on error."""
        with app.app_context():
            try:
                user1 = User(username='user1', email='user1@example.com')
                user2 = User(username='user2', email='user1@example.com')  # Duplicate email
                
                db.session.add(user1)
                db.session.add(user2)
                db.session.commit()
            except Exception:
                db.session.rollback()
            
            # Verify no users were created
            assert User.query.count() == 0
```

## Testing FastAPI

### 1. FastAPI Testing with TestClient
```python
# test_fastapi.py
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from myapp.main import app
from myapp.database import get_db, Base
from myapp.models import User, Post

# Test database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    """Override database dependency for testing."""
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

@pytest.fixture(scope="module")
def client():
    """Create test client."""
    Base.metadata.create_all(bind=engine)
    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as test_client:
        yield test_client
    
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def test_user(client):
    """Create test user."""
    user_data = {
        "username": "testuser",
        "email": "test@example.com",
        "password": "testpassword"
    }
    
    response = client.post("/api/v1/users/", json=user_data)
    return response.json()

@pytest.fixture
def auth_token(client, test_user):
    """Get authentication token."""
    login_data = {
        "username": "testuser",
        "password": "testpassword"
    }
    
    response = client.post("/api/v1/auth/token", data=login_data)
    token_data = response.json()
    return token_data["access_token"]

@pytest.fixture
def auth_headers(auth_token):
    """Create authorization headers."""
    return {"Authorization": f"Bearer {auth_token}"}

class TestFastAPIUsers:
    """Test FastAPI user endpoints."""
    
    def test_create_user(self, client):
        """Test user creation."""
        user_data = {
            "username": "newuser",
            "email": "newuser@example.com",
            "password": "password123"
        }
        
        response = client.post("/api/v1/users/", json=user_data)
        
        assert response.status_code == 201
        data = response.json()
        assert data["username"] == "newuser"
        assert data["email"] == "newuser@example.com"
        assert "id" in data
    
    def test_get_users(self, client, test_user):
        """Test getting users list."""
        response = client.get("/api/v1/users/")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 1
        assert any(user["username"] == "testuser" for user in data)
    
    def test_get_user_by_id(self, client, test_user):
        """Test getting user by ID."""
        user_id = test_user["id"]
        response = client.get(f"/api/v1/users/{user_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == user_id
        assert data["username"] == "testuser"
    
    def test_get_user_not_found(self, client):
        """Test getting non-existent user."""
        response = client.get("/api/v1/users/99999")
        
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()
    
    def test_update_user(self, client, test_user, auth_headers):
        """Test updating user."""
        user_id = test_user["id"]
        update_data = {
            "username": "updateduser",
            "email": "updated@example.com"
        }
        
        response = client.put(
            f"/api/v1/users/{user_id}",
            json=update_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "updateduser"
        assert data["email"] == "updated@example.com"

class TestFastAPIValidation:
    """Test FastAPI request validation."""
    
    def test_user_validation_errors(self, client):
        """Test user creation validation errors."""
        invalid_data = {
            "username": "",  # Too short
            "email": "invalid-email",  # Invalid format
            "password": "123"  # Too short
        }
        
        response = client.post("/api/v1/users/", json=invalid_data)
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
        
        # Check that validation errors are present
        errors = data["detail"]
        assert any(error["loc"] == ["body", "username"] for error in errors)
        assert any(error["loc"] == ["body", "email"] for error in errors)
    
    def test_query_parameter_validation(self, client):
        """Test query parameter validation."""
        # Test invalid skip parameter
        response = client.get("/api/v1/users/?skip=-1")
        assert response.status_code == 422
        
        # Test invalid limit parameter
        response = client.get("/api/v1/users/?limit=0")
        assert response.status_code == 422
    
    def test_request_body_validation(self, client):
        """Test request body validation."""
        # Missing required fields
        incomplete_data = {"username": "testuser"}
        
        response = client.post("/api/v1/users/", json=incomplete_data)
        
        assert response.status_code == 422
        data = response.json()
        errors = data["detail"]
        assert any(error["loc"] == ["body", "email"] for error in errors)

class TestFastAPIAuthentication:
    """Test FastAPI authentication."""
    
    def test_login_success(self, client, test_user):
        """Test successful login."""
        login_data = {
            "username": "testuser",
            "password": "testpassword"
        }
        
        response = client.post("/api/v1/auth/token", data=login_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
    
    def test_login_invalid_credentials(self, client, test_user):
        """Test login with invalid credentials."""
        login_data = {
            "username": "testuser",
            "password": "wrongpassword"
        }
        
        response = client.post("/api/v1/auth/token", data=login_data)
        
        assert response.status_code == 401
        data = response.json()
        assert "detail" in data
    
    def test_protected_endpoint_without_token(self, client):
        """Test accessing protected endpoint without token."""
        response = client.get("/api/v1/users/me")
        
        assert response.status_code == 401
    
    def test_protected_endpoint_with_invalid_token(self, client):
        """Test accessing protected endpoint with invalid token."""
        headers = {"Authorization": "Bearer invalid-token"}
        response = client.get("/api/v1/users/me", headers=headers)
        
        assert response.status_code == 401
    
    def test_protected_endpoint_with_valid_token(self, client, auth_headers):
        """Test accessing protected endpoint with valid token."""
        response = client.get("/api/v1/users/me", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "testuser"

# Async testing
@pytest.mark.asyncio
async def test_async_endpoint():
    """Test async endpoints (if you have any)."""
    from httpx import AsyncClient
    from myapp.main import app
    
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/api/v1/async-endpoint")
        assert response.status_code == 200
```

## API Documentation with OpenAPI/Swagger

### 1. FastAPI Auto-Documentation
```python
# main.py - FastAPI with comprehensive documentation
from fastapi import FastAPI, Depends, HTTPException, status, Query, Path, Body
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional
from enum import Enum

# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="User Management API",
        version="1.0.0",
        description="""
        A comprehensive user management API with the following features:
        
        * **Users**: Create, read, update, and delete users
        * **Posts**: Manage user posts
        * **Authentication**: JWT-based authentication
        * **Authorization**: Role-based access control
        
        ## Authentication
        
        This API uses JWT (JSON Web Tokens) for authentication. To authenticate:
        
        1. Obtain a token by calling the `/api/v1/auth/token` endpoint
        2. Include the token in the Authorization header: `Bearer <token>`
        
        ## Rate Limiting
        
        API endpoints are rate limited:
        * Authentication endpoints: 5 requests per 5 minutes
        * Data endpoints: 1000 requests per hour
        """,
        routes=app.routes,
        contact={
            "name": "API Support",
            "email": "support@example.com",
            "url": "https://example.com/support"
        },
        license_info={
            "name": "MIT License",
            "url": "https://opensource.org/licenses/MIT"
        },
        servers=[
            {"url": "https://api.example.com", "description": "Production server"},
            {"url": "https://staging-api.example.com", "description": "Staging server"},
            {"url": "http://localhost:8000", "description": "Development server"}
        ]
    )
    
    # Add security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "bearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app = FastAPI()
app.openapi = custom_openapi

# Enums for better documentation
class UserRole(str, Enum):
    admin = "admin"
    user = "user"
    moderator = "moderator"

class PostStatus(str, Enum):
    draft = "draft"
    published = "published"
    archived = "archived"

# Detailed Pydantic models
class UserBase(BaseModel):
    """Base user model with common fields."""
    username: str = Field(
        ..., 
        min_length=3, 
        max_length=50,
        description="Username must be between 3 and 50 characters",
        example="johndoe"
    )
    email: EmailStr = Field(
        ...,
        description="Valid email address",
        example="john.doe@example.com"
    )
    full_name: Optional[str] = Field(
        None,
        max_length=100,
        description="User's full name",
        example="John Doe"
    )
    is_active: bool = Field(
        True,
        description="Whether the user account is active"
    )
    role: UserRole = Field(
        UserRole.user,
        description="User role determining permissions"
    )

class UserCreate(UserBase):
    """User creation model."""
    password: str = Field(
        ...,
        min_length=8,
        description="Password must be at least 8 characters long",
        example="secretpassword123"
    )

class UserResponse(UserBase):
    """User response model."""
    id: int = Field(..., description="Unique user identifier", example=1)
    created_at: str = Field(..., description="User creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    
    class Config:
        orm_mode = True
        schema_extra = {
            "example": {
                "id": 1,
                "username": "johndoe",
                "email": "john.doe@example.com",
                "full_name": "John Doe",
                "is_active": True,
                "role": "user",
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:30:00Z"
            }
        }

class PostCreate(BaseModel):
    """Post creation model."""
    title: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Post title",
        example="My First Blog Post"
    )
    content: str = Field(
        ...,
        min_length=1,
        description="Post content in markdown format",
        example="This is the content of my first blog post."
    )
    status: PostStatus = Field(
        PostStatus.draft,
        description="Post publication status"
    )
    tags: Optional[List[str]] = Field(
        None,
        description="List of tags associated with the post",
        example=["python", "fastapi", "tutorial"]
    )

# Documented endpoints
@app.post(
    "/api/v1/users/",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new user",
    description="Create a new user account with the provided information.",
    response_description="The created user",
    tags=["users"]
)
async def create_user(
    user: UserCreate = Body(
        ...,
        description="User data for creation",
        example={
            "username": "johndoe",
            "email": "john.doe@example.com",
            "full_name": "John Doe",
            "password": "secretpassword123",
            "role": "user"
        }
    )
):
    """
    Create a new user account.
    
    This endpoint allows you to create a new user account with the following validations:
    - Username must be unique and between 3-50 characters
    - Email must be valid and unique
    - Password must be at least 8 characters long
    - Role must be one of: admin, user, moderator
    
    Returns the created user information (excluding password).
    """
    # Implementation here
    pass

@app.get(
    "/api/v1/users/",
    response_model=List[UserResponse],
    summary="Get all users",
    description="Retrieve a list of all users with optional pagination.",
    tags=["users"]
)
async def get_users(
    skip: int = Query(
        0,
        ge=0,
        description="Number of records to skip for pagination",
        example=0
    ),
    limit: int = Query(
        10,
        ge=1,
        le=100,
        description="Maximum number of records to return",
        example=10
    ),
    role: Optional[UserRole] = Query(
        None,
        description="Filter users by role"
    ),
    is_active: Optional[bool] = Query(
        None,
        description="Filter users by active status"
    )
):
    """
    Get all users with optional filtering and pagination.
    
    You can filter users by:
    - Role (admin, user, moderator)
    - Active status (true/false)
    
    Pagination parameters:
    - skip: Number of records to skip (default: 0)
    - limit: Maximum records to return (default: 10, max: 100)
    """
    # Implementation here
    pass

@app.get(
    "/api/v1/users/{user_id}",
    response_model=UserResponse,
    summary="Get user by ID",
    description="Retrieve a specific user by their ID.",
    responses={
        200: {"description": "User found and returned"},
        404: {"description": "User not found"},
    },
    tags=["users"]
)
async def get_user(
    user_id: int = Path(
        ...,
        ge=1,
        description="The ID of the user to retrieve",
        example=1
    )
):
    """
    Get a specific user by their ID.
    
    Returns detailed information about the user including:
    - Basic profile information
    - Account status
    - Role and permissions
    - Creation and update timestamps
    """
    # Implementation here
    pass

# Add tags metadata
tags_metadata = [
    {
        "name": "users",
        "description": "Operations with users. The **login** logic is also here.",
        "externalDocs": {
            "description": "User management guide",
            "url": "https://docs.example.com/users/",
        },
    },
    {
        "name": "posts",
        "description": "Manage blog posts. So _fancy_ they have their own docs.",
        "externalDocs": {
            "description": "Posts external docs",
            "url": "https://docs.example.com/posts/",
        },
    },
    {
        "name": "auth",
        "description": "Authentication and authorization endpoints.",
    }
]

app = FastAPI(
    title="User Management API",
    description="A comprehensive user management API",
    version="1.0.0",
    openapi_tags=tags_metadata,
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc",  # ReDoc
    openapi_url="/openapi.json"
)
```

### 2. Flask API Documentation with Flask-RESTX
```python
# app.py - Flask with Flask-RESTX for documentation
from flask import Flask
from flask_restx import Api, Resource, fields, Namespace
from werkzeug.exceptions import BadRequest, NotFound

app = Flask(__name__)

# Configure Flask-RESTX
api = Api(
    app,
    version='1.0',
    title='User Management API',
    description='A comprehensive user management API built with Flask',
    doc='/docs/',  # Documentation URL
    contact='support@example.com',
    contact_email='support@example.com',
    contact_url='https://example.com/support',
    license='MIT',
    license_url='https://opensource.org/licenses/MIT'
)

# Create namespaces
users_ns = Namespace('users', description='User operations')
auth_ns = Namespace('auth', description='Authentication operations')
posts_ns = Namespace('posts', description='Post operations')

api.add_namespace(users_ns, path='/api/v1/users')
api.add_namespace(auth_ns, path='/api/v1/auth')
api.add_namespace(posts_ns, path='/api/v1/posts')

# Define models for documentation
user_model = api.model('User', {
    'id': fields.Integer(readonly=True, description='User unique identifier'),
    'username': fields.String(required=True, description='Username', min_length=3, max_length=50),
    'email': fields.String(required=True, description='Email address'),
    'full_name': fields.String(description='Full name'),
    'is_active': fields.Boolean(description='Account active status'),
    'role': fields.String(description='User role', enum=['admin', 'user', 'moderator']),
    'created_at': fields.DateTime(readonly=True, description='Creation timestamp'),
    'updated_at': fields.DateTime(readonly=True, description='Last update timestamp')
})

user_create_model = api.model('UserCreate', {
    'username': fields.String(required=True, description='Username', min_length=3, max_length=50),
    'email': fields.String(required=True, description='Email address'),
    'password': fields.String(required=True, description='Password', min_length=8),
    'full_name': fields.String(description='Full name'),
    'role': fields.String(description='User role', enum=['admin', 'user', 'moderator'], default='user')
})

user_update_model = api.model('UserUpdate', {
    'username': fields.String(description='Username', min_length=3, max_length=50),
    'email': fields.String(description='Email address'),
    'full_name': fields.String(description='Full name'),
    'is_active': fields.Boolean(description='Account active status'),
    'role': fields.String(description='User role', enum=['admin', 'user', 'moderator'])
})

post_model = api.model('Post', {
    'id': fields.Integer(readonly=True, description='Post unique identifier'),
    'title': fields.String(required=True, description='Post title'),
    'content': fields.String(required=True, description='Post content'),
    'author_id': fields.Integer(required=True, description='Author user ID'),
    'author_name': fields.String(readonly=True, description='Author username'),
    'status': fields.String(description='Post status', enum=['draft', 'published', 'archived']),
    'created_at': fields.DateTime(readonly=True, description='Creation timestamp'),
    'updated_at': fields.DateTime(readonly=True, description='Last update timestamp')
})

# Error models
error_model = api.model('Error', {
    'status': fields.String(description='Error status'),
    'message': fields.String(description='Error message'),
    'errors': fields.List(fields.Raw, description='Detailed error information')
})

# Authentication models
login_model = api.model('Login', {
    'username': fields.String(required=True, description='Username'),
    'password': fields.String(required=True, description='Password')
})

token_model = api.model('Token', {
    'access_token': fields.String(description='JWT access token'),
    'token_type': fields.String(description='Token type'),
    'expires_in': fields.Integer(description='Token expiration time in seconds')
})

# Pagination model
pagination_model = api.model('Pagination', {
    'page': fields.Integer(description='Current page number'),
    'per_page': fields.Integer(description='Items per page'),
    'total': fields.Integer(description='Total number of items'),
    'pages': fields.Integer(description='Total number of pages')
})

users_list_model = api.model('UsersList', {
    'users': fields.List(fields.Nested(user_model)),
    'pagination': fields.Nested(pagination_model)
})

# Users namespace
@users_ns.route('/')
class UserList(Resource):
    @users_ns.doc('list_users')
    @users_ns.marshal_list_with(users_list_model)
    @users_ns.param('page', 'Page number', type='integer', default=1)
    @users_ns.param('per_page', 'Items per page', type='integer', default=10)
    @users_ns.param('role', 'Filter by role', type='string', enum=['admin', 'user', 'moderator'])
    def get(self):
        """Get all users with optional filtering and pagination."""
        # Implementation here
        pass
    
    @users_ns.doc('create_user')
    @users_ns.expect(user_create_model)
    @users_ns.marshal_with(user_model, code=201)
    @users_ns.response(400, 'Validation error', error_model)
    def post(self):
        """Create a new user."""
        # Implementation here
        pass

@users_ns.route('/<int:user_id>')
@users_ns.param('user_id', 'User identifier')
class User(Resource):
    @users_ns.doc('get_user')
    @users_ns.marshal_with(user_model)
    @users_ns.response(404, 'User not found', error_model)
    def get(self, user_id):
        """Get a user by ID."""
        # Implementation here
        pass
    
    @users_ns.doc('update_user')
    @users_ns.expect(user_update_model)
    @users_ns.marshal_with(user_model)
    @users_ns.response(404, 'User not found', error_model)
    @users_ns.response(400, 'Validation error', error_model)
    def put(self, user_id):
        """Update a user."""
        # Implementation here
        pass
    
    @users_ns.doc('delete_user')
    @users_ns.response(204, 'User deleted')
    @users_ns.response(404, 'User not found', error_model)
    def delete(self, user_id):
        """Delete a user."""
        # Implementation here
        pass

# Authentication namespace
@auth_ns.route('/login')
class Login(Resource):
    @auth_ns.doc('login')
    @auth_ns.expect(login_model)
    @auth_ns.marshal_with(token_model)
    @auth_ns.response(401, 'Invalid credentials', error_model)
    def post(self):
        """Authenticate user and get access token."""
        # Implementation here
        pass

# Custom error handlers with documentation
@api.errorhandler(BadRequest)
def handle_bad_request(error):
    """Handle bad request errors."""
    return {'status': 'error', 'message': str(error)}, 400

@api.errorhandler(NotFound)
def handle_not_found(error):
    """Handle not found errors."""
    return {'status': 'error', 'message': 'Resource not found'}, 404

if __name__ == '__main__':
    app.run(debug=True)
```

## Load Testing

### 1. Load Testing with Locust
```python
# locustfile.py
from locust import HttpUser, task, between
import json
import random

class APIUser(HttpUser):
    """Simulate API user behavior."""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Called when a user starts."""
        self.client.verify = False  # Disable SSL verification for testing
        self.auth_token = None
        self.user_id = None
        
        # Create a test user and login
        self.create_user_and_login()
    
    def create_user_and_login(self):
        """Create a test user and obtain auth token."""
        username = f"testuser_{random.randint(1000, 9999)}"
        email = f"{username}@example.com"
        
        # Create user
        user_data = {
            "username": username,
            "email": email,
            "password": "testpassword123"
        }
        
        response = self.client.post("/api/v1/users/", json=user_data)
        if response.status_code == 201:
            user = response.json()
            self.user_id = user.get("id")
        
        # Login
        login_data = {
            "username": username,
            "password": "testpassword123"
        }
        
        response = self.client.post("/api/v1/auth/token", data=login_data)
        if response.status_code == 200:
            token_data = response.json()
            self.auth_token = token_data.get("access_token")
    
    def get_headers(self):
        """Get authorization headers."""
        if self.auth_token:
            return {"Authorization": f"Bearer {self.auth_token}"}
        return {}
    
    @task(10)
    def get_users(self):
        """Get users list (most common operation)."""
        params = {
            "skip": random.randint(0, 50),
            "limit": random.randint(5, 20)
        }
        
        with self.client.get("/api/v1/users/", params=params, 
                           catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")
    
    @task(5)
    def get_user_by_id(self):
        """Get specific user by ID."""
        if self.user_id:
            with self.client.get(f"/api/v1/users/{self.user_id}", 
                               catch_response=True) as response:
                if response.status_code == 200:
                    response.success()
                elif response.status_code == 404:
                    response.failure("User not found")
                else:
                    response.failure(f"Got status code {response.status_code}")
    
    @task(3)
    def create_post(self):
        """Create a new post."""
        if not self.auth_token:
            return
        
        post_data = {
            "title": f"Test Post {random.randint(1, 1000)}",
            "content": "This is a test post content for load testing.",
            "status": random.choice(["draft", "published"])
        }
        
        headers = self.get_headers()
        
        with self.client.post("/api/v1/posts/", json=post_data, 
                            headers=headers, catch_response=True) as response:
            if response.status_code == 201:
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")
    
    @task(2)
    def update_user(self):
        """Update user information."""
        if not self.auth_token or not self.user_id:
            return
        
        update_data = {
            "full_name": f"Test User {random.randint(1, 1000)}"
        }
        
        headers = self.get_headers()
        
        with self.client.put(f"/api/v1/users/{self.user_id}", 
                           json=update_data, headers=headers, 
                           catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")
    
    @task(1)
    def search_users(self):
        """Search users with filters."""
        params = {
            "role": random.choice(["user", "admin", "moderator"]),
            "is_active": random.choice([True, False])
        }
        
        with self.client.get("/api/v1/users/", params=params, 
                           catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")

class AdminUser(HttpUser):
    """Simulate admin user with different behavior patterns."""
    
    wait_time = between(2, 5)
    weight = 1  # 1 admin for every 10 regular users
    
    def on_start(self):
        """Login as admin."""
        self.client.verify = False
        
        # Login with admin credentials
        login_data = {
            "username": "admin",
            "password": "adminpassword"
        }
        
        response = self.client.post("/api/v1/auth/token", data=login_data)
        if response.status_code == 200:
            token_data = response.json()
            self.auth_token = token_data.get("access_token")
    
    @task(5)
    def admin_get_all_users(self):
        """Admin fetching all users."""
        headers = {"Authorization": f"Bearer {self.auth_token}"}
        
        params = {"limit": 100}  # Admins might fetch more data
        
        with self.client.get("/api/v1/users/", params=params, 
                           headers=headers, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")
    
    @task(2)
    def admin_operations(self):
        """Admin-specific operations."""
        headers = {"Authorization": f"Bearer {self.auth_token}"}
        
        with self.client.get("/api/v1/admin/dashboard", 
                           headers=headers, catch_response=True) as response:
            if response.status_code in [200, 404]:  # 404 if endpoint doesn't exist
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")

# Custom load test with different user types
class MixedAPIUser(HttpUser):
    """Mixed user types for realistic load testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Randomly assign user type
        self.user_type = random.choices(
            ["guest", "user", "premium", "admin"],
            weights=[30, 50, 15, 5]  # Realistic distribution
        )[0]
    
    def on_start(self):
        self.setup_user_type()
    
    def setup_user_type(self):
        """Setup based on user type."""
        if self.user_type == "guest":
            self.auth_token = None
        else:
            # Create and login user
            self.create_and_login()
    
    def create_and_login(self):
        """Create user and login based on type."""
        username = f"{self.user_type}_{random.randint(1000, 9999)}"
        user_data = {
            "username": username,
            "email": f"{username}@example.com",
            "password": "password123",
            "role": "admin" if self.user_type == "admin" else "user"
        }
        
        # Create user
        self.client.post("/api/v1/users/", json=user_data)
        
        # Login
        login_data = {
            "username": username,
            "password": "password123"
        }
        
        response = self.client.post("/api/v1/auth/token", data=login_data)
        if response.status_code == 200:
            token_data = response.json()
            self.auth_token = token_data.get("access_token")
    
    @task
    def user_behavior(self):
        """Execute behavior based on user type."""
        if self.user_type == "guest":
            self.guest_behavior()
        elif self.user_type == "user":
            self.regular_user_behavior()
        elif self.user_type == "premium":
            self.premium_user_behavior()
        elif self.user_type == "admin":
            self.admin_behavior()
    
    def guest_behavior(self):
        """Guest user behavior (read-only)."""
        # Guests can only view public content
        self.client.get("/api/v1/users/")
        self.client.get("/api/v1/posts/")
    
    def regular_user_behavior(self):
        """Regular user behavior."""
        headers = {"Authorization": f"Bearer {self.auth_token}"}
        
        # Regular users read more than they write
        actions = ["read"] * 7 + ["write"] * 3
        action = random.choice(actions)
        
        if action == "read":
            self.client.get("/api/v1/users/", headers=headers)
        else:
            post_data = {
                "title": f"Post {random.randint(1, 1000)}",
                "content": "Test content"
            }
            self.client.post("/api/v1/posts/", json=post_data, headers=headers)
    
    def premium_user_behavior(self):
        """Premium user behavior (more active)."""
        headers = {"Authorization": f"Bearer {self.auth_token}"}
        
        # Premium users are more active
        actions = ["read"] * 5 + ["write"] * 4 + ["update"] * 1
        action = random.choice(actions)
        
        if action == "read":
            self.client.get("/api/v1/users/", headers=headers)
        elif action == "write":
            post_data = {
                "title": f"Premium Post {random.randint(1, 1000)}",
                "content": "Premium content with more features"
            }
            self.client.post("/api/v1/posts/", json=post_data, headers=headers)
        elif action == "update":
            # Update profile
            update_data = {"full_name": f"Premium User {random.randint(1, 1000)}"}
            self.client.put("/api/v1/users/me", json=update_data, headers=headers)
    
    def admin_behavior(self):
        """Admin user behavior."""
        headers = {"Authorization": f"Bearer {self.auth_token}"}
        
        # Admins perform management tasks
        self.client.get("/api/v1/users/?limit=100", headers=headers)
        self.client.get("/api/v1/admin/stats", headers=headers)

# Run with: locust -f locustfile.py --host=http://localhost:8000
```

### 2. Performance Testing with Artillery
```yaml
# artillery-config.yml
config:
  target: 'http://localhost:8000'
  phases:
    - duration: 60
      arrivalRate: 10
      name: "Warm up"
    - duration: 120
      arrivalRate: 50
      name: "Sustained load"
    - duration: 60
      arrivalRate: 100
      name: "Peak load"
  defaults:
    headers:
      Content-Type: 'application/json'

scenarios:
  - name: "User Registration and Login Flow"
    weight: 30
    flow:
      - post:
          url: "/api/v1/users/"
          json:
            username: "user{{ $randomString() }}"
            email: "user{{ $randomString() }}@example.com"
            password: "password123"
          capture:
            - json: "$.id"
              as: "userId"
      - post:
          url: "/api/v1/auth/token"
          form:
            username: "user{{ $randomString() }}"
            password: "password123"
          capture:
            - json: "$.access_token"
              as: "authToken"
      - get:
          url: "/api/v1/users/{{ userId }}"
          headers:
            Authorization: "Bearer {{ authToken }}"

  - name: "Browse Users"
    weight: 50
    flow:
      - get:
          url: "/api/v1/users/"
          qs:
            skip: "{{ $randomInt(0, 100) }}"
            limit: "{{ $randomInt(5, 20) }}"
      - think: 2
      - get:
          url: "/api/v1/users/{{ $randomInt(1, 100) }}"

  - name: "Create Posts"
    weight: 20
    flow:
      - post:
          url: "/api/v1/auth/token"
          form:
            username: "testuser"
            password: "testpass"
          capture:
            - json: "$.access_token"
              as: "authToken"
      - post:
          url: "/api/v1/posts/"
          headers:
            Authorization: "Bearer {{ authToken }}"
          json:
            title: "Test Post {{ $randomString() }}"
            content: "This is test content for post {{ $randomString() }}"
            status: "published"
```

This completes the testing and documentation section. The documentation covers comprehensive testing strategies including unit tests, integration tests, load testing, and API documentation best practices for both Flask and FastAPI applications.
