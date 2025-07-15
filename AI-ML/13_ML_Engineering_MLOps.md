# 13. ML Engineering & MLOps

## 🎯 Learning Objectives
- Master MLOps principles and practices
- Learn model deployment strategies and containerization
- Understand model monitoring and versioning
- Implement automated ML pipelines
- Apply production-ready ML engineering practices

---

## 1. MLOps Fundamentals

**MLOps** (Machine Learning Operations) is the practice of deploying, monitoring, and maintaining ML models in production.

### 1.1 MLOps Lifecycle 🟢

#### Core Components:
```python
import os
import json
import pickle
import joblib
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MLPipelineConfig:
    """Configuration management for ML pipelines"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """Load configuration from file or use defaults"""
        default_config = {
            'model': {
                'name': 'default_model',
                'version': '1.0.0',
                'algorithm': 'random_forest',
                'hyperparameters': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'random_state': 42
                }
            },
            'data': {
                'train_path': 'data/train.csv',
                'test_path': 'data/test.csv',
                'validation_split': 0.2,
                'target_column': 'target'
            },
            'deployment': {
                'environment': 'staging',
                'api_version': 'v1',
                'max_requests_per_minute': 1000,
                'timeout_seconds': 30
            },
            'monitoring': {
                'drift_threshold': 0.1,
                'performance_threshold': 0.85,
                'alert_email': 'admin@company.com'
            }
        }
        
        if self.config_path and os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                loaded_config = json.load(f)
                # Merge with defaults
                default_config.update(loaded_config)
        
        return default_config
    
    def save_config(self, path: str = None):
        """Save current configuration to file"""
        save_path = path or self.config_path or 'config.json'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"Configuration saved to {save_path}")
    
    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation"""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value):
        """Set configuration value using dot notation"""
        keys = key_path.split('.')
        config_ref = self.config
        
        for key in keys[:-1]:
            if key not in config_ref:
                config_ref[key] = {}
            config_ref = config_ref[key]
        
        config_ref[keys[-1]] = value

class ModelRegistry:
    """Model versioning and registry system"""
    
    def __init__(self, registry_path: str = "models/registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.registry_path / "metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load model registry metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {'models': {}}
    
    def _save_metadata(self):
        """Save model registry metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def register_model(self, model, name: str, version: str, 
                      metadata: Dict = None, overwrite: bool = False):
        """Register a new model version"""
        
        model_key = f"{name}:{version}"
        model_path = self.registry_path / f"{name}_{version}.pkl"
        
        # Check if model already exists
        if model_key in self.metadata['models'] and not overwrite:
            raise ValueError(f"Model {model_key} already exists. Use overwrite=True to replace.")
        
        # Save model
        joblib.dump(model, model_path)
        
        # Update metadata
        self.metadata['models'][model_key] = {
            'name': name,
            'version': version,
            'path': str(model_path),
            'registered_at': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        self._save_metadata()
        logger.info(f"Model {model_key} registered successfully")
        
        return model_key
    
    def load_model(self, name: str, version: str = None):
        """Load a model from registry"""
        
        if version is None:
            # Get latest version
            versions = [v.split(':')[1] for k, v in self.metadata['models'].items() 
                       if k.startswith(f"{name}:")]
            if not versions:
                raise ValueError(f"No models found for name: {name}")
            version = max(versions)
        
        model_key = f"{name}:{version}"
        
        if model_key not in self.metadata['models']:
            raise ValueError(f"Model {model_key} not found in registry")
        
        model_info = self.metadata['models'][model_key]
        model_path = Path(model_info['path'])
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = joblib.load(model_path)
        logger.info(f"Model {model_key} loaded successfully")
        
        return model, model_info
    
    def list_models(self) -> Dict:
        """List all registered models"""
        return self.metadata['models']
    
    def delete_model(self, name: str, version: str):
        """Delete a model from registry"""
        model_key = f"{name}:{version}"
        
        if model_key not in self.metadata['models']:
            raise ValueError(f"Model {model_key} not found")
        
        # Delete model file
        model_info = self.metadata['models'][model_key]
        model_path = Path(model_info['path'])
        if model_path.exists():
            model_path.unlink()
        
        # Remove from metadata
        del self.metadata['models'][model_key]
        self._save_metadata()
        
        logger.info(f"Model {model_key} deleted successfully")

# Example usage
config = MLPipelineConfig()
registry = ModelRegistry()

# Display configuration
print("Current Configuration:")
print(json.dumps(config.config, indent=2))

# Example model registration
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Create sample model
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Register model
model_metadata = {
    'accuracy': 0.95,
    'features': ['feature_' + str(i) for i in range(10)],
    'training_samples': len(X),
    'algorithm': 'RandomForest'
}

model_key = registry.register_model(
    model, 
    name='customer_churn_predictor', 
    version='1.0.0',
    metadata=model_metadata
)

print(f"\nRegistered model: {model_key}")
print("Registry contents:")
for key, info in registry.list_models().items():
    print(f"  {key}: {info['registered_at']}")
```

### 1.2 Data Versioning and Lineage 🟡

```python
import hashlib
import shutil
from typing import Union

class DataVersioning:
    """Data versioning and lineage tracking system"""
    
    def __init__(self, data_path: str = "data/versions"):
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.lineage_file = self.data_path / "lineage.json"
        self.lineage = self._load_lineage()
    
    def _load_lineage(self) -> Dict:
        """Load data lineage information"""
        if self.lineage_file.exists():
            with open(self.lineage_file, 'r') as f:
                return json.load(f)
        return {'datasets': {}}
    
    def _save_lineage(self):
        """Save data lineage information"""
        with open(self.lineage_file, 'w') as f:
            json.dump(self.lineage, f, indent=2)
    
    def _calculate_hash(self, data: Union[pd.DataFrame, np.ndarray, str]) -> str:
        """Calculate hash of data for versioning"""
        if isinstance(data, pd.DataFrame):
            data_str = data.to_csv(index=False)
        elif isinstance(data, np.ndarray):
            data_str = data.tobytes()
        elif isinstance(data, str):
            with open(data, 'rb') as f:
                data_str = f.read()
        else:
            data_str = str(data).encode()
        
        return hashlib.sha256(data_str).hexdigest()[:16]
    
    def version_data(self, data: Union[pd.DataFrame, str], 
                    name: str, description: str = None,
                    parent_version: str = None) -> str:
        """Create a new version of dataset"""
        
        # Calculate hash
        data_hash = self._calculate_hash(data)
        version = f"{name}_{data_hash}"
        
        # Check if version already exists
        if version in self.lineage['datasets']:
            logger.info(f"Data version {version} already exists")
            return version
        
        # Save data
        if isinstance(data, pd.DataFrame):
            data_file = self.data_path / f"{version}.parquet"
            data.to_parquet(data_file, index=False)
        elif isinstance(data, str) and os.path.exists(data):
            data_file = self.data_path / f"{version}{Path(data).suffix}"
            shutil.copy2(data, data_file)
        else:
            raise ValueError("Data must be DataFrame or file path")
        
        # Update lineage
        self.lineage['datasets'][version] = {
            'name': name,
            'version': version,
            'hash': data_hash,
            'path': str(data_file),
            'created_at': datetime.now().isoformat(),
            'description': description,
            'parent_version': parent_version,
            'size_mb': data_file.stat().st_size / (1024 * 1024)
        }
        
        self._save_lineage()
        logger.info(f"Data version {version} created")
        
        return version
    
    def load_data(self, version: str) -> pd.DataFrame:
        """Load specific version of data"""
        if version not in self.lineage['datasets']:
            raise ValueError(f"Data version {version} not found")
        
        data_info = self.lineage['datasets'][version]
        data_path = Path(data_info['path'])
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        if data_path.suffix == '.parquet':
            return pd.read_parquet(data_path)
        elif data_path.suffix == '.csv':
            return pd.read_csv(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")
    
    def get_lineage(self, version: str = None) -> Dict:
        """Get data lineage information"""
        if version is None:
            return self.lineage['datasets']
        
        if version not in self.lineage['datasets']:
            raise ValueError(f"Data version {version} not found")
        
        # Build lineage chain
        lineage_chain = []
        current_version = version
        
        while current_version:
            data_info = self.lineage['datasets'][current_version]
            lineage_chain.append(data_info)
            current_version = data_info.get('parent_version')
        
        return lineage_chain
    
    def compare_versions(self, version1: str, version2: str) -> Dict:
        """Compare two data versions"""
        data1 = self.load_data(version1)
        data2 = self.load_data(version2)
        
        comparison = {
            'shape_change': {
                'from': data1.shape,
                'to': data2.shape
            },
            'columns_added': list(set(data2.columns) - set(data1.columns)),
            'columns_removed': list(set(data1.columns) - set(data2.columns)),
            'common_columns': list(set(data1.columns) & set(data2.columns))
        }
        
        # Check data types changes
        if comparison['common_columns']:
            common_data1 = data1[comparison['common_columns']]
            common_data2 = data2[comparison['common_columns']]
            
            dtype_changes = {}
            for col in comparison['common_columns']:
                if common_data1[col].dtype != common_data2[col].dtype:
                    dtype_changes[col] = {
                        'from': str(common_data1[col].dtype),
                        'to': str(common_data2[col].dtype)
                    }
            
            comparison['dtype_changes'] = dtype_changes
        
        return comparison

# Example usage
data_versioning = DataVersioning()

# Create sample datasets
np.random.seed(42)
original_data = pd.DataFrame({
    'feature1': np.random.randn(1000),
    'feature2': np.random.randn(1000),
    'target': np.random.choice([0, 1], 1000)
})

# Version original data
v1 = data_versioning.version_data(
    original_data, 
    'customer_data', 
    description='Original customer dataset'
)

# Create modified version
modified_data = original_data.copy()
modified_data['feature3'] = np.random.randn(1000)
modified_data = modified_data.drop('feature2', axis=1)

v2 = data_versioning.version_data(
    modified_data,
    'customer_data',
    description='Added feature3, removed feature2',
    parent_version=v1
)

print(f"Data versions created: {v1}, {v2}")

# Compare versions
comparison = data_versioning.compare_versions(v1, v2)
print("\nVersion comparison:")
print(json.dumps(comparison, indent=2))

# Show lineage
lineage = data_versioning.get_lineage(v2)
print(f"\nLineage for {v2}:")
for i, item in enumerate(lineage):
    print(f"  {i}: {item['name']} - {item['description']} ({item['created_at']})")
```

---

## 2. Model Deployment Strategies

### 2.1 REST API Deployment 🟢

```python
# Model serving API (Flask-based)
from flask import Flask, request, jsonify
import pandas as pd
from datetime import datetime
import traceback

class ModelAPI:
    """RESTful API for model serving"""
    
    def __init__(self, model, preprocessor=None, config=None):
        self.model = model
        self.preprocessor = preprocessor
        self.config = config or {}
        self.app = Flask(__name__)
        self._setup_routes()
        self._setup_logging()
        
        # Performance tracking
        self.request_count = 0
        self.error_count = 0
        self.start_time = datetime.now()
    
    def _setup_logging(self):
        """Setup API logging"""
        self.logger = logging.getLogger('model_api')
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            uptime = datetime.now() - self.start_time
            return jsonify({
                'status': 'healthy',
                'uptime_seconds': uptime.total_seconds(),
                'requests_served': self.request_count,
                'error_rate': self.error_count / max(self.request_count, 1)
            })
        
        @self.app.route('/predict', methods=['POST'])
        def predict():
            """Prediction endpoint"""
            try:
                self.request_count += 1
                start_time = datetime.now()
                
                # Parse input data
                data = request.get_json()
                
                if not data:
                    raise ValueError("No input data provided")
                
                # Convert to DataFrame
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                elif isinstance(data, dict):
                    df = pd.DataFrame([data])
                else:
                    raise ValueError("Input data must be JSON object or array")
                
                # Preprocessing
                if self.preprocessor:
                    df_processed = self.preprocessor.transform(df)
                else:
                    df_processed = df
                
                # Make prediction
                predictions = self.model.predict(df_processed)
                
                # Get prediction probabilities if available
                probabilities = None
                if hasattr(self.model, 'predict_proba'):
                    probabilities = self.model.predict_proba(df_processed).tolist()
                
                # Calculate processing time
                processing_time = (datetime.now() - start_time).total_seconds()
                
                response = {
                    'predictions': predictions.tolist(),
                    'probabilities': probabilities,
                    'processing_time_seconds': processing_time,
                    'model_version': self.config.get('model.version', 'unknown'),
                    'timestamp': datetime.now().isoformat()
                }
                
                self.logger.info(f"Prediction successful - {len(predictions)} samples in {processing_time:.3f}s")
                
                return jsonify(response)
                
            except Exception as e:
                self.error_count += 1
                error_msg = str(e)
                self.logger.error(f"Prediction error: {error_msg}")
                
                return jsonify({
                    'error': error_msg,
                    'traceback': traceback.format_exc(),
                    'timestamp': datetime.now().isoformat()
                }), 400
        
        @self.app.route('/model_info', methods=['GET'])
        def model_info():
            """Model information endpoint"""
            info = {
                'model_type': type(self.model).__name__,
                'model_version': self.config.get('model.version', 'unknown'),
                'features': getattr(self.model, 'feature_names_in_', None),
                'n_features': getattr(self.model, 'n_features_in_', None),
                'config': self.config
            }
            
            return jsonify(info)
        
        @self.app.route('/batch_predict', methods=['POST'])
        def batch_predict():
            """Batch prediction endpoint"""
            try:
                self.request_count += 1
                start_time = datetime.now()
                
                # Parse input data
                data = request.get_json()
                
                if not data or 'instances' not in data:
                    raise ValueError("Input must contain 'instances' key with array of data")
                
                instances = data['instances']
                df = pd.DataFrame(instances)
                
                # Preprocessing
                if self.preprocessor:
                    df_processed = self.preprocessor.transform(df)
                else:
                    df_processed = df
                
                # Make predictions in batches
                batch_size = data.get('batch_size', 1000)
                all_predictions = []
                all_probabilities = []
                
                for i in range(0, len(df_processed), batch_size):
                    batch = df_processed.iloc[i:i+batch_size]
                    batch_preds = self.model.predict(batch)
                    all_predictions.extend(batch_preds.tolist())
                    
                    if hasattr(self.model, 'predict_proba'):
                        batch_probs = self.model.predict_proba(batch)
                        all_probabilities.extend(batch_probs.tolist())
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                response = {
                    'predictions': all_predictions,
                    'probabilities': all_probabilities if all_probabilities else None,
                    'num_instances': len(all_predictions),
                    'processing_time_seconds': processing_time,
                    'throughput_per_second': len(all_predictions) / processing_time,
                    'timestamp': datetime.now().isoformat()
                }
                
                self.logger.info(f"Batch prediction successful - {len(all_predictions)} samples in {processing_time:.3f}s")
                
                return jsonify(response)
                
            except Exception as e:
                self.error_count += 1
                error_msg = str(e)
                self.logger.error(f"Batch prediction error: {error_msg}")
                
                return jsonify({
                    'error': error_msg,
                    'timestamp': datetime.now().isoformat()
                }), 400
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the API server"""
        self.logger.info(f"Starting model API on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

# Example API usage
# Create a simple preprocessor
from sklearn.preprocessing import StandardScaler

# Train a model with preprocessing
X, y = make_classification(n_samples=1000, n_features=5, random_state=42)
feature_names = [f'feature_{i}' for i in range(5)]

# Fit preprocessor
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_scaled, y)

# Create configuration
api_config = MLPipelineConfig()
api_config.set('model.version', '2.0.0')

print("Model API Setup Complete")
print("To run the API server, uncomment the following lines:")
print("# api = ModelAPI(model, scaler, api_config)")
print("# api.run(debug=True)")

# Example client code for testing
def test_api_client():
    """Example client code for testing the API"""
    import requests
    
    # Test data
    test_data = {
        'feature_0': 1.5,
        'feature_1': -0.5,
        'feature_2': 2.0,
        'feature_3': 0.0,
        'feature_4': -1.0
    }
    
    # Single prediction
    response = requests.post('http://localhost:5000/predict', json=test_data)
    print("Single prediction response:", response.json())
    
    # Batch prediction
    batch_data = {
        'instances': [test_data for _ in range(5)],
        'batch_size': 2
    }
    
    response = requests.post('http://localhost:5000/batch_predict', json=batch_data)
    print("Batch prediction response:", response.json())
    
    # Model info
    response = requests.get('http://localhost:5000/model_info')
    print("Model info:", response.json())
    
    # Health check
    response = requests.get('http://localhost:5000/health')
    print("Health check:", response.json())

print("\nExample client test function available: test_api_client()")
```

### 2.2 Containerization with Docker 🟡

```python
def generate_docker_files(project_name: str, model_path: str, 
                         requirements: List[str] = None):
    """Generate Docker files for model deployment"""
    
    # Default requirements
    if requirements is None:
        requirements = [
            'flask==2.3.2',
            'scikit-learn==1.3.0',
            'pandas==2.0.3',
            'numpy==1.24.3',
            'joblib==1.3.1'
        ]
    
    # Dockerfile content
    dockerfile_content = f"""
# Use official Python runtime as base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:5000/health || exit 1

# Run the application
CMD ["python", "app.py"]
"""

    # requirements.txt content
    requirements_content = '\n'.join(requirements)
    
    # docker-compose.yml content
    docker_compose_content = f"""
version: '3.8'

services:
  {project_name}:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
      - MODEL_PATH=/app/models/{model_path}
    volumes:
      - ./models:/app/models:ro
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
        reservations:
          memory: 512M
          cpus: '0.25'

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - {project_name}
    restart: unless-stopped

volumes:
  model_data:
  log_data:
"""

    # nginx.conf content for load balancing
    nginx_config = f"""
events {{
    worker_connections 1024;
}}

http {{
    upstream {project_name}_backend {{
        server {project_name}:5000;
    }}

    server {{
        listen 80;
        
        location / {{
            proxy_pass http://{project_name}_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Timeouts
            proxy_connect_timeout 30s;
            proxy_send_timeout 30s;
            proxy_read_timeout 30s;
        }}
        
        location /health {{
            proxy_pass http://{project_name}_backend/health;
            access_log off;
        }}
    }}
}}
"""

    # App.py content
    app_content = f"""
import os
import sys
from {project_name}_api import ModelAPI
from model_registry import ModelRegistry
from sklearn.externals import joblib

def load_model():
    \"\"\"Load model from registry or file\"\"\"
    model_path = os.environ.get('MODEL_PATH', 'models/model.pkl')
    
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        # Try loading from registry
        registry = ModelRegistry()
        model, _ = registry.load_model('{project_name}')
        return model

if __name__ == '__main__':
    # Load model
    model = load_model()
    
    # Create API
    api = ModelAPI(model)
    
    # Run with production settings
    port = int(os.environ.get('PORT', 5000))
    api.run(host='0.0.0.0', port=port, debug=False)
"""

    # Build script
    build_script = f"""#!/bin/bash

# Build Docker image
echo "Building Docker image for {project_name}..."
docker build -t {project_name}:latest .

# Tag for registry
docker tag {project_name}:latest localhost:5000/{project_name}:latest

# Run with docker-compose
echo "Starting services with docker-compose..."
docker-compose up -d

echo "Deployment complete!"
echo "API available at: http://localhost:5000"
echo "Health check: http://localhost:5000/health"

# Show logs
docker-compose logs -f
"""

    # Create files
    files_to_create = {
        'Dockerfile': dockerfile_content,
        'requirements.txt': requirements_content,
        'docker-compose.yml': docker_compose_content,
        'nginx.conf': nginx_config,
        'app.py': app_content,
        'build.sh': build_script
    }
    
    # Create deployment directory
    deployment_dir = Path(f"deployment_{project_name}")
    deployment_dir.mkdir(exist_ok=True)
    
    for filename, content in files_to_create.items():
        file_path = deployment_dir / filename
        with open(file_path, 'w') as f:
            f.write(content.strip())
        
        # Make shell scripts executable
        if filename.endswith('.sh'):
            os.chmod(file_path, 0o755)
    
    print(f"Docker deployment files created in: {deployment_dir}")
    print("\nTo deploy:")
    print(f"1. cd {deployment_dir}")
    print("2. ./build.sh")
    
    return deployment_dir

# Generate deployment files
deployment_path = generate_docker_files(
    project_name='customer_churn_model',
    model_path='customer_churn_predictor_1.0.0.pkl'
)
```

---

## 3. Model Monitoring and Alerting

### 3.1 Performance Monitoring 🟡

```python
import sqlite3
from collections import defaultdict, deque
import threading
import time

class ModelMonitor:
    """Comprehensive model monitoring system"""
    
    def __init__(self, db_path: str = "monitoring.db", window_size: int = 1000):
        self.db_path = db_path
        self.window_size = window_size
        self.metrics_cache = deque(maxlen=window_size)
        self.alerts = []
        self.thresholds = {
            'accuracy': 0.8,
            'precision': 0.7,
            'recall': 0.7,
            'response_time': 1.0,  # seconds
            'error_rate': 0.05
        }
        
        self._init_database()
        self._start_monitoring_thread()
    
    def _init_database(self):
        """Initialize monitoring database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                input_data TEXT,
                prediction TEXT,
                probability REAL,
                response_time REAL,
                model_version TEXT,
                user_id TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER,
                actual_value TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                feedback_type TEXT,
                FOREIGN KEY (prediction_id) REFERENCES predictions (id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                alert_type TEXT,
                message TEXT,
                severity TEXT,
                resolved BOOLEAN DEFAULT FALSE
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_prediction(self, input_data: Dict, prediction: Any, 
                      probability: float = None, response_time: float = None,
                      model_version: str = None, user_id: str = None) -> int:
        """Log a prediction"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions 
            (input_data, prediction, probability, response_time, model_version, user_id)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            json.dumps(input_data), 
            str(prediction), 
            probability, 
            response_time,
            model_version,
            user_id
        ))
        
        prediction_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # Update metrics cache
        metric = {
            'timestamp': datetime.now(),
            'response_time': response_time,
            'prediction_id': prediction_id
        }
        
        self.metrics_cache.append(metric)
        
        return prediction_id
    
    def log_feedback(self, prediction_id: int, actual_value: Any, 
                    feedback_type: str = 'ground_truth'):
        """Log feedback for a prediction"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO feedback (prediction_id, actual_value, feedback_type)
            VALUES (?, ?, ?)
        ''', (prediction_id, str(actual_value), feedback_type))
        
        conn.commit()
        conn.close()
        
        # Check for performance degradation
        self._check_performance_alerts()
    
    def calculate_metrics(self, hours: int = 24) -> Dict:
        """Calculate performance metrics for recent period"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get recent predictions with feedback
        cursor.execute('''
            SELECT p.prediction, f.actual_value, p.response_time
            FROM predictions p
            JOIN feedback f ON p.id = f.prediction_id
            WHERE p.timestamp > datetime('now', '-{} hours')
        '''.format(hours))
        
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            return {'error': 'No data available for metrics calculation'}
        
        # Calculate metrics
        predictions = [r[0] for r in results]
        actuals = [r[1] for r in results]
        response_times = [r[2] for r in results if r[2] is not None]
        
        # Accuracy (assuming binary classification)
        try:
            correct = sum(1 for p, a in zip(predictions, actuals) if p == a)
            accuracy = correct / len(predictions)
        except:
            accuracy = None
        
        # Response time metrics
        avg_response_time = np.mean(response_times) if response_times else None
        p95_response_time = np.percentile(response_times, 95) if response_times else None
        
        # Error rate (from logs)
        cursor = sqlite3.connect(self.db_path).cursor()
        cursor.execute('''
            SELECT COUNT(*) FROM predictions 
            WHERE timestamp > datetime('now', '-{} hours')
        '''.format(hours))
        total_requests = cursor.fetchone()[0]
        
        error_rate = max(0, (total_requests - len(results)) / total_requests) if total_requests > 0 else 0
        
        metrics = {
            'accuracy': accuracy,
            'avg_response_time': avg_response_time,
            'p95_response_time': p95_response_time,
            'error_rate': error_rate,
            'total_predictions': len(results),
            'total_requests': total_requests,
            'calculation_period_hours': hours,
            'timestamp': datetime.now().isoformat()
        }
        
        return metrics
    
    def _check_performance_alerts(self):
        """Check for performance issues and generate alerts"""
        
        try:
            metrics = self.calculate_metrics(hours=1)  # Last hour
            
            alerts_to_generate = []
            
            # Check accuracy
            if metrics.get('accuracy') and metrics['accuracy'] < self.thresholds['accuracy']:
                alerts_to_generate.append({
                    'type': 'accuracy_degradation',
                    'message': f"Accuracy dropped to {metrics['accuracy']:.3f} (threshold: {self.thresholds['accuracy']})",
                    'severity': 'high'
                })
            
            # Check response time
            if metrics.get('p95_response_time') and metrics['p95_response_time'] > self.thresholds['response_time']:
                alerts_to_generate.append({
                    'type': 'high_latency',
                    'message': f"95th percentile response time: {metrics['p95_response_time']:.3f}s (threshold: {self.thresholds['response_time']}s)",
                    'severity': 'medium'
                })
            
            # Check error rate
            if metrics['error_rate'] > self.thresholds['error_rate']:
                alerts_to_generate.append({
                    'type': 'high_error_rate',
                    'message': f"Error rate: {metrics['error_rate']:.3f} (threshold: {self.thresholds['error_rate']})",
                    'severity': 'high'
                })
            
            # Generate alerts
            for alert in alerts_to_generate:
                self._generate_alert(alert['type'], alert['message'], alert['severity'])
            
        except Exception as e:
            logger.error(f"Error checking performance alerts: {e}")
    
    def _generate_alert(self, alert_type: str, message: str, severity: str = 'medium'):
        """Generate an alert"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO alerts (alert_type, message, severity)
            VALUES (?, ?, ?)
        ''', (alert_type, message, severity))
        
        conn.commit()
        conn.close()
        
        # Add to in-memory alerts
        alert = {
            'timestamp': datetime.now(),
            'type': alert_type,
            'message': message,
            'severity': severity
        }
        
        self.alerts.append(alert)
        logger.warning(f"ALERT [{severity}] {alert_type}: {message}")
    
    def _start_monitoring_thread(self):
        """Start background monitoring thread"""
        
        def monitor_loop():
            while True:
                try:
                    self._check_performance_alerts()
                    time.sleep(300)  # Check every 5 minutes
                except Exception as e:
                    logger.error(f"Monitoring thread error: {e}")
                    time.sleep(60)  # Wait 1 minute before retry
        
        thread = threading.Thread(target=monitor_loop, daemon=True)
        thread.start()
    
    def get_dashboard_data(self) -> Dict:
        """Get data for monitoring dashboard"""
        
        # Recent metrics
        recent_metrics = self.calculate_metrics(hours=24)
        
        # Active alerts
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT alert_type, message, severity, timestamp
            FROM alerts
            WHERE resolved = FALSE
            ORDER BY timestamp DESC
            LIMIT 10
        ''')
        
        active_alerts = [
            {
                'type': row[0],
                'message': row[1],
                'severity': row[2],
                'timestamp': row[3]
            }
            for row in cursor.fetchall()
        ]
        
        # Request volume (last 24 hours)
        cursor.execute('''
            SELECT 
                strftime('%H', timestamp) as hour,
                COUNT(*) as requests
            FROM predictions
            WHERE timestamp > datetime('now', '-24 hours')
            GROUP BY hour
            ORDER BY hour
        ''')
        
        hourly_requests = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            'metrics': recent_metrics,
            'alerts': active_alerts,
            'hourly_requests': hourly_requests,
            'dashboard_updated': datetime.now().isoformat()
        }

# Example usage
monitor = ModelMonitor()

# Simulate some predictions and feedback
for i in range(100):
    # Simulate prediction
    input_data = {'feature1': np.random.randn(), 'feature2': np.random.randn()}
    prediction = np.random.choice([0, 1])
    probability = np.random.random()
    response_time = np.random.uniform(0.1, 0.5)
    
    pred_id = monitor.log_prediction(
        input_data=input_data,
        prediction=prediction,
        probability=probability,
        response_time=response_time,
        model_version='1.0.0'
    )
    
    # Simulate feedback (sometimes wrong to trigger alerts)
    if np.random.random() < 0.3:  # 30% feedback rate
        actual = prediction if np.random.random() < 0.85 else (1 - prediction)  # 85% accuracy
        monitor.log_feedback(pred_id, actual)

# Get dashboard data
dashboard_data = monitor.get_dashboard_data()
print("Monitoring Dashboard Data:")
print(json.dumps(dashboard_data, indent=2, default=str))
```

### 3.2 Data Drift Detection 🔴

```python
from scipy.stats import ks_2samp, chi2_contingency
from sklearn.metrics import mutual_info_score

class DataDriftDetector:
    """Advanced data drift detection system"""
    
    def __init__(self, reference_data: pd.DataFrame, 
                 significance_level: float = 0.05):
        self.reference_data = reference_data
        self.significance_level = significance_level
        self.drift_history = []
        
        # Calculate reference statistics
        self.reference_stats = self._calculate_statistics(reference_data)
    
    def _calculate_statistics(self, data: pd.DataFrame) -> Dict:
        """Calculate comprehensive statistics for data"""
        
        stats = {
            'numerical': {},
            'categorical': {},
            'correlations': {}
        }
        
        # Numerical features
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            stats['numerical'][col] = {
                'mean': data[col].mean(),
                'std': data[col].std(),
                'min': data[col].min(),
                'max': data[col].max(),
                'quantiles': data[col].quantile([0.25, 0.5, 0.75]).to_dict(),
                'distribution': data[col].values  # Store for KS test
            }
        
        # Categorical features
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            value_counts = data[col].value_counts(normalize=True)
            stats['categorical'][col] = {
                'categories': list(value_counts.index),
                'proportions': value_counts.to_dict(),
                'num_categories': len(value_counts)
            }
        
        # Feature correlations
        if len(numerical_cols) > 1:
            corr_matrix = data[numerical_cols].corr()
            stats['correlations'] = corr_matrix.to_dict()
        
        return stats
    
    def detect_drift(self, new_data: pd.DataFrame, 
                    detailed: bool = True) -> Dict:
        """Detect data drift between reference and new data"""
        
        drift_results = {
            'timestamp': datetime.now().isoformat(),
            'overall_drift': False,
            'drift_score': 0.0,
            'feature_drifts': {},
            'summary': {}
        }
        
        # Calculate new data statistics
        new_stats = self._calculate_statistics(new_data)
        
        # Check numerical features
        numerical_drifts = []
        for col in self.reference_stats['numerical'].keys():
            if col not in new_data.columns:
                continue
            
            # Kolmogorov-Smirnov test
            ref_dist = self.reference_stats['numerical'][col]['distribution']
            new_dist = new_data[col].values
            
            ks_stat, ks_p_value = ks_2samp(ref_dist, new_dist)
            
            # Statistical moments comparison
            ref_mean = self.reference_stats['numerical'][col]['mean']
            new_mean = new_stats['numerical'][col]['mean']
            ref_std = self.reference_stats['numerical'][col]['std']
            new_std = new_stats['numerical'][col]['std']
            
            mean_change = abs(new_mean - ref_mean) / (ref_std + 1e-8)
            std_change = abs(new_std - ref_std) / (ref_std + 1e-8)
            
            # Population Stability Index (PSI)
            psi_score = self._calculate_psi(ref_dist, new_dist)
            
            feature_drift = {
                'feature': col,
                'drift_detected': ks_p_value < self.significance_level,
                'ks_statistic': ks_stat,
                'ks_p_value': ks_p_value,
                'mean_change': mean_change,
                'std_change': std_change,
                'psi_score': psi_score,
                'drift_severity': 'high' if psi_score > 0.25 else ('medium' if psi_score > 0.1 else 'low')
            }
            
            drift_results['feature_drifts'][col] = feature_drift
            
            if feature_drift['drift_detected']:
                numerical_drifts.append(col)
        
        # Check categorical features
        categorical_drifts = []
        for col in self.reference_stats['categorical'].keys():
            if col not in new_data.columns:
                continue
            
            ref_props = self.reference_stats['categorical'][col]['proportions']
            new_props = new_data[col].value_counts(normalize=True).to_dict()
            
            # Chi-square test for categorical distribution
            all_categories = set(ref_props.keys()) | set(new_props.keys())
            ref_counts = [ref_props.get(cat, 0) * len(self.reference_data) for cat in all_categories]
            new_counts = [new_props.get(cat, 0) * len(new_data) for cat in all_categories]
            
            # Filter out categories with zero counts in both
            non_zero_mask = [(r > 0 or n > 0) for r, n in zip(ref_counts, new_counts)]
            ref_counts = [r for r, mask in zip(ref_counts, non_zero_mask) if mask]
            new_counts = [n for n, mask in zip(new_counts, non_zero_mask) if mask]
            
            if len(ref_counts) > 1:
                try:
                    chi2_stat, chi2_p_value = chi2_contingency([ref_counts, new_counts])[:2]
                except:
                    chi2_stat, chi2_p_value = 0, 1
            else:
                chi2_stat, chi2_p_value = 0, 1
            
            # Calculate categorical PSI
            cat_psi = self._calculate_categorical_psi(ref_props, new_props)
            
            feature_drift = {
                'feature': col,
                'drift_detected': chi2_p_value < self.significance_level,
                'chi2_statistic': chi2_stat,
                'chi2_p_value': chi2_p_value,
                'psi_score': cat_psi,
                'new_categories': list(set(new_props.keys()) - set(ref_props.keys())),
                'missing_categories': list(set(ref_props.keys()) - set(new_props.keys())),
                'drift_severity': 'high' if cat_psi > 0.25 else ('medium' if cat_psi > 0.1 else 'low')
            }
            
            drift_results['feature_drifts'][col] = feature_drift
            
            if feature_drift['drift_detected']:
                categorical_drifts.append(col)
        
        # Overall drift assessment
        total_features = len(drift_results['feature_drifts'])
        drifted_features = len([f for f in drift_results['feature_drifts'].values() 
                              if f['drift_detected']])
        
        drift_results['overall_drift'] = drifted_features > 0
        drift_results['drift_score'] = drifted_features / total_features if total_features > 0 else 0
        
        # Summary
        drift_results['summary'] = {
            'total_features': total_features,
            'drifted_features': drifted_features,
            'drift_percentage': drift_results['drift_score'] * 100,
            'numerical_drifts': numerical_drifts,
            'categorical_drifts': categorical_drifts,
            'high_severity_drifts': [
                f['feature'] for f in drift_results['feature_drifts'].values()
                if f['drift_severity'] == 'high'
            ]
        }
        
        # Store in history
        self.drift_history.append(drift_results)
        
        return drift_results
    
    def _calculate_psi(self, reference: np.ndarray, new: np.ndarray, 
                      bins: int = 10) -> float:
        """Calculate Population Stability Index for numerical features"""
        
        # Create bins based on reference data
        bin_edges = np.histogram_bin_edges(reference, bins=bins)
        
        # Calculate distributions
        ref_hist, _ = np.histogram(reference, bins=bin_edges)
        new_hist, _ = np.histogram(new, bins=bin_edges)
        
        # Convert to proportions
        ref_props = ref_hist / len(reference)
        new_props = new_hist / len(new)
        
        # Calculate PSI
        psi = 0
        for ref_prop, new_prop in zip(ref_props, new_props):
            if ref_prop > 0 and new_prop > 0:
                psi += (new_prop - ref_prop) * np.log(new_prop / ref_prop)
        
        return psi
    
    def _calculate_categorical_psi(self, ref_props: Dict, new_props: Dict) -> float:
        """Calculate PSI for categorical features"""
        
        all_categories = set(ref_props.keys()) | set(new_props.keys())
        
        psi = 0
        for category in all_categories:
            ref_prop = ref_props.get(category, 1e-6)  # Small epsilon for missing categories
            new_prop = new_props.get(category, 1e-6)
            
            psi += (new_prop - ref_prop) * np.log(new_prop / ref_prop)
        
        return psi
    
    def visualize_drift(self, drift_results: Dict):
        """Visualize data drift results"""
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Overall drift summary
        summary = drift_results['summary']
        
        ax1 = axes[0, 0]
        categories = ['No Drift', 'Drifted']
        values = [summary['total_features'] - summary['drifted_features'], 
                 summary['drifted_features']]
        colors = ['green', 'red']
        
        ax1.pie(values, labels=categories, colors=colors, autopct='%1.1f%%')
        ax1.set_title(f"Feature Drift Overview\n({summary['drift_percentage']:.1f}% features drifted)")
        
        # Drift severity
        ax2 = axes[0, 1]
        severity_counts = {'low': 0, 'medium': 0, 'high': 0}
        
        for feature_drift in drift_results['feature_drifts'].values():
            severity_counts[feature_drift['drift_severity']] += 1
        
        ax2.bar(severity_counts.keys(), severity_counts.values(), 
                color=['green', 'orange', 'red'], alpha=0.7)
        ax2.set_title('Drift Severity Distribution')
        ax2.set_ylabel('Number of Features')
        
        # PSI scores
        ax3 = axes[1, 0]
        features = list(drift_results['feature_drifts'].keys())
        psi_scores = [drift_results['feature_drifts'][f]['psi_score'] 
                     for f in features]
        
        colors = ['red' if score > 0.25 else 'orange' if score > 0.1 else 'green' 
                 for score in psi_scores]
        
        bars = ax3.bar(range(len(features)), psi_scores, color=colors, alpha=0.7)
        ax3.set_title('PSI Scores by Feature')
        ax3.set_ylabel('PSI Score')
        ax3.set_xticks(range(len(features)))
        ax3.set_xticklabels(features, rotation=45, ha='right')
        
        # Add PSI threshold lines
        ax3.axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='Medium threshold')
        ax3.axhline(y=0.25, color='red', linestyle='--', alpha=0.7, label='High threshold')
        ax3.legend()
        
        # P-values heatmap
        ax4 = axes[1, 1]
        p_values = []
        feature_names = []
        
        for feature, drift_info in drift_results['feature_drifts'].items():
            feature_names.append(feature)
            if 'ks_p_value' in drift_info:
                p_values.append(drift_info['ks_p_value'])
            elif 'chi2_p_value' in drift_info:
                p_values.append(drift_info['chi2_p_value'])
            else:
                p_values.append(1.0)
        
        # Create color map for p-values
        colors = ['red' if p < 0.05 else 'orange' if p < 0.1 else 'green' 
                 for p in p_values]
        
        bars = ax4.bar(range(len(feature_names)), p_values, color=colors, alpha=0.7)
        ax4.set_title('Statistical Test P-values')
        ax4.set_ylabel('P-value')
        ax4.set_xticks(range(len(feature_names)))
        ax4.set_xticklabels(feature_names, rotation=45, ha='right')
        ax4.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='Significance threshold')
        ax4.legend()
        
        plt.tight_layout()
        plt.show()

# Example usage
# Create reference data
np.random.seed(42)
reference_data = pd.DataFrame({
    'feature1': np.random.normal(0, 1, 1000),
    'feature2': np.random.normal(5, 2, 1000),
    'category1': np.random.choice(['A', 'B', 'C'], 1000, p=[0.5, 0.3, 0.2]),
    'category2': np.random.choice(['X', 'Y'], 1000, p=[0.7, 0.3])
})

# Create new data with drift
new_data = pd.DataFrame({
    'feature1': np.random.normal(0.5, 1.2, 500),  # Mean and variance shift
    'feature2': np.random.normal(5, 2, 500),      # No drift
    'category1': np.random.choice(['A', 'B', 'C'], 500, p=[0.3, 0.4, 0.3]),  # Distribution shift
    'category2': np.random.choice(['X', 'Y'], 500, p=[0.7, 0.3])  # No drift
})

# Initialize drift detector
drift_detector = DataDriftDetector(reference_data)

# Detect drift
drift_results = drift_detector.detect_drift(new_data)

print("Data Drift Detection Results:")
print(json.dumps(drift_results['summary'], indent=2))

# Visualize results
drift_detector.visualize_drift(drift_results)
```

---

## 🎯 Key Takeaways

### MLOps Best Practices:

#### Configuration Management:
- **Version control**: Track all configs, code, and models
- **Environment isolation**: Use containers and virtual environments
- **Secret management**: Secure API keys and credentials
- **Feature flags**: Enable/disable features without deployment

#### Model Deployment:
- **Blue-green deployments**: Zero-downtime updates
- **Canary releases**: Gradual rollout to catch issues early
- **A/B testing**: Compare model versions in production
- **Rollback strategy**: Quick reversion for failed deployments

#### Monitoring Strategy:
- **Performance monitoring**: Track accuracy, latency, throughput
- **Data drift detection**: Monitor input distribution changes
- **Infrastructure monitoring**: CPU, memory, disk usage
- **Business metrics**: Track impact on KPIs

#### Production Considerations:
- **Scalability**: Handle varying load patterns
- **Fault tolerance**: Graceful degradation and error handling
- **Security**: Input validation, authentication, audit logs
- **Compliance**: GDPR, HIPAA, industry regulations

### Common Pitfalls:
1. **Insufficient monitoring**: Not tracking the right metrics
2. **Model decay**: Ignoring performance degradation over time
3. **Poor versioning**: Inability to reproduce results
4. **Inadequate testing**: Deploying without proper validation
5. **Technical debt**: Quick fixes that compound over time

---

## 📚 Next Steps

Continue your journey with:
- **[Ethics & Bias in AI](14_Ethics_Bias_AI.md)** - Responsible AI development and fairness
- **[Tools & Frameworks](15_Tools_Frameworks.md)** - Complete ML development environment

---

*Next: [Ethics & Bias in AI →](14_Ethics_Bias_AI.md)*
