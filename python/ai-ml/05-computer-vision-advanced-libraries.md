# Computer Vision and Advanced AI Libraries

## Table of Contents
1. [OpenCV - Computer Vision](#opencv---computer-vision)
2. [Pillow (PIL) - Image Processing](#pillow-pil---image-processing)
3. [Streamlit - ML Web Apps](#streamlit---ml-web-apps)
4. [MLflow - ML Lifecycle Management](#mlflow---ml-lifecycle-management)
5. [Optuna - Hyperparameter Optimization](#optuna---hyperparameter-optimization)
6. [SHAP - Model Interpretability](#shap---model-interpretability)

## OpenCV - Computer Vision

OpenCV (Open Source Computer Vision Library) is a powerful library for computer vision, image processing, and machine learning.

### 1. Installation and Basic Setup
```python
# Installation
pip install opencv-python opencv-contrib-python

import cv2
import numpy as np
import matplotlib.pyplot as plt

print(f"OpenCV version: {cv2.__version__}")

# Helper function to display images in Jupyter/Python
def display_image(image, title="Image", figsize=(10, 6)):
    """Display image using matplotlib"""
    plt.figure(figsize=figsize)
    if len(image.shape) == 3:
        # Color image (BGR to RGB conversion for matplotlib)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        # Grayscale image
        plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Create sample images for demonstration
def create_sample_image():
    """Create a sample image for testing"""
    # Create a 300x300 color image
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    
    # Draw some shapes
    cv2.rectangle(img, (50, 50), (150, 150), (255, 0, 0), -1)  # Blue rectangle
    cv2.circle(img, (200, 100), 50, (0, 255, 0), -1)  # Green circle
    cv2.line(img, (0, 200), (300, 250), (0, 0, 255), 5)  # Red line
    
    return img

sample_img = create_sample_image()
print("Sample image created successfully")
```

### 2. Basic Image Operations
```python
# Image loading, saving, and basic operations
def basic_image_operations():
    """Demonstrate basic image operations"""
    
    # Create sample image
    img = create_sample_image()
    
    print("=== Basic Image Operations ===")
    print(f"Image shape: {img.shape}")
    print(f"Image data type: {img.dtype}")
    print(f"Image size: {img.size}")
    
    # Convert color spaces
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    print(f"Grayscale shape: {gray.shape}")
    print(f"HSV shape: {hsv.shape}")
    
    # Basic transformations
    resized = cv2.resize(img, (150, 150))
    rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    flipped = cv2.flip(img, 1)  # Horizontal flip
    
    # Cropping
    cropped = img[50:200, 50:200]
    
    return {
        'original': img,
        'grayscale': gray,
        'hsv': hsv,
        'resized': resized,
        'rotated': rotated,
        'flipped': flipped,
        'cropped': cropped
    }

images = basic_image_operations()

# Image filtering and enhancement
def image_filtering(img):
    """Apply various filters to image"""
    
    # Blur filters
    gaussian_blur = cv2.GaussianBlur(img, (15, 15), 0)
    median_blur = cv2.medianBlur(img, 15)
    bilateral_filter = cv2.bilateralFilter(img, 9, 75, 75)
    
    # Edge detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges_canny = cv2.Canny(gray, 50, 150)
    
    # Morphological operations
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(gray, kernel, iterations=1)
    dilation = cv2.dilate(gray, kernel, iterations=1)
    opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    
    return {
        'gaussian_blur': gaussian_blur,
        'median_blur': median_blur,
        'bilateral_filter': bilateral_filter,
        'edges': edges_canny,
        'erosion': erosion,
        'dilation': dilation,
        'opening': opening,
        'closing': closing
    }

filtered_images = image_filtering(sample_img)
print("Image filtering operations completed")
```

### 3. Feature Detection and Matching
```python
# Feature detection using various algorithms
def feature_detection(img):
    """Detect features using different algorithms"""
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # SIFT (Scale-Invariant Feature Transform)
    sift = cv2.SIFT_create()
    keypoints_sift, descriptors_sift = sift.detectAndCompute(gray, None)
    img_sift = cv2.drawKeypoints(img, keypoints_sift, None, 
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # ORB (Oriented FAST and Rotated BRIEF)
    orb = cv2.ORB_create()
    keypoints_orb, descriptors_orb = orb.detectAndCompute(gray, None)
    img_orb = cv2.drawKeypoints(img, keypoints_orb, None, color=(0, 255, 0))
    
    # Harris Corner Detection
    harris_corners = cv2.cornerHarris(gray, 2, 3, 0.04)
    img_harris = img.copy()
    img_harris[harris_corners > 0.01 * harris_corners.max()] = [0, 0, 255]
    
    print(f"SIFT keypoints: {len(keypoints_sift)}")
    print(f"ORB keypoints: {len(keypoints_orb)}")
    
    return {
        'sift': img_sift,
        'orb': img_orb,
        'harris': img_harris,
        'sift_keypoints': keypoints_sift,
        'orb_keypoints': keypoints_orb,
        'sift_descriptors': descriptors_sift,
        'orb_descriptors': descriptors_orb
    }

features = feature_detection(sample_img)

# Feature matching between two images
def feature_matching(img1, img2):
    """Match features between two images"""
    
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Detect ORB features
    orb = cv2.ORB_create()
    kp1, desc1 = orb.detectAndCompute(gray1, None)
    kp2, desc2 = orb.detectAndCompute(gray2, None)
    
    if desc1 is not None and desc2 is not None:
        # Match features using Brute Force matcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(desc1, desc2)
        
        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Draw matches
        img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None, 
                                     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        print(f"Number of matches found: {len(matches)}")
        return img_matches, matches
    else:
        print("No descriptors found")
        return None, []

# Create a slightly modified version of the image for matching
modified_img = cv2.resize(sample_img, (250, 250))
modified_img = cv2.rotate(modified_img, cv2.ROTATE_90_CLOCKWISE)

matched_img, matches = feature_matching(sample_img, modified_img)
```

### 4. Object Detection and Tracking
```python
# Simple object detection using contours
def detect_objects_contours(img):
    """Detect objects using contour detection"""
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours
    img_contours = img.copy()
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)
    
    # Analyze contours
    objects_info = []
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img_contours, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Centroid
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(img_contours, (cx, cy), 5, (0, 0, 255), -1)
        
        objects_info.append({
            'contour_id': i,
            'area': area,
            'perimeter': perimeter,
            'bounding_box': (x, y, w, h),
            'centroid': (cx, cy) if M["m00"] != 0 else None
        })
    
    print(f"Number of objects detected: {len(contours)}")
    for obj in objects_info:
        print(f"Object {obj['contour_id']}: Area={obj['area']:.1f}, "
              f"Perimeter={obj['perimeter']:.1f}")
    
    return img_contours, objects_info

contour_result, objects = detect_objects_contours(sample_img)

# Color-based object detection
def detect_colored_objects(img, color_range):
    """Detect objects based on color"""
    
    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Create mask for the color range
    lower_bound = np.array(color_range['lower'])
    upper_bound = np.array(color_range['upper'])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw bounding boxes around detected objects
    result_img = img.copy()
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filter small objects
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(result_img, 'Object', (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return result_img, mask, len(contours)

# Define color range for blue objects (in HSV)
blue_range = {
    'lower': [100, 50, 50],
    'upper': [130, 255, 255]
}

blue_detection, blue_mask, blue_count = detect_colored_objects(sample_img, blue_range)
print(f"Blue objects detected: {blue_count}")
```

## Pillow (PIL) - Image Processing

Pillow is a user-friendly image processing library, successor to PIL (Python Imaging Library).

### 1. Installation and Basic Operations
```python
# Installation
pip install Pillow

from PIL import Image, ImageFilter, ImageEnhance, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt

print(f"Pillow version: {Image.__version__}")

# Create sample image with PIL
def create_pil_sample():
    """Create a sample image using PIL"""
    # Create a new image
    img = Image.new('RGB', (300, 200), color='white')
    
    # Draw on the image
    draw = ImageDraw.Draw(img)
    
    # Draw shapes
    draw.rectangle([50, 50, 150, 100], fill='red', outline='black', width=2)
    draw.ellipse([180, 60, 250, 120], fill='blue', outline='black', width=2)
    draw.line([0, 150, 300, 150], fill='green', width=3)
    
    # Add text
    draw.text((10, 10), "Sample Image", fill='black')
    
    return img

pil_sample = create_pil_sample()

# Basic image operations
def pil_basic_operations(img):
    """Demonstrate basic PIL operations"""
    
    print("=== Basic PIL Operations ===")
    print(f"Image size: {img.size}")
    print(f"Image mode: {img.mode}")
    print(f"Image format: {img.format}")
    
    # Resize
    resized = img.resize((150, 100))
    
    # Rotate
    rotated = img.rotate(45, expand=True)
    
    # Crop
    cropped = img.crop((50, 50, 200, 150))
    
    # Flip
    flipped_h = img.transpose(Image.FLIP_LEFT_RIGHT)
    flipped_v = img.transpose(Image.FLIP_TOP_BOTTOM)
    
    # Convert modes
    grayscale = img.convert('L')
    
    return {
        'original': img,
        'resized': resized,
        'rotated': rotated,
        'cropped': cropped,
        'flipped_h': flipped_h,
        'flipped_v': flipped_v,
        'grayscale': grayscale
    }

pil_operations = pil_basic_operations(pil_sample)
```

## Streamlit - ML Web Apps

Streamlit makes it easy to create web applications for machine learning projects.

### 1. Basic Streamlit App Structure
```python
# Installation
pip install streamlit

# Create a file called ml_app.py with the following content:

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px
import plotly.graph_objects as go

# Configure the page
st.set_page_config(
    page_title="ML Demo App",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("🤖 Machine Learning Demo App")
st.markdown("""
This app demonstrates various machine learning concepts and visualizations.
Use the sidebar to navigate between different sections.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox(
    "Choose the app mode",
    ["Dataset Explorer", "Model Training", "Data Visualization", "Prediction"]
)

# Load sample datasets
@st.cache_data
def load_datasets():
    """Load and cache datasets"""
    iris = load_iris()
    wine = load_wine()
    
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_df['target'] = iris.target
    iris_df['species'] = iris_df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    
    wine_df = pd.DataFrame(wine.data, columns=wine.feature_names)
    wine_df['target'] = wine.target
    wine_df['wine_class'] = wine_df['target'].map({0: 'class_0', 1: 'class_1', 2: 'class_2'})
    
    return iris_df, wine_df

iris_df, wine_df = load_datasets()

# Dataset Explorer
if app_mode == "Dataset Explorer":
    st.header("📊 Dataset Explorer")
    
    # Dataset selection
    dataset_choice = st.selectbox("Select Dataset", ["Iris", "Wine"])
    
    if dataset_choice == "Iris":
        df = iris_df
        target_col = 'species'
    else:
        df = wine_df
        target_col = 'wine_class'
    
    # Display dataset info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Number of Samples", df.shape[0])
    
    with col2:
        st.metric("Number of Features", df.shape[1] - 2)  # Excluding target columns
    
    with col3:
        st.metric("Number of Classes", df[target_col].nunique())
    
    # Display raw data
    st.subheader("Raw Data")
    st.dataframe(df.head(20))
    
    # Statistical summary
    st.subheader("Statistical Summary")
    st.dataframe(df.describe())

# Run the app with: streamlit run ml_app.py
```

## MLflow - ML Lifecycle Management

MLflow is an open-source platform for managing the machine learning lifecycle.

### 1. MLflow Tracking
```python
# Installation
pip install mlflow

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt

# Set up MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")  # Local SQLite database
mlflow.set_experiment("Wine Classification Experiment")

def train_and_log_model(model, model_name, X_train, X_test, y_train, y_test, parameters=None):
    """Train a model and log it with MLflow"""
    
    with mlflow.start_run(run_name=f"{model_name}_run") as run:
        # Log parameters
        if parameters:
            mlflow.log_params(parameters)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Log metrics
        mlflow.log_metrics({
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        })
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Run ID: {run.info.run_id}")
        print(f"Model: {model_name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("-" * 50)
        
        return run.info.run_id

# Load and prepare data
wine_data = load_wine()
X, y = wine_data.data, wine_data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("=== MLflow Experiment Tracking ===")

# Experiment with different models
models_to_test = [
    {
        'model': RandomForestClassifier(n_estimators=100, random_state=42),
        'name': 'RandomForest_100',
        'params': {'n_estimators': 100, 'random_state': 42}
    },
    {
        'model': RandomForestClassifier(n_estimators=200, random_state=42),
        'name': 'RandomForest_200',
        'params': {'n_estimators': 200, 'random_state': 42}
    },
    {
        'model': LogisticRegression(random_state=42, max_iter=1000),
        'name': 'LogisticRegression',
        'params': {'random_state': 42, 'max_iter': 1000}
    }
]

run_ids = []
for model_config in models_to_test:
    run_id = train_and_log_model(
        model_config['model'],
        model_config['name'],
        X_train, X_test, y_train, y_test,
        model_config['params']
    )
    run_ids.append(run_id)

# To start MLflow UI, run: mlflow ui --host 127.0.0.1 --port 5000
```

## Optuna - Hyperparameter Optimization

Optuna is an automatic hyperparameter optimization software framework.

### 1. Basic Optimization
```python
# Installation
pip install optuna

import optuna
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Load data
wine_data = load_wine()
X, y = wine_data.data, wine_data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define objective function for Random Forest
def objective_rf(trial):
    """Objective function for Random Forest optimization"""
    
    # Suggest hyperparameters
    n_estimators = trial.suggest_int('n_estimators', 10, 300)
    max_depth = trial.suggest_int('max_depth', 1, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    
    # Create model with suggested parameters
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42
    )
    
    # Evaluate using cross-validation
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    
    return scores.mean()

# Create study and optimize
print("=== Optuna Hyperparameter Optimization ===")
study_rf = optuna.create_study(direction='maximize', study_name='RandomForest_Optimization')
study_rf.optimize(objective_rf, n_trials=100, timeout=300)  # 100 trials or 5 minutes

print("Best trial:")
print(f"  Value: {study_rf.best_value:.4f}")
print("  Params: ")
for key, value in study_rf.best_params.items():
    print(f"    {key}: {value}")

# Train final model with best parameters
best_rf = RandomForestClassifier(**study_rf.best_params, random_state=42)
best_rf.fit(X_train, y_train)
test_score = best_rf.score(X_test, y_test)
print(f"Test accuracy with optimized parameters: {test_score:.4f}")

# Multi-objective optimization example
def objective_multi(trial):
    """Multi-objective optimization: maximize accuracy and minimize model complexity"""
    
    n_estimators = trial.suggest_int('n_estimators', 10, 200)
    max_depth = trial.suggest_int('max_depth', 1, 15)
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    
    # Calculate accuracy
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
    accuracy = scores.mean()
    
    # Calculate model complexity (negative because we want to minimize)
    complexity = -(n_estimators * max_depth) / 1000.0  # Normalized
    
    return accuracy, complexity

# Multi-objective study
study_multi = optuna.create_study(
    directions=['maximize', 'maximize'],  # Both objectives to maximize
    study_name='Multi_Objective_Optimization'
)
study_multi.optimize(objective_multi, n_trials=50)

print(f"\n=== Multi-Objective Optimization Results ===")
print(f"Number of Pareto optimal solutions: {len(study_multi.best_trials)}")

for i, trial in enumerate(study_multi.best_trials):
    print(f"Trial {i}:")
    print(f"  Accuracy: {trial.values[0]:.4f}")
    print(f"  Complexity: {trial.values[1]:.4f}")
    print(f"  Params: {trial.params}")
    print()

# Visualization of optimization history
def plot_optimization_history(study, title="Optimization History"):
    """Plot optimization history"""
    
    fig = optuna.visualization.plot_optimization_history(study)
    fig.update_layout(title=title)
    fig.show()

# plot_optimization_history(study_rf, "Random Forest Optimization History")

# Parameter importance
def plot_param_importance(study, title="Parameter Importance"):
    """Plot parameter importance"""
    
    fig = optuna.visualization.plot_param_importances(study)
    fig.update_layout(title=title)
    fig.show()

# plot_param_importance(study_rf, "Random Forest Parameter Importance")

print("\nOptimization completed!")
```

## SHAP - Model Interpretability

SHAP (SHapley Additive exPlanations) is a library for explaining machine learning model predictions.

### 1. Basic SHAP Usage
```python
# Installation
pip install shap

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine, load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# Initialize SHAP
shap.initjs()

print("=== SHAP Model Interpretability ===")

# Load wine dataset for classification
wine_data = load_wine()
X_wine = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
y_wine = wine_data.target

X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(
    X_wine, y_wine, test_size=0.3, random_state=42
)

# Train Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_wine, y_train_wine)

print(f"Wine classification accuracy: {rf_classifier.score(X_test_wine, y_test_wine):.4f}")

# SHAP Tree Explainer for tree-based models
explainer_tree = shap.TreeExplainer(rf_classifier)
shap_values_tree = explainer_tree.shap_values(X_test_wine)

print(f"SHAP values shape: {np.array(shap_values_tree).shape}")
print("SHAP values calculated for tree model")

# Feature importance using SHAP
def analyze_feature_importance(shap_values, feature_names, class_names):
    """Analyze feature importance using SHAP values"""
    
    print("\n=== Feature Importance Analysis ===")
    
    # Calculate mean absolute SHAP values for each class
    mean_shap = np.mean(np.abs(shap_values), axis=1)  # Mean across samples
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'class_0_importance': mean_shap[0],
        'class_1_importance': mean_shap[1],
        'class_2_importance': mean_shap[2],
        'overall_importance': np.mean(mean_shap, axis=0)
    }).sort_values('overall_importance', ascending=False)
    
    print("Top 10 most important features:")
    print(importance_df.head(10))
    
    return importance_df

importance_df = analyze_feature_importance(
    shap_values_tree, 
    wine_data.feature_names, 
    wine_data.target_names
)

# Individual prediction explanation
def explain_individual_prediction(explainer, shap_values, X_test, sample_idx=0):
    """Explain individual prediction"""
    
    print(f"\n=== Individual Prediction Explanation (Sample {sample_idx}) ===")
    
    # Get SHAP values for specific sample
    sample_shap = [class_shap[sample_idx] for class_shap in shap_values]
    sample_features = X_test.iloc[sample_idx]
    
    print("Feature contributions to prediction:")
    for i, (feature, value) in enumerate(sample_features.items()):
        contributions = [shap_val[i] for shap_val in sample_shap]
        print(f"{feature}: {value:.3f}")
        print(f"  Class 0 contribution: {contributions[0]:.4f}")
        print(f"  Class 1 contribution: {contributions[1]:.4f}")
        print(f"  Class 2 contribution: {contributions[2]:.4f}")
        print()

explain_individual_prediction(explainer_tree, shap_values_tree, X_test_wine, 0)

# SHAP for linear models
print("\n=== SHAP for Linear Models ===")

# Train logistic regression
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_wine, y_train_wine)

print(f"Logistic regression accuracy: {lr_model.score(X_test_wine, y_test_wine):.4f}")

# SHAP Linear Explainer
explainer_linear = shap.LinearExplainer(lr_model, X_train_wine)
shap_values_linear = explainer_linear.shap_values(X_test_wine)

print("SHAP values calculated for linear model")

# Compare explanations between models
def compare_model_explanations(shap_tree, shap_linear, feature_names):
    """Compare SHAP explanations between different models"""
    
    print("\n=== Model Explanation Comparison ===")
    
    # Calculate feature importance for both models
    tree_importance = np.mean(np.abs(np.array(shap_tree)), axis=(0, 1))
    linear_importance = np.mean(np.abs(shap_linear), axis=0)
    
    comparison_df = pd.DataFrame({
        'feature': feature_names,
        'tree_importance': tree_importance,
        'linear_importance': linear_importance
    })
    
    # Calculate correlation between explanations
    correlation = np.corrcoef(tree_importance, linear_importance)[0, 1]
    print(f"Correlation between tree and linear explanations: {correlation:.4f}")
    
    # Top features for each model
    comparison_df = comparison_df.sort_values('tree_importance', ascending=False)
    print("\nTop 5 features by tree model importance:")
    print(comparison_df[['feature', 'tree_importance']].head())
    
    comparison_df = comparison_df.sort_values('linear_importance', ascending=False)
    print("\nTop 5 features by linear model importance:")
    print(comparison_df[['feature', 'linear_importance']].head())
    
    return comparison_df

comparison = compare_model_explanations(shap_values_tree, shap_values_linear, wine_data.feature_names)

# Regression example with SHAP
try:
    from sklearn.datasets import fetch_california_housing
    housing_data = fetch_california_housing()
    X_housing = pd.DataFrame(housing_data.data, columns=housing_data.feature_names)
    y_housing = housing_data.target
    
    print("\n=== SHAP for Regression ===")
    
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_housing, y_housing, test_size=0.3, random_state=42
    )
    
    # Train regression model
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regressor.fit(X_train_reg, y_train_reg)
    
    print(f"Regression R² score: {rf_regressor.score(X_test_reg, y_test_reg):.4f}")
    
    # SHAP for regression
    explainer_reg = shap.TreeExplainer(rf_regressor)
    shap_values_reg = explainer_reg.shap_values(X_test_reg[:100])  # Subset for faster computation
    
    print("SHAP values calculated for regression model")
    
    # Feature importance for regression
    reg_importance = np.mean(np.abs(shap_values_reg), axis=0)
    reg_importance_df = pd.DataFrame({
        'feature': housing_data.feature_names,
        'importance': reg_importance
    }).sort_values('importance', ascending=False)
    
    print("\nTop features for housing price prediction:")
    print(reg_importance_df)

except ImportError:
    print("California housing dataset not available, skipping regression example")

# Summary and best practices
print("\n=== SHAP Best Practices ===")
print("""
1. Choose the right explainer:
   - TreeExplainer: For tree-based models (RF, XGBoost, LightGBM)
   - LinearExplainer: For linear models (LogisticRegression, LinearRegression)
   - KernelExplainer: Model-agnostic but slower
   - DeepExplainer: For deep learning models

2. Interpretation tips:
   - Positive SHAP values push prediction above baseline
   - Negative SHAP values push prediction below baseline
   - Magnitude indicates strength of contribution
   - Sum of SHAP values = prediction - baseline

3. Visualization options:
   - Summary plots: Overall feature importance
   - Waterfall plots: Individual predictions
   - Force plots: Interactive explanations
   - Dependence plots: Feature interactions

4. Performance considerations:
   - TreeExplainer is fastest for tree models
   - Use sampling for large datasets
   - Consider approximate methods for deep models
""")

print("SHAP analysis completed!")
```

### 2. Advanced SHAP Techniques
```python
# Advanced SHAP techniques and visualizations

def advanced_shap_analysis():
    """Advanced SHAP analysis techniques"""
    
    print("=== Advanced SHAP Analysis ===")
    
    # Feature interaction detection
    def detect_feature_interactions(shap_values, feature_names, top_n=5):
        """Detect important feature interactions"""
        
        print(f"\n=== Feature Interactions (Top {top_n}) ===")
        
        # Calculate interaction strength (simplified approach)
        interactions = {}
        n_features = len(feature_names)
        
        for i in range(n_features):
            for j in range(i+1, n_features):
                # Simple interaction metric: correlation of SHAP values
                if len(shap_values.shape) == 3:  # Multi-class
                    interaction_strength = 0
                    for class_idx in range(shap_values.shape[0]):
                        corr = np.corrcoef(shap_values[class_idx][:, i], 
                                         shap_values[class_idx][:, j])[0, 1]
                        interaction_strength += abs(corr)
                    interaction_strength /= shap_values.shape[0]
                else:  # Binary or regression
                    interaction_strength = abs(np.corrcoef(shap_values[:, i], 
                                                         shap_values[:, j])[0, 1])
                
                interactions[f"{feature_names[i]} x {feature_names[j]}"] = interaction_strength
        
        # Sort interactions by strength
        sorted_interactions = sorted(interactions.items(), key=lambda x: x[1], reverse=True)
        
        for i, (interaction, strength) in enumerate(sorted_interactions[:top_n]):
            print(f"{i+1}. {interaction}: {strength:.4f}")
        
        return sorted_interactions
    
    # Detect interactions in wine dataset
    if len(shap_values_tree) > 0:
        interactions = detect_feature_interactions(
            np.array(shap_values_tree), 
            wine_data.feature_names
        )
    
    # Model consistency analysis
    def analyze_model_consistency(explainer, X_data, sample_size=100):
        """Analyze consistency of explanations across similar samples"""
        
        print(f"\n=== Model Consistency Analysis ===")
        
        # Select random samples
        indices = np.random.choice(len(X_data), min(sample_size, len(X_data)), replace=False)
        X_sample = X_data.iloc[indices]
        
        # Calculate SHAP values
        if hasattr(explainer, 'shap_values'):
            shap_vals = explainer.shap_values(X_sample)
        else:
            shap_vals = explainer(X_sample)
        
        # Calculate variance in explanations
        if isinstance(shap_vals, list):  # Multi-class
            avg_variance = []
            for class_shap in shap_vals:
                variance = np.var(class_shap, axis=0)
                avg_variance.append(np.mean(variance))
            consistency_score = 1 / (1 + np.mean(avg_variance))
        else:  # Binary or regression
            variance = np.var(shap_vals, axis=0)
            consistency_score = 1 / (1 + np.mean(variance))
        
        print(f"Model consistency score: {consistency_score:.4f}")
        print("(Higher scores indicate more consistent explanations)")
        
        return consistency_score
    
    # Analyze consistency
    consistency = analyze_model_consistency(explainer_tree, X_test_wine)
    
    # Explanation stability test
    def test_explanation_stability(model, explainer, X_sample, feature_names, n_tests=10):
        """Test stability of explanations with small perturbations"""
        
        print(f"\n=== Explanation Stability Test ===")
        
        original_shap = explainer.shap_values(X_sample.iloc[:1])
        
        stability_scores = []
        
        for i in range(n_tests):
            # Add small random noise
            noise = np.random.normal(0, 0.01, X_sample.iloc[:1].shape)
            perturbed_sample = X_sample.iloc[:1] + noise
            
            # Calculate SHAP values for perturbed sample
            perturbed_shap = explainer.shap_values(perturbed_sample)
            
            # Calculate similarity
            if isinstance(original_shap, list):  # Multi-class
                similarities = []
                for orig, pert in zip(original_shap, perturbed_shap):
                    similarity = np.corrcoef(orig.flatten(), pert.flatten())[0, 1]
                    similarities.append(similarity)
                avg_similarity = np.mean(similarities)
            else:  # Binary or regression
                avg_similarity = np.corrcoef(original_shap.flatten(), 
                                           perturbed_shap.flatten())[0, 1]
            
            stability_scores.append(avg_similarity)
        
        stability_score = np.mean(stability_scores)
        stability_std = np.std(stability_scores)
        
        print(f"Explanation stability: {stability_score:.4f} ± {stability_std:.4f}")
        print("(Values closer to 1.0 indicate more stable explanations)")
        
        return stability_score, stability_std
    
    # Test stability
    stability_mean, stability_std = test_explanation_stability(
        rf_classifier, explainer_tree, X_test_wine, wine_data.feature_names
    )

# Run advanced analysis
advanced_shap_analysis()

print("\nComprehensive SHAP analysis completed!")
print("""
Key Insights from SHAP Analysis:
1. Feature importance varies by class in multi-class problems
2. Model explanations can differ significantly between algorithms
3. Consistency and stability are important for trust in explanations
4. Feature interactions can reveal hidden model behaviors
5. SHAP provides both global and local interpretability
""")
```

## 7. Advanced Computer Vision Techniques

### Object Detection with Deep Learning

```python
import cv2
import numpy as np
from collections import defaultdict

class YOLODetector:
    """YOLO-based object detection wrapper"""
    
    def __init__(self, weights_path=None, config_path=None, classes_path=None):
        self.net = None
        self.classes = []
        self.output_layers = []
        
        if weights_path and config_path and classes_path:
            self.load_model(weights_path, config_path, classes_path)
    
    def load_model(self, weights_path, config_path, classes_path):
        """Load YOLO model"""
        # Load YOLO
        self.net = cv2.dnn.readNet(weights_path, config_path)
        
        # Load classes
        with open(classes_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        # Get output layers
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
    
    def detect_objects(self, image, confidence_threshold=0.5, nms_threshold=0.4):
        """Detect objects in image"""
        if self.net is None:
            print("Model not loaded!")
            return [], [], []
        
        height, width, channels = image.shape
        
        # Detecting objects
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        
        # Extract information
        boxes = []
        confidences = []
        class_ids = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > confidence_threshold:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
        
        final_boxes = []
        final_confidences = []
        final_class_ids = []
        
        if len(indices) > 0:
            for i in indices.flatten():
                final_boxes.append(boxes[i])
                final_confidences.append(confidences[i])
                final_class_ids.append(class_ids[i])
        
        return final_boxes, final_confidences, final_class_ids
    
    def draw_detections(self, image, boxes, confidences, class_ids):
        """Draw detection boxes on image"""
        colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        result_image = image.copy()
        
        for i in range(len(boxes)):
            x, y, w, h = boxes[i]
            label = str(self.classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            
            # Draw bounding box
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label_text = f"{label}: {confidence:.2f}"
            cv2.putText(result_image, label_text, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return result_image

# Example usage (requires YOLO model files)
print("=== YOLO Object Detection ===")
detector = YOLODetector()
print("To use YOLO detection, download:")
print("1. yolov3.weights")
print("2. yolov3.cfg") 
print("3. coco.names")
print("Then initialize: detector.load_model(weights_path, config_path, classes_path)")
```

### Advanced Image Segmentation

```python
class ImageSegmentation:
    """Advanced image segmentation techniques"""
    
    @staticmethod
    def watershed_segmentation(image):
        """Watershed segmentation for separating touching objects"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)
        
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1
        
        # Mark the region of unknown with zero
        markers[unknown == 255] = 0
        
        # Apply watershed
        markers = cv2.watershed(image, markers)
        image[markers == -1] = [255, 0, 0]  # Mark boundaries in red
        
        return image, markers
    
    @staticmethod
    def kmeans_segmentation(image, k=4):
        """K-means clustering for color-based segmentation"""
        # Reshape image to a 2D array of pixels
        data = image.reshape((-1, 3))
        data = np.float32(data)
        
        # Define criteria and apply kmeans
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert back to uint8 and reshape to original image shape
        centers = np.uint8(centers)
        segmented_data = centers[labels.flatten()]
        segmented_image = segmented_data.reshape(image.shape)
        
        return segmented_image, labels.reshape(image.shape[:2])
    
    @staticmethod
    def grabcut_segmentation(image, rect):
        """GrabCut algorithm for interactive foreground extraction"""
        # Initialize mask
        mask = np.zeros(image.shape[:2], np.uint8)
        
        # Initialize background and foreground models
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        # Apply GrabCut
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        
        # Create final mask
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        result = image * mask2[:, :, np.newaxis]
        
        return result, mask2

# Demonstrate segmentation techniques
def demonstrate_segmentation():
    """Demonstrate various segmentation techniques"""
    
    # Create a complex test image
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    
    # Add multiple colored circles (simulating cells or objects)
    cv2.circle(img, (100, 100), 40, (255, 100, 100), -1)
    cv2.circle(img, (120, 120), 35, (100, 255, 100), -1)
    cv2.circle(img, (200, 150), 45, (100, 100, 255), -1)
    cv2.circle(img, (300, 200), 50, (255, 255, 100), -1)
    cv2.circle(img, (150, 250), 30, (255, 100, 255), -1)
    
    print("=== Image Segmentation Techniques ===")
    
    # K-means segmentation
    segmenter = ImageSegmentation()
    kmeans_result, kmeans_labels = segmenter.kmeans_segmentation(img, k=5)
    
    print("✓ K-means segmentation completed")
    
    # Watershed segmentation
    watershed_result, watershed_markers = segmenter.watershed_segmentation(img.copy())
    
    print("✓ Watershed segmentation completed")
    
    # GrabCut segmentation (define rectangle around main object)
    rect = (50, 50, 200, 200)  # x, y, width, height
    grabcut_result, grabcut_mask = segmenter.grabcut_segmentation(img.copy(), rect)
    
    print("✓ GrabCut segmentation completed")
    
    return {
        'original': img,
        'kmeans': kmeans_result,
        'watershed': watershed_result,
        'grabcut': grabcut_result
    }

# Run segmentation demonstration
segmentation_results = demonstrate_segmentation()
```

### Feature Detection and Matching

```python
class FeatureDetector:
    """Advanced feature detection and matching"""
    
    def __init__(self):
        # Initialize different detectors
        self.sift = cv2.SIFT_create()
        self.orb = cv2.ORB_create()
        try:
            self.surf = cv2.xfeatures2d.SURF_create(400)
        except AttributeError:
            self.surf = None
            print("SURF not available (requires opencv-contrib-python)")
    
    def detect_and_compute_features(self, image, method='sift'):
        """Detect and compute features using specified method"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        if method.lower() == 'sift':
            detector = self.sift
        elif method.lower() == 'orb':
            detector = self.orb
        elif method.lower() == 'surf' and self.surf is not None:
            detector = self.surf
        else:
            print(f"Method {method} not available, using SIFT")
            detector = self.sift
        
        keypoints, descriptors = detector.detectAndCompute(gray, None)
        return keypoints, descriptors
    
    def match_features(self, desc1, desc2, method='bf'):
        """Match features between two images"""
        if method == 'bf':
            # Brute Force matcher
            matcher = cv2.BFMatcher()
            matches = matcher.knnMatch(desc1, desc2, k=2)
        else:
            # FLANN matcher
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
            matches = matcher.knnMatch(desc1, desc2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        return good_matches
    
    def find_homography_and_warp(self, img1, img2, matches, kp1, kp2):
        """Find homography and warp image"""
        if len(matches) > 10:
            # Extract location of good matches
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            
            # Find homography
            M, mask = cv2.findHomography(src_pts, dst_pts, 
                                       cv2.RANSAC, 5.0)
            
            if M is not None:
                # Get dimensions
                h, w = img1.shape[:2]
                pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                
                # Transform corners
                dst = cv2.perspectiveTransform(pts, M)
                
                # Warp image
                warped = cv2.warpPerspective(img1, M, (img2.shape[1], img2.shape[0]))
                
                return M, warped, dst
        
        return None, None, None
    
    def draw_matches(self, img1, kp1, img2, kp2, matches):
        """Draw matches between images"""
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, 
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return img3

def demonstrate_feature_matching():
    """Demonstrate feature detection and matching"""
    
    # Create two similar images with some transformation
    img1 = np.zeros((300, 400, 3), dtype=np.uint8)
    img2 = np.zeros((300, 400, 3), dtype=np.uint8)
    
    # Add patterns to both images
    cv2.rectangle(img1, (50, 50), (150, 150), (255, 255, 255), -1)
    cv2.circle(img1, (200, 100), 40, (128, 128, 128), -1)
    cv2.putText(img1, 'TEST', (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    
    # Transform for second image (rotation + translation)
    M = cv2.getRotationMatrix2D((200, 150), 15, 1.0)
    M[0, 2] += 30  # Translation
    M[1, 2] += 20
    img2 = cv2.warpAffine(img1, M, (400, 300))
    
    print("=== Feature Detection and Matching ===")
    
    detector = FeatureDetector()
    
    # Detect features in both images
    kp1, desc1 = detector.detect_and_compute_features(img1, 'sift')
    kp2, desc2 = detector.detect_and_compute_features(img2, 'sift')
    
    print(f"Features detected - Image 1: {len(kp1)}, Image 2: {len(kp2)}")
    
    if desc1 is not None and desc2 is not None:
        # Match features
        matches = detector.match_features(desc1, desc2)
        print(f"Good matches found: {len(matches)}")
        
        # Find homography and warp
        H, warped, corners = detector.find_homography_and_warp(img1, img2, matches, kp1, kp2)
        
        if H is not None:
            print("✓ Homography computed successfully")
        
        # Draw matches
        match_img = detector.draw_matches(img1, kp1, img2, kp2, matches[:20])  # Show top 20 matches
        
        return {
            'img1': img1,
            'img2': img2,
            'matches': match_img,
            'warped': warped if warped is not None else img1
        }
    
    return {'img1': img1, 'img2': img2}

# Run feature matching demonstration
feature_results = demonstrate_feature_matching()
```

### Real-time Video Processing

```python
class VideoProcessor:
    """Real-time video processing with OpenCV"""
    
    def __init__(self):
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2()
        self.tracker = cv2.TrackerCSRT_create()
        self.tracking_initialized = False
    
    def motion_detection(self, frame):
        """Detect motion in video frame"""
        # Apply background subtraction
        fg_mask = self.background_subtractor.apply(frame)
        
        # Morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        min_area = 500
        motion_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        return fg_mask, motion_contours
    
    def object_tracking(self, frame, bbox=None):
        """Track object in video"""
        if bbox is not None and not self.tracking_initialized:
            # Initialize tracker
            self.tracker = cv2.TrackerCSRT_create()
            self.tracker.init(frame, bbox)
            self.tracking_initialized = True
            return True, bbox
        
        if self.tracking_initialized:
            # Update tracker
            success, bbox = self.tracker.update(frame)
            return success, bbox
        
        return False, None
    
    def optical_flow_tracking(self, prev_gray, curr_gray, prev_pts):
        """Track points using optical flow"""
        if prev_pts is not None and len(prev_pts) > 0:
            # Calculate optical flow
            next_pts, status, error = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, prev_pts, None
            )
            
            # Select good points
            good_new = next_pts[status == 1]
            good_old = prev_pts[status == 1]
            
            return good_new, good_old
        
        return None, None
    
    def detect_corners(self, gray_frame, max_corners=100):
        """Detect corners for tracking"""
        corners = cv2.goodFeaturesToTrack(
            gray_frame,
            maxCorners=max_corners,
            qualityLevel=0.01,
            minDistance=10,
            blockSize=3
        )
        return corners

def demonstrate_video_processing():
    """Demonstrate video processing techniques"""
    
    print("=== Video Processing Demonstration ===")
    
    # Create synthetic video frames
    frames = []
    for i in range(10):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Moving object
        x = 50 + i * 20
        y = 100 + int(10 * np.sin(i * 0.5))
        cv2.circle(frame, (x, y), 30, (0, 255, 0), -1)
        
        # Static background
        cv2.rectangle(frame, (500, 300), (600, 400), (255, 0, 0), -1)
        
        frames.append(frame)
    
    processor = VideoProcessor()
    
    # Process frames
    for i, frame in enumerate(frames):
        # Motion detection
        fg_mask, motion_contours = processor.motion_detection(frame)
        
        # Draw motion detection results
        for contour in motion_contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        
        print(f"Frame {i}: {len(motion_contours)} moving objects detected")
    
    # Corner detection and optical flow
    gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
    
    # Detect initial corners
    initial_corners = processor.detect_corners(gray_frames[0])
    
    if initial_corners is not None:
        print(f"Initial corners detected: {len(initial_corners)}")
        
        # Track corners through frames
        prev_gray = gray_frames[0]
        prev_pts = initial_corners
        
        for i in range(1, len(gray_frames)):
            curr_gray = gray_frames[i]
            good_new, good_old = processor.optical_flow_tracking(prev_gray, curr_gray, prev_pts)
            
            if good_new is not None:
                print(f"Frame {i}: {len(good_new)} corners tracked")
                prev_pts = good_new.reshape(-1, 1, 2)
                prev_gray = curr_gray
    
    print("✓ Video processing demonstration completed")

# Run video processing demonstration
demonstrate_video_processing()

print("\n=== Summary of Advanced Computer Vision Capabilities ===")
print("""
Advanced Computer Vision Techniques Covered:

1. Object Detection:
   - YOLO integration for real-time detection
   - Bounding box drawing and NMS

2. Image Segmentation:
   - Watershed algorithm for separating objects
   - K-means clustering for color-based segmentation
   - GrabCut for interactive foreground extraction

3. Feature Detection and Matching:
   - SIFT, ORB, SURF feature detectors
   - Brute force and FLANN-based matching
   - Homography estimation and image warping

4. Video Processing:
   - Motion detection with background subtraction
   - Object tracking with CSRT tracker
   - Optical flow for point tracking
   - Corner detection for feature tracking

5. Applications:
   - Surveillance and security systems
   - Augmented reality applications
   - Image stitching and panorama creation
   - Medical image analysis
   - Industrial quality control
""")
```

This completes the comprehensive computer vision and advanced AI libraries documentation with state-of-the-art computer vision techniques including object detection, advanced segmentation methods, feature detection and matching, and real-time video processing capabilities. The documentation now covers all major aspects of the AI-ML ecosystem from fundamental libraries to advanced specialized techniques.
