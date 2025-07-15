# Machine Learning Libraries

## Table of Contents
1. [Scikit-learn - Machine Learning](#scikit-learn---machine-learning)
2. [TensorFlow - Deep Learning](#tensorflow---deep-learning)
3. [PyTorch - Deep Learning](#pytorch---deep-learning)
4. [Keras - High-level Neural Networks](#keras---high-level-neural-networks)
5. [XGBoost - Gradient Boosting](#xgboost---gradient-boosting)
6. [LightGBM - Gradient Boosting](#lightgbm---gradient-boosting)

## Scikit-learn - Machine Learning

Scikit-learn is the most popular machine learning library for Python, providing simple and efficient tools for data mining and analysis.

### 1. Installation and Basic Setup
```python
# Installation
pip install scikit-learn

# Import conventions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_iris, load_boston, make_classification
import warnings
warnings.filterwarnings('ignore')

print(f"Scikit-learn version: {sklearn.__version__}")
```

### 2. Data Preprocessing
```python
# Loading sample dataset
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target

print(f"Feature names: {iris.feature_names}")
print(f"Target names: {iris.target_names}")
print(f"Data shape: {X.shape}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Original feature means: {X_train.mean(axis=0)}")
print(f"Scaled feature means: {X_train_scaled.mean(axis=0)}")
print(f"Scaled feature stds: {X_train_scaled.std(axis=0)}")

# Label encoding for categorical variables
from sklearn.preprocessing import LabelEncoder

# Example with categorical data
categories = ['cat', 'dog', 'bird', 'cat', 'dog']
label_encoder = LabelEncoder()
encoded_categories = label_encoder.fit_transform(categories)
print(f"Original: {categories}")
print(f"Encoded: {encoded_categories}")
print(f"Classes: {label_encoder.classes_}")

# One-hot encoding
from sklearn.preprocessing import OneHotEncoder
categories_2d = np.array(categories).reshape(-1, 1)
onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoded = onehot_encoder.fit_transform(categories_2d)
print(f"One-hot encoded:\n{onehot_encoded}")

# Handling missing values
from sklearn.impute import SimpleImputer

# Create data with missing values
data_with_missing = np.array([[1, 2, 3], [4, np.nan, 6], [7, 8, np.nan]])
print(f"Data with missing values:\n{data_with_missing}")

# Simple imputation
imputer = SimpleImputer(strategy='mean')  # Can be 'mean', 'median', 'most_frequent', 'constant'
imputed_data = imputer.fit_transform(data_with_missing)
print(f"Imputed data:\n{imputed_data}")
```

### 3. Classification Algorithms
```python
# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Create and train model
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_scaled, y_train)

# Make predictions
y_pred_log = log_reg.predict(X_test_scaled)
y_prob_log = log_reg.predict_proba(X_test_scaled)

print("Logistic Regression Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_log):.3f}")
print(f"Classification Report:\n{classification_report(y_test, y_pred_log)}")

# Decision Tree
from sklearn.tree import DecisionTreeClassifier

dt_classifier = DecisionTreeClassifier(random_state=42, max_depth=3)
dt_classifier.fit(X_train, y_train)  # Decision trees don't require scaling

y_pred_dt = dt_classifier.predict(X_test)
print(f"\nDecision Tree Accuracy: {accuracy_score(y_test, y_pred_dt):.3f}")

# Random Forest
from sklearn.ensemble import RandomForestClassifier

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

y_pred_rf = rf_classifier.predict(X_test)
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf):.3f}")

# Feature importance
feature_importance = rf_classifier.feature_importances_
for i, importance in enumerate(feature_importance):
    print(f"Feature {iris.feature_names[i]}: {importance:.3f}")

# Support Vector Machine
from sklearn.svm import SVC

svm_classifier = SVC(kernel='rbf', random_state=42)
svm_classifier.fit(X_train_scaled, y_train)

y_pred_svm = svm_classifier.predict(X_test_scaled)
print(f"SVM Accuracy: {accuracy_score(y_test, y_pred_svm):.3f}")

# K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train_scaled, y_train)

y_pred_knn = knn_classifier.predict(X_test_scaled)
print(f"KNN Accuracy: {accuracy_score(y_test, y_pred_knn):.3f}")

# Naive Bayes
from sklearn.naive_bayes import GaussianNB

nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

y_pred_nb = nb_classifier.predict(X_test)
print(f"Naive Bayes Accuracy: {accuracy_score(y_test, y_pred_nb):.3f}")
```

### 4. Regression Algorithms
```python
# Load regression dataset
from sklearn.datasets import load_boston
boston = load_boston()
X_reg, y_reg = boston.data, boston.target

# Split data
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

# Scale features
scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

# Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

linear_reg = LinearRegression()
linear_reg.fit(X_train_reg_scaled, y_train_reg)

y_pred_linear = linear_reg.predict(X_test_reg_scaled)

print("Linear Regression Results:")
print(f"MSE: {mean_squared_error(y_test_reg, y_pred_linear):.3f}")
print(f"R² Score: {r2_score(y_test_reg, y_pred_linear):.3f}")

# Ridge Regression (L2 regularization)
from sklearn.linear_model import Ridge

ridge_reg = Ridge(alpha=1.0)
ridge_reg.fit(X_train_reg_scaled, y_train_reg)

y_pred_ridge = ridge_reg.predict(X_test_reg_scaled)
print(f"Ridge R² Score: {r2_score(y_test_reg, y_pred_ridge):.3f}")

# Lasso Regression (L1 regularization)
from sklearn.linear_model import Lasso

lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X_train_reg_scaled, y_train_reg)

y_pred_lasso = lasso_reg.predict(X_test_reg_scaled)
print(f"Lasso R² Score: {r2_score(y_test_reg, y_pred_lasso):.3f}")

# Elastic Net (L1 + L2 regularization)
from sklearn.linear_model import ElasticNet

elastic_reg = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_reg.fit(X_train_reg_scaled, y_train_reg)

y_pred_elastic = elastic_reg.predict(X_test_reg_scaled)
print(f"Elastic Net R² Score: {r2_score(y_test_reg, y_pred_elastic):.3f}")

# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor

rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train_reg, y_train_reg)

y_pred_rf_reg = rf_regressor.predict(X_test_reg)
print(f"Random Forest R² Score: {r2_score(y_test_reg, y_pred_rf_reg):.3f}")

# Support Vector Regression
from sklearn.svm import SVR

svr = SVR(kernel='rbf')
svr.fit(X_train_reg_scaled, y_train_reg)

y_pred_svr = svr.predict(X_test_reg_scaled)
print(f"SVR R² Score: {r2_score(y_test_reg, y_pred_svr):.3f}")
```

### 5. Clustering Algorithms
```python
# Generate sample data for clustering
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score

# Create sample data
X_cluster, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)

# K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
y_kmeans = kmeans.fit_predict(X_cluster)

print("K-Means Clustering:")
print(f"Silhouette Score: {silhouette_score(X_cluster, y_kmeans):.3f}")
print(f"Adjusted Rand Index: {adjusted_rand_score(y_true, y_kmeans):.3f}")

# DBSCAN Clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
y_dbscan = dbscan.fit_predict(X_cluster)

print(f"\nDBSCAN Clustering:")
print(f"Number of clusters: {len(set(y_dbscan)) - (1 if -1 in y_dbscan else 0)}")
print(f"Number of noise points: {list(y_dbscan).count(-1)}")

# Hierarchical Clustering
hierarchical = AgglomerativeClustering(n_clusters=4)
y_hierarchical = hierarchical.fit_predict(X_cluster)

print(f"\nHierarchical Clustering:")
print(f"Silhouette Score: {silhouette_score(X_cluster, y_hierarchical):.3f}")

# Elbow method for optimal K
inertias = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_cluster)
    inertias.append(kmeans.inertia_)

print(f"\nInertias for K=1 to 10: {inertias}")
```

### 6. Model Evaluation and Validation
```python
# Cross-validation
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Using iris dataset again
classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Simple cross-validation
cv_scores = cross_val_score(classifier, X, y, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
stratified_scores = cross_val_score(classifier, X, y, cv=skf)
print(f"Stratified CV scores: {stratified_scores}")

# Learning curves
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    classifier, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10)
)

print(f"Training sizes: {train_sizes}")
print(f"Training scores mean: {train_scores.mean(axis=1)}")
print(f"Validation scores mean: {val_scores.mean(axis=1)}")

# Validation curves
from sklearn.model_selection import validation_curve

param_range = [10, 20, 50, 100, 200]
train_scores_vc, val_scores_vc = validation_curve(
    RandomForestClassifier(random_state=42), X, y,
    param_name='n_estimators', param_range=param_range, cv=5
)

print(f"Validation curve - Training scores: {train_scores_vc.mean(axis=1)}")
print(f"Validation curve - Validation scores: {val_scores_vc.mean(axis=1)}")
```

### 7. Hyperparameter Tuning
```python
# Grid Search
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create classifier
rf = RandomForestClassifier(random_state=42)

# Grid search with cross-validation
grid_search = GridSearchCV(
    rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.3f}")

# Test best model
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
print(f"Test accuracy: {accuracy_score(y_test, y_pred_best):.3f}")

# Random Search
from sklearn.model_selection import RandomizedSearchCV

# Define parameter distributions
param_dist = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 6]
}

random_search = RandomizedSearchCV(
    rf, param_distributions=param_dist, n_iter=20, cv=5, 
    scoring='accuracy', random_state=42, n_jobs=-1
)
random_search.fit(X_train, y_train)

print(f"Random search best parameters: {random_search.best_params_}")
print(f"Random search best score: {random_search.best_score_:.3f}")
```

## TensorFlow - Deep Learning

TensorFlow is Google's open-source machine learning framework, particularly powerful for deep learning applications.

### 1. Installation and Basic Setup
```python
# Installation
pip install tensorflow

# Import TensorFlow
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")

# Check if GPU is available
if tf.test.is_gpu_available():
    print("GPU is available for TensorFlow")
else:
    print("Running on CPU")
```

### 2. Tensors and Basic Operations
```python
# Creating tensors
scalar = tf.constant(7)
vector = tf.constant([1, 2, 3, 4])
matrix = tf.constant([[1, 2], [3, 4]])
tensor_3d = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

print(f"Scalar: {scalar}")
print(f"Vector: {vector}")
print(f"Matrix: {matrix}")
print(f"3D Tensor: {tensor_3d}")

# Tensor properties
print(f"Vector shape: {vector.shape}")
print(f"Vector dtype: {vector.dtype}")
print(f"Matrix rank: {tf.rank(matrix)}")

# Creating tensors with specific values
zeros = tf.zeros((3, 3))
ones = tf.ones((2, 4))
random_normal = tf.random.normal((3, 3), mean=0, stddev=1)
random_uniform = tf.random.uniform((2, 2), minval=0, maxval=1)

print(f"Zeros tensor:\n{zeros}")
print(f"Random normal tensor:\n{random_normal}")

# Basic operations
a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[5, 6], [7, 8]])

# Element-wise operations
add = tf.add(a, b)  # or a + b
multiply = tf.multiply(a, b)  # or a * b
subtract = tf.subtract(a, b)  # or a - b

print(f"Addition:\n{add}")
print(f"Element-wise multiplication:\n{multiply}")

# Matrix operations
matmul = tf.matmul(a, b)  # or a @ b
transpose = tf.transpose(a)

print(f"Matrix multiplication:\n{matmul}")
print(f"Transpose:\n{transpose}")

# Reduction operations
sum_all = tf.reduce_sum(a)
sum_axis_0 = tf.reduce_sum(a, axis=0)
mean = tf.reduce_mean(a)
max_val = tf.reduce_max(a)

print(f"Sum of all elements: {sum_all}")
print(f"Sum along axis 0: {sum_axis_0}")
print(f"Mean: {mean}")
```

### 3. Neural Network with Keras (High-level API)
```python
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Generate sample data
np.random.seed(42)
X_train = np.random.random((1000, 10))
y_train = np.random.randint(0, 2, (1000, 1))
X_test = np.random.random((200, 10))
y_test = np.random.randint(0, 2, (200, 1))

# Create a simple neural network
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(10,)),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Model summary
model.summary()

# Train the model
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=50,
    validation_split=0.2,
    verbose=1
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_accuracy:.3f}")

# Make predictions
predictions = model.predict(X_test)
print(f"First 5 predictions: {predictions[:5].flatten()}")
```

### 4. Convolutional Neural Network (CNN)
```python
# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Normalize pixel values
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convert labels to categorical
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print(f"Training data shape: {x_train.shape}")
print(f"Test data shape: {x_test.shape}")

# Create CNN model
cnn_model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# Compile CNN
cnn_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

cnn_model.summary()

# Train CNN (commented out for brevity - takes time)
# cnn_history = cnn_model.fit(
#     x_train, y_train,
#     batch_size=32,
#     epochs=10,
#     validation_data=(x_test, y_test),
#     verbose=1
# )
```

### 5. Custom Training Loop
```python
# Custom training with GradientTape
import tensorflow as tf

# Simple linear regression example
# Generate synthetic data
np.random.seed(42)
X = np.random.randn(100, 1)
y = 3 * X + 2 + 0.1 * np.random.randn(100, 1)

# Convert to TensorFlow tensors
X_tf = tf.constant(X, dtype=tf.float32)
y_tf = tf.constant(y, dtype=tf.float32)

# Initialize parameters
w = tf.Variable(tf.random.normal((1, 1)), name='weight')
b = tf.Variable(tf.random.normal((1,)), name='bias')

# Define loss function
def mse_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Define optimizer
optimizer = tf.optimizers.Adam(learning_rate=0.01)

# Training loop
epochs = 1000
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        # Forward pass
        y_pred = tf.matmul(X_tf, w) + b
        loss = mse_loss(y_tf, y_pred)
    
    # Compute gradients
    gradients = tape.gradient(loss, [w, b])
    
    # Update parameters
    optimizer.apply_gradients(zip(gradients, [w, b]))
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}, W: {w.numpy()[0,0]:.4f}, B: {b.numpy()[0]:.4f}")

print(f"Final parameters - W: {w.numpy()[0,0]:.4f}, B: {b.numpy()[0]:.4f}")
print(f"True parameters - W: 3.0, B: 2.0")
```

## PyTorch - Deep Learning

PyTorch is Facebook's open-source machine learning library, known for its dynamic computation graphs and ease of use.

### 1. Installation and Basic Setup
```python
# Installation
pip install torch torchvision torchaudio

# Import PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
```

### 2. Tensors and Basic Operations
```python
# Creating tensors
scalar = torch.tensor(7)
vector = torch.tensor([1, 2, 3, 4])
matrix = torch.tensor([[1, 2], [3, 4]])
tensor_3d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

print(f"Vector: {vector}")
print(f"Matrix: {matrix}")
print(f"Matrix shape: {matrix.shape}")
print(f"Matrix dtype: {matrix.dtype}")

# Creating tensors with specific values
zeros = torch.zeros(3, 3)
ones = torch.ones(2, 4)
random_tensor = torch.randn(3, 3)  # Normal distribution
uniform_tensor = torch.rand(2, 2)  # Uniform distribution [0, 1)

print(f"Random tensor:\n{random_tensor}")

# Tensor operations
a = torch.tensor([[1., 2.], [3., 4.]])
b = torch.tensor([[5., 6.], [7., 8.]])

# Element-wise operations
add = a + b
multiply = a * b
power = a ** 2

print(f"Addition:\n{add}")
print(f"Element-wise multiplication:\n{multiply}")

# Matrix operations
matmul = torch.matmul(a, b)  # or a @ b
transpose = a.T

print(f"Matrix multiplication:\n{matmul}")
print(f"Transpose:\n{transpose}")

# Reduction operations
sum_all = torch.sum(a)
mean = torch.mean(a)
max_val = torch.max(a)

print(f"Sum: {sum_all}")
print(f"Mean: {mean}")

# Moving tensors to GPU (if available)
if torch.cuda.is_available():
    a_gpu = a.cuda()
    print(f"Tensor on GPU: {a_gpu.device}")
```

## 7. Advanced Scikit-learn Techniques

### Custom Transformers and Pipelines

```python
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
import numpy as np

class OutlierRemover(BaseEstimator, TransformerMixin):
    """Custom transformer to remove outliers using IQR method"""
    
    def __init__(self, factor=1.5):
        self.factor = factor
        
    def fit(self, X, y=None):
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1
        self.lower_bound = Q1 - self.factor * IQR
        self.upper_bound = Q3 + self.factor * IQR
        return self
    
    def transform(self, X):
        mask = np.all((X >= self.lower_bound) & (X <= self.upper_bound), axis=1)
        return X[mask]

class FeatureSelector(BaseEstimator, TransformerMixin):
    """Custom feature selector based on correlation"""
    
    def __init__(self, threshold=0.8):
        self.threshold = threshold
        
    def fit(self, X, y=None):
        corr_matrix = np.corrcoef(X.T)
        upper_tri = np.triu(np.ones_like(corr_matrix), k=1)
        high_corr = np.where((np.abs(corr_matrix) > self.threshold) & upper_tri)
        self.drop_cols = list(set(high_corr[1]))
        return self
    
    def transform(self, X):
        return np.delete(X, self.drop_cols, axis=1)

# Create complex pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ('outlier_removal', OutlierRemover(factor=2.0)),
    ('feature_selection', FeatureSelector(threshold=0.9)),
    ('scaling', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Fit and evaluate
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)
score = pipeline.score(X_test, y_test)
print(f"Pipeline accuracy: {score:.3f}")
```

### Advanced Model Selection and Hyperparameter Tuning

```python
from sklearn.model_selection import RandomizedSearchCV, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, f1_score
from scipy.stats import randint, uniform

# Define multiple models with parameter distributions
models = {
    'rf': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': randint(50, 200),
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': randint(2, 11),
            'min_samples_leaf': randint(1, 5)
        }
    },
    'svm': {
        'model': SVC(random_state=42),
        'params': {
            'C': uniform(0.1, 10),
            'gamma': ['scale', 'auto'] + list(uniform(0.001, 0.1).rvs(5)),
            'kernel': ['rbf', 'poly', 'sigmoid']
        }
    }
}

# Custom scoring function
def custom_scorer(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted')

custom_score = make_scorer(custom_scorer)

# Perform randomized search for each model
best_models = {}
for name, config in models.items():
    search = RandomizedSearchCV(
        config['model'],
        config['params'],
        n_iter=50,
        cv=5,
        scoring=custom_score,
        random_state=42,
        n_jobs=-1
    )
    
    search.fit(X_train, y_train)
    best_models[name] = search.best_estimator_
    
    print(f"{name.upper()} - Best score: {search.best_score_:.3f}")
    print(f"Best params: {search.best_params_}\n")

# Model comparison with multiple metrics
scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

for name, model in best_models.items():
    scores = cross_validate(model, X_train, y_train, cv=5, scoring=scoring)
    
    print(f"{name.upper()} Cross-validation scores:")
    for metric in scoring:
        mean_score = scores[f'test_{metric}'].mean()
        std_score = scores[f'test_{metric}'].std()
        print(f"  {metric}: {mean_score:.3f} (+/- {std_score * 2:.3f})")
    print()
```

### Model Interpretation and Explainability

```python
from sklearn.inspection import permutation_importance, plot_partial_dependence
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Feature importance for tree-based models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Built-in feature importance
feature_importance = rf_model.feature_importances_
feature_names = iris.feature_names

# Create importance plot
plt.figure(figsize=(10, 6))
indices = np.argsort(feature_importance)[::-1]
plt.bar(range(len(feature_importance)), feature_importance[indices])
plt.xticks(range(len(feature_importance)), [feature_names[i] for i in indices], rotation=45)
plt.title('Feature Importance (Random Forest)')
plt.tight_layout()
plt.show()

# Permutation importance (model-agnostic)
perm_importance = permutation_importance(rf_model, X_test, y_test, n_repeats=10, random_state=42)

plt.figure(figsize=(10, 6))
plt.boxplot([perm_importance.importances[i] for i in indices])
plt.xticks(range(1, len(feature_importance) + 1), [feature_names[i] for i in indices], rotation=45)
plt.title('Permutation Importance')
plt.tight_layout()
plt.show()

# Partial dependence plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
plot_partial_dependence(rf_model, X_train, features=[0, 1, 2, 3], 
                       feature_names=feature_names, ax=axes.ravel())
plt.tight_layout()
plt.show()
```

## 8. Advanced TensorFlow Techniques

### Custom Layers and Models

```python
import tensorflow as tf
from tensorflow.keras import layers, Model

class AttentionLayer(layers.Layer):
    """Custom attention layer"""
    
    def __init__(self, units):
        super(AttentionLayer, self).__init__()
        self.units = units
        
    def build(self, input_shape):
        self.W1 = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True
        )
        self.W2 = self.add_weight(
            shape=(self.units, 1),
            initializer='random_normal',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
        
    def call(self, inputs):
        # Attention mechanism
        score = tf.nn.tanh(tf.matmul(inputs, self.W1))
        attention_weights = tf.nn.softmax(tf.matmul(score, self.W2), axis=1)
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector

class CustomModel(Model):
    """Custom model with attention"""
    
    def __init__(self, num_classes):
        super(CustomModel, self).__init__()
        self.lstm = layers.LSTM(64, return_sequences=True)
        self.attention = AttentionLayer(32)
        self.dropout = layers.Dropout(0.5)
        self.dense = layers.Dense(num_classes, activation='softmax')
        
    def call(self, inputs, training=None):
        x = self.lstm(inputs)
        x = self.attention(x)
        x = self.dropout(x, training=training)
        return self.dense(x)

# Example usage with sequential data
sequence_length = 10
feature_dim = 5
num_classes = 3

# Create sample data
X_seq = tf.random.normal((1000, sequence_length, feature_dim))
y_seq = tf.random.uniform((1000,), maxval=num_classes, dtype=tf.int32)
y_seq = tf.keras.utils.to_categorical(y_seq, num_classes)

# Build and compile model
model = CustomModel(num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_seq, y_seq, epochs=5, batch_size=32, validation_split=0.2, verbose=1)
```

### Advanced Training Techniques

```python
# Custom training loop with gradient accumulation
@tf.function
def train_step(model, optimizer, x_batch, y_batch, accumulate_steps=4):
    """Custom training step with gradient accumulation"""
    accumulated_gradients = []
    accumulated_loss = 0.0
    
    for i in range(accumulate_steps):
        start_idx = i * len(x_batch) // accumulate_steps
        end_idx = (i + 1) * len(x_batch) // accumulate_steps
        
        x_mini = x_batch[start_idx:end_idx]
        y_mini = y_batch[start_idx:end_idx]
        
        with tf.GradientTape() as tape:
            predictions = model(x_mini, training=True)
            loss = tf.keras.losses.categorical_crossentropy(y_mini, predictions)
            loss = tf.reduce_mean(loss) / accumulate_steps
            
        gradients = tape.gradient(loss, model.trainable_variables)
        accumulated_gradients.append(gradients)
        accumulated_loss += loss
    
    # Average gradients
    averaged_gradients = []
    for gradient_list in zip(*accumulated_gradients):
        averaged_gradients.append(tf.reduce_mean(gradient_list, axis=0))
    
    # Apply gradients
    optimizer.apply_gradients(zip(averaged_gradients, model.trainable_variables))
    
    return accumulated_loss

# Learning rate scheduling
def create_lr_schedule():
    """Create custom learning rate schedule"""
    initial_lr = 0.001
    
    def lr_schedule(epoch):
        if epoch < 10:
            return initial_lr
        elif epoch < 20:
            return initial_lr * 0.1
        else:
            return initial_lr * 0.01
    
    return lr_schedule

# Custom callbacks
class CustomCallback(tf.keras.callbacks.Callback):
    """Custom callback for monitoring training"""
    
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('val_accuracy') > 0.95:
            print(f"\nReached 95% validation accuracy at epoch {epoch + 1}!")
            self.model.stop_training = True

# Model checkpointing and early stopping
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    ),
    tf.keras.callbacks.LearningRateScheduler(create_lr_schedule()),
    CustomCallback()
]
```

## 9. Production and Deployment

### Model Serialization and Loading

```python
import joblib
import pickle
from sklearn.ensemble import RandomForestClassifier

# Train a model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Method 1: Joblib (recommended for scikit-learn)
joblib.dump(model, 'model_joblib.pkl')
loaded_model_joblib = joblib.load('model_joblib.pkl')

# Method 2: Pickle
with open('model_pickle.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('model_pickle.pkl', 'rb') as f:
    loaded_model_pickle = pickle.load(f)

# Verify models work
original_pred = model.predict(X_test)
joblib_pred = loaded_model_joblib.predict(X_test)
pickle_pred = loaded_model_pickle.predict(X_test)

print(f"Original accuracy: {accuracy_score(y_test, original_pred):.3f}")
print(f"Joblib accuracy: {accuracy_score(y_test, joblib_pred):.3f}")
print(f"Pickle accuracy: {accuracy_score(y_test, pickle_pred):.3f}")

# TensorFlow model saving
tf_model = tf.keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(4,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(3, activation='softmax')
])

tf_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
tf_model.fit(X_train, y_train, epochs=10, verbose=0)

# Save TensorFlow model
tf_model.save('tf_model.h5')  # HDF5 format
tf_model.save('tf_model_savedmodel')  # SavedModel format

# Load TensorFlow model
loaded_tf_model = tf.keras.models.load_model('tf_model.h5')
```

### Model Monitoring and Validation

```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def comprehensive_model_evaluation(model, X_test, y_test, class_names=None):
    """Comprehensive model evaluation function"""
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = None
    
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)
    
    # Basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.3f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # ROC curve for binary/multiclass
    if y_pred_proba is not None and len(np.unique(y_test)) <= 10:
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize
        
        if len(np.unique(y_test)) == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.show()
        else:
            # Multiclass classification
            y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
            n_classes = y_test_bin.shape[1]
            
            plt.figure(figsize=(10, 8))
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
                roc_auc = auc(fpr, tpr)
                class_name = class_names[i] if class_names else f'Class {i}'
                plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')
            
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves - Multiclass')
            plt.legend()
            plt.show()

# Example usage
comprehensive_model_evaluation(model, X_test, y_test, iris.target_names)
```

This completes the comprehensive machine learning libraries documentation with advanced techniques, custom implementations, and production considerations.
