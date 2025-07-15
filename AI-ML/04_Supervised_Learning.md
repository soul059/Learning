# 04. Supervised Learning

## 🎯 Learning Objectives
- Master classification and regression algorithms
- Understand when to use each algorithm
- Learn model evaluation metrics and techniques
- Implement algorithms from scratch and using libraries

---

## 1. Introduction to Supervised Learning

**Supervised Learning** uses labeled training data to learn a mapping from inputs to outputs. The goal is to make accurate predictions on new, unseen data.

### 1.1 Key Concepts 🟢

#### Components:
- **Training Data**: Input-output pairs (X, y)
- **Features (X)**: Input variables/attributes
- **Target (y)**: Output variable/labels
- **Model**: Mathematical function learned from data
- **Prediction**: Model's output for new inputs

#### Learning Process:
```
Training Data (X, y) → Algorithm → Model → Predictions (ŷ)
```

#### Types of Supervised Learning:
1. **Classification**: Predicting categories/classes
2. **Regression**: Predicting continuous numerical values

### 1.2 Classification vs. Regression 🟢

| Aspect | Classification | Regression |
|--------|----------------|------------|
| **Output** | Discrete categories | Continuous numbers |
| **Examples** | Email spam/ham, Image recognition | House prices, Temperature |
| **Evaluation** | Accuracy, Precision, Recall | MAE, MSE, R² |
| **Algorithms** | Logistic Regression, SVM, Decision Trees | Linear Regression, Polynomial Regression |

---

## 2. Linear Models

### 2.1 Linear Regression 🟢

**Concept**: Find the best line that fits through data points.

#### Mathematical Foundation:
```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
```

Where:
- y: Target variable
- β₀: Intercept (bias)
- βᵢ: Coefficients (weights)
- xᵢ: Features
- ε: Error term

#### Cost Function (Mean Squared Error):
```
J(β) = (1/2m) Σ(hβ(xⁱ) - yⁱ)²
```

#### Normal Equation (Closed-form solution):
```
β = (XᵀX)⁻¹Xᵀy
```

#### Implementation:
```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse:.4f}, R²: {r2:.4f}")
```

#### From Scratch Implementation:
```python
class LinearRegressionScratch:
    def __init__(self):
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        # Add bias term to X
        X_with_bias = np.c_[np.ones(X.shape[0]), X]
        
        # Normal equation: β = (XᵀX)⁻¹Xᵀy
        self.weights = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
        
        self.bias = self.weights[0]
        self.weights = self.weights[1:]
    
    def predict(self, X):
        return X @ self.weights + self.bias
```

#### Assumptions:
1. **Linearity**: Relationship between X and y is linear
2. **Independence**: Observations are independent
3. **Homoscedasticity**: Constant variance of errors
4. **Normality**: Errors are normally distributed
5. **No multicollinearity**: Features are not highly correlated

#### When to Use:
- ✅ Simple, interpretable baseline
- ✅ Feature importance needed
- ✅ Few features, sufficient data
- ❌ Non-linear relationships
- ❌ High-dimensional data without regularization

### 2.2 Polynomial Regression 🟡

**Concept**: Extend linear regression to capture non-linear relationships.

#### Mathematical Form:
```
y = β₀ + β₁x + β₂x² + β₃x³ + ... + βₙxⁿ
```

#### Implementation:
```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# Create polynomial features
poly_reg = Pipeline([
    ('poly', PolynomialFeatures(degree=3)),
    ('linear', LinearRegression())
])

poly_reg.fit(X_train, y_train)
y_pred = poly_reg.predict(X_test)
```

#### Bias-Variance Tradeoff:
- **Low degree**: High bias, low variance (underfitting)
- **High degree**: Low bias, high variance (overfitting)
- **Optimal degree**: Balance between bias and variance

### 2.3 Regularized Linear Models 🟡

#### Ridge Regression (L2 Regularization):
```
J(β) = MSE + α Σβᵢ²
```

**Characteristics:**
- Shrinks coefficients toward zero
- Keeps all features
- Good when many features are somewhat useful

```python
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
```

#### Lasso Regression (L1 Regularization):
```
J(β) = MSE + α Σ|βᵢ|
```

**Characteristics:**
- Can set coefficients to exactly zero
- Performs feature selection
- Good when many features are irrelevant

```python
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=1.0)
lasso.fit(X_train, y_train)
```

#### Elastic Net (L1 + L2):
```
J(β) = MSE + α₁ Σ|βᵢ| + α₂ Σβᵢ²
```

**Characteristics:**
- Combines Ridge and Lasso
- Good when features are correlated

```python
from sklearn.linear_model import ElasticNet

elastic = ElasticNet(alpha=1.0, l1_ratio=0.5)
elastic.fit(X_train, y_train)
```

### 2.4 Logistic Regression 🟢

**Concept**: Linear model for classification using logistic function.

#### Sigmoid Function:
```
σ(z) = 1 / (1 + e⁻ᶻ)
where z = β₀ + β₁x₁ + ... + βₙxₙ
```

#### Probability Interpretation:
```
P(y=1|x) = σ(βᵀx)
P(y=0|x) = 1 - σ(βᵀx)
```

#### Cost Function (Log-likelihood):
```
J(β) = -Σ[y log(σ(βᵀx)) + (1-y) log(1-σ(βᵀx))]
```

#### Implementation:
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Create and train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))
```

#### From Scratch Implementation:
```python
class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -709, 709)))  # Clip to prevent overflow
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for i in range(self.max_iterations):
            # Forward pass
            z = X @ self.weights + self.bias
            predictions = self.sigmoid(z)
            
            # Compute cost
            cost = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
            
            # Compute gradients
            dw = (1 / n_samples) * X.T @ (predictions - y)
            db = (1 / n_samples) * np.sum(predictions - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        z = X @ self.weights + self.bias
        predictions = self.sigmoid(z)
        return (predictions >= 0.5).astype(int)
```

#### Multi-class Classification:
- **One-vs-Rest (OvR)**: Train binary classifier for each class
- **One-vs-One (OvO)**: Train binary classifier for each pair of classes
- **Multinomial**: Direct multi-class extension

```python
# Multi-class logistic regression
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
model.fit(X_train, y_train)
```

---

## 3. Tree-Based Models

### 3.1 Decision Trees 🟢

**Concept**: Create a tree-like model of decisions to reach predictions.

#### How It Works:
1. Start with root node containing all data
2. Find best feature and threshold to split data
3. Create child nodes with split data
4. Repeat until stopping criteria met
5. Assign prediction to each leaf node

#### Splitting Criteria:

**For Classification (Gini Impurity):**
```
Gini = 1 - Σpᵢ²
where pᵢ is probability of class i
```

**For Classification (Entropy):**
```
Entropy = -Σpᵢ log₂(pᵢ)
```

**For Regression (MSE):**
```
MSE = (1/n) Σ(yᵢ - ȳ)²
```

#### Implementation:
```python
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Classification
clf = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Regression
reg = DecisionTreeRegressor(criterion='mse', max_depth=5, random_state=42)
reg.fit(X_train, y_train)

# Visualize tree
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=feature_names, class_names=class_names, filled=True)
plt.show()
```

#### From Scratch Implementation (Simplified):
```python
class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        
    def gini_impurity(self, y):
        proportions = np.bincount(y) / len(y)
        return 1 - np.sum(proportions ** 2)
    
    def information_gain(self, X_column, y, threshold):
        # Split data
        left_mask = X_column <= threshold
        right_mask = ~left_mask
        
        if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
            return 0
        
        # Calculate weighted average of impurities
        n = len(y)
        left_impurity = self.gini_impurity(y[left_mask])
        right_impurity = self.gini_impurity(y[right_mask])
        
        weighted_impurity = (len(y[left_mask]) / n) * left_impurity + \
                           (len(y[right_mask]) / n) * right_impurity
        
        return self.gini_impurity(y) - weighted_impurity
    
    def best_split(self, X, y):
        best_gain = 0
        best_feature = None
        best_threshold = None
        
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gain = self.information_gain(X[:, feature], y, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
```

#### Advantages:
- ✅ Interpretable and visualizable
- ✅ No assumptions about data distribution
- ✅ Handles both numerical and categorical features
- ✅ Automatic feature selection
- ✅ Can capture non-linear relationships

#### Disadvantages:
- ❌ Prone to overfitting
- ❌ Unstable (small data changes can change tree)
- ❌ Biased toward features with more levels
- ❌ Poor with linear relationships

### 3.2 Random Forest 🟡

**Concept**: Ensemble of decision trees using bagging and random feature selection.

#### Algorithm:
1. Create bootstrap samples of training data
2. For each sample, train a decision tree
3. At each split, consider random subset of features
4. Combine predictions by voting (classification) or averaging (regression)

#### Key Features:
- **Bootstrap Aggregating (Bagging)**: Random sampling with replacement
- **Random Subspace Method**: Random subset of features at each split
- **Out-of-Bag (OOB) Error**: Use samples not in bootstrap for validation

#### Implementation:
```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Classification
rf_clf = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=10,          # Max depth of each tree
    min_samples_split=5,   # Min samples to split
    min_samples_leaf=2,    # Min samples in leaf
    max_features='sqrt',   # Features to consider at each split
    random_state=42
)

rf_clf.fit(X_train, y_train)
y_pred = rf_clf.predict(X_test)

# Feature importance
importances = rf_clf.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)
```

#### Hyperparameter Tuning:
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_
```

#### Advantages:
- ✅ Reduces overfitting compared to single trees
- ✅ Provides feature importance
- ✅ Handles missing values well
- ✅ Works well out-of-the-box
- ✅ Can estimate uncertainty via prediction variance

#### Disadvantages:
- ❌ Less interpretable than single trees
- ❌ Can overfit with very noisy data
- ❌ Biased toward categorical variables with many categories

### 3.3 Gradient Boosting 🔴

**Concept**: Build models sequentially, where each model corrects errors of previous models.

#### Algorithm:
1. Initialize with simple model (often just mean)
2. For each iteration:
   - Calculate residuals (errors) from current predictions
   - Train new model to predict residuals
   - Add new model to ensemble with small learning rate
3. Final prediction is sum of all models

#### Mathematical Foundation:
```
F₀(x) = initial prediction
For m = 1 to M:
    rᵢₘ = -[∂L(yᵢ, F(xᵢ))/∂F(xᵢ)]  # Negative gradient
    hₘ(x) = model trained on residuals rᵢₘ
    Fₘ(x) = Fₘ₋₁(x) + ν × hₘ(x)    # ν is learning rate
```

#### XGBoost Implementation:
```python
import xgboost as xgb
from sklearn.metrics import accuracy_score

# Classification
xgb_clf = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb_clf.fit(X_train, y_train)
y_pred = xgb_clf.predict(X_test)

# Regression
xgb_reg = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

xgb_reg.fit(X_train, y_train)
```

#### LightGBM Implementation:
```python
import lightgbm as lgb

# Classification
lgb_clf = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

lgb_clf.fit(X_train, y_train)
```

#### Key Hyperparameters:
- **n_estimators**: Number of boosting rounds
- **learning_rate**: Step size shrinkage
- **max_depth**: Maximum tree depth
- **subsample**: Fraction of samples used for each tree
- **colsample_bytree**: Fraction of features used for each tree

#### Advantages:
- ✅ Often achieves best performance
- ✅ Handles various data types well
- ✅ Built-in regularization
- ✅ Feature importance available

#### Disadvantages:
- ❌ Many hyperparameters to tune
- ❌ Can easily overfit
- ❌ Sensitive to outliers
- ❌ Requires more computational resources

---

## 4. Instance-Based Learning

### 4.1 K-Nearest Neighbors (k-NN) 🟢

**Concept**: Classify/predict based on the k closest training examples.

#### Algorithm:
1. Calculate distance from test point to all training points
2. Find k nearest neighbors
3. For classification: Vote by majority class
4. For regression: Average the target values

#### Distance Metrics:

**Euclidean Distance:**
```
d(x, y) = √Σ(xᵢ - yᵢ)²
```

**Manhattan Distance:**
```
d(x, y) = Σ|xᵢ - yᵢ|
```

**Minkowski Distance:**
```
d(x, y) = (Σ|xᵢ - yᵢ|ᵖ)^(1/p)
```

#### Implementation:
```python
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

# Important: Scale features for k-NN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Classification
knn_clf = KNeighborsClassifier(
    n_neighbors=5,
    weights='uniform',  # or 'distance'
    metric='euclidean'
)

knn_clf.fit(X_train_scaled, y_train)
y_pred = knn_clf.predict(X_test_scaled)

# Find optimal k
from sklearn.model_selection import cross_val_score

k_values = range(1, 21)
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5)
    cv_scores.append(scores.mean())

optimal_k = k_values[np.argmax(cv_scores)]
```

#### From Scratch Implementation:
```python
class KNNClassifier:
    def __init__(self, k=5):
        self.k = k
    
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        predictions = []
        for x in X:
            # Calculate distances to all training points
            distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
            
            # Get k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            
            # Vote by majority
            most_common = max(set(k_nearest_labels), key=k_nearest_labels.count)
            predictions.append(most_common)
        
        return np.array(predictions)
```

#### Advantages:
- ✅ Simple and intuitive
- ✅ No assumptions about data distribution
- ✅ Works well with small datasets
- ✅ Can handle multi-class naturally
- ✅ Good baseline algorithm

#### Disadvantages:
- ❌ Computationally expensive for large datasets
- ❌ Sensitive to irrelevant features
- ❌ Requires feature scaling
- ❌ Poor performance in high dimensions
- ❌ Sensitive to local structure of data

---

## 5. Support Vector Machines (SVM)

### 5.1 Linear SVM 🟡

**Concept**: Find the optimal hyperplane that separates classes with maximum margin.

#### Key Ideas:
- **Hyperplane**: Decision boundary (line in 2D, plane in 3D, etc.)
- **Margin**: Distance between hyperplane and nearest data points
- **Support Vectors**: Data points closest to hyperplane
- **Maximum Margin**: Find hyperplane with largest margin

#### Mathematical Formulation:
```
Objective: Minimize ||w||²/2
Subject to: yᵢ(wᵀxᵢ + b) ≥ 1 for all i
```

#### Implementation:
```python
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler

# Feature scaling is crucial for SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Classification
svm_clf = SVC(
    kernel='linear',
    C=1.0,           # Regularization parameter
    random_state=42
)

svm_clf.fit(X_train_scaled, y_train)
y_pred = svm_clf.predict(X_test_scaled)

# Get support vectors
support_vectors = svm_clf.support_vectors_
n_support = svm_clf.n_support_
```

### 5.2 Non-Linear SVM (Kernel Trick) 🔴

**Concept**: Map data to higher-dimensional space where it becomes linearly separable.

#### Common Kernels:

**Polynomial Kernel:**
```
K(x, y) = (γxᵀy + r)ᵈ
```

**RBF (Gaussian) Kernel:**
```
K(x, y) = exp(-γ||x - y||²)
```

**Sigmoid Kernel:**
```
K(x, y) = tanh(γxᵀy + r)
```

#### Implementation:
```python
# RBF kernel SVM
svm_rbf = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',   # or 'auto', or specific value
    random_state=42
)

svm_rbf.fit(X_train_scaled, y_train)

# Polynomial kernel SVM
svm_poly = SVC(
    kernel='poly',
    degree=3,
    C=1.0,
    random_state=42
)

svm_poly.fit(X_train_scaled, y_train)
```

#### Hyperparameter Tuning:
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
}

grid_search = GridSearchCV(SVC(random_state=42), param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)
best_svm = grid_search.best_estimator_
```

#### Advantages:
- ✅ Effective in high-dimensional spaces
- ✅ Memory efficient (uses support vectors)
- ✅ Versatile (different kernels)
- ✅ Works well with clear margin of separation

#### Disadvantages:
- ❌ Poor performance on large datasets
- ❌ Sensitive to feature scaling
- ❌ No probabilistic output
- ❌ Sensitive to noise

---

## 6. Naive Bayes

### 6.1 Bayes' Theorem Foundation 🟢

**Bayes' Theorem:**
```
P(A|B) = P(B|A) × P(A) / P(B)
```

**For Classification:**
```
P(class|features) = P(features|class) × P(class) / P(features)
```

#### Naive Assumption:
Features are conditionally independent given the class:
```
P(x₁, x₂, ..., xₙ|class) = P(x₁|class) × P(x₂|class) × ... × P(xₙ|class)
```

### 6.2 Types of Naive Bayes 🟢

#### Gaussian Naive Bayes:
For continuous features assuming normal distribution:
```python
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
y_pred_proba = gnb.predict_proba(X_test)
```

#### Multinomial Naive Bayes:
For discrete count features (e.g., word counts):
```python
from sklearn.naive_bayes import MultinomialNB

mnb = MultinomialNB(alpha=1.0)  # Laplace smoothing
mnb.fit(X_train, y_train)
```

#### Bernoulli Naive Bayes:
For binary features:
```python
from sklearn.naive_bayes import BernoulliNB

bnb = BernoulliNB(alpha=1.0)
bnb.fit(X_train, y_train)
```

#### From Scratch Implementation:
```python
class GaussianNaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = {}
        self.var = {}
        self.priors = {}
        
        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = np.mean(X_c, axis=0)
            self.var[c] = np.var(X_c, axis=0)
            self.priors[c] = len(X_c) / len(X)
    
    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = []
            for c in self.classes:
                # Calculate likelihood
                likelihood = np.prod(
                    (1 / np.sqrt(2 * np.pi * self.var[c])) * 
                    np.exp(-0.5 * ((x - self.mean[c]) ** 2) / self.var[c])
                )
                # Calculate posterior
                posterior = likelihood * self.priors[c]
                posteriors.append(posterior)
            
            predictions.append(self.classes[np.argmax(posteriors)])
        
        return np.array(predictions)
```

#### Advantages:
- ✅ Fast and simple
- ✅ Works well with small datasets
- ✅ Not sensitive to irrelevant features
- ✅ Good baseline for text classification
- ✅ Handles multi-class naturally

#### Disadvantages:
- ❌ Strong independence assumption
- ❌ Can be outperformed by more sophisticated methods
- ❌ Requires smoothing for zero probabilities

---

## 7. Model Evaluation and Selection

### 7.1 Classification Metrics 🟢

#### Confusion Matrix:
```
                 Predicted
              Positive  Negative
Actual Pos      TP       FN
       Neg      FP       TN
```

#### Key Metrics:
```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

# Basic metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Comprehensive report
report = classification_report(y_true, y_pred)
```

#### Detailed Metrics:

**Accuracy:**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Precision:**
```
Precision = TP / (TP + FP)
```

**Recall (Sensitivity):**
```
Recall = TP / (TP + FN)
```

**F1-Score:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Specificity:**
```
Specificity = TN / (TN + FP)
```

#### ROC Curve and AUC:
```python
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Get probabilities
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
```

#### Precision-Recall Curve:
```python
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

plt.figure()
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()
```

### 7.2 Regression Metrics 🟢

#### Common Metrics:
```python
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
mape = mean_absolute_percentage_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MAPE: {mape:.4f}")
print(f"R²: {r2:.4f}")
```

#### Metric Definitions:

**Mean Squared Error (MSE):**
```
MSE = (1/n) Σ(yᵢ - ŷᵢ)²
```

**Root Mean Squared Error (RMSE):**
```
RMSE = √MSE
```

**Mean Absolute Error (MAE):**
```
MAE = (1/n) Σ|yᵢ - ŷᵢ|
```

**R-squared (Coefficient of Determination):**
```
R² = 1 - (SS_res / SS_tot)
where SS_res = Σ(yᵢ - ŷᵢ)² and SS_tot = Σ(yᵢ - ȳ)²
```

### 7.3 Cross-Validation 🟡

#### K-Fold Cross-Validation:
```python
from sklearn.model_selection import cross_val_score, KFold

# K-fold CV
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

print(f"CV Scores: {scores}")
print(f"Mean CV Score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
```

#### Stratified K-Fold:
```python
from sklearn.model_selection import StratifiedKFold

# Maintains class distribution
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skfold, scoring='accuracy')
```

#### Leave-One-Out Cross-Validation:
```python
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
scores = cross_val_score(model, X, y, cv=loo, scoring='accuracy')
```

### 7.4 Hyperparameter Tuning 🟡

#### Grid Search:
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

grid_search = GridSearchCV(
    SVC(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_
```

#### Random Search:
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

param_dist = {
    'C': uniform(0.1, 100),
    'kernel': ['linear', 'rbf'],
    'gamma': uniform(0.001, 1)
}

random_search = RandomizedSearchCV(
    SVC(random_state=42),
    param_dist,
    n_iter=100,
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)
```

#### Bayesian Optimization:
```python
from skopt import BayesSearchCV
from skopt.space import Real, Categorical

search_spaces = {
    'C': Real(0.1, 100, prior='log-uniform'),
    'kernel': Categorical(['linear', 'rbf']),
    'gamma': Real(0.001, 1, prior='log-uniform')
}

bayes_search = BayesSearchCV(
    SVC(random_state=42),
    search_spaces,
    n_iter=50,
    cv=5,
    scoring='accuracy',
    random_state=42
)

bayes_search.fit(X_train, y_train)
```

---

## 8. Bias-Variance Tradeoff

### 8.1 Understanding Bias and Variance 🟡

#### Definitions:
- **Bias**: Error from overly simplistic assumptions
- **Variance**: Error from sensitivity to small fluctuations in training set
- **Noise**: Irreducible error in the data

#### Total Error Decomposition:
```
Total Error = Bias² + Variance + Noise
```

#### Bias-Variance Examples:
- **High Bias, Low Variance**: Linear models on non-linear data
- **Low Bias, High Variance**: Complex models (deep trees, k-NN with small k)
- **Optimal**: Balance between bias and variance

### 8.2 Diagnosing Bias-Variance 🟡

#### Learning Curves:
```python
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

def plot_learning_curve(model, X, y, title):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10)
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure()
    plt.plot(train_sizes, train_mean, 'o-', label='Training score')
    plt.plot(train_sizes, val_mean, 'o-', label='Validation score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
    plt.xlabel('Training Set Size')
    plt.ylabel('Score')
    plt.title(title)
    plt.legend()
    plt.show()

# Example usage
plot_learning_curve(LinearRegression(), X, y, 'Linear Regression Learning Curve')
```

#### Validation Curves:
```python
from sklearn.model_selection import validation_curve

param_range = [1, 5, 10, 15, 20, 25, 30]
train_scores, val_scores = validation_curve(
    DecisionTreeClassifier(random_state=42), X, y,
    param_name='max_depth', param_range=param_range,
    cv=5, scoring='accuracy'
)

train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)

plt.figure()
plt.plot(param_range, train_mean, 'o-', label='Training score')
plt.plot(param_range, val_mean, 'o-', label='Validation score')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Validation Curve for Decision Tree')
plt.legend()
plt.show()
```

---

## 🎯 Key Takeaways

### Algorithm Selection Guide:

#### For Classification:
- **Small dataset, interpretability needed**: Logistic Regression, Naive Bayes
- **Non-linear relationships**: SVM with RBF kernel, Random Forest
- **Large dataset, high performance**: Gradient Boosting (XGBoost, LightGBM)
- **Text data**: Naive Bayes, Logistic Regression
- **Image data**: Neural Networks, SVM

#### For Regression:
- **Linear relationships**: Linear Regression, Ridge/Lasso
- **Non-linear relationships**: Random Forest, Gradient Boosting
- **Small dataset**: k-NN, Gaussian Process
- **Interpretability needed**: Linear models, Decision Trees

### Performance Optimization:
1. **Start simple**: Linear models, simple trees
2. **Feature engineering**: Often more important than algorithm choice
3. **Proper validation**: Use cross-validation, separate test set
4. **Hyperparameter tuning**: Grid search, random search, Bayesian optimization
5. **Ensemble methods**: Combine multiple models for better performance

---

## 📚 Next Steps

Continue your ML journey with:
- **[Unsupervised Learning](05_Unsupervised_Learning.md)** - Discover patterns without labels
- **[Model Evaluation & Selection](12_Model_Evaluation_Selection.md)** - Advanced evaluation techniques

---

## 🛠️ Practical Exercises

### Exercise 1: Algorithm Comparison
Compare 5 different algorithms on a classification dataset:
1. Implement each algorithm
2. Perform proper evaluation with cross-validation
3. Create learning curves for each
4. Analyze bias-variance tradeoff

### Exercise 2: Hyperparameter Optimization
For a Random Forest model:
1. Use grid search to find optimal parameters
2. Compare with random search
3. Analyze feature importance
4. Validate on test set

### Exercise 3: End-to-End Pipeline
Build complete supervised learning pipeline:
1. Data preprocessing
2. Feature engineering
3. Model selection
4. Hyperparameter tuning
5. Final evaluation and interpretation

---

*Next: [Unsupervised Learning →](05_Unsupervised_Learning.md)*
