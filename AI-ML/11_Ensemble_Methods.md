# 11. Ensemble Methods

## 🎯 Learning Objectives
- Understand the principles and benefits of ensemble learning
- Master bagging, boosting, and stacking techniques
- Learn to combine different types of models effectively
- Apply ensemble methods to improve prediction accuracy

---

## 1. Introduction to Ensemble Learning

**Ensemble Learning** combines multiple learning algorithms to create a more robust and accurate predictor than any individual algorithm alone.

### 1.1 Why Ensemble Methods Work? 🟢

#### Core Principles:
- **Wisdom of Crowds**: Multiple diverse opinions often lead to better decisions
- **Bias-Variance Trade-off**: Reduce overfitting and improve generalization
- **Error Reduction**: Different models make different types of errors
- **Robustness**: Less sensitive to outliers and noise

#### Mathematical Foundation:
For classification, if each classifier has error rate ε < 0.5, the ensemble error decreases exponentially with the number of classifiers.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb

def ensemble_error_rate(individual_error, n_classifiers):
    """Calculate ensemble error rate for majority voting"""
    error_rate = 0
    for i in range(n_classifiers // 2 + 1, n_classifiers + 1):
        error_rate += comb(n_classifiers, i) * (individual_error ** i) * ((1 - individual_error) ** (n_classifiers - i))
    return error_rate

# Visualize ensemble error improvement
individual_errors = np.linspace(0.1, 0.49, 100)
n_models = [3, 5, 7, 11, 21]

plt.figure(figsize=(12, 8))

for n in n_models:
    ensemble_errors = [ensemble_error_rate(err, n) for err in individual_errors]
    plt.plot(individual_errors, ensemble_errors, label=f'{n} models')

plt.plot(individual_errors, individual_errors, 'k--', label='Individual model', alpha=0.5)
plt.xlabel('Individual Model Error Rate')
plt.ylabel('Ensemble Error Rate')
plt.title('Ensemble Error Reduction with Majority Voting')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

#### Types of Ensemble Methods:
1. **Bagging** (Bootstrap Aggregating): Train models on different subsets
2. **Boosting**: Train models sequentially, focusing on errors
3. **Stacking**: Use a meta-model to combine predictions
4. **Voting**: Simple combination of predictions

### 1.2 Diversity in Ensembles 🟢

#### Sources of Diversity:
- **Data diversity**: Different training subsets
- **Model diversity**: Different algorithms
- **Feature diversity**: Different feature subsets
- **Parameter diversity**: Different hyperparameters

```python
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Create sample dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                          n_redundant=5, n_clusters_per_class=1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Measure diversity using disagreement
def calculate_disagreement(predictions1, predictions2):
    """Calculate disagreement between two prediction sets"""
    return np.mean(predictions1 != predictions2)

def analyze_model_diversity(models, X_test):
    """Analyze diversity among models"""
    predictions = {}
    
    for name, model in models.items():
        pred = model.predict(X_test)
        predictions[name] = pred
    
    # Calculate pairwise disagreement
    model_names = list(models.keys())
    disagreement_matrix = np.zeros((len(model_names), len(model_names)))
    
    for i, name1 in enumerate(model_names):
        for j, name2 in enumerate(model_names):
            if i != j:
                disagreement = calculate_disagreement(predictions[name1], predictions[name2])
                disagreement_matrix[i, j] = disagreement
    
    # Visualize disagreement matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(disagreement_matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Disagreement Rate')
    plt.xticks(range(len(model_names)), model_names, rotation=45)
    plt.yticks(range(len(model_names)), model_names)
    plt.title('Model Disagreement Matrix')
    
    for i in range(len(model_names)):
        for j in range(len(model_names)):
            plt.text(j, i, f'{disagreement_matrix[i, j]:.2f}', 
                    ha='center', va='center', color='white')
    
    plt.tight_layout()
    plt.show()
    
    return disagreement_matrix, predictions

# Train diverse models
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42).fit(X_train, y_train),
    'Logistic Regression': LogisticRegression(random_state=42).fit(X_train, y_train),
    'SVM': SVC(random_state=42).fit(X_train, y_train),
    'Naive Bayes': GaussianNB().fit(X_train, y_train),
    'KNN': KNeighborsClassifier().fit(X_train, y_train)
}

# Analyze diversity
disagreement_matrix, individual_predictions = analyze_model_diversity(models, X_test)

# Calculate individual accuracies
print("Individual Model Accuracies:")
for name, model in models.items():
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"{name}: {accuracy:.4f}")
```

---

## 2. Bagging Methods

### 2.1 Bootstrap Aggregating (Bagging) 🟢

#### From Scratch Implementation:
```python
class BaggingClassifier:
    def __init__(self, base_estimator, n_estimators=10, random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.estimators_ = []
        self.bootstrap_indices_ = []
        
    def _bootstrap_sample(self, X, y, random_state):
        """Create bootstrap sample"""
        n_samples = X.shape[0]
        np.random.seed(random_state)
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices], indices
    
    def fit(self, X, y):
        """Fit bagging ensemble"""
        self.estimators_ = []
        self.bootstrap_indices_ = []
        
        for i in range(self.n_estimators):
            # Create bootstrap sample
            seed = self.random_state + i if self.random_state else None
            X_boot, y_boot, indices = self._bootstrap_sample(X, y, seed)
            
            # Train base estimator
            estimator = self.base_estimator.__class__(**self.base_estimator.get_params())
            estimator.fit(X_boot, y_boot)
            
            self.estimators_.append(estimator)
            self.bootstrap_indices_.append(indices)
        
        return self
    
    def predict(self, X):
        """Make predictions using majority voting"""
        predictions = np.array([estimator.predict(X) for estimator in self.estimators_])
        
        # Majority voting
        final_predictions = []
        for i in range(X.shape[0]):
            votes = predictions[:, i]
            unique_votes, counts = np.unique(votes, return_counts=True)
            majority_vote = unique_votes[np.argmax(counts)]
            final_predictions.append(majority_vote)
        
        return np.array(final_predictions)
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        if not hasattr(self.estimators_[0], 'predict_proba'):
            raise AttributeError("Base estimator doesn't support probability prediction")
        
        probabilities = np.array([estimator.predict_proba(X) for estimator in self.estimators_])
        return np.mean(probabilities, axis=0)

# Test custom bagging classifier
custom_bagging = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=10,
    random_state=42
)

custom_bagging.fit(X_train, y_train)
custom_predictions = custom_bagging.predict(X_test)
custom_accuracy = accuracy_score(y_test, custom_predictions)

print(f"Custom Bagging Accuracy: {custom_accuracy:.4f}")

# Compare with sklearn's BaggingClassifier
from sklearn.ensemble import BaggingClassifier as SklearnBagging

sklearn_bagging = SklearnBagging(
    base_estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=10,
    random_state=42
)

sklearn_bagging.fit(X_train, y_train)
sklearn_predictions = sklearn_bagging.predict(X_test)
sklearn_accuracy = accuracy_score(y_test, sklearn_predictions)

print(f"Sklearn Bagging Accuracy: {sklearn_accuracy:.4f}")
```

### 2.2 Random Forest 🟡

#### Understanding Random Forest:
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

class CustomRandomForest:
    def __init__(self, n_estimators=100, max_features='sqrt', 
                 max_depth=None, min_samples_split=2, random_state=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.trees_ = []
        self.feature_indices_ = []
    
    def _get_n_features(self, n_total_features):
        """Calculate number of features to use"""
        if self.max_features == 'sqrt':
            return int(np.sqrt(n_total_features))
        elif self.max_features == 'log2':
            return int(np.log2(n_total_features))
        elif isinstance(self.max_features, int):
            return self.max_features
        elif isinstance(self.max_features, float):
            return int(self.max_features * n_total_features)
        else:
            return n_total_features
    
    def fit(self, X, y):
        """Fit Random Forest"""
        n_samples, n_features = X.shape
        n_features_to_use = self._get_n_features(n_features)
        
        self.trees_ = []
        self.feature_indices_ = []
        
        for i in range(self.n_estimators):
            # Bootstrap sampling
            np.random.seed(self.random_state + i if self.random_state else None)
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            
            # Random feature selection
            feature_indices = np.random.choice(n_features, size=n_features_to_use, replace=False)
            
            # Prepare data
            X_boot = X[bootstrap_indices][:, feature_indices]
            y_boot = y[bootstrap_indices]
            
            # Train decision tree
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state + i if self.random_state else None
            )
            
            tree.fit(X_boot, y_boot)
            
            self.trees_.append(tree)
            self.feature_indices_.append(feature_indices)
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        tree_predictions = []
        
        for tree, feature_indices in zip(self.trees_, self.feature_indices_):
            X_subset = X[:, feature_indices]
            predictions = tree.predict(X_subset)
            tree_predictions.append(predictions)
        
        tree_predictions = np.array(tree_predictions)
        
        # Majority voting
        final_predictions = []
        for i in range(X.shape[0]):
            votes = tree_predictions[:, i]
            unique_votes, counts = np.unique(votes, return_counts=True)
            majority_vote = unique_votes[np.argmax(counts)]
            final_predictions.append(majority_vote)
        
        return np.array(final_predictions)
    
    def feature_importances_(self):
        """Calculate feature importances"""
        n_features = len(self.feature_indices_[0]) if self.feature_indices_ else 0
        importances = np.zeros(n_features)
        
        for tree, feature_indices in zip(self.trees_, self.feature_indices_):
            tree_importances = tree.feature_importances_
            for i, feature_idx in enumerate(feature_indices):
                importances[feature_idx] += tree_importances[i]
        
        return importances / self.n_estimators

# Compare Random Forest implementations
custom_rf = CustomRandomForest(n_estimators=100, random_state=42)
custom_rf.fit(X_train, y_train)
custom_rf_pred = custom_rf.predict(X_test)
custom_rf_accuracy = accuracy_score(y_test, custom_rf_pred)

sklearn_rf = RandomForestClassifier(n_estimators=100, random_state=42)
sklearn_rf.fit(X_train, y_train)
sklearn_rf_pred = sklearn_rf.predict(X_test)
sklearn_rf_accuracy = accuracy_score(y_test, sklearn_rf_pred)

print(f"Custom Random Forest Accuracy: {custom_rf_accuracy:.4f}")
print(f"Sklearn Random Forest Accuracy: {sklearn_rf_accuracy:.4f}")

# Feature importance comparison
if X.shape[1] <= 20:  # Only plot if reasonable number of features
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.bar(range(len(sklearn_rf.feature_importances_)), sklearn_rf.feature_importances_)
    plt.title('Sklearn Random Forest Feature Importances')
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    
    plt.subplot(1, 2, 2)
    single_tree = DecisionTreeClassifier(random_state=42)
    single_tree.fit(X_train, y_train)
    plt.bar(range(len(single_tree.feature_importances_)), single_tree.feature_importances_)
    plt.title('Single Decision Tree Feature Importances')
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    
    plt.tight_layout()
    plt.show()
```

### 2.3 Extra Trees (Extremely Randomized Trees) 🟡

```python
from sklearn.ensemble import ExtraTreesClassifier

def compare_tree_ensembles(X_train, y_train, X_test, y_test):
    """Compare different tree ensemble methods"""
    
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Bagging': BaggingClassifier(
            base_estimator=DecisionTreeClassifier(random_state=42),
            n_estimators=100, random_state=42
        ),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        results[name] = accuracy
        print(f"{name}: {accuracy:.4f}")
    
    # Visualize results
    plt.figure(figsize=(10, 6))
    model_names = list(results.keys())
    accuracies = list(results.values())
    
    bars = plt.bar(model_names, accuracies, color=['red', 'orange', 'green', 'blue'])
    plt.title('Comparison of Tree Ensemble Methods')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, accuracy in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{accuracy:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return results

# Compare tree ensemble methods
ensemble_results = compare_tree_ensembles(X_train, y_train, X_test, y_test)
```

---

## 3. Boosting Methods

### 3.1 AdaBoost (Adaptive Boosting) 🟡

#### From Scratch Implementation:
```python
class AdaBoostClassifier:
    def __init__(self, base_estimator=None, n_estimators=50, learning_rate=1.0, random_state=None):
        self.base_estimator = base_estimator or DecisionTreeClassifier(max_depth=1)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        self.estimators_ = []
        self.estimator_weights_ = []
        self.estimator_errors_ = []
        
    def fit(self, X, y):
        """Fit AdaBoost classifier"""
        n_samples = X.shape[0]
        
        # Initialize sample weights
        sample_weights = np.ones(n_samples) / n_samples
        
        self.estimators_ = []
        self.estimator_weights_ = []
        self.estimator_errors_ = []
        
        for iboost in range(self.n_estimators):
            # Train weak learner
            estimator = self._make_estimator(random_state=self.random_state)
            estimator.fit(X, y, sample_weight=sample_weights)
            
            # Make predictions
            y_predict = estimator.predict(X)
            
            # Calculate weighted error
            incorrect = y_predict != y
            estimator_error = np.dot(sample_weights, incorrect) / np.sum(sample_weights)
            
            # If perfect classifier, stop
            if estimator_error <= 0:
                if len(self.estimators_) == 0:
                    self.estimators_.append(estimator)
                    self.estimator_weights_.append(1.0)
                    self.estimator_errors_.append(estimator_error)
                break
            
            # If worse than random, stop
            if estimator_error >= 0.5:
                if len(self.estimators_) == 0:
                    # If first estimator is worse than random, just use it
                    self.estimators_.append(estimator)
                    self.estimator_weights_.append(1.0)
                    self.estimator_errors_.append(estimator_error)
                break
            
            # Calculate alpha (estimator weight)
            alpha = self.learning_rate * 0.5 * np.log((1 - estimator_error) / estimator_error)
            
            # Store estimator
            self.estimators_.append(estimator)
            self.estimator_weights_.append(alpha)
            self.estimator_errors_.append(estimator_error)
            
            # Update sample weights
            sample_weights *= np.exp(-alpha * y * y_predict)
            sample_weights /= np.sum(sample_weights)  # Normalize
        
        return self
    
    def _make_estimator(self, random_state=None):
        """Create a copy of base estimator"""
        estimator = self.base_estimator.__class__(**self.base_estimator.get_params())
        if hasattr(estimator, 'random_state'):
            estimator.random_state = random_state
        return estimator
    
    def predict(self, X):
        """Make predictions using weighted majority voting"""
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)
        
        for estimator, weight in zip(self.estimators_, self.estimator_weights_):
            estimator_predictions = estimator.predict(X)
            predictions += weight * estimator_predictions
        
        return np.sign(predictions).astype(int)
    
    def staged_predict(self, X):
        """Return staged predictions for visualization"""
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)
        
        for estimator, weight in zip(self.estimators_, self.estimator_weights_):
            estimator_predictions = estimator.predict(X)
            predictions += weight * estimator_predictions
            yield np.sign(predictions).astype(int)

# Convert labels to -1, 1 for AdaBoost
y_train_ada = np.where(y_train == 0, -1, 1)
y_test_ada = np.where(y_test == 0, -1, 1)

# Train custom AdaBoost
custom_ada = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)

custom_ada.fit(X_train, y_train_ada)
custom_ada_pred = custom_ada.predict(X_test)
custom_ada_accuracy = accuracy_score(y_test_ada, custom_ada_pred)

print(f"Custom AdaBoost Accuracy: {custom_ada_accuracy:.4f}")

# Compare with sklearn AdaBoost
from sklearn.ensemble import AdaBoostClassifier as SklearnAdaBoost

sklearn_ada = SklearnAdaBoost(
    base_estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)

sklearn_ada.fit(X_train, y_train)
sklearn_ada_pred = sklearn_ada.predict(X_test)
sklearn_ada_accuracy = accuracy_score(y_test, sklearn_ada_pred)

print(f"Sklearn AdaBoost Accuracy: {sklearn_ada_accuracy:.4f}")

# Visualize learning process
plt.figure(figsize=(15, 5))

# Plot estimator weights
plt.subplot(1, 3, 1)
plt.plot(custom_ada.estimator_weights_)
plt.title('Estimator Weights Over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Weight (Alpha)')
plt.grid(True, alpha=0.3)

# Plot estimator errors
plt.subplot(1, 3, 2)
plt.plot(custom_ada.estimator_errors_)
plt.axhline(y=0.5, color='red', linestyle='--', label='Random Guess')
plt.title('Estimator Errors Over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Weighted Error')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot staged accuracy
plt.subplot(1, 3, 3)
staged_accuracies = []
for staged_pred in custom_ada.staged_predict(X_test):
    staged_accuracy = accuracy_score(y_test_ada, staged_pred)
    staged_accuracies.append(staged_accuracy)

plt.plot(staged_accuracies)
plt.title('Staged Test Accuracy')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 3.2 Gradient Boosting 🔴

#### Gradient Boosting from Scratch:
```python
class GradientBoostingClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        
        self.estimators_ = []
        self.init_prediction_ = None
        
    def _sigmoid(self, x):
        """Sigmoid function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _log_loss_gradient(self, y, y_pred):
        """Compute gradient of log loss"""
        p = self._sigmoid(y_pred)
        return y - p
    
    def fit(self, X, y):
        """Fit gradient boosting classifier"""
        # Initialize with log-odds
        self.init_prediction_ = np.log(np.mean(y) / (1 - np.mean(y)))
        
        # Initialize predictions
        F = np.full(len(y), self.init_prediction_)
        
        self.estimators_ = []
        
        for m in range(self.n_estimators):
            # Compute residuals (negative gradient)
            residuals = self._log_loss_gradient(y, F)
            
            # Fit regression tree to residuals
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                random_state=self.random_state + m if self.random_state else None
            )
            tree.fit(X, residuals)
            
            # Make predictions
            predictions = tree.predict(X)
            
            # Update F with learning rate
            F += self.learning_rate * predictions
            
            self.estimators_.append(tree)
        
        return self
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        # Start with initial prediction
        F = np.full(X.shape[0], self.init_prediction_)
        
        # Add predictions from all trees
        for tree in self.estimators_:
            F += self.learning_rate * tree.predict(X)
        
        # Convert to probabilities
        probabilities = self._sigmoid(F)
        return np.column_stack([1 - probabilities, probabilities])
    
    def predict(self, X):
        """Make binary predictions"""
        probabilities = self.predict_proba(X)
        return (probabilities[:, 1] > 0.5).astype(int)

# Train custom gradient boosting
from sklearn.tree import DecisionTreeRegressor

custom_gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

custom_gb.fit(X_train, y_train)
custom_gb_pred = custom_gb.predict(X_test)
custom_gb_accuracy = accuracy_score(y_test, custom_gb_pred)

print(f"Custom Gradient Boosting Accuracy: {custom_gb_accuracy:.4f}")

# Compare with sklearn
from sklearn.ensemble import GradientBoostingClassifier as SklearnGB

sklearn_gb = SklearnGB(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

sklearn_gb.fit(X_train, y_train)
sklearn_gb_pred = sklearn_gb.predict(X_test)
sklearn_gb_accuracy = accuracy_score(y_test, sklearn_gb_pred)

print(f"Sklearn Gradient Boosting Accuracy: {sklearn_gb_accuracy:.4f}")
```

### 3.3 XGBoost 🔴

```python
# Note: Install XGBoost with: pip install xgboost
try:
    import xgboost as xgb
    xgboost_available = True
except ImportError:
    xgboost_available = False
    print("XGBoost not available. Install with: pip install xgboost")

if xgboost_available:
    # XGBoost implementation
    def tune_xgboost(X_train, y_train, X_val, y_val):
        """Tune XGBoost hyperparameters"""
        
        best_score = 0
        best_params = {}
        
        # Parameter grid
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        
        # Simple grid search
        for n_est in param_grid['n_estimators']:
            for depth in param_grid['max_depth']:
                for lr in param_grid['learning_rate']:
                    for subsample in param_grid['subsample']:
                        for colsample in param_grid['colsample_bytree']:
                            
                            model = xgb.XGBClassifier(
                                n_estimators=n_est,
                                max_depth=depth,
                                learning_rate=lr,
                                subsample=subsample,
                                colsample_bytree=colsample,
                                random_state=42,
                                eval_metric='logloss'
                            )
                            
                            model.fit(X_train, y_train)
                            score = model.score(X_val, y_val)
                            
                            if score > best_score:
                                best_score = score
                                best_params = {
                                    'n_estimators': n_est,
                                    'max_depth': depth,
                                    'learning_rate': lr,
                                    'subsample': subsample,
                                    'colsample_bytree': colsample
                                }
        
        return best_params, best_score
    
    # Split training data for validation
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Find best parameters (simplified for demo)
    best_xgb_params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }
    
    # Train XGBoost
    xgb_model = xgb.XGBClassifier(**best_xgb_params, random_state=42, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    
    xgb_pred = xgb_model.predict(X_test)
    xgb_accuracy = accuracy_score(y_test, xgb_pred)
    
    print(f"XGBoost Accuracy: {xgb_accuracy:.4f}")
    
    # Feature importance
    plt.figure(figsize=(10, 6))
    feature_importance = xgb_model.feature_importances_
    sorted_idx = np.argsort(feature_importance)[::-1][:10]  # Top 10 features
    
    plt.bar(range(len(sorted_idx)), feature_importance[sorted_idx])
    plt.title('XGBoost Feature Importance (Top 10)')
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.show()
```

---

## 4. Stacking (Stacked Generalization)

### 4.1 Basic Stacking 🔴

```python
from sklearn.model_selection import KFold
from sklearn.base import clone

class StackingClassifier:
    def __init__(self, base_estimators, meta_estimator, cv=5, use_probas=False):
        self.base_estimators = base_estimators
        self.meta_estimator = meta_estimator
        self.cv = cv
        self.use_probas = use_probas
        
        self.base_estimators_ = []
        self.meta_estimator_ = None
        
    def fit(self, X, y):
        """Fit stacking classifier"""
        
        # Prepare meta-features
        meta_features = self._prepare_meta_features(X, y)
        
        # Train base estimators on full data
        self.base_estimators_ = []
        for estimator in self.base_estimators:
            fitted_estimator = clone(estimator)
            fitted_estimator.fit(X, y)
            self.base_estimators_.append(fitted_estimator)
        
        # Train meta-estimator
        self.meta_estimator_ = clone(self.meta_estimator)
        self.meta_estimator_.fit(meta_features, y)
        
        return self
    
    def _prepare_meta_features(self, X, y):
        """Prepare meta-features using cross-validation"""
        
        n_samples = X.shape[0]
        n_base_estimators = len(self.base_estimators)
        
        if self.use_probas:
            # Assume binary classification
            meta_features = np.zeros((n_samples, n_base_estimators * 2))
        else:
            meta_features = np.zeros((n_samples, n_base_estimators))
        
        # K-fold cross-validation
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        
        for train_idx, val_idx in kf.split(X):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold = y[train_idx]
            
            for i, estimator in enumerate(self.base_estimators):
                # Train base estimator on fold
                fitted_estimator = clone(estimator)
                fitted_estimator.fit(X_train_fold, y_train_fold)
                
                # Make predictions on validation fold
                if self.use_probas and hasattr(fitted_estimator, 'predict_proba'):
                    pred = fitted_estimator.predict_proba(X_val_fold)
                    meta_features[val_idx, i*2:(i+1)*2] = pred
                else:
                    pred = fitted_estimator.predict(X_val_fold)
                    meta_features[val_idx, i] = pred
        
        return meta_features
    
    def predict(self, X):
        """Make predictions using stacked model"""
        # Get base estimator predictions
        base_predictions = self._get_base_predictions(X)
        
        # Meta-estimator prediction
        return self.meta_estimator_.predict(base_predictions)
    
    def predict_proba(self, X):
        """Predict probabilities using stacked model"""
        base_predictions = self._get_base_predictions(X)
        
        if hasattr(self.meta_estimator_, 'predict_proba'):
            return self.meta_estimator_.predict_proba(base_predictions)
        else:
            raise AttributeError("Meta-estimator doesn't support probability prediction")
    
    def _get_base_predictions(self, X):
        """Get predictions from base estimators"""
        n_samples = X.shape[0]
        
        if self.use_probas:
            base_predictions = np.zeros((n_samples, len(self.base_estimators_) * 2))
            for i, estimator in enumerate(self.base_estimators_):
                if hasattr(estimator, 'predict_proba'):
                    pred = estimator.predict_proba(X)
                    base_predictions[:, i*2:(i+1)*2] = pred
                else:
                    # Fallback to regular predictions
                    pred = estimator.predict(X)
                    base_predictions[:, i*2] = 1 - pred  # Inverse for class 0
                    base_predictions[:, i*2+1] = pred    # Class 1
        else:
            base_predictions = np.zeros((n_samples, len(self.base_estimators_)))
            for i, estimator in enumerate(self.base_estimators_):
                base_predictions[:, i] = estimator.predict(X)
        
        return base_predictions

# Define base estimators
base_estimators = [
    DecisionTreeClassifier(random_state=42),
    LogisticRegression(random_state=42),
    SVC(probability=True, random_state=42),
    RandomForestClassifier(n_estimators=50, random_state=42)
]

# Meta-estimator
meta_estimator = LogisticRegression(random_state=42)

# Create and train stacking classifier
stacking_clf = StackingClassifier(
    base_estimators=base_estimators,
    meta_estimator=meta_estimator,
    cv=5,
    use_probas=True
)

stacking_clf.fit(X_train, y_train)
stacking_pred = stacking_clf.predict(X_test)
stacking_accuracy = accuracy_score(y_test, stacking_pred)

print(f"Custom Stacking Accuracy: {stacking_accuracy:.4f}")

# Compare with sklearn StackingClassifier
from sklearn.ensemble import StackingClassifier as SklearnStacking

sklearn_stacking = SklearnStacking(
    estimators=[
        ('dt', DecisionTreeClassifier(random_state=42)),
        ('lr', LogisticRegression(random_state=42)),
        ('svc', SVC(probability=True, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42))
    ],
    final_estimator=LogisticRegression(random_state=42),
    cv=5
)

sklearn_stacking.fit(X_train, y_train)
sklearn_stacking_pred = sklearn_stacking.predict(X_test)
sklearn_stacking_accuracy = accuracy_score(y_test, sklearn_stacking_pred)

print(f"Sklearn Stacking Accuracy: {sklearn_stacking_accuracy:.4f}")
```

### 4.2 Multi-level Stacking 🔴

```python
class MultilevelStacking:
    def __init__(self, level_estimators, final_estimator, cv=5):
        """
        Multi-level stacking classifier
        
        Args:
            level_estimators: List of lists, each inner list contains estimators for that level
            final_estimator: Final meta-estimator
            cv: Cross-validation folds
        """
        self.level_estimators = level_estimators
        self.final_estimator = final_estimator
        self.cv = cv
        
        self.fitted_levels_ = []
        self.final_estimator_ = None
        
    def fit(self, X, y):
        """Fit multi-level stacking classifier"""
        
        current_features = X.copy()
        
        # Train each level
        for level_idx, level_estimators in enumerate(self.level_estimators):
            print(f"Training level {level_idx + 1} with {len(level_estimators)} estimators...")
            
            # Create stacking classifier for this level
            if level_idx == 0:
                # First level uses original features
                meta_estimator = LogisticRegression(random_state=42)
            else:
                # Subsequent levels use simpler meta-estimator
                meta_estimator = LogisticRegression(random_state=42)
            
            level_stacking = StackingClassifier(
                base_estimators=level_estimators,
                meta_estimator=meta_estimator,
                cv=self.cv,
                use_probas=True
            )
            
            level_stacking.fit(current_features, y)
            self.fitted_levels_.append(level_stacking)
            
            # Get meta-features for next level
            if level_idx < len(self.level_estimators) - 1:
                meta_features = level_stacking._prepare_meta_features(current_features, y)
                current_features = np.column_stack([current_features, meta_features])
        
        # Train final estimator
        final_meta_features = self.fitted_levels_[-1]._prepare_meta_features(
            current_features if len(self.level_estimators) > 1 else X, y
        )
        
        self.final_estimator_ = clone(self.final_estimator)
        self.final_estimator_.fit(final_meta_features, y)
        
        return self
    
    def predict(self, X):
        """Make predictions using multi-level stacking"""
        current_features = X.copy()
        
        # Pass through each level
        for level_idx, level_stacking in enumerate(self.fitted_levels_):
            if level_idx < len(self.fitted_levels_) - 1:
                # Get meta-features and add to current features
                meta_features = level_stacking._get_base_predictions(current_features)
                current_features = np.column_stack([current_features, meta_features])
        
        # Get final meta-features
        final_meta_features = self.fitted_levels_[-1]._get_base_predictions(
            current_features if len(self.fitted_levels_) > 1 else X
        )
        
        # Final prediction
        return self.final_estimator_.predict(final_meta_features)

# Define multi-level architecture
level_1_estimators = [
    DecisionTreeClassifier(max_depth=5, random_state=42),
    RandomForestClassifier(n_estimators=50, random_state=42),
    SVC(probability=True, random_state=42)
]

level_2_estimators = [
    LogisticRegression(random_state=42),
    GaussianNB()
]

final_estimator = RandomForestClassifier(n_estimators=100, random_state=42)

# Create and train multi-level stacking
multilevel_stacking = MultilevelStacking(
    level_estimators=[level_1_estimators, level_2_estimators],
    final_estimator=final_estimator,
    cv=3  # Reduce CV for demo
)

multilevel_stacking.fit(X_train, y_train)
multilevel_pred = multilevel_stacking.predict(X_test)
multilevel_accuracy = accuracy_score(y_test, multilevel_pred)

print(f"Multi-level Stacking Accuracy: {multilevel_accuracy:.4f}")
```

---

## 5. Voting Classifiers

### 5.1 Hard and Soft Voting 🟡

```python
from sklearn.ensemble import VotingClassifier

class CustomVotingClassifier:
    def __init__(self, estimators, voting='hard', weights=None):
        self.estimators = estimators
        self.voting = voting
        self.weights = weights
        self.fitted_estimators_ = []
        
    def fit(self, X, y):
        """Fit voting classifier"""
        self.fitted_estimators_ = []
        
        for name, estimator in self.estimators:
            fitted_estimator = clone(estimator)
            fitted_estimator.fit(X, y)
            self.fitted_estimators_.append((name, fitted_estimator))
        
        return self
    
    def predict(self, X):
        """Make predictions using voting"""
        if self.voting == 'hard':
            return self._predict_hard(X)
        else:
            return self._predict_soft(X)
    
    def _predict_hard(self, X):
        """Hard voting - majority vote"""
        predictions = np.array([estimator.predict(X) for _, estimator in self.fitted_estimators_])
        
        if self.weights is not None:
            # Weighted voting
            weighted_predictions = []
            for i in range(X.shape[0]):
                votes = {}
                for j, pred in enumerate(predictions[:, i]):
                    weight = self.weights[j] if self.weights else 1
                    votes[pred] = votes.get(pred, 0) + weight
                
                # Get class with highest weighted vote
                best_class = max(votes.keys(), key=lambda k: votes[k])
                weighted_predictions.append(best_class)
            
            return np.array(weighted_predictions)
        else:
            # Simple majority vote
            final_predictions = []
            for i in range(X.shape[0]):
                unique_predictions, counts = np.unique(predictions[:, i], return_counts=True)
                majority_vote = unique_predictions[np.argmax(counts)]
                final_predictions.append(majority_vote)
            
            return np.array(final_predictions)
    
    def _predict_soft(self, X):
        """Soft voting - average probabilities"""
        probabilities = []
        
        for name, estimator in self.fitted_estimators_:
            if hasattr(estimator, 'predict_proba'):
                proba = estimator.predict_proba(X)
            else:
                # Convert predictions to probabilities
                pred = estimator.predict(X)
                proba = np.eye(len(np.unique(pred)))[pred]
            
            probabilities.append(proba)
        
        probabilities = np.array(probabilities)
        
        if self.weights is not None:
            # Weighted average
            weights = np.array(self.weights).reshape(-1, 1, 1)
            avg_probabilities = np.average(probabilities, axis=0, weights=weights.flatten())
        else:
            # Simple average
            avg_probabilities = np.mean(probabilities, axis=0)
        
        return np.argmax(avg_probabilities, axis=1)
    
    def predict_proba(self, X):
        """Predict probabilities using soft voting"""
        if self.voting != 'soft':
            raise ValueError("predict_proba only available for soft voting")
        
        probabilities = []
        
        for name, estimator in self.fitted_estimators_:
            if hasattr(estimator, 'predict_proba'):
                proba = estimator.predict_proba(X)
            else:
                raise ValueError(f"Estimator {name} doesn't support predict_proba")
            
            probabilities.append(proba)
        
        probabilities = np.array(probabilities)
        
        if self.weights is not None:
            weights = np.array(self.weights).reshape(-1, 1, 1)
            return np.average(probabilities, axis=0, weights=weights.flatten())
        else:
            return np.mean(probabilities, axis=0)

# Create voting classifiers
estimators = [
    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
    ('svc', SVC(probability=True, random_state=42)),
    ('lr', LogisticRegression(random_state=42)),
    ('nb', GaussianNB())
]

# Hard voting
hard_voting = CustomVotingClassifier(estimators=estimators, voting='hard')
hard_voting.fit(X_train, y_train)
hard_voting_pred = hard_voting.predict(X_test)
hard_voting_accuracy = accuracy_score(y_test, hard_voting_pred)

# Soft voting
soft_voting = CustomVotingClassifier(estimators=estimators, voting='soft')
soft_voting.fit(X_train, y_train)
soft_voting_pred = soft_voting.predict(X_test)
soft_voting_accuracy = accuracy_score(y_test, soft_voting_pred)

# Weighted soft voting
weights = [0.3, 0.25, 0.25, 0.2]  # Higher weight for Random Forest
weighted_voting = CustomVotingClassifier(estimators=estimators, voting='soft', weights=weights)
weighted_voting.fit(X_train, y_train)
weighted_voting_pred = weighted_voting.predict(X_test)
weighted_voting_accuracy = accuracy_score(y_test, weighted_voting_pred)

print(f"Hard Voting Accuracy: {hard_voting_accuracy:.4f}")
print(f"Soft Voting Accuracy: {soft_voting_accuracy:.4f}")
print(f"Weighted Soft Voting Accuracy: {weighted_voting_accuracy:.4f}")

# Compare with sklearn VotingClassifier
sklearn_hard_voting = VotingClassifier(estimators=estimators, voting='hard')
sklearn_hard_voting.fit(X_train, y_train)
sklearn_hard_pred = sklearn_hard_voting.predict(X_test)
sklearn_hard_accuracy = accuracy_score(y_test, sklearn_hard_pred)

sklearn_soft_voting = VotingClassifier(estimators=estimators, voting='soft')
sklearn_soft_voting.fit(X_train, y_train)
sklearn_soft_pred = sklearn_soft_voting.predict(X_test)
sklearn_soft_accuracy = accuracy_score(y_test, sklearn_soft_pred)

print(f"\nSklearn Hard Voting Accuracy: {sklearn_hard_accuracy:.4f}")
print(f"Sklearn Soft Voting Accuracy: {sklearn_soft_accuracy:.4f}")
```

---

## 6. Ensemble Selection and Optimization

### 6.1 Dynamic Ensemble Selection 🔴

```python
class DynamicEnsembleSelection:
    def __init__(self, base_estimators, k_neighbors=5, selection_method='best'):
        """
        Dynamic Ensemble Selection
        
        Args:
            base_estimators: List of base estimators
            k_neighbors: Number of neighbors to consider
            selection_method: 'best', 'all_accurate', or 'weighted'
        """
        self.base_estimators = base_estimators
        self.k_neighbors = k_neighbors
        self.selection_method = selection_method
        
        self.fitted_estimators_ = []
        self.X_val_ = None
        self.y_val_ = None
        
    def fit(self, X, y, X_val=None, y_val=None):
        """Fit dynamic ensemble selector"""
        
        # If validation data not provided, split training data
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        else:
            X_train, y_train = X, y
        
        # Store validation data for neighborhood search
        self.X_val_ = X_val
        self.y_val_ = y_val
        
        # Fit base estimators
        self.fitted_estimators_ = []
        self.estimator_predictions_ = []
        
        for estimator in self.base_estimators:
            fitted_estimator = clone(estimator)
            fitted_estimator.fit(X_train, y_train)
            
            # Get predictions on validation set
            val_predictions = fitted_estimator.predict(X_val)
            
            self.fitted_estimators_.append(fitted_estimator)
            self.estimator_predictions_.append(val_predictions)
        
        self.estimator_predictions_ = np.array(self.estimator_predictions_).T
        
        return self
    
    def predict(self, X):
        """Make predictions using dynamic selection"""
        from sklearn.neighbors import NearestNeighbors
        
        # Find k nearest neighbors in validation set
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors)
        nbrs.fit(self.X_val_)
        
        predictions = []
        
        for x in X:
            # Find neighbors
            distances, neighbor_indices = nbrs.kneighbors([x])
            neighbor_indices = neighbor_indices[0]
            
            # Calculate accuracy of each estimator on neighbors
            estimator_accuracies = []
            for i in range(len(self.fitted_estimators_)):
                neighbor_predictions = self.estimator_predictions_[neighbor_indices, i]
                neighbor_true = self.y_val_[neighbor_indices]
                accuracy = np.mean(neighbor_predictions == neighbor_true)
                estimator_accuracies.append(accuracy)
            
            # Select estimators based on method
            if self.selection_method == 'best':
                # Use best performing estimator
                best_estimator_idx = np.argmax(estimator_accuracies)
                prediction = self.fitted_estimators_[best_estimator_idx].predict([x])[0]
                
            elif self.selection_method == 'all_accurate':
                # Use all estimators with accuracy > threshold
                threshold = 0.5
                accurate_estimators = np.where(np.array(estimator_accuracies) > threshold)[0]
                
                if len(accurate_estimators) == 0:
                    # Fallback to best estimator
                    best_estimator_idx = np.argmax(estimator_accuracies)
                    prediction = self.fitted_estimators_[best_estimator_idx].predict([x])[0]
                else:
                    # Majority vote among accurate estimators
                    votes = []
                    for idx in accurate_estimators:
                        vote = self.fitted_estimators_[idx].predict([x])[0]
                        votes.append(vote)
                    
                    unique_votes, counts = np.unique(votes, return_counts=True)
                    prediction = unique_votes[np.argmax(counts)]
                    
            elif self.selection_method == 'weighted':
                # Weighted vote based on local accuracy
                weighted_votes = {}
                for i, estimator in enumerate(self.fitted_estimators_):
                    vote = estimator.predict([x])[0]
                    weight = estimator_accuracies[i]
                    
                    if vote in weighted_votes:
                        weighted_votes[vote] += weight
                    else:
                        weighted_votes[vote] = weight
                
                prediction = max(weighted_votes.keys(), key=lambda k: weighted_votes[k])
            
            predictions.append(prediction)
        
        return np.array(predictions)

# Create dynamic ensemble selector
des = DynamicEnsembleSelection(
    base_estimators=[
        DecisionTreeClassifier(random_state=42),
        RandomForestClassifier(n_estimators=50, random_state=42),
        SVC(random_state=42),
        LogisticRegression(random_state=42)
    ],
    k_neighbors=7,
    selection_method='weighted'
)

des.fit(X_train, y_train)
des_pred = des.predict(X_test)
des_accuracy = accuracy_score(y_test, des_pred)

print(f"Dynamic Ensemble Selection Accuracy: {des_accuracy:.4f}")
```

---

## 🎯 Key Takeaways

### Ensemble Method Selection Guide:

#### Problem Characteristics:
- **High bias**: Use boosting methods (AdaBoost, Gradient Boosting, XGBoost)
- **High variance**: Use bagging methods (Random Forest, Extra Trees)
- **Complex patterns**: Use stacking with diverse base models
- **Need interpretability**: Use voting classifiers
- **Large datasets**: Use Random Forest or Extra Trees
- **Small datasets**: Use careful cross-validation and simpler ensembles

#### Performance Optimization:
1. **Diversity is key**: Use different algorithms, features, or data subsets
2. **Quality over quantity**: Better to have fewer good models than many poor ones
3. **Avoid overfitting**: Use proper validation and regularization
4. **Computational trade-offs**: Balance accuracy gains with training/prediction time
5. **Domain knowledge**: Some ensemble types work better for specific domains

### Best Practices:
- Start with simple ensembles (voting, bagging) before complex ones
- Use cross-validation to avoid overfitting in meta-learning
- Monitor individual model performance to identify weak learners
- Consider ensemble diversity when selecting base models
- Test different combination strategies (voting, stacking, etc.)

---

## 📚 Next Steps

Continue your journey with:
- **[Model Evaluation & Selection](12_Model_Evaluation_Selection.md)** - Advanced evaluation techniques
- **[ML Engineering & MLOps](13_ML_Engineering_MLOps.md)** - Production deployment

---

*Next: [Model Evaluation & Selection →](12_Model_Evaluation_Selection.md)*
