# 12. Model Evaluation & Selection

## 🎯 Learning Objectives
- Master comprehensive model evaluation techniques
- Learn advanced cross-validation strategies
- Understand bias-variance trade-off and model selection
- Apply statistical tests for model comparison
- Implement hyperparameter optimization strategies

---

## 1. Evaluation Fundamentals

**Model Evaluation** is the process of assessing how well a machine learning model performs on unseen data.

### 1.1 Train-Validation-Test Split 🟢

#### Data Splitting Strategy:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# Create comprehensive dataset
X, y = make_classification(
    n_samples=10000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_clusters_per_class=2,
    class_sep=0.8,
    random_state=42
)

def create_train_val_test_split(X, y, train_size=0.6, val_size=0.2, test_size=0.2, random_state=42):
    """Create train-validation-test split"""
    
    assert abs(train_size + val_size + test_size - 1.0) < 1e-7, "Sizes should sum to 1"
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: separate train and validation
    val_size_adjusted = val_size / (train_size + val_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# Create splits
X_train, X_val, X_test, y_train, y_val, y_test = create_train_val_test_split(X, y)

print(f"Dataset splits:")
print(f"Total samples: {len(X)}")
print(f"Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"Validation: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
print(f"Test: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

# Visualize class distribution
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, (data, title) in zip(axes, [(y_train, 'Train'), (y_val, 'Validation'), (y_test, 'Test')]):
    unique, counts = np.unique(data, return_counts=True)
    ax.bar(unique, counts, alpha=0.7)
    ax.set_title(f'{title} Set Class Distribution')
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    
    # Add percentage labels
    total = len(data)
    for i, count in enumerate(counts):
        ax.text(unique[i], count + total*0.01, f'{count/total*100:.1f}%', 
                ha='center', va='bottom')

plt.tight_layout()
plt.show()
```

### 1.2 Holdout vs Cross-Validation 🟢

#### Holdout Method Issues:
```python
def demonstrate_holdout_variance(X, y, n_experiments=50):
    """Demonstrate variance in holdout validation"""
    
    holdout_scores = []
    
    for i in range(n_experiments):
        # Different random splits
        X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(
            X, y, test_size=0.3, random_state=i
        )
        
        # Train simple model
        model = LogisticRegression(random_state=42)
        model.fit(X_train_temp, y_train_temp)
        score = model.score(X_test_temp, y_test_temp)
        holdout_scores.append(score)
    
    return holdout_scores

# Demonstrate holdout variance
holdout_scores = demonstrate_holdout_variance(X, y)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(holdout_scores, bins=15, alpha=0.7, edgecolor='black')
plt.axvline(np.mean(holdout_scores), color='red', linestyle='--', 
           label=f'Mean: {np.mean(holdout_scores):.3f}')
plt.axvline(np.median(holdout_scores), color='green', linestyle='--', 
           label=f'Median: {np.median(holdout_scores):.3f}')
plt.title('Distribution of Holdout Validation Scores')
plt.xlabel('Accuracy')
plt.ylabel('Frequency')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(holdout_scores, marker='o', alpha=0.7)
plt.axhline(np.mean(holdout_scores), color='red', linestyle='--', alpha=0.5)
plt.title('Holdout Scores Across Different Splits')
plt.xlabel('Experiment')
plt.ylabel('Accuracy')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Holdout validation statistics:")
print(f"Mean: {np.mean(holdout_scores):.4f}")
print(f"Std: {np.std(holdout_scores):.4f}")
print(f"Min: {np.min(holdout_scores):.4f}")
print(f"Max: {np.max(holdout_scores):.4f}")
print(f"Range: {np.max(holdout_scores) - np.min(holdout_scores):.4f}")
```

---

## 2. Cross-Validation Techniques

### 2.1 K-Fold Cross-Validation 🟢

#### Custom K-Fold Implementation:
```python
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.base import clone

class CustomKFoldCV:
    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
    
    def split(self, X, y=None):
        """Generate train/validation indices for each fold"""
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        if self.shuffle:
            np.random.seed(self.random_state)
            np.random.shuffle(indices)
        
        fold_size = n_samples // self.n_splits
        
        for i in range(self.n_splits):
            start_idx = i * fold_size
            end_idx = start_idx + fold_size if i < self.n_splits - 1 else n_samples
            
            val_indices = indices[start_idx:end_idx]
            train_indices = np.concatenate([indices[:start_idx], indices[end_idx:]])
            
            yield train_indices, val_indices
    
    def cross_val_score(self, estimator, X, y, scoring=None):
        """Perform cross-validation and return scores"""
        scores = []
        
        for train_idx, val_idx in self.split(X, y):
            # Split data
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Clone and train model
            model = clone(estimator)
            model.fit(X_train_fold, y_train_fold)
            
            # Calculate score
            if scoring is None or scoring == 'accuracy':
                score = model.score(X_val_fold, y_val_fold)
            elif scoring == 'f1':
                predictions = model.predict(X_val_fold)
                score = f1_score(y_val_fold, predictions, average='weighted')
            elif scoring == 'precision':
                predictions = model.predict(X_val_fold)
                score = precision_score(y_val_fold, predictions, average='weighted')
            elif scoring == 'recall':
                predictions = model.predict(X_val_fold)
                score = recall_score(y_val_fold, predictions, average='weighted')
            
            scores.append(score)
        
        return np.array(scores)

# Compare custom implementation with sklearn
from sklearn.model_selection import cross_val_score

# Custom K-Fold
custom_kfold = CustomKFoldCV(n_splits=5, shuffle=True, random_state=42)
custom_scores = custom_kfold.cross_val_score(
    LogisticRegression(random_state=42), X, y, scoring='accuracy'
)

# Sklearn K-Fold
sklearn_scores = cross_val_score(
    LogisticRegression(random_state=42), X, y, 
    cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring='accuracy'
)

print("Cross-Validation Results:")
print(f"Custom implementation: {custom_scores.mean():.4f} (+/- {custom_scores.std() * 2:.4f})")
print(f"Sklearn implementation: {sklearn_scores.mean():.4f} (+/- {sklearn_scores.std() * 2:.4f})")

# Visualize CV scores
plt.figure(figsize=(10, 6))
x_pos = np.arange(5)
width = 0.35

plt.bar(x_pos - width/2, custom_scores, width, label='Custom', alpha=0.7)
plt.bar(x_pos + width/2, sklearn_scores, width, label='Sklearn', alpha=0.7)

plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Cross-Validation Scores Comparison')
plt.xticks(x_pos, [f'Fold {i+1}' for i in range(5)])
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 2.2 Stratified Cross-Validation 🟡

```python
def compare_cv_strategies(X, y, model, n_splits=5):
    """Compare different cross-validation strategies"""
    
    # Regular K-Fold
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    kfold_scores = cross_val_score(model, X, y, cv=kfold)
    
    # Stratified K-Fold
    stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    stratified_scores = cross_val_score(model, X, y, cv=stratified_kfold)
    
    # Analyze class distribution in each fold
    fold_distributions = {'kfold': [], 'stratified': []}
    
    for train_idx, val_idx in kfold.split(X, y):
        fold_y = y[val_idx]
        class_dist = np.bincount(fold_y) / len(fold_y)
        fold_distributions['kfold'].append(class_dist)
    
    for train_idx, val_idx in stratified_kfold.split(X, y):
        fold_y = y[val_idx]
        class_dist = np.bincount(fold_y) / len(fold_y)
        fold_distributions['stratified'].append(class_dist)
    
    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # CV Scores comparison
    axes[0, 0].bar(['K-Fold', 'Stratified K-Fold'], 
                   [kfold_scores.mean(), stratified_scores.mean()],
                   yerr=[kfold_scores.std(), stratified_scores.std()],
                   alpha=0.7, capsize=5)
    axes[0, 0].set_title('Mean CV Scores with Error Bars')
    axes[0, 0].set_ylabel('Accuracy')
    
    # Score distributions
    axes[0, 1].boxplot([kfold_scores, stratified_scores], 
                       labels=['K-Fold', 'Stratified K-Fold'])
    axes[0, 1].set_title('CV Score Distributions')
    axes[0, 1].set_ylabel('Accuracy')
    
    # Class distribution variance - K-Fold
    kfold_dists = np.array(fold_distributions['kfold'])
    axes[1, 0].plot(kfold_dists, marker='o', alpha=0.7)
    axes[1, 0].set_title('K-Fold: Class Distribution per Fold')
    axes[1, 0].set_xlabel('Fold')
    axes[1, 0].set_ylabel('Class Proportion')
    axes[1, 0].legend(['Class 0', 'Class 1'])
    axes[1, 0].grid(True, alpha=0.3)
    
    # Class distribution variance - Stratified K-Fold
    stratified_dists = np.array(fold_distributions['stratified'])
    axes[1, 1].plot(stratified_dists, marker='s', alpha=0.7)
    axes[1, 1].set_title('Stratified K-Fold: Class Distribution per Fold')
    axes[1, 1].set_xlabel('Fold')
    axes[1, 1].set_ylabel('Class Proportion')
    axes[1, 1].legend(['Class 0', 'Class 1'])
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Calculate class distribution variance
    kfold_class_var = np.var(kfold_dists, axis=0)
    stratified_class_var = np.var(stratified_dists, axis=0)
    
    print("Cross-Validation Strategy Comparison:")
    print(f"K-Fold CV: {kfold_scores.mean():.4f} (+/- {kfold_scores.std() * 2:.4f})")
    print(f"Stratified K-Fold CV: {stratified_scores.mean():.4f} (+/- {stratified_scores.std() * 2:.4f})")
    print(f"\nClass Distribution Variance:")
    print(f"K-Fold - Class 0: {kfold_class_var[0]:.6f}, Class 1: {kfold_class_var[1]:.6f}")
    print(f"Stratified K-Fold - Class 0: {stratified_class_var[0]:.6f}, Class 1: {stratified_class_var[1]:.6f}")
    
    return kfold_scores, stratified_scores

# Compare CV strategies
model = RandomForestClassifier(n_estimators=100, random_state=42)
kfold_scores, stratified_scores = compare_cv_strategies(X, y, model)
```

### 2.3 Specialized Cross-Validation 🟡

#### Time Series Cross-Validation:
```python
from sklearn.model_selection import TimeSeriesSplit

class CustomTimeSeriesSplit:
    def __init__(self, n_splits=5, max_train_size=None, test_size=None, gap=0):
        self.n_splits = n_splits
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.gap = gap
    
    def split(self, X, y=None, groups=None):
        """Generate train/test indices for time series data"""
        n_samples = len(X)
        
        if self.test_size is None:
            test_size = n_samples // (self.n_splits + 1)
        else:
            test_size = self.test_size
        
        for i in range(self.n_splits):
            # Calculate test start and end
            test_start = (i + 1) * test_size
            test_end = test_start + test_size
            
            if test_end > n_samples:
                test_end = n_samples
                test_start = test_end - test_size
            
            # Calculate train end (with gap)
            train_end = test_start - self.gap
            
            # Calculate train start
            if self.max_train_size is None:
                train_start = 0
            else:
                train_start = max(0, train_end - self.max_train_size)
            
            # Generate indices
            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices

# Create time series data
np.random.seed(42)
n_samples = 1000
time_index = pd.date_range('2020-01-01', periods=n_samples, freq='D')
trend = np.linspace(0, 10, n_samples)
seasonal = 3 * np.sin(2 * np.pi * np.arange(n_samples) / 365.25)
noise = np.random.normal(0, 1, n_samples)
ts_data = trend + seasonal + noise

# Create features (lagged values)
def create_ts_features(ts, lags=[1, 2, 3, 5, 7]):
    """Create lagged features for time series"""
    features = []
    max_lag = max(lags)
    
    for i in range(max_lag, len(ts)):
        feature_row = []
        for lag in lags:
            feature_row.append(ts[i - lag])
        features.append(feature_row)
    
    return np.array(features), ts[max_lag:]

X_ts, y_ts = create_ts_features(ts_data)

# Compare regular CV vs Time Series CV
def compare_ts_cv(X, y, model):
    """Compare regular CV with time series CV"""
    
    # Regular K-Fold (WRONG for time series!)
    regular_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    regular_scores = cross_val_score(model, X, y, cv=regular_cv)
    
    # Time Series Split (CORRECT for time series)
    ts_cv = TimeSeriesSplit(n_splits=5)
    ts_scores = cross_val_score(model, X, y, cv=ts_cv)
    
    # Custom Time Series Split
    custom_ts_cv = CustomTimeSeriesSplit(n_splits=5, gap=1)
    custom_ts_scores = []
    
    for train_idx, test_idx in custom_ts_cv.split(X):
        model_clone = clone(model)
        model_clone.fit(X[train_idx], y[train_idx])
        score = model_clone.score(X[test_idx], y[test_idx])
        custom_ts_scores.append(score)
    
    custom_ts_scores = np.array(custom_ts_scores)
    
    # Visualize splits
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    # Regular K-Fold visualization
    y_pos = 0
    for i, (train_idx, test_idx) in enumerate(regular_cv.split(X)):
        axes[0].barh(y_pos, len(train_idx), left=min(train_idx), 
                    color=colors[i], alpha=0.6, label=f'Fold {i+1} Train')
        axes[0].barh(y_pos + 0.3, len(test_idx), left=min(test_idx), 
                    color=colors[i], alpha=1.0, label=f'Fold {i+1} Test')
        y_pos += 1
    
    axes[0].set_title('Regular K-Fold CV (WRONG for Time Series)')
    axes[0].set_xlabel('Sample Index')
    axes[0].set_ylabel('Fold')
    
    # Time Series Split visualization
    y_pos = 0
    for i, (train_idx, test_idx) in enumerate(ts_cv.split(X)):
        axes[1].barh(y_pos, len(train_idx), left=min(train_idx), 
                    color=colors[i], alpha=0.6)
        axes[1].barh(y_pos + 0.3, len(test_idx), left=min(test_idx), 
                    color=colors[i], alpha=1.0)
        y_pos += 1
    
    axes[1].set_title('Time Series Split (CORRECT)')
    axes[1].set_xlabel('Sample Index')
    axes[1].set_ylabel('Fold')
    
    # Score comparison
    axes[2].bar(['Regular K-Fold', 'Time Series Split', 'Custom TS Split'], 
               [regular_scores.mean(), ts_scores.mean(), custom_ts_scores.mean()],
               yerr=[regular_scores.std(), ts_scores.std(), custom_ts_scores.std()],
               alpha=0.7, capsize=5)
    axes[2].set_title('Cross-Validation Score Comparison')
    axes[2].set_ylabel('R² Score')
    
    plt.tight_layout()
    plt.show()
    
    print("Time Series Cross-Validation Comparison:")
    print(f"Regular K-Fold: {regular_scores.mean():.4f} (+/- {regular_scores.std() * 2:.4f})")
    print(f"Time Series Split: {ts_scores.mean():.4f} (+/- {ts_scores.std() * 2:.4f})")
    print(f"Custom TS Split: {custom_ts_scores.mean():.4f} (+/- {custom_ts_scores.std() * 2:.4f})")
    
    return regular_scores, ts_scores, custom_ts_scores

# Use regression model for time series
from sklearn.ensemble import RandomForestRegressor
ts_model = RandomForestRegressor(n_estimators=50, random_state=42)
regular_ts_scores, ts_split_scores, custom_ts_scores = compare_ts_cv(X_ts, y_ts, ts_model)
```

---

## 3. Classification Metrics

### 3.1 Comprehensive Classification Metrics 🟢

```python
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc, precision_recall_curve

def comprehensive_classification_evaluation(y_true, y_pred, y_proba=None, class_names=None):
    """Comprehensive classification evaluation"""
    
    if class_names is None:
        class_names = [f'Class {i}' for i in np.unique(y_true)]
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)
    
    # Macro and micro averages
    precision_macro = precision_score(y_true, y_pred, average='macro')
    recall_macro = recall_score(y_true, y_pred, average='macro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    
    precision_micro = precision_score(y_true, y_pred, average='micro')
    recall_micro = recall_score(y_true, y_pred, average='micro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 15))
    
    # Confusion Matrix
    ax1 = plt.subplot(3, 4, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Normalized Confusion Matrix
    ax2 = plt.subplot(3, 4, 2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Precision, Recall, F1 by class
    ax3 = plt.subplot(3, 4, 3)
    x_pos = np.arange(len(class_names))
    width = 0.25
    
    plt.bar(x_pos - width, precision, width, label='Precision', alpha=0.7)
    plt.bar(x_pos, recall, width, label='Recall', alpha=0.7)
    plt.bar(x_pos + width, f1, width, label='F1-Score', alpha=0.7)
    
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.title('Per-Class Metrics')
    plt.xticks(x_pos, class_names)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Macro vs Micro averages
    ax4 = plt.subplot(3, 4, 4)
    metrics = ['Precision', 'Recall', 'F1-Score']
    macro_scores = [precision_macro, recall_macro, f1_macro]
    micro_scores = [precision_micro, recall_micro, f1_micro]
    
    x_pos = np.arange(len(metrics))
    plt.bar(x_pos - 0.2, macro_scores, 0.4, label='Macro Average', alpha=0.7)
    plt.bar(x_pos + 0.2, micro_scores, 0.4, label='Micro Average', alpha=0.7)
    
    plt.xlabel('Metric')
    plt.ylabel('Score')
    plt.title('Macro vs Micro Averages')
    plt.xticks(x_pos, metrics)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if y_proba is not None and len(np.unique(y_true)) == 2:
        # ROC Curve
        ax5 = plt.subplot(3, 4, 5)
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # Precision-Recall Curve
        ax6 = plt.subplot(3, 4, 6)
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_proba[:, 1])
        pr_auc = auc(recall_curve, precision_curve)
        
        plt.plot(recall_curve, precision_curve, color='blue', lw=2,
                label=f'PR curve (AUC = {pr_auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
    
    # Class distribution
    ax7 = plt.subplot(3, 4, 7)
    unique_true, counts_true = np.unique(y_true, return_counts=True)
    unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
    
    x_pos = np.arange(len(class_names))
    plt.bar(x_pos - 0.2, counts_true, 0.4, label='True', alpha=0.7)
    plt.bar(x_pos + 0.2, counts_pred, 0.4, label='Predicted', alpha=0.7)
    
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Distribution Comparison')
    plt.xticks(x_pos, class_names)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Error analysis
    ax8 = plt.subplot(3, 4, 8)
    errors = y_true != y_pred
    error_rate = np.mean(errors)
    
    plt.bar(['Correct', 'Incorrect'], 
           [1 - error_rate, error_rate], 
           color=['green', 'red'], alpha=0.7)
    plt.title(f'Prediction Accuracy\n(Error Rate: {error_rate:.1%})')
    plt.ylabel('Proportion')
    
    for i, v in enumerate([1 - error_rate, error_rate]):
        plt.text(i, v + 0.01, f'{v:.1%}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed metrics
    print("CLASSIFICATION EVALUATION REPORT")
    print("=" * 50)
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Overall Error Rate: {1 - accuracy:.4f}")
    print(f"\nMacro Averages:")
    print(f"  Precision: {precision_macro:.4f}")
    print(f"  Recall: {recall_macro:.4f}")
    print(f"  F1-Score: {f1_macro:.4f}")
    print(f"\nMicro Averages:")
    print(f"  Precision: {precision_micro:.4f}")
    print(f"  Recall: {recall_micro:.4f}")
    print(f"  F1-Score: {f1_micro:.4f}")
    
    print(f"\nPer-Class Metrics:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name}:")
        print(f"    Precision: {precision[i]:.4f}")
        print(f"    Recall: {recall[i]:.4f}")
        print(f"    F1-Score: {f1[i]:.4f}")
    
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro
    }

# Train model and evaluate
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# Comprehensive evaluation
results = comprehensive_classification_evaluation(
    y_test, y_pred, y_proba, class_names=['Class 0', 'Class 1']
)
```

### 3.2 Custom Metrics 🟡

```python
def custom_business_metric(y_true, y_pred, cost_matrix=None):
    """
    Custom business metric considering misclassification costs
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        cost_matrix: Cost of misclassification (2x2 matrix)
                    [[TN_cost, FP_cost],
                     [FN_cost, TP_cost]]
    """
    
    if cost_matrix is None:
        # Default: False positives cost 2x, False negatives cost 5x
        cost_matrix = np.array([[0, 2],    # TN, FP
                               [5, 0]])    # FN, TP
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate total cost
    total_cost = np.sum(cm * cost_matrix)
    
    # Calculate cost per sample
    cost_per_sample = total_cost / len(y_true)
    
    # Calculate cost-based accuracy (lower cost = higher accuracy)
    max_possible_cost = len(y_true) * np.max(cost_matrix)
    cost_based_accuracy = 1 - (total_cost / max_possible_cost)
    
    return {
        'total_cost': total_cost,
        'cost_per_sample': cost_per_sample,
        'cost_based_accuracy': cost_based_accuracy,
        'confusion_matrix': cm,
        'cost_matrix': cost_matrix
    }

def balanced_accuracy_detailed(y_true, y_pred):
    """Detailed balanced accuracy calculation"""
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate sensitivity (recall) for each class
    sensitivities = []
    for i in range(len(cm)):
        sensitivity = cm[i, i] / np.sum(cm[i, :]) if np.sum(cm[i, :]) > 0 else 0
        sensitivities.append(sensitivity)
    
    balanced_acc = np.mean(sensitivities)
    
    return {
        'balanced_accuracy': balanced_acc,
        'class_sensitivities': sensitivities,
        'unbalanced_accuracy': accuracy_score(y_true, y_pred)
    }

# Apply custom metrics
business_results = custom_business_metric(y_test, y_pred)
balanced_results = balanced_accuracy_detailed(y_test, y_pred)

print("Custom Business Metric Results:")
print(f"Total Cost: {business_results['total_cost']}")
print(f"Cost per Sample: {business_results['cost_per_sample']:.2f}")
print(f"Cost-based Accuracy: {business_results['cost_based_accuracy']:.4f}")

print(f"\nBalanced Accuracy Results:")
print(f"Balanced Accuracy: {balanced_results['balanced_accuracy']:.4f}")
print(f"Regular Accuracy: {balanced_results['unbalanced_accuracy']:.4f}")
print(f"Class Sensitivities: {balanced_results['class_sensitivities']}")
```

---

## 4. Regression Metrics

### 4.1 Comprehensive Regression Evaluation 🟢

```python
# Create regression dataset
X_reg, y_reg = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

def comprehensive_regression_evaluation(y_true, y_pred, feature_names=None):
    """Comprehensive regression evaluation"""
    
    # Basic metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Additional metrics
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    max_error = np.max(np.abs(y_true - y_pred))
    
    # Residual analysis
    residuals = y_true - y_pred
    residual_std = np.std(residuals)
    residual_mean = np.mean(residuals)
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    # Actual vs Predicted
    axes[0, 0].scatter(y_true, y_pred, alpha=0.6)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Values')
    axes[0, 0].set_ylabel('Predicted Values')
    axes[0, 0].set_title(f'Actual vs Predicted (R² = {r2:.3f})')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Residuals vs Predicted
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
    axes[0, 1].axhline(y=0, color='red', linestyle='--')
    axes[0, 1].set_xlabel('Predicted Values')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residuals vs Predicted')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Residual histogram
    axes[0, 2].hist(residuals, bins=30, density=True, alpha=0.7, edgecolor='black')
    axes[0, 2].axvline(residual_mean, color='red', linestyle='--', 
                      label=f'Mean: {residual_mean:.3f}')
    axes[0, 2].set_xlabel('Residuals')
    axes[0, 2].set_ylabel('Density')
    axes[0, 2].set_title('Residual Distribution')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Q-Q plot for residuals
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot of Residuals')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Residuals vs Index (to check for patterns)
    axes[1, 1].plot(residuals, alpha=0.7)
    axes[1, 1].axhline(y=0, color='red', linestyle='--')
    axes[1, 1].set_xlabel('Sample Index')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].set_title('Residuals vs Sample Order')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Error metrics comparison
    metrics_names = ['MAE', 'RMSE', 'MAPE (%)', 'Max Error']
    metrics_values = [mae, rmse, mape, max_error]
    
    axes[1, 2].bar(metrics_names, metrics_values, alpha=0.7, color=['blue', 'green', 'orange', 'red'])
    axes[1, 2].set_title('Error Metrics')
    axes[1, 2].set_ylabel('Error Value')
    
    # Add value labels on bars
    for i, v in enumerate(metrics_values):
        axes[1, 2].text(i, v + max(metrics_values) * 0.01, f'{v:.2f}', 
                        ha='center', va='bottom')
    
    axes[1, 2].grid(True, alpha=0.3)
    
    # Prediction error vs actual values
    abs_errors = np.abs(residuals)
    axes[2, 0].scatter(y_true, abs_errors, alpha=0.6)
    axes[2, 0].set_xlabel('Actual Values')
    axes[2, 0].set_ylabel('Absolute Error')
    axes[2, 0].set_title('Absolute Error vs Actual Values')
    axes[2, 0].grid(True, alpha=0.3)
    
    # Cumulative error distribution
    sorted_abs_errors = np.sort(abs_errors)
    cumulative_percentiles = np.arange(1, len(sorted_abs_errors) + 1) / len(sorted_abs_errors)
    
    axes[2, 1].plot(sorted_abs_errors, cumulative_percentiles)
    axes[2, 1].axvline(np.percentile(abs_errors, 90), color='red', linestyle='--', 
                      label='90th percentile')
    axes[2, 1].set_xlabel('Absolute Error')
    axes[2, 1].set_ylabel('Cumulative Probability')
    axes[2, 1].set_title('Cumulative Error Distribution')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    # Performance summary
    axes[2, 2].axis('off')
    performance_text = f"""
    REGRESSION PERFORMANCE SUMMARY
    
    R² Score: {r2:.4f}
    MAE: {mae:.4f}
    MSE: {mse:.4f}
    RMSE: {rmse:.4f}
    MAPE: {mape:.2f}%
    Max Error: {max_error:.4f}
    
    Residual Statistics:
    Mean: {residual_mean:.4f}
    Std: {residual_std:.4f}
    
    90th Percentile Error: {np.percentile(abs_errors, 90):.4f}
    95th Percentile Error: {np.percentile(abs_errors, 95):.4f}
    """
    
    axes[2, 2].text(0.1, 0.9, performance_text, transform=axes[2, 2].transAxes, 
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.show()
    
    # Detailed metrics dictionary
    metrics_dict = {
        'r2_score': r2,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
        'max_error': max_error,
        'residual_mean': residual_mean,
        'residual_std': residual_std,
        'percentile_90_error': np.percentile(abs_errors, 90),
        'percentile_95_error': np.percentile(abs_errors, 95)
    }
    
    return metrics_dict

# Train regression model and evaluate
from sklearn.ensemble import RandomForestRegressor

reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
reg_model.fit(X_train_reg, y_train_reg)

y_pred_reg = reg_model.predict(X_test_reg)

# Comprehensive evaluation
reg_results = comprehensive_regression_evaluation(y_test_reg, y_pred_reg)
```

---

## 5. Model Comparison and Statistical Testing

### 5.1 Statistical Significance Testing 🔴

```python
from scipy import stats
from sklearn.model_selection import cross_validate

def compare_models_statistical(models, X, y, cv=5, scoring='accuracy', alpha=0.05):
    """
    Compare multiple models using statistical tests
    
    Args:
        models: Dictionary of models to compare
        X, y: Data
        cv: Cross-validation strategy
        scoring: Scoring metric
        alpha: Significance level
    """
    
    # Collect CV scores for each model
    cv_results = {}
    
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        cv_results[name] = scores
        print(f"{name}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    # Pairwise statistical tests
    model_names = list(models.keys())
    n_models = len(model_names)
    
    # Create results matrix
    p_values = np.zeros((n_models, n_models))
    test_statistics = np.zeros((n_models, n_models))
    
    print(f"\nPairwise t-test results (α = {alpha}):")
    print("=" * 60)
    
    for i in range(n_models):
        for j in range(i + 1, n_models):
            model1, model2 = model_names[i], model_names[j]
            scores1, scores2 = cv_results[model1], cv_results[model2]
            
            # Paired t-test
            t_stat, p_value = stats.ttest_rel(scores1, scores2)
            
            p_values[i, j] = p_value
            p_values[j, i] = p_value
            test_statistics[i, j] = t_stat
            test_statistics[j, i] = -t_stat
            
            # Interpretation
            significant = "***" if p_value < alpha else "n.s."
            direction = ">" if np.mean(scores1) > np.mean(scores2) else "<"
            
            print(f"{model1} {direction} {model2}: t={t_stat:.3f}, p={p_value:.4f} {significant}")
    
    # Friedman test for multiple comparisons
    all_scores = [cv_results[name] for name in model_names]
    friedman_stat, friedman_p = stats.friedmanchisquare(*all_scores)
    
    print(f"\nFriedman test (overall difference):")
    print(f"χ² = {friedman_stat:.3f}, p = {friedman_p:.4f}")
    
    if friedman_p < alpha:
        print("*** Significant difference between models detected")
        
        # Post-hoc analysis with Nemenyi test approximation
        print(f"\nPost-hoc pairwise comparisons:")
        
        # Calculate critical difference
        k = len(model_names)
        N = cv
        q_alpha = 2.344  # Critical value for α=0.05, k=number of models
        
        cd = q_alpha * np.sqrt(k * (k + 1) / (6 * N))
        
        # Calculate average ranks
        ranks_matrix = np.zeros((N, k))
        for fold in range(N):
            fold_scores = [cv_results[name][fold] for name in model_names]
            ranks_matrix[fold] = stats.rankdata([-score for score in fold_scores])  # Negative for descending
        
        avg_ranks = np.mean(ranks_matrix, axis=0)
        
        print(f"Critical Difference (CD): {cd:.3f}")
        print(f"Average ranks:")
        for i, name in enumerate(model_names):
            print(f"  {name}: {avg_ranks[i]:.2f}")
        
        # Find significantly different pairs
        print(f"\nSignificantly different pairs (rank difference > {cd:.3f}):")
        for i in range(k):
            for j in range(i + 1, k):
                rank_diff = abs(avg_ranks[i] - avg_ranks[j])
                if rank_diff > cd:
                    better_model = model_names[i] if avg_ranks[i] < avg_ranks[j] else model_names[j]
                    worse_model = model_names[j] if avg_ranks[i] < avg_ranks[j] else model_names[i]
                    print(f"  {better_model} > {worse_model} (diff: {rank_diff:.3f})")
    else:
        print("No significant difference between models detected")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Box plot of CV scores
    scores_data = [cv_results[name] for name in model_names]
    axes[0, 0].boxplot(scores_data, labels=model_names)
    axes[0, 0].set_title('Cross-Validation Score Distributions')
    axes[0, 0].set_ylabel(scoring.capitalize())
    axes[0, 0].grid(True, alpha=0.3)
    
    # Bar plot with error bars
    means = [np.mean(cv_results[name]) for name in model_names]
    stds = [np.std(cv_results[name]) for name in model_names]
    
    axes[0, 1].bar(model_names, means, yerr=stds, capsize=5, alpha=0.7)
    axes[0, 1].set_title('Mean CV Scores with Standard Deviation')
    axes[0, 1].set_ylabel(scoring.capitalize())
    axes[0, 1].grid(True, alpha=0.3)
    
    # P-value heatmap
    mask = np.triu(np.ones_like(p_values), k=1).astype(bool)
    p_values_display = p_values.copy()
    p_values_display[mask] = np.nan
    
    im = axes[1, 0].imshow(p_values_display, cmap='RdYlBu_r', vmin=0, vmax=alpha*2)
    axes[1, 0].set_xticks(range(n_models))
    axes[1, 0].set_yticks(range(n_models))
    axes[1, 0].set_xticklabels(model_names, rotation=45)
    axes[1, 0].set_yticklabels(model_names)
    axes[1, 0].set_title('P-values from Pairwise t-tests')
    
    # Add text annotations
    for i in range(n_models):
        for j in range(n_models):
            if not mask[i, j] and i != j:
                text = f'{p_values[i, j]:.3f}'
                axes[1, 0].text(j, i, text, ha='center', va='center',
                               color='white' if p_values[i, j] < alpha/2 else 'black')
    
    plt.colorbar(im, ax=axes[1, 0])
    
    # Ranking visualization
    if 'avg_ranks' in locals():
        sorted_indices = np.argsort(avg_ranks)
        sorted_names = [model_names[i] for i in sorted_indices]
        sorted_ranks = [avg_ranks[i] for i in sorted_indices]
        
        axes[1, 1].barh(range(len(sorted_names)), sorted_ranks, alpha=0.7)
        axes[1, 1].set_yticks(range(len(sorted_names)))
        axes[1, 1].set_yticklabels(sorted_names)
        axes[1, 1].set_xlabel('Average Rank')
        axes[1, 1].set_title('Model Ranking (Lower is Better)')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].axis('off')
        axes[1, 1].text(0.5, 0.5, 'No significant\ndifferences found\n(Friedman test n.s.)', 
                        ha='center', va='center', transform=axes[1, 1].transAxes,
                        fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray'))
    
    plt.tight_layout()
    plt.show()
    
    return cv_results, p_values, test_statistics

# Compare multiple models
models_to_compare = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(random_state=42),
    'Naive Bayes': GaussianNB()
}

comparison_results = compare_models_statistical(
    models_to_compare, X, y, cv=5, scoring='accuracy'
)
```

---

## 🎯 Key Takeaways

### Model Evaluation Best Practices:

#### Data Splitting:
- **Always** use separate test set for final evaluation
- Use stratified sampling for imbalanced datasets
- Consider temporal structure for time series data
- Maintain data distribution across splits

#### Cross-Validation:
- Use appropriate CV strategy for your data type
- Stratified K-Fold for classification
- Time Series Split for temporal data
- Group K-Fold for grouped data
- Consider computational cost vs. reliability trade-off

#### Metric Selection:
- **Classification**: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **Regression**: R², MAE, RMSE, MAPE
- **Business metrics**: Custom cost-sensitive measures
- **Imbalanced data**: Balanced accuracy, PR-AUC

#### Statistical Testing:
- Use paired t-tests for comparing two models
- Friedman test for multiple model comparison
- Consider practical significance, not just statistical significance
- Account for multiple comparison problems

### Common Pitfalls:
1. **Data leakage**: Information from future in past
2. **Overfitting to validation set**: Too much hyperparameter tuning
3. **Wrong CV strategy**: Using K-Fold for time series
4. **Ignoring class imbalance**: Using accuracy for skewed datasets
5. **Not testing significance**: Assuming differences are meaningful

---

## 📚 Next Steps

Continue your journey with:
- **[ML Engineering & MLOps](13_ML_Engineering_MLOps.md)** - Production deployment and lifecycle management
- **[Ethics & Bias in AI](14_Ethics_Bias_AI.md)** - Responsible AI development

---

*Next: [ML Engineering & MLOps →](13_ML_Engineering_MLOps.md)*
