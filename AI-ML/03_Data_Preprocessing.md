# 03. Data Preprocessing

## 🎯 Learning Objectives
- Master data cleaning and preparation techniques
- Understand feature engineering and selection
- Learn data transformation and normalization methods
- Handle missing data and outliers effectively

---

## 1. Introduction to Data Preprocessing

**Data preprocessing** is the process of preparing raw data for machine learning algorithms. It's often said that 80% of a data scientist's time is spent on data preparation, making this a crucial skill.

### 1.1 Why Data Preprocessing Matters 🟢

#### Garbage In, Garbage Out (GIGO):
- Poor quality data leads to poor model performance
- Even the best algorithms can't overcome bad data
- Preprocessing can dramatically improve results

#### Common Data Quality Issues:
- **Missing values**: Incomplete records
- **Inconsistent formats**: Different date formats, capitalization
- **Outliers**: Extreme values that may be errors
- **Noise**: Random errors in data
- **Duplicate records**: Same information recorded multiple times
- **Irrelevant features**: Variables that don't contribute to predictions

### 1.2 Data Preprocessing Pipeline 🟢

```
Raw Data → Data Cleaning → Feature Engineering → Feature Selection → 
Data Transformation → Data Splitting → Ready for ML
```

#### Typical Steps:
1. **Data Collection**: Gathering data from various sources
2. **Data Exploration**: Understanding data structure and quality
3. **Data Cleaning**: Handling missing values, outliers, duplicates
4. **Feature Engineering**: Creating new features from existing ones
5. **Feature Selection**: Choosing most relevant features
6. **Data Transformation**: Scaling, encoding, normalization
7. **Data Splitting**: Train/validation/test sets

---

## 2. Data Understanding and Exploration

### 2.1 Exploratory Data Analysis (EDA) 🟢

#### Initial Data Inspection:
```python
# Basic information about dataset
df.info()          # Data types, memory usage, non-null counts
df.describe()      # Statistical summary for numerical columns
df.head()          # First few rows
df.tail()          # Last few rows
df.shape           # Number of rows and columns
df.columns         # Column names
df.dtypes          # Data types of each column
```

#### Data Quality Assessment:
```python
# Missing values
df.isnull().sum()
df.isnull().sum() / len(df) * 100  # Percentage missing

# Duplicate records
df.duplicated().sum()

# Unique values
df.nunique()
df['column'].value_counts()
```

### 2.2 Statistical Analysis 🟢

#### Univariate Analysis:
- **Numerical variables**: Histograms, box plots, summary statistics
- **Categorical variables**: Bar charts, frequency tables

#### Bivariate Analysis:
- **Numerical vs. Numerical**: Scatter plots, correlation
- **Categorical vs. Numerical**: Box plots by category
- **Categorical vs. Categorical**: Cross-tabulation, chi-square test

#### Multivariate Analysis:
- **Correlation matrices**: Heatmaps showing relationships
- **Pair plots**: All variables against each other
- **Principal Component Analysis**: Dimensionality reduction for visualization

### 2.3 Data Visualization 🟢

#### Essential Plots:
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Distribution plots
plt.hist(df['column'])           # Histogram
sns.boxplot(df['column'])        # Box plot
sns.violinplot(df['column'])     # Violin plot

# Relationship plots
plt.scatter(df['x'], df['y'])    # Scatter plot
sns.heatmap(df.corr())          # Correlation heatmap
sns.pairplot(df)                # Pair plot

# Categorical data
sns.countplot(df['category'])    # Count plot
sns.barplot(x='cat', y='num', data=df)  # Bar plot
```

#### Key Insights to Look For:
- **Distribution shape**: Normal, skewed, bimodal
- **Outliers**: Points far from typical values
- **Missing data patterns**: Random vs. systematic
- **Relationships**: Linear, non-linear, no relationship
- **Class imbalance**: Unequal representation in target variable

---

## 3. Data Cleaning

### 3.1 Handling Missing Data 🟢

#### Types of Missing Data:
1. **Missing Completely at Random (MCAR)**: Missingness independent of all variables
2. **Missing at Random (MAR)**: Missingness depends on observed variables
3. **Missing Not at Random (MNAR)**: Missingness depends on unobserved variables

#### Detection Strategies:
```python
# Missing data patterns
import missingno as msno
msno.matrix(df)      # Visualize missing data pattern
msno.heatmap(df)     # Correlation of missingness between variables
msno.dendrogram(df)  # Hierarchical clustering of missing data
```

#### Handling Strategies:

**1. Deletion Methods:**
```python
# Remove rows with any missing values
df_clean = df.dropna()

# Remove rows with missing values in specific columns
df_clean = df.dropna(subset=['important_column'])

# Remove columns with high percentage of missing values
threshold = 0.5  # 50% threshold
df_clean = df.dropna(axis=1, thresh=len(df) * threshold)
```

**2. Imputation Methods:**

**Simple Imputation:**
```python
from sklearn.impute import SimpleImputer

# Mean/median/mode imputation
imputer = SimpleImputer(strategy='mean')  # or 'median', 'most_frequent'
df['column'] = imputer.fit_transform(df[['column']])

# Forward fill / backward fill
df['column'].fillna(method='ffill')  # Forward fill
df['column'].fillna(method='bfill')  # Backward fill
```

**Advanced Imputation:**
```python
from sklearn.impute import KNNImputer, IterativeImputer

# K-Nearest Neighbors imputation
knn_imputer = KNNImputer(n_neighbors=5)
df_imputed = knn_imputer.fit_transform(df)

# Iterative imputation (MICE)
iterative_imputer = IterativeImputer()
df_imputed = iterative_imputer.fit_transform(df)
```

**Domain-Specific Imputation:**
```python
# Time series: interpolation
df['column'].interpolate(method='linear')

# Categorical: create "Unknown" category
df['category'].fillna('Unknown')

# Numerical: use regression to predict missing values
from sklearn.linear_model import LinearRegression
# Train model on non-missing data, predict missing values
```

### 3.2 Handling Outliers 🟡

#### Detection Methods:

**Statistical Methods:**
```python
# Z-score method
from scipy import stats
z_scores = np.abs(stats.zscore(df['column']))
threshold = 3
outliers = df[z_scores > threshold]

# IQR method
Q1 = df['column'].quantile(0.25)
Q3 = df['column'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df['column'] < lower_bound) | (df['column'] > upper_bound)]
```

**Visual Methods:**
```python
# Box plots
plt.boxplot(df['column'])

# Scatter plots
plt.scatter(df['x'], df['y'])

# Histograms
plt.hist(df['column'], bins=50)
```

**Advanced Methods:**
```python
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

# Isolation Forest
iso_forest = IsolationForest(contamination=0.1)
outlier_labels = iso_forest.fit_predict(df)

# One-Class SVM
oc_svm = OneClassSVM(nu=0.1)
outlier_labels = oc_svm.fit_predict(df)
```

#### Handling Strategies:
1. **Remove outliers**: Delete extreme values
2. **Transform data**: Log transformation, Box-Cox
3. **Cap outliers**: Replace with threshold values
4. **Separate analysis**: Analyze outliers separately
5. **Robust methods**: Use algorithms less sensitive to outliers

### 3.3 Handling Duplicates 🟢

#### Detection and Removal:
```python
# Find duplicate rows
duplicates = df.duplicated()
print(f"Number of duplicates: {duplicates.sum()}")

# Remove duplicates
df_clean = df.drop_duplicates()

# Keep first occurrence
df_clean = df.drop_duplicates(keep='first')

# Keep last occurrence
df_clean = df.drop_duplicates(keep='last')

# Remove duplicates based on specific columns
df_clean = df.drop_duplicates(subset=['column1', 'column2'])
```

#### Fuzzy Matching for Similar Records:
```python
from fuzzywuzzy import fuzz

# Find similar strings
similarity = fuzz.ratio("string1", "string2")
if similarity > 80:  # Threshold for similarity
    # Handle as potential duplicate
```

---

## 4. Feature Engineering

Feature engineering is the process of creating new features from existing data to improve model performance.

### 4.1 Creating New Features 🟡

#### Mathematical Transformations:
```python
# Polynomial features
df['x_squared'] = df['x'] ** 2
df['x_cubed'] = df['x'] ** 3

# Logarithmic transformation
df['log_x'] = np.log(df['x'])

# Square root transformation
df['sqrt_x'] = np.sqrt(df['x'])

# Exponential transformation
df['exp_x'] = np.exp(df['x'])
```

#### Interaction Features:
```python
# Multiplication
df['feature1_x_feature2'] = df['feature1'] * df['feature2']

# Division (with zero handling)
df['feature1_div_feature2'] = df['feature1'] / (df['feature2'] + 1e-8)

# Addition/subtraction
df['feature_sum'] = df['feature1'] + df['feature2']
df['feature_diff'] = df['feature1'] - df['feature2']
```

#### Binning/Discretization:
```python
# Equal-width binning
df['age_group'] = pd.cut(df['age'], bins=5, labels=['Young', 'Adult', 'Middle', 'Senior', 'Elder'])

# Equal-frequency binning
df['income_quartile'] = pd.qcut(df['income'], q=4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])

# Custom binning
bins = [0, 18, 65, 100]
labels = ['Minor', 'Adult', 'Senior']
df['age_category'] = pd.cut(df['age'], bins=bins, labels=labels)
```

### 4.2 Time-Based Features 🟡

#### Date/Time Decomposition:
```python
# Convert to datetime
df['date'] = pd.to_datetime(df['date'])

# Extract components
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['day_of_week'] = df['date'].dt.dayofweek
df['day_of_year'] = df['date'].dt.dayofyear
df['week_of_year'] = df['date'].dt.week
df['quarter'] = df['date'].dt.quarter

# Boolean features
df['is_weekend'] = df['day_of_week'].isin([5, 6])
df['is_month_start'] = df['date'].dt.is_month_start
df['is_month_end'] = df['date'].dt.is_month_end
```

#### Time Since Features:
```python
# Time since reference date
reference_date = pd.to_datetime('2020-01-01')
df['days_since_reference'] = (df['date'] - reference_date).dt.days

# Time between events
df = df.sort_values('date')
df['days_since_last_purchase'] = df.groupby('customer_id')['date'].diff().dt.days
```

#### Lag and Lead Features:
```python
# Lag features
df['value_lag_1'] = df.groupby('id')['value'].shift(1)
df['value_lag_7'] = df.groupby('id')['value'].shift(7)

# Lead features
df['value_lead_1'] = df.groupby('id')['value'].shift(-1)

# Rolling statistics
df['value_rolling_mean_7'] = df.groupby('id')['value'].rolling(7).mean()
df['value_rolling_std_7'] = df.groupby('id')['value'].rolling(7).std()
```

### 4.3 Text Feature Engineering 🟡

#### Basic Text Features:
```python
# Length features
df['text_length'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()
df['sentence_count'] = df['text'].str.count('\.')

# Character features
df['uppercase_count'] = df['text'].str.count('[A-Z]')
df['digit_count'] = df['text'].str.count('\d')
df['special_char_count'] = df['text'].str.count('[^a-zA-Z0-9\s]')
```

#### Bag of Words (BoW):
```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
bow_features = vectorizer.fit_transform(df['text'])
feature_names = vectorizer.get_feature_names_out()
```

#### TF-IDF (Term Frequency-Inverse Document Frequency):
```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_features=1000)
tfidf_features = tfidf_vectorizer.fit_transform(df['text'])
```

#### N-grams:
```python
# Bigrams and trigrams
vectorizer = CountVectorizer(ngram_range=(1, 3))  # Unigrams, bigrams, trigrams
ngram_features = vectorizer.fit_transform(df['text'])
```

### 4.4 Categorical Feature Engineering 🟡

#### Encoding Techniques:

**One-Hot Encoding:**
```python
# Using pandas
df_encoded = pd.get_dummies(df, columns=['category'])

# Using sklearn
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
encoded_features = encoder.fit_transform(df[['category']])
```

**Label Encoding:**
```python
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
df['category_encoded'] = label_encoder.fit_transform(df['category'])
```

**Ordinal Encoding:**
```python
from sklearn.preprocessing import OrdinalEncoder

# For ordered categories
ordinal_encoder = OrdinalEncoder(categories=[['Low', 'Medium', 'High']])
df['priority_encoded'] = ordinal_encoder.fit_transform(df[['priority']])
```

**Target Encoding:**
```python
# Mean target encoding
target_means = df.groupby('category')['target'].mean()
df['category_target_encoded'] = df['category'].map(target_means)
```

**Frequency Encoding:**
```python
# Frequency of each category
freq_encoding = df['category'].value_counts().to_dict()
df['category_frequency'] = df['category'].map(freq_encoding)
```

---

## 5. Feature Selection

Feature selection identifies the most relevant features for the prediction task.

### 5.1 Why Feature Selection? 🟢

#### Benefits:
- **Reduced overfitting**: Fewer features = simpler model
- **Improved performance**: Remove irrelevant/noisy features
- **Faster training**: Fewer computations required
- **Better interpretability**: Focus on important features
- **Reduced storage**: Less memory and disk space

#### Curse of Dimensionality:
As the number of features increases:
- Data becomes sparse in high-dimensional space
- Distance measures become less meaningful
- More data needed to maintain performance

### 5.2 Filter Methods 🟢

Based on statistical properties of features, independent of ML algorithm.

#### Univariate Statistical Tests:
```python
from sklearn.feature_selection import SelectKBest, chi2, f_regression

# For classification (categorical target)
selector = SelectKBest(score_func=chi2, k=10)
X_selected = selector.fit_transform(X, y)

# For regression (continuous target)
selector = SelectKBest(score_func=f_regression, k=10)
X_selected = selector.fit_transform(X, y)
```

#### Correlation-based Selection:
```python
# Remove highly correlated features
correlation_matrix = df.corr().abs()
upper_triangle = correlation_matrix.where(
    np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
)

# Find features with correlation > threshold
high_corr_features = [column for column in upper_triangle.columns 
                      if any(upper_triangle[column] > 0.95)]

# Remove one feature from each highly correlated pair
df_reduced = df.drop(columns=high_corr_features)
```

#### Variance Threshold:
```python
from sklearn.feature_selection import VarianceThreshold

# Remove features with low variance
selector = VarianceThreshold(threshold=0.01)
X_selected = selector.fit_transform(X)
```

### 5.3 Wrapper Methods 🟡

Use ML algorithm performance to evaluate feature subsets.

#### Recursive Feature Elimination (RFE):
```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Select top 10 features
estimator = LogisticRegression()
selector = RFE(estimator, n_features_to_select=10)
X_selected = selector.fit_transform(X, y)

# Get selected features
selected_features = X.columns[selector.support_]
```

#### Forward/Backward Selection:
```python
from mlxtend.feature_selection import SequentialFeatureSelector

# Forward selection
sfs = SequentialFeatureSelector(
    estimator=LogisticRegression(),
    k_features=10,
    forward=True,
    scoring='accuracy'
)
sfs.fit(X, y)
selected_features = list(sfs.k_feature_names_)
```

### 5.4 Embedded Methods 🟡

Feature selection integrated into model training.

#### L1 Regularization (Lasso):
```python
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel

# Lasso with cross-validation
lasso = LassoCV(cv=5)
lasso.fit(X, y)

# Select features with non-zero coefficients
selector = SelectFromModel(lasso, prefit=True)
X_selected = selector.transform(X)
```

#### Tree-based Feature Importance:
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

# Train random forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X, y)

# Select features based on importance
selector = SelectFromModel(rf, prefit=True)
X_selected = selector.transform(X)

# Get feature importances
importances = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
```

---

## 6. Data Transformation

### 6.1 Scaling and Normalization 🟢

#### Why Scale Data?
- Different features have different scales
- Some algorithms sensitive to scale (SVM, Neural Networks, k-NN)
- Gradient descent converges faster with scaled data

#### Standardization (Z-score normalization):
```python
from sklearn.preprocessing import StandardScaler

# Mean = 0, Std = 1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Formula: (x - mean) / std
```

#### Min-Max Scaling:
```python
from sklearn.preprocessing import MinMaxScaler

# Scale to range [0, 1]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Formula: (x - min) / (max - min)
```

#### Robust Scaling:
```python
from sklearn.preprocessing import RobustScaler

# Use median and IQR (less sensitive to outliers)
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Formula: (x - median) / IQR
```

#### Unit Vector Scaling:
```python
from sklearn.preprocessing import Normalizer

# Scale individual samples to have unit norm
normalizer = Normalizer(norm='l2')  # L2 norm (Euclidean)
X_normalized = normalizer.fit_transform(X)
```

### 6.2 Distribution Transformations 🟡

#### Log Transformation:
```python
# For right-skewed data
df['log_feature'] = np.log(df['feature'] + 1)  # +1 to handle zeros
```

#### Box-Cox Transformation:
```python
from scipy.stats import boxcox

# Automatically find best lambda
transformed_data, lambda_param = boxcox(df['feature'])
```

#### Yeo-Johnson Transformation:
```python
from sklearn.preprocessing import PowerTransformer

# Works with negative values
transformer = PowerTransformer(method='yeo-johnson')
X_transformed = transformer.fit_transform(X)
```

#### Quantile Transformation:
```python
from sklearn.preprocessing import QuantileTransformer

# Transform to uniform or normal distribution
transformer = QuantileTransformer(output_distribution='normal')
X_transformed = transformer.fit_transform(X)
```

### 6.3 Handling Categorical Variables 🟢

Already covered in Feature Engineering section, but key points:

#### When to Use Each Encoding:
- **One-Hot**: Nominal categories, few categories
- **Label**: Ordinal categories, many categories
- **Target**: High cardinality categories
- **Frequency**: When frequency matters

#### Handling High Cardinality:
```python
# Group rare categories
value_counts = df['category'].value_counts()
rare_categories = value_counts[value_counts < 100].index
df['category_grouped'] = df['category'].replace(rare_categories, 'Other')
```

---

## 7. Data Splitting

### 7.1 Train-Validation-Test Split 🟢

#### Purpose of Each Set:
- **Training set**: Train the model
- **Validation set**: Tune hyperparameters, select model
- **Test set**: Final evaluation, estimate generalization performance

#### Typical Split Ratios:
- **Large datasets**: 70% train, 15% validation, 15% test
- **Medium datasets**: 60% train, 20% validation, 20% test
- **Small datasets**: Use cross-validation instead

#### Implementation:
```python
from sklearn.model_selection import train_test_split

# First split: separate test set
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Second split: separate train and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)
```

### 7.2 Cross-Validation 🟡

#### K-Fold Cross-Validation:
```python
from sklearn.model_selection import KFold, cross_val_score

# K-fold CV
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
```

#### Stratified K-Fold:
```python
from sklearn.model_selection import StratifiedKFold

# Maintains class distribution in each fold
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skfold, scoring='accuracy')
```

#### Time Series Split:
```python
from sklearn.model_selection import TimeSeriesSplit

# For time series data (no data leakage)
tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(model, X, y, cv=tscv, scoring='mse')
```

### 7.3 Handling Imbalanced Data 🟡

#### Detection:
```python
# Check class distribution
print(y.value_counts())
print(y.value_counts(normalize=True))
```

#### Sampling Techniques:

**Under-sampling:**
```python
from imblearn.under_sampling import RandomUnderSampler

undersampler = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = undersampler.fit_resample(X, y)
```

**Over-sampling:**
```python
from imblearn.over_sampling import RandomOverSampler, SMOTE

# Random over-sampling
oversampler = RandomOverSampler(random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(X, y)

# SMOTE (Synthetic Minority Oversampling Technique)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
```

**Combined Sampling:**
```python
from imblearn.combine import SMOTETomek

# SMOTE + Tomek links
smote_tomek = SMOTETomek(random_state=42)
X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
```

---

## 8. Advanced Preprocessing Techniques

### 8.1 Dimensionality Reduction 🟡

#### Principal Component Analysis (PCA):
```python
from sklearn.decomposition import PCA

# Reduce to n components
pca = PCA(n_components=10)
X_reduced = pca.fit_transform(X_scaled)

# Explained variance ratio
print(pca.explained_variance_ratio_)
print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.2f}")
```

#### t-SNE (t-Distributed Stochastic Neighbor Embedding):
```python
from sklearn.manifold import TSNE

# Non-linear dimensionality reduction (mainly for visualization)
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)
```

#### Linear Discriminant Analysis (LDA):
```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Supervised dimensionality reduction
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_scaled, y)
```

### 8.2 Feature Creation from Domain Knowledge 🟡

#### Domain-Specific Features:
```python
# E-commerce example
df['price_per_unit'] = df['total_price'] / df['quantity']
df['discount_percentage'] = (df['original_price'] - df['sale_price']) / df['original_price']

# Time-based features for retail
df['is_holiday_season'] = df['date'].dt.month.isin([11, 12])
df['is_payday'] = df['date'].dt.day.isin([15, 30, 31])

# Geographic features
df['distance_to_city_center'] = np.sqrt(
    (df['latitude'] - city_center_lat)**2 + 
    (df['longitude'] - city_center_lon)**2
)
```

### 8.3 Automated Feature Engineering 🔴

#### Featuretools:
```python
import featuretools as ft

# Create entity set
es = ft.EntitySet(id='data')
es = es.entity_from_dataframe(
    entity_id='customers', 
    dataframe=customers_df, 
    index='customer_id'
)

# Deep feature synthesis
features, feature_defs = ft.dfs(
    entityset=es,
    target_entity='customers',
    max_depth=2
)
```

---

## 9. Data Quality Assessment

### 9.1 Data Quality Metrics 🟢

#### Completeness:
```python
completeness = (1 - df.isnull().sum() / len(df)) * 100
print("Completeness by column:")
print(completeness)
```

#### Consistency:
```python
# Check for inconsistent formats
date_formats = df['date'].apply(lambda x: type(x).__name__).value_counts()
print("Date format consistency:", date_formats)

# Check for inconsistent categorical values
category_variations = df['category'].str.lower().value_counts()
print("Category variations:", category_variations)
```

#### Accuracy:
```python
# Range checks
invalid_ages = df[(df['age'] < 0) | (df['age'] > 150)]
print(f"Invalid ages: {len(invalid_ages)}")

# Format validation
import re
invalid_emails = df[~df['email'].str.match(r'^[\w\.-]+@[\w\.-]+\.\w+$')]
print(f"Invalid emails: {len(invalid_emails)}")
```

#### Uniqueness:
```python
# Check for duplicates in supposed unique columns
duplicate_ids = df['id'].duplicated().sum()
print(f"Duplicate IDs: {duplicate_ids}")
```

### 9.2 Data Profiling 🟡

#### Automated Profiling:
```python
import pandas_profiling

# Generate comprehensive data report
profile = pandas_profiling.ProfileReport(df)
profile.to_file("data_profile_report.html")
```

#### Custom Profiling Function:
```python
def profile_dataframe(df):
    """Custom data profiling function"""
    profile = {}
    
    for column in df.columns:
        col_profile = {
            'dtype': str(df[column].dtype),
            'missing_count': df[column].isnull().sum(),
            'missing_percentage': df[column].isnull().sum() / len(df) * 100,
            'unique_count': df[column].nunique(),
            'unique_percentage': df[column].nunique() / len(df) * 100
        }
        
        if df[column].dtype in ['int64', 'float64']:
            col_profile.update({
                'mean': df[column].mean(),
                'std': df[column].std(),
                'min': df[column].min(),
                'max': df[column].max(),
                'median': df[column].median()
            })
        
        profile[column] = col_profile
    
    return profile
```

---

## 10. Best Practices and Common Pitfalls

### 10.1 Best Practices 🟢

#### Data Preprocessing Pipeline:
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Create preprocessing pipeline
numeric_features = ['age', 'income']
categorical_features = ['category', 'region']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Combine with model
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])
```

#### Key Principles:
1. **Understand your data first**: Always do EDA before preprocessing
2. **Document everything**: Keep track of all transformations
3. **Version control**: Save different versions of processed data
4. **Validate transformations**: Check that transformations make sense
5. **Consider domain knowledge**: Use expertise to guide feature engineering
6. **Test on unseen data**: Ensure preprocessing generalizes

### 10.2 Common Pitfalls 🔴

#### Data Leakage:
```python
# WRONG: Scale entire dataset then split
X_scaled = StandardScaler().fit_transform(X)
X_train, X_test = train_test_split(X_scaled, test_size=0.2)

# CORRECT: Split first, then scale
X_train, X_test = train_test_split(X, test_size=0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Only transform, don't fit
```

#### Target Leakage:
```python
# WRONG: Using future information to predict past
df['target_mean_by_category'] = df.groupby('category')['target'].mean()

# CORRECT: Use only past information
df['target_mean_by_category'] = df.groupby('category')['target'].expanding().mean().shift(1)
```

#### Inconsistent Preprocessing:
```python
# Ensure same preprocessing for train and test
def preprocess_data(df, is_training=True):
    if is_training:
        # Fit and transform
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df)
        return df_scaled, scaler
    else:
        # Only transform using fitted scaler
        df_scaled = scaler.transform(df)
        return df_scaled
```

---

## 🎯 Key Takeaways

### Essential Preprocessing Steps:
1. **Explore and understand** your data thoroughly
2. **Clean the data**: Handle missing values, outliers, duplicates
3. **Engineer features**: Create meaningful features from raw data
4. **Select features**: Choose most relevant features
5. **Transform data**: Scale, encode, normalize as needed
6. **Split data properly**: Avoid data leakage
7. **Validate results**: Ensure preprocessing improves model performance

### Remember:
- **Quality over quantity**: Better features beat more features
- **Domain knowledge is crucial**: Use expertise to guide decisions
- **Preprocessing is iterative**: Refine based on model performance
- **Document everything**: Reproducibility is key
- **Test thoroughly**: Ensure preprocessing generalizes to new data

---

## 📚 Next Steps

Ready to apply your preprocessed data? Continue with:
- **[Supervised Learning](04_Supervised_Learning.md)** - Apply preprocessing to classification and regression
- **[Unsupervised Learning](05_Unsupervised_Learning.md)** - Explore patterns in your clean data

---

## 🛠️ Practical Exercises

### Exercise 1: Data Cleaning
Given a messy dataset with missing values, outliers, and duplicates:
1. Perform EDA to understand data quality issues
2. Handle missing values appropriately
3. Detect and handle outliers
4. Remove duplicates

### Exercise 2: Feature Engineering
Create new features for a sales dataset:
1. Time-based features from date columns
2. Interaction features between price and quantity
3. Categorical encoding for product categories
4. Text features from product descriptions

### Exercise 3: Complete Preprocessing Pipeline
Build an end-to-end preprocessing pipeline:
1. Handle mixed data types (numerical, categorical, text)
2. Create robust preprocessing steps
3. Ensure no data leakage
4. Validate preprocessing improves model performance

---

*Next: [Supervised Learning →](04_Supervised_Learning.md)*
