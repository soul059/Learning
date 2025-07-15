# 05. Unsupervised Learning

## 🎯 Learning Objectives
- Master clustering algorithms and techniques
- Understand dimensionality reduction methods
- Learn association rule mining
- Apply unsupervised learning to real-world problems

---

## 1. Introduction to Unsupervised Learning

**Unsupervised Learning** discovers hidden patterns in data without labeled examples. Unlike supervised learning, there's no target variable to predict.

### 1.1 Key Characteristics 🟢

#### What Makes It Different:
- **No labels**: Only input data (X), no target variable (y)
- **Pattern discovery**: Find hidden structures in data
- **Exploratory**: Often used for data exploration and understanding
- **Validation challenges**: No clear "correct" answer to compare against

#### Common Applications:
- **Customer segmentation**: Group customers by behavior
- **Market basket analysis**: Find products bought together
- **Data compression**: Reduce dimensionality while preserving information
- **Anomaly detection**: Identify unusual patterns
- **Data visualization**: Reduce dimensions for plotting

### 1.2 Types of Unsupervised Learning 🟢

#### Main Categories:
1. **Clustering**: Group similar data points
2. **Association Rules**: Find relationships between variables
3. **Dimensionality Reduction**: Reduce number of features
4. **Density Estimation**: Estimate probability distributions
5. **Anomaly Detection**: Identify outliers

---

## 2. Clustering

**Clustering** groups data points so that points within the same cluster are more similar to each other than to points in other clusters.

### 2.1 K-Means Clustering 🟢

**Concept**: Partition data into k clusters by minimizing within-cluster sum of squares.

#### Algorithm:
1. Choose number of clusters (k)
2. Initialize k centroids randomly
3. Assign each point to nearest centroid
4. Update centroids to mean of assigned points
5. Repeat steps 3-4 until convergence

#### Mathematical Foundation:
```
Objective: Minimize WCSS = Σᵢ₌₁ᵏ Σₓ∈Cᵢ ||x - μᵢ||²
where μᵢ is centroid of cluster i
```

#### Implementation:
```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# Create and fit model
kmeans = KMeans(
    n_clusters=3,
    init='k-means++',  # Smart initialization
    n_init=10,         # Number of random initializations
    max_iter=300,      # Maximum iterations
    random_state=42
)

cluster_labels = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_

# Visualize results (for 2D data)
plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200)
plt.title('K-Means Clustering')
plt.show()
```

#### From Scratch Implementation:
```python
class KMeansFromScratch:
    def __init__(self, k=3, max_iters=100, random_state=None):
        self.k = k
        self.max_iters = max_iters
        self.random_state = random_state
    
    def fit(self, X):
        if self.random_state:
            np.random.seed(self.random_state)
        
        # Initialize centroids randomly
        n_samples, n_features = X.shape
        self.centroids = np.random.rand(self.k, n_features)
        
        for _ in range(self.max_iters):
            # Assign points to nearest centroid
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)
            
            # Update centroids
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.k)])
            
            # Check for convergence
            if np.allclose(self.centroids, new_centroids):
                break
            
            self.centroids = new_centroids
        
        return labels
    
    def predict(self, X):
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)
```

#### Choosing Optimal k:

**Elbow Method:**
```python
def find_optimal_k_elbow(X, max_k=10):
    wcss = []
    k_range = range(1, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, wcss, 'bo-')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS')
    plt.show()
    
    return k_range, wcss
```

**Silhouette Analysis:**
```python
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.cm as cm

def silhouette_analysis(X, max_k=10):
    silhouette_scores = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, silhouette_scores, 'bo-')
    plt.title('Silhouette Analysis')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Average Silhouette Score')
    plt.show()
    
    optimal_k = k_range[np.argmax(silhouette_scores)]
    return optimal_k, silhouette_scores
```

#### Advantages:
- ✅ Simple and fast
- ✅ Works well with spherical clusters
- ✅ Scales well to large datasets
- ✅ Guaranteed convergence

#### Disadvantages:
- ❌ Need to specify k in advance
- ❌ Sensitive to initialization
- ❌ Assumes spherical clusters
- ❌ Sensitive to outliers
- ❌ Struggles with varying cluster sizes

### 2.2 Hierarchical Clustering 🟡

**Concept**: Create tree-like hierarchy of clusters.

#### Types:
1. **Agglomerative (Bottom-up)**: Start with individual points, merge similar clusters
2. **Divisive (Top-down)**: Start with all points, recursively split clusters

#### Linkage Criteria:
- **Single**: Minimum distance between clusters
- **Complete**: Maximum distance between clusters
- **Average**: Average distance between all pairs
- **Ward**: Minimize within-cluster variance

#### Implementation:
```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Agglomerative clustering
agg_clustering = AgglomerativeClustering(
    n_clusters=3,
    linkage='ward'  # or 'single', 'complete', 'average'
)

cluster_labels = agg_clustering.fit_predict(X)

# Create dendrogram
linkage_matrix = linkage(X, method='ward')
plt.figure(figsize=(12, 8))
dendrogram(linkage_matrix, truncate_mode='level', p=3)
plt.title('Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()
```

#### Dendrogram Interpretation:
- **Height**: Distance at which clusters merge
- **Horizontal lines**: Cluster merges
- **Vertical lines**: Represent clusters
- **Cut line**: Determines number of clusters

#### Advantages:
- ✅ No need to specify number of clusters in advance
- ✅ Produces hierarchy of clusters
- ✅ Deterministic results
- ✅ Works with any distance metric

#### Disadvantages:
- ❌ Computationally expensive O(n³)
- ❌ Sensitive to noise and outliers
- ❌ Difficult to handle large datasets
- ❌ Greedy algorithm (can't undo merges)

### 2.3 DBSCAN (Density-Based Clustering) 🟡

**Concept**: Group points that are closely packed while marking outliers in low-density regions.

#### Key Parameters:
- **eps (ε)**: Maximum distance between two points to be neighbors
- **min_samples**: Minimum points needed to form dense region

#### Point Types:
- **Core points**: Have at least min_samples neighbors within eps
- **Border points**: Not core but within eps of core point
- **Noise points**: Neither core nor border

#### Algorithm:
1. For each point, find all neighbors within eps distance
2. If point has ≥ min_samples neighbors, mark as core point
3. Create clusters by connecting core points within eps distance
4. Assign border points to nearest cluster
5. Mark remaining points as noise

#### Implementation:
```python
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np

# Standardize features (important for DBSCAN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply DBSCAN
dbscan = DBSCAN(
    eps=0.5,           # Maximum distance between neighbors
    min_samples=5      # Minimum samples in neighborhood
)

cluster_labels = dbscan.fit_predict(X_scaled)

# Number of clusters (excluding noise)
n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
n_noise = list(cluster_labels).count(-1)

print(f'Number of clusters: {n_clusters}')
print(f'Number of noise points: {n_noise}')

# Visualize results
unique_labels = set(cluster_labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black for noise
        col = [0, 0, 0, 1]
    
    class_member_mask = (cluster_labels == k)
    xy = X_scaled[class_member_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('DBSCAN Clustering')
plt.show()
```

#### Parameter Selection:
```python
def find_optimal_eps(X, min_samples=5):
    from sklearn.neighbors import NearestNeighbors
    
    # Calculate k-distance graph
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(X)
    distances, indices = neighbors_fit.kneighbors(X)
    
    # Sort distances
    distances = np.sort(distances[:, min_samples-1], axis=0)
    
    # Plot k-distance graph
    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.title('k-distance Graph')
    plt.xlabel('Data Points sorted by distance')
    plt.ylabel(f'{min_samples}-NN distance')
    plt.show()
    
    return distances
```

#### Advantages:
- ✅ Automatically determines number of clusters
- ✅ Can find arbitrary shaped clusters
- ✅ Robust to outliers (marks them as noise)
- ✅ Doesn't require cluster centers

#### Disadvantages:
- ❌ Sensitive to hyperparameters (eps, min_samples)
- ❌ Struggles with varying densities
- ❌ Can be sensitive to distance metric
- ❌ Memory intensive for large datasets

### 2.4 Gaussian Mixture Models (GMM) 🔴

**Concept**: Assume data comes from mixture of Gaussian distributions.

#### Mathematical Foundation:
```
P(x) = Σₖ₌₁ᴷ πₖ N(x|μₖ, Σₖ)
```

Where:
- πₖ: Mixing coefficient (weight) of component k
- N(x|μₖ, Σₖ): Gaussian distribution with mean μₖ and covariance Σₖ

#### Expectation-Maximization (EM) Algorithm:
1. **E-step**: Calculate probability of each point belonging to each component
2. **M-step**: Update parameters based on weighted assignments
3. Repeat until convergence

#### Implementation:
```python
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Fit GMM
gmm = GaussianMixture(
    n_components=3,
    covariance_type='full',  # 'full', 'tied', 'diag', 'spherical'
    random_state=42
)

gmm.fit(X)

# Get cluster assignments
cluster_labels = gmm.predict(X)
probabilities = gmm.predict_proba(X)

# Get parameters
weights = gmm.weights_
means = gmm.means_
covariances = gmm.covariances_

print(f"Weights: {weights}")
print(f"Means: {means}")

# Calculate AIC and BIC for model selection
aic = gmm.aic(X)
bic = gmm.bic(X)
print(f"AIC: {aic:.2f}, BIC: {bic:.2f}")
```

#### Model Selection:
```python
def select_gmm_components(X, max_components=10):
    n_components_range = range(1, max_components + 1)
    aic_scores = []
    bic_scores = []
    
    for n_components in n_components_range:
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit(X)
        aic_scores.append(gmm.aic(X))
        bic_scores.append(gmm.bic(X))
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(n_components_range, aic_scores, 'bo-')
    plt.title('AIC Score')
    plt.xlabel('Number of Components')
    plt.ylabel('AIC')
    
    plt.subplot(1, 2, 2)
    plt.plot(n_components_range, bic_scores, 'ro-')
    plt.title('BIC Score')
    plt.xlabel('Number of Components')
    plt.ylabel('BIC')
    
    plt.tight_layout()
    plt.show()
    
    optimal_aic = n_components_range[np.argmin(aic_scores)]
    optimal_bic = n_components_range[np.argmin(bic_scores)]
    
    return optimal_aic, optimal_bic
```

#### Advantages:
- ✅ Probabilistic clustering (soft assignments)
- ✅ Can model elliptical clusters
- ✅ Provides cluster probabilities
- ✅ Statistical foundation

#### Disadvantages:
- ❌ Assumes Gaussian distributions
- ❌ Sensitive to initialization
- ❌ Can overfit with too many components
- ❌ Computationally expensive

### 2.5 Clustering Evaluation 🟡

#### Internal Measures (No ground truth needed):

**Silhouette Score:**
```python
from sklearn.metrics import silhouette_score

# Calculate average silhouette score
silhouette_avg = silhouette_score(X, cluster_labels)
print(f"Average silhouette score: {silhouette_avg:.3f}")

# Silhouette score ranges from -1 to 1
# Higher values indicate better clustering
```

**Calinski-Harabasz Index:**
```python
from sklearn.metrics import calinski_harabasz_score

ch_score = calinski_harabasz_score(X, cluster_labels)
print(f"Calinski-Harabasz Index: {ch_score:.3f}")
# Higher values indicate better clustering
```

**Davies-Bouldin Index:**
```python
from sklearn.metrics import davies_bouldin_score

db_score = davies_bouldin_score(X, cluster_labels)
print(f"Davies-Bouldin Index: {db_score:.3f}")
# Lower values indicate better clustering
```

#### External Measures (Ground truth available):

**Adjusted Rand Index:**
```python
from sklearn.metrics import adjusted_rand_score

ari = adjusted_rand_score(true_labels, cluster_labels)
print(f"Adjusted Rand Index: {ari:.3f}")
# Ranges from -1 to 1, higher is better
```

**Normalized Mutual Information:**
```python
from sklearn.metrics import normalized_mutual_info_score

nmi = normalized_mutual_info_score(true_labels, cluster_labels)
print(f"Normalized Mutual Information: {nmi:.3f}")
# Ranges from 0 to 1, higher is better
```

---

## 3. Dimensionality Reduction

**Dimensionality Reduction** reduces the number of features while preserving important information.

### 3.1 Principal Component Analysis (PCA) 🟢

**Concept**: Find orthogonal directions (principal components) that capture maximum variance in data.

#### Mathematical Foundation:
1. Standardize data
2. Compute covariance matrix
3. Find eigenvalues and eigenvectors
4. Sort by eigenvalue magnitude
5. Transform data using top k eigenvectors

#### Key Concepts:
- **Principal Components**: Directions of maximum variance
- **Explained Variance**: Amount of variance captured by each component
- **Loadings**: Weights of original features in components

#### Implementation:
```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Standardize data (important for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Explained variance
explained_variance = pca.explained_variance_ratio_
print(f"Explained variance ratio: {explained_variance}")
print(f"Total explained variance: {explained_variance.sum():.3f}")

# Components (loadings)
components = pca.components_
print(f"Principal components shape: {components.shape}")

# Visualize results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel(f'PC1 ({explained_variance[0]:.1%} variance)')
plt.ylabel(f'PC2 ({explained_variance[1]:.1%} variance)')
plt.title('PCA Result')

plt.subplot(1, 2, 2)
cumulative_variance = np.cumsum(explained_variance)
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
plt.xlabel('Principal Component')
plt.ylabel('Cumulative Explained Variance')
plt.title('Scree Plot')

plt.tight_layout()
plt.show()
```

#### Choosing Number of Components:
```python
def choose_pca_components(X, threshold=0.95):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit PCA with all components
    pca = PCA()
    pca.fit(X_scaled)
    
    # Calculate cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    # Find number of components for threshold
    n_components = np.argmax(cumulative_variance >= threshold) + 1
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
    plt.axhline(y=threshold, color='r', linestyle='--', 
                label=f'{threshold:.0%} variance threshold')
    plt.axvline(x=n_components, color='r', linestyle='--',
                label=f'{n_components} components needed')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Component Selection')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return n_components
```

#### From Scratch Implementation:
```python
class PCAFromScratch:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        
    def fit(self, X):
        # Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Compute covariance matrix
        cov_matrix = np.cov(X_centered.T)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Sort by eigenvalues (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Store top n_components
        self.components = eigenvectors[:, :self.n_components].T
        self.explained_variance_ratio_ = eigenvalues[:self.n_components] / np.sum(eigenvalues)
        
    def transform(self, X):
        X_centered = X - self.mean
        return X_centered @ self.components.T
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
```

#### Advantages:
- ✅ Removes redundancy and noise
- ✅ Reduces computational complexity
- ✅ Helps with visualization
- ✅ Statistical foundation

#### Disadvantages:
- ❌ Components may not be interpretable
- ❌ Linear transformation only
- ❌ May lose important information
- ❌ Sensitive to scaling

### 3.2 t-SNE (t-Distributed Stochastic Neighbor Embedding) 🟡

**Concept**: Non-linear dimensionality reduction for visualization by preserving local similarities.

#### Key Ideas:
- Preserves local neighborhood structure
- Converts similarities to probabilities
- Minimizes KL divergence between high and low dimensional probabilities

#### Implementation:
```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Apply t-SNE
tsne = TSNE(
    n_components=2,      # Usually 2 or 3 for visualization
    perplexity=30,       # Number of nearest neighbors
    learning_rate=200,   # Learning rate
    n_iter=1000,        # Number of iterations
    random_state=42
)

X_tsne = tsne.fit_transform(X_scaled)

# Visualize
plt.figure(figsize=(10, 8))
if y is not None:  # If labels available
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')
    plt.colorbar(scatter)
else:
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1])

plt.title('t-SNE Visualization')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()
```

#### Hyperparameter Tuning:
```python
def tune_tsne_perplexity(X, perplexities=[5, 30, 50, 100]):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, perplexity in enumerate(perplexities):
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        X_tsne = tsne.fit_transform(X)
        
        axes[i].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')
        axes[i].set_title(f'Perplexity = {perplexity}')
        axes[i].set_xlabel('t-SNE Component 1')
        axes[i].set_ylabel('t-SNE Component 2')
    
    plt.tight_layout()
    plt.show()
```

#### Best Practices:
- Apply PCA first to reduce to ~50 dimensions
- Try different perplexity values
- Run multiple times with different random seeds
- Don't interpret distances between clusters

#### Advantages:
- ✅ Excellent for visualization
- ✅ Preserves local structure
- ✅ Can reveal hidden patterns
- ✅ Works well with complex data

#### Disadvantages:
- ❌ Non-deterministic results
- ❌ Computationally expensive
- ❌ Global structure not preserved
- ❌ Hyperparameter sensitive

### 3.3 UMAP (Uniform Manifold Approximation and Projection) 🟡

**Concept**: Preserve both local and global structure of data using topological methods.

#### Implementation:
```python
import umap

# Apply UMAP
umap_reducer = umap.UMAP(
    n_neighbors=15,      # Local neighborhood size
    min_dist=0.1,        # Minimum distance between points
    n_components=2,      # Output dimensions
    metric='euclidean',  # Distance metric
    random_state=42
)

X_umap = umap_reducer.fit_transform(X_scaled)

# Visualize
plt.figure(figsize=(10, 8))
if y is not None:
    scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='viridis')
    plt.colorbar(scatter)
else:
    plt.scatter(X_umap[:, 0], X_umap[:, 1])

plt.title('UMAP Visualization')
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.show()
```

#### Advantages over t-SNE:
- ✅ Faster computation
- ✅ Preserves global structure better
- ✅ Can be used for general dimensionality reduction
- ✅ More consistent results

### 3.4 Linear Discriminant Analysis (LDA) 🟡

**Concept**: Supervised dimensionality reduction that maximizes class separability.

#### Implementation:
```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# LDA requires labels
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_scaled, y)

# Explained variance ratio
explained_variance = lda.explained_variance_ratio_
print(f"Explained variance ratio: {explained_variance}")

# Visualize
plt.figure(figsize=(10, 6))
if len(np.unique(y)) > 2:
    scatter = plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap='viridis')
    plt.colorbar(scatter)
else:
    for class_value in np.unique(y):
        mask = y == class_value
        plt.scatter(X_lda[mask, 0], X_lda[mask, 1], label=f'Class {class_value}')
    plt.legend()

plt.xlabel(f'LD1 ({explained_variance[0]:.1%} variance)')
plt.ylabel(f'LD2 ({explained_variance[1]:.1%} variance)')
plt.title('Linear Discriminant Analysis')
plt.show()
```

#### Differences from PCA:
- **PCA**: Maximizes variance (unsupervised)
- **LDA**: Maximizes class separability (supervised)
- **When to use**: LDA for classification, PCA for general dimensionality reduction

---

## 4. Association Rule Mining

**Association Rule Mining** discovers relationships between variables, commonly used in market basket analysis.

### 4.1 Basic Concepts 🟢

#### Key Terms:
- **Itemset**: Collection of items
- **Transaction**: Single instance (e.g., shopping basket)
- **Support**: Frequency of itemset in dataset
- **Confidence**: Conditional probability
- **Lift**: Ratio of observed vs. expected co-occurrence

#### Rule Format:
```
If {A, B} then {C}
```

#### Metrics:
```
Support(A) = P(A) = |A| / |D|
Confidence(A → B) = P(B|A) = Support(A ∪ B) / Support(A)
Lift(A → B) = Confidence(A → B) / Support(B)
```

### 4.2 Apriori Algorithm 🟡

**Concept**: Generate frequent itemsets by level-wise search.

#### Algorithm:
1. Find frequent 1-itemsets
2. Use frequent k-itemsets to generate candidate (k+1)-itemsets
3. Prune candidates using minimum support
4. Repeat until no frequent itemsets found
5. Generate rules from frequent itemsets

#### Implementation:
```python
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

# Sample transaction data (binary matrix)
# Rows = transactions, Columns = items, Values = 0/1
transactions = pd.DataFrame({
    'bread': [1, 1, 0, 1, 1],
    'milk': [1, 1, 1, 0, 1],
    'butter': [1, 0, 0, 1, 1],
    'jam': [0, 1, 1, 0, 1],
    'eggs': [1, 0, 1, 1, 0]
})

# Find frequent itemsets
frequent_itemsets = apriori(transactions, min_support=0.4, use_colnames=True)
print("Frequent Itemsets:")
print(frequent_itemsets)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
print("\nAssociation Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
```

#### Rule Evaluation:
```python
def evaluate_rules(rules):
    # Sort by lift (descending)
    rules_sorted = rules.sort_values('lift', ascending=False)
    
    print("Top 10 rules by lift:")
    print(rules_sorted[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))
    
    # Filter by confidence and lift
    strong_rules = rules[(rules['confidence'] >= 0.8) & (rules['lift'] >= 1.2)]
    print(f"\nStrong rules (confidence >= 0.8, lift >= 1.2): {len(strong_rules)}")
    
    return strong_rules
```

#### Real-World Example - Market Basket:
```python
# Load retail dataset
def create_basket_analysis(transactions_df):
    # Group by transaction ID and aggregate items
    basket = transactions_df.groupby(['transaction_id', 'item'])['quantity'].sum().unstack().fillna(0)
    
    # Convert to binary (0/1)
    basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)
    
    # Apply Apriori
    frequent_itemsets = apriori(basket_sets, min_support=0.01, use_colnames=True)
    
    # Generate rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
    
    return frequent_itemsets, rules
```

### 4.3 FP-Growth Algorithm 🔴

**Concept**: More efficient algorithm using FP-tree data structure.

#### Advantages over Apriori:
- No candidate generation
- Compresses database into tree structure
- Faster for large datasets

#### Implementation:
```python
from mlxtend.frequent_patterns import fpgrowth

# Apply FP-Growth
frequent_itemsets_fp = fpgrowth(transactions, min_support=0.4, use_colnames=True)
print("Frequent Itemsets (FP-Growth):")
print(frequent_itemsets_fp)

# Generate rules
rules_fp = association_rules(frequent_itemsets_fp, metric="confidence", min_threshold=0.6)
```

### 4.4 Applications 🟡

#### Market Basket Analysis:
```python
def market_basket_insights(rules):
    # Product recommendations
    print("Product Recommendations:")
    for _, rule in rules.iterrows():
        antecedent = list(rule['antecedents'])[0]
        consequent = list(rule['consequents'])[0]
        confidence = rule['confidence']
        print(f"Customers who buy {antecedent} also buy {consequent} ({confidence:.1%} confidence)")
    
    # Cross-selling opportunities
    high_lift_rules = rules[rules['lift'] > 1.5]
    print(f"\nCross-selling opportunities (lift > 1.5): {len(high_lift_rules)}")
```

#### Web Usage Patterns:
```python
# Analyze web page navigation patterns
def web_usage_analysis(clickstream_data):
    # Convert clickstream to transaction format
    sessions = clickstream_data.groupby('session_id')['page'].apply(list)
    
    # Create transaction matrix
    all_pages = set([page for session in sessions for page in session])
    transactions = pd.DataFrame(0, index=sessions.index, columns=list(all_pages))
    
    for session_id, pages in sessions.items():
        for page in pages:
            transactions.loc[session_id, page] = 1
    
    # Apply association rule mining
    frequent_itemsets = apriori(transactions, min_support=0.05)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)
    
    return rules
```

---

## 5. Anomaly Detection

**Anomaly Detection** identifies data points that deviate significantly from normal patterns.

### 5.1 Statistical Methods 🟢

#### Z-Score Method:
```python
from scipy import stats

def detect_anomalies_zscore(data, threshold=3):
    z_scores = np.abs(stats.zscore(data))
    anomalies = data[z_scores > threshold]
    return anomalies, z_scores > threshold

# Example usage
anomalies, is_anomaly = detect_anomalies_zscore(X[:, 0])
print(f"Number of anomalies: {np.sum(is_anomaly)}")
```

#### Isolation Forest:
```python
from sklearn.ensemble import IsolationForest

# Isolation Forest
iso_forest = IsolationForest(
    contamination=0.1,  # Expected proportion of anomalies
    random_state=42
)

anomaly_labels = iso_forest.fit_predict(X)
anomalies = X[anomaly_labels == -1]

print(f"Number of anomalies detected: {len(anomalies)}")

# Visualize
plt.figure(figsize=(10, 6))
normal_points = X[anomaly_labels == 1]
anomaly_points = X[anomaly_labels == -1]

plt.scatter(normal_points[:, 0], normal_points[:, 1], c='blue', label='Normal')
plt.scatter(anomaly_points[:, 0], anomaly_points[:, 1], c='red', label='Anomaly')
plt.legend()
plt.title('Isolation Forest Anomaly Detection')
plt.show()
```

### 5.2 Distance-Based Methods 🟡

#### Local Outlier Factor (LOF):
```python
from sklearn.neighbors import LocalOutlierFactor

# LOF
lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.1
)

anomaly_labels = lof.fit_predict(X)
outlier_scores = lof.negative_outlier_factor_

# Visualize
plt.figure(figsize=(10, 6))
normal_points = X[anomaly_labels == 1]
anomaly_points = X[anomaly_labels == -1]

plt.scatter(normal_points[:, 0], normal_points[:, 1], c='blue', label='Normal')
plt.scatter(anomaly_points[:, 0], anomaly_points[:, 1], c='red', label='Anomaly')
plt.legend()
plt.title('LOF Anomaly Detection')
plt.show()
```

### 5.3 Clustering-Based Methods 🟡

#### Using K-Means for Anomaly Detection:
```python
def kmeans_anomaly_detection(X, n_clusters=8, threshold_percentile=95):
    # Fit K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    
    # Calculate distances to centroids
    centroids = kmeans.cluster_centers_
    distances = []
    
    for i, point in enumerate(X):
        cluster = cluster_labels[i]
        distance = np.linalg.norm(point - centroids[cluster])
        distances.append(distance)
    
    distances = np.array(distances)
    
    # Define threshold
    threshold = np.percentile(distances, threshold_percentile)
    anomalies = distances > threshold
    
    return anomalies, distances

anomalies, distances = kmeans_anomaly_detection(X)
print(f"Number of anomalies: {np.sum(anomalies)}")
```

---

## 6. Evaluation and Validation

### 6.1 Clustering Validation 🟡

#### Silhouette Analysis in Detail:
```python
def detailed_silhouette_analysis(X, cluster_labels):
    from sklearn.metrics import silhouette_samples, silhouette_score
    
    # Calculate silhouette scores
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    silhouette_avg = silhouette_score(X, cluster_labels)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Silhouette plot
    y_lower = 10
    for i in range(len(np.unique(cluster_labels))):
        cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        cluster_silhouette_values.sort()
        
        size_cluster_i = cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = plt.cm.nipy_spectral(float(i) / len(np.unique(cluster_labels)))
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_values,
                         facecolor=color, edgecolor=color, alpha=0.7)
        
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    
    ax1.set_xlabel('Silhouette coefficient values')
    ax1.set_ylabel('Cluster label')
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    
    # Scatter plot
    colors = plt.cm.nipy_spectral(cluster_labels.astype(float) / len(np.unique(cluster_labels)))
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7, c=colors)
    ax2.set_title('Clustered data')
    
    plt.tight_layout()
    plt.show()
    
    return silhouette_avg
```

### 6.2 Dimensionality Reduction Validation 🟡

#### Reconstruction Error for PCA:
```python
def pca_reconstruction_error(X, n_components_list):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    reconstruction_errors = []
    
    for n_components in n_components_list:
        pca = PCA(n_components=n_components)
        X_transformed = pca.fit_transform(X_scaled)
        X_reconstructed = pca.inverse_transform(X_transformed)
        
        error = np.mean((X_scaled - X_reconstructed) ** 2)
        reconstruction_errors.append(error)
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_components_list, reconstruction_errors, 'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Reconstruction Error')
    plt.title('PCA Reconstruction Error')
    plt.grid(True)
    plt.show()
    
    return reconstruction_errors
```

---

## 🎯 Key Takeaways

### Algorithm Selection Guide:

#### For Clustering:
- **K-Means**: Spherical clusters, known number of clusters
- **Hierarchical**: Unknown number of clusters, small datasets
- **DBSCAN**: Arbitrary shapes, noise handling, varying densities
- **GMM**: Overlapping clusters, probabilistic assignments

#### For Dimensionality Reduction:
- **PCA**: Linear relationships, interpretability needed
- **t-SNE**: Visualization, non-linear relationships
- **UMAP**: Faster than t-SNE, preserves global structure
- **LDA**: Supervised, maximizing class separability

#### For Association Rules:
- **Apriori**: Educational, small datasets
- **FP-Growth**: Large datasets, better performance

### Best Practices:
1. **Start with exploration**: Use visualization and simple methods first
2. **Preprocess carefully**: Scaling is crucial for many algorithms
3. **Validate results**: Use multiple evaluation metrics
4. **Domain knowledge**: Interpret results in context
5. **Parameter tuning**: Systematically test different parameters

---

## 📚 Next Steps

Continue your ML journey with:
- **[Reinforcement Learning](06_Reinforcement_Learning.md)** - Learning through interaction
- **[Deep Learning](07_Deep_Learning.md)** - Neural networks and deep architectures

---

## 🛠️ Practical Exercises

### Exercise 1: Customer Segmentation
Use clustering to segment customers:
1. Apply K-means, hierarchical, and DBSCAN
2. Compare results using silhouette analysis
3. Interpret business meaning of clusters
4. Create customer profiles for each segment

### Exercise 2: Dimensionality Reduction Pipeline
Build complete dimensionality reduction pipeline:
1. Start with high-dimensional dataset
2. Apply PCA for initial reduction
3. Use t-SNE for visualization
4. Compare with UMAP results
5. Evaluate information preservation

### Exercise 3: Market Basket Analysis
Analyze transaction data:
1. Apply Apriori algorithm
2. Generate association rules
3. Identify cross-selling opportunities
4. Create recommendation system
5. Validate with business metrics

---

*Next: [Reinforcement Learning →](06_Reinforcement_Learning.md)*
