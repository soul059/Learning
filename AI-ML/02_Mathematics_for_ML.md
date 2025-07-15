# 02. Mathematics for Machine Learning

## 🎯 Learning Objectives
- Master essential linear algebra concepts for ML
- Understand calculus applications in optimization
- Learn probability and statistics fundamentals
- Apply mathematical concepts to ML algorithms

---

## 1. Linear Algebra Foundations

Linear algebra is the backbone of machine learning, providing the mathematical framework for data representation and algorithm implementation.

### 1.1 Vectors 🟢

**Definition**: A vector is an ordered list of numbers representing magnitude and direction.

#### Vector Notation:
```
Column Vector: v = [v₁]    Row Vector: v = [v₁, v₂, v₃]
                  [v₂]
                  [v₃]
```

#### Vector Operations:

**Addition/Subtraction:**
```
a = [1, 2, 3]
b = [4, 5, 6]
a + b = [5, 7, 9]
a - b = [-3, -3, -3]
```

**Scalar Multiplication:**
```
c = 3
c * a = [3, 6, 9]
```

**Dot Product (Inner Product):**
```
a · b = a₁b₁ + a₂b₂ + a₃b₃ = 1×4 + 2×5 + 3×6 = 32
```

**Magnitude (Euclidean Norm):**
```
||a|| = √(a₁² + a₂² + a₃²) = √(1² + 2² + 3²) = √14
```

#### ML Applications:
- **Feature vectors**: Representing data points
- **Weight vectors**: Model parameters
- **Similarity measurement**: Cosine similarity using dot product

### 1.2 Matrices 🟢

**Definition**: A rectangular array of numbers arranged in rows and columns.

#### Matrix Notation:
```
A = [a₁₁  a₁₂  a₁₃]  (m × n matrix: m rows, n columns)
    [a₂₁  a₂₂  a₂₃]
```

#### Matrix Operations:

**Addition/Subtraction:**
```
A + B = [a₁₁+b₁₁  a₁₂+b₁₂]
        [a₂₁+b₂₁  a₂₂+b₂₂]
```

**Matrix Multiplication:**
```
C = A × B where C[i,j] = Σ(A[i,k] × B[k,j])

Example:
[1  2] × [5  6] = [1×5+2×7  1×6+2×8] = [19  22]
[3  4]   [7  8]   [3×5+4×7  3×6+4×8]   [43  50]
```

**Transpose:**
```
A = [1  2  3]  →  Aᵀ = [1  4]
    [4  5  6]          [2  5]
                       [3  6]
```

**Identity Matrix:**
```
I = [1  0  0]  (AI = IA = A)
    [0  1  0]
    [0  0  1]
```

**Inverse Matrix:**
```
A⁻¹ such that AA⁻¹ = A⁻¹A = I
```

#### Special Matrices:

**Symmetric Matrix:** A = Aᵀ
**Orthogonal Matrix:** AAᵀ = I
**Diagonal Matrix:** Non-zero elements only on diagonal

#### ML Applications:
- **Data matrices**: Rows = samples, columns = features
- **Transformation matrices**: Linear transformations
- **Covariance matrices**: Feature relationships

### 1.3 Eigenvalues and Eigenvectors 🟡

**Definition**: For matrix A, vector v is an eigenvector with eigenvalue λ if:
```
Av = λv
```

#### Key Properties:
- Eigenvectors show directions of maximum variance
- Eigenvalues indicate the magnitude of variance
- Used in dimensionality reduction (PCA)

#### Calculation Example:
```
For A = [3  1], find eigenvalues:
        [0  2]

det(A - λI) = 0
det([3-λ  1  ]) = (3-λ)(2-λ) = 0
   ([0   2-λ])

λ₁ = 3, λ₂ = 2
```

#### ML Applications:
- **Principal Component Analysis (PCA)**
- **Spectral clustering**
- **Facial recognition** (eigenfaces)

### 1.4 Matrix Decompositions 🟡

#### Singular Value Decomposition (SVD):
```
A = UΣVᵀ
```
Where:
- U: Left singular vectors
- Σ: Diagonal matrix of singular values
- V: Right singular vectors

#### ML Applications:
- **Dimensionality reduction**
- **Recommendation systems**
- **Image compression**
- **Latent Semantic Analysis**

---

## 2. Calculus for Optimization

Calculus provides the mathematical foundation for optimizing machine learning models.

### 2.1 Derivatives 🟢

**Definition**: Rate of change of a function with respect to its variable.

#### Basic Rules:
```
d/dx(c) = 0                    (constant)
d/dx(x^n) = nx^(n-1)          (power rule)
d/dx(e^x) = e^x               (exponential)
d/dx(ln(x)) = 1/x             (logarithm)
d/dx(sin(x)) = cos(x)         (trigonometric)
```

#### Chain Rule:
```
d/dx[f(g(x))] = f'(g(x)) × g'(x)
```

#### ML Applications:
- **Gradient calculation**: Finding steepest ascent/descent
- **Backpropagation**: Training neural networks
- **Optimization**: Minimizing loss functions

### 2.2 Partial Derivatives 🟢

**Definition**: Derivative with respect to one variable while keeping others constant.

#### Notation:
```
∂f/∂x = partial derivative of f with respect to x
```

#### Example:
```
f(x,y) = x² + 3xy + y²
∂f/∂x = 2x + 3y
∂f/∂y = 3x + 2y
```

#### Gradient Vector:
```
∇f = [∂f/∂x₁]
     [∂f/∂x₂]
     [  ⋮  ]
     [∂f/∂xₙ]
```

### 2.3 Optimization 🟡

#### Critical Points:
Points where ∇f = 0 (all partial derivatives are zero)

#### Types of Critical Points:
- **Global minimum**: Lowest point overall
- **Local minimum**: Lowest in neighborhood
- **Saddle point**: Neither maximum nor minimum

#### Hessian Matrix:
```
H = [∂²f/∂x₁²    ∂²f/∂x₁∂x₂  ⋯]
    [∂²f/∂x₂∂x₁  ∂²f/∂x₂²    ⋯]
    [     ⋮           ⋮       ⋱]
```

#### Second Derivative Test:
- **Positive definite Hessian**: Local minimum
- **Negative definite Hessian**: Local maximum
- **Indefinite Hessian**: Saddle point

### 2.4 Gradient Descent 🟡

**Algorithm**: Iteratively move in direction of steepest descent

```
θ(t+1) = θ(t) - α∇f(θ(t))
```

Where:
- θ: Parameters
- α: Learning rate
- ∇f: Gradient of cost function

#### Variants:
- **Batch Gradient Descent**: Use entire dataset
- **Stochastic Gradient Descent**: Use one sample at a time
- **Mini-batch Gradient Descent**: Use small batches

#### ML Applications:
- **Linear regression**: Minimizing mean squared error
- **Logistic regression**: Maximizing likelihood
- **Neural networks**: Backpropagation algorithm

---

## 3. Probability Theory

Probability provides the foundation for handling uncertainty in machine learning.

### 3.1 Basic Probability 🟢

#### Sample Space and Events:
- **Sample Space (Ω)**: Set of all possible outcomes
- **Event (A)**: Subset of sample space
- **Probability P(A)**: Number between 0 and 1

#### Axioms of Probability:
1. P(A) ≥ 0 for any event A
2. P(Ω) = 1
3. P(A ∪ B) = P(A) + P(B) if A and B are disjoint

#### Basic Rules:
```
P(A') = 1 - P(A)                    (complement)
P(A ∪ B) = P(A) + P(B) - P(A ∩ B)   (union)
```

### 3.2 Conditional Probability 🟢

**Definition**: Probability of A given that B has occurred

```
P(A|B) = P(A ∩ B) / P(B)
```

#### Independence:
Events A and B are independent if:
```
P(A|B) = P(A)  or  P(A ∩ B) = P(A)P(B)
```

#### Bayes' Theorem:
```
P(A|B) = P(B|A)P(A) / P(B)
```

Where:
- P(A|B): Posterior probability
- P(B|A): Likelihood
- P(A): Prior probability
- P(B): Marginal probability

#### ML Applications:
- **Naive Bayes classifier**
- **Bayesian inference**
- **Medical diagnosis**
- **Spam filtering**

### 3.3 Random Variables 🟡

**Definition**: Function that assigns numerical values to outcomes of random experiments.

#### Types:
- **Discrete**: Countable values (coin flips, dice)
- **Continuous**: Uncountable values (height, temperature)

#### Probability Mass Function (PMF):
For discrete random variable X:
```
P(X = x) = probability that X takes value x
```

#### Probability Density Function (PDF):
For continuous random variable X:
```
f(x) such that P(a ≤ X ≤ b) = ∫[a to b] f(x)dx
```

#### Cumulative Distribution Function (CDF):
```
F(x) = P(X ≤ x)
```

### 3.4 Important Distributions 🟡

#### Bernoulli Distribution:
```
X ~ Bernoulli(p)
P(X = 1) = p, P(X = 0) = 1-p
```

#### Binomial Distribution:
```
X ~ Binomial(n, p)
P(X = k) = C(n,k) × p^k × (1-p)^(n-k)
```

#### Normal (Gaussian) Distribution:
```
X ~ N(μ, σ²)
f(x) = (1/√(2πσ²)) × e^(-(x-μ)²/(2σ²))
```

#### Standard Normal Distribution:
```
Z ~ N(0, 1)
```

#### ML Applications:
- **Gaussian distribution**: Assumption in many algorithms
- **Bernoulli**: Binary classification
- **Multinomial**: Multi-class classification

### 3.5 Expectation and Variance 🟡

#### Expectation (Mean):
```
E[X] = Σ x × P(X = x)  (discrete)
E[X] = ∫ x × f(x)dx    (continuous)
```

#### Variance:
```
Var(X) = E[(X - E[X])²] = E[X²] - (E[X])²
```

#### Standard Deviation:
```
σ = √Var(X)
```

#### Properties:
```
E[aX + b] = aE[X] + b
Var(aX + b) = a²Var(X)
E[X + Y] = E[X] + E[Y]
Var(X + Y) = Var(X) + Var(Y)  (if X,Y independent)
```

---

## 4. Statistics for Machine Learning

Statistics provides tools for analyzing data and making inferences.

### 4.1 Descriptive Statistics 🟢

#### Measures of Central Tendency:
- **Mean**: Average value
- **Median**: Middle value when sorted
- **Mode**: Most frequent value

#### Measures of Spread:
- **Range**: Max - Min
- **Variance**: Average squared deviation from mean
- **Standard Deviation**: Square root of variance
- **Interquartile Range (IQR)**: Q3 - Q1

#### Distribution Shape:
- **Skewness**: Asymmetry of distribution
- **Kurtosis**: Tail heaviness

### 4.2 Inferential Statistics 🟡

#### Sampling:
- **Population**: Complete set of items
- **Sample**: Subset of population
- **Sampling bias**: Non-representative sample

#### Central Limit Theorem:
Sample means approach normal distribution as sample size increases.

#### Confidence Intervals:
Range likely to contain population parameter with given confidence level.

#### Hypothesis Testing:
- **Null hypothesis (H₀)**: No effect/difference
- **Alternative hypothesis (H₁)**: Effect exists
- **p-value**: Probability of observing data given H₀ is true
- **Type I error**: Rejecting true H₀ (false positive)
- **Type II error**: Accepting false H₀ (false negative)

### 4.3 Correlation and Regression 🟡

#### Correlation:
Measure of linear relationship between variables.

**Pearson Correlation Coefficient:**
```
r = Σ[(xi - x̄)(yi - ȳ)] / √[Σ(xi - x̄)²Σ(yi - ȳ)²]
```

Values: -1 (perfect negative) to +1 (perfect positive)

#### Simple Linear Regression:
```
y = β₀ + β₁x + ε
```

Where:
- β₀: Intercept
- β₁: Slope
- ε: Error term

#### Least Squares Estimation:
```
β₁ = Σ[(xi - x̄)(yi - ȳ)] / Σ(xi - x̄)²
β₀ = ȳ - β₁x̄
```

### 4.4 Information Theory 🟡

#### Entropy:
Measure of uncertainty/information content.

```
H(X) = -Σ P(xi) log₂ P(xi)
```

#### Cross-Entropy:
```
H(p,q) = -Σ p(xi) log q(xi)
```

#### Kullback-Leibler (KL) Divergence:
```
KL(p||q) = Σ p(xi) log(p(xi)/q(xi))
```

#### Mutual Information:
```
I(X;Y) = Σ P(x,y) log(P(x,y)/(P(x)P(y)))
```

#### ML Applications:
- **Decision trees**: Information gain
- **Neural networks**: Cross-entropy loss
- **Feature selection**: Mutual information

---

## 5. Mathematical Optimization

Optimization is central to training machine learning models.

### 5.1 Optimization Problems 🟡

#### General Form:
```
minimize f(x)
subject to: gi(x) ≤ 0, i = 1,...,m
           hj(x) = 0, j = 1,...,p
```

#### Types:
- **Unconstrained**: No constraints on variables
- **Constrained**: Variables must satisfy constraints
- **Convex**: Global minimum exists and is findable
- **Non-convex**: Multiple local minima may exist

### 5.2 Convex Optimization 🟡

#### Convex Function:
```
f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y)
```
for all x, y in domain and λ ∈ [0,1]

#### Properties:
- Local minimum is global minimum
- Gradient descent converges to global minimum
- Many ML problems are convex

#### Examples:
- Linear functions
- Quadratic functions (positive definite)
- Exponential functions
- Logarithmic functions (on positive domain)

### 5.3 Lagrange Multipliers 🔴

For constrained optimization:
```
L(x,λ) = f(x) + Σλigi(x)
```

#### Necessary Conditions (KKT):
```
∇f(x*) + Σλi∇gi(x*) = 0
gi(x*) ≤ 0
λi ≥ 0
λigi(x*) = 0
```

#### ML Applications:
- **Support Vector Machines**
- **Constrained optimization in neural networks**

---

## 6. Numerical Methods

Practical computation methods for implementing ML algorithms.

### 6.1 Root Finding 🟡

#### Newton's Method:
```
xn+1 = xn - f(xn)/f'(xn)
```

#### Bisection Method:
Repeatedly halve interval containing root.

### 6.2 Numerical Integration 🟡

#### Trapezoidal Rule:
```
∫[a to b] f(x)dx ≈ (b-a)/2n × Σ[f(xi) + f(xi+1)]
```

#### Monte Carlo Integration:
Use random sampling to estimate integrals.

### 6.3 Linear System Solving 🟡

#### Gaussian Elimination:
Systematic method to solve Ax = b.

#### LU Decomposition:
```
A = LU
```
Where L is lower triangular, U is upper triangular.

#### Iterative Methods:
- **Jacobi method**
- **Gauss-Seidel method**
- **Conjugate gradient**

---

## 7. Practical Applications in ML

### 7.1 Linear Regression Mathematics 🟢

#### Matrix Form:
```
y = Xβ + ε
```

#### Normal Equation:
```
β̂ = (XᵀX)⁻¹Xᵀy
```

#### Cost Function:
```
J(β) = (1/2m)||Xβ - y||²
```

#### Gradient:
```
∇J(β) = (1/m)Xᵀ(Xβ - y)
```

### 7.2 Logistic Regression Mathematics 🟡

#### Sigmoid Function:
```
σ(z) = 1/(1 + e⁻ᶻ)
```

#### Prediction:
```
P(y=1|x) = σ(βᵀx)
```

#### Log-Likelihood:
```
ℓ(β) = Σ[yi log σ(βᵀxi) + (1-yi) log(1-σ(βᵀxi))]
```

#### Gradient:
```
∇ℓ(β) = Σ(yi - σ(βᵀxi))xi
```

### 7.3 Neural Network Mathematics 🟡

#### Forward Propagation:
```
z⁽ˡ⁾ = W⁽ˡ⁾a⁽ˡ⁻¹⁾ + b⁽ˡ⁾
a⁽ˡ⁾ = σ(z⁽ˡ⁾)
```

#### Backpropagation:
```
δ⁽ˡ⁾ = (W⁽ˡ⁺¹⁾)ᵀδ⁽ˡ⁺¹⁾ ⊙ σ'(z⁽ˡ⁾)
```

#### Parameter Updates:
```
W⁽ˡ⁾ := W⁽ˡ⁾ - α × δ⁽ˡ⁺¹⁾(a⁽ˡ⁾)ᵀ
b⁽ˡ⁾ := b⁽ˡ⁾ - α × δ⁽ˡ⁺¹⁾
```

---

## 🎯 Key Mathematical Concepts Summary

### Essential for ML Success:
1. **Linear Algebra**: Matrix operations, eigenvalues, SVD
2. **Calculus**: Derivatives, gradients, optimization
3. **Probability**: Distributions, Bayes' theorem, expectation
4. **Statistics**: Hypothesis testing, correlation, regression
5. **Optimization**: Gradient descent, convex optimization

### Common Patterns:
- **Data as matrices**: Rows = samples, columns = features
- **Optimization**: Most ML = optimization problem
- **Probability**: Handle uncertainty and make predictions
- **Linear algebra**: Efficient computation with vectors/matrices

---

## 📚 Next Steps

Continue your journey with:
- **[Data Preprocessing](03_Data_Preprocessing.md)** - Applying math to prepare real data
- **[Supervised Learning](04_Supervised_Learning.md)** - See math in action with algorithms

---

## 🔢 Practice Problems

### Problem 1: Matrix Operations
Given matrices:
```
A = [1  2]    B = [5  6]
    [3  4]        [7  8]
```
Calculate: A + B, AB, Aᵀ, and det(A)

### Problem 2: Gradient Calculation
For f(x,y) = x² + 2xy + y², find:
- ∇f
- Critical points
- Hessian matrix

### Problem 3: Probability
If P(Disease) = 0.01 and a test has:
- P(Positive|Disease) = 0.95
- P(Positive|No Disease) = 0.05

Find P(Disease|Positive) using Bayes' theorem.

---

*Next: [Data Preprocessing →](03_Data_Preprocessing.md)*
