# Calculate Covariance Matrix (Easy, Statistics)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Covariance Matrix](#learn-covariance-matrix)
- [Solutions](#solutions)
  - [Custom Implementation](#custom-implementation)
  - [NumPy Implementation](#numpy-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

## Problem Statement

### [Calculate Covariance Matrix](https://www.deep-ml.com/problems/10)

Write a Python function that computes the **covariance matrix** for a given set of feature vectors.

Each inner list represents one feature and contains all of its observations. The function should return an **n × n covariance matrix**, where each element represents the covariance between a pair of features.

---

## Example

### Input

```python
vectors = [
    [1, 2, 3],
    [4, 5, 6]
]
```

### Output

```python
[
    [1.0, 1.0],
    [1.0, 1.0]
]
```

### Reasoning

Feature means are

$$
\bar{X}_1 = \frac{1+2+3}{3} = 2
$$

$$
\bar{X}_2 = \frac{4+5+6}{3} = 5
$$

The covariance between the two features is

$$
\operatorname{cov}(X_1,X_2)=\frac{(-1)(-1)+0+1}{2}=1
$$

Since both features vary together perfectly, the covariance matrix becomes

$$
\begin{bmatrix}
1 & 1\\
1 & 1
\end{bmatrix}
$$

---

## Learn: Covariance Matrix

### What is Covariance?

Covariance measures how two variables change together.

- **Positive covariance** means both variables tend to increase or decrease together.
- **Negative covariance** means one variable tends to increase while the other decreases.
- **Zero covariance** indicates no linear relationship between the variables.

Unlike correlation, covariance depends on the units of measurement and is therefore not bounded between -1 and 1.

---

### Covariance Formula

Given two variables

$$
X=(X_1,X_2,\dots,X_m)
$$

and

$$
Y=(Y_1,Y_2,\dots,Y_m)
$$

their sample covariance is

$$
\operatorname{cov}(X,Y)=\frac{\sum_{k=1}^{m}(X_k-\bar{X})(Y_k-\bar{Y})}{m-1}
$$

where

- $X_k$ and $Y_k$ are observations.
- $\bar{X}$ and $\bar{Y}$ are the sample means.
- $m$ is the number of observations.

Using $m-1$ in the denominator gives the **sample covariance**, which is the standard estimator used in statistics and by NumPy's `cov()`.

---

### What is a Covariance Matrix?

For a dataset containing $n$ features, the covariance matrix is

$$
\Sigma=
\begin{bmatrix}
\operatorname{cov}(X_1,X_1) & \operatorname{cov}(X_1,X_2) & \cdots & \operatorname{cov}(X_1,X_n)\\
\operatorname{cov}(X_2,X_1) & \operatorname{cov}(X_2,X_2) & \cdots & \operatorname{cov}(X_2,X_n)\\
\vdots & \vdots & \ddots & \vdots\\
\operatorname{cov}(X_n,X_1) & \operatorname{cov}(X_n,X_2) & \cdots & \operatorname{cov}(X_n,X_n)
\end{bmatrix}
$$

Each element describes the relationship between two features.

The diagonal entries are simply the **variances** of each feature because

$$
\operatorname{cov}(X,X)=\operatorname{Var}(X)
$$

---

### Step-by-Step Algorithm

To compute the covariance matrix:

1. Compute the mean of every feature.
2. Subtract the mean from each observation (mean centering).
3. Compute the covariance between every pair of features.
4. Store the computed covariance in the matrix.
5. Since covariance is symmetric,

$$
\operatorname{cov}(X,Y)=\operatorname{cov}(Y,X)
$$

only half the matrix needs to be computed explicitly.

---

### Example

Consider two features

$$
X=[1,2,3]
$$

$$
Y=[4,5,6]
$$

Their means are

$$
\bar{X}=2
$$

$$
\bar{Y}=5
$$

The covariance is

$$
\operatorname{cov}(X,Y)=\frac{(1-2)(4-5)+(2-2)(5-5)+(3-2)(6-5)}{3-1}=1
$$

Similarly,

$$
\operatorname{Var}(X)=1
$$

and

$$
\operatorname{Var}(Y)=1
$$

Therefore,

$$
\Sigma=
\begin{bmatrix}
1 & 1\\
1 & 1
\end{bmatrix}
$$

---

### Characteristics / Key Points

- Measures linear relationships between variables.
- Produces a symmetric matrix.

$$
\Sigma=\Sigma^T
$$

- Diagonal entries represent variances.
- Off-diagonal entries represent covariances.
- Positive covariance indicates variables move together.
- Negative covariance indicates variables move in opposite directions.
- Zero covariance implies no linear dependence (though nonlinear relationships may still exist).

---

### Covariance vs Correlation

| Covariance               | Correlation                              |
| ------------------------ | ---------------------------------------- |
| Depends on units         | Unitless                                 |
| Unbounded                | Always between -1 and 1                  |
| Measures joint variation | Measures strength of linear relationship |

Correlation is simply the normalized version of covariance.

$$
\rho_{XY}=\frac{\operatorname{cov}(X,Y)}{\sigma_X\sigma_Y}
$$

---

### Why is it Used?

Covariance matrices are essential in statistics, machine learning, and data analysis.

Common applications include:

- Principal Component Analysis (PCA)
- Multivariate Gaussian distributions
- Feature engineering
- Portfolio optimization in finance
- Dimensionality reduction
- Kalman Filters
- Signal processing
- Statistical modeling

In PCA, the covariance matrix identifies directions with the greatest variance, allowing high-dimensional data to be projected into fewer dimensions while preserving most of the information.

---

> 💡 **Important Note**
>
> A large covariance value does **not** necessarily indicate a strong relationship because covariance depends on the scale of the variables. When comparing relationships between features measured in different units, **correlation** is generally more appropriate since it is normalized to lie between -1 and 1.

---

## Solutions

### Custom Implementation

```python
def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:

    def cov(x, y):
        mean_x = sum(x) / len(x)
        mean_y = sum(y) / len(y)

        return sum(
            (x[k] - mean_x) * (y[k] - mean_y)
            for k in range(len(x))
        ) / (len(x) - 1)

    n = len(vectors)

    result = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i, n):
            value = cov(vectors[i], vectors[j])
            result[i][j] = value
            result[j][i] = value

    return result
```

### NumPy Implementation

```python
import numpy as np

def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
    return np.cov(vectors).tolist()
```

---

## Code Explanation

### 1. Compute the Mean of Each Feature

For every feature, calculate its average value.

$$
\bar{X}=\frac{1}{m}\sum_{i=1}^{m}X_i
$$

These means are used to center the data.

---

### 2. Compute Pairwise Covariance

For each pair of features,

```python
cov(x, y)
```

computes

$$
\operatorname{cov}(X,Y)=\frac{\sum (X-\bar{X})(Y-\bar{Y})}{m-1}
$$

This measures how the two variables vary together.

---

### 3. Fill the Covariance Matrix

```python
for i in range(n):
    for j in range(i, n):
```

Only the upper triangular part is computed because

$$
\operatorname{cov}(X,Y)=\operatorname{cov}(Y,X)
$$

The symmetric value is copied to the lower triangular part.

---

### 4. Return the Matrix

The completed covariance matrix is returned as a list of lists.

The NumPy implementation achieves the same result using

```python
np.cov(vectors)
```

which internally performs mean centering and covariance computation efficiently.

---

## Time & Space Complexity

Let

- $n$ = number of features.
- $m$ = number of observations per feature.

Each covariance calculation processes all observations, and there are approximately $n^2$ feature pairs.

| Complexity | Value         |
| ---------- | ------------- |
| Time       | **O(n² × m)** |
| Space      | **O(n²)**     |

The covariance matrix itself requires $n \times n$ storage.
