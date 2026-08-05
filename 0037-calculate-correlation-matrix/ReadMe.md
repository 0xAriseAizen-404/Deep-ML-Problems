# Calculate Correlation Matrix (Medium, Linear Algebra)

## Table of Contents

- Problem Statement
- Example
- Learn: Understanding Correlation Matrix
- Solution
- Code Explanation
- Time & Space Complexity

---

## Problem Statement

### [Calculate Correlation Matrix](https://www.deep-ml.com/problems/37)

Write a Python function that computes the **correlation matrix** for a given dataset.

The function should:

- Accept a 2-D NumPy array `X`.
- Optionally accept another 2-D NumPy array `Y`.
- If `Y` is not provided, compute the correlation matrix of `X` with itself.
- Return the resulting correlation matrix as a 2-D NumPy array.

Each entry of the matrix represents the Pearson correlation coefficient between two features.

---

## Example

### Input

```python
X = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])

output = calculate_correlation_matrix(X)

print(output)
```

### Output

```python
[
    [1., 1.],
    [1., 1.]
]
```

### Reasoning

The two columns increase proportionally.

```text
Feature 1 : 1 3 5

Feature 2 : 2 4 6
```

Since one feature is a perfect linear transformation of the other, their Pearson correlation coefficient is

$$
1
$$

Therefore, every feature is perfectly correlated with itself and with the other feature.

---

## Learn: Understanding Correlation Matrix

### What is it?

A **Correlation Matrix** is a square matrix that measures the strength and direction of the **linear relationship** between pairs of variables.

Each element of the matrix is a **correlation coefficient**, whose value lies between

$$
-1
\le
r
\le
1
$$

where

- **1** indicates a perfect positive linear relationship.
- **-1** indicates a perfect negative linear relationship.
- **0** indicates no linear relationship.

Unlike covariance, correlation is **scale-independent**, making it easier to compare relationships between different features.

---

### Mathematical Definition

The Pearson correlation coefficient between two variables $X$ and $Y$ is

$$
\mathrm{corr}(X,Y) =
\frac{\mathrm{cov}(X,Y)}
{\sigma_X \sigma_Y}
$$

where

- $\mathrm{cov}(X,Y)$ is the covariance.
- $\sigma_X$ is the standard deviation of $X$.
- $\sigma_Y$ is the standard deviation of $Y$.

The covariance is

$$
\mathrm{cov}(X,Y) =
\frac{
\sum_{i=1}^{n}
(x_i-\bar{x})
(y_i-\bar{y})
}
{n-1}
$$

The correlation matrix for a dataset with $d$ features is

$$
R =
\begin{bmatrix}
r_{11} & r_{12} & \cdots & r_{1d}\\
r_{21} & r_{22} & \cdots & r_{2d}\\
\vdots & \vdots & \ddots & \vdots\\
r_{d1} & r_{d2} & \cdots & r_{dd}
\end{bmatrix}
$$

where

$$
r_{ij} =
\mathrm{corr}(X_i,X_j)
$$

---

### How Correlation is Calculated

The algorithm follows these steps:

1. Compute the mean of every feature.
2. Standardize each feature by subtracting its mean and dividing by its standard deviation.
3. Compute the dot product of the standardized features.
4. Divide by

$$
n-1
$$

to obtain the correlation coefficients.

For example,

Feature A

```text
1 2 3 4
```

Feature B

```text
2 4 6 8
```

Although the values differ, they change proportionally.

Therefore,

$$
\mathrm{corr}(A,B)=1
$$

---

### Standardization

Before computing correlation, every feature is standardized.

For a feature

$$
x
$$

the standardized version is

$$
z =
\frac{x-\mu}{\sigma}
$$

where

- $\mu$ is the feature mean.
- $\sigma$ is the sample standard deviation.

After standardization,

- Mean becomes

$$
0
$$

- Standard deviation becomes

$$
1
$$

This removes differences in scale and allows direct comparison between variables.

---

### Correlation vs Covariance

| Covariance               | Correlation                              |
| ------------------------ | ---------------------------------------- |
| Depends on feature scale | Independent of feature scale             |
| Can take any real value  | Always lies between -1 and 1             |
| Harder to interpret      | Easy to interpret                        |
| Measures joint variation | Measures strength of linear relationship |

For example,

Two variables measured in

- kilograms
- grams

have different covariance values but exactly the same correlation.

---

### Characteristics / Key Points

- Produces a square symmetric matrix.
- Every diagonal element equals **1**.
- Values always lie between **-1** and **1**.
- Measures only linear relationships.
- Independent of feature units.
- Based on standardized variables.
- Symmetric when computing `corr(X, X)`.

---

### Why is it used? / Applications

Correlation matrices are widely used during exploratory data analysis and feature engineering.

Applications include

- Feature selection
- Multicollinearity detection
- Exploratory Data Analysis (EDA)
- Principal Component Analysis (PCA)
- Financial asset analysis
- Scientific data analysis
- Recommendation systems
- Biological data analysis

Highly correlated features often contain redundant information and may be removed before training certain machine learning models.

---

> 💡 **Important Note**
>
> Correlation measures **linear association**, not causation. Two variables can have a correlation close to **1** even if one does not cause the other. Likewise, variables with a strong nonlinear relationship may have a correlation close to **0**.

---

## Solution

### Custom Implementation

```python
import numpy as np

def calculate_correlation_matrix(X, Y=None):
    X = np.asarray(X).T

    if Y is None:
        Y = X
    else:
        Y = np.asarray(Y).T

    corr_matrix = []

    for f1 in X:
        row = []

        for f2 in Y:
            cov = np.sum(
                (f1 - f1.mean()) *
                (f2 - f2.mean())
            ) / (len(f1) - 1)

            corr = cov / (
                f1.std(ddof=1) *
                f2.std(ddof=1)
            )

            row.append(corr)

        corr_matrix.append(row)

    return np.asarray(corr_matrix)
```

### Optimized NumPy Implementation

```python
import numpy as np

def calculate_correlation_matrix(X, Y=None):
    X = np.asarray(X)

    if Y is None:
        Y = X
    else:
        Y = np.asarray(Y)

    X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    Y = (Y - Y.mean(axis=0)) / Y.std(axis=0, ddof=1)

    return (X.T @ Y) / (X.shape[0] - 1)
```

---

## Code Explanation

### Step 1

Convert the input into NumPy arrays.

```python
X = np.asarray(X)
```

If another dataset is supplied, convert it as well.

---

### Step 2

Standardize every feature.

```python
(X - X.mean(axis=0))
/
X.std(axis=0, ddof=1)
```

Each feature is centered around zero and scaled to unit variance.

---

### Step 3

Repeat the same standardization for `Y`.

Both datasets must have standardized features before correlation is computed.

---

### Step 4

Compute the matrix multiplication.

```python
X.T @ Y
```

This simultaneously computes the covariance between every pair of standardized features.

---

### Step 5

Normalize by the number of samples.

```python
(X.T @ Y) / (X.shape[0] - 1)
```

Dividing by

$$
n-1
$$

converts the standardized covariance into the Pearson correlation coefficient.

---

### Step 6

Return the correlation matrix.

The output matrix contains the correlation between every feature pair.

---

## Time & Space Complexity

| Complexity | Value          |
| ---------- | -------------- |
| Time       | **O(nd²)**     |
| Space      | **O(nd + d²)** |

where

- $n$ is the number of samples.
- $d$ is the number of features.

Standardization requires processing every feature once, while the matrix multiplication computes the correlation between every pair of features.
