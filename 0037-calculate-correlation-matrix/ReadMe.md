# Calculate Correlation Matrix (Medium, Linear Algebra)

## Problem Statement

[Calculate Correlation Matrix](https://www.deep-ml.com/problems/37)

Write a Python function to calculate the **correlation matrix** for a given dataset. The function should take a 2D NumPy array `X` and an optional 2D NumPy array `Y`. If `Y` is not provided, compute the correlation matrix of `X` with itself. Return the resulting correlation matrix as a 2D NumPy array.

---

## Example

```python
X = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])

print(calculate_correlation_matrix(X))
```

**Output**

```python
[[1. 1.]
 [1. 1.]]
```

**Explanation**

The two features increase together perfectly, so their correlation coefficient is **1**, indicating a perfect positive linear relationship.

---

# Learn About the Topic

## Understanding Correlation Matrix

A **correlation matrix** is a square matrix that measures the linear relationship between every pair of features in a dataset. Each entry contains the **Pearson correlation coefficient**, which ranges from **-1 to 1**.

- **1** → Perfect positive linear relationship.
- **0** → No linear relationship.
- **-1** → Perfect negative linear relationship.

The diagonal elements are always **1** because every feature is perfectly correlated with itself.

### Pearson Correlation Coefficient

The correlation between two variables \(X\) and \(Y\) is defined as

\[
\text{corr}(X,Y)=\frac{\text{cov}(X,Y)}{\sigma_X\sigma_Y}
\]

where

- \(\text{cov}(X,Y)\) is the covariance between the variables.
- \(\sigma_X\) and \(\sigma_Y\) are their standard deviations.

Since covariance depends on the scale of the variables, dividing by the standard deviations normalizes the value to the range **[-1, 1]**.

### Computing a Correlation Matrix

Given a dataset with multiple features:

1. Compute the mean of each feature.
2. Standardize each feature by subtracting its mean and dividing by its standard deviation.
3. Compute the covariance of the standardized features.
4. The resulting matrix is the correlation matrix.

For standardized data \(Z\),

\[
R=\frac{Z^T Z}{n-1}
\]

where:

- \(Z\) is the standardized data matrix.
- \(n\) is the number of samples.
- \(R\) is the correlation matrix.

### Applications

Correlation matrices are widely used in machine learning and statistics for:

- Feature selection
- Detecting highly correlated (redundant) features
- Exploratory Data Analysis (EDA)
- Multicollinearity detection before regression
- Principal Component Analysis (PCA)

### Important Notes

- Correlation measures **only linear relationships**. Two variables may have a strong nonlinear relationship while having a correlation close to zero.
- A high correlation does **not** imply causation.
- The correlation matrix is always **symmetric**, meaning:

\[
\text{corr}(X,Y)=\text{corr}(Y,X)
\]

- Every diagonal entry is exactly **1**.

> **Note:** Standardizing the features before computing covariance is an efficient way to obtain the correlation matrix, which is exactly what many numerical libraries (including NumPy implementations) do internally.

---

# Solution

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

# Code Explanation

1. Convert the inputs to NumPy arrays.
2. If `Y` is not provided, compute the correlation of `X` with itself.
3. Standardize every feature using its mean and sample standard deviation (`ddof=1`).
4. Multiply the transpose of the standardized `X` with the standardized `Y`.
5. Divide by `(n - 1)` to obtain the Pearson correlation matrix.
6. Return the resulting matrix.