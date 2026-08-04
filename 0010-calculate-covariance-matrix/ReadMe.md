# Calculate Covariance Matrix (Easy, Statistics)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Covariance Matrix](#learn-covariance-matrix)
- [Things to Note](#things-to-note)
- [Solutions](#solutions)
  - [NumPy Implementation](#numpy-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

---

# Problem Statement

[Calculate Covariance Matrix](https://www.deep-ml.com/problems/10)

Write a Python function `calculate_covariance_matrix(vectors)` that computes the **covariance matrix** for a given collection of feature vectors.

Each inner list represents a **feature**, and each element inside that list represents one observation of that feature.

The function should return the covariance matrix as a nested Python list.

---

# Example

```python
vectors = [
    [1, 2, 3],
    [4, 5, 6]
]

print(calculate_covariance_matrix(vectors))
```

### Output

```text
[
    [1.0, 1.0],
    [1.0, 1.0]
]
```

### Explanation

Both features increase together.

Their covariance is positive:

$$
\operatorname{Cov}(X,Y)=1
$$

Since each feature is perfectly linearly related to the other,

the covariance matrix becomes

$$
\begin{bmatrix}
1 & 1\\
1 & 1
\end{bmatrix}
$$

---

# Learn: Covariance Matrix

## What is Covariance?

Covariance measures how **two variables change together**.

- Positive covariance → Both variables increase or decrease together.
- Negative covariance → One increases while the other decreases.
- Zero covariance → No linear relationship.

Unlike correlation, covariance does **not** have a fixed range.

Its value depends on the scale of the data.

---

## Covariance Formula

Suppose we have two variables

$$
X=(x_1,x_2,\ldots,x_n)
$$

and

$$
Y=(y_1,y_2,\ldots,y_n)
$$

Their sample covariance is

$$
\operatorname{Cov}(X,Y)
=
\frac{1}{n-1}
\sum_{i=1}^{n}
(x_i-\bar{x})(y_i-\bar{y})
$$

where

- $\bar{x}$ is the mean of $X$.
- $\bar{y}$ is the mean of $Y$.
- $n$ is the number of observations.

---

## What is a Covariance Matrix?

When there are multiple features,

we compute the covariance between **every pair of features**.

If there are

$$
m
$$

features,

the covariance matrix has size

$$
m \times m
$$

For three features,

$$
\mathbf{C}
=
\begin{bmatrix}
\operatorname{Cov}(X_1,X_1) &
\operatorname{Cov}(X_1,X_2) &
\operatorname{Cov}(X_1,X_3)
\\
\operatorname{Cov}(X_2,X_1) &
\operatorname{Cov}(X_2,X_2) &
\operatorname{Cov}(X_2,X_3)
\\
\operatorname{Cov}(X_3,X_1) &
\operatorname{Cov}(X_3,X_2) &
\operatorname{Cov}(X_3,X_3)
\end{bmatrix}
$$

---

## Properties of a Covariance Matrix

A covariance matrix always has the following properties:

- It is a **square matrix**.
- It is **symmetric**.

$$
\operatorname{Cov}(X,Y)
=
\operatorname{Cov}(Y,X)
$$

- Every diagonal element represents the variance of a feature.

$$
\operatorname{Cov}(X,X)
=
\operatorname{Var}(X)
$$

---

## Example

Consider

$$
X=[1,2,3]
$$

and

$$
Y=[4,5,6]
$$

Mean values are

$$
\bar{x}=2
$$

$$
\bar{y}=5
$$

The covariance is

$$
\frac{
(-1)(-1)+0(0)+1(1)
}{2}
=
1
$$

Therefore,

$$
\mathbf{C}
=
\begin{bmatrix}
1&1\\
1&1
\end{bmatrix}
$$

---

## Applications

Covariance matrices are widely used in Machine Learning and Statistics.

Some common applications include:

- Principal Component Analysis (PCA)
- Multivariate statistics
- Feature analysis
- Portfolio optimization
- Gaussian distributions
- Data preprocessing

---

# Things to Note

- The covariance matrix compares **every feature with every other feature**.
- The diagonal entries are the variances of each feature.
- The covariance matrix is always symmetric.
- Positive covariance indicates features move together.
- Negative covariance indicates features move in opposite directions.
- NumPy's `np.cov()` computes the **sample covariance**, dividing by $n-1$.

---

# Solutions

## NumPy Implementation

```python
import numpy as np

def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
    return np.cov(vectors).tolist()
```

---

# Code Explanation

## Step 1: Convert the Input

```python
np.cov(vectors)
```

The input list is interpreted as a collection of feature vectors.

Each row represents one feature,

and each column represents one observation.

---

## Step 2: Compute Feature Means

Internally, NumPy computes the mean of every feature.

For a feature

$$
X
$$

the mean is

$$
\bar{x}
=
\frac1n
\sum_{i=1}^{n}x_i
$$

---

## Step 3: Compute Covariance

For every pair of features,

NumPy computes

$$
\operatorname{Cov}(X,Y)
=
\frac{1}{n-1}
\sum_{i=1}^{n}
(x_i-\bar{x})(y_i-\bar{y})
$$

This is repeated for every pair of features.

---

## Step 4: Build the Covariance Matrix

The computed covariance values are arranged into a symmetric matrix.

The diagonal entries become

$$
\operatorname{Var}(X)
=
\operatorname{Cov}(X,X)
$$

while the off-diagonal entries represent the covariance between different features.

---

## Step 5: Convert Back to a Python List

```python
.tolist()
```

NumPy returns a NumPy array.

Calling `.tolist()` converts it into a nested Python list.

---

# Time & Space Complexity

Assume there are

- $m$ features.
- $n$ observations per feature.

| Complexity | Value |
|------------|-------|
| Time | **O(m² × n)** |
| Space | **O(m²)** |

The covariance between every pair of features is computed, producing an $m \times m$ covariance matrix.