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

---

## Problem Statement

[Calculate Covariance Matrix](https://www.deep-ml.com/problems/10)

Write a Python function to calculate the **covariance matrix** for a given set of feature vectors.

Each inner list represents a **feature** and its observations.

---

## Example

```python
Input:
vectors = [
    [1, 2, 3],
    [4, 5, 6]
]

Output:
[
    [1.0, 1.0],
    [1.0, 1.0]
]
```

**Reasoning**

Both features increase together, so their covariance is **1.0**.

---

## Learn: Covariance Matrix

Covariance measures **how two features vary together**.

- Positive covariance → Both increase together.
- Negative covariance → One increases while the other decreases.
- Zero covariance → No linear relationship.

Covariance Formula:

```text
cov(X, Y) = Σ[(Xi - X̄)(Yi - Ȳ)] / (n - 1)
```

A covariance matrix stores the covariance between **every pair of features**.

```text
[
 [cov(X₁,X₁), cov(X₁,X₂)],
 [cov(X₂,X₁), cov(X₂,X₂)]
]
```

The matrix is always **square** and **symmetric**.

---

## Solutions

### Custom Implementation

```python
def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:

    def covariance(x, y):
        mean_x = sum(x) / len(x)
        mean_y = sum(y) / len(y)

        return sum(
            (x[i] - mean_x) * (y[i] - mean_y)
            for i in range(len(x))
        ) / (len(x) - 1)

    n = len(vectors)
    result = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i, n):
            value = covariance(vectors[i], vectors[j])
            result[i][j] = value
            result[j][i] = value

    return result


# Example Usage
vectors = [
    [1, 2, 3],
    [4, 5, 6]
]

print(calculate_covariance_matrix(vectors))
```

---

### NumPy Implementation

```python
import numpy as np


def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
    return np.cov(vectors).tolist()


# Example Usage
print(calculate_covariance_matrix(vectors))
```

---

## Code Explanation

- Compute the mean of each feature.
- Calculate covariance for every pair of features.
- Since covariance is symmetric:

```python
cov(X, Y) == cov(Y, X)
```

only the upper half is computed, and the lower half is filled with the same values.

NumPy's `np.cov()` performs all these calculations internally.

---

## Time & Space Complexity

### Custom Implementation

| Complexity | Value |
|------------|-------|
| Time | **O(f² × n)** |
| Space | **O(f²)** |

Where:

- **f** = number of features
- **n** = number of observations per feature

### NumPy Implementation

- **Time:** Optimized (implemented in C)
- **Space:** **O(f²)**