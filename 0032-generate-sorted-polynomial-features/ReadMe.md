# Generate Sorted Polynomial Features (Medium, Machine Learning)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Generate Sorted Polynomial Features](#learn-generate-sorted-polynomial-features)
- [Solutions](#solutions)
  - [Custom Implementation](#custom-implementation)
  - [Using itertools](#using-itertools)
- [Code Explanation](#code-explanation)

## Problem Statement

[Generate Sorted Polynomial Features](https://www.deep-ml.com/problems/32)

Write a Python function that generates all polynomial feature combinations up to a given degree, then sorts the features of each sample in ascending order.

## Example

```python
Input:
X = np.array([
    [2, 3],
    [3, 4],
    [5, 6]
])

degree = 2

Output:
[
    [1., 2., 3., 4., 6., 9.],
    [1., 3., 4., 9., 12., 16.],
    [1., 5., 6., 25., 30., 36.]
]
```

## Learn: Generate Sorted Polynomial Features

Polynomial features help linear models learn **non-linear relationships**.

For degree = 2 and features `(x₁, x₂)`, the generated features are:

```text
1, x₁, x₂, x₁², x₁x₂, x₂²
```

Finally, each row is sorted in ascending order.

## Solutions

### Custom Implementation

```python
import numpy as np
from itertools import combinations_with_replacement

def polynomial_features(X, degree):
    n_samples, n_features = X.shape

    combs = [
        c
        for d in range(degree + 1)
        for c in combinations_with_replacement(range(n_features), d)
    ]

    X_poly = np.empty((n_samples, len(combs)))

    for i, idx in enumerate(combs):
        X_poly[:, i] = (
            1 if len(idx) == 0
            else np.prod(X[:, idx], axis=1)
        )

    return np.sort(X_poly, axis=1)
```

### Using itertools

```python
# Same implementation as above.
# itertools.combinations_with_replacement()
# generates all valid polynomial feature combinations.
```

## Code Explanation

- Generate all feature index combinations up to the given degree.
- Compute the product for each combination.
- Include the constant feature `1`.
- Sort each row using `np.sort(axis=1)`.
- Return the sorted polynomial feature matrix.

This approach efficiently creates polynomial features and is suitable for feature engineering in machine learning.