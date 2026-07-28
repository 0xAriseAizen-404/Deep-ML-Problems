# Generate Random Subsets of a Dataset (Medium, Machine Learning)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Generate Random Subsets of a Dataset](#learn-generate-random-subsets-of-a-dataset)
- [Solutions](#solutions)
  - [Custom Implementation](#custom-implementation)
  - [NumPy Implementation](#numpy-implementation)
- [Code Explanation](#code-explanation)

## Problem Statement

[Generate Random Subsets of a Dataset](https://www.deep-ml.com/problems/33)

Write a Python function to generate random subsets of a dataset for ensemble methods like **Bagging**.

- **With replacement:** subset size = original dataset size.
- **Without replacement:** subset size = half of the dataset.

Return a list of `(X_subset, y_subset)` tuples.

## Example

```python
Input:
X = np.array([
    [1, 2],
    [3, 4],
    [5, 6],
    [7, 8],
    [9, 10]
])

y = np.array([1, 2, 3, 4, 5])

n_subsets = 3
replacements = False

Output:
[
    ([[3, 4], [9, 10]], [2, 5]),
    ([[7, 8], [3, 4]], [4, 2]),
    ([[3, 4], [1, 2]], [2, 1])
]
```

## Learn: Generate Random Subsets of a Dataset

Random subsets are widely used in **Bagging** and **Random Forests**.

- **With replacement:** samples may repeat (bootstrap sampling).
- **Without replacement:** each sample appears at most once per subset.

NumPy's `np.random.choice()` supports both sampling methods.

## Solutions

### Custom Implementation

```python
import numpy as np

def get_random_subsets(X, y, n_subsets, replacements=True):
    n = len(X)
    subset_size = n if replacements else n // 2

    subsets = []

    for _ in range(n_subsets):
        idx = np.random.choice(n, subset_size, replace=replacements)
        subsets.append((X[idx].tolist(), y[idx].tolist()))

    return subsets
```

### NumPy Implementation

```python
import numpy as np

def get_random_subsets(X, y, n_subsets, replacements=True):
    n, _ = X.shape
    subset_size = n if replacements else n // 2

    idx = np.array([
        np.random.choice(n, subset_size, replace=replacements)
        for _ in range(n_subsets)
    ])

    return [
        (X[idx[i]].tolist(), y[idx[i]].tolist())
        for i in range(n_subsets)
    ]
```

## Code Explanation

- Determine the subset size based on the `replacements` flag.
- Randomly generate sample indices using `np.random.choice()`.
- Select rows from `X` and `y` using those indices.
- Convert the subsets to Python lists before returning.

This is the standard approach used to create bootstrap samples for ensemble learning algorithms.
