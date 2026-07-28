# Divide Dataset Based on Feature Threshold (Medium, Machine Learning)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Divide Dataset Based on Feature Threshold](#learn-divide-dataset-based-on-feature-threshold)
- [Solutions](#solutions)
  - [Custom Implementation](#custom-implementation)
  - [NumPy Implementation](#numpy-implementation)
- [Code Explanation](#code-explanation)

## Problem Statement

[Divide Dataset Based on Feature Threshold](https://www.deep-ml.com/problems/31)

Write a Python function to divide a dataset based on whether the value of a specified feature is greater than or equal to a given threshold. Return two subsets:

- Samples with `feature >= threshold`
- Samples with `feature < threshold`

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

feature_i = 0
threshold = 5

Output:
[
    array([[ 5,  6],
           [ 7,  8],
           [ 9, 10]]),

    array([[1, 2],
           [3, 4]])
]
```

## Learn: Divide Dataset Based on Feature Threshold

This type of split is commonly used in **Decision Trees**.

For a feature index `i` and threshold `t`:

```text
Left  = X[:, i] >= t
Right = X[:, i] <  t
```

A boolean mask makes this operation simple and efficient in NumPy.

## Solutions

### Custom Implementation

```python
import numpy as np

def divide_on_feature(X, feature_i, threshold):
    left, right = [], []

    for row in X:
        if row[feature_i] >= threshold:
            left.append(row)
        else:
            right.append(row)

    return [np.array(left), np.array(right)]
```

### NumPy Implementation

```python
import numpy as np

def divide_on_feature(X, feature_i, threshold):
    mask = X[:, feature_i] >= threshold
    return [X[mask], X[~mask]]
```

## Code Explanation

- Create a boolean mask using the selected feature.
- `True` selects rows where the feature is **greater than or equal** to the threshold.
- `False` selects the remaining rows.
- Return both subsets while preserving the original row order.

The NumPy approach is shorter, faster, and preferred for real-world ML preprocessing.
