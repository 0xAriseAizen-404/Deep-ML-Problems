# Random Shuffle of Dataset (Easy, Machine Learning)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Dataset Shuffling](#learn-dataset-shuffling)
- [Solutions](#solutions)
  - [NumPy Implementation](#numpy-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

---

## Problem Statement

[Random Shuffle of Dataset](https://www.deep-ml.com/problems/29)

Write a Python function to randomly shuffle the samples in **X** and **y** while maintaining the correspondence between features and labels. The function should optionally accept a random seed for reproducibility.

---

## Example

```python
Input:

X = np.array([[1, 2],
              [3, 4],
              [5, 6],
              [7, 8]])

y = np.array([1, 2, 3, 4])

Output:

(array([[5, 6],
        [1, 2],
        [7, 8],
        [3, 4]]),

 array([3, 1, 4, 2]))
```

**Reasoning**

A random permutation of indices is generated and applied to both **X** and **y**, preserving the feature-label mapping.

---

## Learn: Dataset Shuffling

Shuffling helps randomize the order of training samples before model training.

**Steps**

1. Generate shuffled indices.
2. Reorder **X** using those indices.
3. Reorder **y** using the same indices.

This ensures the correspondence between samples and labels is preserved.

---

## Solutions

### NumPy Implementation

```python
import numpy as np


def shuffle_data(X, y, seed=None):

    np.random.seed(seed)

    indices = np.random.permutation(len(X))

    return X[indices], y[indices]


# Example Usage
X = np.array([[1, 2],
              [3, 4],
              [5, 6],
              [7, 8]])

y = np.array([1, 2, 3, 4])

print(shuffle_data(X, y, seed=42))
```

---

## Code Explanation

- Set the random seed (optional) for reproducibility.

```python
np.random.seed(seed)
```

- Generate a random permutation of indices.

```python
indices = np.random.permutation(len(X))
```

- Shuffle both arrays using the same indices.

```python
X[indices], y[indices]
```

---

## Time & Space Complexity

| Complexity | Value |
|------------|-------|
| Time | **O(n)** |
| Space | **O(n)** |

Where **n** is the number of samples.