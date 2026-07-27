# Convert Vector to Diagonal Matrix (Easy, Linear Algebra)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Diagonal Matrix](#learn-diagonal-matrix)
- [Solutions](#solutions)
  - [Custom Implementation](#custom-implementation)
  - [NumPy Implementation](#numpy-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

---

## Problem Statement

[Convert Vector to Diagonal Matrix](https://www.deep-ml.com/problems/35)

Write a Python function to convert a **1D NumPy array** into a **diagonal matrix**, where the elements of the input vector form the main diagonal.

---

## Example

```python
Input:

x = np.array([1, 2, 3])

Output:

[[1. 0. 0.]
 [0. 2. 0.]
 [0. 0. 3.]]
```

**Reasoning**

Each element of the input vector is placed on the **main diagonal**, while all other elements are **0**.

---

## Learn: Diagonal Matrix

A **diagonal matrix** is a square matrix where all **non-diagonal elements are zero**.

Example:

```text
[1, 2, 3]

↓

[[1, 0, 0],
 [0, 2, 0],
 [0, 0, 3]]
```

Diagonal matrices are commonly used in linear algebra because they simplify many matrix operations.

---

## Solutions

### Custom Implementation

```python
import numpy as np


def make_diagonal(x):

    size = len(x)

    matrix = [[0 for _ in range(size)] for _ in range(size)]

    for i in range(size):
        matrix[i][i] = x[i]

    return np.array(matrix)


# Example Usage
x = np.array([1, 2, 3])

print(make_diagonal(x))
```

### NumPy Implementation

```python
import numpy as np


def make_diagonal(x):
    return np.diag(x)
```

---

## Code Explanation

- Create a square matrix filled with zeros.
- Place each vector element on the main diagonal.

```python
matrix[i][i] = x[i]
```

- Convert the result into a NumPy array.

---

## Time & Space Complexity

| Complexity | Value |
|------------|-------|
| Time | **O(n²)** |
| Space | **O(n²)** |

Where **n** is the length of the input vector.