# Transformation Matrix from Basis B to C (Easy, Linear Algebra)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Transformation Matrix](#learn-transformation-matrix)
- [Solutions](#solutions)
  - [Custom Implementation](#custom-implementation)
  - [NumPy Implementation](#numpy-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

---

## Problem Statement

[Transformation Matrix from Basis B to C](https://www.deep-ml.com/problems/27)

Write a Python function to compute the **transformation matrix** from basis **B** to basis **C** for **ℝ³**.

---

## Example

```python
Input:

B = [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]]

C = [[1, 2.3, 3],
     [4.4, 25, 6],
     [7.4, 8, 9]]

Output:

[[-0.6772, -0.0126, 0.2342],
 [-0.0184,  0.0505, -0.0275],
 [ 0.5732, -0.0345, -0.0569]]
```

**Reasoning**

The transformation matrix is computed using:

```text
P = C⁻¹ × B
```

---

## Learn: Transformation Matrix

A transformation matrix converts vector coordinates from one basis to another.

Formula:

```text
P = C⁻¹ × B
```

Steps:

1. Compute the inverse of **C**.
2. Multiply **C⁻¹** with **B**.
3. The result is the transformation matrix.

---

## Solutions

### Custom Implementation

```python
def transform_basis(B: list[list[int]], C: list[list[int]]) -> list[list[float]]:

    def det_help(mat):
        n = len(mat)

        if n == 1:
            return mat[0][0]

        if n == 2:
            return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]

        det = 0

        for j in range(n):
            minor = [row[:j] + row[j + 1:] for row in mat[1:]]
            det += (-1) ** j * mat[0][j] * det_help(minor)

        return det

    def adjoint_help(mat):
        n = len(mat)

        if n == 1:
            return [[1]]

        cof = [[0] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                minor = [
                    row[:j] + row[j + 1:]
                    for r, row in enumerate(mat)
                    if r != i
                ]

                cof[i][j] = (-1) ** (i + j) * det_help(minor)

        return [[cof[j][i] for j in range(n)] for i in range(n)]

    det_C = det_help(C)
    adjoint_C = adjoint_help(C)

    inverse_C = [
        [x / det_C for x in row]
        for row in adjoint_C
    ]

    n = len(C)

    P = [
        [
            sum(inverse_C[i][k] * B[k][j] for k in range(n))
            for j in range(n)
        ]
        for i in range(n)
    ]

    return P


# Example Usage
print(transform_basis(B, C))
```

### NumPy Implementation

```python
import numpy as np


def transform_basis(B, C):
    return (np.linalg.inv(C) @ np.array(B)).tolist()
```

---

## Code Explanation

- Compute the determinant of **C** recursively.
- Find the adjoint matrix and compute **C⁻¹**.
- Multiply **C⁻¹** with **B**.

```python
P = C⁻¹ × B
```

- Return the transformation matrix.

---

## Time & Space Complexity

| Complexity | Value |
|------------|-------|
| Time | **O(n⁴)** |
| Space | **O(n²)** |