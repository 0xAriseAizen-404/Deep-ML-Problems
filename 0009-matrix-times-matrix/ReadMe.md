# Matrix × Matrix Multiplication (Medium, Linear Algebra)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Matrix Multiplication](#learn-matrix-multiplication)
- [Solutions](#solutions)
  - [Custom Implementation](#custom-implementation)
  - [NumPy Implementation](#numpy-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

---

## Problem Statement

[Matrix times Matrix](https://www.deep-ml.com/problems/9)

Write a Python function to multiply two matrices **A** and **B**.

Return **`-1`** if the matrices cannot be multiplied.

---

## Example

```python
Input:
A = [[1, 2],
     [2, 4]]

B = [[2, 1],
     [3, 4]]

Output:
[[8, 9],
 [16, 18]]
```

**Reasoning**

```
8  = 1×2 + 2×3
9  = 1×1 + 2×4
16 = 2×2 + 4×3
18 = 2×1 + 4×4
```

---

## Learn: Matrix Multiplication

Two matrices can be multiplied **only if**:

```
Columns of A == Rows of B
```

If

```
A = (m × n)

B = (n × p)
```

then

```
A × B = (m × p)
```

Each element of the result is calculated by taking the **dot product** of a row from **A** and a column from **B**.

Formula:

```
C[i][j] = Σ (A[i][k] × B[k][j])
```

---

## Solutions

### Custom Implementation

```python
def matrix_multiply(A: list[list[int | float]],
                    B: list[list[int | float]]) -> list[list[int | float]] | int:

    if len(A[0]) != len(B):
        return -1

    rows = len(A)
    cols = len(B[0])
    common = len(B)

    result = [[0] * cols for _ in range(rows)]

    for i in range(rows):
        for j in range(cols):
            for k in range(common):
                result[i][j] += A[i][k] * B[k][j]

    return result


# Example Usage
A = [[1, 2],
     [2, 4]]

B = [[2, 1],
     [3, 4]]

print(matrix_multiply(A, B))
```

---

### NumPy Implementation

```python
import numpy as np


def matrix_multiply(A: list[list[int | float]],
                    B: list[list[int | float]]):

    A = np.array(A)
    B = np.array(B)

    if A.shape[1] != B.shape[0]:
        return -1

    return np.matmul(A, B)


# Example Usage
print(matrix_multiply(A, B))
```

---

## Code Explanation

- Check if multiplication is possible.

```python
if len(A[0]) != len(B):
    return -1
```

- Create a result matrix filled with zeros.

```python
result = [[0] * cols for _ in range(rows)]
```

- Multiply each row of **A** with each column of **B**.

```python
result[i][j] += A[i][k] * B[k][j]
```

- Return the resulting matrix.

---

## Time & Space Complexity

### Custom Implementation

| Complexity | Value |
|------------|-------|
| Time | **O(m × n × p)** |
| Space | **O(m × p)** |

where

- **m** = rows of A
- **n** = columns of A (or rows of B)
- **p** = columns of B

### NumPy Implementation

- **Time:** Optimized (internally uses highly efficient BLAS/LAPACK libraries)
- **Space:** O(m × p)