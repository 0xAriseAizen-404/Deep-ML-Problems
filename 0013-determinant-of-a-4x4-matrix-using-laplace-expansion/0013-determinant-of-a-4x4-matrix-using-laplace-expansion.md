# Determinant of a 4×4 Matrix using Laplace's Expansion (Hard, Linear Algebra)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Laplace's Expansion](#learn-laplaces-expansion)
- [Solutions](#solutions)
  - [Recursive Implementation](#recursive-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

---

## Problem Statement

[Determinant of a 4×4 Matrix using Laplace's Expansion](https://www.deep-ml.com/problems/13)

Write a Python function to calculate the **determinant of a 4×4 matrix** using **Laplace's Expansion**. Implement the solution recursively to compute the determinants of the minor matrices.

---

## Example

```python
Input:

A = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
]

Output:

0
```

**Reasoning**

The rows of the matrix are linearly dependent, so its determinant is **0**.

---

## Learn: Laplace's Expansion

Laplace's Expansion computes the determinant by expanding along any row or column.

Formula (first row):

```text
det(A) = a₁₁M₁₁ - a₁₂M₁₂ + a₁₃M₁₃ - a₁₄M₁₄
```

where:

- **Minor (Mᵢⱼ)** → Determinant of the matrix after removing row *i* and column *j*.
- **Cofactor** → Applies alternating signs (+ − + −).

The process is recursive until reaching a **2×2** matrix.

---

## Solutions

### Recursive Implementation

```python
def determinant_4x4(matrix: list[list[int | float]]) -> float:

    def determinant(mat: list[list[int | float]]) -> float:
        n = len(mat)

        if n == 1:
            return mat[0][0]

        if n == 2:
            return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]

        det = 0

        for j in range(n):
            minor = [
                row[:j] + row[j + 1:]
                for row in mat[1:]
            ]

            det += (-1) ** j * mat[0][j] * determinant(minor)

        return det

    return determinant(matrix)


# Example Usage
A = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
]

print(determinant_4x4(A))
```

---

## Code Explanation

- Base case:
  - **1×1** → Return the element.
  - **2×2** → Compute `ad - bc`.

- For larger matrices:
  - Remove the current row and column to form the **minor**.
  - Recursively compute its determinant.
  - Multiply by the corresponding element and alternating sign.
  - Sum all cofactors.

---

## Time & Space Complexity

| Complexity | Value |
|------------|-------|
| Time | **O(n!)** |
| Space | **O(n²)** (recursive call stack and minor matrices) |

> For this problem, `n = 4`, so the recursion depth is small and performs well.