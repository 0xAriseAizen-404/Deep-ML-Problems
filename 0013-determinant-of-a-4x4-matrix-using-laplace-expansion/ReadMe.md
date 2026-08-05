# Determinant of a 4×4 Matrix using Laplace's Expansion (Hard, Linear Algebra)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Determinant using Laplace's Expansion](#learn-determinant-using-laplaces-expansion)
- [Solutions](#solutions)
  - [Custom Implementation](#custom-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

## Problem Statement

### [Determinant of a 4×4 Matrix using Laplace's Expansion](https://www.deep-ml.com/problems/13)

Write a Python function that computes the **determinant of a 4×4 matrix** using **Laplace's Expansion (Cofactor Expansion)**.

The implementation should recursively compute determinants of the smaller **3×3**, **2×2**, and **1×1** minor matrices until reaching the base case.

---

## Example

### Input

```python
matrix = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
]
```

### Output

```python
0
```

### Reasoning

Expanding along the first row,

$$
\det(A)=a_{11}M_{11}-a_{12}M_{12}+a_{13}M_{13}-a_{14}M_{14}
$$

where each $M_{1j}$ is the determinant of the corresponding **3×3 minor**.

For this matrix, every row is linearly dependent, so

$$
\det(A)=0
$$

indicating that the matrix is **singular**.

---

## Learn: Determinant using Laplace's Expansion

### What is a Determinant?

The determinant is a scalar associated with every square matrix.

It provides valuable information about the matrix, including:

- Whether the matrix is invertible.
- The scaling factor of a linear transformation.
- Whether the rows or columns are linearly independent.
- The volume scaling factor in higher-dimensional space.

For any square matrix

$$
A
$$

its determinant is denoted by

$$
\det(A)
$$

---

### Why is the Determinant Important?

If

$$
\det(A)\neq0
$$

then

- the matrix is invertible,
- the rows are linearly independent,
- the columns are linearly independent.

If

$$
\det(A)=0
$$

then the matrix is **singular**, meaning information is lost during the transformation and an inverse does not exist.

---

### Laplace's Expansion

Laplace's Expansion computes determinants recursively by expanding along any row or column.

For a matrix

$$
A=[a_{ij}]
$$

expanding along the first row gives

$$
\det(A)=\sum_{j=1}^{n}a_{1j}C_{1j}
$$

where

$$
C_{ij}=(-1)^{i+j}M_{ij}
$$

is the **cofactor**.

Substituting the cofactors,

$$
\det(A)=a_{11}M_{11}-a_{12}M_{12}+a_{13}M_{13}-a_{14}M_{14}
$$

for a 4×4 matrix.

---

### Minor

The **minor** of an element is obtained by removing its row and column.

For element

$$
a_{ij}
$$

its minor is

$$
M_{ij}
$$

which is the determinant of the remaining matrix.

For a 4×4 matrix, removing one row and one column produces a **3×3 minor**.

---

### Cofactor

The cofactor applies alternating signs to the minors.

It is defined as

$$
C_{ij}=(-1)^{i+j}M_{ij}
$$

The sign pattern is

$$
\begin{bmatrix}
+ & - & + & -\\
- & + & - & +\\
+ & - & + & -\\
- & + & - & +
\end{bmatrix}
$$

This checkerboard pattern ensures the determinant is computed correctly.

---

### Recursive Computation

Laplace's Expansion naturally leads to recursion.

The determinant of a

- 4×4 matrix requires several 3×3 determinants.
- 3×3 determinant requires several 2×2 determinants.
- 2×2 determinant is computed directly.

For a 2×2 matrix,

$$
\begin{bmatrix}
a & b\\
c & d
\end{bmatrix}
$$

the determinant is

$$
ad-bc
$$

This serves as the recursion's base mathematical formula.

---

### Recursive Tree

For a 4×4 matrix,

```text
4×4
│
├── 3×3
│   ├── 2×2
│   ├── 2×2
│   └── 2×2
│
├── 3×3
│   ├── 2×2
│   ├── 2×2
│   └── 2×2
│
├── ...
```

Each recursive call solves a smaller determinant until reaching the base case.

---

### Characteristics / Key Points

- Works for matrices of any size.
- Naturally implemented using recursion.
- Uses minors and cofactors.
- Expansion may be performed along any row or column.
- Choosing a row or column with many zeros reduces computation.
- Produces the exact determinant.

---

### Why is it Used?

Determinants are fundamental in many Linear Algebra applications.

Some common uses include:

- Matrix inversion
- Solving linear systems
- Eigenvalue computation
- Change of variables
- Computer graphics
- Robotics
- Physics simulations
- Differential equations
- Volume computation in higher dimensions

---

> 💡 **Important Note**
>
> Laplace's Expansion is excellent for understanding determinant computation, but it is **computationally expensive**. Modern numerical libraries (such as NumPy) compute determinants using **LU decomposition** or Gaussian Elimination, reducing the complexity from factorial time to approximately **O(n³)**.

---

## Solutions

### Custom Implementation

```python
def determinant_4x4(matrix: list[list[int | float]]) -> float:

    def det_help(mat):

        n = len(mat)

        if n == 1:
            return mat[0][0]

        if n == 2:
            return (
                mat[0][0] * mat[1][1]
                - mat[0][1] * mat[1][0]
            )

        det = 0

        for j in range(n):

            minor = [
                row[:j] + row[j + 1:]
                for row in mat[1:]
            ]

            det += (-1) ** j * mat[0][j] * det_help(minor)

        return det

    return det_help(matrix)
```

---

## Code Explanation

### 1. Recursive Helper Function

```python
def det_help(mat):
```

The recursive helper computes the determinant of matrices of any size.

Each recursive call reduces the matrix dimension by one.

---

### 2. Base Case: 1×1 Matrix

```python
if n == 1:
    return mat[0][0]
```

A single-element matrix has determinant equal to that element.

---

### 3. Base Case: 2×2 Matrix

```python
return (
    mat[0][0] * mat[1][1]
    - mat[0][1] * mat[1][0]
)
```

This directly evaluates

$$
\det(A)=ad-bc
$$

which terminates the recursion.

---

### 4. Expand Along the First Row

```python
for j in range(n):
```

Each element of the first row contributes one cofactor term.

The determinant is computed as

$$
\det(A)=\sum_{j=1}^{n}a_{1j}C_{1j}
$$

---

### 5. Construct the Minor Matrix

```python
minor = [
    row[:j] + row[j + 1:]
    for row in mat[1:]
]
```

The first row and the current column are removed to construct the corresponding minor matrix.

---

### 6. Recursive Cofactor Expansion

```python
det += (-1) ** j * mat[0][j] * det_help(minor)
```

This implements

$$
\det(A)=a_{11}M_{11}-a_{12}M_{12}+a_{13}M_{13}-\cdots
$$

The recursive call computes the determinant of the smaller matrix.

---

### 7. Return the Final Determinant

After all cofactors have been evaluated, the accumulated determinant is returned.

---

## Time & Space Complexity

Let

- $n$ = size of the square matrix.

Laplace's Expansion recursively computes determinants of smaller matrices, resulting in factorial growth.

| Complexity | Value     |
| ---------- | --------- |
| Time       | **O(n!)** |
| Space      | **O(n²)** |

The recursion stack has depth **O(n)**, while each recursive call constructs a new minor matrix, requiring additional memory proportional to the matrix size.
