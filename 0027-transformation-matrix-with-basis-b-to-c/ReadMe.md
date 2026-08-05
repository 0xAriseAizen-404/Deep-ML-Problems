# Transformation Matrix from Basis B to C (Easy, Linear Algebra)

## Table of Contents

- Problem Statement
- Example
- Learn: Understanding Transformation Matrices
- Solution
- Code Explanation
- Time & Space Complexity

## Problem Statement

### [Transformation Matrix from Basis B to C](https://www.deep-ml.com/problems/27)

Given two bases **B** and **C** of $\mathbb{R}^3$, write a Python function that computes the transformation matrix that converts vector coordinates from basis **B** to basis **C**.

If **B** and **C** are represented as matrices whose columns are basis vectors, compute the transformation matrix

$$
P = C^{-1}B
$$

and return the resulting matrix.

---

## Example

**Input**

```python
B = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
]

C = [
    [1, 2.3, 3],
    [4.4, 25, 6],
    [7.4, 8, 9]
]
```

**Output**

```python
[
    [-0.6772, -0.0126, 0.2342],
    [-0.0184, 0.0505, -0.0275],
    [0.5732, -0.0345, -0.0569]
]
```

**Reasoning**

Since **B** is the standard basis (identity matrix), the transformation matrix is simply the inverse of **C**.

$$
P = C^{-1}I = C^{-1}
$$

---

## Learn: Understanding Transformation Matrices

### What is a Transformation Matrix?

A transformation matrix converts the coordinates of a vector from one basis to another.

Suppose a vector has coordinates with respect to basis **B**, but we want its coordinates in basis **C**.

Instead of changing the vector itself, we change its coordinate representation using a transformation matrix.

If

- **B** is the source basis,
- **C** is the destination basis,

then the coordinate transformation is

$$
[x]_C = P[x]_B
$$

where $P$ is the change-of-basis matrix.

---

### Mathematical Definition

Let

$$
B =
\begin{bmatrix}
| & | & |\\
b_1 & b_2 & b_3\\
| & | & |
\end{bmatrix}
$$

and

$$
C =
\begin{bmatrix}
| & | & |\\
c_1 & c_2 & c_3\\
| & | & |
\end{bmatrix}
$$

where each column is a basis vector.

The transformation matrix from basis **B** to basis **C** is

$$
P = C^{-1}B
$$

For any coordinate vector,

$$
[x]_C = P[x]_B
$$

If **B** is the standard basis,

$$
B = I
$$

then

$$
P = C^{-1}
$$

---

### Why Does This Formula Work?

A vector expressed in basis **B** can first be converted into its standard coordinate representation by multiplying with **B**.

$$
x = B[x]_B
$$

To obtain its coordinates in basis **C**, multiply by the inverse of **C**.

$$
[x]_C = C^{-1}x
$$

Substituting

$$
x = B[x]_B
$$

gives

$$
[x]_C = C^{-1}B[x]_B
$$

Therefore,

$$
P = C^{-1}B
$$

---

### Computing the Inverse

To compute the inverse manually, three steps are required.

First compute the determinant.

$$
\det(C)
$$

If

$$
\det(C)=0
$$

the matrix is singular and no inverse exists.

Next compute the cofactor matrix.

$$
C_{ij}=(-1)^{i+j}M_{ij}
$$

where

- $M_{ij}$ is the determinant of the corresponding minor.

Transpose the cofactor matrix to obtain the adjoint.

$$
Adj(C)=Cof(C)^T
$$

Finally,

$$
C^{-1}=\frac{1}{\det(C)}Adj(C)
$$

---

### Characteristics / Key Points

- A change-of-basis matrix converts coordinate systems without changing the vector.
- The destination basis matrix must be invertible.
- The determinant must be non-zero.
- Columns of a basis matrix are linearly independent.
- If both bases are identical, the transformation matrix is the identity matrix.
- If the source basis is the standard basis, the transformation matrix equals the inverse of the destination basis.

---

### Why is it used? / Applications

Transformation matrices are widely used in mathematics and machine learning.

Applications include:

- Coordinate system conversion
- Computer graphics
- Robotics and kinematics
- Computer vision
- 3D rendering
- Eigenvector basis transformations
- Principal Component Analysis (PCA)
- Linear transformations in numerical computing

---

> 💡 **Important Note**
>
> A common misconception is that changing the basis changes the vector. It does not. The geometric vector remains the same; only its coordinate representation changes. The transformation matrix changes how the vector is described, not the vector itself.

---

## Solution

### Custom Implementation

```python
def transform_basis(B: list[list[int]], C: list[list[int]]) -> list[list[float]]:

    def det_help(mat: list[list[int]]) -> float:
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

    def adjoint_help(mat: list[list[int]]) -> list[list[float]]:
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
                cof[i][j] = ((-1) ** (i + j)) * det_help(minor)

        adj = [[cof[j][i] for j in range(n)] for i in range(n)]
        return adj

    det_C = det_help(C)
    adjoint_mat_C = adjoint_help(C)

    inverse_C = [
        [x / det_C for x in row]
        for row in adjoint_mat_C
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
```

### NumPy Implementation

```python
import numpy as np

def transform_basis(B, C):
    return (np.linalg.inv(C) @ np.array(B)).tolist()
```

---

## Code Explanation

### Step 1

Compute the determinant of matrix **C** recursively using Laplace Expansion.

If the determinant is zero, the inverse does not exist.

---

### Step 2

Compute the cofactor matrix.

Each element is obtained by

- Removing one row
- Removing one column
- Computing the determinant of the remaining minor
- Multiplying by the alternating sign

$$
(-1)^{i+j}
$$

---

### Step 3

Transpose the cofactor matrix to obtain the adjoint matrix.

$$
Adj(C)=Cof(C)^T
$$

---

### Step 4

Compute the inverse.

$$
C^{-1}=\frac{1}{\det(C)}Adj(C)
$$

Each element of the adjoint matrix is divided by the determinant.

---

### Step 5

Multiply the inverse of **C** with **B**.

$$
P=C^{-1}B
$$

Each element of the resulting matrix is computed as the dot product between

- a row of $C^{-1}$
- a column of $B$

---

### Step 6

Return the transformation matrix.

This matrix converts coordinates from basis **B** into basis **C**.

---

## Time & Space Complexity

| Complexity | Value          |
| ---------- | -------------- |
| Time       | **O(n! + n³)** |
| Space      | **O(n²)**      |

where

- **n** is the dimension of the square basis matrices.
- Recursive Laplace Expansion dominates the running time with **O(n!)**, while the final matrix multiplication requires **O(n³)**. For this problem, $n=3$, so the computation is effectively constant time.
