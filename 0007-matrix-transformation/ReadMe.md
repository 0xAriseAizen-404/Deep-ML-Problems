# Matrix Transformation (Medium, Linear Algebra)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Matrix Transformation using $T^{-1}AS$](#learn-matrix-transformation-using-t-1as)
- [Solutions](#solutions)
  - [NumPy Implementation](#numpy-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

## Problem Statement

### [Matrix Transformation](https://www.deep-ml.com/problems/7)

Write a Python function that transforms a matrix using the matrix transformation

$$
A' = T^{-1}AS
$$

where:

- $A$ is the original matrix.
- $T$ and $S$ are invertible matrices.
- $T^{-1}$ denotes the inverse of matrix $T$.

Before performing the transformation, verify that both $T$ and $S$ are invertible. If either matrix is singular (non-invertible), return `-1`.

---

## Example

### Input

```python
A = [
    [1, 2],
    [3, 4]
]

T = [
    [2, 0],
    [0, 2]
]

S = [
    [1, 1],
    [0, 1]
]
```

### Output

```python
[
    [0.5, 1.5],
    [1.5, 3.5]
]
```

### Reasoning

First compute

$$
T^{-1} = \begin{bmatrix}
0.5 & 0\\
0 & 0.5
\end{bmatrix}
$$

Then,

$$
A' = T^{-1}AS =
\begin{bmatrix}
0.5 & 0\\
0 & 0.5
\end{bmatrix}
\begin{bmatrix}
1 & 2\\
3 & 4
\end{bmatrix}
\begin{bmatrix}
1 & 1\\
0 & 1
\end{bmatrix} =
\begin{bmatrix}
0.5 & 1.5\\
1.5 & 3.5
\end{bmatrix}
$$

---

## Learn: Matrix Transformation using $T^{-1}AS$

### What is Matrix Transformation?

Matrix transformation modifies a matrix by applying linear transformations on its rows and columns.

The transformation

$$
A' = T^{-1}AS
$$

uses two invertible matrices:

- $T^{-1}$ transforms the row space.
- $S$ transforms the column space.

Unlike simple scalar multiplication or transpose, this operation changes the representation of the matrix while preserving its essential linear properties when the transformation matrices are invertible.

---

### Mathematical Definition

Given

$$
A \in \mathbb{R}^{m \times n}
$$

and invertible matrices

$$
T \in \mathbb{R}^{m \times m}
$$

and

$$
S \in \mathbb{R}^{n \times n}
$$

the transformed matrix is

$$
A' = T^{-1}AS
$$

The dimensions remain unchanged.

---

### Why Must $T$ and $S$ be Invertible?

A matrix is invertible only if its determinant is non-zero.

For matrix $T$,

$$
\det(T) \ne 0
$$

Similarly,

$$
\det(S) \ne 0
$$

If either determinant equals zero,

$$
\det(T)=0
$$

or

$$
\det(S)=0
$$

the inverse does not exist, making the transformation impossible.

---

### Matrix Inverse

For a matrix

$$
T
$$

its inverse satisfies

$$
TT^{-1}=T^{-1}T=I
$$

where

$$
I
$$

is the identity matrix.

Multiplying a matrix by its inverse recovers the original matrix.

---

### Transformation Process

The transformation consists of three sequential operations.

#### Step 1

Verify that

$$
\det(T) \ne 0
$$

and

$$
\det(S) \ne 0
$$

---

#### Step 2

Compute

$$
T^{-1}
$$

---

#### Step 3

Perform matrix multiplication

$$
A' = T^{-1}AS
$$

Since matrix multiplication is associative,

$$
(T^{-1}A)S = T^{-1}(AS)
$$

Both expressions produce the same result.

---

### Characteristics / Key Points

- Requires both transformation matrices to be invertible.
- Preserves matrix dimensions.
- Changes the coordinate representation of the matrix.
- Uses standard matrix multiplication.
- Matrix multiplication is associative but **not** commutative.
- Widely used in change-of-basis problems.

---

### Similarity Transformation

A special case occurs when

$$
S=T
$$

The transformation becomes

$$
A' = T^{-1}AT
$$

This is called a **similarity transformation**.

Two similar matrices have:

- The same eigenvalues.
- The same determinant.
- The same trace.
- The same characteristic polynomial.

Only their representation changes.

---

### Why is it Used?

Matrix transformations are fundamental in many scientific and engineering applications.

Some common applications include:

- Change of basis
- Coordinate transformations
- Principal Component Analysis (PCA)
- Diagonalization
- Eigenvalue decomposition
- Robotics
- Computer graphics
- Control systems
- Quantum mechanics
- Numerical linear algebra

Many machine learning algorithms transform data into a new coordinate system where computations become simpler or more interpretable.

---

> 💡 **Important Note**
>
> Do not confuse matrix transformation with element-wise operations. The expression $T^{-1}AS$ performs **matrix multiplication**, which depends on matrix dimensions and multiplication order. Since matrix multiplication is **not commutative**, changing the order (for example, computing $SAT^{-1}$) generally produces a completely different result.

---

## Solutions

### NumPy Implementation

```python
import numpy as np

def transform_matrix(A: list[list[int | float]],
                     T: list[list[int | float]],
                     S: list[list[int | float]]) -> list[list[int | float]]:

    detT = np.linalg.det(np.asarray(T))
    detS = np.linalg.det(np.asarray(S))

    if detT == 0 or detS == 0:
        return -1

    result = np.linalg.inv(np.asarray(T)) @ np.asarray(A) @ np.asarray(S)

    return np.round(result, 10).tolist()
```

---

## Code Explanation

### 1. Compute the Determinants

```python
detT = np.linalg.det(np.asarray(T))
detS = np.linalg.det(np.asarray(S))
```

The determinant determines whether a matrix is invertible.

If

$$
\det(T)=0
$$

or

$$
\det(S)=0
$$

the transformation cannot be performed.

---

### 2. Validate Invertibility

```python
if detT == 0 or detS == 0:
    return -1
```

Only invertible matrices possess an inverse.

If either matrix is singular, the function immediately returns `-1`.

---

### 3. Compute the Inverse of $T$

```python
np.linalg.inv(np.asarray(T))
```

NumPy computes

$$
T^{-1}
$$

which satisfies

$$
TT^{-1}=I
$$

---

### 4. Perform Matrix Multiplication

```python
result = np.linalg.inv(T) @ A @ S
```

The `@` operator performs matrix multiplication.

The computation follows the formula

$$
A' = T^{-1}AS
$$

producing the transformed matrix.

---

### 5. Round Numerical Errors

```python
np.round(result, 10)
```

Floating-point arithmetic can introduce tiny numerical inaccuracies such as

```text
0.49999999997
```

Rounding improves readability while preserving numerical accuracy.

---

### 6. Convert Back to a Python List

```python
.tolist()
```

The final NumPy array is converted into a standard Python list before returning.

---

## Time & Space Complexity

Let

- $n$ = number of rows (and columns for square matrices)

Computing a matrix inverse dominates the overall complexity.

| Complexity | Value     |
| ---------- | --------- |
| Time       | **O(n³)** |
| Space      | **O(n²)** |

The inverse computation and matrix multiplications both require storing intermediate matrices of size $n \times n$.
