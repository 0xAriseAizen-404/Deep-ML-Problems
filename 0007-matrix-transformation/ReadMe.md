# Matrix Transformation (Medium, Linear Algebra)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Matrix Transformation](#learn-matrix-transformation)
- [Things to Note](#things-to-note)
- [Solutions](#solutions)
  - [NumPy Implementation](#numpy-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

---

# Problem Statement

[Matrix Transformation](https://www.deep-ml.com/problems/7)

Write a Python function `transform_matrix(A, T, S)` that transforms a matrix using

$$
T^{-1}AS
$$

where $T$ and $S$ are invertible matrices.

If either transformation matrix is **not invertible**, return `-1`.

---

# Example

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

print(transform_matrix(A, T, S))
```

### Output

```text
[
    [0.5, 1.5],
    [1.5, 3.5]
]
```

### Explanation

The determinants are

$$
\det(T)=4
$$

$$
\det(S)=1
$$

Since both are non-zero, both matrices are invertible.

The transformed matrix is

$$
T^{-1}AS
=
\begin{bmatrix}
0.5&1.5\\
1.5&3.5
\end{bmatrix}
$$

---

# Learn: Matrix Transformation

## What is a Matrix Transformation?

A matrix transformation changes the representation of a matrix by multiplying it with other matrices.

In this problem, the transformation is

$$
A'=T^{-1}AS
$$

where

- $A$ is the original matrix.
- $T^{-1}$ changes the basis on the left.
- $S$ changes the basis on the right.

The transformed matrix represents the same linear transformation under different coordinate systems.

---

## Mathematical Formula

The transformation is defined as

$$
A'
=
T^{-1}AS
$$

where

- $T^{-1}$ is the inverse of $T$.
- $A$ is the original matrix.
- $S$ is another invertible matrix.

Matrix multiplication is performed from left to right.

---

## Why Must $T$ and $S$ Be Invertible?

A matrix is invertible only if its determinant is non-zero.

Mathematically,

$$
\det(T)\ne0
$$

and

$$
\det(S)\ne0
$$

If

$$
\det(T)=0
$$

or

$$
\det(S)=0,
$$

then the inverse does not exist, making the transformation impossible.

---

## Example

Suppose

$$
A=
\begin{bmatrix}
1&2\\
3&4
\end{bmatrix}
$$

$$
T=
\begin{bmatrix}
2&0\\
0&2
\end{bmatrix}
$$

$$
S=
\begin{bmatrix}
1&1\\
0&1
\end{bmatrix}
$$

First compute

$$
T^{-1}
=
\begin{bmatrix}
\frac12&0\\
0&\frac12
\end{bmatrix}
$$

Then

$$
A'
=
T^{-1}AS
=
\begin{bmatrix}
0.5&1.5\\
1.5&3.5
\end{bmatrix}
$$

---

## Applications

Matrix transformations are fundamental in

- Linear Algebra
- Computer Graphics
- Robotics
- Computer Vision
- Coordinate Transformations
- Change of Basis
- Quantum Computing
- Machine Learning

---

# Things to Note

- Both $T$ and $S$ must be invertible.
- A matrix is invertible only if its determinant is non-zero.
- Matrix multiplication is **not commutative**, meaning

$$
AB\ne BA
$$

- The order

$$
T^{-1}AS
$$

must not be changed.

---

# Solutions

## NumPy Implementation

```python
import numpy as np

def transform_matrix(
    A: list[list[int | float]],
    T: list[list[int | float]],
    S: list[list[int | float]]
) -> list[list[int | float]]:

    detT = np.linalg.det(np.asarray(T))
    detS = np.linalg.det(np.asarray(S))

    if detT == 0 or detS == 0:
        return -1

    result = np.linalg.inv(np.asarray(T)) @ np.asarray(A) @ np.asarray(S)

    return np.round(result, 10).tolist()
```

---

# Code Explanation

## Step 1: Convert Inputs to NumPy Arrays

```python
np.asarray(T)
np.asarray(A)
np.asarray(S)
```

These convert Python lists into NumPy arrays so that optimized linear algebra operations can be used.

---

## Step 2: Compute the Determinants

```python
detT = np.linalg.det(np.asarray(T))
detS = np.linalg.det(np.asarray(S))
```

The determinant of each transformation matrix is computed.

Mathematically,

$$
\det(T)
$$

and

$$
\det(S)
$$

are calculated.

---

## Step 3: Check Whether the Matrices are Invertible

```python
if detT == 0 or detS == 0:
    return -1
```

If either determinant equals zero,

$$
\det(T)=0
$$

or

$$
\det(S)=0,
$$

then the corresponding inverse does not exist.

The function immediately returns `-1`.

---

## Step 4: Compute the Inverse of $T$

```python
np.linalg.inv(np.asarray(T))
```

This computes

$$
T^{-1}
$$

using NumPy's optimized inverse algorithm.

---

## Step 5: Perform the Matrix Transformation

```python
np.linalg.inv(np.asarray(T)) @ np.asarray(A) @ np.asarray(S)
```

The `@` operator performs matrix multiplication.

This directly computes

$$
T^{-1}AS
$$

in the required order.

---

## Step 6: Round the Result

```python
np.round(result, 10)
```

Floating-point arithmetic can introduce tiny numerical errors.

Rounding to 10 decimal places removes insignificant precision errors.

---

## Step 7: Convert Back to a Python List

```python
.tolist()
```

Converts the NumPy array back into a nested Python list.

---

# Time & Space Complexity

Assume the matrices are of size $n\times n$.

| Complexity | Value |
|------------|-------|
| Time | **O(n³)** |
| Space | **O(n²)** |

The dominant operations are computing the matrix inverse and performing matrix multiplication, both of which require cubic time for dense matrices.