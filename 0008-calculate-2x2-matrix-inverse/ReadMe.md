# Inverse of a 2×2 Matrix (Easy, Linear Algebra)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Inverse of a 2×2 Matrix](#learn-inverse-of-a-2x2-matrix)
- [Things to Note](#things-to-note)
- [Solutions](#solutions)
  - [Custom Implementation](#custom-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

---

# Problem Statement

[Calculate 2×2 Matrix Inverse](https://www.deep-ml.com/problems/8)

Write a Python function `inverse_2x2(matrix)` that computes the inverse of a **2×2 matrix**.

If the matrix is **singular** (its determinant is zero), return `None`.

---

# Example

```python
matrix = [
    [4, 7],
    [2, 6]
]

print(inverse_2x2(matrix))
```

### Output

```text
[
    [0.6, -0.7],
    [-0.2, 0.4]
]
```

### Explanation

Given

$$
A=
\begin{bmatrix}
4&7\\
2&6
\end{bmatrix}
$$

its determinant is

$$
\det(A)=4\times6-7\times2=10
$$

Since

$$
\det(A)\neq0
$$

the inverse exists.

---

# Learn: Inverse of a 2×2 Matrix

## What is the Inverse of a Matrix?

The inverse of a matrix is analogous to the reciprocal of a number.

For numbers,

$$
5\times\frac15=1
$$

Similarly, for matrices,

$$
AA^{-1}=I
$$

where

$$
I=
\begin{bmatrix}
1&0\\
0&1
\end{bmatrix}
$$

is called the **Identity Matrix**.

Multiplying a matrix by its inverse always produces the identity matrix.

---

## Formula

Suppose

$$
A=
\begin{bmatrix}
a&b\\
c&d
\end{bmatrix}
$$

First compute its determinant

$$
\det(A)=ad-bc
$$

If

$$
\det(A)\neq0
$$

then

$$
A^{-1}
=
\frac1{\det(A)}
\begin{bmatrix}
d&-b\\
-c&a
\end{bmatrix}
$$

---

## Why Does the Determinant Matter?

The determinant measures whether a matrix preserves dimensionality.

- If $\det(A)\neq0$, the rows (and columns) are linearly independent, so the matrix is invertible.
- If $\det(A)=0$, the matrix is singular and no inverse exists.

For example,

$$
\begin{bmatrix}
1&2\\
2&4
\end{bmatrix}
$$

has

$$
\det(A)=1\times4-2\times2=0
$$

Therefore, the inverse does not exist.

---

## How the Formula is Derived

For

$$
A=
\begin{bmatrix}
a&b\\
c&d
\end{bmatrix}
$$

the inverse is obtained by:

1. Swapping the diagonal elements $a$ and $d$.
2. Negating the off-diagonal elements $b$ and $c$.
3. Dividing every element by the determinant.

This produces

$$
A^{-1}
=
\frac1{ad-bc}
\begin{bmatrix}
d&-b\\
-c&a
\end{bmatrix}
$$

---

## Applications

The inverse of a matrix is used in:

- Solving systems of linear equations
- Linear Regression
- Computer Graphics
- Robotics
- Control Systems
- Signal Processing
- Machine Learning
- Numerical Analysis

---

# Things to Note

- Only **square matrices** can have inverses.
- A $2\times2$ matrix is invertible only when $\det(A)\neq0$.
- If $\det(A)=0$, the matrix is singular.
- Swapping the diagonal elements and negating the off-diagonal elements is specific to the **2×2 inverse formula**.
- Larger matrices require methods such as Gaussian Elimination or LU Decomposition.

---

# Solutions

## Custom Implementation

```python
def inverse_2x2(matrix: list[list[float]]) -> list[list[float]] | None:
    """
    Calculate the inverse of a 2x2 matrix.

    Args:
        matrix: A 2x2 matrix represented as [[a, b], [c, d]]

    Returns:
        The inverse matrix as a 2x2 list, or None if the matrix is singular
        (i.e., determinant equals zero)
    """
    [[a, b], [c, d]] = matrix

    det = a * d - b * c

    if det == 0:
        return None

    return [
        [d / det, -b / det],
        [-c / det, a / det]
    ]
```

---

# Code Explanation

## Step 1: Extract Matrix Elements

```python
[[a, b], [c, d]] = matrix
```

The matrix

$$
\begin{bmatrix}
a&b\\
c&d
\end{bmatrix}
$$

is unpacked into four variables.

---

## Step 2: Compute the Determinant

```python
det = a * d - b * c
```

This computes

$$
\det(A)=ad-bc
$$

The determinant determines whether the inverse exists.

---

## Step 3: Check if the Matrix is Invertible

```python
if det == 0:
    return None
```

If

$$
\det(A)=0
$$

the matrix is singular, so no inverse exists.

---

## Step 4: Apply the Inverse Formula

```python
return [
    [d / det, -b / det],
    [-c / det, a / det]
]
```

This directly implements

$$
A^{-1}
=
\frac1{\det(A)}
\begin{bmatrix}
d&-b\\
-c&a
\end{bmatrix}
$$

Each element is divided by the determinant to produce the inverse matrix.

---

# Time & Space Complexity

Since the matrix size is fixed at $2\times2$:

| Complexity | Value |
|------------|-------|
| Time | **O(1)** |
| Space | **O(1)** |

The algorithm performs a constant number of arithmetic operations regardless of the input values.