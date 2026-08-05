# Calculate 2×2 Matrix Inverse (Easy, Linear Algebra)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Calculating the Inverse of a 2×2 Matrix](#learn-calculating-the-inverse-of-a-22-matrix)
- [Solutions](#solutions)
  - [Custom Implementation](#custom-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

## Problem Statement

### [Calculate 2×2 Matrix Inverse](https://www.deep-ml.com/problems/8)

Write a Python function that computes the **inverse of a 2×2 matrix**.

For a matrix

$$
A = \begin{bmatrix}
a & b\\
c & d
\end{bmatrix}
$$

the inverse exists only when its determinant is non-zero.

If the determinant is zero, the matrix is **singular** (non-invertible), and the function should return `None`.

---

## Example

### Input

```python
matrix = [
    [4, 7],
    [2, 6]
]
```

### Output

```python
[
    [0.6, -0.7],
    [-0.2, 0.4]
]
```

### Reasoning

The determinant is

$$
\det(A) = (4)(6) - (7)(2) = 24 - 14 = 10
$$

Since

$$
\det(A) \neq 0
$$

the matrix is invertible.

Using the inverse formula,

$$
A^{-1} = \frac{1}{10}
\begin{bmatrix}
6 & -7\\
-2 & 4
\end{bmatrix} =
\begin{bmatrix}
0.6 & -0.7\\
-0.2 & 0.4
\end{bmatrix}
$$

---

## Learn: Calculating the Inverse of a 2×2 Matrix

### What is a Matrix Inverse?

The inverse of a square matrix is another matrix that "undoes" the effect of the original matrix.

For a matrix

$$
A
$$

its inverse

$$
A^{-1}
$$

satisfies

$$
AA^{-1} = A^{-1}A = I
$$

where

$$
I = \begin{bmatrix}
1 & 0\\
0 & 1
\end{bmatrix}
$$

is the **identity matrix**.

Multiplying a matrix by its inverse produces the identity matrix, just as multiplying a number by its reciprocal produces 1.

---

### Formula for a 2×2 Matrix

Consider

$$
A = \begin{bmatrix}
a & b\\
c & d
\end{bmatrix}
$$

Its determinant is

$$
\det(A) = ad - bc
$$

If

$$
\det(A) \neq 0
$$

then the inverse is

$$
A^{-1} = \frac{1}{\det(A)}
\begin{bmatrix}
d & -b\\
-c & a
\end{bmatrix}
$$

Notice the pattern:

- Swap the diagonal elements.
- Negate the off-diagonal elements.
- Divide every element by the determinant.

---

### When Does the Inverse Exist?

A matrix is invertible **if and only if** its determinant is non-zero.

If

$$
\det(A) \neq 0
$$

the inverse exists.

If

$$
\det(A) = 0
$$

the matrix is called **singular**, and no inverse exists.

A singular matrix loses information during transformation, making it impossible to recover the original values.

---

### Step-by-Step Algorithm

Given

$$
A = \begin{bmatrix}
a & b\\
c & d
\end{bmatrix}
$$

#### Step 1

Compute the determinant.

$$
\det(A) = ad - bc
$$

#### Step 2

If

$$
\det(A) = 0
$$

return `None`.

#### Step 3

Otherwise compute

$$
A^{-1} = \frac{1}{\det(A)}
\begin{bmatrix}
d & -b\\
-c & a
\end{bmatrix}
$$

---

### Geometric Interpretation

A matrix represents a linear transformation.

Its inverse performs the **reverse transformation**, mapping transformed vectors back to their original positions.

If the determinant is zero, the transformation collapses the space into a lower dimension (for example, a plane into a line), making recovery impossible.

---

### Characteristics / Key Points

- Defined only for square matrices.
- A non-zero determinant is required.
- The inverse is unique.
- Singular matrices do not have an inverse.
- Multiplying a matrix by its inverse gives the identity matrix.
- Every invertible matrix represents a reversible transformation.

---

### Why is it Used?

Matrix inverses are fundamental in Linear Algebra and Machine Learning.

Some applications include:

- Solving systems of linear equations
- Linear Regression (Normal Equation)
- Computer Graphics
- Robotics
- Coordinate transformations
- Cryptography
- Control systems
- Scientific simulations

For example, a linear system

$$
Ax = b
$$

can be solved as

$$
x = A^{-1}b
$$

provided that

$$
A
$$

is invertible.

---

> 💡 **Important Note**
>
> Although the inverse provides a direct solution for linear systems, explicitly computing a matrix inverse is often **not recommended** for large matrices due to numerical instability and computational cost. In practice, libraries such as NumPy use methods like **LU decomposition** (`numpy.linalg.solve`) instead of calculating the inverse directly.

---

## Solutions

### Custom Implementation

```python
def inverse_2x2(matrix: list[list[float]]) -> list[list[float]] | None:
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

## Code Explanation

### 1. Extract the Matrix Elements

```python
[[a, b], [c, d]] = matrix
```

The four matrix elements are unpacked for easier access.

---

### 2. Compute the Determinant

```python
det = a * d - b * c
```

This evaluates

$$
\det(A) = ad - bc
$$

The determinant determines whether the inverse exists.

---

### 3. Check for Singularity

```python
if det == 0:
    return None
```

If the determinant is zero, the matrix is singular and cannot be inverted.

---

### 4. Apply the Inverse Formula

```python
return [
    [d / det, -b / det],
    [-c / det, a / det]
]
```

The implementation directly follows

$$
A^{-1} = \frac{1}{\det(A)}
\begin{bmatrix}
d & -b\\
-c & a
\end{bmatrix}
$$

Each element is divided by the determinant to produce the inverse matrix.

---

## Time & Space Complexity

Since the matrix size is fixed (**2 × 2**), the algorithm performs a constant number of arithmetic operations.

| Complexity | Value |
| ---------- | ----- |
| Time | **O(1)** |
| Space | **O(1)** |

The computation requires only a few scalar variables and returns a fixed-size 2×2 matrix.