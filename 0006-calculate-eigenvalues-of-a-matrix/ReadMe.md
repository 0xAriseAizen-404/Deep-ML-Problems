# Calculate Eigenvalues of a Matrix (Medium, Linear Algebra)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Calculate Eigenvalues](#learn-calculate-eigenvalues)
- [Solutions](#solutions)
  - [Custom Implementation](#custom-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

## Problem Statement

### [Calculate Eigenvalues of a Matrix](https://www.deep-ml.com/problems/6)

Write a Python function that computes the **eigenvalues** of a **2 × 2** matrix. The function should return a list containing the eigenvalues sorted in **descending order**.

For a matrix

$$
A = \begin{bmatrix}
a & b\\
c & d
\end{bmatrix}
$$

the eigenvalues are obtained by solving its characteristic equation.

---

## Example

### Input

```python
matrix = [
    [2, 1],
    [1, 2]
]
```

### Output

```python
[3.0, 1.0]
```

### Reasoning

The trace of the matrix is

$$
tr(A) = 2 + 2 = 4
$$

The determinant is

$$
\det(A) = (2)(2) - (1)(1) = 3
$$

The characteristic equation becomes

$$
\lambda^2 - 4\lambda + 3 = 0
$$

Solving this quadratic equation gives

$$
\lambda_1 = 3,\quad \lambda_2 = 1
$$

---

## Learn: Calculate Eigenvalues

### What are Eigenvalues?

Eigenvalues are special scalars associated with a square matrix that describe how the matrix transforms vectors.

When a matrix acts on certain vectors, called **eigenvectors**, their direction remains unchanged. Only their magnitude is scaled.

If

$$
A
$$

is a square matrix and

$$
v \ne 0
$$

is an eigenvector, then

$$
Av = \lambda v
$$

where

- $A$ is the square matrix.
- $v$ is a non-zero eigenvector.
- $\lambda$ is the corresponding eigenvalue.

The matrix transforms the vector by scaling it rather than changing its direction.

---

### Mathematical Definition

For a square matrix

$$
A \in \mathbb{R}^{n \times n}
$$

the eigenvalues satisfy

$$
Av = \lambda v
$$

Rearranging,

$$
Av - \lambda v = 0
$$

Factoring out the vector,

$$
(A - \lambda I)v = 0
$$

For a non-zero solution,

$$
\det(A - \lambda I) = 0
$$

This equation is known as the **characteristic equation**.

---

### Eigenvalues of a 2 × 2 Matrix

Consider

$$
A = \begin{bmatrix}
a & b\\
c & d
\end{bmatrix}
$$

Then

$$
A - \lambda I = \begin{bmatrix}
a-\lambda & b\\
c & d-\lambda
\end{bmatrix}
$$

The characteristic equation is

$$
\det(A-\lambda I) = 0
$$

Expanding the determinant,

$$
(a-\lambda)(d-\lambda)-bc = 0
$$

This simplifies to

$$
\lambda^2-tr(A)\lambda+\det(A)=0
$$

where

$$
tr(A)=a+d
$$

is the **trace**, and

$$
\det(A)=ad-bc
$$

is the **determinant**.

---

### Solving for the Eigenvalues

The characteristic equation is a quadratic equation.

Using the quadratic formula,

$$
\lambda = \frac{tr(A)\pm\sqrt{tr(A)^2-4\det(A)}}{2}
$$

The quantity

$$
\Delta = tr(A)^2-4\det(A)
$$

is called the **discriminant**.

Depending on its value:

- $\Delta > 0$ → Two distinct real eigenvalues.
- $\Delta = 0$ → One repeated real eigenvalue.
- $\Delta < 0$ → Two complex conjugate eigenvalues.

---

### Characteristics / Key Points

- Eigenvalues exist only for square matrices.
- Every eigenvalue has at least one corresponding eigenvector.
- The trace equals the sum of the eigenvalues.

$$
\lambda_1+\lambda_2=tr(A)
$$

- The determinant equals the product of the eigenvalues.

$$
\lambda_1\lambda_2=\det(A)
$$

- Symmetric matrices always have real eigenvalues.
- Complex eigenvalues occur in conjugate pairs for real-valued matrices.

---

### Geometric Interpretation

A linear transformation generally changes both the magnitude and direction of a vector.

However, eigenvectors are special because their direction remains unchanged.

If

$$
Av=\lambda v
$$

then

- $\lambda>1$ stretches the vector.
- $0<\lambda<1$ shrinks the vector.
- $\lambda<0$ reverses its direction.
- $\lambda=0$ collapses it to the origin.

---

### Why are Eigenvalues Important?

Eigenvalues appear throughout mathematics, engineering, and machine learning.

Some important applications include:

- Principal Component Analysis (PCA)
- Spectral Clustering
- Dimensionality Reduction
- Stability Analysis
- Control Systems
- Graph Theory
- Computer Vision
- Vibration Analysis
- Quantum Mechanics
- Google's PageRank algorithm

In PCA, the covariance matrix is decomposed into eigenvalues and eigenvectors. The eigenvalues indicate how much variance each principal component explains.

---

> 💡 **Important Note**
>
> Eigenvalues describe **how much** a transformation scales vectors, while eigenvectors describe **which directions** remain unchanged. A common mistake is thinking every vector is scaled by an eigenvalue—only eigenvectors satisfy the equation $Av=\lambda v$.

---

## Solutions

### Custom Implementation

```python
import math
import cmath

def calculate_eigenvalues(matrix: list[list[float | int]]) -> list[complex]:
    if len(matrix[0]) != 2:
        raise ValueError("Only 2x2 matrices are supported")

    trace = matrix[0][0] + matrix[1][1]
    trace = -trace

    determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    disc = trace ** 2 - 4 * determinant

    if disc >= 0:
        sqrt_disc = math.sqrt(disc)
    else:
        sqrt_disc = cmath.sqrt(disc)

    root1 = (-trace + sqrt_disc) / 2
    root2 = (-trace - sqrt_disc) / 2

    return sorted([root1, root2], reverse=True)
```

---

## Code Explanation

### 1. Validate the Matrix Size

```python
if len(matrix[0]) != 2:
    raise ValueError(...)
```

The implementation is derived specifically for a **2 × 2** matrix.

---

### 2. Compute the Trace

```python
trace = matrix[0][0] + matrix[1][1]
```

The trace is

$$
tr(A)=a+d
$$

The code stores its negative because the quadratic equation is written as

$$
x^2+bx+c=0
$$

where

$$
b=-tr(A)
$$

---

### 3. Compute the Determinant

```python
determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
```

This evaluates

$$
\det(A)=ad-bc
$$

---

### 4. Compute the Discriminant

```python
disc = trace ** 2 - 4 * determinant
```

This corresponds to

$$
\Delta=b^2-4ac
$$

If the discriminant is negative, the eigenvalues are complex.

---

### 5. Compute the Square Root

```python
math.sqrt(disc)
```

is used for real roots, while

```python
cmath.sqrt(disc)
```

handles complex roots.

---

### 6. Apply the Quadratic Formula

The two eigenvalues are computed using

$$
\lambda=\frac{-b\pm\sqrt{b^2-4ac}}{2a}
$$

Since

- $a=1$
- $b=-tr(A)$
- $c=\det(A)$

the roots are the required eigenvalues.

---

### 7. Sort the Eigenvalues

```python
return sorted([root1, root2], reverse=True)
```

The problem requires the eigenvalues to be returned from **largest to smallest**.

---

## Time & Space Complexity

Since the matrix size is fixed (**2 × 2**), every operation takes constant time.

| Complexity | Value    |
| ---------- | -------- |
| Time       | **O(1)** |
| Space      | **O(1)** |

The algorithm performs a fixed number of arithmetic operations regardless of the input values.
