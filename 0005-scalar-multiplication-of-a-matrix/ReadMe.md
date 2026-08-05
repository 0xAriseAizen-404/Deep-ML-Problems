# Scalar Multiplication of a Matrix (Easy, Linear Algebra)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Scalar Multiplication of a Matrix](#learn-scalar-multiplication-of-a-matrix)
- [Solutions](#solutions)
  - [Custom Implementation](#custom-implementation)
  - [NumPy Implementation](#numpy-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

## Problem Statement

### [Scalar Multiplication of a Matrix](https://www.deep-ml.com/problems/5)

Write a Python function that multiplies every element of a matrix by a given scalar and returns the resulting matrix.

The dimensions of the matrix remain unchanged; only the values are scaled.

---

## Example

### Input

```python
matrix = [
    [1, 2],
    [3, 4]
]

scalar = 2
```

### Output

```python
[
    [2, 4],
    [6, 8]
]
```

### Reasoning

Each element is multiplied by the scalar value.

$$
2 \times
\begin{bmatrix}
1 & 2\\
3 & 4
\end{bmatrix}
=
\begin{bmatrix}
2 & 4\\
6 & 8
\end{bmatrix}
$$

---

## Learn: Scalar Multiplication of a Matrix

### What is it?

Scalar multiplication is one of the simplest matrix operations. It multiplies **every element** of a matrix by a single numerical value called a **scalar**.

The operation changes the magnitude of every matrix element while preserving:

- The matrix dimensions
- The arrangement of elements
- The relative proportions between elements

If the scalar is greater than 1, the matrix is enlarged. If it is between 0 and 1, the matrix is compressed. A negative scalar additionally reverses the sign of every element.

---

### Mathematical Definition

Let

$$
A \in \mathbb{R}^{m \times n}
$$

be a matrix and

$$
k \in \mathbb{R}
$$

be a scalar.

The scalar multiplication is defined as

$$
kA=
\begin{bmatrix}
ka_{11} & ka_{12} & \cdots & ka_{1n}\\
ka_{21} & ka_{22} & \cdots & ka_{2n}\\
\vdots & \vdots & \ddots & \vdots\\
ka_{m1} & ka_{m2} & \cdots & ka_{mn}
\end{bmatrix}
$$

Equivalently,

$$
(kA)_{ij}=k \cdot a_{ij}
$$

Every element is multiplied independently by the same scalar.

---

### Example

Original matrix

$$
A=
\begin{bmatrix}
1 & 2\\
3 & 4
\end{bmatrix}
$$

Multiply by

$$
k=3
$$

Result

$$
3A=
\begin{bmatrix}
3 & 6\\
9 & 12
\end{bmatrix}
$$

Only the values change; the matrix remains **2 × 2**.

---

### Geometric Interpretation

A matrix can be viewed as a collection of vectors.

Multiplying by a scalar scales every vector uniformly.

- $k>1$ stretches the vectors.
- $0<k<1$ shrinks the vectors.
- $k=1$ leaves the matrix unchanged.
- $k=0$ converts every element into zero.
- $k<0$ scales the matrix and reverses the direction of every vector.

---

### Properties of Scalar Multiplication

#### Identity

Multiplying by 1 leaves the matrix unchanged.

$$
1A=A
$$

---

#### Zero Scalar

Multiplying by zero produces the zero matrix.

$$
0A=0
$$

---

#### Associativity

$$
(ab)A=a(bA)
$$

---

#### Distributive over Matrix Addition

$$
k(A+B)=kA+kB
$$

---

#### Distributive over Scalar Addition

$$
(a+b)A=aA+bA
$$

---

### Characteristics / Key Points

- Every element is multiplied independently.
- Matrix dimensions never change.
- The relative arrangement of elements remains unchanged.
- Negative scalars change the sign of every element.
- Zero produces the zero matrix.
- Computationally simple and highly parallelizable.

---

### Why is it Used?

Scalar multiplication is frequently used in Linear Algebra and Machine Learning.

Some applications include:

- Feature scaling
- Gradient updates during optimization
- Weight initialization
- Image brightness adjustment
- Data normalization
- Physics simulations
- Computer graphics transformations
- Scientific computing

For example, in Gradient Descent, model parameters are updated using

$$
W_{new}=W-\eta\nabla J(W)
$$

where

- $W$ is the weight matrix.
- $\eta$ is the learning rate (a scalar).
- $\nabla J(W)$ is the gradient matrix.

The learning rate scales the entire gradient matrix before the update.

---

> 💡 **Important Note**
>
> Scalar multiplication affects only the **values** of a matrix—not its shape. Unlike matrix multiplication, there are no dimension compatibility rules because every element is processed independently.

---

## Solutions

### Custom Implementation

```python
def scalar_multiply(matrix: list[list[int | float]], scalar: int | float) -> list[list[int | float]]:
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            matrix[i][j] *= scalar

    return matrix
```

### NumPy Implementation

```python
import numpy as np

def scalar_multiply(matrix: list[list[int | float]], scalar: int | float) -> list[list[int | float]]:
    return (np.asarray(matrix) * scalar).tolist()
```

---

## Code Explanation

### 1. Convert the Matrix to a NumPy Array

```python
np.asarray(matrix)
```

The input list is converted into a NumPy array to enable efficient element-wise operations.

---

### 2. Multiply by the Scalar

```python
np.asarray(matrix) * scalar
```

NumPy automatically multiplies every element by the given scalar.

Mathematically,

$$
(kA)_{ij}=k \times a_{ij}
$$

No explicit loops are required because NumPy performs vectorized computation internally.

---

### 3. Convert Back to a Python List

```python
.tolist()
```

The resulting NumPy array is converted back into a standard Python list before returning.

---

## Time & Space Complexity

Let

- $m$ = number of rows
- $n$ = number of columns

Every element is multiplied exactly once.

| Complexity | Value |
| ---------- | ----- |
| Time | **O(m × n)** |
| Space | **O(m × n)** |

The returned matrix contains the same number of elements as the original matrix.