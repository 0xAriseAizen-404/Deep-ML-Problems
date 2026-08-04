# Scalar Multiplication of a Matrix (Easy, Linear Algebra)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Scalar Multiplication of a Matrix](#learn-scalar-multiplication-of-a-matrix)
- [Things to Note](#things-to-note)
- [Solutions](#solutions)
  - [Custom Implementation](#custom-implementation)
  - [NumPy Implementation](#numpy-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

---

# Problem Statement

[Scalar Multiplication of a Matrix](https://www.deep-ml.com/problems/5)

Write a Python function `scalar_multiply(matrix, scalar)` that performs scalar multiplication on a matrix.

The function should multiply every element of the matrix by the given scalar and return the resulting matrix.

---

# Example

```python
matrix = [
    [1, 2],
    [3, 4]
]

scalar = 2

print(scalar_multiply(matrix, scalar))
```

### Output

```text
[
    [2, 4],
    [6, 8]
]
```

### Explanation

Each element of the matrix is multiplied by the scalar value.

$$
2
\begin{bmatrix}
1&2\\
3&4
\end{bmatrix}
=
\begin{bmatrix}
2&4\\
6&8
\end{bmatrix}
$$

---

# Learn: Scalar Multiplication of a Matrix

## What is Scalar Multiplication?

A **scalar** is a single numerical value.

Examples include

$$
2,\;-3,\;0.5,\;\pi
$$

When a matrix is multiplied by a scalar, **every element** of the matrix is multiplied by that scalar.

The dimensions of the matrix never change.

Only the values of its elements change.

---

## Mathematical Definition

Suppose

$$
A=
\begin{bmatrix}
a_{11}&a_{12}&\cdots&a_{1n}\\
a_{21}&a_{22}&\cdots&a_{2n}\\
\vdots&\vdots&\ddots&\vdots\\
a_{m1}&a_{m2}&\cdots&a_{mn}
\end{bmatrix}
$$

and let the scalar be $k$.

Then

$$
kA=
\begin{bmatrix}
ka_{11}&ka_{12}&\cdots&ka_{1n}\\
ka_{21}&ka_{22}&\cdots&ka_{2n}\\
\vdots&\vdots&\ddots&\vdots\\
ka_{m1}&ka_{m2}&\cdots&ka_{mn}
\end{bmatrix}
$$

Every element is multiplied independently by the scalar.

---

## Example

Let

$$
A=
\begin{bmatrix}
1&2\\
3&4
\end{bmatrix}
$$

Multiply by

$$
k=3
$$

Then

$$
3A=
\begin{bmatrix}
3&6\\
9&12
\end{bmatrix}
$$

Notice that the matrix shape remains

$$
2\times2
$$

Only the numerical values change.

---

## Properties of Scalar Multiplication

For scalars $a$ and $b$, and matrices $A$ and $B$,

### Identity Property

Multiplying by $1$ leaves the matrix unchanged.

$$
1A=A
$$

---

### Zero Property

Multiplying by $0$ produces the zero matrix.

$$
0A=0
$$

---

### Distributive Property over Matrix Addition

$$
k(A+B)=kA+kB
$$

---

### Distributive Property over Scalar Addition

$$
(a+b)A=aA+bA
$$

---

### Associative Property

$$
a(bA)=(ab)A
$$

---

## Applications

Scalar multiplication is widely used in:

- Feature scaling
- Image brightness adjustment
- Linear transformations
- Neural network weight updates
- Physics (scaling vectors and matrices)
- Computer graphics

---

# Things to Note

- Every matrix element is multiplied independently.
- The dimensions of the matrix never change.
- Multiplying by $1$ leaves the matrix unchanged.
- Multiplying by $0$ produces a zero matrix.
- Negative scalars reverse the sign of every element.

---

# Solutions

## Custom Implementation

```python
def scalar_multiply(matrix: list[list[int | float]], scalar: int | float) -> list[list[int | float]]:
    l = []

    for row in matrix:
        l.append([value * scalar for value in row])

    return l
```

---

## NumPy Implementation

```python
import numpy as np

def scalar_multiply(matrix: list[list[int | float]], scalar: int | float):
    return np.multiply(matrix, scalar)
```

---

# Code Explanation

## Custom Implementation

### Step 1: Create the Result Matrix

```python
l = []
```

A new matrix is created to store the scaled values.

---

### Step 2: Iterate Through Every Row

```python
for row in matrix:
```

The loop visits each row of the matrix one at a time.

---

### Step 3: Multiply Every Element

```python
[value * scalar for value in row]
```

The list comprehension multiplies every element of the current row by the scalar.

Mathematically,

$$
ka_{ij}
$$

is computed for every matrix element $a_{ij}$.

---

### Step 4: Store the New Row

```python
l.append(...)
```

The scaled row is appended to the result matrix.

---

### Step 5: Return the Result

```python
return l
```

The function returns the scalar-multiplied matrix.

---

## NumPy Implementation

### Convert and Multiply

```python
np.multiply(matrix, scalar)
```

NumPy performs **element-wise multiplication**.

Each element

$$
a_{ij}
$$

is transformed into

$$
ka_{ij}
$$

using optimized vectorized operations.

---

# Time & Space Complexity

Assume the matrix contains

- $m$ rows.
- $n$ columns.

| Complexity | Value |
|------------|-------|
| Time | **O(mn)** |
| Space | **O(mn)** |

Every element is visited exactly once, and a new matrix of the same size is returned.