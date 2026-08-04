# Matrix × Matrix Multiplication (Medium, Linear Algebra)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Matrix Multiplication](#learn-matrix-multiplication)
- [Things to Note](#things-to-note)
- [Solutions](#solutions)
  - [NumPy Implementation](#numpy-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

---

# Problem Statement

[Matrix times Matrix](https://www.deep-ml.com/problems/9)

Write a Python function `matrixmul(a, b)` that multiplies two matrices.

If the matrices are incompatible for multiplication, return `-1`.

---

# Example

```python
A = [
    [1, 2],
    [2, 4]
]

B = [
    [2, 1],
    [3, 4]
]

print(matrixmul(A, B))
```

### Output

```text
[
    [8, 9],
    [16, 18]
]
```

### Explanation

The first element is

$$
1\times2+2\times3=8
$$

The second element is

$$
1\times1+2\times4=9
$$

Similarly,

$$
2\times2+4\times3=16
$$

and

$$
2\times1+4\times4=18
$$

---

# Learn: Matrix Multiplication

## What is Matrix Multiplication?

Matrix multiplication combines two matrices to produce a new matrix.

Unlike element-wise multiplication, each element of the result is obtained by taking the **dot product** of a row from the first matrix with a column from the second matrix.

---

## Condition for Matrix Multiplication

Suppose

$$
A\in\mathbb{R}^{m\times n}
$$

and

$$
B\in\mathbb{R}^{n\times p}
$$

Then

$$
AB\in\mathbb{R}^{m\times p}
$$

Matrix multiplication is possible **only if**

$$
\text{Columns of }A=\text{Rows of }B
$$

or equivalently,

$$
n=n
$$

If this condition is not satisfied, the multiplication is undefined.

---

## Mathematical Formula

Each element of the resulting matrix is computed as

$$
C_{ij}
=
\sum_{k=1}^{n}
A_{ik}B_{kj}
$$

where

- $A_{ik}$ is the element from row $i$ and column $k$ of matrix $A$.
- $B_{kj}$ is the element from row $k$ and column $j$ of matrix $B$.
- $C_{ij}$ is the element at row $i$ and column $j$ of the resulting matrix.

---

## Example

Let

$$
A=
\begin{bmatrix}
1&2\\
2&4
\end{bmatrix}
$$

and

$$
B=
\begin{bmatrix}
2&1\\
3&4
\end{bmatrix}
$$

The product is

$$
AB=
\begin{bmatrix}
1\times2+2\times3 & 1\times1+2\times4\\
2\times2+4\times3 & 2\times1+4\times4
\end{bmatrix}
=
\begin{bmatrix}
8&9\\
16&18
\end{bmatrix}
$$

---

## Why Matrix Multiplication Matters

Matrix multiplication is one of the most fundamental operations in Linear Algebra.

Applications include:

- Linear Transformations
- Neural Networks
- Computer Graphics
- Robotics
- Principal Component Analysis (PCA)
- Recommendation Systems
- Scientific Computing

---

# Things to Note

- Matrix multiplication is **not commutative**.

$$
AB\neq BA
$$

in general.

- The number of columns of the first matrix must equal the number of rows of the second matrix.
- The resulting matrix has dimensions

$$
m\times p
$$

if

$$
A\in\mathbb{R}^{m\times n}
$$

and

$$
B\in\mathbb{R}^{n\times p}
$$

- NumPy's `np.dot()` performs matrix multiplication for two-dimensional arrays.
- If the dimensions are incompatible, NumPy raises a `ValueError`.

---

# Solutions

## NumPy Implementation

```python
import numpy as np

def matrixmul(a: list[list[int | float]],
              b: list[list[int | float]]) -> list[list[int | float]]:
    try:
        return np.dot(np.asarray(a), np.asarray(b)).tolist()
    except ValueError:
        return -1
```

---

# Code Explanation

## Step 1: Convert Inputs into NumPy Arrays

```python
np.asarray(a)
np.asarray(b)
```

Converts the input Python lists into NumPy arrays.

This allows NumPy to perform optimized matrix operations.

---

## Step 2: Perform Matrix Multiplication

```python
np.dot(np.asarray(a), np.asarray(b))
```

Computes

$$
C=AB
$$

where every element is calculated using

$$
C_{ij}
=
\sum_{k=1}^{n}
A_{ik}B_{kj}
$$

NumPy automatically verifies whether the matrices satisfy the multiplication condition.

---

## Step 3: Convert the Result Back to a Python List

```python
.tolist()
```

Converts the NumPy array into a nested Python list so that the function returns the required output format.

---

## Step 4: Handle Invalid Dimensions

```python
except ValueError:
    return -1
```

If the matrices cannot be multiplied because

$$
\text{Columns of }A
\neq
\text{Rows of }B
$$

NumPy raises a `ValueError`.

The function catches this exception and returns `-1`.

---

# Time & Space Complexity

Suppose

- $A$ has dimensions $m\times n$.
- $B$ has dimensions $n\times p$.

| Complexity | Value |
|------------|-------|
| Time | **O(mnp)** |
| Space | **O(mp)** |

Matrix multiplication computes every element of the output matrix, and each element requires $n$ multiplications and additions.