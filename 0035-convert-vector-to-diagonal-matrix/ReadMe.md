# Convert Vector to Diagonal Matrix (Easy, Linear Algebra)

## Table of Contents

- Problem Statement
- Example
- Learn: Understanding Diagonal Matrices
- Solution
- Code Explanation
- Time & Space Complexity

---

## Problem Statement

### [Convert Vector to Diagonal Matrix](https://www.deep-ml.com/problems/35)

Write a Python function that converts a one-dimensional NumPy array into a diagonal matrix.

The function should:

- Accept a 1-D NumPy array `x`.
- Create a square matrix whose main diagonal contains the elements of `x`.
- Fill all remaining positions with zeros.
- Return the resulting 2-D NumPy array.

---

## Example

### Input

```python
x = np.array([1, 2, 3])

output = make_diagonal(x)
print(output)
```

### Output

```python
[
    [1., 0., 0.],
    [0., 2., 0.],
    [0., 0., 3.]
]
```

### Reasoning

The vector

```text
[1, 2, 3]
```

contains three elements, so the resulting matrix has size

$$
3 \times 3
$$

Each element is placed along the **main diagonal**, while every off-diagonal element is assigned the value `0`.

---

## Learn: Understanding Diagonal Matrices

### What is it?

A diagonal matrix is a **square matrix** in which every element outside the main diagonal is zero.

The **main diagonal** consists of the entries extending from the top-left corner to the bottom-right corner of the matrix.

Only the diagonal elements may contain non-zero values.

Diagonal matrices are among the simplest and most important matrices in linear algebra because many mathematical operations become much easier when a matrix is diagonal.

---

### Mathematical Definition

Given a vector

$$
x = [x_1,\;x_2,\;\ldots,\;x_n]
$$

its corresponding diagonal matrix is

$$
D =
\begin{bmatrix}
x_1 & 0 & 0 & \cdots & 0\\
0 & x_2 & 0 & \cdots & 0\\
0 & 0 & x_3 & \cdots & 0\\
\vdots & \vdots & \vdots & \ddots & \vdots\\
0 & 0 & 0 & \cdots & x_n
\end{bmatrix}
$$

The entries satisfy

$$
D_{ij} =
\begin{cases}
x_i,& i=j\\
0,& i\ne j
\end{cases}
$$

where

- $i$ is the row index.
- $j$ is the column index.

---

### How a Diagonal Matrix is Constructed

The algorithm performs the following steps:

1. Determine the length of the vector.
2. Create an $n \times n$ matrix filled with zeros.
3. Copy each vector element onto the main diagonal.
4. Return the completed matrix.

For example,

Input vector

$$
[4,\;7,\;2]
$$

Resulting matrix

$$
\begin{bmatrix}
4 & 0 & 0\\
0 & 7 & 0\\
0 & 0 & 2
\end{bmatrix}
$$

---

### Properties of Diagonal Matrices

Diagonal matrices have several useful mathematical properties.

#### Matrix Addition

Two diagonal matrices can be added by adding their corresponding diagonal elements.

$$
\begin{bmatrix}
1 & 0\\
0 & 2
\end{bmatrix}
+
\begin{bmatrix}
3 & 0\\
0 & 4
\end{bmatrix}
=
\begin{bmatrix}
4 & 0\\
0 & 6
\end{bmatrix}
$$

---

#### Matrix Multiplication

Multiplying two diagonal matrices multiplies only the diagonal entries.

$$
\begin{bmatrix}
2 & 0\\
0 & 3
\end{bmatrix}
\begin{bmatrix}
4 & 0\\
0 & 5
\end{bmatrix}
=
\begin{bmatrix}
8 & 0\\
0 & 15
\end{bmatrix}
$$

---

#### Matrix Inverse

A diagonal matrix is invertible only if every diagonal element is non-zero.

Its inverse is

$$
D^{-1} =
\begin{bmatrix}
\frac{1}{x_1} & 0 & \cdots & 0\\
0 & \frac{1}{x_2} & \cdots & 0\\
\vdots & \vdots & \ddots & \vdots\\
0 & 0 & \cdots & \frac{1}{x_n}
\end{bmatrix}
$$

---

### Characteristics / Key Points

- Always a square matrix.
- All off-diagonal elements are zero.
- The main diagonal stores the important values.
- Easy to construct from a vector.
- Requires only one value per row and column.
- Matrix multiplication and inversion are computationally efficient.
- Widely used in linear algebra and machine learning.

---

### Why is it used? / Applications

Diagonal matrices appear throughout mathematics, machine learning, and scientific computing.

Applications include:

- Covariance matrices (special cases)
- Scaling transformations
- Eigenvalue decomposition
- Singular Value Decomposition (SVD)
- Principal Component Analysis (PCA)
- Neural network weight initialization
- Matrix factorization
- Numerical optimization
- Scientific simulations

Because only the diagonal contains non-zero values, many computations become significantly faster than with general matrices.

---

> 💡 **Important Note**
>
> A diagonal matrix is **not** the same as an identity matrix. An identity matrix is a special diagonal matrix where every diagonal element equals **1**. Every identity matrix is diagonal, but not every diagonal matrix is an identity matrix.

---

## Solution

### Custom Implementation

```python
import numpy as np

def make_diagonal(x):
    sz = len(x)

    mat = [[0 for _ in range(sz)] for _ in range(sz)]

    for ind in range(sz):
        mat[ind][ind] = x[ind]

    return np.array(mat)
```

### NumPy Implementation

```python
import numpy as np

def make_diagonal(x):
    return np.diag(x)
```

---

## Code Explanation

### Step 1

Determine the size of the input vector.

```python
sz = len(x)
```

This determines the dimensions of the output matrix.

---

### Step 2

Create a square matrix filled with zeros.

```python
mat = [[0 for _ in range(sz)] for _ in range(sz)]
```

Initially, every element is zero.

---

### Step 3

Fill the main diagonal.

```python
for ind in range(sz):
    mat[ind][ind] = x[ind]
```

For every position `(i, i)`, place the corresponding vector element.

All off-diagonal elements remain zero.

---

### Step 4

Convert the matrix into a NumPy array.

```python
return np.array(mat)
```

The function returns the completed diagonal matrix.

---

## Time & Space Complexity

| Complexity | Value     |
| ---------- | --------- |
| Time       | **O(n²)** |
| Space      | **O(n²)** |

where

- $n$ is the number of elements in the input vector.

Although only $n$ diagonal elements are assigned, an entire $n \times n$ matrix must be created, resulting in quadratic time and space complexity. The built-in `np.diag()` function also produces an output matrix of the same size.
