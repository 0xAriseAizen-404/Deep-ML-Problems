# Transpose of a Matrix (Easy, Linear Algebra)

## Table of Contents

- Problem Statement
- Example
- Learn: Transpose of a Matrix
- Solutions
  - Custom Implementation
- Code Explanation
- Time & Space Complexity

---

## Problem Statement

[Transpose of a Matrix](https://www.deep-ml.com/problems/2)

Write a Python function `transpose_matrix(a)` that computes the **transpose** of a given matrix.

The transpose of a matrix is obtained by **interchanging its rows and columns**.

---

## Example

```python
a = [
    [1, 2, 3],
    [4, 5, 6]
]

print(transpose_matrix(a))
```

### Output

```text
[
    [1, 4],
    [2, 5],
    [3, 6]
]
```

### Explanation

Given

$$
A=
\begin{bmatrix}
1&2&3\\
4&5&6
\end{bmatrix}
$$

its transpose is

$$
A^T=
\begin{bmatrix}
1&4\\
2&5\\
3&6
\end{bmatrix}
$$

Rows become columns and columns become rows.

---

# Learn: Transpose of a Matrix

## What is it?

The **transpose** of a matrix is formed by swapping its rows and columns.

If an element is located at row $i$ and column $j$ in the original matrix, it moves to row $j$ and column $i$ in the transposed matrix.

The transpose is one of the most fundamental operations in Linear Algebra and appears frequently in Machine Learning, Deep Learning, Computer Vision, and Statistics.

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

Its transpose is

$$
A^T=
\begin{bmatrix}
a_{11}&a_{21}&\cdots&a_{m1}\\
a_{12}&a_{22}&\cdots&a_{m2}\\
\vdots&\vdots&\ddots&\vdots\\
a_{1n}&a_{2n}&\cdots&a_{mn}
\end{bmatrix}
$$

In general,

$$
(A^T)_{ij}
=
A_{ji}
$$

---

## Characteristics / Key Points

- Rows become columns.
- Columns become rows.
- If $A$ is an $m\times n$ matrix, then $A^T$ is an $n\times m$ matrix.
- Applying transpose twice returns the original matrix.

$$
(A^T)^T=A
$$

- The transpose of a square matrix has the same dimensions.

---

## Why is it used? / Applications

Matrix transpose is widely used in:

- Matrix Multiplication
- Linear Regression
- Principal Component Analysis (PCA)
- Covariance Matrix Computation
- Singular Value Decomposition (SVD)
- Neural Networks
- Computer Graphics
- Scientific Computing

Many machine learning algorithms rely on transposed matrices during optimization and gradient computation.

> 💡 **Important Note**
>
> The provided implementation supports **ragged matrices** (rows with different lengths). Missing elements are skipped instead of causing an error, making it more flexible than the standard transpose operation, which assumes every row has the same number of columns.

---

# Solution

## Custom Implementation

```python
def transpose_matrix(a: list[list[int | float]]) -> list[list[int | float]]:
    m = max(len(row) for row in a)
    return [[row[i] for row in a if i < len(row)] for i in range(m)]
```

---

# Code Explanation

### Step 1: Find the Maximum Number of Columns

```python
m = max(len(row) for row in a)
```

This computes

$$
m
=
\max_{1\le i\le r}
\left(
\text{length}(A_i)
\right)
$$

where

- $r$ is the number of rows.
- $A_i$ is the $i^{th}$ row of the matrix.

The value $m$ determines how many rows the transposed matrix will contain.

---

### Step 2: Iterate Over Every Column Index

```python
for i in range(m)
```

For each column index,

$$
i=0,1,2,\ldots,m-1
$$

a new row of the transposed matrix is constructed.

---

### Step 3: Collect Elements from Each Row

```python
[row[i] for row in a if i < len(row)]
```

For every row,

the element

$$
A_{ji}
$$

is selected only if the column exists.

Mathematically,

$$
A^T_i
=
\{A_{ji}\mid i<\text{length}(A_j)\}
$$

Rows that do not contain the current column index are ignored.

---

### Step 4: Construct the Transposed Matrix

```python
return [[row[i] for row in a if i < len(row)] for i in range(m)]
```

Each generated row becomes one row of the transpose.

The resulting matrix satisfies

$$
(A^T)_{ij}
=
A_{ji}
$$

for every valid element in the original matrix.

---

## Time & Space Complexity

Let

- $r$ = Number of rows.
- $c$ = Maximum row length.

| Complexity | Value |
| ---------- | ----- |
| Time | **O(r × c)** |
| Space | **O(r × c)** |