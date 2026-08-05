# Transpose of a Matrix (Easy, Linear Algebra)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Transpose of a Matrix](#learn-transpose-of-a-matrix)
- [Solutions](#solutions)
  - [Custom Implementation](#custom-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

## Problem Statement

### [Transpose of a Matrix](https://www.deep-ml.com/problems/2)

Write a Python function that computes the **transpose** of a given 2D matrix. The transpose is obtained by converting every row of the original matrix into a column and every column into a row.

For an **m × n** matrix, the transpose has dimensions **n × m**.

---

## Example

### Input

```python
a = [
    [1, 2, 3],
    [4, 5, 6]
]
```

### Output

```python
[
    [1, 4],
    [2, 5],
    [3, 6]
]
```

### Reasoning

The input matrix has **2 rows** and **3 columns**.

Original matrix:

$$
A=
\begin{bmatrix}
1 & 2 & 3\\
4 & 5 & 6
\end{bmatrix}
$$

After swapping rows and columns,

$$
A^T=
\begin{bmatrix}
1 & 4\\
2 & 5\\
3 & 6
\end{bmatrix}
$$

The resulting matrix has **3 rows** and **2 columns**.

---

## Learn: Transpose of a Matrix

### What is it?

The **transpose** of a matrix is formed by interchanging its rows and columns. Every element moves from position **(i, j)** to **(j, i)** without changing its value.

If the original matrix has **m rows** and **n columns**, the transposed matrix will have **n rows** and **m columns**.

Unlike matrix multiplication, the transpose does not perform any arithmetic operations. It simply rearranges the existing elements.

---

### Mathematical Definition

Let

$$
A \in \mathbb{R}^{m \times n}
$$

Then its transpose is

$$
A^T \in \mathbb{R}^{n \times m}
$$

where

$$
(A^T)_{ij}=A_{ji}
$$

If

$$
A=
\begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1n}\\
a_{21} & a_{22} & \cdots & a_{2n}\\
\vdots & \vdots & \ddots & \vdots\\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{bmatrix}
$$

then

$$
A^T=
\begin{bmatrix}
a_{11} & a_{21} & \cdots & a_{m1}\\
a_{12} & a_{22} & \cdots & a_{m2}\\
\vdots & \vdots & \ddots & \vdots\\
a_{1n} & a_{2n} & \cdots & a_{mn}
\end{bmatrix}
$$

---

### Index Transformation

Every element follows a simple rule:

$$
(i,j)\rightarrow(j,i)
$$

For example,

| Original Position | Transposed Position |
|------------------|---------------------|
| $(0,1)$ | $(1,0)$ |
| $(2,3)$ | $(3,2)$ |
| $(4,0)$ | $(0,4)$ |

Only the indices change; the values remain the same.

---

### Properties of Transpose

The transpose operation satisfies several important mathematical properties.

#### Double Transpose

Applying transpose twice returns the original matrix.

$$
(A^T)^T=A
$$

---

#### Addition

The transpose distributes over matrix addition.

$$
(A+B)^T=A^T+B^T
$$

---

#### Scalar Multiplication

A scalar can be factored outside the transpose.

$$
(cA)^T=cA^T
$$

---

#### Matrix Multiplication

When transposing a product, the multiplication order reverses.

$$
(AB)^T=B^TA^T
$$

This is one of the most frequently tested transpose properties.

---

#### Symmetric Matrix

A matrix is symmetric if

$$
A=A^T
$$

Only square matrices can be symmetric.

---

### Characteristics / Key Points

- Swaps rows and columns.
- Matrix dimensions change from **m × n** to **n × m**.
- No arithmetic computation is performed.
- Every element keeps its value.
- Only the element positions change.
- Works for square and rectangular matrices.
- Applying transpose twice returns the original matrix.

---

### Pythonic Implementation

Python provides a concise way to transpose a matrix using `zip()`.

```python
list(map(list, zip(*matrix)))
```

Here,

- `*matrix` unpacks every row as a separate argument.
- `zip()` groups elements with the same column index.
- `map(list, ...)` converts tuples into lists.

This approach is clean and efficient for rectangular matrices.

---

### Why is it Used?

Transpose is a fundamental operation across mathematics and machine learning.

Some common applications include:

- Linear Algebra computations
- Matrix multiplication
- Covariance matrix computation
- Feature matrix transformations
- Data preprocessing
- Principal Component Analysis (PCA)
- Computer Graphics transformation matrices
- Scientific computing
- Neural network weight manipulation

Many machine learning algorithms rely on transposed matrices when deriving gradients and performing matrix multiplications.

---

> 💡 **Important Note**
>
> The transpose only changes the **position** of each element—it never changes the element itself. A common mistake is assuming values are recalculated. Every entry simply moves from index $(i,j)$ to $(j,i)$, causing the matrix dimensions to flip from **m × n** to **n × m**.

---

## Solutions

### Custom Implementation

```python
def transpose_matrix(a: list[list[int | float]]) -> list[list[int | float]]:
    m = max(len(row) for row in a)
    return [[row[i] for row in a if i < len(row)] for i in range(m)]
```

---

## Code Explanation

### 1. Find the Maximum Row Length

```python
m = max(len(row) for row in a)
```

The implementation supports matrices whose rows may have different lengths.

The maximum row length determines how many columns the transposed matrix can have.

---

### 2. Iterate Over Every Column Index

```python
for i in range(m)
```

Each column of the original matrix becomes one row in the transposed matrix.

---

### 3. Collect Elements from Every Row

```python
[row[i] for row in a if i < len(row)]
```

For the current column index:

- Visit every row.
- Check whether the row contains that column.
- Append the element if it exists.

This naturally handles irregular (jagged) matrices without raising an index error.

---

### 4. Construct the Result

The outer list comprehension combines all generated rows into the final transposed matrix.

The overall logic is equivalent to swapping every element from

$$
(i,j)
$$

to

$$
(j,i)
$$

---

## Time & Space Complexity

Let

- $m$ = number of rows
- $n$ = maximum number of columns

Every element is visited exactly once.

| Complexity | Value |
| ---------- | ----- |
| Time | **O(m × n)** |
| Space | **O(m × n)** |

The additional space is required to store the transposed matrix.