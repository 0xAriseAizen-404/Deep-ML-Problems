# Matrix times Matrix (Medium, Linear Algebra)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Matrix Multiplication](#learn-matrix-multiplication)
- [Solutions](#solutions)
  - [Custom Implementation](#custom-implementation)
  - [NumPy Implementation](#numpy-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

## Problem Statement

### [Matrix times Matrix](https://www.deep-ml.com/problems/9)

Write a Python function that multiplies two matrices.

Given matrices **A** and **B**, compute

$$
C = AB
$$

If the number of columns in **A** does not equal the number of rows in **B**, matrix multiplication is undefined and the function should return `-1`.

---

## Example

### Input

```python
A = [
    [1, 2],
    [2, 4]
]

B = [
    [2, 1],
    [3, 4]
]
```

### Output

```python
[
    [8, 9],
    [16, 18]
]
```

### Reasoning

Each element is computed as the dot product of one row of **A** with one column of **B**.

$$
C_{11} = (1)(2) + (2)(3) = 8
$$

$$
C_{12} = (1)(1) + (2)(4) = 9
$$

$$
C_{21} = (2)(2) + (4)(3) = 16
$$

$$
C_{22} = (2)(1) + (4)(4) = 18
$$

Therefore,

$$
AB =
\begin{bmatrix}
8 & 9\\
16 & 18
\end{bmatrix}
$$

---

## Learn: Matrix Multiplication

### What is Matrix Multiplication?

Matrix multiplication combines two matrices to produce a new matrix.

Unlike element-wise multiplication, each element of the result is obtained by taking the **dot product** of a row from the first matrix with a column from the second matrix.

Matrix multiplication is one of the most fundamental operations in Linear Algebra and forms the mathematical foundation of Machine Learning, Deep Learning, Computer Graphics, Robotics, and Scientific Computing.

---

### Dimension Requirement

Suppose

$$
A \in \mathbb{R}^{m \times n}
$$

and

$$
B \in \mathbb{R}^{n \times p}
$$

Then

$$
AB \in \mathbb{R}^{m \times p}
$$

Matrix multiplication is valid only when

$$
\text{Columns of }A = \text{Rows of }B
$$

or equivalently,

$$
n = n
$$

If the dimensions do not satisfy this condition, multiplication is undefined.

---

### Mathematical Definition

The element at row $i$ and column $j$ of the product matrix is

$$
C_{ij} = \sum_{k=1}^{n} A_{ik}B_{kj}
$$

Each output element is the dot product between:

- Row $i$ of matrix $A$
- Column $j$ of matrix $B$

---

### Example

Consider

$$
A =
\begin{bmatrix}
1 & 2\\
2 & 4
\end{bmatrix}
$$

and

$$
B =
\begin{bmatrix}
2 & 1\\
3 & 4
\end{bmatrix}
$$

Then

$$
AB =
\begin{bmatrix}
(1)(2)+(2)(3) & (1)(1)+(2)(4)\\
(2)(2)+(4)(3) & (2)(1)+(4)(4)
\end{bmatrix} =
\begin{bmatrix}
8 & 9\\
16 & 18
\end{bmatrix}
$$

---

### Step-by-Step Algorithm

To compute matrix multiplication:

1. Verify that the dimensions are compatible.
2. Select one row from matrix **A**.
3. Select one column from matrix **B**.
4. Compute their dot product.
5. Store the result in the corresponding position.
6. Repeat for every row and every column.

---

### Characteristics / Key Points

- Requires compatible matrix dimensions.
- Uses row-column dot products.
- Produces a new matrix.
- Matrix multiplication is **associative**.

$$
(AB)C = A(BC)
$$

- Matrix multiplication is **distributive**.

$$
A(B+C) = AB + AC
$$

- Matrix multiplication is generally **not commutative**.

$$
AB \ne BA
$$

---

### Why is it Used?

Matrix multiplication is one of the most frequently used operations in Machine Learning and Deep Learning.

Some applications include:

- Neural network forward propagation
- Linear Regression
- Logistic Regression
- Principal Component Analysis (PCA)
- Computer Graphics transformations
- Robotics
- Signal processing
- Scientific simulations
- Graph algorithms
- Numerical optimization

For example, a dense neural network layer computes

$$
y = Wx + b
$$

where

- $W$ is the weight matrix.
- $x$ is the input vector.
- $b$ is the bias vector.

The primary computation is matrix multiplication.

---

> 💡 **Important Note**
>
> Matrix multiplication is **not** element-wise multiplication. Another common mistake is assuming that $AB = BA$. In general, changing the multiplication order either produces a different result or may even be mathematically invalid due to incompatible dimensions.

---

## Solutions

### Custom Implementation

```python
def matrixmul(a: list[list[int | float]],
              b: list[list[int | float]]) -> list[list[int | float]]:

    if len(a[0]) != len(b):
        return -1

    res = []

    for row in a:
        new_row = []

        for col in range(len(b[0])):
            value = sum(row[k] * b[k][col] for k in range(len(b)))
            new_row.append(value)

        res.append(new_row)

    return res
```

### NumPy Implementation

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

## Code Explanation

### 1. Validate the Matrix Dimensions

```python
if len(a[0]) != len(b):
    return -1
```

The number of columns in matrix **A** must equal the number of rows in matrix **B**.

Otherwise, multiplication cannot be performed.

---

### 2. Iterate Through Every Row

```python
for row in a:
```

Each row of **A** contributes one row to the resulting matrix.

---

### 3. Iterate Through Every Column

```python
for col in range(len(b[0])):
```

Each column of **B** contributes one column to the output.

---

### 4. Compute the Dot Product

```python
value = sum(row[k] * b[k][col] for k in range(len(b)))
```

This implements

$$
C_{ij} = \sum_{k=1}^{n} A_{ik}B_{kj}
$$

The row of **A** and the column of **B** are multiplied element by element, and the products are summed.

---

### 5. Store the Computed Value

```python
new_row.append(value)
```

Each computed dot product becomes one element of the output row.

---

### 6. Return the Result

After every row-column pair has been processed, the completed matrix is returned.

The NumPy implementation performs the same computation using

```python
np.dot()
```

which is highly optimized and internally uses efficient numerical libraries.

---

## Time & Space Complexity

Let

- $m$ = number of rows in **A**
- $n$ = number of columns in **A** (rows in **B**)
- $p$ = number of columns in **B**

The algorithm computes one dot product for every element of the output matrix.

| Complexity | Value            |
| ---------- | ---------------- |
| Time       | **O(m × n × p)** |
| Space      | **O(m × p)**     |

The additional space is required to store the resulting matrix of dimensions $m \times p$.
