# Matrix Times Vector (Easy, Linear Algebra)

## Table of Contents

- Problem Statement
- Example
- Learn: Matrix Times Vector
- Solution
- Code Explanation
- Time & Space Complexity

---

## Problem Statement

[Matrix Times Vector](https://www.deep-ml.com/problems/1)

Write a Python function `matrix_dot_vector(a, b)` that computes the **matrix-vector product** between a matrix and a vector.

Return **-1** if the matrix and vector cannot be multiplied.

---

## Example

```python
a = [
    [1, 2],
    [2, 4]
]

b = [1, 2]

print(matrix_dot_vector(a, b))
```

### Output

```text
[5, 10]
```

### Explanation

The first output element is

$$
1\times1+2\times2=5
$$

The second output element is

$$
2\times1+4\times2=10
$$

Therefore,

$$
Av=
\begin{bmatrix}
5\\
10
\end{bmatrix}
$$

---

# Learn: Matrix Times Vector

## What is it?

Matrix-vector multiplication is one of the most fundamental operations in Linear Algebra.

It applies the linear transformation represented by a matrix to an input vector. Each row of the matrix is multiplied with the vector using the **dot product**, producing one value in the output vector.

If a matrix has $m$ rows, the resulting vector will also contain $m$ elements.

---

## Mathematical Definition

Suppose we have the matrix

$$
A=
\begin{bmatrix}
a_{11}&a_{12}&\cdots&a_{1n}\\
a_{21}&a_{22}&\cdots&a_{2n}\\
\vdots&\vdots&\ddots&\vdots\\
a_{m1}&a_{m2}&\cdots&a_{mn}
\end{bmatrix}
$$

and the vector

$$
v=
\begin{bmatrix}
v_1\\
v_2\\
\vdots\\
v_n
\end{bmatrix}
$$

The matrix-vector product is

$$
Av=
\begin{bmatrix}
\sum_{j=1}^{n}a_{1j}v_j\\
\sum_{j=1}^{n}a_{2j}v_j\\
\vdots\\
\sum_{j=1}^{n}a_{mj}v_j
\end{bmatrix}
$$

Each output element is simply the **dot product** of one row of the matrix with the input vector.

---

## Dimension Requirement

Suppose

- Matrix $A$ has dimensions $m\times n$
- Vector $v$ has length $n$

Then the multiplication is valid.

$$
(m\times n)
\times
(n\times1)
=
(m\times1)
$$

If the number of matrix columns does not equal the vector length, the multiplication is undefined.

---

## Characteristics / Key Points

- Each row produces exactly one output value.
- Every output value is computed using a dot product.
- The output vector has the same number of elements as the number of matrix rows.
- Matrix-vector multiplication represents a linear transformation.
- The number of columns must equal the size of the vector.

---

## Why is it used? / Applications

Matrix-vector multiplication is used throughout Machine Learning and Linear Algebra.

Applications include:

- Linear Regression
- Logistic Regression
- Neural Networks
- Principal Component Analysis (PCA)
- Computer Graphics
- Physics Simulations
- Recommendation Systems
- Scientific Computing

Nearly every neural network layer performs matrix-vector multiplication during forward propagation.

> 💡 **Important Note**
>
> The provided implementation checks `len(a) == len(b)`, which is only correct for square matrices. The mathematically correct condition is:
>
> ```python
> len(a[0]) == len(b)
> ```
>
> since the **number of columns** of the matrix must equal the **length of the vector**.

---

# Solution

## Custom Implementation

```python
def matrix_dot_vector(a, b):
    if len(a) == len(b):
        l = []
        for i in a:
            l.append(sum([j * k for j, k in zip(i, b)]))
        return l
    return -1
```

---

# Code Explanation

### Step 1: Check the Dimensions

```python
if len(a) == len(b):
```

The implementation checks whether

$$
\text{Rows}(A)
=
\text{Length}(v)
$$

Although this works for square matrices, the correct mathematical condition is

$$
\text{Columns}(A)
=
\text{Length}(v)
$$

---

### Step 2: Iterate Through Each Matrix Row

```python
for i in a:
```

Each iteration selects one row

$$
A_i=
\begin{bmatrix}
a_{i1}&a_{i2}&\cdots&a_{in}
\end{bmatrix}
$$

Each row contributes one value to the output vector.

---

### Step 3: Compute the Dot Product

```python
sum([j * k for j, k in zip(i, b)])
```

For the current row,

$$
y_i
=
\sum_{j=1}^{n}
a_{ij}v_j
$$

This computes the dot product between the current row and the input vector.

---

### Step 4: Store the Result

```python
l.append(...)
```

The computed value is appended to the output vector.

After processing every row,

$$
y=
Av
$$

where

$$
y=
\begin{bmatrix}
y_1\\
y_2\\
\vdots\\
y_m
\end{bmatrix}
$$

---

### Step 5: Return the Result

```python
return l
```

The function returns the complete matrix-vector product.

If the dimension check fails,

```python
return -1
```

indicates that the multiplication is not valid.

---

## Time & Space Complexity

Let

- $m$ = Number of rows
- $n$ = Number of columns

| Complexity | Value |
| ---------- | ----- |
| Time | **O(m × n)** |
| Space | **O(m)** |