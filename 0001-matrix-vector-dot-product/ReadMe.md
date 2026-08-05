# Matrix-Vector Dot Product (Easy, Linear Algebra)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Matrix-Vector Dot Product](#learn-matrix-vector-dot-product)
- [Solutions](#solutions)
  - [Custom Implementation](#custom-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

## Problem Statement

### [Matrix-Vector Dot Product](https://www.deep-ml.com/problems/1)

Given a matrix and a vector, compute their matrix-vector dot product.

A matrix with dimensions **n × m** can only be multiplied by a vector of length **m**. The result is a new vector of length **n**, where each element is obtained by taking the dot product of one row of the matrix with the input vector.

If the number of columns in the matrix does not equal the length of the vector, the multiplication is undefined, and the function should return **-1**.

---

## Example

### Input

```python
a = [[1, 2],
     [2, 4]]

b = [1, 2]
```

### Output

```python
[5, 10]
```

### Reasoning

For the first row:

$$
(1 \times 1) + (2 \times 2) = 5
$$

For the second row:

$$
(2 \times 1) + (4 \times 2) = 10
$$

Therefore,

```python
[5, 10]
```

---

## Learn: Matrix-Vector Dot Product

### What is it?

A **matrix-vector dot product** is one of the most fundamental operations in Linear Algebra. It transforms an input vector into another vector by combining the values in each matrix row with the corresponding elements of the vector.

If the matrix has **n rows** and **m columns**, and the vector has **m elements**, the multiplication produces a vector containing **n elements**.

Each output element is simply the **dot product** between one row of the matrix and the input vector.

This operation is widely used in Machine Learning, Deep Learning, Computer Vision, Scientific Computing, and numerical optimization.

---

### Mathematical Definition

Let

$$
A \in \mathbb{R}^{n \times m}
$$

be a matrix and

$$
v \in \mathbb{R}^{m}
$$

be a vector.

Their product is

$$
A v \in \mathbb{R}^{n}
$$

where

$$
(A v)_i = \sum_{j=1}^{m} a_{ij}v_j
$$

Expanding this,

$$
A=
\begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1m}\\
a_{21} & a_{22} & \cdots & a_{2m}\\
\vdots & \vdots & \ddots & \vdots\\
a_{n1} & a_{n2} & \cdots & a_{nm}
\end{bmatrix}
$$

and

$$
v=
\begin{bmatrix}
v_1\\
v_2\\
\vdots\\
v_m
\end{bmatrix}
$$

Then,

$$
Av=
\begin{bmatrix}
a_{11}v_1+a_{12}v_2+\cdots+a_{1m}v_m\\
a_{21}v_1+a_{22}v_2+\cdots+a_{2m}v_m\\
\vdots\\
a_{n1}v_1+a_{n2}v_2+\cdots+a_{nm}v_m
\end{bmatrix}
$$

---

### Dimension Requirement

For matrix-vector multiplication to be valid,

$$
A_{n \times m}\times v_{m}=y_n
$$

The **number of columns** in the matrix must equal the **length of the vector**.

In other words,

$$
m = |v|
$$

If

$$
m \ne |v|
$$

the multiplication is undefined.

---

### Geometric Interpretation

A matrix can be viewed as a transformation.

Instead of simply scaling a vector, the matrix can:

- Rotate it
- Stretch it
- Compress it
- Reflect it
- Project it into another space

The resulting vector represents the transformed version of the original vector.

---

### Characteristics / Key Points

- Matrix dimensions are **n × m**.
- Vector length must be **m**.
- Output vector length is **n**.
- Each output element is computed independently.
- Every row contributes exactly one value to the output.
- Matrix-vector multiplication is **not** element-wise multiplication.
- The operation is deterministic and linear.

---

### Step-by-Step Algorithm

Suppose

```text
Matrix:
[
 [1,2],
 [2,4]
]

Vector:
[1,2]
```

For each row:

Row 1

$$
1\times1+2\times2=5
$$

Row 2

$$
2\times1+4\times2=10
$$

Output

```text
[5,10]
```

---

### Relationship to the Dot Product

The dot product between two vectors

$$
x=[x_1,x_2,\dots,x_m]
$$

and

$$
y=[y_1,y_2,\dots,y_m]
$$

is

$$
x\cdot y=\sum_{i=1}^{m}x_i y_i
$$

Matrix-vector multiplication simply performs this dot product for **every row** of the matrix.

---

### Why is it Used?

Matrix-vector multiplication appears almost everywhere in Machine Learning.

Some common applications include:

- Linear Regression predictions
- Logistic Regression
- Fully Connected (Dense) Neural Network layers
- Feature transformations
- Coordinate transformations
- Physics simulations
- Computer Graphics
- Robotics
- Scientific Computing
- Optimization algorithms

In Deep Learning, a dense layer computes

$$
y=Wx+b
$$

where

- $W$ is the weight matrix
- $x$ is the input vector
- $b$ is the bias vector

The first operation performed is exactly a matrix-vector multiplication.

---

> 💡 **Important Note**
>
> A common mistake is confusing **matrix multiplication** with **element-wise multiplication**. In matrix-vector multiplication, each output value is obtained from the dot product of an entire matrix row with the input vector—not by multiplying corresponding positions of the matrix and vector directly. Always verify that the matrix columns equal the vector length before performing the operation.

---

## Solutions

### Custom Implementation

```python
def matrix_dot_vector(a: list[list[int | float]], b: list[int | float]):

    if not a:
        return []

    if len(a[0]) != len(b):
        return -1

    res = []

    for row in a:
        total = 0

        for x, y in zip(row, b):
            total += x * y

        res.append(total)

    return res
```

---

## Code Explanation

### 1. Handle an Empty Matrix

```python
if not a:
    return []
```

If the matrix has no rows, there is nothing to multiply, so the resulting vector is also empty.

---

### 2. Validate Dimensions

```python
if len(a[0]) != len(b):
    return -1
```

Before performing multiplication, verify that the number of columns equals the vector length.

If the dimensions are incompatible, matrix-vector multiplication is undefined.

---

### 3. Iterate Through Every Row

```python
for row in a:
```

Each row produces one value in the output vector.

---

### 4. Compute the Dot Product

```python
total = 0

for x, y in zip(row, b):
    total += x * y
```

`zip()` pairs each matrix element with its corresponding vector element.

For every pair,

$$
\text{total}=\text{total}+xy
$$

After the loop finishes,

$$
\text{total}=row\cdot b
$$

---

### 5. Store the Result

```python
res.append(total)
```

Each computed dot product becomes one element of the resulting vector.

---

### 6. Return the Final Vector

```python
return res
```

After processing every row, return the transformed vector.

---

## Time & Space Complexity

Let

- $n$ = number of rows in the matrix
- $m$ = number of columns in the matrix (also the vector length)

Every row requires iterating through all **m** columns.

Therefore,

| Complexity | Value |
|------------|-------|
| Time | **O(n × m)** |
| Space | **O(n)** |

The additional space is used only for storing the resulting vector containing **n** elements.