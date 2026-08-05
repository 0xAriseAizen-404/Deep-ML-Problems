# Reshape Matrix (Easy, Linear Algebra)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Reshaping a Matrix](#learn-reshaping-a-matrix)
- [Solutions](#solutions)
  - [NumPy Implementation](#numpy-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

## Problem Statement

### [Reshape Matrix](https://www.deep-ml.com/problems/3)

Write a Python function that reshapes a given matrix into a new specified shape.

Reshaping changes only the **dimensions** of the matrix while preserving the order of its elements. If the requested shape is impossible because the total number of elements differs, return an empty list `[]`.

---

## Example

### Input

```python
a = [
    [1, 2, 3, 4],
    [5, 6, 7, 8]
]

new_shape = (4, 2)
```

### Output

```python
[
    [1, 2],
    [3, 4],
    [5, 6],
    [7, 8]
]
```

### Reasoning

The original matrix has shape **2 × 4**, containing

$$
2 \times 4 = 8
$$

elements.

The requested shape is **4 × 2**, which also contains

$$
4 \times 2 = 8
$$

elements.

Since the total number of elements remains unchanged, the reshape operation is valid.

---

## Learn: Reshaping a Matrix

### What is it?

Matrix reshaping is the process of changing the dimensions of a matrix without modifying its data.

Unlike the transpose operation, reshaping **does not move elements based on their indices**. Instead, the elements are read in their original order and placed sequentially into a matrix with the new dimensions.

Reshaping is one of the most common preprocessing operations in Machine Learning and Deep Learning, where data often needs to match the expected input shape of an algorithm or neural network.

---

### Mathematical Definition

Suppose

$$
A \in \mathbb{R}^{m \times n}
$$

contains

$$
m \times n
$$

elements.

It can be reshaped into

$$
A' \in \mathbb{R}^{p \times q}
$$

only if

$$
m \times n = p \times q
$$

The total number of elements must remain constant.

---

### Example

Original matrix

$$
A=
\begin{bmatrix}
1 & 2 & 3 & 4\\
5 & 6 & 7 & 8
\end{bmatrix}
$$

Flattening in row-major order gives

$$
[1,2,3,4,5,6,7,8]
$$

Reshaping into a **4 × 2** matrix produces

$$
A'=
\begin{bmatrix}
1 & 2\\
3 & 4\\
5 & 6\\
7 & 8
\end{bmatrix}
$$

Notice that the values remain unchanged—only the shape changes.

---

### Reshape Condition

A reshape is valid only when

$$
\text{Original Elements}=\text{New Elements}
$$

or equivalently,

$$
m \times n = p \times q
$$

If this condition is not satisfied, reshaping is impossible.

For example,

$$
2 \times 3 \ne 4 \times 2
$$

since

$$
6 \ne 8
$$

Therefore, such a reshape must fail.

---

### Characteristics / Key Points

- Changes only the matrix dimensions.
- Does not modify element values.
- Preserves the order of elements.
- Requires the total number of elements to remain constant.
- Commonly performed in row-major (C-style) order.
- Widely supported by numerical computing libraries such as NumPy.

---

### Row-Major Ordering

NumPy reshapes matrices using **row-major order** by default.

For example,

```text
Original

1 2 3
4 5 6
```

is internally viewed as

```text
1 2 3 4 5 6
```

Reshaping into a **3 × 2** matrix gives

```text
1 2
3 4
5 6
```

The sequence of elements never changes.

---

### Why is it Used?

Reshaping is frequently used while preparing data for machine learning models.

Common applications include:

- Preparing feature matrices
- Converting images into vectors
- Transforming vectors back into images
- Mini-batch creation
- Tensor manipulation
- Deep Learning input preprocessing
- Computer Vision pipelines
- Scientific computing
- Data engineering workflows

For example, grayscale images of size **28 × 28** are often reshaped into vectors of length

$$
28 \times 28 = 784
$$

before being fed into traditional machine learning models.

---

> 💡 **Important Note**
>
> Reshaping does **not** rearrange or recompute data—it only changes how the existing elements are interpreted in memory. Always verify that the product of the new dimensions equals the product of the original dimensions; otherwise, the reshape operation is invalid.

---

## Solutions

### NumPy Implementation

```python
import numpy as np

def reshape_matrix(a: list[list[int | float]], new_shape: tuple[int, int]) -> list[list[int | float]]:
    try:
        return np.asarray(a).reshape(new_shape).tolist()
    except ValueError:
        return []
```

---

## Code Explanation

### 1. Convert the Input into a NumPy Array

```python
np.asarray(a)
```

The input Python list is converted into a NumPy array so that NumPy's built-in reshape functionality can be used.

---

### 2. Attempt the Reshape

```python
.reshape(new_shape)
```

NumPy checks whether the requested shape is compatible with the original number of elements.

If

$$
m \times n = p \times q
$$

the reshape succeeds.

Otherwise, a `ValueError` is raised.

---

### 3. Convert Back to a Python List

```python
.tolist()
```

Since the function should return a standard Python list, the reshaped NumPy array is converted back using `tolist()`.

---

### 4. Handle Invalid Shapes

```python
except ValueError:
    return []
```

If the reshape is impossible because the total number of elements differs, the function catches the exception and returns an empty list.

---

## Time & Space Complexity

Let

- $m$ = number of rows
- $n$ = number of columns
- $N = m \times n$ = total number of elements

| Complexity | Value |
| ---------- | ----- |
| Time | **O(N)** |
| Space | **O(N)** |

The reshape operation processes all elements, and converting the result back into a Python list requires storing all **N** elements.