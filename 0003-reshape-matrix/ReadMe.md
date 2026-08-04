# Reshape Matrix (Easy, Linear Algebra)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Reshaping a Matrix](#learn-reshaping-a-matrix)
- [Things to Note](#things-to-note)
- [Solution](#solution)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

---

# Problem Statement

[Reshape Matrix](https://www.deep-ml.com/problems/3)

Write a Python function `reshape_matrix(a, new_shape)` that reshapes a given matrix into the specified dimensions.

If the reshape operation is impossible (the total number of elements changes), return an empty list `[]`.

---

# Example

```python
a = [
    [1, 2, 3, 4],
    [5, 6, 7, 8]
]

new_shape = (4, 2)

print(reshape_matrix(a, new_shape))
```

### Output

```text
[
    [1, 2],
    [3, 4],
    [5, 6],
    [7, 8]
]
```

### Explanation

The original matrix has shape $2 \times 4$.

Therefore, it contains

$$
2 \times 4 = 8
$$

elements.

After reshaping, the new matrix has shape $4 \times 2$.

Again,

$$
4 \times 2 = 8
$$

elements are present.

Only the arrangement changes—the order of the elements remains exactly the same.

---

# Learn: Reshaping a Matrix

## What is Matrix Reshaping?

Matrix reshaping changes the dimensions of a matrix **without changing its data**.

The elements remain in the same order in memory.

Only the arrangement into rows and columns changes.

---

## Example

Consider the matrix

$$
A=
\begin{bmatrix}
1&2&3&4\\
5&6&7&8
\end{bmatrix}
$$

Its shape is $2 \times 4$.

If we flatten it into a one-dimensional sequence, we obtain

$$
[1,2,3,4,5,6,7,8]
$$

Now reshape it into a matrix of shape $4 \times 2$.

The resulting matrix becomes

$$
A'=
\begin{bmatrix}
1&2\\
3&4\\
5&6\\
7&8
\end{bmatrix}
$$

Notice that the numbers appear in exactly the same order.

Only their grouping into rows and columns changes.

---

## Mathematical Condition

Suppose the original matrix has

- $m$ rows
- $n$ columns

The total number of elements is

$$
m \times n
$$

Suppose the new shape is

- $r$ rows
- $c$ columns

Reshaping is only possible if

$$
m \times n = r \times c
$$

Otherwise, some elements would either be lost or new elements would have to be created, making the reshape invalid.

---

## How NumPy Reshape Works

NumPy first reads the matrix as a continuous one-dimensional sequence

$$
[a_1,a_2,a_3,\ldots,a_N]
$$

where $N$ is the total number of elements.

It then fills the new matrix **row by row**.

For example,

Original matrix

$$
\begin{bmatrix}
1&2&3\\
4&5&6
\end{bmatrix}
$$

is interpreted as

$$
[1,2,3,4,5,6]
$$

Reshaping it into a $3 \times 2$ matrix produces

$$
\begin{bmatrix}
1&2\\
3&4\\
5&6
\end{bmatrix}
$$

The ordering of the values never changes.

---

# Things to Note

- Reshaping only changes the **shape**, not the data.
- The total number of elements must remain constant.
- NumPy reshapes matrices in **row-major (C-order)** by default.
- If the requested shape is invalid, NumPy raises a `ValueError`.
- `reshape()` returns a reshaped array, while `tolist()` converts it back into a nested Python list.

---

# Solution

## NumPy Implementation

```python
import numpy as np

def reshape_matrix(a: list[list[int | float]], new_shape: tuple[int, int]) -> list[list[int | float]]:
    try:
        return np.asarray(a).reshape(new_shape).tolist()
    except ValueError:
        return []
```

---

# Code Explanation

## Step 1: Convert the Matrix into a NumPy Array

```python
np.asarray(a)
```

Converts the input Python list into a NumPy array.

This allows NumPy to perform optimized matrix operations.

---

## Step 2: Reshape the Array

```python
.reshape(new_shape)
```

NumPy rearranges the elements into the requested dimensions.

Internally, it first verifies that

$$
\text{Original Number of Elements}
=
\text{New Number of Elements}
$$

or equivalently,

$$
m \times n
=
r \times c
$$

If this condition is not satisfied, NumPy raises a `ValueError`.

---

## Step 3: Convert Back to a Python List

```python
.tolist()
```

Converts the reshaped NumPy array back into a nested Python list.

This matches the required return type.

---

## Step 4: Handle Invalid Shapes

```python
except ValueError:
    return []
```

If the reshape operation is impossible,

the function catches the exception and returns an empty list instead of terminating the program.

---

# Time & Space Complexity

Assume the matrix contains $N$ total elements.

| Complexity | Value |
|-----------|-------|
| Time | **O(N)** |
| Space | **O(N)** |

The reshape operation processes all $N$ elements, and the returned matrix stores the same $N$ elements in a different arrangement.