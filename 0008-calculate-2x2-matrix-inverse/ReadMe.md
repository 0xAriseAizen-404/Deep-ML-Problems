# Inverse of a 2×2 Matrix (Easy) ✔

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Inverse of a 2×2 Matrix](#learn-inverse-of-a-2x2-matrix)
- [Approach](#approach)
- [Solutions](#solutions)
  - [Custom Implementation](#custom-implementation)
  - [NumPy Implementation](#numpy-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)
- [Edge Cases](#edge-cases)

---

## Problem Statement

[Calculate 2x2 Matrix Inverse](https://www.deep-ml.com/problems/8)

Write a Python function that calculates the **inverse of a 2×2 matrix**.

The inverse of a matrix **A** is another matrix **A⁻¹** such that:

\[
A \times A^{-1} = I
\]

where **I** is the identity matrix.

For a matrix

\[
A =
\begin{pmatrix}
a & b \\
c & d
\end{pmatrix}
\]

the inverse exists **only if** its determinant is non-zero.

Return **`None`** if the matrix is not invertible.

---

## Example

```python
Input:
matrix = [[4, 7],
          [2, 6]]

Output:
[[0.6, -0.7],
 [-0.2, 0.4]]
```

### Reasoning

For

\[
A=
\begin{pmatrix}
4 & 7 \\
2 & 6
\end{pmatrix}
\]

Determinant:

\[
det = ad - bc
\]

\[
=4\times6-7\times2
\]

\[
=24-14=10
\]

Since the determinant is **non-zero**, the inverse exists.

Using the inverse formula:

\[
A^{-1}
=
\frac1{10}
\begin{pmatrix}
6 & -7\\
-2 & 4
\end{pmatrix}
\]

Result:

```python
[[0.6, -0.7],
 [-0.2, 0.4]]
```

---

# Learn: Inverse of a 2×2 Matrix

## What is the Inverse of a Matrix?

The inverse of a matrix is similar to the reciprocal of a number.

For numbers,

\[
5\times\frac15=1
\]

Likewise, for matrices,

\[
A\times A^{-1}=I
\]

where

\[
I=
\begin{pmatrix}
1&0\\
0&1
\end{pmatrix}
\]

is called the **Identity Matrix**.

---

## Formula

For

\[
A=
\begin{pmatrix}
a&b\\
c&d
\end{pmatrix}
\]

First compute the determinant:

\[
det(A)=ad-bc
\]

If

\[
det(A)\neq0
\]

then

\[
A^{-1}
=
\frac1{det(A)}
\begin{pmatrix}
d&-b\\
-c&a
\end{pmatrix}
\]

---

## When Does the Inverse Exist?

The matrix is invertible **only if**

\[
det(A)\neq0
\]

If

\[
det(A)=0
\]

the matrix is called **singular**, and **no inverse exists**.

---

## Why Does the Determinant Matter?

The determinant tells us whether the rows (or columns) are linearly independent.

- Determinant ≠ 0 → Matrix preserves dimensions → Invertible
- Determinant = 0 → Matrix collapses space → Not invertible

For example,

```
[[1,2],
 [2,4]]
```

Second row = 2 × First row

These rows are dependent.

Determinant:

```
1×4 − 2×2 = 0
```

No inverse exists.

---

## Step-by-Step Algorithm

Given

```python
[[a, b],
 [c, d]]
```

### Step 1

Extract the values.

```python
a, b = matrix[0]
c, d = matrix[1]
```

---

### Step 2

Calculate determinant.

```python
det = a*d - b*c
```

---

### Step 3

If determinant is zero,

```python
return None
```

---

### Step 4

Otherwise apply the formula.

```python
[
 [ d/det, -b/det],
 [-c/det,  a/det]
]
```

---

# Approach

## Intuition

The inverse of a 2×2 matrix has a direct mathematical formula.

Instead of performing Gaussian elimination, we simply:

1. Compute the determinant.
2. Check whether it is zero.
3. Swap the diagonal elements.
4. Negate the off-diagonal elements.
5. Divide every element by the determinant.

---

## Algorithm

1. Read matrix elements.
2. Compute determinant.
3. If determinant equals zero:
   - Return `None`.
4. Otherwise:
   - Swap `a` and `d`.
   - Negate `b` and `c`.
   - Divide each element by determinant.
5. Return the inverse matrix.

---

## Dry Run

Input

```python
[[4,7],
 [2,6]]
```

Extract

```
a = 4
b = 7
c = 2
d = 6
```

Determinant

```
4×6 − 7×2

=24−14

=10
```

Construct inverse

```
[[6,-7],
 [-2,4]]
```

Divide every element by 10

```
[[0.6,-0.7],
 [-0.2,0.4]]
```

Output

```python
[[0.6,-0.7],
 [-0.2,0.4]]
```

---

# Solutions

## Custom Implementation

```python
def inverse_2x2(matrix: list[list[int | float]]) -> list[list[float]] | None:
    a, b = matrix[0]
    c, d = matrix[1]

    determinant = (a * d) - (b * c)

    if determinant == 0:
        return None

    return [
        [d / determinant, -b / determinant],
        [-c / determinant, a / determinant]
    ]


# Example Usage
matrix = [[4, 7],
          [2, 6]]

print(inverse_2x2(matrix))
```

**Output**

```python
[[0.6, -0.7],
 [-0.2, 0.4]]
```

---

## NumPy Implementation

```python
import numpy as np


def inverse_2x2(matrix: list[list[int | float]]) -> list[list[float]] | None:
    matrix = np.array(matrix)

    if np.linalg.det(matrix) == 0:
        return None

    return np.linalg.inv(matrix)


# Example Usage
matrix = [[4, 7],
          [2, 6]]

print(inverse_2x2(matrix))
```

> **Note:** `np.linalg.inv()` works for square matrices of any size, not just 2×2 matrices.

---

# Code Explanation

### Step 1

Extract the four matrix elements.

```python
a, b = matrix[0]
c, d = matrix[1]
```

For

```python
[[4,7],
 [2,6]]
```

```
a = 4
b = 7
c = 2
d = 6
```

---

### Step 2

Calculate the determinant.

```python
determinant = (a * d) - (b * c)
```

```
4×6 − 7×2

=10
```

---

### Step 3

Check if the inverse exists.

```python
if determinant == 0:
    return None
```

A zero determinant means the matrix is singular and cannot be inverted.

---

### Step 4

Apply the inverse formula.

```python
return [
    [d / determinant, -b / determinant],
    [-c / determinant, a / determinant]
]
```

For the example,

```
[[6,-7],
 [-2,4]]
```

Divide each element by 10.

Result:

```
[[0.6,-0.7],
 [-0.2,0.4]]
```

---

# Time & Space Complexity

### Custom Implementation

| Complexity | Value |
|------------|-------|
| Time | **O(1)** |
| Space | **O(1)** |

Since a 2×2 matrix always contains exactly four elements, the number of operations is constant.

---

### NumPy Implementation

| Complexity | Value |
|------------|-------|
| Time | **O(1)** (for a fixed 2×2 matrix) |
| Space | **O(1)** |

For larger matrices, NumPy uses optimized algorithms with approximately **O(n³)** time complexity.

---

# Edge Cases

### 1. Singular Matrix

```python
matrix = [
    [1,2],
    [2,4]
]
```

Determinant

```
1×4 − 2×2

=0
```

Output

```python
None
```

---

### 2. Identity Matrix

```python
matrix = [
    [1,0],
    [0,1]
]
```

Output

```python
[[1,0],
 [0,1]]
```

The identity matrix is its own inverse.

---

### 3. Negative Numbers

```python
matrix = [
    [-2,1],
    [3,4]
]
```

The same formula applies.

---

### 4. Floating Point Values

```python
matrix = [
    [1.5,2.5],
    [3.2,4.1]
]
```

The function works correctly because Python supports floating-point arithmetic.

---

## Key Takeaways

- The inverse exists **only if the determinant is non-zero**.
- The determinant is computed as **ad − bc**.
- The inverse of a 2×2 matrix is obtained by:
  - Swapping the diagonal elements.
  - Negating the off-diagonal elements.
  - Dividing each element by the determinant.
- If the determinant is **0**, return **`None`** since the matrix is singular.
- The custom implementation runs in **O(1)** time and uses **O(1)** space.