# Calculate Mean by Row or Column (Easy, Linear Algebra)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Calculate Mean by Row or Column](#learn-calculate-mean-by-row-or-column)
- [Things to Note](#things-to-note)
- [Solutions](#solutions)
  - [Custom Implementation](#custom-implementation)
  - [NumPy Implementation](#numpy-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

---

# Problem Statement

[Calculate Mean by Row or Column](https://www.deep-ml.com/problems/4)

Write a Python function `calculate_matrix_mean(matrix, mode)` that computes the mean of a matrix either **row-wise** or **column-wise**.

The parameter `mode` determines whether the function calculates:

- `"row"` → Mean of every row.
- `"column"` → Mean of every column.

The function should return a list containing the computed means.

---

# Example

```python
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

print(calculate_matrix_mean(matrix, "column"))
```

### Output

```text
[4.0, 5.0, 6.0]
```

### Explanation

Column means are

$$
\frac{1+4+7}{3}=4
$$

$$
\frac{2+5+8}{3}=5
$$

$$
\frac{3+6+9}{3}=6
$$

---

```python
print(calculate_matrix_mean(matrix, "row"))
```

### Output

```text
[2.0, 5.0, 8.0]
```

### Explanation

Row means are

$$
\frac{1+2+3}{3}=2
$$

$$
\frac{4+5+6}{3}=5
$$

$$
\frac{7+8+9}{3}=8
$$

---

# Learn: Calculate Mean by Row or Column

## What is the Mean?

The **mean (average)** is the central value of a collection of numbers.

For a set containing $n$ numbers,

the mean is

$$
\mu=\frac{1}{n}\sum_{i=1}^{n}x_i
$$

where

- $x_i$ is the $i^{th}$ value.
- $n$ is the total number of values.

---

## Row Mean

Suppose a matrix

$$
A=
\begin{bmatrix}
a_{11}&a_{12}&\cdots&a_{1n}\\
a_{21}&a_{22}&\cdots&a_{2n}\\
\vdots&\vdots&\ddots&\vdots\\
a_{m1}&a_{m2}&\cdots&a_{mn}
\end{bmatrix}
$$

The mean of row $i$ is

$$
\mu_{\text{row }i}
=
\frac{1}{n}
\sum_{j=1}^{n}
a_{ij}
$$

where

- $n$ is the number of columns.
- $a_{ij}$ is the element at row $i$ and column $j$.

---

## Column Mean

Similarly, the mean of column $j$ is

$$
\mu_{\text{column }j}
=
\frac{1}{m}
\sum_{i=1}^{m}
a_{ij}
$$

where

- $m$ is the number of rows.
- $a_{ij}$ is the element at row $i$ and column $j$.

---

## Visualization

Consider

$$
A=
\begin{bmatrix}
1&2&3\\
4&5&6\\
7&8&9
\end{bmatrix}
$$

### Row Means

$$
\left[
\frac{1+2+3}{3},
\frac{4+5+6}{3},
\frac{7+8+9}{3}
\right]
=
[2,5,8]
$$

### Column Means

$$
\left[
\frac{1+4+7}{3},
\frac{2+5+8}{3},
\frac{3+6+9}{3}
\right]
=
[4,5,6]
$$

---

## Applications

Computing row and column means is one of the most common preprocessing operations in Machine Learning.

Applications include:

- Feature normalization
- Data standardization
- Mean-centering datasets
- Principal Component Analysis (PCA)
- Image processing
- Statistical analysis

---

# Things to Note

- Row mean computes one average **per row**.
- Column mean computes one average **per column**.
- The output length depends on the selected mode.
- A matrix with $m$ rows and $n$ columns produces:
  - $m$ row means.
  - $n$ column means.

---

# Solutions

## Custom Implementation

```python
import numpy as np

def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    res = []

    if mode == "row":
        for vec in matrix:
            res.append(sum(vec) / len(vec))
    else:
        for ind in range(len(matrix[0])):
            summ = sum(vec[ind] for vec in matrix)
            res.append(summ / len(matrix))

    return res
```

---

## NumPy Implementation

```python
import numpy as np

def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    if mode == "column":
        return np.asarray(matrix).mean(axis=0).tolist()
    else:
        return np.asarray(matrix).mean(axis=1).tolist()
```

---

# Code Explanation

## Custom Implementation

### Step 1: Create the Result List

```python
res = []
```

This list stores the computed means.

---

### Step 2: Check the Mode

```python
if mode == "row":
```

If the mode is `"row"`, the function computes one mean for every row.

Otherwise, it computes one mean for every column.

---

### Step 3: Compute Row Means

```python
for vec in matrix:
    res.append(sum(vec) / len(vec))
```

For every row,

```python
sum(vec)
```

computes

$$
\sum_{j=1}^{n}a_{ij}
$$

and

```python
len(vec)
```

returns the number of elements in that row.

The computed mean is therefore

$$
\mu_{\text{row }i}
=
\frac{1}{n}
\sum_{j=1}^{n}
a_{ij}
$$

The result is appended to `res`.

---

### Step 4: Iterate Through Every Column

```python
for ind in range(len(matrix[0])):
```

The loop visits every column index from left to right.

Here, `ind` represents the current column index.

---

### Step 5: Compute the Sum of a Column

```python
summ = sum(vec[ind] for vec in matrix)
```

This adds together all elements belonging to column `ind`.

Mathematically,

$$
\sum_{i=1}^{m}
a_{ij}
$$

where $j$ is the current column.

---

### Step 6: Compute the Column Mean

```python
res.append(summ / len(matrix))
```

Since `len(matrix)` equals the number of rows, the computed value is

$$
\mu_{\text{column }j}
=
\frac{1}{m}
\sum_{i=1}^{m}
a_{ij}
$$

The result is appended to `res`.

---

### Step 7: Return the Result

```python
return res
```

The function returns the list containing either the row means or the column means.

---

## NumPy Implementation

### Compute Column Means

```python
np.asarray(matrix).mean(axis=0)
```

First,

```python
np.asarray(matrix)
```

converts the Python list into a NumPy array.

Then,

```python
.mean(axis=0)
```

computes the mean along axis $0$, meaning values are averaged **down each column**.

This directly computes

$$
\mu_{\text{column }j}
=
\frac{1}{m}
\sum_{i=1}^{m}
a_{ij}
$$

---

### Compute Row Means

```python
np.asarray(matrix).mean(axis=1)
```

Using `axis=1` computes the mean **across each row**.

This corresponds to

$$
\mu_{\text{row }i}
=
\frac{1}{n}
\sum_{j=1}^{n}
a_{ij}
$$

Finally,

```python
.tolist()
```

converts the NumPy array back into a standard Python list.

---

# Time & Space Complexity

Assume the matrix contains

- $m$ rows.
- $n$ columns.

| Complexity | Value |
|------------|-------|
| Time | **O(mn)** |
| Space | **O(m+n)** |

Every element of the matrix is visited exactly once while computing the means. The output stores either $m$ row means or $n$ column means.