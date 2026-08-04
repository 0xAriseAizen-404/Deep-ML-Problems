# Implement Compressed Row Sparse Matrix (CSR) Format Conversion (Easy, Linear Algebra)

## Table of Contents

- Problem Statement
- Example
- Learn: Compressed Row Sparse (CSR) Format
- Solution
- Code Explanation
- Time & Space Complexity

---

## Problem Statement

[Implement Compressed Row Sparse Matrix (CSR) Format Conversion](https://www.deep-ml.com/problems/65)

Write a Python function `compressed_row_sparse_matrix(dense_matrix)` that converts a dense matrix into the **Compressed Row Sparse (CSR)** format.

The function should return three arrays:

- **Values Array:** Stores all non-zero elements in row-major order.
- **Column Indices Array:** Stores the column index corresponding to each non-zero value.
- **Row Pointer Array:** Stores the cumulative count of non-zero elements at the start of each row.

CSR is one of the most widely used sparse matrix storage formats because it significantly reduces memory usage when most matrix elements are zero.

---

## Example

```python
dense_matrix = [
    [1, 0, 0, 0],
    [0, 2, 0, 0],
    [3, 0, 4, 0],
    [1, 0, 0, 5]
]

vals, col_idx, row_ptr = compressed_row_sparse_matrix(dense_matrix)

print("Values array:", vals)
print("Column indices array:", col_idx)
print("Row pointer array:", row_ptr)
```

### Output

```text
Values array: [1, 2, 3, 4, 1, 5]
Column indices array: [0, 1, 0, 2, 0, 3]
Row pointer array: [0, 1, 2, 4, 6]
```

### Explanation

The non-zero values are stored in row-major order.

**Values**

```text
[1, 2, 3, 4, 1, 5]
```

Their corresponding column indices are

```text
[0, 1, 0, 2, 0, 3]
```

The row pointer array indicates where each row begins inside the values array.

| Row | Non-zero Elements | Start Index |
| --- | ----------------- | ----------: |
| 0 | [1] | 0 |
| 1 | [2] | 1 |
| 2 | [3, 4] | 2 |
| 3 | [1, 5] | 4 |

Thus,

```text
Row Pointer = [0, 1, 2, 4, 6]
```

---

# Learn: Compressed Row Sparse (CSR) Format

## What is it?

A **Sparse Matrix** is a matrix in which most of the elements are zero.

For example,

```text
1 0 0 0
0 2 0 0
3 0 4 0
1 0 0 5
```

Although this matrix contains **16 elements**, only **6** are non-zero.

Storing every element wastes memory, especially for very large matrices.

The **Compressed Row Sparse (CSR)** format stores **only the non-zero values** and their positions, making it one of the most memory-efficient representations for sparse matrices.

Instead of storing the complete matrix, CSR stores three one-dimensional arrays.

---

## CSR Data Structure

Suppose the matrix is

```text
1 0 0 0
0 2 0 0
3 0 4 0
1 0 0 5
```

CSR stores

### Values Array

Contains every non-zero element.

```text
Values = [1, 2, 3, 4, 1, 5]
```

---

### Column Indices Array

Stores the column position of every value.

```text
Column Indices = [0, 1, 0, 2, 0, 3]
```

For example,

- Value **1** is in column **0**
- Value **2** is in column **1**
- Value **4** is in column **2**

---

### Row Pointer Array

The row pointer stores **where each row begins inside the values array**.

```text
Row Pointer = [0, 1, 2, 4, 6]
```

Interpretation:

| Row | Values Array Range |
| --- | ------------------ |
| Row 0 | Values[0 : 1] |
| Row 1 | Values[1 : 2] |
| Row 2 | Values[2 : 4] |
| Row 3 | Values[4 : 6] |

Notice that the last value (**6**) equals the total number of non-zero elements.

---

## Mathematical Representation

Suppose a sparse matrix

\[
A
\in
\mathbb{R}^{m \times n}
\]

contains **k** non-zero elements.

CSR stores three arrays:

### Values

\[
\text{values}
=
[a_1,a_2,\dots,a_k]
\]

---

### Column Indices

\[
\text{col\_indices}
=
[c_1,c_2,\dots,c_k]
\]

where

\[
0 \le c_i < n
\]

---

### Row Pointer

\[
\text{row\_ptr}
=
[p_0,p_1,\dots,p_m]
\]

where

- \(p_0=0\)
- \(p_m=k\)

The non-zero values belonging to row \(i\) are located in

\[
\text{values}
[
p_i
:
p_{i+1}
]
\]

---

## Characteristics / Key Points

- Stores only non-zero values.
- Memory efficient for sparse matrices.
- Excellent for row-wise traversal.
- Matrix-vector multiplication is very fast.
- Eliminates storage of unnecessary zeros.
- Widely supported by scientific computing libraries such as **SciPy**.

### Memory Comparison

For an \(m \times n\) dense matrix:

Dense storage requires

\[
m \times n
\]

elements.

CSR storage requires approximately

\[
2k+m+1
\]

elements,

where

- \(k\) = number of non-zero entries.

When

\[
k
\ll
mn
\]

CSR uses dramatically less memory.

---

## Why is it used? / Applications

CSR is widely used in scientific computing and machine learning because many real-world matrices are sparse.

Common applications include:

- Sparse Linear Algebra
- Graph adjacency matrices
- PageRank algorithm
- Recommendation systems
- Natural Language Processing (Bag-of-Words and TF-IDF matrices)
- Finite Element Analysis (FEA)
- Solving sparse systems of linear equations
- Sparse feature representations in machine learning

> 💡 **Important Note**
>
> CSR is optimized for **row-wise operations** such as matrix-vector multiplication. If your application frequently accesses or modifies **columns**, the **Compressed Sparse Column (CSC)** format is often a better choice.

---

# Solution

## Custom Implementation

```python
import numpy as np

def compressed_row_sparse_matrix(dense_matrix):
    """
    Convert a dense matrix to its Compressed Row Sparse (CSR) representation.

    Parameters
    ----------
    dense_matrix : list[list]
        Dense matrix.

    Returns
    -------
    tuple
        (values, column_indices, row_pointer)
    """

    values = []
    col_indices = []
    row_ptr = [0]

    for row in range(len(dense_matrix)):
        for col in range(len(dense_matrix[0])):
            if dense_matrix[row][col] != 0:
                values.append(dense_matrix[row][col])
                col_indices.append(col)

        row_ptr.append(len(values))

    return values, col_indices, row_ptr
```

---

# Code Explanation

### Step 1: Initialize the CSR Arrays

```python
values = []
col_indices = []
row_ptr = [0]
```

- `values` stores all non-zero entries.
- `col_indices` stores their column positions.
- `row_ptr` starts with **0**, indicating that the first row begins at index 0.

---

### Step 2: Traverse the Matrix

```python
for row in range(len(dense_matrix)):
```

Process each row one at a time.

Inside each row,

```python
for col in range(len(dense_matrix[0])):
```

visit every column.

---

### Step 3: Store Non-zero Elements

Whenever a non-zero element is found,

```python
values.append(dense_matrix[row][col])
col_indices.append(col)
```

- Save the value.
- Save its column index.

Zeros are ignored completely.

---

### Step 4: Update the Row Pointer

After finishing a row,

```python
row_ptr.append(len(values))
```

`len(values)` equals the total number of non-zero elements processed so far.

This becomes the starting position of the next row.

For example,

```text
After Row 0:
values = [1]

row_ptr = [0, 1]
```

After Row 2,

```text
values = [1, 2, 3, 4]

row_ptr = [0, 1, 2, 4]
```

---

### Step 5: Return the CSR Representation

Finally,

```python
return values, col_indices, row_ptr
```

returns the complete CSR representation of the matrix.

---

## Time & Space Complexity

Let

- \(m\) = Number of rows.
- \(n\) = Number of columns.
- \(k\) = Number of non-zero elements.

| Complexity | Value |
| ---------- | ----- |
| Time | **O(m × n)** |
| Space | **O(k + m)** |

The algorithm scans every element of the dense matrix exactly once. The additional space consists of:

- **Values Array:** `k`
- **Column Indices Array:** `k`
- **Row Pointer Array:** `m + 1`

Thus, the total auxiliary space is proportional to the number of non-zero elements and the number of rows.