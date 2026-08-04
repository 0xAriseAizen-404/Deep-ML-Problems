# Implement Compressed Column Sparse Matrix Format (CSC) (Easy, Linear Algebra)

## Table of Contents

- Problem Statement
- Example
- Learn: Compressed Column Sparse (CSC) Format
- Solution
- Code Explanation
- Time & Space Complexity

---

## Problem Statement

[Implement Compressed Column Sparse Matrix Format (CSC)](https://www.deep-ml.com/problems/67)

Write a Python function `compressed_col_sparse_matrix(dense_matrix)` that converts a dense matrix into its **Compressed Column Sparse (CSC)** representation.

The function should return three arrays:

- **Values Array:** Stores all non-zero elements in **column-major order**.
- **Row Indices Array:** Stores the row index corresponding to each non-zero value.
- **Column Pointer Array:** Stores the cumulative count of non-zero elements at the start of each column.

CSC is an efficient storage format for sparse matrices where most elements are zero, especially when column-wise operations are common.

---

## Example

```python
dense_matrix = [
    [0, 0, 3, 0],
    [1, 0, 0, 4],
    [0, 2, 0, 0]
]

vals, row_idx, col_ptr = compressed_col_sparse_matrix(dense_matrix)

print(vals)
print(row_idx)
print(col_ptr)
```

### Output

```text
[1, 2, 3, 4]
[1, 2, 0, 1]
[0, 1, 2, 3, 4]
```

### Explanation

The matrix is scanned **column by column**.

```text
0 0 3 0
1 0 0 4
0 2 0 0
```

The non-zero values are encountered in this order:

```text
1 → 2 → 3 → 4
```

Therefore,

```text
Values = [1, 2, 3, 4]
```

Their row positions are

```text
Row Indices = [1, 2, 0, 1]
```

The column pointer indicates where each column begins inside the values array.

```text
Column Pointer = [0, 1, 2, 3, 4]
```

---

# Learn: Compressed Column Sparse (CSC) Format

## What is it?

A **Sparse Matrix** is a matrix that contains mostly zero values.

For example,

```text
1 0 0 0
0 2 0 0
3 0 4 0
1 0 0 5
```

Although the matrix contains **16 entries**, only **6** contain useful information.

Instead of storing every zero, the **Compressed Column Sparse (CSC)** format stores only:

- Non-zero values
- Their row positions
- Where each column begins

This greatly reduces memory usage for large sparse matrices.

Unlike **CSR (Compressed Sparse Row)**, CSC stores information **column by column** instead of row by row.

---

## CSC Data Structure

Consider the matrix

```text
1 0 0 0
0 2 0 0
3 0 4 0
1 0 0 5
```

---

### Values Array

Stores all non-zero elements in **column-major order**.

```text
Values = [1, 3, 1, 2, 4, 5]
```

---

### Row Indices Array

Stores the row position of every value.

```text
Row Indices = [0, 2, 3, 1, 2, 3]
```

For example,

- Value **1** is in row **0**
- Value **3** is in row **2**
- Value **5** is in row **3**

---

### Column Pointer Array

Stores the starting index of every column inside the values array.

```text
Column Pointer = [0, 3, 4, 5, 6]
```

Interpretation:

| Column | Values Array Range |
| ------- | ------------------ |
| Column 0 | Values[0 : 3] |
| Column 1 | Values[3 : 4] |
| Column 2 | Values[4 : 5] |
| Column 3 | Values[5 : 6] |

The last value (**6**) equals the total number of non-zero elements.

---

## Mathematical Representation

Suppose

\[
A
\in
\mathbb{R}^{m\times n}
\]

contains **k** non-zero elements.

CSC stores three arrays.

### Values

\[
\text{values}
=
[a_1,a_2,\ldots,a_k]
\]

---

### Row Indices

\[
\text{row\_indices}
=
[r_1,r_2,\ldots,r_k]
\]

where

\[
0\le r_i<m
\]

---

### Column Pointer

\[
\text{col\_ptr}
=
[p_0,p_1,\ldots,p_n]
\]

where

- \(p_0=0\)
- \(p_n=k\)

The non-zero entries belonging to column \(j\) are stored in

\[
\text{values}
[
p_j
:
p_{j+1}
]
\]

---

## Characteristics / Key Points

- Stores only non-zero values.
- Traverses the matrix **column-wise**.
- Memory efficient for sparse matrices.
- Excellent for column operations.
- Eliminates unnecessary storage of zeros.
- Widely supported by scientific computing libraries such as **SciPy**.

### Memory Requirement

For an

\[
m\times n
\]

dense matrix,

dense storage requires

\[
m\times n
\]

elements.

CSC stores approximately

\[
2k+n+1
\]

elements,

where

- \(k\) = number of non-zero entries.

When

\[
k\ll mn
\]

CSC requires much less memory.

---

## CSC vs CSR

| CSC | CSR |
| ---- | ---- |
| Column-major storage | Row-major storage |
| Fast column access | Fast row access |
| Uses row indices | Uses column indices |
| Column pointer array | Row pointer array |
| Preferred for column operations | Preferred for row operations |

---

## Why is it used? / Applications

CSC is commonly used in applications involving sparse matrices where efficient column access is important.

Common applications include:

- Sparse Linear Algebra
- Scientific Computing
- Numerical Optimization
- Sparse Matrix Factorization
- Machine Learning
- Graph Algorithms
- Recommendation Systems
- Natural Language Processing
- Finite Element Analysis (FEA)

Libraries such as **SciPy** use CSC internally for many sparse matrix algorithms.

> 💡 **Important Note**
>
> CSC and CSR store exactly the same information but in different traversal orders. **CSC is optimized for column-wise operations**, while **CSR is optimized for row-wise operations**. Choosing the correct format can significantly improve the performance of sparse matrix computations.

---

# Solution

## Custom Implementation

```python
def compressed_col_sparse_matrix(dense_matrix):
    """
    Convert a dense matrix into its Compressed Column Sparse (CSC) representation.

    Parameters
    ----------
    dense_matrix : list[list]
        Dense matrix.

    Returns
    -------
    tuple
        (values, row_indices, column_pointer)
    """

    values = []
    row_indices = []
    col_ptr = [0]

    for col in range(len(dense_matrix[0])):
        for row in range(len(dense_matrix)):
            if dense_matrix[row][col] != 0:
                values.append(dense_matrix[row][col])
                row_indices.append(row)

        col_ptr.append(len(values))

    return values, row_indices, col_ptr
```

---

# Code Explanation

### Step 1: Initialize the CSC Arrays

```python
values = []
row_indices = []
col_ptr = [0]
```

- `values` stores all non-zero elements.
- `row_indices` stores the corresponding row positions.
- `col_ptr` starts with **0**, indicating that the first column begins at index 0.

---

### Step 2: Traverse the Matrix Column by Column

```python
for col in range(len(dense_matrix[0])):
```

Process one column at a time.

Inside each column,

```python
for row in range(len(dense_matrix)):
```

visit every row.

---

### Step 3: Store Non-zero Elements

Whenever a non-zero element is found,

```python
values.append(dense_matrix[row][col])
row_indices.append(row)
```

- Store the value.
- Store the row in which it appears.

Zeros are ignored completely.

---

### Step 4: Update the Column Pointer

After scanning an entire column,

```python
col_ptr.append(len(values))
```

`len(values)` equals the total number of non-zero elements processed so far.

This becomes the starting position of the next column.

For example,

```text
After Column 0:

Values = [1, 3, 1]

Column Pointer = [0, 3]
```

After Column 2,

```text
Values = [1, 3, 1, 2, 4]

Column Pointer = [0, 3, 4, 5]
```

---

### Step 5: Return the CSC Representation

Finally,

```python
return values, row_indices, col_ptr
```

returns the complete CSC representation of the dense matrix.

---

## Time & Space Complexity

Let

- \(m\) = Number of rows.
- \(n\) = Number of columns.
- \(k\) = Number of non-zero elements.

| Complexity | Value |
| ---------- | ----- |
| Time | **O(m × n)** |
| Space | **O(k + n)** |

The algorithm visits every element of the matrix exactly once.

The additional storage consists of:

- **Values Array:** `k`
- **Row Indices Array:** `k`
- **Column Pointer Array:** `n + 1`

Thus, the auxiliary space is proportional to the number of non-zero elements and the number of columns.