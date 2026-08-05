# Implement Compressed Column Sparse Matrix Format (CSC) (Easy, Linear Algebra)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Compressed Column Sparse (CSC) Format](#learn-compressed-column-sparse-csc-format)
- [Solution](#solution)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

---

## Problem Statement

### [Implement Compressed Column Sparse Matrix Format (CSC)](https://www.deep-ml.com/problems/67)

Write a Python function that converts a dense matrix into its **Compressed Column Sparse (CSC)** representation.

The function should:

- Accept a 2-D dense matrix.
- Traverse the matrix column by column.
- Store all non-zero values in column-major order.
- Store the corresponding row index for every non-zero value.
- Store the starting position of every column in the values array.
- Return the CSC representation as three arrays.

---

## Example

### Input

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

```python
[1, 2, 3, 4]
[1, 2, 0, 1]
[0, 1, 2, 3, 4]
```

### Reasoning

The matrix is scanned one column at a time.

Whenever a non-zero element is found,

- Store its value.
- Store its row index.

After each column is processed,

- Store the total number of non-zero elements encountered so far in the column pointer array.

Repeating this for every column produces the CSC representation.

---

## Learn: Compressed Column Sparse (CSC) Format

### What is it?

A **Sparse Matrix** is a matrix containing mostly zero values.

Instead of storing every element, the **Compressed Column Sparse (CSC)** format stores only:

- Non-zero values
- Their row indices
- The starting position of every column

This significantly reduces memory usage when the matrix contains many zeros.

Unlike **CSR (Compressed Sparse Row)**, CSC stores data **column by column**.

---

### Example

Suppose the matrix is

```text
1 0 0 0
0 2 0 0
3 0 4 0
1 0 0 5
```

The CSC representation becomes

```text
Values = [1, 3, 1, 2, 4, 5]
```

```text
Row Indices = [0, 2, 3, 1, 2, 3]
```

```text
Column Pointer = [0, 3, 4, 5, 6]
```

Interpretation:

| Column   | Values Array Range |
| -------- | ------------------ |
| Column 0 | Values[0 : 3]      |
| Column 1 | Values[3 : 4]      |
| Column 2 | Values[4 : 5]      |
| Column 3 | Values[5 : 6]      |

The final pointer equals the total number of non-zero elements.

---

### Mathematical Representation

Suppose

- Matrix dimensions

$$
m \times n
$$

- Number of non-zero elements

$$
k
$$

CSC stores three arrays.

Values:

$$
\text{values} =
[a_1,a_2,\ldots,a_k]
$$

Row Indices:

$$
\text{row\_indices} =
[a_1,a_2,\ldots,a_k]
$$

where

$$
0 \le r_i < m
$$

Column Pointer:

$$
\text{col\_ptr} =
[p_0,p_1,\ldots,p_n]
$$

where

$$
p_0=0,\qquad p_n=k
$$

The non-zero elements of column

$$
j
$$

are stored in

$$
\text{values}[p_j:p_{j+1}]
$$

---

### Characteristics / Key Points

- Stores only non-zero values.
- Traverses the matrix column-wise.
- Memory efficient for sparse matrices.
- Fast column access.
- Eliminates storage of zeros.
- Widely used by scientific computing libraries such as SciPy.

---

### Memory Requirement

For a dense matrix of size

$$
m \times n
$$

dense storage requires

$$
m \times n
$$

elements.

CSC stores approximately

$$
2k+n+1
$$

elements,

where

- $k$ is the number of non-zero entries.

When

$$
k \ll mn
$$

CSC requires much less memory.

---

### CSC vs CSR

| CSC                             | CSR                          |
| ------------------------------- | ---------------------------- |
| Column-major storage            | Row-major storage            |
| Fast column access              | Fast row access              |
| Uses row indices                | Uses column indices          |
| Column pointer array            | Row pointer array            |
| Preferred for column operations | Preferred for row operations |

---

### Why is it used? / Applications

CSC is commonly used for sparse matrices where efficient column access is required.

Applications include:

- Sparse Linear Algebra
- Scientific Computing
- Numerical Optimization
- Sparse Matrix Factorization
- Machine Learning
- Graph Algorithms
- Recommendation Systems
- Natural Language Processing
- Finite Element Analysis (FEA)

Libraries such as **SciPy** internally use CSC for many sparse matrix algorithms.

> 💡 **Important Note**
>
> CSC and CSR store exactly the same information but traverse the matrix differently. CSC is optimized for column-wise operations, while CSR is optimized for row-wise operations.

---

## Solution

### Custom Implementation

```python
def compressed_col_sparse_matrix(dense_matrix):
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

## Code Explanation

### Step 1

Initialize the CSC arrays.

```python
values = []
row_indices = []
col_ptr = [0]
```

- `values` stores all non-zero elements.
- `row_indices` stores the corresponding row index.
- `col_ptr` stores the starting position of every column.

---

### Step 2

Traverse the matrix column by column.

```python
for col in range(len(dense_matrix[0])):
    for row in range(len(dense_matrix)):
```

The outer loop processes columns, while the inner loop visits every row.

---

### Step 3

Store every non-zero element.

```python
if dense_matrix[row][col] != 0:
    values.append(dense_matrix[row][col])
    row_indices.append(row)
```

Each non-zero value and its row index are stored.

---

### Step 4

Update the column pointer.

```python
col_ptr.append(len(values))
```

After processing one column, the total number of non-zero elements processed so far becomes the starting index of the next column.

---

### Step 5

Return the CSC representation.

```python
return values, row_indices, col_ptr
```

The function returns the values array, row indices array, and column pointer array.

---

## Time & Space Complexity

Let

- $m$ = Number of rows.
- $n$ = Number of columns.
- $k$ = Number of non-zero elements.

| Complexity | Value        |
| ---------- | ------------ |
| Time       | **O(m × n)** |
| Space      | **O(k + n)** |

The algorithm scans every element exactly once. Additional storage consists of the values array, row indices array, and column pointer array.
