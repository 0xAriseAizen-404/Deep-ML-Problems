# Implement Compressed Row Sparse Matrix (CSR) Format Conversion (Easy, Linear Algebra)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Understanding the Compressed Row Sparse (CSR) Format](#learn-understanding-the-compressed-row-sparse-csr-format)
- [Solution](#solution)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

---

## Problem Statement

### [Implement Compressed Row Sparse Matrix (CSR) Format Conversion](https://www.deep-ml.com/problems/65)

Write a Python function that converts a dense matrix into its **Compressed Row Sparse (CSR)** representation.

The function should:

- Accept a 2D dense matrix.
- Store only the non-zero elements.
- Record the column index of every non-zero element.
- Maintain a row pointer array indicating where each row begins in the values array.
- Return the CSR representation as a tuple containing:
  - Values array
  - Column indices array
  - Row pointer array

---

## Example

### Input

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

```python
Values array: [1, 2, 3, 4, 1, 5]
Column indices array: [0, 1, 0, 2, 0, 3]
Row pointer array: [0, 1, 2, 4, 6]
```

### Reasoning

The algorithm scans the matrix row by row.

Whenever a non-zero element is found,

- Store its value.
- Store its column index.

After processing each row, record the current number of stored values in the row pointer array.

This allows every row to be reconstructed without storing the zero elements.

---

## Learn: Understanding the Compressed Row Sparse (CSR) Format

### What is it?

A **Sparse Matrix** is a matrix in which most elements are zero.

For large sparse matrices, storing every element wastes both memory and computation.

The **Compressed Row Sparse (CSR)** format stores only the useful (non-zero) elements, making it one of the most popular sparse matrix representations used in scientific computing and machine learning.

Instead of storing every matrix entry, CSR represents a matrix using three one-dimensional arrays.

---

### CSR Representation

A CSR matrix consists of

- **Values array**
- **Column indices array**
- **Row pointer array**

These three arrays contain enough information to reconstruct the original matrix.

---

### Values Array

The values array stores every non-zero element in **row-major order**.

Example

```text
1 0 0 0
0 2 0 0
3 0 4 0
1 0 0 5
```

Values array

```text
[1, 2, 3, 4, 1, 5]
```

---

### Column Indices Array

For every value stored, record the column where it appears.

Column indices

```text
[0, 1, 0, 2, 0, 3]
```

The first value (`1`) appears in column `0`.

The second value (`2`) appears in column `1`, and so on.

---

### Row Pointer Array

The row pointer stores the starting index of every row inside the values array.

For the example,

```text
Row 0 -> values[0:1]
Row 1 -> values[1:2]
Row 2 -> values[2:4]
Row 3 -> values[4:6]
```

Therefore,

```text
Row Pointer = [0, 1, 2, 4, 6]
```

Notice that the row pointer always contains

$$ \text{Number of Rows} + 1 $$

elements.

The last value equals the total number of stored non-zero elements.

---

### Matrix Example

Dense matrix

```text
1 0 0 0
0 2 0 0
3 0 4 0
1 0 0 5
```

CSR representation

```text
Values      = [1, 2, 3, 4, 1, 5]
Column Index= [0, 1, 0, 2, 0, 3]
Row Pointer = [0, 1, 2, 4, 6]
```

---

### Reconstructing a Row

Suppose we want the third row.

Using

```text
Row Pointer = [0, 1, 2, 4, 6]
```

The third row begins at

```text
values[2]
```

and ends before

```text
values[4]
```

Therefore,

```text
Values      = [3, 4]
Columns     = [0, 2]
```

The reconstructed row becomes

```text
3 0 4 0
```

---

### Empty Rows

CSR naturally supports empty rows.

Example

```text
1 0 0 0
0 0 0 0
3 0 0 0
1 0 0 5
```

Values

```text
[1, 3, 1, 5]
```

Column indices

```text
[0, 0, 0, 3]
```

Row pointer

```text
[0, 1, 1, 2, 4]
```

Notice that

```text
Row Pointer[1] == Row Pointer[2]
```

which indicates that the second row contains no non-zero elements.

---

### Why CSR is Efficient

A dense matrix requires memory for every element.

For an

$$ m \times n $$

matrix,

memory usage is proportional to

$$ O(mn) $$

CSR stores only the non-zero elements.

If there are

$$ k $$

non-zero entries,

memory usage becomes

$$ O(k+m) $$

Since

$$ k \ll mn $$

for sparse matrices,

CSR provides significant memory savings.

---

### Characteristics / Key Points

- Stores only non-zero values.
- Uses three one-dimensional arrays.
- Efficient row access.
- Excellent memory efficiency for sparse matrices.
- Preserves row-major ordering.
- Widely supported by scientific computing libraries.
- Ideal when most matrix entries are zero.

---

### Advantages

- Greatly reduces memory usage.
- Faster sparse matrix operations.
- Efficient row slicing.
- Easy matrix-vector multiplication.
- Standard sparse representation in many numerical libraries.

---

### Limitations

- Slow column access compared to row access.
- Inserting new non-zero elements is expensive.
- Less suitable for frequently changing sparse matrices.

---

### Applications

CSR is widely used in

- Sparse linear algebra
- Scientific computing
- Finite Element Analysis (FEA)
- Graph algorithms
- Recommendation systems
- Machine learning with sparse features
- Sparse neural network computations
- PageRank
- Large-scale numerical simulations

It is the default sparse matrix format in libraries such as **SciPy**.

---

### Practical Example

Suppose a matrix contains

- 1,000 rows
- 1,000 columns
- Only 2,000 non-zero elements

A dense representation stores

```text
1,000,000 values
```

whereas CSR stores approximately

```text
2,000 values
2,000 column indices
1,001 row pointers
```

This dramatically reduces memory consumption.

---

### Common Mistakes

- Forgetting to append the final row pointer.
- Storing zero values.
- Recording incorrect column indices.
- Updating the row pointer before processing the row.
- Assuming every row contains at least one non-zero element.

---

> 💡 **Important Note**
>
> CSR is optimized for row-wise operations. If frequent column access is required, formats such as **Compressed Sparse Column (CSC)** are usually more efficient.

---

## Solution

### Custom Implementation

```python
import numpy as np

def compressed_row_sparse_matrix(dense_matrix):
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

## Code Explanation

### Step 1

Initialize the three arrays.

```python
values = []
col_indices = []
row_ptr = [0]
```

These arrays will store the complete CSR representation.

---

### Step 2

Traverse the matrix row by row.

```python
for row in range(len(dense_matrix)):
```

Each row is processed independently.

---

### Step 3

Visit every column in the current row.

```python
for col in range(len(dense_matrix[0])):
```

Every matrix element is inspected exactly once.

---

### Step 4

Store non-zero elements.

```python
if dense_matrix[row][col] != 0:
```

For each non-zero value,

- Append the value.
- Append its corresponding column index.

Zero elements are ignored.

---

### Step 5

Update the row pointer.

```python
row_ptr.append(len(values))
```

After finishing a row, the current number of stored values indicates where the next row begins.

---

### Step 6

Return the CSR representation.

```python
return values, col_indices, row_ptr
```

The three arrays together completely describe the original sparse matrix.

---

## Time & Space Complexity

| Complexity | Value        |
| ---------- | ------------ |
| Time       | **O(m × n)** |
| Space      | **O(k + m)** |

where

- $m$ is the number of rows.
- $n$ is the number of columns.
- $k$ is the number of non-zero elements.
- Every matrix element is visited exactly once.
- Only the non-zero values, their column indices, and the row pointer array are stored.
