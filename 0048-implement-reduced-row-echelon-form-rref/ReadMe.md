# Implement Reduced Row Echelon Form (RREF) Function (Medium, Linear Algebra)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Understanding Reduced Row Echelon Form](#learn-understanding-reduced-row-echelon-form)
  - [What is RREF?](#what-is-rref)
  - [Row Echelon Form vs RREF](#row-echelon-form-vs-rref)
  - [Elementary Row Operations](#elementary-row-operations)
  - [Mathematical Definition](#mathematical-definition)
  - [Pivot](#pivot)
  - [RREF Algorithm](#rref-algorithm)
  - [Handling Zero Columns](#handling-zero-columns)
  - [Handling Zero Rows](#handling-zero-rows)
  - [Step-by-Step Example](#step-by-step-example)
  - [Characteristics / Key Points](#characteristics--key-points)
  - [Why is it used? / Applications](#why-is-it-used--applications)
- [Solutions](#solutions)
  - [Custom Implementation](#custom-implementation)
  - [NumPy Equivalent](#numpy-equivalent)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

---

## Problem Statement

### [Implement Reduced Row Echelon Form (RREF) Function](https://www.deep-ml.com/problems/48)

Implement a Python function that converts a given matrix into its **Reduced Row Echelon Form (RREF)**.

The RREF of a matrix satisfies the following conditions:

- Every non-zero row has a leading entry equal to `1`.
- Each leading `1` is the only non-zero element in its column.
- Every leading `1` occurs to the right of the leading `1` in the row above it.
- Rows consisting entirely of zeros appear below non-zero rows.
- A pivot may occur in a column other than the corresponding diagonal position.
- A diagonal entry can be zero when the matrix does not have a pivot at that position.

The function should perform the required elementary row operations and return the RREF of the input matrix.

---

## Example

### Input

```python
import numpy as np

matrix = np.array([
    [1, 2, -1, -4],
    [2, 3, -1, -11],
    [-2, 0, -3, 22]
])

rref_matrix = rref(matrix)

print(rref_matrix)
```

### Output

```text
[
    [ 1.  0.  0. -8.],
    [ 0.  1.  0.  1.],
    [-0. -0.  1. -2.]
]
```

### Reasoning

The algorithm scans the matrix from left to right and searches for a non-zero pivot in each column.

Once a pivot is found:

1. The corresponding row is moved into the current pivot position if necessary.
2. The pivot row is divided by the pivot value so that the pivot becomes `1`.
3. Every other row is modified to make the remaining values in the pivot column equal to zero.
4. The algorithm moves to the next pivot position.

For this matrix, the resulting RREF is

$$
\begin{bmatrix}
1 & 0 & 0 & -8\
0 & 1 & 0 & 1\
0 & 0 & 1 & -2
\end{bmatrix}
$$

Each pivot is `1`, and every other element in a pivot column is zero.

---

## Learn: Understanding Reduced Row Echelon Form

### What is RREF?

**Reduced Row Echelon Form (RREF)** is a canonical form of a matrix obtained through a sequence of elementary row operations.

A matrix is in RREF when every non-zero row has a leading `1`, and that leading `1` is the only non-zero value in its column.

For example,

$$
\begin{bmatrix}
1 & 0 & 3\
0 & 1 & -2\
0 & 0 & 0
\end{bmatrix}
$$

is in RREF.

The first pivot is in column `1`, the second pivot is in column `2`, and every pivot column contains zeros everywhere except at the pivot.

---

## Row Echelon Form vs RREF

**Row Echelon Form (REF)** only requires zeros below each pivot.

For example,

$$
\begin{bmatrix}
1 & 2 & 3\
0 & 1 & 4\
0 & 0 & 1
\end{bmatrix}
$$

is in row echelon form.

However, it is not reduced row echelon form because the pivot columns contain non-zero values above the pivots.

RREF additionally requires zeros both above and below every pivot.

For example,

$$
\begin{bmatrix}
1 & 0 & -5\
0 & 1 & 2\
0 & 0 & 0
\end{bmatrix}
$$

is in RREF.

The distinction is important because Gaussian elimination typically produces REF, while **Gauss-Jordan elimination** continues the elimination process to produce RREF.

---

## Elementary Row Operations

RREF is obtained using three elementary row operations.

### Row Swap

Two rows can be exchanged.

$$
R_i \leftrightarrow R_j
$$

This is useful when the current pivot position contains zero but a non-zero value exists below it.

---

### Row Scaling

A row can be multiplied by a non-zero scalar.

$$
R_i \leftarrow cR_i
$$

where

$$
c \neq 0
$$

To convert a pivot value $p$ into `1`, divide the entire row by $p$.

$$
R_i \leftarrow \frac{R_i}{p}
$$

---

### Row Replacement

A multiple of one row can be added to another row.

$$
R_i \leftarrow R_i-cR_j
$$

This operation is used to eliminate values above and below a pivot.

---

## Mathematical Definition

Consider a matrix

$$
A \in \mathbb{R}^{m\times n}
$$

The RREF of $A$ is a matrix obtained using elementary row operations such that:

1. Every non-zero row has a leading `1`.
2. Each leading `1` is the only non-zero entry in its column.
3. Each pivot is located to the right of the pivot in the row above.
4. Any zero rows occur below all non-zero rows.

For a pivot located at row $r$ and column $c$,

$$
A_{r,c}=1
$$

and for every other row $i$,

$$
A_{i,c}=0
$$

This makes the pivot column a unit vector.

---

## Pivot

A **pivot** is the leading non-zero element of a row after the matrix has been transformed into echelon form.

During RREF computation, every pivot is normalized to `1`.

For example,

$$
\begin{bmatrix}
2 & 4\
0 & 3
\end{bmatrix}
$$

has potential pivots `2` and `3`.

After scaling the rows,

$$
\begin{bmatrix}
1 & 2\
0 & 1
\end{bmatrix}
$$

the pivots are both `1`.

RREF then eliminates the value above the second pivot:

$$
\begin{bmatrix}
1 & 0\
0 & 1
\end{bmatrix}
$$

The number of pivots is the **rank** of the matrix.

---

## RREF Algorithm

The algorithm maintains a current pivot row and scans columns from left to right.

Suppose the current pivot position is `(pivot, col)`.

The algorithm performs the following steps:

### Step 1: Check the Current Pivot

If the current entry is non-zero, it can be used as the pivot.

If it is zero, search rows below for a non-zero value in the same column.

---

### Step 2: Find a Non-Zero Pivot

If

$$
A_{pivot,col}=0
$$

search for some row $r$ below the current pivot row such that

$$
A_{r,col}\neq0
$$

If such a row exists, swap it with the current pivot row.

$$
R_{pivot}\leftrightarrow R_r
$$

If no such row exists, the current column contains no pivot.

The algorithm skips the column and continues to the right.

---

### Step 3: Normalize the Pivot

Suppose the pivot value is $p$.

Divide the complete pivot row by $p$.

$$
R_{pivot}\leftarrow\frac{R_{pivot}}{p}
$$

Now,

$$
A_{pivot,col}=1
$$

---

### Step 4: Eliminate the Pivot Column

For every row other than the pivot row, let the value in the pivot column be $c$.

Perform

$$
R_i\leftarrow R_i-cR_{pivot}
$$

This makes

$$
A_{i,col}=0
$$

for every $i\neq pivot$.

Therefore, the pivot becomes the only non-zero value in its column.

---

### Step 5: Move to the Next Pivot

After processing a pivot, increment the pivot row and move to the next column.

The algorithm continues until either:

- Every row has been processed.
- Every column has been examined.

---

## Handling Zero Columns

A column may contain no non-zero values in the remaining rows.

For example,

$$
\begin{bmatrix}
1 & 2 & 0\
0 & 0 & 0
\end{bmatrix}
$$

The third column contains no pivot.

The algorithm should not attempt to divide by zero.

Instead, it skips the column and searches for the next possible pivot.

This is why the implementation uses a separate column counter.

---

## Handling Zero Rows

Some matrices have linearly dependent rows.

For example,

$$
\begin{bmatrix}
1 & 2\
2 & 4
\end{bmatrix}
$$

The second row is twice the first row.

After elimination,

$$
\begin{bmatrix}
1 & 2\
0 & 0
\end{bmatrix}
$$

The zero row remains below the non-zero row.

More generally, an RREF matrix places all zero rows after the non-zero rows.

---

## Handling a Zero Diagonal Entry

A common misconception is that every diagonal element must become `1`.

That is not required.

For example,

$$
\begin{bmatrix}
0 & 1\
0 & 0
\end{bmatrix}
$$

is already in RREF.

The first diagonal entry is zero because there is no pivot in the first column.

The pivot is instead located at row `1`, column `2`.

Therefore, the algorithm must search columns independently rather than assuming that every diagonal position contains a pivot.

---

## Step-by-Step Example

Consider

$$
A =
\begin{bmatrix}
1 & 2\
2 & 4
\end{bmatrix}
$$

The first pivot is already `1`.

We eliminate the value below it.

$$
R_2\leftarrow R_2-2R_1
$$

giving

$$
\begin{bmatrix}
1 & 2\
0 & 0
\end{bmatrix}
$$

The second row is now a zero row.

There is no remaining pivot, so the final RREF is

$$
\begin{bmatrix}
1 & 2\
0 & 0
\end{bmatrix}
$$

---

## Example with a Row Swap

Consider

$$
A =
\begin{bmatrix}
0 & 2\
1 & 3
\end{bmatrix}
$$

The first pivot position contains zero.

Search below the current row and find the non-zero value `1`.

Swap the rows:

$$
\begin{bmatrix}
1 & 3\
0 & 2
\end{bmatrix}
$$

The first pivot is already `1`.

Now normalize the second row:

$$
R_2\leftarrow\frac{R_2}{2}
$$

giving

$$
\begin{bmatrix}
1 & 3\
0 & 1
\end{bmatrix}
$$

Eliminate the value above the second pivot:

$$
R_1\leftarrow R_1-3R_2
$$

The RREF becomes

$$
\begin{bmatrix}
1 & 0\
0 & 1
\end{bmatrix}
$$

---

## RREF and Linear Systems

RREF is especially useful when solving systems of linear equations.

Consider

$$
\begin{bmatrix}
1 & 1 & 3\
2 & 3 & 8
\end{bmatrix}
$$

where the final column represents the right-hand side.

The matrix can be reduced to

$$
\begin{bmatrix}
1 & 0 & 1\
0 & 1 & 2
\end{bmatrix}
$$

This corresponds to

$$
x_1+x_3=1
$$

and

$$
x_2+2x_3=2
$$

The pivot variables are $x_1$ and $x_2$, while $x_3$ is a free variable.

Thus, RREF makes the structure of the solution directly visible.

---

## RREF and Rank

The number of pivot positions in the RREF is the rank of the matrix.

For example,

$$
\begin{bmatrix}
1 & 2 & 3\
0 & 1 & 4\
0 & 0 & 0
\end{bmatrix}
$$

contains two pivots.

Therefore,

$$
rank(A)=2
$$

RREF therefore provides a direct way to determine the rank through elimination.

---

## Characteristics / Key Points

- RREF is produced using elementary row operations.
- Every pivot is equal to `1`.
- Every pivot is the only non-zero value in its column.
- Pivot positions move strictly to the right as row number increases.
- Zero rows appear at the bottom.
- A diagonal element does not necessarily need to be a pivot.
- Columns without pivots are skipped.
- Rows may need to be swapped to find a non-zero pivot.
- Pivot rows must be normalized before elimination.
- Elimination is performed both above and below each pivot.
- RREF is unique for a given matrix.
- Different sequences of valid row operations produce the same RREF.
- The number of pivots equals the rank of the matrix.
- RREF can reveal free variables in a linear system.
- RREF is a form of Gauss-Jordan elimination.
- Floating-point arithmetic can introduce very small numerical errors.
- Exact symbolic arithmetic may be preferable when exact results are required.

---

## Why is it used? / Applications

RREF is a fundamental operation in linear algebra and is widely used in machine learning and numerical computing.

Applications include

- Solving systems of linear equations.
- Finding matrix rank.
- Identifying linearly independent rows or columns.
- Finding the null space.
- Finding free variables.
- Solving homogeneous systems.
- Computing relationships between variables.
- Understanding linear transformations.
- Checking whether a system has a unique solution.
- Checking whether a system has infinitely many solutions.
- Determining whether a system is inconsistent.
- Basis computation.
- Linear regression derivations.
- Least-squares analysis.
- Understanding matrix structure.

---

> 💡 **Important Note**
>
> Do not assume that the pivot must always lie on the diagonal. If the current column contains only zeros, the algorithm must skip that column and continue searching to the right. This is essential for correctly handling rank-deficient and rectangular matrices.

---

> 💡 **Interview Tip**
>
> The key idea behind RREF is simple: **find a pivot, make it `1`, eliminate its column, then move right and down**. The important implementation detail is handling zero pivot positions by searching below for a non-zero value and swapping rows when necessary.

---

> 💡 **Common Mistake**
>
> Gaussian elimination only eliminates values below pivots to produce row echelon form. RREF requires eliminating values **both above and below** every pivot. Therefore, the elimination loop must process every row except the pivot row.

---

## Solutions

### Custom Implementation

```python
import numpy as np

def rref(mat):
    matrix = mat.copy().astype(float)

    rows, cols = matrix.shape

    pivot = 0
    col = 0

    while pivot < rows and col < cols:

        if matrix[pivot][col] == 0:
            row_ind = -1

            for temp in range(
                pivot + 1,
                rows
            ):
                if matrix[temp][col] != 0:
                    row_ind = temp
                    break

            if row_ind == -1:
                col += 1
                continue

            matrix[[pivot, row_ind]] = (
                matrix[[row_ind, pivot]]
            )

        matrix[pivot] = (
            matrix[pivot]
            / matrix[pivot][col]
        )

        for row in range(rows):
            if row != pivot:
                matrix[row] = (
                    matrix[row]
                    - (
                        matrix[row][col]
                        * matrix[pivot]
                    )
                )

        pivot += 1
        col += 1

    return matrix
```

### NumPy Equivalent

NumPy does not provide a direct `np.rref()` function.

For practical numerical work, libraries such as **SymPy** provide an RREF implementation:

```python
import sympy as sp

matrix = sp.Matrix([
    [1, 2, -1, -4],
    [2, 3, -1, -11],
    [-2, 0, -3, 22]
])

rref_matrix, pivots = matrix.rref()

print(rref_matrix)
print(pivots)
```

The `pivots` result identifies the pivot columns.

For learning the underlying algorithm, however, implementing the row operations manually is useful because it exposes the complete Gauss-Jordan elimination process.

---

## Code Explanation

### Step 1: Copy the Matrix

```python
matrix = mat.copy().astype(float)
```

The original matrix is copied so that the input is not modified.

The matrix is converted to floating-point representation because row normalization can produce fractional values.

---

### Step 2: Store Matrix Dimensions

```python
rows, cols = matrix.shape
```

These values determine the boundaries of the row and column traversal.

Let

- `rows` be the number of rows.
- `cols` be the number of columns.

---

### Step 3: Initialize the Pivot Position

```python
pivot = 0
col = 0
```

`pivot` represents the row where the next pivot should be placed.

`col` represents the column currently being examined.

They are kept separately because a column may contain no pivot.

---

### Step 4: Continue While Pivots Are Possible

```python
while pivot < rows and col < cols:
```

The algorithm stops when either all rows have been assigned pivots or all columns have been examined.

---

### Step 5: Check for a Zero Pivot

```python
if matrix[pivot][col] == 0:
```

If the current position is zero, it cannot be used as a pivot.

The algorithm searches the rows below.

---

### Step 6: Search for a Non-Zero Entry

```python
row_ind = -1

for temp in range(
    pivot + 1,
    rows
):
    if matrix[temp][col] != 0:
        row_ind = temp
        break
```

The search begins below the current pivot row.

The first row containing a non-zero value in the current column is selected.

---

### Step 7: Skip a Column Without a Pivot

```python
if row_ind == -1:
    col += 1
    continue
```

If no non-zero value exists below the current pivot position, the column contains no pivot.

The algorithm moves to the next column without incrementing the pivot row.

This distinction is important for rank-deficient matrices.

---

### Step 8: Swap Rows

```python
matrix[[pivot, row_ind]] = (
    matrix[[row_ind, pivot]]
)
```

The row containing the non-zero value is moved into the current pivot position.

After this operation, the pivot candidate is non-zero.

---

### Step 9: Normalize the Pivot

```python
matrix[pivot] = (
    matrix[pivot]
    / matrix[pivot][col]
)
```

The entire pivot row is divided by the pivot value.

If the pivot was $p$, this performs

$$
R_{pivot}\leftarrow\frac{R_{pivot}}{p}
$$

The pivot therefore becomes

$$
A_{pivot,col}=1
$$

---

### Step 10: Eliminate the Pivot Column

```python
for row in range(rows):
    if row != pivot:
        matrix[row] = (
            matrix[row]
            - (
                matrix[row][col]
                * matrix[pivot]
            )
        )
```

Every row except the pivot row is processed.

If the current value in the pivot column is $c$, the algorithm performs

$$
R_i\leftarrow R_i-cR_{pivot}
$$

This changes the pivot-column entry to zero.

Because the loop processes every other row, zeros are created both above and below the pivot.

This is what makes the result **reduced** row echelon form.

---

### Step 11: Move to the Next Pivot

```python
pivot += 1
col += 1
```

Once the current pivot has been completely processed, the algorithm moves to the next row and next column.

The next pivot must therefore occur below the current pivot and to its right.

---

### Complete Algorithm Flow

```text
RREF(A)

1. Copy A as a floating-point matrix.
2. Set pivot row = 0.
3. For each column:
4.     If current pivot is zero:
5.         Search below for a non-zero entry.
6.         If none exists:
7.             Skip this column.
8.         Otherwise:
9.             Swap rows.
10.    Normalize the pivot row.
11.    Eliminate the pivot column from every other row.
12.    Move to the next pivot row.
13. Return the matrix.
```

The important invariant is that after processing a pivot, its column contains exactly one non-zero element: the pivot `1`.

---

## Time & Space Complexity

Let

- $m$ be the number of rows.
- $n$ be the number of columns.

For each pivot, the algorithm may scan rows to find a non-zero pivot.

After finding the pivot, it performs elimination across every row and every column.

In the general case, the elimination work is bounded by

$$
O(mn\min(m,n))
$$

For a square matrix where

$$
m=n=N
$$

this becomes

$$
O(N^3)
$$

The algorithm modifies a copy of the matrix and does not create a separate matrix proportional to every elimination step.

Therefore, the matrix storage requires

$$
O(mn)
$$

additional space.

| Complexity | Value              |
| ---------- | ------------------ |
| Time       | **O(mn min(m,n))** |
| Space      | **O(mn)**          |

where

- $m$ is the number of rows.
- $n$ is the number of columns.

For an $N\times N$ matrix, the time complexity is **O(N³)** and the space complexity is **O(N²)**.
