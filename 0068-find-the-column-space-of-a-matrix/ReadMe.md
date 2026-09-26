# Find the Column Space of a Matrix (Medium, Linear Algebra)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Matrix Image, Spans, and Column Space](#learn-matrix-image-spans-and-column-space)
  - [What is the Column Space?](#what-is-the-column-space)
  - [Span](#span)
  - [Image of a Matrix](#image-of-a-matrix)
  - [Linear Independence](#linear-independence)
  - [Rank and Column Space](#rank-and-column-space)
  - [Finding Pivot Columns](#finding-pivot-columns)
  - [Why Pivot Columns Must Come from the Original Matrix](#why-pivot-columns-must-come-from-the-original-matrix)
  - [RREF](#rref)
  - [Applications](#applications)

- [Solutions](#solutions)
  - [Custom Implementation](#custom-implementation)
  - [NumPy Implementation](#numpy-implementation)

- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

## Problem Statement

### [Find the column space of a matrix](https://www.deep-ml.com/problems/68)

Implement a function `matrix_image(A)` that calculates the column space, also called the image or span, of a matrix.

The column space consists of every possible linear combination of the columns of the matrix.

Given

\(A=[a_1\ a_2\ \cdots\ a_n]\)

the column space is

\(Col(A)=span\{a_1,a_2,\ldots,a_n\}\)

Not every column necessarily contributes a new independent direction.

The task is therefore to identify the independent columns and return those columns from the **original matrix**.

The returned columns should form a basis for the column space.

## Example

### Input

```python id="h3c1fy"
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

print(matrix_image(matrix))
```

### Output

```text id="qg3x6s"
[[1, 2],
 [4, 5],
 [7, 8]]
```

### Reasoning

The matrix is

$$
A=
\begin{bmatrix}
1&2&3\\
4&5&6\\
7&8&9
\end{bmatrix}
$$

Its three columns are

$$
a_1=
\begin{bmatrix}
1\\4\\7
\end{bmatrix}
$$

$$
a_2=
\begin{bmatrix}
2\\5\\8
\end{bmatrix}
$$

$$
a_3=
\begin{bmatrix}
3\\6\\9
\end{bmatrix}
$$

The third column is linearly dependent on the first two.

In fact,

\(a_3=2a_2-a_1\)

Therefore, only the first two columns are required to span the entire column space.

The rank is therefore:

\(rank(A)=2\)

A basis for the column space is:

$$
\left\{
\begin{bmatrix}
1\\4\\7
\end{bmatrix},
\begin{bmatrix}
2\\5\\8
\end{bmatrix}
\right\}
$$

The implementation returns these columns from the original matrix.

## Learn: Matrix Image, Spans, and Column Space

### What is the Column Space?

The column space of a matrix is the set of every vector that can be produced by taking a linear combination of its columns.

If

\(A=[a_1\ a_2\ \cdots\ a_n]\)

then:

\(Col(A)=span\{a_1,a_2,\ldots,a_n\}\)

For a matrix with $m$ rows, every column contains $m$ elements.

Therefore:

\(Col(A)\subseteq\mathbb{R}^m\)

The column space is a subspace of the output space.

### Matrix as a Linear Transformation

A matrix can be interpreted as a function.

\(T(x)=Ax\)

Suppose $A$ is an $m\times n$ matrix.

Then:

\(A:\mathbb{R}^n\rightarrow\mathbb{R}^m\)

For

$$
x=
\begin{bmatrix}
x_1\\
x_2\\
\vdots\\
x_n
\end{bmatrix}
$$

we have:

\(Ax=x_1a_1+x_2a_2+\cdots+x_na_n\)

Therefore, every output produced by $A$ is a linear combination of its columns.

This means:

\(Im(A)=Col(A)\)

The image of the linear transformation is exactly the column space of the matrix.

### Span

The span of a set of vectors is the collection of all possible linear combinations of those vectors.

For vectors $v_1,\ldots,v_k$:

\(span\{v_1,\ldots,v_k\}=\{c_1v_1+\cdots+c_kv_k:c_i\in\mathbb{R}\}\)

For the example matrix:

\(Col(A)=span\{a_1,a_2,a_3\}\)

Even though three columns are present, only two are needed because the third is dependent.

Thus:

\(Col(A)=span\{a_1,a_2\}\)

### Linear Independence

Vectors are linearly independent when the only solution to

\(c_1v_1+c_2v_2+\cdots+c_kv_k=0\)

is

\(c_1=c_2=\cdots=c_k=0\)

If one vector can be written as a combination of the others, the vectors are linearly dependent.

For example:

\(a_3=2a_2-a_1\)

means that $a_3$ does not add a new direction.

Therefore, the set ${a_1,a_2,a_3}$ is linearly dependent.

### Basis of the Column Space

A basis is a set of linearly independent vectors that spans the entire space.

Therefore, a basis for the column space must satisfy two conditions:

1. The vectors are linearly independent.
2. Their span equals the entire column space.

The pivot columns of the original matrix provide such a basis.

If a matrix has rank $r$, its column space has dimension $r$.

Therefore, a basis for the column space contains exactly $r$ vectors.

### Rank and Column Space

The rank of a matrix is the dimension of its column space.

\(rank(A)=dim(Col(A))\)

For the example:

\(rank(A)=2\)

Therefore:

\(dim(Col(A))=2\)

This means the column space needs exactly two independent vectors to form a basis.

The rank also equals the dimension of the row space.

Thus:

\(rank(A)=dim(Col(A))=dim(Row(A))\)

### Rank as Number of Pivot Columns

When a matrix is reduced to row-echelon form or RREF, the number of pivot positions equals the rank.

Therefore:

\(rank(A)=\text{number of pivot columns}\)

This gives us a practical method for finding a basis for the column space.

1. Reduce $A$ to RREF.
2. Identify the pivot columns.
3. Take those same columns from the original $A$.

### Why RREF Helps

Suppose:

$$
A=
\begin{bmatrix}
1&2&3\\
4&5&6\\
7&8&9
\end{bmatrix}
$$

Its RREF is:

$$
R=
\begin{bmatrix}
1&0&-1\\
0&1&2\\
0&0&0
\end{bmatrix}
$$

The leading ones occur in columns $1$ and $2$.

Therefore, columns $1$ and $2$ are pivot columns.

The third column is non-pivot.

The important point is that RREF tells us **which column indices are independent**, but the basis vectors themselves must come from the original matrix.

### Finding Pivot Columns

A pivot is a leading non-zero entry in a row-echelon representation.

In RREF, every pivot column contains a unit vector.

For the matrix:

$$
R=
\begin{bmatrix}
1&0&-1\\
0&1&2\\
0&0&0
\end{bmatrix}
$$

the first column is:

$$
\begin{bmatrix}
1\\0\\0
\end{bmatrix}
$$

and the second column is:

$$
\begin{bmatrix}
0\\1\\0
\end{bmatrix}
$$

These are pivot columns.

The third column:

$$
\begin{bmatrix}
-1\\2\\0
\end{bmatrix}
$$

is not a pivot column.

### Pivot Columns vs Non-Pivot Columns

A pivot column corresponds to an independent direction.

A non-pivot column can be expressed as a linear combination of pivot columns.

For the example:

\(a_3=2a_2-a_1\)

Therefore, $a_3$ does not need to be included in a basis.

The original matrix may contain many redundant columns.

RREF exposes this redundancy.

### Why Pivot Columns Must Come from the Original Matrix

This is one of the most important details in the problem.

Suppose:

$$
A=
\begin{bmatrix}
1&2&3\\
4&5&6\\
7&8&9
\end{bmatrix}
$$

After row reduction:

$$
R=
\begin{bmatrix}
1&0&-1\\
0&1&2\\
0&0&0
\end{bmatrix}
$$

The pivot columns of $R$ are:

$$
\begin{bmatrix}
1\\0\\0
\end{bmatrix},
\begin{bmatrix}
0\\1\\0
\end{bmatrix}
$$

These vectors are generally **not** the basis vectors of the original column space.

Instead, they identify the corresponding columns in the original matrix.

Therefore, use the pivot indices:

\([0,1]\)

and extract:

\(A[:,[0,1]]\)

This gives:

$$
\begin{bmatrix}
1&2\\
4&5\\
7&8
\end{bmatrix}
$$

which is a basis for the original column space.

### Row Operations Preserve Row Space, Not Column Vectors

Elementary row operations change the actual columns of a matrix.

Therefore, row reduction does not preserve the original column vectors.

However, row operations preserve the relationships needed to determine which columns are pivot columns.

This is why RREF is used to identify pivot **positions**, while the actual basis vectors are extracted from the original matrix.

### RREF

Reduced Row-Echelon Form is obtained by applying elementary row operations until the matrix satisfies stronger conditions than ordinary row-echelon form.

For every pivot:

- The pivot is $1$.
- It is the only non-zero entry in its column.
- Each pivot moves to the right as rows increase.
- Zero rows appear at the bottom.

For example:

$$
\begin{bmatrix}
1&0&2\\
0&1&-3\\
0&0&0
\end{bmatrix}
$$

is in RREF.

### RREF Algorithm

A typical Gauss-Jordan procedure maintains:

- A current pivot row.
- A current column.
- A search for a non-zero pivot.
- Row swapping when necessary.
- Pivot normalization.
- Elimination above and below the pivot.

The general process is:

```text
pivot_row = 0

for each column:

    find a non-zero entry at or below pivot_row

    if none exists:
        continue to next column

    swap the selected row with pivot_row

    divide pivot row by pivot

    eliminate this column from every other row

    increment pivot_row
```

The implementation in this problem follows this general structure.

### RREF and Numerical Precision

Exact zero comparisons are appropriate for symbolic mathematics but can be unreliable with floating-point values.

For example, a mathematically zero value may become:

\(-2.2\times10^{-16}\)

after floating-point operations.

Therefore, numerical implementations often use a tolerance.

For example:

```python id="juxjcc"
np.abs(value) > 1e-10
```

can be used to determine whether a value should be treated as non-zero.

This is why the solution uses a tolerance when identifying pivot columns.

### Column Space Dimension

If $A$ is an $m\times n$ matrix and has rank $r$, then:

\(dim(Col(A))=r\)

The dimension cannot exceed either matrix dimension:

\(r\leq\min(m,n)\)

Therefore:

\(dim(Col(A))\leq m\)

and

\(dim(Col(A))\leq n\)

For a matrix with more columns than rows, some columns must necessarily be linearly dependent.

### Full Column Rank

An $m\times n$ matrix has full column rank when:

\(rank(A)=n\)

This requires:

\(n\leq m\)

Every column is then independent.

The column space has dimension $n$.

### Full Row Rank

An $m\times n$ matrix has full row rank when:

\(rank(A)=m\)

If $m\leq n$, the column space fills the entire codomain:

\(Col(A)=\mathbb{R}^m\)

Every possible output vector can then be represented as $Ax$.

### Square Full-Rank Matrix

For an $n\times n$ matrix:

\(rank(A)=n\)

means all columns are linearly independent.

Therefore, all $n$ columns form a basis for $\mathbb{R}^n$.

The matrix is also invertible.

Equivalent conditions include:

\(det(A)\neq0\)

and

\(A^{-1}\text{ exists}\)

and

\(rank(A)=n\)

### Connection to Linear Systems

The column space directly determines whether the system

\(Ax=b\)

has a solution.

A solution exists exactly when:

\(b\in Col(A)\)

In other words, $b$ must be expressible as a linear combination of the columns of $A$.

If:

\(b\notin Col(A)\)

then the system has no solution.

This makes column space fundamental to understanding linear systems.

### Connection to the Null Space

The null space is defined as:

\(Null(A)=\{x:Ax=0\}\)

The column space describes possible outputs.

The null space describes inputs that produce the zero output.

For an $m\times n$ matrix:

\(rank(A)+nullity(A)=n\)

This is the rank-nullity theorem.

Therefore:

\(dim(Col(A))+dim(Null(A))=n\)

### Column Space and Data

A matrix can represent a collection of data vectors as columns.

If many columns are linearly dependent, the data contains redundant directions.

The rank measures the number of independent directions represented by the data.

This idea is closely related to dimensionality reduction.

### Applications

The column space has important applications in:

- Solving linear systems.
- Determining matrix rank.
- Understanding linear transformations.
- Dimensionality reduction.
- Data compression.
- Computer graphics.
- Signal processing.
- Least-squares problems.
- Numerical linear algebra.
- Machine learning.

In machine learning, matrix rank can reveal redundancy in feature representations and parameter matrices.

### Important Distinction

The column space itself is not simply a list of independent columns.

The column space is the **entire span** generated by those columns.

For example:

\(Col(A)=span\{a_1,a_2\}\)

The vectors $a_1$ and $a_2$ form a basis.

The column space contains infinitely many vectors:

\(c_1a_1+c_2a_2\)

for arbitrary scalars $c_1,c_2$.

Therefore, `matrix_image(A)` returns a basis representation of the column space rather than explicitly enumerating the entire space.

> 💡 **Important Note**
>
> The pivot columns must be extracted from the original matrix, not from the RREF matrix. RREF identifies which column positions are pivots, but its transformed columns generally are not the original column-space basis vectors.

> 💡 **Implementation Insight**
>
> Your solution correctly follows the two-stage idea: compute RREF to identify pivot positions, then use those positions to slice the original matrix. This is the key conceptual step of the problem.

## Solutions

### Custom Implementation

```python id="w7k9p2"
import numpy as np

def rref(mat):

    matrix = mat.copy().astype(float)

    rows, cols = matrix.shape

    pivot = 0

    col = 0

    while pivot < rows and col < cols:

        if matrix[pivot][col] == 0:

            row_ind = -1

            for temp in range(pivot + 1, rows):

                if matrix[temp][col] != 0:

                    row_ind = temp

                    break

            if row_ind == -1:

                col += 1

                continue

            matrix[[pivot, row_ind]] = matrix[[row_ind, pivot]]

        matrix[pivot] = matrix[pivot] / matrix[pivot][col]

        for row in range(rows):

            if row != pivot:

                matrix[row] = matrix[row] - (matrix[row][col] * matrix[pivot])

        pivot += 1

        col += 1

    return matrix


def matrix_image(A):

    r = rref(A)

    pivot_cols = []

    for j in range(r.shape[1]):

        col = r[:, j]

        if np.count_nonzero(np.abs(col) > 1e-10) == 1 and np.isclose(col.max(), 1):

            pivot_cols.append(j)

    return A[:, pivot_cols]
```

### NumPy Implementation

For practical numerical linear algebra, the pivot-column problem can be handled more robustly using matrix rank and QR/SVD-based methods.

A direct RREF routine is not provided as a core NumPy operation because numerical linear algebra libraries generally prefer decompositions such as QR or SVD.

For example, an SVD can reveal the numerical rank:

```python id="0e3x8q"
import numpy as np

def matrix_rank(A):
    return np.linalg.matrix_rank(A)
```

However, `np.linalg.matrix_rank` returns only the rank.

It does not directly return the original pivot columns.

For this problem, the custom RREF approach is therefore useful because the required output is specifically the original independent columns.

## Code Explanation

### 1. Copy the Matrix

```python id="b0pqg8"
matrix = mat.copy().astype(float)
```

The input matrix is copied so that row operations do not modify the original matrix.

This is especially important because `matrix_image` must eventually return columns from the original matrix `A`.

The conversion to floating point is necessary because RREF requires division.

### 2. Track Matrix Dimensions

```python id="x3y6vv"
rows, cols = matrix.shape
```

The algorithm supports rectangular matrices.

`rows` is the number of equations or row vectors.

`cols` is the number of columns.

The number of pivot columns cannot exceed:

\(\min(rows,cols)\)

### 3. Track the Current Pivot

```python id="3g8jwk"
pivot = 0
col = 0
```

`pivot` identifies the row where the next pivot should be placed.

`col` identifies the column currently being inspected.

The algorithm advances through columns while pivot rows advance only when a pivot is found.

### 4. Find a Non-Zero Pivot

```python id="p3x1qz"
if matrix[pivot][col] == 0:
```

If the current candidate is zero, the implementation searches lower rows for a non-zero value in the same column.

```python id="6s3kaf"
row_ind = -1

for temp in range(pivot + 1, rows):

    if matrix[temp][col] != 0:

        row_ind = temp

        break
```

If a non-zero entry is found, that row can be moved into the pivot position.

### 5. Swap Rows

```python id="q2y7vf"
matrix[[pivot, row_ind]] = matrix[[row_ind, pivot]]
```

The selected row becomes the current pivot row.

Row swapping is an elementary row operation and preserves the row-equivalence class of the matrix.

### 6. Skip a Zero Column

If no non-zero entry exists below the current pivot position:

```python id="w4n8mz"
if row_ind == -1:

    col += 1

    continue
```

This means the current column does not contain a pivot at this stage.

The algorithm therefore moves to the next column without advancing the pivot row.

This distinction is important for rank-deficient and rectangular matrices.

### 7. Normalize the Pivot

```python id="v9r3kc"
matrix[pivot] = matrix[pivot] / matrix[pivot][col]
```

The entire pivot row is divided by the pivot value.

Therefore, the pivot becomes:

\(1\)

This is one of the defining properties of RREF.

### 8. Eliminate the Pivot Column

```python id="m8d4zq"
for row in range(rows):

    if row != pivot:

        matrix[row] = matrix[row] - (matrix[row][col] * matrix[pivot])
```

Unlike ordinary Gaussian Elimination, which only eliminates entries below a pivot, RREF eliminates entries both above and below it.

For another row $i$:

\(R*i\leftarrow R_i-A*{ic}R_p\)

After this operation, the pivot column contains zeros everywhere except at the pivot.

### 9. Move to the Next Pivot

```python id="r5t2vk"
pivot += 1
col += 1
```

Once a pivot has been established, both the pivot row and column advance.

The process continues until either all rows or all columns have been processed.

### 10. Return the RREF

```python id="n1q6cy"
return matrix
```

The helper function returns the reduced matrix.

For the example, the result is equivalent to:

```text id="m5j8qa"
[[1, 0, -1],
 [0, 1,  2],
 [0, 0,  0]]
```

### 11. Compute RREF of the Original Matrix

```python id="u8f2pd"
r = rref(A)
```

The original matrix is reduced to RREF.

The resulting matrix is used only to determine pivot locations.

The original matrix is retained separately.

### 12. Search for Pivot Columns

```python id="k3v9ha"
pivot_cols = []

for j in range(r.shape[1]):

    col = r[:, j]
```

The implementation examines every column of the RREF.

A pivot column in RREF contains exactly one non-zero element, and that element is the pivot value $1$.

### 13. Identify Unit Pivot Columns

```python id="d2p7mc"
if np.count_nonzero(np.abs(col) > 1e-10) == 1 and np.isclose(col.max(), 1):

    pivot_cols.append(j)
```

The first condition checks whether only one element is significantly non-zero.

The tolerance:

\(10^{-10}\)

helps account for floating-point noise.

The second condition verifies that the non-zero element is approximately $1$.

For the example:

$$
\begin{bmatrix}
1\\0\\0
\end{bmatrix}
$$

and

$$
\begin{bmatrix}
0\\1\\0
\end{bmatrix}
$$

are recognized as pivot columns.

### 14. Extract Original Columns

```python id="y4m1xk"
return A[:, pivot_cols]
```

This is the most important line in the entire solution.

The pivot indices were discovered using RREF.

But the vectors are extracted from the original matrix.

Therefore, the returned columns remain actual vectors from the original column space.

For the example:

```python
pivot_cols = [0, 1]
```

so:

```python
A[:, pivot_cols]
```

returns:

```text
[[1, 2],
 [4, 5],
 [7, 8]]
```

### Why the Algorithm Works

The correctness follows from the relationship between pivot columns and linear independence.

RREF preserves the rank of the matrix.

The number of pivots equals the rank.

Each pivot corresponds to an independent original column.

Therefore, if the pivot indices are:

\(p_1,p_2,\ldots,p_r\)

then:

\(\{A*{:,p_1},A*{:,p*2},\ldots,A*{:,p_r}\}\)

forms a basis for the column space.

Consequently:

\(Col(A)=span\{A*{:,p_1},\ldots,A*{:,p_r}\}\)

### Important Numerical Caveat

The RREF implementation checks:

```python
matrix[pivot][col] == 0
```

This is an exact floating-point comparison.

For numerical data, a tolerance-based check is generally safer:

```python id="p6t1wx"
np.isclose(matrix[pivot, col], 0, atol=1e-10)
```

Similarly, the pivot search can use a tolerance.

The current implementation works well for the intended educational problem, but a production numerical implementation should be more careful about floating-point thresholds.

### Another Important Caveat

The implementation identifies RREF pivot columns using the property that a pivot column has exactly one significant non-zero value equal to $1$.

This works because `rref()` explicitly constructs reduced row-echelon form.

If the helper produced only ordinary row-echelon form, this detection method would not be reliable.

For ordinary REF, pivot columns may contain non-zero entries above the pivot.

### Alternative Pivot Detection

A more explicit approach is to identify the leading non-zero position in each non-zero row.

For each row:

1. Find its first significant non-zero element.
2. Record its column index.
3. Those indices are pivot columns.

Conceptually:

```python id="e7s4bd"
pivot_cols = []

for row in r:
    nonzero = np.flatnonzero(np.abs(row) > 1e-10)

    if len(nonzero) > 0:
        pivot_cols.append(nonzero[0])
```

This approach is often easier to reason about because it directly follows the definition of a pivot.

The current implementation's unit-column detection is valid for a correctly computed RREF.

## Time & Space Complexity

Let $A$ be an $m\times n$ matrix.

### RREF Complexity

The Gauss-Jordan elimination process examines pivot positions and performs row operations across the matrix.

For a general $m\times n$ matrix, the complexity is approximately:

\(O(mn\min(m,n))\)

For a square $n\times n$ matrix:

\(O(n^3)\)

### Pivot Column Detection

After RREF has been computed, every column is inspected.

There are $n$ columns and up to $m$
