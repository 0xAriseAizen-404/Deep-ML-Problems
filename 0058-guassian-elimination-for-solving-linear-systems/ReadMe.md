# Gaussian Elimination for Solving Linear Systems (Medium, Linear Algebra)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Gaussian Elimination](#learn-gaussian-elimination)
  - [What is Gaussian Elimination?](#what-is-gaussian-elimination)
  - [Linear System Representation](#linear-system-representation)
  - [Augmented Matrix](#augmented-matrix)
  - [Row-Echelon Form](#row-echelon-form)
  - [Partial Pivoting](#partial-pivoting)
  - [Elimination Step](#elimination-step)
  - [Backward Substitution](#backward-substitution)
  - [Numerical Stability](#numerical-stability)
  - [Applications](#applications)

- [Solutions](#solutions)
  - [Custom Implementation](#custom-implementation)
  - [NumPy Implementation](#numpy-implementation)

- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

## Problem Statement

### [Gaussian Elimination for Solving Linear Systems](https://www.deep-ml.com/problems/58)

Implement the Gaussian Elimination method to solve a system of linear equations:

\(Ax=b\)

The method transforms the coefficient matrix $A$ into an upper triangular matrix using elementary row operations.

Once the matrix is triangular, the unknown variables can be recovered using backward substitution.

The implementation must use **partial pivoting** to improve numerical stability and avoid division by zero when a suitable non-zero pivot exists.

The function should return the solution vector $x$.

## Example

### Input

```python
A = np.array([[2, 8, 4], [2, 5, 1], [4, 10, -1]], dtype=float)
b = np.array([2, 5, 1], dtype=float)

print(gaussian_elimination(A, b))
```

### Output

```text
[11.0, -4.0, 3.0]
```

### Reasoning

The original system is

\(2x_1+8x_2+4x_3=2\)

\(2x_1+5x_2+x_3=5\)

\(4x_1+10x_2-x_3=1\)

Gaussian Elimination performs row operations to eliminate the entries below each pivot.

The resulting coefficient matrix is upper triangular.

An upper triangular system has the form

\(Ux=c\)

where entries below the diagonal are zero.

The final variable can therefore be solved immediately.

The remaining variables are then solved from bottom to top using backward substitution.

Partial pivoting is performed before each elimination step by selecting the row containing the largest absolute value in the current pivot column.

## Learn: Gaussian Elimination

### What is Gaussian Elimination?

Gaussian Elimination is a fundamental algorithm for solving systems of linear equations.

Given

\(Ax=b\)

the algorithm transforms $A$ into an upper triangular matrix $U$ while applying the same row operations to $b$.

The transformed system becomes

\(Ux=c\)

where $U$ has zeros below its main diagonal.

The variables can then be obtained efficiently using backward substitution.

The algorithm consists of two major phases:

1. Forward elimination
2. Backward substitution

Forward elimination transforms the matrix.

Backward substitution extracts the solution.

### Linear System Representation

A system of $n$ linear equations with $n$ unknowns can be represented as

\(Ax=b\)

where:

- $A$ is the $n\times n$ coefficient matrix.
- $x$ is the unknown solution vector.
- $b$ is the right-hand-side vector.

For example,

$$
A=
\begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1n}\\
a_{21} & a_{22} & \cdots & a_{2n}\\
\vdots & \vdots & \ddots & \vdots\\
a_{n1} & a_{n2} & \cdots & a_{nn}
\end{bmatrix}
$$

and

$$
x=
\begin{bmatrix}
x_1\\
x_2\\
\vdots\\
x_n
\end{bmatrix}
$$

with

$$
b=
\begin{bmatrix}
b_1\\
b_2\\
\vdots\\
b_n
\end{bmatrix}
$$

The objective is to determine the vector $x$.

### Augmented Matrix

The coefficient matrix and right-hand-side vector can be combined into an augmented matrix.

\([A|b]\)

For a $3\times3$ system:

$$
[A|b]=
\begin{bmatrix}
a_{11} & a_{12} & a_{13} & | & b_1\\
a_{21} & a_{22} & a_{23} & | & b_2\\
a_{31} & a_{32} & a_{33} & | & b_3
\end{bmatrix}
$$

Gaussian Elimination operates on this augmented system.

Whenever a row operation is applied to a row of $A$, the corresponding element of $b$ must receive the same operation.

This preserves the original system of equations.

### Elementary Row Operations

Gaussian Elimination relies on three elementary row operations.

#### Row Swapping

Two rows can be exchanged.

\(R_i\leftrightarrow R_j\)

This is used by partial pivoting.

#### Row Scaling

A row can be multiplied by a non-zero scalar.

\(R_i\leftarrow cR_i\)

#### Row Replacement

A multiple of one row can be subtracted from another.

\(R_i\leftarrow R_i-cR_j\)

The implementation mainly uses row swapping and row replacement.

### Row-Echelon Form

The goal of forward elimination is to produce an upper triangular matrix.

A matrix is in row-echelon form when:

- All non-zero rows appear above zero rows.
- Each leading entry moves to the right as we move downward.
- Entries below each pivot are zero.

For example:

$$
\begin{bmatrix}
a & b & c\\
0 & d & e\\
0 & 0 & f
\end{bmatrix}
$$

is upper triangular.

The pivots are approximately the diagonal elements:

\(a,\ d,\ f\)

The version of row-echelon form described in the problem also mentions leading entries being $1$.

However, standard Gaussian Elimination does not need to normalize every pivot to $1$.

It is enough to eliminate the entries below each pivot.

Normalizing pivots to $1$ is more closely associated with reduced row-echelon form or certain implementations of elimination.

### Upper Triangular Matrix

After forward elimination, the system has the structure

$$
\begin{bmatrix}
u_{11} & u_{12} & u_{13}\\
0 & u_{22} & u_{23}\\
0 & 0 & u_{33}
\end{bmatrix}
\begin{bmatrix}
x_1\\
x_2\\
x_3
\end{bmatrix}
=
\begin{bmatrix}
c_1\\
c_2\\
c_3
\end{bmatrix}
$$

The final equation contains only one unknown:

\(u\_{33}x_3=c_3\)

Therefore,

\(x*3=\frac{c_3}{u*{33}}\)

The previous equation contains $x_2$ and the already known $x_3$.

The process continues upward.

### Partial Pivoting

A pivot is the matrix element currently being used to eliminate entries below it.

Without pivoting, the algorithm could encounter a zero pivot.

For example:

$$
\begin{bmatrix}
0 & 2\\
3 & 4
\end{bmatrix}
$$

The first pivot is zero.

Dividing by it would be impossible.

Partial pivoting fixes this by searching the current column for the row with the largest absolute value.

For pivot column $k$:

\(p=\arg\max*{i\geq k}|A*{ik}|\)

The selected row is exchanged with the current pivot row.

This produces a non-zero pivot whenever the remaining column contains a non-zero entry.

### Why Use the Largest Absolute Pivot?

Suppose the pivot is extremely small:

\(|A\_{kk}|\approx0\)

The elimination factor

\(m*{ik}=\frac{A*{ik}}{A\_{kk}}\)

can become extremely large.

Large elimination factors can amplify floating-point rounding errors.

Partial pivoting attempts to avoid unnecessarily small pivots by selecting the largest available magnitude in the current column.

Therefore, partial pivoting generally improves numerical stability.

It does not make floating-point computation perfectly exact.

### Forward Elimination

For each pivot position $k$, rows below the pivot are modified.

The elimination factor for row $i$ is

\(m*{ik}=\frac{A*{ik}}{A\_{kk}}\)

The row update is

\(R*i\leftarrow R_i-m*{ik}R_k\)

The right-hand side must be updated simultaneously:

\(b*i\leftarrow b_i-m*{ik}b_k\)

After this operation,

\(A\_{ik}=0\)

for the current pivot column.

Repeating this for every pivot produces an upper triangular matrix.

### Elimination Process

Consider

$$
A=
\begin{bmatrix}
2 & 8 & 4\\
2 & 5 & 1\\
4 & 10 & -1
\end{bmatrix}
$$

At the first pivot, the values in column zero are

\(2,\ 2,\ 4\)

Partial pivoting selects the third row because

\(|4|>|2|\)

The rows are therefore rearranged before elimination.

The new first pivot is $4$.

For the second row, the elimination factor is

\(m\_{21}=\frac{2}{4}=0.5\)

For the third row, the elimination factor is

\(m\_{31}=\frac{2}{4}=0.5\)

The corresponding row operations eliminate the first-column entries below the pivot.

The process then continues using the second column as the next pivot column.

### Generic Gaussian Elimination Algorithm

For each pivot column $k$:

1. Find the row with the largest absolute entry in column $k$.
2. Swap that row with row $k$.
3. Use $A_{kk}$ as the pivot.
4. Eliminate entries below the pivot.
5. Update both $A$ and $b$.

In pseudocode:

```text
for k = 0 ... n-1:

    pivot_row = row with largest |A[row][k]|

    swap rows k and pivot_row

    for i = k+1 ... n-1:

        factor = A[i][k] / A[k][k]

        for j = k ... n-1:
            A[i][j] = A[i][j] - factor * A[k][j]

        b[i] = b[i] - factor * b[k]
```

After this process, $A$ is upper triangular.

### Backward Substitution

Once forward elimination has produced

\(Ux=c\)

the variables can be solved starting from the final row.

For row $i$:

\(u*{ii}x_i+\sum*{j=i+1}^{n-1}u\_{ij}x_j=c_i\)

Rearranging gives

\(x*i=\frac{c_i-\sum*{j=i+1}^{n-1}u*{ij}x_j}{u*{ii}}\)

The implementation processes rows from bottom to top.

```text
for i = n-1 ... 0:

    x[i] = (b[i] - sum(A[i][j] * x[j])) / A[i][i]
```

At each iteration, all variables to the right of $x_i$ have already been computed.

### Why Backward Substitution Works

Consider:

$$
\begin{aligned}
u_{11}x_1+u_{12}x_2+u_{13}x_3 &= c_1\\
u_{22}x_2+u_{23}x_3 &= c_2\\
u_{33}x_3 &= c_3
\end{aligned}
$$

The third equation contains only $x_3$.

Therefore:

\(x*3=\frac{c_3}{u*{33}}\)

Substitute $x_3$ into the second equation:

\(x*2=\frac{c_2-u*{23}x*3}{u*{22}}\)

Finally:

\(x*1=\frac{c_1-u*{12}x*2-u*{13}x*3}{u*{11}}\)

This is the fundamental idea behind backward substitution.

### Complete Mathematical Formulation

For pivot $k$, select

\(p=\arg\max*{i=k,\ldots,n-1}|A*{ik}|\)

Swap rows $k$ and $p$.

Then for every row $i>k$:

\(m*{ik}=\frac{A*{ik}}{A\_{kk}}\)

Update the coefficients:

\(A*{ij}\leftarrow A*{ij}-m*{ik}A*{kj}\)

for $j\geq k$.

Update the right-hand side:

\(b*i\leftarrow b_i-m*{ik}b_k\)

After elimination, solve:

\(x*i=\frac{b_i-\sum*{j=i+1}^{n-1}A*{ij}x_j}{A*{ii}}\)

for $i=n-1,\ldots,0$.

### Numerical Stability

Floating-point numbers cannot represent every real number exactly.

As a result, operations such as division and subtraction can introduce rounding errors.

Gaussian Elimination without pivoting can be especially sensitive when pivots are very small.

Partial pivoting reduces this problem by selecting a relatively large pivot from the current column.

However, partial pivoting does not guarantee perfect numerical stability for every possible matrix.

For difficult numerical problems, specialized linear algebra libraries are preferred.

### Singular Systems

If every possible pivot in a column is zero, the matrix may be singular or rank-deficient.

A singular matrix does not have a unique solution.

Possible outcomes include:

- No solution.
- Infinitely many solutions.

The provided implementation assumes the system has a unique solution.

A production-quality solver should explicitly check whether the selected pivot is sufficiently close to zero.

For floating-point computations, checking

\(|A\_{kk}|<\epsilon\)

is generally safer than checking

\(A\_{kk}=0\)

exactly.

### Unique Solution

A square system has a unique solution when $A$ is invertible.

An important equivalent condition is:

\(\det(A)\neq0\)

Another equivalent condition is:

\(rank(A)=n\)

for an $n\times n$ square matrix.

During Gaussian Elimination, this corresponds to obtaining a non-zero pivot at every elimination stage.

### Gaussian Elimination vs Gauss-Jordan Elimination

Gaussian Elimination transforms the matrix into upper triangular form.

Gauss-Jordan Elimination continues further and eliminates entries both above and below every pivot.

Gaussian Elimination therefore uses backward substitution after forward elimination.

Gauss-Jordan can directly produce reduced row-echelon form.

For solving a single system, Gaussian Elimination generally avoids unnecessary elimination work.

### Gaussian Elimination vs Matrix Inverse

A system

\(Ax=b\)

does not normally need to be solved by explicitly computing

\(x=A^{-1}b\)

Although mathematically valid, explicitly computing the inverse is usually more expensive and can be less numerically desirable.

Gaussian Elimination solves the system directly.

This is particularly important when solving systems numerically.

### LU Decomposition Connection

Gaussian Elimination is closely related to LU decomposition.

The matrix can conceptually be factored as

\(PA=LU\)

where:

- $P$ is a permutation matrix caused by pivoting.
- $L$ is a lower triangular matrix.
- $U$ is an upper triangular matrix.

Then

\(PAx=Pb\)

becomes

\(LUx=Pb\)

which can be solved using:

1. Forward substitution for $Ly=Pb$.
2. Backward substitution for $Ux=y$.

This connection is important because LU factorization allows multiple systems with the same matrix $A$ but different vectors $b$ to be solved efficiently.

### Applications

Gaussian Elimination is a fundamental building block in numerical computing.

Applications include:

- Machine learning.
- Solving linear regression systems.
- Numerical optimization.
- Computational fluid dynamics.
- Physics simulations.
- Electrical circuit analysis.
- Structural engineering.
- 3D graphics.
- Computer vision.
- Scientific computing.

Many higher-level numerical algorithms ultimately rely on matrix factorization or elimination techniques.

> 💡 **Important Note**
>
> Do not confuse Gaussian Elimination with simply "turning every pivot into 1." The essential operation is eliminating entries below each pivot. Pivot normalization is optional for solving the system and is more characteristic of Gauss-Jordan-style implementations.

> 💡 **Implementation Insight**
>
> The user's implementation correctly combines partial pivoting, forward elimination, and backward substitution. The important invariant is that every row operation applied to `A` must also be applied to the corresponding element of `b`.

## Solutions

### Custom Implementation

```python
import numpy as np

def gaussian_elimination(A, b):

    A = A.astype(float)

    b = b.astype(float)

    n = len(b)

    for pivot in range(n):

        max_row = pivot + np.argmax(np.abs(A[pivot:, pivot]))

        A[[pivot, max_row]] = A[[max_row, pivot]]

        b[[pivot, max_row]] = b[[max_row, pivot]]

        for row in range(pivot + 1, n):

            factor = A[row, pivot] / A[pivot, pivot]

            A[row] -= factor * A[pivot]

            b[row] -= factor * b[pivot]

    x = np.zeros(n)

    for i in range(n - 1, -1, -1):

        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]

    return x
```

### NumPy Implementation

For practical numerical work, NumPy provides higher-level linear algebra routines.

```python
import numpy as np

def solve_system(A, b):
    return np.linalg.solve(A, b)
```

`np.linalg.solve` is preferable to explicitly computing the inverse.

Instead of calculating:

```python
np.linalg.inv(A) @ b
```

use:

```python
np.linalg.solve(A, b)
```

The latter directly solves the linear system.

## Code Explanation

### 1. Convert Inputs to Floating Point

```python
A = A.astype(float)
b = b.astype(float)
```

Gaussian Elimination requires division when calculating elimination factors.

Converting to floating point prevents integer arithmetic from destroying fractional results.

For example, an elimination factor may be:

\(m\_{ik}=0.5\)

Integer arithmetic would not represent this correctly.

### 2. Determine the System Size

```python
n = len(b)
```

For an $n\times n$ system, `b` contains $n$ equations.

The implementation assumes a square coefficient matrix with the same number of rows as elements in `b`.

### 3. Find the Pivot Row

```python
max_row = pivot + np.argmax(np.abs(A[pivot:, pivot]))
```

`A[pivot:, pivot]` selects the remaining entries in the current pivot column.

`np.abs` compares their magnitudes.

`np.argmax` returns the position of the largest absolute value.

The `pivot` offset converts that local position into the corresponding row index in the original matrix.

This implements partial pivoting.

### 4. Swap Rows

```python
A[[pivot, max_row]] = A[[max_row, pivot]]
b[[pivot, max_row]] = b[[max_row, pivot]]
```

The selected pivot row is exchanged with the current row.

The same permutation must be applied to `b`.

Otherwise, the coefficient matrix and right-hand side would represent different systems.

### 5. Compute the Elimination Factor

```python
factor = A[row, pivot] / A[pivot, pivot]
```

The factor determines how much of the pivot row must be subtracted from the current row.

The goal is to make:

\(A\_{row,pivot}=0\)

### 6. Eliminate the Current Column

```python
A[row] -= factor * A[pivot]
```

This performs the row operation

\(R*{row}\leftarrow R*{row}-factorR\_{pivot}\)

As a result, the current pivot-column entry becomes zero.

The operation is vectorized across the entire row.

### 7. Update the Right-Hand Side

```python
b[row] -= factor * b[pivot]
```

The same row operation must be applied to the right-hand side.

This preserves the equivalence of the linear system.

The augmented system effectively undergoes:

\([A|b]\rightarrow[A'|b']\)

### 8. Produce an Upper Triangular Matrix

After all pivot iterations, the entries below the diagonal have been eliminated.

The coefficient matrix approximately has the form:

$$
U=
\begin{bmatrix}
u_{11} & u_{12} & \cdots & u_{1n}\\
0 & u_{22} & \cdots & u_{2n}\\
0 & 0 & \ddots & \vdots\\
0 & 0 & \cdots & u_{nn}
\end{bmatrix}
$$

The corresponding vector is the transformed right-hand side.

### 9. Initialize the Solution

```python
x = np.zeros(n)
```

The solution vector stores the values obtained during backward substitution.

Initially, none of the variables have been solved.

### 10. Iterate Backward

```python
for i in range(n - 1, -1, -1):
```

The final row is solved first.

This is necessary because the final row contains only the final unknown after elimination.

The algorithm then proceeds toward the first row.

### 11. Compute the Known Contribution

```python
np.dot(A[i, i + 1:], x[i + 1:])
```

This calculates:

\(\sum*{j=i+1}^{n-1}A*{ij}x_j\)

All of these $x_j$ values have already been computed because the algorithm moves from right to left.

### 12. Solve the Current Variable

```python
x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]
```

This directly implements:

\(x*i=\frac{b_i-\sum*{j=i+1}^{n-1}A*{ij}x_j}{A*{ii}}\)

The result is stored in `x[i]`.

### 13. Return the Solution

```python
return x
```

After the backward-substitution loop completes, every variable has been determined.

For the provided example, the solution is:

```text
[11.0, -4.0, 3.0]
```

### Algorithm Summary

The complete algorithm can be viewed as:

```text
Input: A, b

1. Convert A and b to floating point.

2. For each pivot column:
   a. Find the largest absolute entry below the pivot.
   b. Swap that row into the pivot position.
   c. Eliminate all entries below the pivot.

3. A is now upper triangular.

4. Starting from the final row:
   a. Compute the contribution of already-known variables.
   b. Subtract it from b.
   c. Divide by the diagonal pivot.

5. Return x.
```

The critical invariant is:

\(Ax=b\quad\Longleftrightarrow\quad Ux=c\)

because every row operation is applied consistently to both sides.

## Time & Space Complexity

Let $n$ be the number of equations and unknowns.

### Partial Pivoting

At each pivot, the algorithm searches up to $n$ rows.

Across all pivots, this contributes:

\(O(n^2)\)

### Forward Elimination

For each pivot, approximately $n$ rows are updated.

Each row update touches approximately $n$ columns.

Therefore:

\(O(n^3)\)

### Backward Substitution

For each of the $n$ variables, up to $n$ previously solved variables may be used.

Therefore:

\(O(n^2)\)

### Overall Time Complexity

The dominant operation is forward elimination.

\(O(n^3)\)

### Space Complexity

The input matrix is modified in place after conversion.

The solution vector requires:

\(O(n)\)

additional space.

The NumPy operations used for row updates can create temporary arrays, but the algorithm's primary storage is the matrix and vectors.

The overall storage including the matrix is:

\(O(n^2)\)

The auxiliary solution storage is:

\(O(n)\)

| Operation                | Time      |
| ------------------------ | --------- |
| Partial Pivoting         | **O(n²)** |
| Forward Elimination      | **O(n³)** |
| Backward Substitution    | **O(n²)** |
| Overall                  | **O(n³)** |
| Auxiliary Solution Space | **O(n)**  |
| Matrix Storage           | **O(n²)** |

For a general $m\times n$ matrix, the elimination cost is more precisely related to the dimensions of the matrix, but for the square systems used here, $O(n^3)$ is the standard complexity.
