# Matrix Determinant & Trace (Easy, Linear Algebra)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Matrix Determinant and Trace](#learn-matrix-determinant-and-trace)
  - [What is the Trace?](#what-is-the-trace)
  - [Trace Definition](#trace-definition)
  - [Properties of Trace](#properties-of-trace)
  - [What is the Determinant?](#what-is-the-determinant)
  - [Determinant of a 2×2 Matrix](#determinant-of-a-2x2-matrix)
  - [Cofactor Expansion](#cofactor-expansion)
  - [3×3 Determinant Example](#3x3-determinant-example)
  - [Properties of the Determinant](#properties-of-the-determinant)
  - [Geometric Interpretation](#geometric-interpretation)
  - [Determinant and Invertibility](#determinant-and-invertibility)
  - [Computational Considerations](#computational-considerations)
  - [Relationship Between Determinant and Trace](#relationship-between-determinant-and-trace)
  - [Applications](#applications)

- [Solutions](#solutions)
  - [Custom Recursive Implementation](#custom-recursive-implementation)
  - [NumPy Implementation](#numpy-implementation)

- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

---

## Problem Statement

### [Matrix Determinant & Trace](https://www.deep-ml.com/problems/195)

Implement a function that computes both the determinant and trace of a square matrix.

The determinant is a scalar value that captures important properties of a square matrix, including whether the matrix is invertible and how the corresponding linear transformation scales volume.

The trace is the sum of the elements on the main diagonal.

The function should return:

```text
(determinant, trace)
```

as a tuple.

For an $n\times n$ matrix $A$:

\(\operatorname{tr}(A)=\sum*{i=1}^{n}a*{ii}\)

For a $2\times2$ matrix:

\(\det(A)=ad-bc\)

For larger matrices, the determinant can be computed recursively using cofactor expansion.

---

## Example

### Input

```python
matrix = [[2, 3], [1, 4]]
```

### Output

```text
(5.0, 6.0)
```

### Reasoning

For:

```text
A = [2  3]
    [1  4]
```

the determinant is:

\(\det(A)=(2)(4)-(3)(1)\)

Therefore:

\(\det(A)=8-3=5\)

The trace is the sum of the diagonal elements:

\(\operatorname{tr}(A)=2+4=6\)

Therefore the result is:

```text
(5.0, 6.0)
```

---

# Learn: Matrix Determinant and Trace

## What is the Trace?

The trace is one of the simplest scalar quantities associated with a square matrix.

It is obtained by summing the elements on the main diagonal.

For:

\(A=\begin{bmatrix}a*{11}&a*{12}&\cdots&a*{1n}\\a*{21}&a*{22}&\cdots&a*{2n}\\\vdots&\vdots&\ddots&\vdots\\a*{n1}&a*{n2}&\cdots&a\_{nn}\end{bmatrix}\)

the diagonal elements are:

\(a*{11},a*{22},\ldots,a\_{nn}\)

Therefore:

\(\operatorname{tr}(A)=\sum*{i=1}^{n}a*{ii}\)

The trace is defined only for square matrices.

---

## Trace Definition

For a $3\times3$ matrix:

\(A=\begin{bmatrix}2&3&1\\0&5&4\\7&6&8\end{bmatrix}\)

the diagonal elements are:

\(2,5,8\)

Therefore:

\(\operatorname{tr}(A)=2+5+8=15\)

Only the main diagonal contributes to the trace.

Elements above or below the diagonal are ignored.

---

## Computing the Trace

The algorithm is direct.

For each index:

\(i=0,1,\ldots,n-1\)

add:

\(A\_{ii}\)

to an accumulator.

Pseudocode:

```text
trace(A):
    result = 0

    for i = 0 to n-1:
        result += A[i][i]

    return result
```

There are exactly `n` diagonal elements.

Therefore, trace computation requires:

\(O(n)\)

time.

---

## Properties of Trace

### Linearity

For two square matrices of the same dimensions:

\(\operatorname{tr}(A+B)=\operatorname{tr}(A)+\operatorname{tr}(B)\)

For a scalar $c$:

\(\operatorname{tr}(cA)=c\operatorname{tr}(A)\)

Together:

\(\operatorname{tr}(cA+dB)=c\operatorname{tr}(A)+d\operatorname{tr}(B)\)

---

## Cyclic Property

For compatible matrices:

\(\operatorname{tr}(ABC)=\operatorname{tr}(BCA)=\operatorname{tr}(CAB)\)

This is called the cyclic property of trace.

However, trace generally does not allow arbitrary reordering.

In general:

\(\operatorname{tr}(ABC)\neq\operatorname{tr}(ACB)\)

The order can be cyclically rotated, but not freely permuted.

---

## Trace and Transpose

The trace remains unchanged after transposition:

\(\operatorname{tr}(A^T)=\operatorname{tr}(A)\)

This happens because transposition does not change the diagonal elements.

If:

\(A\_{ii}=x\)

then:

\(A^T\_{ii}=x\)

for every diagonal position.

---

## Trace and Eigenvalues

For an $n\times n$ matrix, the trace equals the sum of its eigenvalues, counting algebraic multiplicity:

\(\operatorname{tr}(A)=\sum\_{i=1}^{n}\lambda_i\)

This provides an important connection between the entries of a matrix and its spectral properties.

For a $2\times2$ matrix with eigenvalues $\lambda_1$ and $\lambda_2$:

\(\operatorname{tr}(A)=\lambda_1+\lambda_2\)

---

# What is the Determinant?

The determinant is a scalar associated with a square matrix.

Unlike the trace, the determinant is not obtained by simply summing selected elements.

It captures several important properties of the matrix.

In particular:

\(\det(A)=0\)

means that the matrix is singular and therefore not invertible.

A nonzero determinant means:

\(\det(A)\neq0\)

and therefore the matrix is invertible.

The determinant also describes the signed volume scaling of the linear transformation represented by the matrix.

---

## Determinant of a 2×2 Matrix

For:

\(A=\begin{bmatrix}a&b\\c&d\end{bmatrix}\)

the determinant is:

\(\det(A)=ad-bc\)

For:

\(A=\begin{bmatrix}2&3\\1&4\end{bmatrix}\)

we obtain:

\(\det(A)=(2)(4)-(3)(1)\)

Therefore:

\(\det(A)=5\)

The determinant requires multiplying the main diagonal terms and subtracting the product of the opposite diagonal terms.

---

# Cofactor Expansion

For matrices larger than $2\times2$, the determinant can be computed recursively using cofactor expansion.

Suppose $A$ is an $n\times n$ matrix.

Expanding along row $i$:

\(\det(A)=\sum*{j=1}^{n}(-1)^{i+j}a*{ij}M\_{ij}\)

where:

- $a_{ij}$ is the element at row $i$, column $j$.
- $M_{ij}$ is the determinant of the minor obtained by removing row $i$ and column $j$.
- $(-1)^{i+j}$ determines the alternating sign.

---

## Minor Matrix

The minor associated with $a_{ij}$ is formed by removing:

1. Row $i$
2. Column $j$

from the original matrix.

For example, consider:

\(A=\begin{bmatrix}1&2&3\\4&5&6\\7&8&9\end{bmatrix}\)

For the element $a_{11}=1$, remove the first row and first column.

The resulting minor is:

\(M\_{11}=\begin{bmatrix}5&6\\8&9\end{bmatrix}\)

Its determinant is:

\(\det(M\_{11})=(5)(9)-(6)(8)\)

Therefore:

\(\det(M\_{11})=-3\)

---

## Cofactor Sign Pattern

The factor:

\((-1)^{i+j}\)

produces the alternating sign pattern:

\(\begin{bmatrix}+&-&+\\-&+&-\\+&-&+\end{bmatrix}\)

For the first row:

\(+,-,+\)

Therefore, expansion along the first row has the form:

\(\det(A)=a*{11}M*{11}-a*{12}M*{12}+a*{13}M*{13}\)

This sign pattern is essential for a correct recursive implementation.

---

# 3×3 Determinant Example

Consider:

\(A=\begin{bmatrix}1&2&3\\0&1&4\\5&6&0\end{bmatrix}\)

Expand along the first row:

\(\det(A)=1M*{11}-2M*{12}+3M\_{13}\)

The first minor is:

\(M\_{11}=\begin{vmatrix}1&4\\6&0\end{vmatrix}\)

Therefore:

\(M\_{11}=(1)(0)-(4)(6)=-24\)

The second minor is:

\(M\_{12}=\begin{vmatrix}0&4\\5&0\end{vmatrix}\)

Therefore:

\(M\_{12}=(0)(0)-(4)(5)=-20\)

The third minor is:

\(M\_{13}=\begin{vmatrix}0&1\\5&6\end{vmatrix}\)

Therefore:

\(M\_{13}=(0)(6)-(1)(5)=-5\)

Substituting:

\(\det(A)=1(-24)-2(-20)+3(-5)\)

Therefore:

\(\det(A)=-24+40-15\)

and:

\(\det(A)=1\)

---

# Properties of the Determinant

## Invertibility

A square matrix is invertible exactly when its determinant is nonzero.

\(A\text{ is invertible}\iff\det(A)\neq0\)

If:

\(\det(A)=0\)

then the matrix is singular.

---

## Multiplicativity

For square matrices of compatible dimensions:

\(\det(AB)=\det(A)\det(B)\)

This is one of the most important determinant properties.

It means the determinant of a product can be computed from the determinants of the individual matrices.

---

## Transpose

Transposing a matrix does not change its determinant:

\(\det(A^T)=\det(A)\)

Therefore, a matrix and its transpose have the same determinant.

---

## Inverse

If $A$ is invertible:

\(\det(A^{-1})=\frac{1}{\det(A)}\)

This follows from:

\(AA^{-1}=I\)

and:

\(\det(AA^{-1})=\det(I)=1\)

Therefore:

\(\det(A)\det(A^{-1})=1\)

---

## Row Swapping

Swapping two rows changes the sign of the determinant.

If $B$ is obtained from $A$ by exchanging two rows:

\(\det(B)=-\det(A)\)

This is important when computing determinants using elimination.

---

## Scaling a Row

Multiplying a row by a scalar $c$ multiplies the determinant by $c$.

If one row is multiplied by $c$:

\(\det(B)=c\det(A)\)

For example, multiplying one row by `2` doubles the determinant.

---

## Adding a Multiple of Another Row

Adding a multiple of one row to another does not change the determinant.

If:

\(R_i\leftarrow R_i+cR_j\)

then:

\(\det(B)=\det(A)\)

This property is heavily used when computing determinants through Gaussian elimination.

---

## Determinant and Eigenvalues

The determinant equals the product of the eigenvalues, counting algebraic multiplicity:

\(\det(A)=\prod\_{i=1}^{n}\lambda_i\)

For a $2\times2$ matrix:

\(\det(A)=\lambda_1\lambda_2\)

Combined with the trace:

\(\operatorname{tr}(A)=\lambda_1+\lambda_2\)

these provide useful information about the eigenvalue structure.

---

# Geometric Interpretation

## Determinant as Area and Volume Scaling

A matrix represents a linear transformation.

The absolute value of the determinant describes how that transformation changes volume.

For a set $S$:

\(\operatorname{Volume}(T(S))=|\det(A)|\operatorname{Volume}(S)\)

For a two-dimensional transformation, this becomes area scaling.

\(\operatorname{Area}(T(S))=|\det(A)|\operatorname{Area}(S)\)

For a three-dimensional transformation, it describes volume scaling.

---

## Determinant Magnitude

If:

\(|\det(A)|=2\)

the transformation doubles volume.

If:

\(|\det(A)|=0.5\)

the transformation halves volume.

If:

\(|\det(A)|=1\)

the transformation preserves volume magnitude.

If:

\(\det(A)=0\)

the transformation collapses the space into a lower-dimensional space.

---

## Sign of the Determinant

The sign carries orientation information.

If:

\(\det(A)>0\)

the transformation preserves orientation.

If:

\(\det(A)<0\)

the transformation reverses orientation.

If:

\(\det(A)=0\)

the transformation collapses at least one dimension.

For example, a reflection has a negative determinant because it reverses orientation.

---

# Determinant and Invertibility

The determinant provides a direct test for whether a square matrix has an inverse.

Consider:

\(A=\begin{bmatrix}2&4\\1&2\end{bmatrix}\)

Its determinant is:

\(\det(A)=(2)(2)-(4)(1)=0\)

Therefore:

\(\det(A)=0\)

and the matrix is singular.

The rows are linearly dependent:

\(R_1=2R_2\)

Consequently, the transformation collapses a dimension and cannot be reversed uniquely.

---

## Nonzero Determinant

Consider:

\(A=\begin{bmatrix}2&3\\1&4\end{bmatrix}\)

Its determinant is:

\(\det(A)=8-3=5\)

Since:

\(5\neq0\)

the matrix is invertible.

This means the associated linear transformation does not collapse the entire space into a lower-dimensional subspace.

---

# Computational Considerations

## Cofactor Expansion Complexity

The recursive cofactor implementation is useful for understanding how determinants are defined.

However, it is computationally expensive.

Each expansion creates multiple smaller determinant problems.

The naive recursive approach has factorial-scale time complexity:

\(O(n!)\)

for an $n\times n$ matrix.

Therefore, it becomes impractical quickly as `n` increases.

---

## Why Recursion Becomes Expensive

For an $n\times n$ matrix, expansion along one row creates `n` determinant calculations of size:

\(n-1\)

Each of those creates approximately:

\(n-1\)

more calculations.

This branching continues until reaching $2\times2$ matrices.

The number of recursive subproblems grows extremely quickly.

For this reason, cofactor expansion is primarily useful for:

- Learning the determinant definition.
- Small matrices.
- Symbolic calculations.
- Demonstrating the recursive structure of determinants.

---

# Efficient Determinant Computation

For practical numerical computation, elimination-based methods are much more efficient.

A common approach is LU decomposition:

\(A=LU\)

where:

- $L$ is lower triangular.
- $U$ is upper triangular.

For triangular matrices, the determinant equals the product of diagonal elements.

Therefore:

\(\det(L)=\prod*{i=1}^{n}L*{ii}\)

and:

\(\det(U)=\prod*{i=1}^{n}U*{ii}\)

Since:

\(\det(A)=\det(L)\det(U)\)

we obtain:

\(\det(A)=\left(\prod*{i=1}^{n}L*{ii}\right)\left(\prod*{i=1}^{n}U*{ii}\right)\)

For standard LU decomposition with a unit diagonal in $L$:

\(\det(L)=1\)

so:

\(\det(A)=\prod*{i=1}^{n}U*{ii}\)

Accounting for row permutations gives the appropriate sign adjustment.

---

## Complexity of LU-Based Determinants

LU decomposition requires approximately:

\(O(n^3)\)

time for an $n\times n$ dense matrix.

This is dramatically better than the factorial-scale recursive cofactor approach.

Practical numerical libraries therefore use elimination or factorization methods rather than naive cofactor expansion.

---

# Computing the Trace Efficiently

Trace is much simpler than determinant.

Only the diagonal needs to be accessed.

For an $n\times n$ matrix:

```text
for i in range(n):
    result += A[i][i]
```

There are exactly `n` operations.

Therefore:

\(T(n)=O(n)\)

No recursive computation is necessary.

---

# Relationship Between Determinant and Trace

The trace and determinant provide different information about a matrix.

For a $2\times2$ matrix with eigenvalues:

\(\lambda_1,\lambda_2\)

the trace is:

\(\operatorname{tr}(A)=\lambda_1+\lambda_2\)

while the determinant is:

\(\det(A)=\lambda_1\lambda_2\)

Therefore, the characteristic polynomial can be written as:

\(\lambda^2-\operatorname{tr}(A)\lambda+\det(A)=0\)

This means that for a $2\times2$ matrix, trace and determinant together determine the coefficients of the characteristic polynomial.

---

## Example

Suppose:

\(\operatorname{tr}(A)=6\)

and:

\(\det(A)=5\)

Then the characteristic equation is:

\(\lambda^2-6\lambda+5=0\)

Factoring:

\(\lambda^2-6\lambda+5=(\lambda-1)(\lambda-5)\)

Therefore:

\(\lambda_1=1\)

and:

\(\lambda_2=5\)

Notice:

\(1+5=6\)

and:

\(1\cdot5=5\)

which agrees with the trace and determinant.

---

# Applications

## Computer Graphics

Determinants are used to determine orientation.

For example, the sign of a determinant can indicate whether a transformation preserves or reverses orientation.

This is useful in geometric computations and operations such as backface culling.

---

## Machine Learning

Determinants appear in multivariate probability distributions.

For example, the multivariate Gaussian density contains the determinant of its covariance matrix:

\(p(x)=\frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}}\exp\left(-\frac12(x-\mu)^T\Sigma^{-1}(x-\mu)\right)\)

Here:

\(|\Sigma|=\det(\Sigma)\)

The determinant therefore contributes to the normalization of the probability density.

---

## Differential Equations

For linear dynamical systems:

\(x'=Ax\)

the eigenvalues of $A$ determine important aspects of system behavior.

Trace and determinant provide compact information about the eigenvalues, especially for low-dimensional systems.

---

## Optimization

The Hessian matrix contains second-order derivative information.

For a two-variable function, the determinant of the Hessian can help distinguish between different types of critical points when combined with the signs of the Hessian entries.

For example, for a Hessian:

\(H=\begin{bmatrix}f*{xx}&f*{xy}\\f*{yx}&f*{yy}\end{bmatrix}\)

the determinant is:

\(\det(H)=f*{xx}f*{yy}-f*{xy}f*{yx}\)

Its value contributes to the second-derivative test.

---

> 💡 **Important Note**
>
> The recursive implementation in this problem is excellent for understanding the mathematical definition of a determinant, but it is not the algorithm you would normally use for large numerical matrices.
>
> In practical numerical computing, determinant calculations are typically based on matrix factorization or elimination methods with approximately $O(n^3)$ complexity.

---

# Solutions

## Custom Recursive Implementation

```python
def matrix_determinant_and_trace(
    matrix: list[list[float]]
) -> tuple[float, float]:
    """
    Compute the determinant and trace of a square matrix.
    """
    def det_help(mat: list[list[float]]) -> float:
        n = len(mat)

        if n == 1:
            return mat[0][0]

        if n == 2:
            return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]

        det = 0

        for j in range(n):
            minor = [row[:j] + row[j + 1:] for row in mat[1:]]
            det += (-1) ** j * mat[0][j] * det_help(minor)

        return det

    def trace(mat: list[list[float]]) -> float:
        n = len(mat)
        trace_mat = 0.0

        for ind in range(n):
            trace_mat += mat[ind][ind]

        return trace_mat

    return (det_help(matrix), trace(matrix))
```

---

## NumPy Implementation

For practical numerical computation, NumPy already provides optimized routines for both operations.

```python
import numpy as np

def matrix_determinant_and_trace(
    matrix: list[list[float]]
) -> tuple[float, float]:
    matrix = np.asarray(matrix, dtype=float)

    determinant = np.linalg.det(matrix)
    trace = np.trace(matrix)

    return float(determinant), float(trace)
```

The NumPy implementation delegates determinant computation to optimized numerical linear-algebra routines rather than explicitly performing recursive cofactor expansion.

---

# Code Explanation

## 1. Recursive Determinant Helper

The function defines an internal helper:

```python
def det_help(mat):
```

This isolates determinant computation from trace computation.

The helper recursively reduces the matrix until reaching the base cases.

---

## 2. 1×1 Base Case

For:

```text
[a]
```

the determinant is simply:

\(\det(A)=a\)

The implementation returns:

```python
if n == 1:
    return mat[0][0]
```

This is the smallest determinant problem.

---

## 3. 2×2 Base Case

For:

\(A=\begin{bmatrix}a&b\\c&d\end{bmatrix}\)

the determinant is:

\(\det(A)=ad-bc\)

The implementation directly applies this formula:

```python
if n == 2:
    return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]
```

This avoids unnecessary recursive expansion.

---

## 4. Expand Along the First Row

For larger matrices, the implementation uses the first row.

```python
for j in range(n):
```

Each column corresponds to one cofactor term.

The expansion is:

\(\det(A)=\sum*{j=1}^{n}(-1)^{1+j}a*{1j}M\_{1j}\)

Because Python uses zero-based indexing, the implementation uses:

```python
(-1) ** j
```

instead of:

```python
(-1) ** (1 + j)
```

The sign pattern is still correct because the index starts at zero.

---

## 5. Construct the Minor

The minor removes the first row and the current column:

```python
minor = [row[:j] + row[j + 1:] for row in mat[1:]]
```

The expression:

```python
mat[1:]
```

removes the first row.

Then:

```python
row[:j]
```

keeps elements before column `j`.

And:

```python
row[j + 1:]
```

keeps elements after column `j`.

Concatenating them removes the selected column.

Therefore, each generated matrix has dimensions:

\((n-1)\times(n-1)\)

---

## 6. Recursive Expansion

The determinant contribution from each column is:

```python
(-1) ** j * mat[0][j] * det_help(minor)
```

This corresponds to:

\((-1)^ja*{0j}\det(M*{0j})\)

The recursive call computes the determinant of the smaller minor.

The contributions are accumulated into:

```python
det = 0
```

and finally returned.

---

## 7. Computing the Trace

The trace helper is much simpler:

```python
def trace(mat):
```

It initializes:

```python
trace_mat = 0.0
```

Then iterates through diagonal positions:

```python
for ind in range(n):
    trace_mat += mat[ind][ind]
```

The expression:

```python
mat[ind][ind]
```

selects:

\(a\_{ii}\)

Therefore the loop computes:

\(\sum*{i=1}^{n}a*{ii}\)

---

## 8. Returning Both Results

The final line:

```python
return (det_help(matrix), trace(matrix))
```

returns both scalar values as a tuple.

For:

```python
matrix = [[2, 3], [1, 4]]
```

the result is:

```text
(5, 6.0)
```

Conceptually:

```text
(
    determinant,
    trace
)
```

---

## 9. Why the Determinant Uses Recursion

The recursive structure directly follows the mathematical definition.

For an $n\times n$ matrix:

\(\det(A)=\sum*j(-1)^ja*{1j}\det(M\_{1j})\)

Each minor has one fewer row and one fewer column.

Therefore:

\(n\times n\rightarrow(n-1)\times(n-1)\rightarrow(n-2)\times(n-2)\rightarrow\cdots\)

until reaching a $2\times2$ or $1\times1$ matrix.

This makes recursion a natural implementation of cofactor expansion.

---

## 10. Why Trace Does Not Need Recursion

Trace has no recursive structure.

Its definition is simply:

\(\operatorname{tr}(A)=\sum*{i=1}^{n}a*{ii}\)

Therefore, directly iterating over the diagonal is both simpler and more efficient.

---

# Time & Space Complexity

Let `n` be the dimension of the square matrix.

## Trace

The trace visits exactly `n` diagonal elements.

Therefore:

\(T\_{\text{trace}}(n)=O(n)\)

The accumulator uses constant auxiliary space:

\(S\_{\text{trace}}(n)=O(1)\)

excluding the input matrix.

---

## Recursive Determinant

The naive cofactor expansion recursively creates many minor matrices.

Its factorial-scale time complexity is:

\(T\_{\text{det}}(n)=O(n!)\)

This makes it unsuitable for large matrices.

The recursion depth is:

\(O(n)\)

and constructing minors introduces additional memory overhead.

A useful high-level characterization is:

\(S\_{\text{det}}(n)=O(n^2)\)

for the active recursion and temporary minor construction, although the exact allocation behavior depends on implementation details.

---

## Overall Complexity

Because determinant computation dominates:

\(T(n)=O(n!)\)

for the custom recursive implementation.

The trace computation contributes only:

\(O(n)\)

which is negligible compared with the determinant calculation.

| Operation             | Time      | Auxiliary Space |
| --------------------- | --------- | --------------- |
| Trace                 | **O(n)**  | **O(1)**        |
| Recursive determinant | **O(n!)** | **O(n²)**       |
| Combined function     | **O(n!)** | **O(n²)**       |

For practical numerical determinant computation, LU/elimination-based methods reduce the determinant calculation to approximately:

\(O(n^3)\)

time.

Thus, the recursive solution is best viewed as a direct implementation of the mathematical definition rather than a production-scale numerical algorithm.
