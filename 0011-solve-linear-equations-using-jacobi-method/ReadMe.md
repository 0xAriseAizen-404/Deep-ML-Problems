# Solve Linear Equations using Jacobi Method (Medium, Linear Algebra)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Jacobi Method](#learn-jacobi-method)
- [Solutions](#solutions)
  - [Custom Implementation](#custom-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

## Problem Statement

### [Solve Linear Equations using Jacobi Method](https://www.deep-ml.com/problems/11)

Write a Python function that solves a system of linear equations

$$
Ax=b
$$

using the **Jacobi Iterative Method**.

The algorithm should:

- Initialize the solution vector with zeros.
- Perform **exactly** `n` iterations.
- Update every variable using only the values from the previous iteration.
- Carry full floating-point precision throughout the iterations.
- Round the final solution to **4 decimal places** before returning it.

If only a few iterations are performed, the returned solution is merely an approximation and may not exactly satisfy

$$
Ax=b
$$

---

## Example

### Input

```python
A = [
    [5, -2, 3],
    [-3, 9, 1],
    [2, -1, -7]
]

b = [-1, 2, 3]

n = 2
```

### Output

```python
[0.1460, 0.2032, -0.5175]
```

### Reasoning

Starting with

$$
x^{(0)}=
\begin{bmatrix}
0\\
0\\
0
\end{bmatrix}
$$

each iteration computes a new approximation using only the values from the previous iteration.

After two iterations, the approximate solution becomes

$$
x^{(2)}=
\begin{bmatrix}
0.1460\\
0.2032\\
-0.5175
\end{bmatrix}
$$

---

## Learn: Jacobi Method

### What is the Jacobi Method?

The Jacobi Method is an **iterative algorithm** for solving systems of linear equations.

Instead of eliminating variables like Gaussian Elimination, Jacobi repeatedly improves an initial guess until the solution converges.

It is especially useful for:

- Large sparse systems
- Scientific computing
- Numerical simulations
- Parallel computing

because every variable can be updated independently.

---

### Linear System

Given

$$
Ax=b
$$

where

- $A$ is an $n \times n$ coefficient matrix.
- $x$ is the unknown vector.
- $b$ is the constant vector.

The goal is to find

$$
x
$$

such that the equation holds.

---

### Jacobi Update Formula

For the $i^{th}$ equation,

$$
x_i^{(k+1)}=\frac{1}{a_{ii}}\left(b_i-\sum_{j\ne i}a_{ij}x_j^{(k)}\right)
$$

where

- $k$ is the iteration number.
- $a_{ii}$ is the diagonal element.
- $x_j^{(k)}$ comes from the **previous iteration**.

Unlike the Gauss-Seidel method, newly computed values are **not** reused during the same iteration.

---

### Matrix Form

If

$$
A=D+L+U
$$

where

- $D$ is the diagonal matrix.
- $L$ is the lower triangular matrix.
- $U$ is the upper triangular matrix.

then the Jacobi iteration is

$$
x^{(k+1)}=D^{-1}\left(b-(L+U)x^{(k)}\right)
$$

This compact matrix form is commonly used in numerical linear algebra.

---

### Step-by-Step Algorithm

1. Initialize

$$
x^{(0)}=0
$$

2. For each iteration:
   - Compute every variable independently.
   - Use only values from the previous iteration.
   - Store the new values separately.

3. Replace the old solution with the new one.

4. Repeat for the specified number of iterations.

---

### Convergence

Jacobi does **not** always converge.

A sufficient condition for convergence is that the matrix is **strictly diagonally dominant**.

A matrix is diagonally dominant if

$$
|a_{ii}|>\sum_{j\ne i}|a_{ij}|
$$

for every row.

Symmetric positive definite matrices also guarantee convergence.

---

### Characteristics / Key Points

- Iterative numerical method.
- Starts from an initial guess.
- Updates all variables simultaneously.
- Requires storing two solution vectors.
- Easy to parallelize.
- May fail to converge for some matrices.
- Accuracy improves with more iterations when convergence conditions are satisfied.

---

### Jacobi vs Gauss-Seidel

| Jacobi                              | Gauss-Seidel                           |
| ----------------------------------- | -------------------------------------- |
| Uses only previous iteration values | Uses newly computed values immediately |
| Easier to parallelize               | Harder to parallelize                  |
| Requires two solution vectors       | Can update in-place                    |
| Usually converges more slowly       | Usually converges faster               |

---

### Why is it Used?

The Jacobi Method is widely used in:

- Numerical Linear Algebra
- Finite Element Analysis (FEA)
- Computational Fluid Dynamics (CFD)
- Heat diffusion simulations
- Electrical circuit analysis
- Sparse matrix solvers
- Scientific computing
- Parallel and distributed computing

Many large engineering problems contain millions of unknowns where direct methods become computationally expensive.

---

> 💡 **Important Note**
>
> A common implementation mistake is updating the solution vector **in-place**. Doing so changes the algorithm into the **Gauss-Seidel Method**. The Jacobi Method must compute every new value from the previous iteration only. Also, round **only the final answer**—rounding after each iteration introduces numerical error and changes the result.

---

## Solutions

### Custom Implementation

```python
import numpy as np

def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:

    x = np.zeros(len(A))

    for _ in range(n):

        x_new = np.zeros_like(x)

        for i in range(len(A)):
            x_new[i] = (
                b[i] -
                sum(A[i][j] * x[j] for j in range(len(A)) if j != i)
            ) / A[i][i]

        x = x_new

    return np.round(x, 4).tolist()
```

---

## Code Explanation

### 1. Initialize the Solution Vector

```python
x = np.zeros(len(A))
```

The algorithm starts with

$$
x^{(0)}=
\begin{bmatrix}
0\\
0\\
\vdots\\
0
\end{bmatrix}
$$

This is the initial approximation.

---

### 2. Perform the Required Iterations

```python
for _ in range(n):
```

The algorithm performs exactly `n` iterations, regardless of whether convergence has already occurred.

---

### 3. Create a New Solution Vector

```python
x_new = np.zeros_like(x)
```

A separate vector stores the next iteration.

This ensures every update uses only values from

$$
x^{(k)}
$$

instead of partially updated values.

---

### 4. Compute Each Variable

```python
x_new[i] = (
    b[i] -
    sum(A[i][j] * x[j] for j in range(len(A)) if j != i)
) / A[i][i]
```

This directly implements

$$
x_i^{(k+1)}=\frac{1}{a_{ii}}\left(b_i-\sum_{j\ne i}a_{ij}x_j^{(k)}\right)
$$

Each variable is solved independently.

---

### 5. Replace the Old Approximation

```python
x = x_new
```

After every variable has been computed, the new approximation becomes the current solution for the next iteration.

---

### 6. Round Only the Final Answer

```python
return np.round(x, 4).tolist()
```

The problem specifies that rounding should occur **only once**, after all iterations have completed.

This avoids cumulative rounding errors.

---

## Time & Space Complexity

Let

- $n$ = number of variables.
- $k$ = number of Jacobi iterations.

Each iteration computes every variable using all other variables.

| Complexity | Value         |
| ---------- | ------------- |
| Time       | **O(k × n²)** |
| Space      | **O(n)**      |

The algorithm stores two vectors (`x` and `x_new`), each containing `n` elements.
