# Solve Linear Equations Using the Jacobi Method (Medium, Linear Algebra)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Jacobi Method](#learn-jacobi-method)
- [Solutions](#solutions)
  - [Custom Implementation](#custom-implementation)
  - [NumPy Implementation](#numpy-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

---

## Problem Statement

[Solve Linear Equations Using the Jacobi Method](https://www.deep-ml.com/problems/11)

Write a Python function that uses the **Jacobi Method** to solve a system of linear equations **Ax = b**.

- Initialize the solution vector **x** with all zeros.
- Perform **n** iterations.
- Round the solution after each iteration to **4 decimal places**.
- Return the approximate solution.

---

## Example

```python
Input:

A = [[5, -2, 3],
     [-3, 9, 1],
     [2, -1, -7]]

b = [-1, 2, 3]

n = 2

Output:

[0.146, 0.2032, -0.5175]
```

**Reasoning**

Each iteration updates every variable using the values from the **previous iteration**.

---

## Learn: Jacobi Method

The Jacobi Method is an **iterative algorithm** for solving linear systems.

For each variable:

```text
x[i] = (b[i] - Σ(A[i][j] × x[j])) / A[i][i]
         (j ≠ i)
```

Steps:

1. Initialize `x` with zeros.
2. Compute a new solution vector using the previous values.
3. Repeat for `n` iterations.

> **Note:** The method converges reliably when the matrix is **diagonally dominant**.

---

## Solutions

### Custom Implementation

```python
def solve_jacobi(A: list[list[float]], b: list[float], n: int) -> list[float]:

    x = [0.0] * len(A)

    for _ in range(n):
        x_new = [0.0] * len(A)

        for i in range(len(A)):
            total = sum(
                A[i][j] * x[j]
                for j in range(len(A))
                if i != j
            )

            x_new[i] = (b[i] - total) / A[i][i]

        x = [round(value, 4) for value in x_new]

    return x


# Example Usage
A = [[5, -2, 3],
     [-3, 9, 1],
     [2, -1, -7]]

b = [-1, 2, 3]

print(solve_jacobi(A, b, 2))
```

---

### NumPy Implementation

```python
import numpy as np


def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:

    x = np.zeros(len(A))

    for _ in range(n):
        x_new = np.zeros_like(x)

        for i in range(len(A)):
            x_new[i] = (
                b[i] - sum(
                    A[i][j] * x[j]
                    for j in range(len(A))
                    if i != j
                )
            ) / A[i][i]

        x = np.round(x_new, 4)

    return x.tolist()


# Example Usage
print(solve_jacobi(np.array(A), np.array(b), 2))
```

---

## Code Explanation

- Initialize the solution vector with zeros.

```python
x = [0.0] * len(A)
```

- For each iteration, create a new solution vector.

- Compute each variable using the values from the **previous iteration**.

```python
x_new[i] = (b[i] - total) / A[i][i]
```

- Round the updated values to **4 decimal places**.

```python
x = [round(value, 4) for value in x_new]
```

---

## Time & Space Complexity

### Custom Implementation

| Complexity | Value |
|------------|-------|
| Time | **O(n × m²)** |
| Space | **O(m)** |

Where:

- **n** = number of iterations
- **m** = number of variables

### NumPy Implementation

- **Time:** **O(n × m²)**
- **Space:** **O(m)**