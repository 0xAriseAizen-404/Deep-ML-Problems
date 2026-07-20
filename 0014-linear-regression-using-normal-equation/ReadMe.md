# Linear Regression Using Normal Equation (Easy) ✔

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Normal Equation](#learn-normal-equation)
- [Solutions](#solutions)
  - [NumPy Implementation](#numpy-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

---

## Problem Statement

[Linear Regression Using Normal Equation](https://www.deep-ml.com/problems/14)

Write a Python function that performs **Linear Regression** using the **Normal Equation**. Given the feature matrix `X` and target vector `y`, return the regression coefficients rounded to **4 decimal places**.

---

## Example

```python
Input:

X = [[1, 1],
     [1, 2],
     [1, 3]]

y = [1, 2, 3]

Output:

[0.0, 1.0]
```

**Reasoning**

The fitted model is:

```text
y = 0.0 + 1.0x
```

which perfectly fits the given data.

---

## Learn: Normal Equation

The Normal Equation computes the optimal parameters directly without iterative optimization.

Formula:

```text
θ = (XᵀX)⁻¹ Xᵀy
```

Where:

- `X` → Feature matrix
- `Xᵀ` → Transpose of `X`
- `θ` → Regression coefficients

**Advantages**

- No learning rate required.
- No iterations.
- Gives the exact solution.

**Limitation**

- Computing `(XᵀX)⁻¹` is expensive for datasets with many features.

---

## Solutions

### NumPy Implementation

```python
import numpy as np


def linear_regression_normal_equation(
    X: list[list[float]],
    y: list[float]
) -> list[float]:

    X = np.asarray(X)
    y = np.asarray(y).reshape(-1, 1)

    theta = np.linalg.inv(X.T @ X) @ X.T @ y

    return np.round(theta, 4).flatten().tolist()


# Example Usage
X = [[1, 1],
     [1, 2],
     [1, 3]]

y = [1, 2, 3]

print(linear_regression_normal_equation(X, y))
```

---

## Code Explanation

- Convert the input lists into NumPy arrays.

```python
X = np.asarray(X)
y = np.asarray(y).reshape(-1, 1)
```

- Compute the coefficients using the Normal Equation.

```python
theta = np.linalg.inv(X.T @ X) @ X.T @ y
```

- Round the coefficients and return them as a list.

```python
return np.round(theta, 4).flatten().tolist()
```

---

## Time & Space Complexity

| Complexity | Value |
|------------|-------|
| Time | **O(n³)** |
| Space | **O(n²)** |

Where **n** is the number of features.