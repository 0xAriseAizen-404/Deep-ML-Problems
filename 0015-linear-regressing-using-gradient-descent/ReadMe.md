# Linear Regression Using Gradient Descent (Easy, Machine Learning)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Gradient Descent](#learn-gradient-descent)
- [Solutions](#solutions)
  - [NumPy Implementation](#numpy-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

---

## Problem Statement

[Linear Regression Using Gradient Descent](https://www.deep-ml.com/problems/15)

Write a Python function to perform **Linear Regression** using **Batch Gradient Descent**. Initialize all weights to zero and update them iteratively to minimize the **Mean Squared Error (MSE)**.

---

## Example

```python
Input:

X = np.array([[1, 1],
              [1, 2],
              [1, 3]])

y = np.array([3, 5, 7])

alpha = 0.1
iterations = 1000

Output:

[1.0, 2.0]
```

**Reasoning**

The data follows the equation:

```text
y = 1 + 2x
```

Gradient Descent learns the intercept **1** and slope **2** after enough iterations.

---

## Learn: Gradient Descent

Gradient Descent minimizes the MSE loss by repeatedly updating the weights.

Update Rule:

```text
θ = θ - α × (1/m) × Xᵀ(Xθ - y)
```

Where:

- `θ` → Model weights
- `α` → Learning rate
- `m` → Number of training examples
- `Xθ` → Predictions

**Steps**

1. Initialize weights to zero.
2. Compute predictions.
3. Calculate the gradient.
4. Update the weights.
5. Repeat for the given iterations.

---

## Solutions

### NumPy Implementation

```python
import numpy as np


def linear_regression_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    iterations: int
) -> np.ndarray:

    m, n = X.shape

    y = y.reshape(-1, 1)
    theta = np.zeros((n, 1))

    for _ in range(iterations):
        predictions = X @ theta
        errors = predictions - y
        gradient = (1 / m) * (X.T @ errors)
        theta -= alpha * gradient

    return theta.flatten()


# Example Usage
X = np.array([[1, 1],
              [1, 2],
              [1, 3]])

y = np.array([3, 5, 7])

print(linear_regression_gradient_descent(X, y, 0.1, 1000))
```

---

## Code Explanation

- Initialize all weights to zero.

```python
theta = np.zeros((n, 1))
```

- Compute predictions.

```python
predictions = X @ theta
```

- Compute the gradient.

```python
gradient = (1 / m) * (X.T @ (predictions - y))
```

- Update the weights.

```python
theta -= alpha * gradient
```

Repeat the process until all iterations are completed.

---

## Time & Space Complexity

| Complexity | Value                     |
| ---------- | ------------------------- |
| Time       | **O(iterations × m × n)** |
| Space      | **O(n)**                  |

Where:

- **m** = Number of training examples
- **n** = Number of features (including the bias term)
