# Implement Ridge Regression Loss Function (Easy, Machine Learning)

## Table of Contents

- Problem Statement
- Example
- Learn: Understanding Ridge Regression Loss
- Solution
- Code Explanation
- Time & Space Complexity

---

## Problem Statement

### [Implement Ridge Regression Loss Function](https://www.deep-ml.com/problems/43)

Write a Python function that computes the **Ridge Regression Loss**.

The function should:

- Accept a feature matrix `X`.
- Accept a coefficient (weight) vector `w`.
- Accept the true target values `y_true`.
- Accept a regularization parameter `alpha`.
- Compute the Mean Squared Error (MSE).
- Add the L2 regularization penalty.
- Return the total Ridge loss.

---

## Example

### Input

```python
import numpy as np

X = np.array([
    [1, 1],
    [2, 1],
    [3, 1],
    [4, 1]
])

w = np.array([0.2, 2])

y_true = np.array([2, 3, 4, 5])

alpha = 0.1

loss = ridge_loss(X, w, y_true, alpha)

print(loss)
```

### Output

```python
2.204
```

### Reasoning

First, the model predicts the outputs using

$$
\hat{y}=Xw
$$

The prediction error is measured using the **Mean Squared Error (MSE)**.

Finally, a penalty proportional to the squared magnitude of the coefficients is added to discourage overly large weights.

---

## Learn: Understanding Ridge Regression Loss

### What is it?

**Ridge Regression** is an extension of Linear Regression that adds an **L2 regularization** term to the loss function.

Instead of minimizing only the prediction error, Ridge Regression also penalizes large model coefficients.

This helps reduce **overfitting**, improves generalization, and produces more stable models, especially when features are highly correlated.

Unlike ordinary linear regression, Ridge Regression prefers smaller weights even if doing so slightly increases the training error.

---

### Mathematical Definition

The prediction of a linear regression model is

$$
\hat{y}=Xw
$$

where

- $X$ is the feature matrix.
- $w$ is the weight vector.
- $\hat{y}$ is the predicted output.

The Mean Squared Error is

$$
\mathrm{MSE}=
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
$$

The L2 regularization term is

$$
\lambda
\sum_{j=1}^{p}
w_j^2
$$

The Ridge Regression loss combines both terms.

$$
L(w)=
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
+
\lambda
\sum_{j=1}^{p}
w_j^2
$$

where

- $n$ is the number of samples.
- $p$ is the number of features.
- $\lambda$ (called `alpha` in many libraries) controls the regularization strength.

---

### How Ridge Loss is Computed

The algorithm follows these steps.

1. Compute the predictions.

$$
\hat{y}=Xw
$$

2. Compute the Mean Squared Error.

$$
\mathrm{MSE}=
\frac{1}{n}
\sum
(y-\hat{y})^2
$$

3. Compute the L2 penalty.

$$
\lambda
\sum
w^2
$$

4. Add both quantities.

$$
\mathrm{Loss}=
\mathrm{MSE}
+
\mathrm{L2\ Penalty}
$$

---

### Effect of the Regularization Parameter

The value of

$$
\lambda
$$

controls how strongly large coefficients are penalized.

- Small $\lambda$ behaves similarly to ordinary linear regression.
- Large $\lambda$ forces coefficients toward zero.
- Extremely large values may cause underfitting.

For example,

```text
λ = 0
```

No regularization.

```text
λ = 0.1
```

Weak regularization.

```text
λ = 100
```

Very strong regularization.

---

### Ridge vs Linear Regression

| Linear Regression              | Ridge Regression                     |
| ------------------------------ | ------------------------------------ |
| Minimizes only MSE             | Minimizes MSE + L2 penalty           |
| Can overfit easily             | Reduces overfitting                  |
| Large coefficients allowed     | Penalizes large coefficients         |
| Sensitive to multicollinearity | More stable with correlated features |

---

### Ridge vs Lasso

| Ridge (L2)                     | Lasso (L1)                              |
| ------------------------------ | --------------------------------------- |
| Penalizes squared coefficients | Penalizes absolute coefficients         |
| Shrinks coefficients           | Can shrink coefficients exactly to zero |
| Keeps all features             | Performs feature selection              |
| Smooth optimization            | Produces sparse models                  |

---

### Characteristics / Key Points

- Uses **L2 Regularization**.
- Penalizes large coefficient values.
- Helps reduce overfitting.
- Improves model generalization.
- Works well with correlated features.
- Keeps every feature in the model.
- Produces a convex optimization problem with a unique solution.

---

### Why is it used? / Applications

Ridge Regression is commonly used when a dataset contains many correlated features.

Applications include

- House Price Prediction
- Stock Price Forecasting
- Medical Data Analysis
- Financial Modeling
- Engineering Regression Problems
- Time Series Forecasting
- Scientific Data Modeling
- High-dimensional Machine Learning

It is also frequently used as a baseline regression model before trying more complex algorithms.

---

> 💡 **Important Note**
>
> In practice, the **bias/intercept term is usually not regularized**. Libraries such as **scikit-learn** apply the L2 penalty only to the feature weights, not the intercept. This prevents unnecessary bias in the learned predictions.

---

## Solution

### Custom Implementation

```python
import numpy as np

def ridge_loss(X, w, y_true, alpha):
    y_pred = X @ w

    return (
        np.mean((y_true - y_pred) ** 2)
        + alpha * np.sum(w ** 2)
    )
```

---

## Code Explanation

### Step 1

Compute the predicted values.

```python
y_pred = X @ w
```

Matrix multiplication calculates the prediction for every sample simultaneously.

---

### Step 2

Compute the Mean Squared Error.

```python
np.mean((y_true - y_pred) ** 2)
```

This measures the average squared prediction error.

---

### Step 3

Compute the L2 regularization penalty.

```python
alpha * np.sum(w ** 2)
```

Each coefficient is squared and summed.

Multiplying by `alpha` controls the penalty strength.

---

### Step 4

Return the total Ridge loss.

```python
MSE + L2 Penalty
```

The optimization algorithm attempts to minimize this combined objective.

---

## Time & Space Complexity

| Complexity | Value     |
| ---------- | --------- |
| Time       | **O(np)** |
| Space      | **O(n)**  |

where

- $n$ is the number of samples.
- $p$ is the number of features.

Computing the predictions requires matrix-vector multiplication, while the regularization term depends only on the weight vector.
