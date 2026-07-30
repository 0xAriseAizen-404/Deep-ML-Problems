# Implement Ridge Regression Loss Function (Easy, Machine Learning)

## Table of Contents

- Problem Statement
- Example
- Learn: Ridge Regression Loss
- Solution
- Code Explanation
- Time & Space Complexity

---

## Problem Statement

[Implement Ridge Regression Loss Function](https://www.deep-ml.com/problems/43)

Write a Python function `ridge_loss(X, w, y_true, alpha)` that computes the **Ridge Regression loss**.

The function takes:

- `X` — Feature matrix of shape `(n_samples, n_features)`.
- `w` — Weight (coefficient) vector.
- `y_true` — Actual target values.
- `alpha` — Regularization parameter.

The Ridge loss combines the **Mean Squared Error (MSE)** with an **L2 regularization** term that penalizes large model coefficients, helping reduce overfitting.

---

## Example

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

```text
2.204
```

### Explanation

First compute the predicted values:

\[
\hat{y}
=
Xw
\]

\[
=
\begin{bmatrix}
1 & 1\\
2 & 1\\
3 & 1\\
4 & 1
\end{bmatrix}
\begin{bmatrix}
0.2\\
2
\end{bmatrix}
=
\begin{bmatrix}
2.2\\
2.4\\
2.6\\
2.8
\end{bmatrix}
\]

Compute the Mean Squared Error:

\[
\text{MSE}
=
\frac{1}{4}
\left[
(2-2.2)^2+
(3-2.4)^2+
(4-2.6)^2+
(5-2.8)^2
\right]
=
1.8
\]

Compute the L2 penalty:

\[
\alpha
\sum_{j=1}^{p}
w_j^2
=
0.1
(0.2^2+2^2)
=
0.404
\]

Therefore,

\[
\text{Ridge Loss}
=
1.8+0.404
=
2.204
\]

---

# Learn: Ridge Regression Loss

## What is it?

**Ridge Regression** is a regularized version of **Linear Regression** that reduces overfitting by discouraging the model from learning excessively large coefficients.

Instead of minimizing only the prediction error, Ridge Regression also penalizes the magnitude of the model weights.

This penalty is known as **L2 Regularization**.

The optimization objective becomes:

- Fit the training data well.
- Keep the model coefficients as small as possible.

This improves the model's ability to generalize to unseen data.

---

## Mathematical Definition / Formula

The prediction of a linear regression model is

\[
\hat{y}
=
Xw
\]

where

- \(X\) is the feature matrix.
- \(w\) is the coefficient vector.

---

### Mean Squared Error (MSE)

The prediction loss is

\[
\text{MSE}
=
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
\]

---

### L2 Regularization

The penalty term is

\[
\lambda
\sum_{j=1}^{p}
w_j^2
\]

where

- \(\lambda\) (or `alpha`) controls the regularization strength.
- \(p\) is the number of features.

---

### Ridge Loss

The complete Ridge objective is

\[
L(w)
=
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
+
\lambda
\sum_{j=1}^{p}
w_j^2
\]

---

## Characteristics / Key Points

- Uses **L2 Regularization**.
- Penalizes large model coefficients.
- Helps reduce overfitting.
- Keeps all features in the model by shrinking coefficients instead of removing them.
- Produces a smoother and more stable model.
- Differentiable everywhere, making optimization straightforward using Gradient Descent.

---

### Effect of the Regularization Parameter

| Alpha (λ) | Effect |
| ---------- | ------ |
| λ = 0 | Ordinary Linear Regression |
| Small λ | Slight coefficient shrinkage |
| Large λ | Strong regularization and smaller coefficients |
| Very Large λ | Model may underfit |

---

### Ridge vs Linear Regression

| Linear Regression | Ridge Regression |
| ----------------- | ---------------- |
| Minimizes only MSE | Minimizes MSE + L2 penalty |
| Can overfit | Reduces overfitting |
| Large coefficients possible | Coefficients are shrunk |
| Sensitive to multicollinearity | Handles multicollinearity better |

---

## Why is it used? / Applications

Ridge Regression is commonly used when:

- The dataset contains many correlated features.
- Overfitting is observed.
- The number of features is large.
- Stable coefficient estimates are required.
- Building regression models for finance, healthcare, economics, and forecasting.

It is one of the standard regularization techniques available in **Scikit-Learn**.

> 💡 **Important Note**
>
> Unlike **Lasso Regression (L1 Regularization)**, Ridge Regression **does not make coefficients exactly zero**. It only shrinks them toward zero. Therefore, Ridge is useful when you believe **most features contain useful information**, while Lasso is preferred when automatic feature selection is desired.

---

# Solution

## Custom Implementation

```python
import numpy as np

def ridge_loss(
    X: np.ndarray,
    w: np.ndarray,
    y_true: np.ndarray,
    alpha: float
) -> float:

    y_pred = X @ w

    return (
        np.mean((y_true - y_pred) ** 2)
        + alpha * np.sum(w ** 2)
    )
```

---

# Code Explanation

### Step 1: Compute Predictions

```python
y_pred = X @ w
```

Matrix multiplication computes the predicted values using the linear regression equation

\[
\hat{y}=Xw
\]

---

### Step 2: Compute the Mean Squared Error

```python
np.mean((y_true - y_pred) ** 2)
```

This calculates

\[
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
\]

which measures how far the predictions are from the true values.

---

### Step 3: Compute the L2 Regularization Term

```python
alpha * np.sum(w ** 2)
```

This computes

\[
\lambda
\sum_{j=1}^{p}
w_j^2
\]

Large coefficients contribute a larger penalty.

---

### Step 4: Compute the Ridge Loss

Finally,

```python
MSE + Regularization
```

returns

\[
\text{Ridge Loss}
=
\text{MSE}
+
\lambda
\sum_{j=1}^{p}
w_j^2
\]

This value is minimized during model training to obtain the optimal regression coefficients.

---

## Time & Space Complexity

Let

- \(n\) = Number of samples.
- \(p\) = Number of features.

| Complexity | Value |
| ---------- | ----- |
| Time | **O(np)** |
| Space | **O(n)** |

- Computing `X @ w` requires **O(np)** time.
- Computing the MSE requires **O(n)** time.
- Computing the L2 penalty requires **O(p)** time.
- The prediction vector `y_pred` stores **n** values, requiring **O(n)** additional space.