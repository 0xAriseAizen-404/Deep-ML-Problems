# Calculate R-squared for Regression Analysis (Easy, Machine Learning)

## Table of Contents

- Problem Statement
- Example
- Learn: R-squared (Coefficient of Determination)
- Solution
- Code Explanation
- Time & Space Complexity

---

## Problem Statement

[Calculate R-squared for Regression Analysis](https://www.deep-ml.com/problems/69)

Write a Python function `r_squared(y_true, y_pred)` that computes the **R-squared (R²)** value for a regression model.

R-squared, also known as the **Coefficient of Determination**, measures how well the predicted values explain the variability of the true target values.

The function should return the R² value.

---

## Example

```python
import numpy as np

y_true = np.array([1, 2, 3, 4, 5])
y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])

print(r_squared(y_true, y_pred))
```

### Output

```text
0.989
```

### Explanation

First compute the **Sum of Squared Residuals (SSR)**

\[
SSR
=
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
\]

which measures the prediction error.

Next compute the **Total Sum of Squares (SST)**

\[
SST
=
\sum_{i=1}^{n}
(y_i-\bar{y})^2
\]

which measures the total variation in the target values.

Finally,

\[
R^2
=
1-
\frac{SSR}{SST}
\]

Since the prediction errors are very small, the resulting value is

```text
R² = 0.989
```

meaning the model explains **98.9%** of the variance in the data.

---

# Learn: R-squared (Coefficient of Determination)

## What is it?

**R-squared (R²)** is one of the most commonly used evaluation metrics for **regression models**.

It measures **how much of the variation in the target variable is explained by the model**.

Instead of looking directly at prediction errors like **MAE** or **RMSE**, R² compares your model against a very simple baseline:

> **What if we always predicted the mean of the target values?**

If your regression model performs much better than predicting the mean, the R² score will be close to **1**.

---

## Mathematical Definition / Formula

### Sum of Squared Residuals (SSR)

Also called the **Residual Sum of Squares (RSS)**,

\[
SSR
=
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
\]

where

- \(y_i\) = Actual value
- \(\hat{y}_i\) = Predicted value

SSR measures the prediction error.

---

### Total Sum of Squares (SST)

\[
SST
=
\sum_{i=1}^{n}
(y_i-\bar{y})^2
\]

where

\[
\bar{y}
=
\frac{1}{n}
\sum_{i=1}^{n}
y_i
\]

is the mean of the target values.

SST measures the total variation present in the data.

---

### R-squared Formula

\[
R^2
=
1-
\frac{SSR}{SST}
\]

The closer **SSR** is to zero, the closer **R²** is to one.

---

## Geometric Interpretation

Suppose the true values are

```text
1   2   3   4   5
```

The simplest possible prediction is their mean.

```text
Mean Prediction

3   3   3   3   3
```

A regression model should outperform this constant prediction.

R² measures **how much better your model is than always predicting the mean**.

If the model removes most of the prediction error left by the mean, R² approaches **1**.

---

## Characteristics / Key Points

- Used only for **Regression**.
- Measures the goodness of fit.
- Compares the model against predicting the mean.
- Dimensionless metric.
- Easy to interpret.

### Range of R²

| R² Value | Meaning |
| --------- | ------- |
| **1** | Perfect prediction |
| **0** | Model performs no better than predicting the mean |
| **< 0** | Model performs worse than predicting the mean |

Unlike many metrics, **R² can be negative** if the model performs very poorly.

---

## Relationship Between SSR and SST

The prediction quality depends on the relationship between

- **SSR** (prediction error)
- **SST** (total variation)

### Perfect Model

\[
SSR=0
\]

\[
R^2=1
\]

---

### Mean Predictor

\[
SSR=SST
\]

\[
R^2=0
\]

---

### Poor Model

If

\[
SSR>SST
\]

then

\[
R^2<0
\]

meaning the regression model performs worse than simply predicting the average value.

---

## Why is it used? / Applications

R² is widely used to evaluate regression models such as

- Linear Regression
- Ridge Regression
- Lasso Regression
- Polynomial Regression
- Decision Tree Regression
- Random Forest Regression
- Gradient Boosting Regression
- XGBoost Regression

It is commonly reported in

- Machine Learning
- Statistics
- Economics
- Finance
- Forecasting
- Scientific Research

However, R² should **not** be used as the only evaluation metric. Metrics such as **RMSE**, **MAE**, and **Adjusted R²** often provide additional insight into model performance.

> 💡 **Important Note**
>
> A **high R² does not necessarily mean the model is good**. A highly overfitted model can achieve an excellent R² on the training data while performing poorly on unseen data. Always evaluate regression models on a separate validation or test set and consider metrics like **RMSE** or **MAE** alongside R².

---

# Solution

## Custom Implementation

```python
import numpy as np

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)

    ss_tot = np.sum(
        (y_true - np.mean(y_true)) ** 2
    )

    return 1 - (ss_res / ss_tot)
```

---

# Code Explanation

### Step 1: Compute the Residual Sum of Squares (SSR)

```python
ss_res = np.sum((y_true - y_pred) ** 2)
```

This computes

\[
SSR
=
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
\]

which represents the total squared prediction error.

---

### Step 2: Compute the Total Sum of Squares (SST)

```python
ss_tot = np.sum(
    (y_true - np.mean(y_true)) ** 2
)
```

This computes

\[
SST
=
\sum_{i=1}^{n}
(y_i-\bar{y})^2
\]

which measures the total variance in the target values.

---

### Step 3: Compute R²

```python
1 - (ss_res / ss_tot)
```

This applies the standard formula

\[
R^2
=
1-
\frac{SSR}{SST}
\]

A smaller prediction error (SSR) produces a larger R² value.

---

## Time & Space Complexity

Let

- \(n\) = Number of samples.

| Complexity | Value |
| ---------- | ----- |
| Time | **O(n)** |
| Space | **O(1)** |

The algorithm scans the arrays a constant number of times, giving **O(n)** time complexity. Aside from a few scalar variables (`ss_res`, `ss_tot`, and the mean), it does not allocate additional memory proportional to the input size, resulting in **O(1)** auxiliary space.