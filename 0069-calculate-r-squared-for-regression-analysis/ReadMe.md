# Calculate R-squared for Regression Analysis (Easy, Machine Learning)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: R-squared (Coefficient of Determination)](#learn-r-squared-coefficient-of-determination)
- [Solution](#solution)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

---

## Problem Statement

### [Calculate R-squared for Regression Analysis](https://www.deep-ml.com/problems/69)

Write a Python function that computes the **R-squared (R²)** value for a regression model.

The function should:

- Accept the true target values.
- Accept the predicted target values.
- Compute the Residual Sum of Squares (SSR).
- Compute the Total Sum of Squares (SST).
- Return the R² value.

R-squared, also known as the **Coefficient of Determination**, measures how well the predicted values explain the variability of the true target values.

---

## Example

### Input

```python
import numpy as np

y_true = np.array([1, 2, 3, 4, 5])
y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])

print(r_squared(y_true, y_pred))
```

### Output

```python
0.989
```

### Reasoning

First compute the **Residual Sum of Squares (SSR)**

$$
SSR=\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

which measures the prediction error.

Next compute the **Total Sum of Squares (SST)**

$$
SST=\sum_{i=1}^{n}(y_i-\bar{y})^2
$$

which measures the total variation in the target values.

Finally,

$$
R^2=1-\frac{SSR}{SST}
$$

Since the prediction errors are very small, the resulting value is **0.989**, meaning the model explains **98.9%** of the variance in the data.

---

## Learn: R-squared (Coefficient of Determination)

### What is it?

**R-squared (R²)** is one of the most commonly used evaluation metrics for **regression models**.

It measures **how much of the variation in the target variable is explained by the model**.

Instead of looking directly at prediction errors like **MAE** or **RMSE**, R² compares your model against a simple baseline: always predicting the mean of the target values.

If the regression model performs much better than predicting the mean, the R² score approaches **1**.

---

### Mathematical Definition

#### Residual Sum of Squares (SSR)

$$
SSR=\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

where

- $y_i$ is the actual value.
- $\hat{y}_i$ is the predicted value.

SSR measures the total prediction error.

---

#### Total Sum of Squares (SST)

$$
SST=\sum_{i=1}^{n}(y_i-\bar{y})^2
$$

where

$$
\bar{y}=\frac{1}{n}\sum_{i=1}^{n}y_i
$$

is the mean of the target values.

SST measures the total variation in the data.

---

#### R-squared Formula

$$
R^2=1-\frac{SSR}{SST}
$$

The closer **SSR** is to zero, the closer **R²** is to one.

---

### Geometric Interpretation

Suppose the true values are

```text
1   2   3   4   5
```

The simplest prediction is their mean.

```text
Mean Prediction

3   3   3   3   3
```

A regression model should outperform this constant prediction.

R² measures **how much better the model performs than always predicting the mean**.

If the model removes most of the prediction error left by the mean, R² approaches **1**.

---

### Characteristics / Key Points

- Used only for regression problems.
- Measures the goodness of fit.
- Compares the model against predicting the mean.
- Dimensionless metric.
- Easy to interpret.

---

### Range of R²

| R² Value | Meaning                                           |
| -------- | ------------------------------------------------- |
| **1**    | Perfect prediction                                |
| **0**    | Model performs no better than predicting the mean |
| **< 0**  | Model performs worse than predicting the mean     |

Unlike many evaluation metrics, **R² can be negative** if the model performs poorly.

---

### Relationship Between SSR and SST

The prediction quality depends on the relationship between

- **SSR** (prediction error)
- **SST** (total variation)

#### Perfect Model

$$
SSR=0
$$

$$
R^2=1
$$

---

#### Mean Predictor

$$
SSR=SST
$$

$$
R^2=0
$$

---

#### Poor Model

If

$$
SSR>SST
$$

then

$$
R^2<0
$$

meaning the regression model performs worse than simply predicting the average value.

---

### Why is it used? / Applications

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
> A high **R²** does not necessarily mean the model is good. An overfitted model can achieve an excellent R² on the training data while performing poorly on unseen data. Always evaluate regression models on separate validation or test data and consider metrics like **RMSE** or **MAE** alongside R².

---

## Solution

### Custom Implementation

```python
import numpy as np

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    return 1 - (ss_res / ss_tot)
```

---

## Code Explanation

### Step 1

Compute the Residual Sum of Squares (SSR).

```python
ss_res = np.sum((y_true - y_pred) ** 2)
```

This computes

$$
SSR=\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

which represents the total squared prediction error.

---

### Step 2

Compute the Total Sum of Squares (SST).

```python
ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
```

This computes

$$
SST=\sum_{i=1}^{n}(y_i-\bar{y})^2
$$

which measures the total variance in the target values.

---

### Step 3

Compute the R-squared value.

```python
1 - (ss_res / ss_tot)
```

This applies the formula

$$
R^2=1-\frac{SSR}{SST}
$$

A smaller prediction error (SSR) produces a larger R² value.

---

## Time & Space Complexity

Let

- $n$ be the number of samples.

| Complexity | Value    |
| ---------- | -------- |
| Time       | **O(n)** |
| Space      | **O(1)** |

The algorithm scans the input arrays a constant number of times, resulting in **O(n)** time complexity. Aside from a few scalar variables, it does not allocate additional memory proportional to the input size, giving **O(1)** auxiliary space.
