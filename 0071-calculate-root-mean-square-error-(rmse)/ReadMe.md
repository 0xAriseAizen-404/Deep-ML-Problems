# Calculate Root Mean Square Error (RMSE) (Easy, Machine Learning)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Root Mean Square Error (RMSE)](#learn-root-mean-square-error-rmse)
  - [What is it?](#what-is-it)
  - [Mathematical Definition / Formula](#mathematical-definition--formula)
  - [Characteristics / Key Points](#characteristics--key-points)
  - [Why is it used? / Applications](#why-is-it-used--applications)
- [Solution](#solution)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

---

## Problem Statement

[Calculate Root Mean Square Error (RMSE)](https://www.deep-ml.com/problems/71)

Write a Python function `rmse(y_true, y_pred)` that computes the **Root Mean Square Error (RMSE)** between the true values and predicted values.

The function should:

- Calculate the RMSE between `y_true` and `y_pred`.
- Return the result rounded to **three decimal places**.
- Handle invalid inputs such as:
  - Non-NumPy arrays.
  - Empty arrays.
  - Arrays with mismatched shapes.

RMSE is one of the most commonly used metrics for evaluating **regression models** because it measures the average magnitude of prediction errors while giving larger errors a higher penalty.

---

## Example

```python
import numpy as np

y_true = np.array([3, -0.5, 2, 7])
y_pred = np.array([2.5, 0.0, 2, 8])

print(rmse(y_true, y_pred))
```

### Output

```text
0.612
```

### Explanation

Compute the prediction errors:

| Actual | Predicted | Error | Squared Error |
| :----: | :-------: | :---: | :-----------: |
|   3    |    2.5    |  0.5  |     0.25      |
|  -0.5  |    0.0    | -0.5  |     0.25      |
|   2    |     2     |   0   |       0       |
|   7    |     8     |  -1   |       1       |

Mean Squared Error (MSE):

$$
\begin{aligned}
\text{MSE}
&=
\frac{0.25+0.25+0+1}{4} \\
&=
0.375
\end{aligned}
$$

Root Mean Square Error:

$$
\begin{aligned}
\text{RMSE}
&=
\sqrt{0.375} \\
&=
0.612
\end{aligned}
$$

---

# Learn: Root Mean Square Error (RMSE)

## What is it?

**Root Mean Square Error (RMSE)** is one of the most popular evaluation metrics for **regression models**.

It measures the average difference between the **actual values** and the **predicted values**, indicating how accurately a model makes predictions.

Unlike **Mean Absolute Error (MAE)**, RMSE squares each error before averaging. As a result, **larger errors contribute much more heavily**, making RMSE sensitive to outliers.

A lower RMSE indicates better model performance, while an RMSE of **0** represents perfect predictions.

---

## Mathematical Definition / Formula

Suppose we have:

- Actual values

$$
y_{\text{true}} =
[y_1,y_2,\dots,y_n]
$$

- Predicted values

$$
y_{\text{pred}} =
[\hat{y}_1,\hat{y}_2,\dots,\hat{y}_n]
$$

The prediction error for each observation is

$$
e_i =
y_i-\hat{y}_i
$$

The **Mean Squared Error (MSE)** is

$$
\boxed{
\text{MSE} =
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
}
$$

The **Root Mean Square Error (RMSE)** is simply the square root of the MSE:

$$
\boxed{
\text{RMSE} =
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
}
}
$$

where

- $n$ = Number of observations.
- $y_i$ = Actual value.
- $\hat{y}_i$ = Predicted value.

---

## Characteristics / Key Points

- RMSE ranges from **0** to **∞**.
- Lower values indicate better model performance.
- RMSE has the **same units as the target variable**, making it easy to interpret.
- Large prediction errors are penalized more heavily because the errors are squared.
- Sensitive to outliers.

### RMSE vs MSE

| Metric | Formula                   |     Units      |
| :----: | :------------------------ | :------------: |
|  MSE   | Average of squared errors | Squared units  |
|  RMSE  | Square root of MSE        | Original units |

Because RMSE returns the error in the **same units as the data**, it is usually easier to interpret than MSE.

---

### RMSE vs MAE

| RMSE                                        | MAE                                             |
| :------------------------------------------ | :---------------------------------------------- |
| Squares the errors                          | Uses absolute errors                            |
| Penalizes large errors heavily              | Treats every error equally                      |
| Sensitive to outliers                       | More robust to outliers                         |
| Preferred when large errors are undesirable | Preferred when all errors are equally important |

---

## Why is it used? / Applications

RMSE is widely used for evaluating regression models in areas such as:

- House price prediction
- Stock price forecasting
- Weather prediction
- Energy demand forecasting
- Sales forecasting
- Time series forecasting
- Deep Learning regression models

It is one of the standard evaluation metrics available in libraries such as **Scikit-Learn**, **TensorFlow**, and **PyTorch**.

> 💡 **Important Note**
>
> RMSE is sensitive to outliers because the errors are squared before averaging. A few large prediction errors can significantly increase the RMSE. If your dataset contains many outliers, **Mean Absolute Error (MAE)** may provide a more representative measure of model performance.

---

# Solution

## Custom Implementation

```python
import numpy as np

def rmse(y_true, y_pred):
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("y_true and y_pred must be NumPy arrays.")

    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("Input arrays cannot be empty.")

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true shape {y_true.shape} != y_pred shape {y_pred.shape}"
        )

    return np.round(
        np.sqrt(np.mean((y_true - y_pred) ** 2)),
        3
    )
```

---

# Code Explanation

### Step 1: Validate the Inputs

```python
if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
```

Ensure both inputs are NumPy arrays.

```python
if y_true.size == 0 or y_pred.size == 0:
```

Prevent computations on empty arrays.

```python
if y_true.shape != y_pred.shape:
```

Ensure both arrays have identical dimensions before computing element-wise operations.

---

### Step 2: Compute the Prediction Errors

```python
y_true - y_pred
```

This produces the residual (prediction error) for each observation.

Example:

```text
[3, -0.5, 2, 7]
-
[2.5, 0, 2, 8]

=

[0.5, -0.5, 0, -1]
```

---

### Step 3: Square the Errors

```python
(y_true - y_pred) ** 2
```

Result:

```text
[0.25, 0.25, 0, 1]
```

Squaring ensures:

- Negative errors become positive.
- Larger errors receive greater penalties.

---

### Step 4: Compute the Mean Squared Error

```python
np.mean(...)
```

Computes

$$
\boxed{
\text{MSE}
=
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
}
$$

---

### Step 5: Compute the RMSE

```python
np.sqrt(...)
```

Takes the square root of the MSE to convert the error back to the original units of the target variable.

---

### Step 6: Round the Result

```python
np.round(..., 3)
```

Rounds the RMSE to three decimal places before returning it.

---

## Time & Space Complexity

Let **n** be the number of observations.

| Complexity |  Value   |
| :--------: | :------: |
|    Time    | **O(n)** |
|   Space    | **O(n)** |

The algorithm performs a constant number of passes over the input arrays. The intermediate error array created by `(y_true - y_pred) ** 2` requires **O(n)** additional space.
