# Calculate Mean Absolute Error (MAE) (Easy, Machine Learning)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Mean Absolute Error (MAE)](#learn-mean-absolute-error-mae)
- [Solutions](#solutions)
  - [Custom Implementation](#custom-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

## Problem Statement

### [Calculate Mean Absolute Error (MAE)](https://www.deep-ml.com/problems/93)

Implement a Python function `mae(y_true, y_pred)` that calculates the Mean Absolute Error between actual values and predicted values.

The function receives:

- `y_true`: Array containing actual target values.
- `y_pred`: Array containing predicted values from the model.

MAE measures the average magnitude of prediction errors without considering whether predictions are higher or lower than the actual values.

The formula for Mean Absolute Error is:

$$ MAE = \frac{1}{n}\sum\_{i=1}^{n}|y_i-\hat{y_i}| $$

where:

- $n$ represents the number of samples.
- $y_i$ represents the actual value.
- $\hat{y_i}$ represents the predicted value.

The function should return the MAE as a floating-point value.

## Example

### Input

```python
import numpy as np

y_true = np.array([3, -0.5, 2, 7])
y_pred = np.array([2.5, 0.0, 2, 8])

result = mae(y_true, y_pred)
print(result)
```

### Output

```text
0.5
```

### Reasoning

Calculate the absolute differences between actual and predicted values:

$$ |3-2.5| = 0.5 $$

$$ |-0.5-0.0| = 0.5 $$

$$ |2-2| = 0 $$

$$ |7-8| = 1 $$

The average error is:

$$ MAE = \frac{0.5+0.5+0+1}{4} $$

$$ MAE = \frac{2}{4} = 0.5 $$

## Learn: Mean Absolute Error (MAE)

### What is it?

Mean Absolute Error (MAE) is a regression evaluation metric that measures the average absolute difference between actual values and predicted values.

It shows how far predictions are from the real values on average.

Unlike signed error, MAE ignores the direction of the error by taking the absolute value.

Example:

- Prediction higher than actual value:
  - Error is positive.

- Prediction lower than actual value:
  - Error is negative.

Taking absolute values makes both contribute equally to the final error.

MAE is commonly used to evaluate regression models because it is simple to interpret and remains in the same units as the target variable.

### Mathematical Definition / Formula

The Mean Absolute Error is:

$$ MAE = \frac{1}{n}\sum\_{i=1}^{n}|y_i-\hat{y_i}| $$

where:

- $n$ is the number of observations.
- $y_i$ is the actual value.
- $\hat{y_i}$ is the predicted value.
- $|y_i-\hat{y_i}|$ is the absolute prediction error.

The error for each sample is:

$$ Error_i = |y_i-\hat{y_i}| $$

The final MAE is the average of all individual errors.

### Characteristics / Key Points

- MAE is always non-negative.

$$ MAE \geq 0 $$

- Perfect predictions produce:

$$ MAE = 0 $$

- MAE is measured in the same units as the target variable.
- All errors contribute equally.
- MAE is less sensitive to outliers compared to MSE.

Example:

For house price prediction:

- Target:
  - Price in dollars.

- MAE:
  - Average prediction error in dollars.

### MAE vs MSE

Mean Squared Error (MSE) is another common regression metric:

$$ MSE = \frac{1}{n}\sum\_{i=1}^{n}(y_i-\hat{y_i})^2 $$

Differences:

| Metric | Error Calculation   | Effect                      |
| ------ | ------------------- | --------------------------- |
| MAE    | Absolute difference | Treats all errors equally   |
| MSE    | Squared difference  | Penalizes large errors more |

MSE gives more importance to outliers because errors are squared.

Example:

An error of:

$$ Error = 10 $$

becomes:

$$ Error^2 = 100 $$

in MSE.

### Why is it used? / Applications

MAE is widely used in regression problems.

Applications:

- House price prediction.
- Stock price prediction.
- Weather forecasting.
- Demand forecasting.
- Machine learning regression evaluation.

> 💡 **Important Note**
>
> MAE is easier to interpret because it remains in the original unit of the target variable. For example, an MAE of $5000 in house price prediction means the model is wrong by approximately $5000 on average.

## Solutions

### Custom Implementation

```python id="5u3qnm"
import numpy as np

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))
```

## Code Explanation

### Step 1: Calculate Prediction Errors

```python id="9m4f8p"
y_true - y_pred
```

The difference between actual and predicted values gives the prediction error.

Example:

```text id="5c7l2e"
y_true = [3, 5]
y_pred = [2, 7]

errors = [1, -2]
```

### Step 2: Convert Errors to Absolute Values

```python id="3n9v0d"
np.abs(y_true - y_pred)
```

Absolute values remove the negative sign so all errors contribute positively.

Example:

```text id="z3w4qk"
[1, -2] -> [1, 2]
```

### Step 3: Calculate Mean Error

```python id="x7r9ak"
np.mean(np.abs(y_true - y_pred))
```

The function averages all absolute errors to produce the final MAE value.

## Time & Space Complexity

| Complexity | Value    |
| ---------- | -------- |
| Time       | **O(n)** |
| Space      | **O(1)** |

Where:

- **n** is the number of samples.
- Each element is processed once.
- No additional storage is required apart from intermediate NumPy operations.
