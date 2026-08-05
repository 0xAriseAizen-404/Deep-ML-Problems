# Linear Regression Using Normal Equation (Easy, Machine Learning)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Linear Regression Using the Normal Equation](#learn-linear-regression-using-the-normal-equation)
- [Solutions](#solutions)
  - [Custom Implementation](#custom-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

## Problem Statement

### [Linear Regression Using Normal Equation](https://www.deep-ml.com/problems/14)

Write a Python function that computes the coefficients of a **Linear Regression** model using the **Normal Equation**. The function takes a feature matrix `X` and a target vector `y` as input and returns the model parameters rounded to **4 decimal places**.

The Normal Equation computes the optimal parameters directly without using iterative optimization methods like Gradient Descent.

---

## Example

### Input

```python
X = [
    [1, 1],
    [1, 2],
    [1, 3]
]

y = [1, 2, 3]
```

### Output

```python
[0.0, 1.0]
```

### Reasoning

The first column of `X` represents the bias feature (all ones).

The resulting model is

$$
y=0+1x
$$

which perfectly fits every training sample.

---

## Learn: Linear Regression Using the Normal Equation

### What is Linear Regression?

Linear Regression is one of the most fundamental supervised machine learning algorithms. It models the relationship between one or more input features and a continuous target variable by fitting the best possible linear equation.

For a single feature,

$$
y=\theta_0+\theta_1x
$$

For multiple features,

$$
y=\theta_0+\theta_1x_1+\theta_2x_2+\cdots+\theta_nx_n
$$

where

- $\theta_0$ is the intercept (bias).
- $\theta_i$ are the feature coefficients.
- $x_i$ are the input features.

The objective is to find the coefficients that minimize the prediction error.

---

### Cost Function

Linear Regression minimizes the **Mean Squared Error (MSE)** cost function.

$$
J(\theta)=\frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})^2
$$

where

- $m$ is the number of training samples.
- $h_\theta(x)$ is the predicted value.

The Normal Equation finds the parameters that minimize this cost analytically.

---

### Normal Equation

Instead of updating parameters iteratively, Linear Regression has a closed-form solution.

The optimal parameter vector is

$$
\theta=(X^TX)^{-1}X^Ty
$$

where

- $X$ is the feature matrix.
- $X^T$ is the transpose of $X$.
- $(X^TX)^{-1}$ is the inverse of the Gram matrix.
- $y$ is the target vector.

The resulting vector

$$
\theta=
\begin{bmatrix}
\theta_0\\
\theta_1\\
\vdots\\
\theta_n
\end{bmatrix}
$$

contains all learned model parameters.

---

### Why Does it Work?

The Normal Equation is obtained by differentiating the Mean Squared Error cost function with respect to every parameter and setting the derivatives equal to zero.

This produces the unique least-squares solution whenever

$$
X^TX
$$

is invertible.

---

### Bias Term

To learn the intercept automatically, the feature matrix usually contains a column of ones.

For example,

$$
X=
\begin{bmatrix}
1 & x_1\\
1 & x_2\\
1 & x_3
\end{bmatrix}
$$

The first coefficient learned becomes the intercept

$$
\theta_0
$$

while the remaining coefficients correspond to feature weights.

---

### Characteristics / Key Points

- Produces the exact least-squares solution.
- No learning rate is required.
- No iterative optimization.
- No epochs or convergence criteria.
- Does not require feature scaling for correctness.
- Requires matrix inversion.
- Works well when the number of features is relatively small.

---

### Limitations

The Normal Equation becomes computationally expensive for datasets with many features because matrix inversion has cubic complexity.

Additionally,

$$
X^TX
$$

must be invertible.

If it is singular, alternatives include:

- Moore-Penrose Pseudoinverse
- Ridge Regression
- Feature removal
- Regularization

---

### Normal Equation vs Gradient Descent

| Normal Equation               | Gradient Descent                     |
| ----------------------------- | ------------------------------------ |
| Analytical solution           | Iterative optimization               |
| No learning rate              | Requires learning rate               |
| No epochs                     | Multiple epochs                      |
| Matrix inversion required     | Matrix inversion not required        |
| Better for small feature sets | Better for high-dimensional datasets |

---

### Why is it Used?

Linear Regression is widely used in:

- House price prediction
- Stock price forecasting
- Sales prediction
- Demand forecasting
- Medical data analysis
- Economic modeling
- Recommendation systems
- Baseline machine learning models

It is often the first model built for regression problems because of its simplicity and interpretability.

---

> 💡 **Important Note**
>
> In practice, directly computing $(X^TX)^{-1}$ is numerically less stable than solving the system using matrix factorization. Libraries such as NumPy typically recommend `np.linalg.solve()` or the Moore-Penrose pseudoinverse (`np.linalg.pinv`) instead of explicitly computing the inverse, especially for large or nearly singular matrices.

---

## Solutions

### Custom Implementation

```python
import numpy as np

def linear_regression_normal_equation(
    X: list[list[float]],
    y: list[float]
) -> list[float]:

    X = np.asarray(X)
    y = np.asarray(y)

    theta = np.linalg.inv(
        X.T @ X
    ) @ X.T @ y

    return np.round(theta, 4).tolist()
```

---

## Code Explanation

### 1. Convert Inputs into NumPy Arrays

```python
X = np.asarray(X)
y = np.asarray(y)
```

Converting the input lists into NumPy arrays enables efficient matrix operations.

---

### 2. Compute the Transpose

```python
X.T
```

The transpose swaps rows and columns.

If

$$
X\in\mathbb{R}^{m\times n}
$$

then

$$
X^T\in\mathbb{R}^{n\times m}
$$

---

### 3. Compute the Gram Matrix

```python
X.T @ X
```

This computes

$$
X^TX
$$

which is a square matrix of size

$$
n\times n
$$

---

### 4. Compute the Inverse

```python
np.linalg.inv(X.T @ X)
```

This evaluates

$$
(X^TX)^{-1}
$$

which exists only if the matrix is invertible.

---

### 5. Compute the Coefficient Vector

```python
theta = (
    np.linalg.inv(X.T @ X)
    @ X.T
    @ y
)
```

This directly implements

$$
\theta=(X^TX)^{-1}X^Ty
$$

producing the optimal least-squares solution.

---

### 6. Round the Coefficients

```python
return np.round(theta, 4).tolist()
```

The coefficients are rounded to four decimal places before returning them as a Python list.

---

## Time & Space Complexity

Let

- $m$ = number of training samples.
- $n$ = number of features.

The dominant operation is inverting the

$$
n\times n
$$

matrix.

| Complexity | Value           |
| ---------- | --------------- |
| Time       | **O(mn² + n³)** |
| Space      | **O(n²)**       |

The $mn^2$ term comes from computing $X^TX$, while the $n^3$ term comes from matrix inversion. For large feature spaces, iterative methods like Gradient Descent are generally preferred.
