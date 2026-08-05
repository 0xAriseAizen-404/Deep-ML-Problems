# Linear Regression Using Gradient Descent (Easy, Machine Learning)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Linear Regression Using Gradient Descent](#learn-linear-regression-using-gradient-descent)
- [Solutions](#solutions)
  - [Custom Implementation](#custom-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

## Problem Statement

### [Linear Regression Using Gradient Descent](https://www.deep-ml.com/problems/15)

Write a Python function that trains a **Linear Regression** model using **Batch Gradient Descent**. The function receives the feature matrix `X` (already containing a bias column of ones), the target vector `y`, the learning rate `alpha`, and the number of iterations.

The algorithm should:

- Initialize all model weights to zero.
- Use the **Mean Squared Error (MSE)** loss function.
- Perform **Batch Gradient Descent**, where every update uses the entire training dataset.
- Return the learned coefficients as a NumPy array.

---

## Example

### Input

```python
X = np.array([
    [1, 1],
    [1, 2],
    [1, 3]
])

y = np.array([3, 5, 7])

alpha = 0.1
iterations = 1000
```

### Output

```python
[1.0, 2.0]
```

### Reasoning

The training data follows the equation

$$
y=1+2x
$$

Starting with

$$
\theta=
\begin{bmatrix}
0\\
0
\end{bmatrix}
$$

Gradient Descent repeatedly updates the parameters until they converge to

$$
\theta=
\begin{bmatrix}
1\\
2
\end{bmatrix}
$$

which perfectly fits the data.

---

## Learn: Linear Regression Using Gradient Descent

### What is Linear Regression?

Linear Regression is a supervised learning algorithm used to predict a continuous target variable.

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
- $\theta_i$ are feature coefficients.
- $x_i$ are input features.

The objective is to learn the parameters that minimize prediction error.

---

### Why Gradient Descent?

The **Normal Equation** computes the optimal solution directly, but it requires matrix inversion, which becomes expensive when the number of features is large.

Gradient Descent avoids matrix inversion by **iteratively** improving the parameters.

Instead of solving for the optimum in one step, it gradually moves toward the minimum of the loss function.

---

### Prediction Function

The prediction (hypothesis) is

$$
h_\theta(x)=X\theta
$$

where

- $X$ is the feature matrix.
- $\theta$ is the parameter vector.

The predicted values are

$$
\hat{y}=X\theta
$$

---

### Mean Squared Error (MSE)

The objective is to minimize the Mean Squared Error loss.

$$
J(\theta)=\frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})^2
$$

where

- $m$ is the number of training examples.
- $y^{(i)}$ is the true value.
- $h_\theta(x^{(i)})$ is the predicted value.

The factor

$$
\frac{1}{2}
$$

is included only to simplify differentiation.

---

### Gradient of the Loss

Taking the derivative of the loss function gives

$$
\nabla J(\theta)=\frac{1}{m}X^T(X\theta-y)
$$

The gradient points toward the direction of **maximum increase** in the loss.

To minimize the loss, we move in the opposite direction.

---

### Gradient Descent Update Rule

The parameters are updated using

$$
\theta:=\theta-\alpha\nabla J(\theta)
$$

Substituting the gradient,

$$
\theta:=\theta-\alpha\frac{1}{m}X^T(X\theta-y)
$$

where

- $\alpha$ is the learning rate.

This update is repeated until convergence or until the desired number of iterations is reached.

---

### Batch Gradient Descent

Batch Gradient Descent computes the gradient using **every training example** during each iteration.

Algorithm:

1. Compute predictions.
2. Compute errors.
3. Compute the gradient.
4. Update all parameters simultaneously.
5. Repeat.

This produces stable updates but can become slow on very large datasets.

---

### Choosing the Learning Rate

The learning rate determines how large each update step is.

- Small $\alpha$ → slow convergence.
- Large $\alpha$ → may overshoot the minimum.
- Proper $\alpha$ → fast and stable convergence.

Choosing an appropriate learning rate is critical for successful training.

---

### Characteristics / Key Points

- Iterative optimization algorithm.
- No matrix inversion required.
- Works well for large datasets.
- Easily extends to many ML models.
- Requires selecting a learning rate.
- May converge slowly depending on hyperparameters.
- Supports high-dimensional feature spaces.

---

### Batch vs Stochastic vs Mini-Batch

| Batch Gradient Descent | Stochastic Gradient Descent | Mini-Batch Gradient Descent         |
| ---------------------- | --------------------------- | ----------------------------------- |
| Uses all samples       | Uses one sample             | Uses a small batch                  |
| Stable updates         | Noisy updates               | Balanced approach                   |
| Slower per iteration   | Fast updates                | Most commonly used in Deep Learning |
| Deterministic          | Randomized                  | Randomized                          |

---

### Why is it Used?

Gradient Descent is one of the most important optimization algorithms in Machine Learning and Deep Learning.

Applications include:

- Linear Regression
- Logistic Regression
- Neural Networks
- Deep Learning
- Recommendation Systems
- Computer Vision
- Natural Language Processing
- Reinforcement Learning

Nearly every deep learning model is trained using Gradient Descent or one of its variants.

---

> 💡 **Important Note**
>
> Gradient Descent is an optimization algorithm, not a Machine Learning model. Linear Regression defines the model, while Gradient Descent is one method for learning its parameters. In practice, optimizers such as **Momentum**, **RMSProp**, and **Adam** are often preferred because they typically converge faster and more reliably than vanilla Gradient Descent.

---

## Solutions

### Custom Implementation

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

        y_pred = X @ theta

        errors = y_pred - y

        gradient = (1 / m) * X.T @ errors

        theta = theta - alpha * gradient

    return theta.flatten()
```

---

## Code Explanation

### 1. Initialize Parameters

```python
theta = np.zeros((n, 1))
```

The algorithm starts with all weights equal to zero.

This is the initial model before learning begins.

---

### 2. Predict the Outputs

```python
y_pred = X @ theta
```

This computes

$$
\hat{y}=X\theta
$$

which represents the model's predictions for every training sample.

---

### 3. Compute Prediction Errors

```python
errors = y_pred - y
```

The error vector is

$$
e=\hat{y}-y
$$

Each element measures how far the prediction is from the true target.

---

### 4. Compute the Gradient

```python
gradient = (1 / m) * X.T @ errors
```

This implements

$$
\nabla J(\theta)=\frac{1}{m}X^T(X\theta-y)
$$

The gradient tells us how each parameter affects the loss.

---

### 5. Update the Parameters

```python
theta = theta - alpha * gradient
```

This applies the Gradient Descent update rule

$$
\theta:=\theta-\alpha\frac{1}{m}X^T(X\theta-y)
$$

The learning rate controls how large each update step is.

---

### 6. Return the Learned Weights

```python
return theta.flatten()
```

The final parameter vector is converted into a one-dimensional NumPy array before returning.

---

## Time & Space Complexity

Let

- $m$ = number of training samples.
- $n$ = number of features.
- $k$ = number of Gradient Descent iterations.

Each iteration performs matrix multiplication involving all samples and features.

| Complexity | Value            |
| ---------- | ---------------- |
| Time       | **O(k × m × n)** |
| Space      | **O(n)**         |

The algorithm stores the parameter vector and gradient, both proportional to the number of features.
