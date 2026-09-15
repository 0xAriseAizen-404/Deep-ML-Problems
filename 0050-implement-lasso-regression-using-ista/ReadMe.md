# [Implement Lasso Regression using ISTA](https://www.deep-ml.com/problems/50) (Medium, Machine Learning)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Lasso Regression with ISTA](#learn-lasso-regression-with-ista)
  - [Lasso Regression](#lasso-regression)
  - [Objective Function](#objective-function)
  - [L1 Regularization](#l1-regularization)
  - [Why L1 Produces Sparsity](#why-l1-produces-sparsity)
  - [Why Regular Gradient Descent Is Not Enough](#why-regular-gradient-descent-is-not-enough)
  - [ISTA](#ista)
  - [Gradient Step](#gradient-step)
  - [Proximal Step](#proximal-step)
  - [Soft-Thresholding](#soft-thresholding)
  - [Bias Update](#bias-update)
  - [Why the Bias Is Not Regularized](#why-the-bias-is-not-regularized)
  - [Convergence](#convergence)
  - [ISTA vs Ridge](#ista-vs-ridge)
  - [Feature Selection](#feature-selection)
  - [Practical Considerations](#practical-considerations)
- [Solutions](#solutions)
  - [Custom Implementation](#custom-implementation)
  - [NumPy Implementation](#numpy-implementation)
- [Code Explanation](#code-explanation)
  - [Soft Threshold Function](#soft-threshold-function)
  - [Initializing Parameters](#initializing-parameters)
  - [Computing the Prediction Error](#computing-the-prediction-error)
  - [Computing the Weight Gradient](#computing-the-weight-gradient)
  - [Computing the Bias Gradient](#computing-the-bias-gradient)
  - [Gradient Descent Step](#gradient-descent-step)
  - [Proximal Step](#proximal-step)
  - [Updating the Bias](#updating-the-bias)
  - [Convergence Check](#convergence-check)
  - [Returning the Model](#returning-the-model)
- [Step-by-Step Example](#step-by-step-example)
- [Understanding the Soft-Thresholding Operation](#understanding-the-soft-thresholding-operation)
- [Important Mathematical Details](#important-mathematical-details)
- [Common Mistakes](#common-mistakes)
- [Interview and Practical Notes](#interview-and-practical-notes)
- [Time and Space Complexity](#time-and-space-complexity)
- [Key Takeaways](#key-takeaways)

## Problem Statement

The task is to implement Lasso Regression using the Iterative
Shrinkage-Thresholding Algorithm (ISTA).

Lasso Regression is linear regression with an L1 regularization
penalty added to the ordinary Mean Squared Error loss.

The objective is to find weights $w$ and bias $b$ that minimize:

$$ J(w,b) = \frac{1}{2n}\sum*{i=1}^{n}(y_i-\hat{y}\_i)^2 + \alpha\sum*{j=1}^{p}|w_j| $$

where:

- $n$ is the number of training samples.
- $p$ is the number of features.
- $w$ is the weight vector.
- $b$ is the bias.
- $\hat{y}$ is the predicted output.
- $\alpha$ controls the strength of L1 regularization.

The prediction equation is:

$$ \hat{y} = Xw+b $$

The important property of Lasso is that the L1 penalty can
shrink some coefficients exactly to zero.

Therefore, Lasso can perform automatic feature selection.

The difficulty is that $|w|$ is not differentiable at zero.

ISTA solves this problem by separating the objective into:

1. A smooth MSE component handled using gradient descent.
2. A non-smooth L1 component handled using a proximal operator.

## Example

### Input

```python
import numpy as np

X = np.array([[1, 0.01], [2, 0.02], [3, 0.03], [4, 0.04], [5, 0.05]])
y = np.array([2, 4, 6, 8, 10])

weights, bias = l1_regularization_gradient_descent(X, y, alpha=0.5, learning_rate=0.01, max_iter=1000)
```

### Output

```text
weights = array([1.96, 0.0])
bias ≈ 0.1
```

### Reasoning

The target approximately follows:

\(y \approx 2x_1\)

The second feature is:

```text
0.01, 0.02, 0.03, 0.04, 0.05
```

and contributes little independent information.

The L1 penalty encourages unnecessary coefficients to become zero.

Therefore, ISTA can produce:

```text
w1 ≈ 1.96
w2 = 0
```

The second feature is effectively removed from the model.

This is the central advantage of Lasso over ordinary linear
regression and Ridge Regression.

## Learn: Lasso Regression with ISTA

## Lasso Regression

Lasso stands for Least Absolute Shrinkage and Selection Operator.

It is a regularized version of linear regression.

Ordinary linear regression minimizes only the prediction error:

\(J(w,b) = \frac{1}{2n}\sum\_{i=1}^{n}(y_i-\hat{y}\_i)^2\)

Lasso adds an L1 penalty:

\(J(w,b) = \frac{1}{2n}\sum*{i=1}^{n}(y_i-\hat{y}\_i)^2+\alpha\sum*{j=1}^{p}|w_j|\)

The first term tries to fit the data.

The second term discourages large coefficients.

Unlike L2 regularization, L1 regularization can force coefficients
to become exactly zero.

That makes Lasso useful when we have many features and believe that
only a subset of them is actually important.

## Objective Function

The complete Lasso objective can be written as:

\(J(w,b) = \underbrace{\frac{1}{2n}\|Xw+b-y\|_2^2}_{\text{smooth MSE loss}}+\underbrace{\alpha\|w\|_1}_{\text{non-smooth L1 penalty}}\)

The objective consists of two different mathematical components.

The MSE term is smooth and differentiable.

The L1 term is not differentiable at zero.

This distinction is the reason we use a proximal optimization method.

The parameter $\alpha$ determines the regularization strength.

A larger $\alpha$ means stronger pressure toward zero.

A smaller $\alpha$ means the model behaves more like ordinary
linear regression.

## L1 Regularization

The L1 norm of a vector is:

\(\|w\|_1 = \sum_{j=1}^{p}|w_j|\)

For example, if:

```text
w = [3, -2, 0.5]
```

then:

\(\|w\|\_1 = |3|+|-2|+|0.5|=5.5\)

The regularization penalty is therefore:

\(\alpha\|w\|\_1\)

L1 regularization penalizes coefficients according to their
absolute magnitude.

The absolute value function has a sharp corner at zero:

```text
        /
       /
------/------
     /
    /
```

That corner is the source of the non-differentiability.

## Why L1 Produces Sparsity

The key property of L1 regularization is that it can produce
exact zeros.

Suppose a weight after the gradient step is:

```text
w_temp = 0.03
```

and the threshold is:

```text
lambda = 0.05
```

Soft-thresholding gives:

\(S(0.03,0.05)=0\)

The coefficient is not merely made small.

It becomes exactly zero.

For a negative value:

\(S(-0.03,0.05)=0\)

For a larger positive value:

\(S(0.2,0.05)=0.15\)

For a larger negative value:

\(S(-0.2,0.05)=-0.15\)

Thus, Lasso performs two operations simultaneously:

- Shrink sufficiently large coefficients toward zero.
- Set sufficiently small coefficients exactly to zero.

This creates a sparse weight vector.

## Why Regular Gradient Descent Is Not Enough

For the smooth MSE term, gradient descent works normally.

The gradient with respect to $w$ is:

\(\nabla_w L = \frac{1}{n}X^T(Xw+b-y)\)

However, the derivative of $|w|$ is problematic at zero.

For non-zero values:

\(\frac{d}{dw}|w| = \operatorname{sign}(w)\)

But at zero:

\(\frac{d}{dw}|w|\)

does not have a unique value.

A subgradient can be used, but the standard proximal approach provides
a cleaner solution for this particular objective.

Instead of directly differentiating the L1 term, ISTA handles it
using its proximal operator.

## ISTA

ISTA stands for Iterative Shrinkage-Thresholding Algorithm.

It is a proximal gradient method.

The central idea is to split the objective:

\(J(w)=f(w)+g(w)\)

where:

\(f(w)=\frac{1}{2n}\|Xw+b-y\|\_2^2\)

is smooth, and:

\(g(w)=\alpha\|w\|\_1\)

is non-smooth.

ISTA then performs two steps at every iteration.

### Step 1: Gradient Step

Move in the direction that decreases the smooth MSE loss:

\(w\_{\text{temp}}=w-\eta\nabla_w f(w)\)

For Lasso:

\(w\_{\text{temp}}=w-\eta\frac{1}{n}X^T(Xw+b-y)\)

### Step 2: Proximal Step

Apply soft-thresholding:

\(w*{\text{new}}=S(w*{\text{temp}},\eta\alpha)\)

The threshold is:

\(\lambda=\eta\alpha\)

The combination of these two operations gives ISTA.

## Gradient Step

The prediction is:

\(\hat{y}=Xw+b\)

The error is:

\(e=\hat{y}-y\)

The MSE component is:

\(L(w,b)=\frac{1}{2n}\|e\|\_2^2\)

Its gradient with respect to the weights is:

\(\nabla_wL=\frac{1}{n}X^Te\)

Substituting the error:

\(\nabla_wL=\frac{1}{n}X^T(Xw+b-y)\)

The gradient descent update is therefore:

\(w\_{\text{temp}}=w-\eta\frac{1}{n}X^T(Xw+b-y)\)

This step does not handle the L1 penalty yet.

That happens in the next step.

## Proximal Step

The proximal operator handles the non-smooth part.

For the L1 norm, the proximal operator has a closed-form solution
called soft-thresholding.

The update is:

\(w*{\text{new}}=\operatorname{prox}*{\eta\alpha\|\cdot\|_1}(w_{\text{temp}})\)

which is equivalent to:

\(w*{\text{new}}=S(w*{\text{temp}},\eta\alpha)\)

This is why ISTA is computationally attractive.

We do not need to solve a complicated optimization problem for the
L1 penalty at every iteration.

A simple element-wise operation is sufficient.

## Soft-Thresholding

The soft-thresholding operator is:

\(S(w,\lambda)=\operatorname{sign}(w)\max(|w|-\lambda,0)\)

It works independently on every coefficient.

For a positive coefficient:

\(w>\lambda \Rightarrow S(w,\lambda)=w-\lambda\)

For a small coefficient:

\(|w|\leq\lambda \Rightarrow S(w,\lambda)=0\)

For a negative coefficient:

\(w<-\lambda \Rightarrow S(w,\lambda)=w+\lambda\)

An equivalent piecewise definition is:

\(S(w,\lambda)=\begin{cases}w-\lambda&w>\lambda\\0&|w|\leq\lambda\\w+\lambda&w<-\lambda\end{cases}\)

The important behavior is the middle case.

Every coefficient inside the interval:

\([-\lambda,\lambda]\)

is mapped exactly to zero.

## Bias Update

The bias is part of the smooth MSE loss.

Its gradient is:

\(\nabla*bL=\frac{1}{n}\sum*{i=1}^{n}(\hat{y}\_i-y_i)\)

Equivalently:

\(\nabla_bL=\operatorname{mean}(\hat{y}-y)\)

The bias update is therefore:

\(b\_{\text{new}}=b-\eta\nabla_bL\)

In the implementation:

```python
grad_b = np.mean(error)
bias -= learning_rate * grad_b
```

Notice that the bias does not go through soft-thresholding.

## Why the Bias Is Not Regularized

Lasso typically regularizes the feature coefficients but not the
intercept.

The objective is:

\(J(w,b)=L(w,b)+\alpha\|w\|\_1\)

and not:

\(J(w,b)=L(w,b)+\alpha(\|w\|\_1+|b|)\)

The bias represents the baseline level of the target.

Regularizing it can unnecessarily constrain the model's ability
to shift predictions vertically.

Therefore:

```text
weights → regularized
bias    → not regularized
```

This is the standard convention for linear regression with L1
regularization.

## Algorithm Summary

ISTA can be summarized as:

```text
Initialize:
    w = 0
    b = 0

Repeat:
    1. Compute predictions:
       y_hat = Xw + b

    2. Compute error:
       error = y_hat - y

    3. Compute MSE gradient:
       grad_w = (1/n) X.T @ error

    4. Gradient step:
       w_temp = w - learning_rate * grad_w

    5. Proximal step:
       w = soft_threshold(w_temp, learning_rate * alpha)

    6. Compute bias gradient:
       grad_b = mean(error)

    7. Update bias:
       b = b - learning_rate * grad_b

    8. Check convergence
```

The key point is that the weight update is not a single ordinary
gradient descent operation.

It is:

```text
gradient descent
       ↓
soft thresholding
```

That is the defining structure of ISTA.

## Why ISTA Works

The proximal gradient framework works because the objective is
separable into smooth and non-smooth components.

The smooth component can be handled using a normal gradient.

The non-smooth component can be handled using a proximal operator.

For Lasso:

\(f(w)=\frac{1}{2n}\|Xw+b-y\|\_2^2\)

and:

\(g(w)=\alpha\|w\|\_1\)

ISTA computes:

\(w*{k+1}=\operatorname{prox}*{\eta g}(w_k-\eta\nabla f(w_k))\)

For the L1 norm, the proximal operator is soft-thresholding.

Therefore:

\(w\_{k+1}=S(w_k-\eta\nabla f(w_k),\eta\alpha)\)

This gives an efficient iterative algorithm for Lasso optimization.

## Convergence

The learning rate controls how far the algorithm moves during
each gradient step.

If the learning rate is too large, the optimization can become
unstable or oscillate.

If it is too small, convergence can be very slow.

For a standard proximal gradient method, the step size is generally
chosen according to the Lipschitz constant of the smooth gradient.

For the MSE objective, the relevant quantity is related to the
largest eigenvalue of:

\(\frac{1}{n}X^TX\)

A conservative learning rate can therefore improve stability.

The implementation instead exposes `learning_rate` directly, allowing
the caller to choose an appropriate value.

## ISTA vs Ridge

Ridge Regression uses L2 regularization:

\(J\_{Ridge}=L(w,b)+\alpha\sum_jw_j^2\)

Lasso uses L1 regularization:

\(J\_{Lasso}=L(w,b)+\alpha\sum_j|w_j|\)

The major difference is sparsity.

Ridge generally shrinks coefficients toward zero but rarely makes
them exactly zero.

Lasso can make coefficients exactly zero.

Therefore:

```text
Ridge → coefficient shrinkage
Lasso → coefficient shrinkage + feature selection
```

This makes Lasso particularly useful when the number of features
is large and only some features are expected to be relevant.

## Feature Selection

Suppose a dataset contains:

```text
Feature 1
Feature 2
Feature 3
Feature 4
Feature 5
```

After training, Lasso might produce:

```text
w = [1.82, 0.00, -0.43, 0.00, 0.02]
```

Features 2 and 4 have coefficients exactly equal to zero.

The model therefore does not use them in its predictions.

The effective model becomes:

\(\hat{y}=1.82x_1-0.43x_3+0.02x_5+b\)

This can improve interpretability and reduce the number of
features required by the model.

However, zero coefficients do not automatically mean that those
features are universally useless.

Their selection depends on the dataset, regularization strength,
feature scaling, and correlations between features.

## Correlated Features

Lasso can behave differently when multiple features contain
similar information.

Suppose:

```text
x1 ≈ x2
```

Both features may explain the target equally well.

Lasso may keep one coefficient and shrink the other toward zero.

Therefore, feature selection by Lasso should be interpreted with
care when predictors are strongly correlated.

Elastic Net is often useful when we want both L1 sparsity and
L2 stabilization.

Its regularization combines:

\(\alpha_1\|w\|\_1+\alpha_2\|w\|\_2^2\)

## Practical Considerations

### Feature Scaling

Lasso is sensitive to feature scale.

Suppose:

```text
x1 ranges from 0 to 1
x2 ranges from 0 to 10000
```

The same coefficient magnitude can have very different effects
depending on the feature scale.

Regularization also interacts with coefficient magnitudes.

Therefore, standardizing features is usually important before
applying Lasso.

A common transformation is:

\(x'=\frac{x-\mu}{\sigma}\)

This gives features comparable scales.

### Choosing Alpha

The regularization strength $\alpha$ controls sparsity.

Small $\alpha$:

```text
less regularization
more features retained
```

Large $\alpha$:

```text
stronger regularization
more coefficients pushed toward zero
```

An excessively large value can cause underfitting.

In practical machine learning workflows, $\alpha$ is commonly chosen
using validation or cross-validation.

### Choosing Learning Rate

The learning rate controls the gradient step:

\(w\_{\text{temp}}=w-\eta\nabla L\)

A value that is too large can cause unstable updates.

A value that is too small can make convergence unnecessarily slow.

The learning rate and $\alpha$ play different roles:

```text
learning_rate → optimization step size
alpha         → regularization strength
```

They should not be confused.

## Solutions

## Custom Implementation

The submitted solution directly implements the two components of
ISTA: gradient descent and soft-thresholding.

```python
import numpy as np

def soft_threshold(w: np.ndarray, threshold: float) -> np.ndarray:
    sign_w = np.where(w < 0, -1, 1)
    # sign_w = np.sign(w)
    return sign_w * np.maximum(np.abs(w) - threshold, 0)

def l1_regularization_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float = 0.1, learning_rate: float = 0.01, max_iter: int = 1000, tol: float = 1e-4) -> tuple:
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0.0

    for _ in range(max_iter):
        error = np.dot(X, weights) + bias - y
        grad_w = (1 / n_samples) * np.dot(X.T, error)
        grad_b = np.mean(error)

        w_temp = weights - learning_rate * grad_w
        weights_new = soft_threshold(w_temp, learning_rate * alpha)
        bias -= learning_rate * grad_b
        if np.linalg.norm(weights_new - weights) < tol:
            weights = weights_new
            break
        weights = weights_new

    return weights, bias
```

## NumPy Implementation

The core NumPy implementation can also be written more compactly
while keeping exactly the same optimization procedure.

```python
import numpy as np

def soft_threshold(w, threshold):
    return np.sign(w) * np.maximum(np.abs(w) - threshold, 0)

def lasso_ista(X, y, alpha=0.1, learning_rate=0.01, max_iter=1000, tol=1e-4):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0.0

    for _ in range(max_iter):
        error = X @ weights + bias - y
        grad_w = (X.T @ error) / n_samples
        grad_b = np.mean(error)
        w_temp = weights - learning_rate * grad_w
        weights_new = soft_threshold(w_temp, learning_rate * alpha)
        bias -= learning_rate * grad_b

        if np.linalg.norm(weights_new - weights) < tol:
            weights = weights_new
            break

        weights = weights_new

    return weights, bias
```

The matrix operation:

```python
X @ weights
```

computes predictions from all samples at once.

The gradient:

```python
(X.T @ error) / n_samples
```

is the vectorized form of the MSE gradient.

The optimization logic is unchanged.

## Code Explanation

## Soft Threshold Function

The helper function is:

```python
def soft_threshold(w: np.ndarray, threshold: float) -> np.ndarray:
```

Its job is to apply the proximal operator of the L1 norm
element-wise.

The mathematical definition is:

\(S(w,\lambda)=\operatorname{sign}(w)\max(|w|-\lambda,0)\)

The code obtains the sign using:

```python
sign_w = np.where(w < 0, -1, 1)
```

For positive values, the sign is `1`.

For negative values, the sign is `-1`.

The magnitude reduction is computed using:

```python
np.maximum(np.abs(w) - threshold, 0)
```

The `maximum` operation is critical.

It prevents the result from becoming negative when the magnitude
is smaller than the threshold.

For example:

```text
abs(w) = 0.2
threshold = 0.5
```

then:

```text
max(0.2 - 0.5, 0)
= max(-0.3, 0)
= 0
```

Therefore, the coefficient becomes exactly zero.

## Sign Handling

The commented alternative is:

```python
# sign_w = np.sign(w)
```

`np.sign` is a natural NumPy implementation of the sign function.

The current implementation:

```python
np.where(w < 0, -1, 1)
```

assigns positive sign to zero as well.

This does not cause a practical issue here because the magnitude
term becomes zero whenever `w == 0`.

For example:

\(\operatorname{sign}(0)\max(|0|-\lambda,0)=0\)

regardless of whether the sign is represented as `0` or `1`.

## Initializing Parameters

The function begins with:

```python
n_samples, n_features = X.shape
```

This extracts the dimensions of the dataset.

If:

```text
X.shape = (5, 2)
```

then:

```text
n_samples = 5
n_features = 2
```

The weights are initialized to zero:

```python
weights = np.zeros(n_features)
```

Therefore:

```text
weights = [0, 0]
```

The bias is also initialized to zero:

```python
bias = 0.0
```

This gives the algorithm a simple starting point.

## Computing the Prediction Error

At each iteration, the model first computes:

```python
error = np.dot(X, weights) + bias - y
```

The prediction equation is:

\(\hat{y}=Xw+b\)

Therefore:

\(e=\hat{y}-y\)

The error vector contains one value for every training sample.

If the model predicts:

```text
[1, 3, 5]
```

while the true values are:

```text
[2, 4, 6]
```

then:

```text
error = [-1, -1, -1]
```

The gradient is computed from this error.

## Computing the Weight Gradient

The weight gradient is:

```python
grad_w = (1 / n_samples) * np.dot(X.T, error)
```

Mathematically:

\(\nabla_wL=\frac{1}{n}X^T(Xw+b-y)\)

The transpose of `X` is required because we need one gradient
value for every feature.

If:

```text
X.shape = (n, p)
```

then:

```text
X.T.shape = (p, n)
```

and:

```text
error.shape = (n,)
```

so:

```text
X.T @ error
```

produces:

```text
(p,)
```

which matches the shape of the weight vector.

## Computing the Bias Gradient

The bias gradient is:

```python
grad_b = np.mean(error)
```

Mathematically:

\(\nabla*bL=\frac{1}{n}\sum*{i=1}^{n}e_i\)

The bias affects every prediction equally.

Therefore, its gradient is simply the average prediction error.

If the average error is positive, predictions are generally too
large and the bias is decreased.

If the average error is negative, predictions are generally too
small and the bias is increased.

## Gradient Descent Step

The temporary weight vector is computed using:

```python
w_temp = weights - learning_rate * grad_w
```

This is ordinary gradient descent:

\(w\_{\text{temp}}=w-\eta\nabla_wL\)

At this point, the L1 penalty has not yet been applied.

This is intentional.

ISTA separates the smooth optimization step from the non-smooth
regularization step.

## Proximal Step

The next operation is:

```python
weights_new = soft_threshold(w_temp, learning_rate * alpha)
```

The threshold is:

\(\lambda=\eta\alpha\)

Therefore:

\(w*{\text{new}}=S(w*{\text{temp}},\eta\alpha)\)

This performs the L1 proximal operation.

For example, suppose:

```text
w_temp = [1.2, 0.03, -0.8]
learning_rate = 0.1
alpha = 0.5
```

Then:

\(\lambda=0.1\times0.5=0.05\)

Applying soft-thresholding:

```text
1.2  →  1.15
0.03 →  0
-0.8 → -0.75
```

The second coefficient disappears completely.

This is how the implementation performs feature selection.

## Updating the Bias

The bias is updated separately:

```python
bias -= learning_rate * grad_b
```

or mathematically:

\(b\_{\text{new}}=b-\eta\nabla_bL\)

There is no soft-thresholding operation on the bias.

Therefore, the L1 regularization affects only the feature weights.

## Convergence Check

After calculating the new weights, the implementation checks:

```python
if np.linalg.norm(weights_new - weights) < tol:
```

The difference is:

\(\Delta w=w*{\text{new}}-w*{\text{old}}\)

Its Euclidean norm is:

\(||\Delta w||\_2\)

If:

\(||w*{\text{new}}-w*{\text{old}}||\_2<\text{tol}\)

the algorithm considers the weight update sufficiently small.

The loop then terminates.

This prevents unnecessary iterations after the solution has mostly
stabilized.

## Updating the Current Weights

If convergence has not been reached:

```python
weights = weights_new
```

The new weights become the starting point for the next iteration.

The algorithm therefore repeatedly performs:

```text
current weights
       ↓
gradient step
       ↓
temporary weights
       ↓
soft thresholding
       ↓
new weights
       ↓
next iteration
```

## Returning the Model

After convergence or after reaching `max_iter`, the function returns:

```python
return weights, bias
```

The result is therefore a trained linear model represented by:

```text
weights
bias
```

The prediction function can then be written as:

```python
predictions = X @ weights + bias
```

## Step-by-Step Example

Consider:

```python
X = np.array([[1], [2], [3]])
y = np.array([2, 4, 6])
```

Initialize:

```text
w = 0
b = 0
```

Suppose:

```text
learning_rate = 0.1
alpha = 0.5
```

The first prediction is:

\(\hat{y}=Xw+b=0\)

Therefore:

```text
y_hat = [0, 0, 0]
```

The error is:

```text
[-2, -4, -6]
```

The weight gradient is:

\(\nabla_wL=\frac{1}{3}X^Te\)

which gives:

\(\nabla_wL=\frac{1}{3}(1(-2)+2(-4)+3(-6))=-\frac{28}{3}\)

The gradient step produces:

\(w\_{\text{temp}}=0-0.1\left(-\frac{28}{3}\right)\)

so:

\(w\_{\text{temp}}\approx0.9333\)

The soft-threshold value is:

\(\lambda=\eta\alpha=0.1(0.5)=0.05\)

Therefore:

\(w\_{\text{new}}=0.9333-0.05=0.8833\)

The coefficient remains positive but is slightly shrunk.

The next iteration repeats the same process.

Over many iterations, the model moves toward a solution balancing
prediction accuracy and the L1 penalty.

## Understanding the Soft-Thresholding Operation

Consider:

```text
w = [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]
```

and:

```text
lambda = 1
```

Soft-thresholding produces:

```text
[-2, -1, 0, 0, 0, 0, 0, 1, 2]
```

The transformation can be visualized as:

```text
Input:     -3  -2  -1  -0.5   0   0.5   1   2   3
Output:    -2  -1   0   0     0    0    0   1   2
                         └─────────────┘
                           zero region
```

The interval:

\([-\lambda,\lambda]\)

is mapped to zero.

Outside that interval, values are moved toward zero by exactly
$\lambda$.

This is why the operation is called "shrinkage-thresholding":

```text
shrink large values
+
threshold small values
```

## Important Mathematical Details

### Gradient of the MSE

The MSE term is:

\(L(w,b)=\frac{1}{2n}\|Xw+b-y\|\_2^2\)

Its gradient is:

\(\nabla_wL=\frac{1}{n}X^T(Xw+b-y)\)

and:

\(\nabla_bL=\frac{1}{n}\mathbf{1}^T(Xw+b-y)\)

The implementation uses:

```python
grad_b = np.mean(error)
```

which is exactly the same operation.

### Why the Factor 1/2 Exists

The objective uses:

\(\frac{1}{2n}\sum_i(y_i-\hat{y}\_i)^2\)

The factor $\frac{1}{2}$ simplifies differentiation.

Without it, the gradient would contain an additional factor of `2`:

\(\nabla_wL=\frac{2}{n}X^T(Xw+b-y)\)

The factor does not fundamentally change the optimization objective,
but it changes the scale of the gradient and therefore interacts
with the learning rate.

### Proximal Operator

The proximal operator of a function $g$ is:

\(\operatorname{prox}\_{\lambda g}(v)=\arg\min_z\left(\frac{1}{2}\|z-v\|\_2^2+\lambda g(z)\right)\)

For the L1 norm:

\(\operatorname{prox}\_{\lambda\|\cdot\|\_1}(v)=S(v,\lambda)\)

Therefore:

\(\operatorname{prox}\_{\lambda\|\cdot\|\_1}(v)=\operatorname{sign}(v)\max(|v|-\lambda,0)\)

This closed-form solution is what makes Lasso particularly convenient
for proximal gradient optimization.

## Common Mistakes

### Mistake 1: Applying L1 Through Normal Gradient Descent

A common implementation attempt is:

```python
grad_w += alpha * np.sign(weights)
```

This is related to a subgradient method, but it is not the ISTA
procedure requested by this problem.

ISTA explicitly separates:

```text
gradient step
+
proximal soft-thresholding
```

### Mistake 2: Forgetting the Threshold Scale

The threshold is not simply `alpha`.

It is:

\(\lambda=\eta\alpha\)

Therefore, the implementation correctly uses:

```python
soft_threshold(w_temp, learning_rate * alpha)
```

### Mistake 3: Regularizing the Bias

The bias should normally not be passed through soft-thresholding.

The implementation correctly performs:

```python
bias -= learning_rate * grad_b
```

without applying the L1 penalty.

### Mistake 4: Forgetting the Sample Average

The gradient uses:

\(\frac{1}{n}X^Te\)

not simply:

\(X^Te\)

The division by the number of samples keeps the gradient scale
consistent with the stated MSE objective.

### Mistake 5: Using the Wrong Error Sign

The implementation uses:

```python
error = prediction - y
```

which gives:

\(e=Xw+b-y\)

The gradient then follows:

\(\nabla_wL=\frac{1}{n}X^Te\)

Changing the sign without changing the update direction can cause
the algorithm to move away from the optimum.

### Mistake 6: Forgetting Soft Thresholding

If the implementation only performs:

```python
weights -= learning_rate * grad_w
```

then it is ordinary gradient descent on the MSE.

It does not implement ISTA.

The proximal step is essential:

```python
weights_new = soft_threshold(w_temp, learning_rate * alpha)
```

### Mistake 7: Not Scaling Features

Large differences in feature scale can distort the effect of
regularization.

Standardization is generally recommended before fitting Lasso.

### Mistake 8: Using an Extremely Large Alpha

A very large $\alpha$ can force most or all coefficients to zero.

The resulting model may underfit badly.

Regularization strength should therefore be selected carefully.

## Interview and Practical Notes

### What Is the Main Advantage of Lasso?

Lasso can produce exact zero coefficients.

This means it performs feature selection as part of model fitting.

### Why Is L1 Non-Differentiable?

The absolute value function:

\(|w|\)

has different one-sided derivatives around zero.

For positive values:

\(\frac{d}{dw}|w|=1\)

For negative values:

\(\frac{d}{dw}|w|=-1\)

At zero, there is no unique derivative.

### What Does ISTA Stand For?

ISTA stands for:

```text
Iterative Shrinkage-Thresholding Algorithm
```

It is a proximal gradient optimization algorithm.

### What Does "Proximal" Mean Here?

Instead of directly differentiating the non-smooth L1 term,
we solve a small optimization problem represented by its proximal
operator.

For L1, this operation has a simple closed-form solution:
soft-thresholding.

### Why Does Lasso Perform Feature Selection?

Because the proximal operator maps sufficiently small coefficients
to exactly zero:

\(|w|\leq\eta\alpha \Rightarrow w\_{\text{new}}=0\)

Repeated application of this operation can therefore eliminate
features from the model.

### Is Lasso the Same as Feature Selection Before Training?

No.

Lasso performs feature selection as part of model optimization.

It does not require a separate feature-selection algorithm first.

### Can Lasso Handle Many Features?

Yes.

Lasso is especially useful when the feature space is large and
we expect many features to be irrelevant.

Its sparse solution can produce a simpler model.

### What Happens When Alpha Is Zero?

If:

\(\alpha=0\)

then:

\(S(w,\eta\alpha)=S(w,0)=w\)

The proximal step does nothing.

ISTA therefore reduces to ordinary gradient descent on the MSE.

### What Happens When Alpha Becomes Very Large?

The threshold:

\(\eta\alpha\)

becomes large.

More coefficients are pushed to zero.

If the regularization is too strong, the model can underfit.

### Why Is Feature Scaling Important?

Lasso penalizes coefficient magnitude.

If features have very different scales, their coefficients are not
directly comparable.

Standardizing features makes the regularization behavior more
balanced across dimensions.

## Lasso vs Ordinary Linear Regression

| Property             | Linear Regression | Lasso                  |
| -------------------- | ----------------- | ---------------------- |
| Loss                 | MSE               | MSE + L1               |
| Regularization       | None              | L1                     |
| Shrinks coefficients | No                | Yes                    |
| Exact zeros          | Not generally     | Yes                    |
| Feature selection    | No                | Yes                    |
| Non-smooth objective | No                | Yes                    |
| ISTA required        | No                | One possible optimizer |

## Lasso vs Ridge

| Property            | Lasso          | Ridge                    |     |                    |
| ------------------- | -------------- | ------------------------ | --- | ------------------ |
| Penalty             | L1             | L2                       |     |                    |
| Penalty form        | $\alpha\sum    | w_j                      | $   | $\alpha\sum w_j^2$ |
| Exact zeros         | Yes            | Usually no               |     |                    |
| Feature selection   | Yes            | No                       |     |                    |
| Correlated features | Can select one | Often distributes weight |     |                    |
| Typical use         | Sparse models  | Stable shrinkage         |     |                    |

## ISTA vs Gradient Descent

Ordinary gradient descent on MSE performs:

\(w\_{k+1}=w_k-\eta\nabla L(w_k)\)

ISTA performs:

\(w\_{k+1}=S(w_k-\eta\nabla L(w_k),\eta\alpha)\)

The difference is the proximal step.

That single additional operation is what allows the algorithm to
optimize the L1-regularized objective effectively.

## ISTA vs Coordinate Descent

Coordinate descent updates one coefficient at a time.

ISTA updates the entire weight vector simultaneously.

Coordinate descent is another popular optimization approach for
Lasso and can be highly effective.

ISTA is particularly useful for understanding the connection between
gradient methods and proximal optimization.

## Acceleration

ISTA can be accelerated using FISTA, which stands for
Fast Iterative Shrinkage-Thresholding Algorithm.

FISTA introduces a momentum-like extrapolation step and can achieve
faster convergence in many convex optimization settings.

The conceptual progression is:

```text
Gradient Descent
      ↓
Proximal Gradient
      ↓
ISTA
      ↓
FISTA
```

Understanding ISTA first makes FISTA easier to understand.

## Numerical Considerations

The implementation uses:

```python
np.linalg.norm(weights_new - weights)
```

for convergence.

This checks the magnitude of the weight update rather than directly
checking the objective function.

That is a reasonable stopping criterion for this implementation.

A more complete optimizer could also monitor:

- Objective value.
- Relative parameter change.
- Gradient norm.
- Maximum number of iterations.

The current implementation keeps the convergence condition simple.

## One Important Implementation Detail

The convergence test occurs after calculating `weights_new` but before
assigning it to `weights`:

```python
if np.linalg.norm(weights_new - weights) < tol:
    weights = weights_new
    break
```

This assignment is important.

Without it, the function could stop with the previous weight vector
instead of the converged weight vector.

The implementation correctly stores the new weights before returning.

## Time and Space Complexity

Let:

- $n$ = number of samples.
- $p$ = number of features.
- $k$ = number of ISTA iterations.

For each iteration, the major operations are:

```text
X @ weights
X.T @ error
```

The prediction requires:

\(O(np)\)

operations.

The weight gradient also requires:

\(O(np)\)

operations.

Soft-thresholding operates on all $p$ weights:

\(O(p)\)

The bias update is:

\(O(n)\)

Therefore, the dominant cost per iteration is:

\(O(np)\)

For $k$ iterations:

\(O(knp)\)

The weight vector requires:

\(O(p)\)

space.

The input matrix itself occupies:

\(O(np)\)

space.

The output model requires:

\(O(p)\)

space for the weights and:

\(O(1)\)

for the bias.

If the input matrix is considered external storage, the algorithm's
additional working space is approximately:

\(O(p)\)

## Complexity Summary

| Operation                | Complexity |
| ------------------------ | ---------: |
| Prediction               |    $O(np)$ |
| Weight gradient          |    $O(np)$ |
| Bias gradient            |     $O(n)$ |
| Soft-thresholding        |     $O(p)$ |
| One ISTA iteration       |    $O(np)$ |
| $k$ ISTA iterations      |   $O(knp)$ |
| Additional working space |     $O(p)$ |
| Input storage            |    $O(np)$ |

## Key Takeaways

1. Lasso Regression adds an L1 penalty to the MSE objective.

2. The Lasso objective is:

\(J(w,b)=\frac{1}{2n}\|Xw+b-y\|\_2^2+\alpha\|w\|\_1\)

3. The MSE component is smooth and differentiable.

4. The L1 penalty is non-differentiable at zero.

5. ISTA solves this problem using proximal gradient descent.

6. The first step is an ordinary gradient descent update:

\(w\_{\text{temp}}=w-\eta\frac{1}{n}X^T(Xw+b-y)\)

7. The second step is soft-thresholding:

\(w*{\text{new}}=S(w*{\text{temp}},\eta\alpha)\)

8. The soft-thresholding operator is:

\(S(w,\lambda)=\operatorname{sign}(w)\max(|w|-\lambda,0)\)

9. Values satisfying:

\(|w|\leq\lambda\)

become exactly zero.

10. This exact-zero behavior allows Lasso to perform automatic
    feature selection.

11. The bias is updated using the MSE gradient and is normally not
    regularized.

12. The regularization strength $\alpha$ controls the amount of
    sparsity.

13. The learning rate controls the optimization step size.

14. Feature scaling is important because L1 regularization depends
    on coefficient magnitude.

15. Lasso differs from Ridge because Lasso can produce exact zeros,
    while Ridge generally only shrinks coefficients.

16. The dominant computational cost is the matrix multiplication
    in each iteration.

17. For $k$ iterations, $n$ samples, and $p$ features, the time
    complexity is:

\(O(knp)\)

18. The core idea of ISTA is:

```text
gradient descent
       +
soft thresholding
       =
L1-regularized optimization
```
