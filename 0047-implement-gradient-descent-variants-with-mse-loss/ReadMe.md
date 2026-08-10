# Implement Gradient Descent Variants with MSE Loss (Medium, Machine Learning)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Gradient Descent Variants with MSE Loss](#learn-gradient-descent-variants-with-mse-loss)
- [Solution](#solution)
  - [Custom Implementation](#custom-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

---

## Problem Statement

### [Implement Gradient Descent Variants with MSE Loss](https://www.deep-ml.com/problems/47)

Implement a function that performs three variants of gradient descent for a linear regression model using **Mean Squared Error (MSE)** as the loss function:

- Batch Gradient Descent
- Stochastic Gradient Descent
- Mini-Batch Gradient Descent

The function should accept a `method` parameter that determines which variant is used.

The data must be processed in its original order without shuffling.

For:

- **Batch GD:** use the complete dataset for one gradient update per epoch.
- **Stochastic GD:** update the weights after every individual sample.
- **Mini-Batch GD:** divide consecutive samples into non-overlapping batches and update after each batch.
- **Epoch:** one complete pass through the training dataset.

---

## Example

### Input

```python
import numpy as np

X = np.array([[1, 1], [2, 1], [3, 1], [4, 1]])
y = np.array([2, 3, 4, 5])

learning_rate = 0.01
n_epochs = 1000
batch_size = 2

weights = np.zeros(X.shape[1])

batch_weights = gradient_descent(
    X, y, weights, learning_rate, n_epochs, method="batch"
)

stochastic_weights = gradient_descent(
    X, y, weights, learning_rate, n_epochs, method="stochastic"
)

mini_batch_weights = gradient_descent(
    X, y, weights, learning_rate, n_epochs,
    batch_size, method="mini_batch"
)
```

### Output

```text
[float, float]
[float, float]
[float, float]
```

### Reasoning

The model uses the linear prediction

$$
\hat{y}^{(i)}=x^{(i)T}\theta
$$

where $\theta$ represents the model weights.

At every update, the gradient of the MSE loss is calculated using either the complete dataset, one sample, or a mini-batch.

The weights are then updated using

$$
\theta=\theta-\alpha\nabla J(\theta)
$$

where $\alpha$ is the learning rate.

After the specified number of epochs, the final weight vector is returned.

---

## Learn: Gradient Descent Variants with MSE Loss

### What is it?

**Gradient Descent** is an iterative optimization algorithm used to minimize a loss function.

In linear regression, the model predicts a target using a weighted combination of input features:

$$
\hat{y}=X\theta
$$

The goal is to find weights $\theta$ that minimize the difference between predictions and the actual targets.

For regression, **Mean Squared Error (MSE)** is commonly used as the loss function:

$$
J(\theta)=\frac{1}{m}\sum_{i=1}^{m}(\hat{y}^{(i)}-y^{(i)})^2
$$

where

- $m$ is the number of training samples.
- $y^{(i)}$ is the actual target.
- $\hat{y}^{(i)}$ is the prediction.
- $\theta$ is the vector of model parameters.

Gradient descent repeatedly calculates the gradient of the loss with respect to the weights and moves the weights in the opposite direction.

---

### Gradient of MSE

For a linear model,

$$
\hat{y}=X\theta
$$

the MSE gradient with respect to the weight vector is

$$
\nabla J(\theta)=\frac{2}{m}X^T(X\theta-y)
$$

Therefore, the standard gradient descent update is

$$
\theta=\theta-\alpha\frac{2}{m}X^T(X\theta-y)
$$

The factor `2` comes from differentiating the squared error.

---

### Epoch and Iteration

An **epoch** is one complete pass through the entire training dataset.

An **iteration** or **update** is one weight update.

The number of updates per epoch depends on the gradient descent variant.

| Method        | Samples per Update |  Updates per Epoch |
| ------------- | -----------------: | -----------------: |
| Batch GD      |                $m$ |                $1$ |
| Stochastic GD |                $1$ |                $m$ |
| Mini-Batch GD |                $b$ | $\lceil m/b\rceil$ |

where $m$ is the number of samples and $b$ is the batch size.

---

### 1. Batch Gradient Descent

**Batch Gradient Descent** calculates the gradient using the entire training dataset before updating the weights.

For one epoch:

$$
\nabla J(\theta)=\frac{2}{m}\sum_{i=1}^{m}(\hat{y}^{(i)}-y^{(i)})x^{(i)}
$$

The update becomes

$$
\theta=\theta-\alpha\frac{2}{m}\sum_{i=1}^{m}(\hat{y}^{(i)}-y^{(i)})x^{(i)}
$$

Therefore, one epoch produces exactly **one weight update**.

#### Characteristics

- Uses the entire dataset for every update.
- Gradient is relatively stable.
- Updates are deterministic when the data order and initialization are fixed.
- Can become computationally expensive for very large datasets.
- Usually requires more memory for processing a large batch.

---

### 2. Stochastic Gradient Descent

**Stochastic Gradient Descent (SGD)** updates the weights after processing each individual training sample.

For sample $i$:

$$
\nabla J_i(\theta)=2(\hat{y}^{(i)}-y^{(i)})x^{(i)}
$$

The update is

$$
\theta=\theta-\alpha2(\hat{y}^{(i)}-y^{(i)})x^{(i)}
$$

Therefore, one epoch produces $m$ updates.

In this problem, the samples must be processed sequentially:

```text
Sample 0 → Sample 1 → Sample 2 → ... → Sample m-1
```

No random sampling or shuffling is performed.

#### Characteristics

- Uses one sample per update.
- Produces frequent weight updates.
- Requires very little memory per update.
- The gradient can be noisy.
- Can make rapid progress toward a minimum.
- The noise can help optimization escape some unfavorable regions.

---

### 3. Mini-Batch Gradient Descent

**Mini-Batch Gradient Descent** divides the dataset into smaller batches.

For a batch containing $b$ samples:

$$
\nabla J_B(\theta)=\frac{2}{b}\sum_{i\in B}(\hat{y}^{(i)}-y^{(i)})x^{(i)}
$$

The update becomes

$$
\theta=\theta-\alpha\frac{2}{b}\sum_{i\in B}(\hat{y}^{(i)}-y^{(i)})x^{(i)}
$$

For example, with `batch_size=2`:

```text
Batch 1 → indices [0, 1]
Batch 2 → indices [2, 3]
Batch 3 → indices [4, 5]
...
```

The final batch can contain fewer than `batch_size` samples if the number of samples is not evenly divisible by the batch size.

#### Characteristics

- Uses multiple samples per update.
- Generally less noisy than SGD.
- Requires less computation per update than Batch GD.
- Usually provides a good balance between stability and efficiency.
- Widely used in practical machine learning and deep learning.

---

### Comparison

| Method        | Samples per Update |  Updates per Epoch | Gradient Stability | Memory |
| ------------- | -----------------: | -----------------: | ------------------ | ------ |
| Batch GD      |                $m$ |                $1$ | High               | High   |
| SGD           |                $1$ |                $m$ | Low                | Low    |
| Mini-Batch GD |                $b$ | $\lceil m/b\rceil$ | Medium             | Medium |

The choice depends on the dataset size and optimization requirements.

---

### Why the Data Order Matters

In this problem, the data must **not be shuffled**.

For Batch GD, the order does not affect the gradient because every sample is processed together.

For SGD and Mini-Batch GD, however, the sequence of updates affects the optimization trajectory.

This problem explicitly requires:

```text
0 → 1 → 2 → 3 → ... → m-1
```

for every epoch.

Therefore, no permutation of the training data should be performed.

---

### Mini-Batch With a Partial Final Batch

Suppose there are `5` samples and the batch size is `2`.

The batches are

```text
[0, 1]
[2, 3]
[4]
```

The final batch contains only one sample.

Its gradient must be normalized using the actual batch size:

$$
\nabla J_B(\theta)=\frac{2}{1}\sum_{i\in B}(\hat{y}^{(i)}-y^{(i)})x^{(i)}
$$

rather than assuming that every batch contains exactly `2` samples.

---

### Learning Rate

The **learning rate** $\alpha$ controls the size of each weight update.

The general update rule is

$$
\theta=\theta-\alpha\nabla J(\theta)
$$

A learning rate that is too large can cause the optimization to overshoot the minimum or diverge.

A learning rate that is too small can make convergence very slow.

---

### Characteristics / Key Points

- Gradient descent minimizes a differentiable objective by moving opposite to its gradient.
- MSE is commonly used for regression.
- Batch GD performs one update per epoch.
- SGD performs one update per sample.
- Mini-batch GD performs one update per batch.
- The learning rate controls the update magnitude.
- More frequent updates do not necessarily mean better convergence.
- SGD generally has noisier gradients.
- Batch GD generally has more stable gradients.
- Mini-batch GD provides a practical compromise.
- The final mini-batch may contain fewer samples.
- In this problem, samples must remain in their original order.
- Every method performs the specified number of complete epochs.

---

### Why is it used? / Applications

Gradient descent variants are fundamental to machine learning optimization.

They are used for:

- Linear Regression
- Logistic Regression
- Neural Networks
- Deep Neural Networks
- Large-scale Machine Learning
- Matrix Factorization
- Embedding Optimization
- Recommendation Systems

Mini-batch gradient descent is particularly common in deep learning because modern datasets are often too large to process efficiently as one complete batch.

---

> 💡 **Important Note**
>
> The three methods differ primarily in **how much data is used to estimate the gradient before each update**. The underlying optimization rule remains the same: calculate a gradient and move the weights in the opposite direction.

> 💡 **Interview Tip**
>
> Remember the simplest distinction: **Batch = all samples per update, SGD = one sample per update, Mini-Batch = a subset of samples per update.**

---

## Solution

### Custom Implementation

```python
import numpy as np

class GradientDescent:
    def __init__(self, X, y, weights, learning_rate, n_epochs):
        self.X = X
        self.m = X.shape[0]
        self.y = y.reshape(-1, 1)
        self.weights = np.array(weights, dtype=float).reshape(-1, 1)
        self.learning_rate = learning_rate
        self.epochs = n_epochs

    def batchGD(self):
        for _ in range(self.epochs):
            errors = self.X.dot(self.weights) - self.y
            gradients = (2 / self.m) * self.X.T.dot(errors)
            self.weights -= self.learning_rate * gradients
        return self.weights.flatten()

    def stochasticGD(self):
        for _ in range(self.epochs):
            for i in range(self.m):
                sample = self.X[i].reshape(1, -1)
                target = self.y[i].reshape(1, -1)
                errors = sample.dot(self.weights) - target
                gradients = 2 * sample.T.dot(errors)
                self.weights -= self.learning_rate * gradients
        return self.weights.flatten()

    def miniBatchGD(self, batch_size):
        for _ in range(self.epochs):
            for i in range(0, self.m, batch_size):
                samples = self.X[i:i + batch_size]
                targets = self.y[i:i + batch_size]
                errors = samples.dot(self.weights) - targets
                gradients = (2 / len(samples)) * samples.T.dot(errors)
                self.weights -= self.learning_rate * gradients
        return self.weights.flatten()


def gradient_descent(
    X, y, weights, learning_rate, n_epochs,
    batch_size=1, method="batch"
):
    obj = GradientDescent(X, y, weights, learning_rate, n_epochs)

    if method == "batch":
        return obj.batchGD()
    if method == "stochastic":
        return obj.stochasticGD()
    if method == "mini_batch":
        return obj.miniBatchGD(batch_size)

    raise ValueError(
        "method must be 'batch', 'stochastic', or 'mini_batch'"
    )
```

---

## Code Explanation

### Step 1: Store the Training Data

The constructor stores the feature matrix, targets, initial weights, learning rate, and number of epochs.

The target vector and weights are reshaped into column vectors so that matrix multiplication can be performed consistently.

```python
self.y = y.reshape(-1, 1)
self.weights = np.array(weights, dtype=float).reshape(-1, 1)
```

If $X$ has shape $(m,n)$ and $\theta$ has shape $(n,1)$, the prediction is

$$
\hat{y}=X\theta
$$

and has shape $(m,1)$.

---

### Step 2: Batch Gradient Descent

The Batch GD method first computes predictions for the entire dataset:

```python
errors = self.X.dot(self.weights) - self.y
```

The gradient is then calculated as

```python
gradients = (2 / self.m) * self.X.T.dot(errors)
```

This corresponds to

$$
\nabla J(\theta)=\frac{2}{m}X^T(X\theta-y)
$$

The weights are updated once:

```python
self.weights -= self.learning_rate * gradients
```

Therefore, each epoch performs exactly one update.

---

### Step 3: Stochastic Gradient Descent

SGD processes one sample at a time:

```python
for i in range(self.m):
```

Since `range()` starts at `0` and increases sequentially, the samples are processed in their original order.

For the current sample,

```python
sample = self.X[i].reshape(1, -1)
target = self.y[i].reshape(1, -1)
```

The prediction error is calculated:

```python
errors = sample.dot(self.weights) - target
```

The gradient for one sample is

```python
gradients = 2 * sample.T.dot(errors)
```

The weights are immediately updated.

Therefore, one epoch performs exactly $m$ updates.

---

### Step 4: Mini-Batch Gradient Descent

Mini-batch GD processes consecutive sections of the dataset:

```python
for i in range(0, self.m, batch_size):
```

For example, with `batch_size=2`:

```text
i = 0 → [0, 1]
i = 2 → [2, 3]
i = 4 → [4, 5]
```

The current batch is selected using:

```python
samples = self.X[i:i + batch_size]
targets = self.y[i:i + batch_size]
```

No shuffling is performed.

---

### Step 5: Calculate the Mini-Batch Gradient

The prediction error for the batch is

```python
errors = samples.dot(self.weights) - targets
```

The gradient is

```python
gradients = (2 / len(samples)) * samples.T.dot(errors)
```

The denominator uses `len(samples)` instead of the requested `batch_size`.

This is important because the final batch may contain fewer samples than `batch_size`.

---

### Step 6: Select the Optimization Method

The wrapper function creates a `GradientDescent` object and selects the appropriate method:

```python
if method == "batch":
    return obj.batchGD()
if method == "stochastic":
    return obj.stochasticGD()
if method == "mini_batch":
    return obj.miniBatchGD(batch_size)
```

This allows one function to expose all three optimization strategies.

---

### Step 7: Validate the Method

If the caller provides an unsupported method, the function raises an error:

```python
raise ValueError(
    "method must be 'batch', 'stochastic', or 'mini_batch'"
)
```

This prevents silently running an unintended optimization method.

---

### Update Comparison

The three implementations use the same fundamental rule:

$$
\theta=\theta-\alpha\nabla J(\theta)
$$

They differ only in how the gradient $\nabla J(\theta)$ is estimated.

| Method        | Gradient Uses  |
| ------------- | -------------- |
| Batch GD      | Entire dataset |
| SGD           | One sample     |
| Mini-Batch GD | Current batch  |

This is the central idea behind the three gradient descent variants.

---

## Time & Space Complexity

Let:

- $m$ be the number of samples.
- $n$ be the number of features.
- $b$ be the mini-batch size.
- $E$ be the number of epochs.

### Batch Gradient Descent

Each epoch processes all $m$ samples and computes a matrix-vector gradient.

The time complexity is

$$
O(Emn)
$$

The main stored matrices require

$$
O(mn)
$$

space, while the weight vector requires $O(n)$ space.

---

### Stochastic Gradient Descent

Each epoch performs $m$ individual updates, and each update processes $n$ features.

Therefore,

$$
O(Emn)
$$

time is required.

The additional working memory for each individual sample is

$$
O(n)
$$

apart from the stored dataset.

---

### Mini-Batch Gradient Descent

Each epoch still processes all $m$ samples, although they are divided into batches.

Therefore, the total time complexity remains

$$
O(Emn)
$$

The working memory for a batch is

$$
O(bn)
$$

apart from the stored dataset.

---

| Method        | Time         | Update Memory |
| ------------- | ------------ | ------------- |
| Batch GD      | **$O(Emn)$** | **$O(mn)$**   |
| SGD           | **$O(Emn)$** | **$O(n)$**    |
| Mini-Batch GD | **$O(Emn)$** | **$O(bn)$**   |

Where **$E$** is the number of epochs, **$m$** is the number of samples, **$n$** is the number of features, and **$b$** is the mini-batch size.
