# Single Neuron with Backpropagation (Medium, Deep Learning)

## Table of Contents

- Problem Statement
- Example
- Learn: Neural Network Learning with Backpropagation
- Solution
- Code Explanation
- Time & Space Complexity

## Problem Statement

### [Single Neuron with Backpropagation](https://www.deep-ml.com/problems/25)

Write a Python function that trains a single neuron using the **sigmoid activation function** and **batch gradient descent with backpropagation**.

The function receives:

- Feature matrix
- Binary labels
- Initial weights
- Initial bias
- Learning rate
- Number of epochs

For every epoch:

1. Perform a forward pass.
2. Compute predictions.
3. Compute the Mean Squared Error (MSE).
4. Calculate gradients using backpropagation.
5. Update weights and bias once using the entire batch.

Return:

- Updated weights
- Updated bias
- List of MSE values before each parameter update

Round all returned values to **4 decimal places**.

---

## Example

**Input**

```python
features = [[1.0, 2.0],
            [2.0, 1.0],
            [-1.0, -2.0]]

labels = [1, 0, 0]

initial_weights = [0.1, -0.2]
initial_bias = 0.0

learning_rate = 0.1
epochs = 2
```

**Output**

```python
updated_weights = [0.1036, -0.1425]
updated_bias = -0.0167
mse_values = [0.3033, 0.2942]
```

**Reasoning**

The neuron computes probabilities using the sigmoid activation function. The prediction error is measured using Mean Squared Error (MSE). Backpropagation computes the gradients of the loss with respect to each parameter, and batch gradient descent updates the weights and bias once per epoch.

---

## Learn: Neural Network Learning with Backpropagation

### What is Backpropagation?

Backpropagation is the learning algorithm used to train neural networks. It computes how much each parameter (weights and bias) contributes to the prediction error and adjusts them to reduce the loss.

Training follows four main steps:

1. Forward Pass
2. Loss Calculation
3. Backward Pass (Gradient Computation)
4. Parameter Update

The objective is to minimize the loss over many iterations.

---

### Forward Pass

For each training sample, compute the weighted sum:

$$
z = w_1x_1 + w_2x_2 + \cdots + w_nx_n + b
$$

or in vector notation,

$$
z = \mathbf{w}^T\mathbf{x} + b
$$

The sigmoid activation converts this value into a probability.

$$
\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}
$$

The output always lies between **0 and 1**, making it suitable for binary classification.

---

### Mean Squared Error (MSE)

The prediction error is measured using Mean Squared Error.

$$
MSE = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i-y_i)^2
$$

where

- $n$ is the number of training samples
- $\hat{y}_i$ is the predicted probability
- $y_i$ is the true label

Lower MSE indicates better predictions.

---

### Backward Pass

Backpropagation applies the **Chain Rule** from calculus to compute gradients.

Since

$$
MSE \rightarrow \hat{y} \rightarrow z \rightarrow w,b
$$

the derivatives are computed step-by-step.

Derivative of MSE with respect to predictions:

$$
\frac{\partial MSE}{\partial \hat{y}} = \frac{2}{n}(\hat{y}-y)
$$

Derivative of sigmoid:

$$
\frac{\partial \hat{y}}{\partial z} = \hat{y}(1-\hat{y})
$$

Gradient of each weight:

$$
\frac{\partial MSE}{\partial w_j} = \frac{2}{n}\sum_{i=1}^{n}(\hat{y}_i-y_i)\hat{y}_i(1-\hat{y}_i)x_{ij}
$$

Gradient of the bias:

$$
\frac{\partial MSE}{\partial b} = \frac{2}{n}\sum_{i=1}^{n}(\hat{y}_i-y_i)\hat{y}_i(1-\hat{y}_i)
$$

These gradients indicate the direction that increases the loss the most.

---

### Parameter Update

Gradient Descent updates the parameters in the opposite direction of the gradient.

Weights:

$$
w = w-\alpha\frac{\partial MSE}{\partial w}
$$

Bias:

$$
b = b-\alpha\frac{\partial MSE}{\partial b}
$$

where

- $\alpha$ is the learning rate.

A larger learning rate results in larger updates, while a smaller learning rate produces slower but more stable learning.

---

### Batch Gradient Descent

This problem uses **Batch Gradient Descent**.

Instead of updating after every sample, the algorithm:

- Performs one forward pass over the entire dataset.
- Computes gradients using every training example.
- Updates parameters once.

This produces smoother optimization than stochastic updates.

---

### Characteristics / Key Points

- Uses sigmoid activation for binary classification.
- MSE measures prediction error.
- Gradients are computed using the Chain Rule.
- Batch Gradient Descent updates parameters once per epoch.
- All weights are initialized to zero or provided values.
- Parameters gradually converge toward values that minimize loss.
- Learning rate controls the update magnitude.
- Sigmoid derivatives become very small for extremely positive or negative inputs, causing slow learning (vanishing gradients).

---

### Why is it used? / Applications

Backpropagation is the foundation of nearly every deep learning model.

Applications include:

- Binary classification
- Spam detection
- Disease prediction
- Credit risk analysis
- Logistic Regression training
- Deep Neural Networks
- Convolutional Neural Networks (CNNs)
- Recurrent Neural Networks (RNNs)

Without backpropagation, neural networks cannot learn from data.

---

> 💡 **Important Note**
>
> Although this problem uses **Mean Squared Error (MSE)**, binary classification models typically use **Binary Cross-Entropy (BCE)** because it provides stronger gradients and generally converges faster. MSE is mainly used here to simplify the mathematics of backpropagation.

---

## Solution

### Custom Implementation

```python
import numpy as np

def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
    weights = initial_weights[:].reshape(-1, 1)
    bias = initial_bias
    mse_values = []
    labels = labels.reshape(-1, 1)

    for _ in range(epochs):
        # Forward Pass
        z = features.dot(weights) + bias
        probs = 1 / (1 + np.exp(-z))

        # Loss
        mse = np.mean((probs - labels) ** 2)
        mse_values.append(mse)

        # Backpropagation
        d_probs = (2 / len(probs)) * (probs - labels)
        d_z = d_probs * (probs * (1 - probs))

        d_weights = features.T.dot(d_z)
        d_bias = np.sum(d_z)

        # Gradient Descent Update
        weights = weights - learning_rate * d_weights
        bias = bias - learning_rate * d_bias

    return (
        np.round(weights.ravel(), 4),
        round(float(bias), 4),
        [round(float(x), 4) for x in mse_values]
    )
```

---

## Code Explanation

### Step 1

Initialize the model parameters.

- Convert weights into a column vector.
- Store the bias.
- Create a list to record MSE values.
- Convert labels into a column vector.

---

### Step 2

Perform the forward pass.

Compute the linear combination:

$$
z = Xw + b
$$

Apply the sigmoid activation:

$$
\hat{y} = \frac{1}{1+e^{-z}}
$$

This produces predicted probabilities.

---

### Step 3

Compute the loss.

Calculate the Mean Squared Error between predictions and labels.

$$
MSE = \frac{1}{n}\sum(\hat{y}-y)^2
$$

Store the loss before updating the parameters.

---

### Step 4

Compute gradients using backpropagation.

First compute

$$
\frac{\partial MSE}{\partial \hat{y}}
$$

Then multiply by the sigmoid derivative

$$
\hat{y}(1-\hat{y})
$$

Finally compute

$$
\frac{\partial MSE}{\partial w}
$$

and

$$
\frac{\partial MSE}{\partial b}
$$

using matrix multiplication.

---

### Step 5

Update the parameters using Gradient Descent.

$$
w = w-\alpha\nabla_w
$$

$$
b = b-\alpha\nabla_b
$$

Repeat this process for the specified number of epochs.

---

### Step 6

Return

- Final weights
- Final bias
- MSE history

rounded to four decimal places.

---

## Time & Space Complexity

| Complexity | Value                 |
| ---------- | --------------------- |
| Time       | **O(epochs × m × n)** |
| Space      | **O(m + n)**          |

where

- **m** is the number of training samples.
- **n** is the number of input features.
- **epochs** is the number of gradient descent iterations.
