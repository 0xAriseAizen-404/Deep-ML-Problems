# Single Neuron with Backpropagation (Medium, Deep Learning)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Backpropagation](#learn-backpropagation)
- [Solutions](#solutions)
  - [NumPy Implementation](#numpy-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

---

## Problem Statement

[Single Neuron with Backpropagation](https://www.deep-ml.com/problems/25)

Write a Python function that trains a **single neuron** using **batch backpropagation** with a **sigmoid activation**. Return the updated weights, bias, and the MSE before each parameter update.

---

## Example

```python
Input:

features = [[1.0, 2.0],
            [2.0, 1.0],
            [-1.0, -2.0]]

labels = [1, 0, 0]

initial_weights = [0.1, -0.2]
initial_bias = 0.0

learning_rate = 0.1
epochs = 2

Output:

updated_weights = [0.1036, -0.1425]
updated_bias = -0.0167

mse_values = [0.3033, 0.2942]
```

**Reasoning**

The neuron performs a forward pass, computes the MSE, calculates gradients using backpropagation, and updates the weights and bias after each epoch.

---

## Learn: Backpropagation

Backpropagation updates model parameters by computing the gradient of the loss.

**Forward Pass**

```text
z = XW + b
```

```text
ŷ = sigmoid(z)
```

**Loss**

```text
MSE = (1/n) × Σ(ŷ - y)²
```

**Update Rule**

```text
W = W - α × ∂Loss/∂W

b = b - α × ∂Loss/∂b
```

---

## Solutions

### NumPy Implementation

```python
import numpy as np


def train_neuron(
    features: np.ndarray,
    labels: np.ndarray,
    initial_weights: np.ndarray,
    initial_bias: float,
    learning_rate: float,
    epochs: int
) -> tuple[np.ndarray, float, list[float]]:

    weights = initial_weights.reshape(-1, 1)
    bias = initial_bias
    labels = labels.reshape(-1, 1)

    mse_values = []

    for _ in range(epochs):

        # Forward Pass
        z = features @ weights + bias
        predictions = 1 / (1 + np.exp(-z))

        # Loss
        mse = np.mean((predictions - labels) ** 2)
        mse_values.append(round(float(mse), 4))

        # Backpropagation
        d_predictions = (2 / len(labels)) * (predictions - labels)
        d_z = d_predictions * predictions * (1 - predictions)

        d_weights = features.T @ d_z
        d_bias = np.sum(d_z)

        # Update Parameters
        weights -= learning_rate * d_weights
        bias -= learning_rate * d_bias

    return (
        np.round(weights.ravel(), 4),
        round(float(bias), 4),
        mse_values
    )


# Example Usage
weights, bias, mse = train_neuron(
    features,
    labels,
    initial_weights,
    initial_bias,
    0.1,
    2
)

print(weights)
print(bias)
print(mse)
```

---

## Code Explanation

- Compute the weighted sum and apply the sigmoid activation.

```python
predictions = 1 / (1 + np.exp(-z))
```

- Compute the Mean Squared Error (MSE).

```python
mse = np.mean((predictions - labels) ** 2)
```

- Compute gradients using backpropagation.

```python
d_weights = features.T @ d_z
d_bias = np.sum(d_z)
```

- Update the weights and bias.

```python
weights -= learning_rate * d_weights
bias -= learning_rate * d_bias
```

---

## Time & Space Complexity

| Complexity | Value |
|------------|-------|
| Time | **O(epochs × n × d)** |
| Space | **O(n + d)** |

Where:

- **n** = Number of training samples
- **d** = Number of features