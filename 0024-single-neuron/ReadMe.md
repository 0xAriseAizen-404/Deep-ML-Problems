# Single Neuron (Easy, Deep Learning)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Single Neuron Model](#learn-single-neuron-model)
- [Solutions](#solutions)
  - [Python Implementation](#python-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

---

## Problem Statement

[Single Neuron](https://www.deep-ml.com/problems/24)

Write a Python function that simulates a **single neuron** with a **Sigmoid Activation Function** for binary classification. Return the predicted probabilities and the **Mean Squared Error (MSE)**, both rounded to **4 decimal places**.

---

## Example

```python
Input:

features = [[0.5, 1.0],
            [-1.5, -2.0],
            [2.0, 1.5]]

labels = [0, 1, 0]

weights = [0.7, -0.4]

bias = -0.1

Output:

([0.4626, 0.4134, 0.6682], 0.3349)
```

**Reasoning**

For each sample:

1. Compute the weighted sum.
2. Add the bias.
3. Apply the sigmoid function.
4. Compute the Mean Squared Error (MSE).

---

## Learn: Single Neuron Model

A single neuron computes a weighted sum of the inputs and passes it through an activation function.

**Weighted Sum**

```text
z = Σ(weight × feature) + bias
```

**Sigmoid Activation**

```text
σ(z) = 1 / (1 + e⁻ᶻ)
```

**Mean Squared Error**

```text
MSE = (1 / n) × Σ(predicted - actual)²
```

---

## Solutions

### Python Implementation

```python
import math


def single_neuron_model(
    features: list[list[float]],
    labels: list[int],
    weights: list[float],
    bias: float
) -> tuple[list[float], float]:

    probabilities = []
    mse = 0

    for feature, label in zip(features, labels):

        z = sum(w * x for w, x in zip(weights, feature)) + bias

        prediction = round(1 / (1 + math.exp(-z)), 4)

        probabilities.append(prediction)

        mse += (label - prediction) ** 2

    mse /= len(probabilities)

    return probabilities, round(mse, 4)


# Example Usage
features = [[0.5, 1.0],
            [-1.5, -2.0],
            [2.0, 1.5]]

labels = [0, 1, 0]

weights = [0.7, -0.4]

bias = -0.1

print(single_neuron_model(features, labels, weights, bias))
```

---

## Code Explanation

- Compute the weighted sum for each input.

```python
z = sum(w * x for w, x in zip(weights, feature)) + bias
```

- Apply the sigmoid activation.

```python
1 / (1 + math.exp(-z))
```

- Compute the Mean Squared Error.

```python
mse += (label - prediction) ** 2
```

- Return the predicted probabilities and the average MSE.

---

## Time & Space Complexity

| Complexity | Value |
|------------|-------|
| Time | **O(n × d)** |
| Space | **O(n)** |

Where:

- **n** = Number of samples
- **d** = Number of features