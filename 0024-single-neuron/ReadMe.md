# Single Neuron (Easy, Deep Learning)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Single Neuron Model](#learn-single-neuron-model)
- [Solutions](#solutions)
  - [Custom Implementation](#custom-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

## Problem Statement

### [Single Neuron](https://www.deep-ml.com/problems/24)

Write a Python function that simulates a **single artificial neuron** for binary classification.

The function receives:

- A list of feature vectors.
- Their corresponding binary labels.
- One weight for each feature.
- A bias term.

For every input sample, compute the weighted sum, apply the **Sigmoid Activation Function**, and return:

1. The predicted probabilities.
2. The Mean Squared Error (MSE) between the predicted probabilities and the true labels.

Round both the probabilities and the final MSE to **4 decimal places**.

---

## Example

### Input

```python
features = [
    [0.5, 1.0],
    [-1.5, -2.0],
    [2.0, 1.5]
]

labels = [0, 1, 0]

weights = [0.7, -0.4]

bias = -0.1
```

### Output

```python
(
    [0.4626, 0.4134, 0.6682],
    0.3349
)
```

### Reasoning

For the first sample,

$$
z=(0.7\times0.5)+(-0.4\times1.0)-0.1=-0.15
$$

Applying the sigmoid function,

$$
\sigma(z)=\frac{1}{1+e^{0.15}}\approx0.4626
$$

The same computation is repeated for every sample.

Finally, the Mean Squared Error is computed as

$$
\text{MSE}=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

---

## Learn: Single Neuron Model

### What is a Single Neuron?

A **Single Artificial Neuron** is the basic building block of a neural network.

It receives several input features, multiplies each feature by its corresponding weight, adds a bias, and passes the result through an activation function.

For binary classification, the activation function is usually the **Sigmoid Function**, producing a probability between 0 and 1.

This model is essentially the same as **Logistic Regression**, viewed from a neural network perspective.

---

### Structure of a Neuron

A neuron consists of:

- Input features
- Weights
- Bias
- Weighted sum
- Activation function
- Output probability

The computation follows the sequence

```text
Inputs
   │
Weights
   │
Weighted Sum
   │
 + Bias
   │
Sigmoid
   │
Probability
```

---

### Weighted Sum

Suppose an input contains

$$
x=(x_1,x_2,\ldots,x_n)
$$

and corresponding weights

$$
w=(w_1,w_2,\ldots,w_n)
$$

The neuron first computes

$$
z=\sum_{i=1}^{n}w_ix_i+b
$$

where

- $w_i$ is the weight of feature $i$.
- $x_i$ is the input feature.
- $b$ is the bias.

This value is often called the **logit** or **pre-activation value**.

---

### Sigmoid Activation

The weighted sum is converted into a probability using

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

The output satisfies

$$
0<\sigma(z)<1
$$

Larger values of $z$ produce probabilities closer to 1, while smaller values produce probabilities closer to 0.

---

### Mean Squared Error (MSE)

The prediction error is measured using

$$
\text{MSE}=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

where

- $y_i$ is the true label.
- $\hat{y}_i$ is the predicted probability.
- $n$ is the number of samples.

A smaller MSE indicates better predictions.

---

### Characteristics / Key Points

- Smallest possible neural network.
- Handles multiple input features.
- Produces probabilities for binary classification.
- Uses learnable weights and bias.
- Fully differentiable, allowing gradient-based optimization.
- Acts as the foundation for deeper neural networks.
- Equivalent to Logistic Regression when trained for binary classification.

---

### Why is it Used?

Single neurons are used for:

- Binary classification
- Logistic Regression
- Spam detection
- Fraud detection
- Medical diagnosis
- Credit approval
- Understanding neural network fundamentals
- Building blocks of deep neural networks

Although modern networks contain millions of neurons, every neuron performs this same basic computation.

---

> 💡 **Important Note**
>
> While this problem evaluates predictions using **Mean Squared Error (MSE)**, real-world binary classification models almost always use **Binary Cross-Entropy (BCE) Loss** because it provides stronger gradients and faster learning. MSE is mainly used here for simplicity and educational purposes.

---

## Solutions

### Custom Implementation

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

        z = sum(
            w * x
            for w, x in zip(weights, feature)
        ) + bias

        probability = round(
            1 / (1 + math.exp(-z)),
            4
        )

        probabilities.append(probability)

        mse = (
            (
                mse * (len(probabilities) - 1)
            )
            + (label - probability) ** 2
        ) / len(probabilities)

    return probabilities, round(mse, 4)
```

---

## Code Explanation

### 1. Process Each Training Sample

```python
for feature, label in zip(features, labels):
```

Each feature vector is processed together with its corresponding target label.

---

### 2. Compute the Weighted Sum

```python
z = sum(
    w * x
    for w, x in zip(weights, feature)
) + bias
```

This implements

$$
z=\sum_{i=1}^{n}w_ix_i+b
$$

Each feature contributes according to its weight.

---

### 3. Apply the Sigmoid Function

```python
1 / (1 + math.exp(-z))
```

This computes

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

The result represents the probability that the sample belongs to the positive class.

---

### 4. Store the Prediction

```python
probabilities.append(probability)
```

The predicted probability is added to the output list.

---

### 5. Compute the Running Mean Squared Error

```python
(label - probability) ** 2
```

The squared prediction error is computed for the current sample.

The running average updates the Mean Squared Error without storing every individual error.

This follows

$$
\text{MSE}=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

---

### 6. Return the Final Results

The function returns

- The list of predicted probabilities.
- The final Mean Squared Error rounded to four decimal places.

---

## Time & Space Complexity

Let

- $m$ = number of training samples.
- $n$ = number of input features.

Each sample computes one weighted sum across all features.

| Complexity | Value        |
| ---------- | ------------ |
| Time       | **O(m × n)** |
| Space      | **O(m)**     |

The additional space is required to store the predicted probabilities.
