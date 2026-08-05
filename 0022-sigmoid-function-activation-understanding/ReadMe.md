# Sigmoid Activation Function Understanding (Easy, Deep Learning)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Sigmoid Activation Function](#learn-sigmoid-activation-function)
- [Solutions](#solutions)
  - [Custom Implementation](#custom-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

## Problem Statement

### [Sigmoid Activation Function Understanding](https://www.deep-ml.com/problems/22)

Write a Python function that computes the output of the **Sigmoid Activation Function** for a given input value.

The function should return the output rounded to **4 decimal places**.

The sigmoid function maps any real-valued input to a value between **0 and 1**, making it ideal for representing probabilities in binary classification problems.

---

## Example

### Input

```python
z = 0
```

### Output

```python
0.5
```

### Reasoning

Using the sigmoid function,

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

Substituting

$$
z=0
$$

gives

$$
\sigma(0)=\frac{1}{1+e^0}=\frac{1}{2}=0.5
$$

---

## Learn: Sigmoid Activation Function

### What is the Sigmoid Function?

The **Sigmoid Activation Function** is one of the earliest activation functions used in Artificial Neural Networks.

It transforms any real-valued input into a number between **0 and 1**, allowing the output to be interpreted as a probability.

Because of this property, sigmoid is commonly used in the **output layer of binary classification models**.

---

### Mathematical Definition

The sigmoid function is defined as

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

where

- $z$ is the input (also called the **logit**).
- $e$ is Euler's constant ($e \approx 2.71828$).

---

### Sigmoid Curve

The sigmoid function produces an **S-shaped (logistic) curve**.

Some common values are

| Input ($z$) | Output ($\sigma(z)$) |
| ----------: | -------------------: |
|          -5 |               0.0067 |
|          -2 |               0.1192 |
|          -1 |               0.2689 |
|           0 |               0.5000 |
|           1 |               0.7311 |
|           2 |               0.8808 |
|           5 |               0.9933 |

As the input increases,

$$
\sigma(z)\rightarrow1
$$

As the input decreases,

$$
\sigma(z)\rightarrow0
$$

---

### Properties of the Sigmoid Function

#### Output Range

The sigmoid output always lies within

$$
0<\sigma(z)<1
$$

The output never becomes exactly 0 or exactly 1.

---

#### Midpoint

At

$$
z=0
$$

the output is

$$
\sigma(0)=0.5
$$

This is the center of the sigmoid curve.

---

#### Monotonic Function

As the input increases, the output also increases.

Therefore, sigmoid is a monotonically increasing function.

---

### Derivative of Sigmoid

One reason sigmoid became popular is its elegant derivative.

If

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

then

$$
\frac{d\sigma(z)}{dz}=\sigma(z)(1-\sigma(z))
$$

Notice that the derivative depends only on the sigmoid output itself.

This makes gradient computation efficient during backpropagation.

---

### Vanishing Gradient Problem

For very large positive or negative values,

$$
\sigma(z)\approx1
$$

or

$$
\sigma(z)\approx0
$$

In these regions,

$$
\frac{d\sigma(z)}{dz}\approx0
$$

As gradients become extremely small, weight updates nearly stop.

This phenomenon is known as the **Vanishing Gradient Problem**, one of the main reasons sigmoid is rarely used in hidden layers of modern deep neural networks.

---

### Characteristics / Key Points

- Smooth and differentiable everywhere.
- Produces outputs between 0 and 1.
- Monotonically increasing.
- Non-linear activation function.
- Converts logits into probabilities.
- Suffers from vanishing gradients for very large positive or negative inputs.
- Computationally more expensive than ReLU due to the exponential operation.

---

### Why is it Used?

The sigmoid activation function is commonly used in:

- Binary classification
- Logistic Regression
- Output layer of binary neural networks
- Probability estimation
- Medical diagnosis models
- Spam detection
- Fraud detection
- Binary image segmentation

Although modern hidden layers often use **ReLU**, sigmoid remains the standard activation for binary output neurons.

---

> 💡 **Important Note**
>
> A common mistake is using sigmoid in every hidden layer of a deep neural network. Because sigmoid saturates for large positive or negative inputs, gradients become very small, slowing or even stopping learning. Modern architectures typically use **ReLU** (or its variants) for hidden layers and reserve **Sigmoid** mainly for the **output layer of binary classification models**.

---

## Solutions

### Custom Implementation

```python
import math

def sigmoid(z: float) -> float:
    return round(
        1 / (1 + math.exp(-z)),
        4
    )
```

---

## Code Explanation

### 1. Compute the Exponential

```python
math.exp(-z)
```

This computes

$$
e^{-z}
$$

which determines how quickly the sigmoid curve approaches 0 or 1.

---

### 2. Add One to the Denominator

```python
1 + math.exp(-z)
```

This forms the denominator of the sigmoid equation,

$$
1+e^{-z}
$$

---

### 3. Compute the Sigmoid Value

```python
1 / (1 + math.exp(-z))
```

This directly implements

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

The resulting value always lies strictly between 0 and 1.

---

### 4. Round the Result

```python
round(..., 4)
```

The final output is rounded to four decimal places before being returned.

---

## Time & Space Complexity

The function performs a constant number of arithmetic operations regardless of the input value.

| Complexity | Value    |
| ---------- | -------- |
| Time       | **O(1)** |
| Space      | **O(1)** |
