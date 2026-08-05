# Softmax Activation Function Implementation (Easy, Deep Learning)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Softmax Activation Function](#learn-softmax-activation-function)
- [Solutions](#solutions)
  - [Custom Implementation](#custom-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

## Problem Statement

### [Softmax Activation Function Implementation](https://www.deep-ml.com/problems/23)

Write a Python function that computes the **Softmax Activation Function** for a given list of scores.

The implementation should be **numerically stable**, meaning it should prevent overflow when computing exponentials of large values. Return the resulting probability distribution as a list of floating-point numbers.

---

## Example

### Input

```python
scores = [1, 2, 3]
```

### Output

```python
[0.09, 0.2447, 0.6652]
```

### Reasoning

The softmax function converts the input scores into probabilities.

First, subtract the maximum score for numerical stability:

```text
[-2, -1, 0]
```

Compute exponentials

```text
[e⁻², e⁻¹, e⁰]
=
[0.1353, 0.3679, 1]
```

Normalize by their sum

$$
\frac{[0.1353,\;0.3679,\;1]}{0.1353+0.3679+1} =
[0.0900,\;0.2447,\;0.6652]
$$

The probabilities sum to

$$
1
$$

---

## Learn: Softmax Activation Function

### What is the Softmax Function?

The **Softmax Activation Function** converts a vector of arbitrary real-valued scores (called **logits**) into a **probability distribution**.

Unlike the sigmoid function, which produces a single probability, softmax produces probabilities for **multiple classes**.

Each output represents the probability that the input belongs to one particular class.

---

### Mathematical Definition

For an input vector

$$
z=(z_1,z_2,\ldots,z_n)
$$

the softmax output for the $i^{th}$ element is

$$
\text{Softmax}(z_i)=\frac{e^{z_i}}{\sum_{j=1}^{n}e^{z_j}}
$$

where

- $z_i$ is the score (logit) for class $i$.
- $e$ is Euler's constant.

---

### Output Properties

Every softmax output satisfies

$$
0<P_i<1
$$

and all probabilities add up to

$$
\sum_{i=1}^{n}P_i=1
$$

This makes softmax ideal for multi-class classification problems.

---

### Numerical Stability

Directly computing

$$
e^{z_i}
$$

can cause overflow when the input scores are very large.

To avoid this, subtract the largest score before exponentiation.

The stable softmax equation is

$$
\text{Softmax}(z_i)=\frac{e^{z_i-\max(z)}}{\sum_{j=1}^{n}e^{z_j-\max(z)}}
$$

Subtracting the maximum shifts every value by the same constant, leaving the final probabilities unchanged while preventing numerical overflow.

---

### Why Does This Work?

Suppose

```text
scores = [1000, 1001, 1002]
```

Computing

$$
e^{1002}
$$

would overflow in most programming languages.

Instead,

```text
scores = [-2, -1, 0]
```

after subtracting the maximum value.

Now

$$
e^0=1
$$

and

$$
e^{-2}=0.1353
$$

These values are safe to compute and produce exactly the same probability distribution.

---

### Softmax vs Sigmoid

| Softmax                                     | Sigmoid                                |
| ------------------------------------------- | -------------------------------------- |
| Used for multiple classes                   | Used for binary classification         |
| Produces a probability distribution         | Produces one independent probability   |
| Probabilities sum to 1                      | Outputs are independent                |
| Output layer for multi-class classification | Output layer for binary classification |

---

### Characteristics / Key Points

- Produces probabilities between 0 and 1.
- Output probabilities always sum to 1.
- Converts logits into a probability distribution.
- Sensitive to differences between input scores.
- Larger scores receive exponentially larger probabilities.
- Requires numerical stabilization for practical implementations.
- Differentiable, making it suitable for gradient-based optimization.

---

### Why is it Used?

Softmax is widely used in Deep Learning for:

- Multi-class image classification
- Handwritten digit recognition
- Object recognition
- Language modeling
- Machine translation
- Text classification
- Speech recognition
- Large Language Models (LLMs)

It is typically paired with the **Cross-Entropy Loss** function during training.

---

> 💡 **Important Note**
>
> Softmax should generally be used **only in the output layer** of a multi-class classifier. During training, many deep learning libraries (such as PyTorch's `CrossEntropyLoss`) expect **raw logits**, not softmax probabilities, because they internally combine softmax with cross-entropy in a numerically stable way.

---

## Solutions

### Custom Implementation

```python
import math

def softmax(scores: list[float]) -> list[float]:

    mx = max(scores)

    scores = [x - mx for x in scores]

    exp_scores = [
        math.exp(x)
        for x in scores
    ]

    total = sum(exp_scores)

    return [
        round(x / total, 4)
        for x in exp_scores
    ]
```

---

## Code Explanation

### 1. Find the Maximum Score

```python
mx = max(scores)
```

The maximum value is used to improve numerical stability before exponentiation.

---

### 2. Shift Every Score

```python
scores = [x - mx for x in scores]
```

This implements

$$
z_i-\max(z)
$$

The relative differences between scores remain unchanged, so the final probabilities are identical to the original softmax.

---

### 3. Compute the Exponentials

```python
exp_scores = [
    math.exp(x)
    for x in scores
]
```

Each shifted score is transformed into

$$
e^{z_i-\max(z)}
$$

These values are always positive.

---

### 4. Compute the Normalization Constant

```python
total = sum(exp_scores)
```

The denominator of the softmax equation is

$$
\sum_{j=1}^{n}e^{z_j-\max(z)}
$$

---

### 5. Normalize the Values

```python
x / total
```

Each exponential value is divided by the total sum.

This implements

$$
\text{Softmax}(z_i)=\frac{e^{z_i-\max(z)}}{\sum_{j=1}^{n}e^{z_j-\max(z)}}
$$

ensuring the outputs form a valid probability distribution.

---

### 6. Round the Output

```python
round(x / total, 4)
```

The probabilities are rounded to four decimal places before being returned.

---

## Time & Space Complexity

Let

- $n$ = number of input scores.

The algorithm performs three linear passes over the input:

- Find the maximum.
- Compute exponentials.
- Normalize the probabilities.

| Complexity | Value    |
| ---------- | -------- |
| Time       | **O(n)** |
| Space      | **O(n)** |

Additional space is required to store the exponentiated values before normalization.
