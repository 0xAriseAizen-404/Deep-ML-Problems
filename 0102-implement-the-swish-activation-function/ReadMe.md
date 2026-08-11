# Implement the Swish Activation Function (Easy, Deep Learning)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Understanding the Swish Activation Function](#learn-understanding-the-swish-activation-function)
- [Solution](#solution)
  - [Custom Implementation](#custom-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

---

## Problem Statement

### [Implement the Swish Activation Function](https://www.deep-ml.com/problems/102)

Implement the **Swish activation function**, a smooth, non-monotonic, self-gated activation function used in neural networks.

The function should take a numerical input $x$ and return the Swish activation value rounded to **4 decimal places**.

Swish is defined as the product of the input and its sigmoid value.

---

## Example

### Input

```python
swish(1)
````

### Output

```text
0.7311
```

### Reasoning

The Swish function is

$$
{Swish}(x)=x\sigma(x)
$$

where the sigmoid function is

$$
\sigma(x)=\frac{1}{1+e^{-x}}
$$

For $x=1$:

$$
\sigma(1)=\frac{1}{1+e^{-1}}\approx0.7311
$$

Therefore,

$$
{Swish}(1)=1\times0.7311=0.7311
$$

---

## Learn: Understanding the Swish Activation Function

### What is it?

**Swish** is a smooth, non-monotonic, self-gated activation function used in neural networks.

It was introduced by researchers at **Google Brain** and was proposed as an alternative to traditional activation functions such as ReLU.

The defining idea behind Swish is that the input controls its own gating through the sigmoid function.

The function is defined as

$$
{Swish}(x)=x\sigma(x)
$$

where $\sigma(x)$ is the sigmoid function.

This gives Swish a smooth transition between negative and positive values.

---

### Mathematical Definition

The Swish activation function is

$$
f(x)=x\sigma(x)
$$

The sigmoid function is

$$
\sigma(x)=\frac{1}{1+e^{-x}}
$$

Therefore, Swish can also be written directly as

$$
f(x)=\frac{x}{1+e^{-x}}
$$

The sigmoid term acts as a gate that determines how strongly the input should contribute to the output.

---

### How Self-Gating Works

Unlike ReLU, which applies a fixed threshold,

$$
{ReLU}(x)=\max(0,x)
$$

Swish multiplies the input by a value between `0` and `1`:

$$
0<\sigma(x)<1
$$

Therefore,

$$
{Swish}(x)=x\sigma(x)
$$

For positive values, the sigmoid approaches `1`, allowing the input to pass through almost unchanged.

For negative values, the sigmoid approaches `0`, suppressing the magnitude of the input.

This input-dependent behavior is known as **self-gating**.

---

### Behavior for Large Positive Inputs

When $x$ becomes very large,

$$
x\rightarrow\infty
$$

the sigmoid approaches `1`:

$$
\sigma(x)\rightarrow1
$$

Therefore,

$$
{Swish}(x)\rightarrow x
$$

So for large positive values, Swish behaves approximately like the identity function.

For example:

$$
{Swish}(10)\approx10
$$

This allows positive signals to pass through with very little attenuation.

---

### Behavior for Large Negative Inputs

When $x$ becomes very negative,

$$
x\rightarrow-\infty
$$

the sigmoid approaches `0`:

$$
\sigma(x)\rightarrow0
$$

Therefore,

$$
{Swish}(x)\rightarrow0
$$

However, Swish approaches zero from the negative side.

For example, for a sufficiently negative input:

$$
x\sigma(x)<0
$$

This differs from ReLU, which sets every negative input exactly to zero.

---

### Non-Monotonic Behavior

One of the important characteristics of Swish is that it is **non-monotonic**.

For some negative values, the function decreases below zero before increasing toward positive values.

The function reaches its minimum at approximately

$$
x\approx-1.28
$$

This creates a small negative region before the function rises toward its approximately linear positive region.

The slight negative output can allow information to flow through the network instead of completely removing negative activations.

---

### Smoothness

Swish is smooth and differentiable everywhere.

Unlike ReLU, it does not contain a sharp corner at $x=0$.

ReLU is

$$
{ReLU}(x)=\max(0,x)
$$

while Swish is

$$
{Swish}(x)=\frac{x}{1+e^{-x}}
$$

The sigmoid component makes the transition between negative and positive regions gradual.

This smoothness can make optimization behavior different from piecewise-linear activations.

---

### Derivative

Starting with

$$
f(x)=x\sigma(x)
$$

using the product rule:

$$
f'(x)=\sigma(x)+x\sigma'(x)
$$

The derivative of sigmoid is

$$
\sigma'(x)=\sigma(x)(1-\sigma(x))
$$

Therefore,

$$
f'(x)=\sigma(x)+x\sigma(x)(1-\sigma(x))
$$

Factoring sigmoid gives

$$
f'(x)=\sigma(x)(1+x(1-\sigma(x)))
$$

This derivative is smooth for all real values of $x$.

---

### Important Values

At $x=0$:

$$
{Swish}(0)=0\times\sigma(0)=0
$$

At $x=1$:

$$
{Swish}(1)\approx0.7311
$$

At $x=-1$:

$$
{Swish}(-1)\approx-0.2689
$$

This demonstrates that Swish preserves a small negative value instead of setting it to zero.

---

### Swish vs ReLU

| Property                  | ReLU        | Swish             |
| ------------------------- | ----------- | ----------------- |
| Formula                   | $\max(0,x)$ | $x\sigma(x)$      |
| Smooth                    | No          | Yes               |
| Differentiable everywhere | No          | Yes               |
| Monotonic                 | Yes         | No                |
| Negative outputs          | No          | Yes               |
| Positive-region behavior  | $x$         | Approximately $x$ |
| Self-gated                | No          | Yes               |
| Bounded                   | No          | No                |

Swish and ReLU both behave approximately linearly for large positive inputs, but Swish provides a smooth and non-monotonic transition around zero.

---

### Swish vs Sigmoid

Swish and sigmoid are closely related because Swish uses sigmoid as its gating function.

| Property      | Sigmoid              | Swish                |
| ------------- | -------------------- | -------------------- |
| Formula       | $\frac{1}{1+e^{-x}}$ | $\frac{x}{1+e^{-x}}$ |
| Output Range  | $(0,1)$              | $(-\infty,\infty)$   |
| Zero-Centered | No                   | Yes                  |
| Monotonic     | Yes                  | No                   |
| Smooth        | Yes                  | Yes                  |
| Self-Gated    | No                   | Yes                  |

Swish avoids restricting the output to the range `(0, 1)` because it multiplies sigmoid by the original input.

---

### Characteristics / Key Points

* Swish is a smooth activation function.
* It is defined as $x\sigma(x)$.
* It uses sigmoid as a self-gating mechanism.
* It is differentiable everywhere.
* It is non-monotonic.
* It can produce negative outputs.
* It has an unbounded positive range.
* It approaches $x$ for large positive inputs.
* It approaches zero for large negative inputs.
* It has a small negative dip around $x\approx-1.28$.
* It does not have the hard zero threshold of ReLU.
* The sigmoid gate controls how much of the input passes through.
* It can provide useful optimization behavior in deep networks.

---

### Output Range

Unlike sigmoid or tanh, Swish is not bounded.

For large positive $x$:

$$
{Swish}(x)\rightarrow\infty
$$

For large negative $x$:

$$
{Swish}(x)\rightarrow0^-
$$

Therefore, the overall range is

$$
(-\infty,\infty)
$$

although the negative part has a finite minimum near $x=-1.28$.

---

### Why is it used? / Applications

Swish can be used in:

* Deep neural networks.
* Computer vision models.
* Image classification.
* Large-scale deep learning architectures.
* Models where smooth nonlinearities are desirable.
* Architectures where non-monotonic activation behavior can improve optimization.

Swish has been shown to perform well in several deep neural network architectures and is particularly interesting as an alternative to ReLU.

---

> 💡 **Important Note**
>
> Swish is not simply a smoothed version of ReLU. Its key property is **self-gating**: the input is multiplied by its own sigmoid value, ${Swish}(x)=x\sigma(x)$. It is also non-monotonic and can produce small negative outputs.

> 💡 **Interview Tip**
>
> Remember the three important facts: **Swish = $x\times$ sigmoid$(x)$, it is smooth, and it is non-monotonic**. For large positive $x$, Swish behaves approximately like $x$; for large negative $x$, it approaches zero from below.

---

## Solution

### Custom Implementation

```python
import math

def swish(x: float) -> float:
    return round(x * (1 / (1 + math.exp(-x))), 4)
```

---

## Code Explanation

### Step 1: Calculate the Sigmoid Gate

```python
1 / (1 + math.exp(-x))
```

This computes the sigmoid function:

$$
\sigma(x)=\frac{1}{1+e^{-x}}
$$

The result always lies between `0` and `1`.

---

### Step 2: Multiply by the Input

```python
x * (1 / (1 + math.exp(-x)))
```

This directly implements the Swish definition:

$$
{Swish}(x)=x\sigma(x)
$$

For $x=1$:

$$
{Swish}(1)=1\times\frac{1}{1+e^{-1}}
$$

which gives approximately

```text
0.7311
```

---

### Step 3: Round the Result

```python
round(..., 4)
```

The output is rounded to four decimal places as required.

For example:

$$
0.731058\ldots\rightarrow0.7311
$$

---

### Numerical Consideration

The direct implementation is simple and works for ordinary inputs, but `math.exp(-x)` can overflow for extremely large negative values.

A numerically stable sigmoid implementation can avoid this by treating positive and negative inputs separately.

For the Deep-ML problem, the direct implementation captures the mathematical definition clearly.

---

## Time & Space Complexity

The function performs a constant number of arithmetic operations, including one exponential calculation.

Therefore, the time complexity is

$$
O(1)
$$

Only a constant number of scalar values are stored.

Therefore, the auxiliary space complexity is

$$
O(1)
$$

| Complexity | Value      |
| ---------- | ---------- |
| Time       | **$O(1)$** |
| Space      | **$O(1)$** |

The complexity is constant because the function processes a single scalar input.