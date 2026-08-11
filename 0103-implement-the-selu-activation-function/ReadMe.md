# Implement the SELU Activation Function (Easy, Deep Learning)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Understanding the SELU Activation Function](#learn-understanding-the-selu-activation-function)
- [Solution](#solution)
  - [Custom Implementation](#custom-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

---

## Problem Statement

### [Implement the SELU Activation Function](https://www.deep-ml.com/problems/103)

Implement the **SELU (Scaled Exponential Linear Unit)** activation function, a self-normalizing variant of the ELU activation function.

The function should take a numerical input $x$ and return its SELU activation value while maintaining numerical stability.

SELU uses carefully chosen scale and alpha parameters to encourage activations to maintain approximately zero mean and unit variance.

---

## Example

### Input

```python
selu(-1.0)
```

### Output

```text
-1.1113
```

### Reasoning

For negative inputs, SELU is defined as

$$
{SELU}(x)=\lambda\alpha(e^x-1)
$$

Using

$$
\lambda\approx1.0507
$$

and

$$
\alpha\approx1.6733
$$

for $x=-1$:

$$
{SELU}(-1)=1.0507\times1.6733\times(e^{-1}-1)
$$

which gives approximately

$$
{SELU}(-1)\approx-1.1113
$$

---

## Learn: Understanding the SELU Activation Function

### What is it?

**SELU (Scaled Exponential Linear Unit)** is an activation function designed to create **self-normalizing neural networks**.

SELU is closely related to the ELU activation function, but introduces a scaling factor $\lambda$.

The carefully selected values of $\lambda$ and $\alpha$ allow activations to tend toward zero mean and unit variance under suitable conditions.

SELU was introduced as part of the **Self-Normalizing Neural Networks** approach and is particularly useful in deep feed-forward networks.

---

### Mathematical Definition

SELU is defined piecewise:

$$
{SELU}(x)=
\begin{cases}
\lambda x & \text{if } x>0\
\lambda\alpha(e^x-1) & \text{if } x\leq0
\end{cases}
$$

where

$$
\lambda\approx1.0507
$$

and

$$
\alpha\approx1.6733
$$

More precise values commonly used are

$$
\lambda=1.0507009873554804
$$

and

$$
\alpha=1.6732632423543772
$$

These values are chosen specifically to provide the self-normalizing behavior of SELU.

---

### Positive Region

For positive inputs, SELU behaves like a scaled linear function:

$$
{SELU}(x)=\lambda x
$$

Since $\lambda>1$, the positive input is slightly amplified.

For example, for $x=2$:

$$
{SELU}(2)=1.0507\times2\approx2.1014
$$

Unlike ELU, the positive region is not simply $x$ because SELU applies the scale factor.

---

### Negative Region

For non-positive inputs:

$$
{SELU}(x)=\lambda\alpha(e^x-1)
$$

The exponential term provides a smooth transition toward a negative saturation value.

As $x$ becomes very negative,

$$
e^x\rightarrow0
$$

Therefore,

$$
{SELU}(x)\rightarrow-\lambda\alpha
$$

The lower bound is approximately

$$
-\lambda\alpha\approx-1.7581
$$

Thus, negative activations are bounded while positive activations remain unbounded.

---

### Continuity

SELU is continuous at $x=0$.

From the positive side:

$$
\lambda(0)=0
$$

From the negative side:

$$
\lambda\alpha(e^0-1)=\lambda\alpha(1-1)=0
$$

Therefore, both branches meet at zero.

The derivative is also continuous at zero.

For $x>0$:

$$
f'(x)=\lambda
$$

For $x<0$:

$$
f'(x)=\lambda\alpha e^x
$$

At $x=0$:

$$
f'(0)=\lambda\alpha
$$

The SELU parameters are selected such that

$$
\lambda\alpha\approx1.7581
$$

This means the basic SELU definition has a smooth value transition, while its derivative behavior differs from a simple ELU.

---

### Output Range

For negative inputs, SELU approaches

$$
-\lambda\alpha
$$

For positive inputs, the output grows without bound.

Therefore, the range is

$$
(-\lambda\alpha,\infty)
$$

Using the standard parameters:

$$
{Range}({SELU})\approx(-1.7581,\infty)
$$

---

### Self-Normalizing Property

The key idea behind SELU is **self-normalization**.

Under suitable conditions, repeated application of SELU causes the mean and variance of activations to move toward stable values:

$$
\mu\rightarrow0
$$

and

$$
\sigma^2\rightarrow1
$$

This can help maintain stable activation distributions as information passes through many layers.

The self-normalizing behavior depends on assumptions such as appropriate initialization, network structure, and input distributions.

It is therefore not a guarantee that every SELU network will automatically have exactly zero mean and unit variance.

---

### SELU vs ELU

SELU is based directly on ELU.

ELU is

$$
{ELU}(x)=
\begin{cases}
x & \text{if } x>0\
\alpha(e^x-1) & \text{if } x\leq0
\end{cases}
$$

SELU adds a scaling factor:

$$
{SELU}(x)=\lambda{ELU}(x)
$$

This scaling is essential to the self-normalizing behavior.

| Property              | ELU             | SELU                           |
| --------------------- | --------------- | ------------------------------ |
| Positive Region       | $x$             | $\lambda x$                    |
| Negative Region       | $\alpha(e^x-1)$ | $\lambda\alpha(e^x-1)$         |
| Bounded Negative Side | Yes             | Yes                            |
| Self-Normalizing      | No              | Yes, under suitable conditions |
| Scale Parameter       | No              | $\lambda$                      |
| Alpha Parameter       | Yes             | Yes                            |

---

### SELU vs ReLU

ReLU is defined as

$$
{ReLU}(x)=\max(0,x)
$$

SELU differs by allowing negative activations and scaling both regions.

| Property            | ReLU | SELU                           |
| ------------------- | ---- | ------------------------------ |
| Negative Output     | No   | Yes                            |
| Smooth              | No   | Yes                            |
| Positive Region     | $x$  | $\lambda x$                    |
| Negative Region     | $0$  | $\lambda\alpha(e^x-1)$         |
| Negative Saturation | $0$  | $-\lambda\alpha$               |
| Self-Normalizing    | No   | Yes, under suitable conditions |

ReLU is simpler and widely used, while SELU is specifically designed for self-normalizing architectures.

---

### Derivative

For positive inputs:

$$
{SELU}'(x)=\lambda
$$

For negative inputs:

$$
{SELU}'(x)=\lambda\alpha e^x
$$

Therefore,

$$
{SELU}'(x)=
\begin{cases}
\lambda & \text{if } x>0\
\lambda\alpha e^x & \text{if } x\leq0
\end{cases}
$$

For large negative values, the derivative approaches zero because

$$
e^x\rightarrow0
$$

---

### Characteristics / Key Points

- SELU stands for **Scaled Exponential Linear Unit**.
- It is a self-normalizing activation function.
- SELU is based on ELU with an additional scale factor.
- The standard scale is approximately $\lambda=1.0507$.
- The standard alpha is approximately $\alpha=1.6733$.
- Positive values are scaled linearly.
- Negative values follow an exponential curve.
- Negative outputs are bounded below by approximately $-1.7581$.
- Positive outputs are unbounded.
- SELU is continuous.
- SELU provides smooth negative activations.
- Its self-normalizing behavior can help stabilize deep networks.
- Proper initialization and architecture are important for self-normalization.

---

### Why is it used? / Applications

SELU can be useful in:

- Deep feed-forward neural networks.
- Self-normalizing neural networks.
- Networks where maintaining stable activation statistics is important.
- Architectures where batch normalization is undesirable.
- Situations where normalized activations are needed without explicitly applying batch normalization.

SELU is particularly associated with **Self-Normalizing Neural Networks (SNNs)**.

---

> 💡 **Important Note**
>
> SELU does not guarantee zero mean and unit variance for arbitrary networks. Its self-normalizing behavior relies on specific assumptions about initialization, architecture, and input distributions.

> 💡 **Interview Tip**
>
> Remember that **SELU = scaled ELU**. The two important constants are $\lambda\approx1.0507$ and $\alpha\approx1.6733$. The positive branch is $\lambda x$, while the negative branch is $\lambda\alpha(e^x-1)$.

---

## Solution

### Custom Implementation

```python
import math

def selu(x: float) -> float:
    scale = 1.0507009873554804
    alpha = 1.6732632423543772
    if x > 0:
        return scale * x
    return scale * alpha * (math.exp(x) - 1)
```

---

## Code Explanation

### Step 1: Define SELU Constants

```python
scale = 1.0507009873554804
alpha = 1.6732632423543772
```

These are the standard SELU constants.

They are chosen to provide the self-normalizing behavior of the activation function.

---

### Step 2: Handle Positive Inputs

```python
if x > 0:
    return scale * x
```

For positive values, SELU uses the linear branch:

$$
{SELU}(x)=\lambda x
$$

For example:

$$
{SELU}(2)=1.0507\times2\approx2.1014
$$

---

### Step 3: Handle Non-Positive Inputs

```python
return scale * alpha * (math.exp(x) - 1)
```

For $x\leq0$, the exponential branch is used:

$$
{SELU}(x)=\lambda\alpha(e^x-1)
$$

For $x=-1$:

$$
{SELU}(-1)=1.0507\times1.6733\times(e^{-1}-1)
$$

which produces approximately

```text
-1.1113
```

---

### Step 4: Why the Branching Is Necessary

The SELU function is piecewise-defined.

The positive branch is linear:

$$
\lambda x
$$

while the non-positive branch is exponential:

$$
\lambda\alpha(e^x-1)
$$

Therefore, the implementation must determine which branch applies before calculating the result.

---

### Numerical Stability

The negative branch uses

```python
math.exp(x)
```

For very negative $x$, the exponential approaches zero:

$$
e^x\rightarrow0
$$

which is numerically safe for typical floating-point inputs.

The positive branch avoids computing $e^{-x}$ entirely, preventing unnecessary exponential calculations for large positive inputs.

---

## Time & Space Complexity

The function performs a constant number of arithmetic operations and at most one exponential calculation.

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
