# Leaky ReLU Activation Function (Easy, Deep Learning)

## Table of Contents

- Problem Statement
- Example
- Learn: Understanding the Leaky ReLU Activation Function
- Solution
- Code Explanation
- Time & Space Complexity

---

## Problem Statement

### [Leaky ReLU Activation Function](https://www.deep-ml.com/problems/44)

Write a Python function that implements the **Leaky Rectified Linear Unit (Leaky ReLU)** activation function.

The function should:

- Accept a floating-point input `z`.
- Accept an optional parameter `alpha`, which controls the slope for negative values.
- Return the input unchanged if it is positive.
- Return `alpha × z` if the input is negative.

The default value of `alpha` is `0.01`.

---

## Example

### Input

```python
print(leaky_relu(0))
print(leaky_relu(1))
print(leaky_relu(-1))
print(leaky_relu(-2, alpha=0.1))
```

### Output

```python
0
1
-0.01
-0.2
```

### Reasoning

For positive inputs, Leaky ReLU behaves exactly like ReLU.

For negative inputs, instead of returning zero, it returns a small fraction of the input.

```text
Input :   0     1    -1    -2

Output:   0     1  -0.01  -0.2
```

---

## Learn: Understanding the Leaky ReLU Activation Function

### What is it?

**Leaky ReLU (Leaky Rectified Linear Unit)** is a variant of the standard ReLU activation function.

Unlike ReLU, which completely blocks negative values by mapping them to zero, Leaky ReLU allows a small portion of negative values to pass through.

This small negative slope helps prevent neurons from permanently becoming inactive during training, solving one of ReLU's biggest drawbacks—the **Dying ReLU Problem**.

Leaky ReLU is widely used in deep learning models where stable gradient flow is important.

---

### Mathematical Definition

The Leaky ReLU function is defined as

$$
f(z)=
\begin{cases}
z,& z\ge0\\
\alpha z,& z<0
\end{cases}
$$

where

- $z$ is the input value.
- $\alpha$ is a small positive constant, typically

$$
\alpha=0.01
$$

---

### Graph of Leaky ReLU

The activation function has a small slope for negative inputs.

```text
Output
 ^
 |
 |          /
 |         /
 |        /
 |_______/
 |      /
 |     /
 +----------------------> Input
```

Unlike ReLU, the left side is not completely flat.

---

### Gradient of Leaky ReLU

During backpropagation, the derivative is

$$
f'(z)=
\begin{cases}
1,& z\ge0\\
\alpha,& z<0
\end{cases}
$$

Unlike ReLU, the gradient never becomes zero for negative inputs.

As a result, neurons continue receiving updates during training.

---

### ReLU vs Leaky ReLU

| ReLU                              | Leaky ReLU                               |
| --------------------------------- | ---------------------------------------- |
| Negative values become 0          | Negative values are scaled by $\alpha$   |
| Gradient is 0 for negative inputs | Gradient is $\alpha$ for negative inputs |
| Can suffer from dying neurons     | Greatly reduces dying neurons            |
| Simpler and slightly faster       | Slightly more flexible                   |

---

### Why Leaky ReLU Helps

Consider a neuron receiving only negative inputs.

With ReLU,

$$
f(z)=0
$$

The gradient also becomes

$$
0
$$

meaning the neuron's weights stop updating.

With Leaky ReLU,

$$
f(z)=\alpha z
$$

and

$$
f'(z)=\alpha
$$

Since the gradient is still non-zero, learning continues.

---

### Characteristics / Key Points

- Introduces non-linearity into neural networks.
- Keeps positive values unchanged.
- Preserves a small negative slope.
- Reduces the dying ReLU problem.
- Allows gradients to flow even for negative inputs.
- Computationally inexpensive.
- Often performs better than standard ReLU in deeper networks.

---

### Why is it used? / Applications

Leaky ReLU is commonly used in modern deep learning architectures.

Applications include

- Convolutional Neural Networks (CNNs)
- Fully Connected Neural Networks
- Generative Adversarial Networks (GANs)
- Image Classification
- Object Detection
- Speech Recognition
- Natural Language Processing
- Deep Reinforcement Learning

It is especially popular in GAN architectures, where maintaining gradient flow is essential for stable training.

---

> 💡 **Important Note**
>
> Leaky ReLU uses a **fixed negative slope** (`alpha`). Variants such as **Parametric ReLU (PReLU)** learn this slope during training, allowing the network to determine the optimal value automatically. While PReLU can improve performance in some cases, it also introduces additional trainable parameters.

---

## Solution

### Custom Implementation

```python
def leaky_relu(z: float, alpha: float = 0.01) -> float | int:
    return z if z >= 0 else alpha * z
```

### NumPy Implementation

```python
import numpy as np

def leaky_relu(z, alpha=0.01):
    return np.where(z >= 0, z, alpha * z)
```

### PyTorch Implementation

```python
import torch
import torch.nn.functional as F

output = F.leaky_relu(input_tensor, negative_slope=0.01)
```

---

## Code Explanation

### Step 1

Receive the input value and the negative slope.

```python
z
alpha
```

The parameter `alpha` controls how much of the negative input is retained.

---

### Step 2

Check whether the input is non-negative.

```python
z >= 0
```

If true, return the input unchanged.

---

### Step 3

Otherwise, scale the negative value.

```python
alpha * z
```

Instead of clipping negative values to zero, Leaky ReLU preserves a small negative output.

---

### Step 4

Return the activated value.

The function behaves exactly like ReLU for positive inputs while maintaining a small gradient for negative inputs.

---

## Time & Space Complexity

| Complexity | Value    |
| ---------- | -------- |
| Time       | **O(1)** |
| Space      | **O(1)** |

where

- The function performs a single comparison and one multiplication, requiring constant time and constant additional space.
