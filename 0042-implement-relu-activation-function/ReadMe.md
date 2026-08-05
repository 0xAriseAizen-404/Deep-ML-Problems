# Implement ReLU Activation Function (Easy, Deep Learning)

## Table of Contents

- Problem Statement
- Example
- Learn: Understanding the ReLU Activation Function
- Solution
- Code Explanation
- Time & Space Complexity

---

## Problem Statement

### [Implement ReLU Activation Function](https://www.deep-ml.com/problems/42)

Write a Python function that implements the **Rectified Linear Unit (ReLU)** activation function.

The function should:

- Accept a single floating-point value.
- Return the input itself if it is positive.
- Return `0` if the input is negative or zero.

ReLU is one of the most widely used activation functions in modern deep learning.

---

## Example

### Input

```python
print(relu(0))
print(relu(1))
print(relu(-1))
```

### Output

```python
0.0
1.0
0.0
```

### Reasoning

The ReLU function keeps positive values unchanged while replacing negative values with zero.

```text
Input :  0   1  -1

Output:  0   1   0
```

---

## Learn: Understanding the ReLU Activation Function

### What is it?

The **Rectified Linear Unit (ReLU)** is one of the most popular activation functions used in neural networks.

An activation function introduces **non-linearity**, allowing neural networks to learn complex patterns that cannot be represented using only linear transformations.

ReLU is simple, computationally efficient, and helps train deep neural networks much faster than earlier activation functions such as Sigmoid and Tanh.

Because of these advantages, ReLU has become the default activation function for most hidden layers in deep learning.

---

### Mathematical Definition

The ReLU function is defined as

$$
f(z)=\max(0,z)
$$

or equivalently,

$$
f(z)=
\begin{cases}
z,& z>0\\
0,& z\le0
\end{cases}
$$

where

- $z$ is the input to the activation function.

---

### Graph of ReLU

The function has a simple shape.

```text
Output
 ^
 |
 |         /
 |        /
 |       /
 |______/
 |
 +---------------------> Input
```

For negative inputs, the output remains zero.

For positive inputs, the output increases linearly.

---

### Gradient of ReLU

During backpropagation, the derivative of ReLU is

$$
f'(z)=
\begin{cases}
1,& z>0\\
0,& z<0
\end{cases}
$$

The derivative at

$$
z=0
$$

is mathematically undefined.

In practice, deep learning libraries simply choose either

$$
0
$$

or

$$
1
$$

since this occurs only at a single point and has negligible impact during training.

---

### Why ReLU Works Well

Earlier activation functions such as Sigmoid compress outputs into a very small range.

For large positive or negative inputs, their gradients become extremely small.

This causes the **Vanishing Gradient Problem**, making deep networks difficult to train.

ReLU avoids this issue for positive inputs because its derivative is

$$
1
$$

allowing gradients to flow through many layers efficiently.

---

### ReLU vs Sigmoid vs Tanh

| Activation | Output Range | Gradient Issue                                     |
| ---------- | ------------ | -------------------------------------------------- |
| Sigmoid    | $(0,1)$      | Suffers from vanishing gradients                   |
| Tanh       | $(-1,1)$     | Reduced but still suffers from vanishing gradients |
| ReLU       | $[0,\infty)$ | Efficient for positive values                      |

ReLU is significantly faster because it only requires a comparison instead of expensive exponential computations.

---

### Characteristics / Key Points

- Simple and computationally efficient.
- Introduces non-linearity.
- Outputs zero for negative inputs.
- Leaves positive values unchanged.
- Helps alleviate the vanishing gradient problem.
- Sparse activations improve computational efficiency.
- Most commonly used activation function in hidden layers.

---

### Why is it used? / Applications

ReLU is used throughout modern deep learning architectures.

Applications include

- Convolutional Neural Networks (CNNs)
- Fully Connected Neural Networks
- Residual Networks (ResNet)
- Object Detection
- Image Classification
- Natural Language Processing
- Speech Recognition
- Reinforcement Learning
- Generative Models

Nearly every state-of-the-art deep learning architecture uses ReLU or one of its variants.

---

> 💡 **Important Note**
>
> A common issue with ReLU is the **Dying ReLU Problem**. If a neuron's weights cause its output to remain negative for all inputs, its gradient becomes zero and it stops learning permanently. Variants such as **Leaky ReLU**, **Parametric ReLU (PReLU)**, and **ELU** were introduced to address this problem.

---

## Solution

### Custom Implementation

```python
def relu(z: float) -> float:
    return z if z > 0.0 else 0.0
```

### NumPy Implementation

```python
import numpy as np

def relu(z):
    return np.maximum(0, z)
```

### PyTorch Implementation

```python
import torch
import torch.nn.functional as F

output = F.relu(input_tensor)
```

---

## Code Explanation

### Step 1

Receive the input value.

```python
z
```

This represents the neuron's pre-activation value (also called the **logit**).

---

### Step 2

Check whether the input is positive.

```python
z > 0.0
```

If the condition is true, return the input unchanged.

---

### Step 3

Otherwise, return zero.

```python
0.0
```

Negative values are clipped to zero, introducing non-linearity into the network.

---

## Time & Space Complexity

| Complexity | Value    |
| ---------- | -------- |
| Time       | **O(1)** |
| Space      | **O(1)** |

where

- The function performs a single comparison and returns one value, requiring constant time and constant additional space.
