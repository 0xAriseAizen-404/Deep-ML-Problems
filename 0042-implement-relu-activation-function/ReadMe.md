# Implement ReLU Activation Function (Easy, Deep Learning)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: ReLU Activation Function](#learn-relu-activation-function)
- [Solutions](#solutions)
  - [Custom Implementation](#custom-implementation)
- [Code Explanation](#code-explanation)

## Problem Statement

[Implement ReLU Activation Function](https://www.deep-ml.com/problems/42)

Write a Python function `relu` that implements the Rectified Linear Unit (ReLU) activation function. The function takes a single float as input and returns the input if it is positive; otherwise, it returns `0.0`.

## Example

```python
print(relu(0))
print(relu(1))
print(relu(-1))

# Output
0.0
1.0
0.0
```

## Learn: ReLU Activation Function

The ReLU (Rectified Linear Unit) activation function is one of the most widely used activation functions in deep learning.

**Formula**

$$
\text{ReLU}(x) = \max(0, x)
$$

- If `x > 0`, return `x`
- Otherwise, return `0`

## Solutions

### Custom Implementation

```python
def relu(z: float) -> float:
    return z if z > 0.0 else 0.0
```

## Code Explanation

The function checks whether the input is positive:

1. If `z > 0`, it returns `z`.
2. Otherwise, it returns `0.0`.

ReLU is simple, computationally efficient, and is the default activation function used in most modern neural networks because it helps reduce the vanishing gradient problem.