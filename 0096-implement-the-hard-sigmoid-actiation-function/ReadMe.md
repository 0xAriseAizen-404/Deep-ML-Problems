# Implement the Hard Sigmoid Activation Function (Easy, Deep Learning)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Hard Sigmoid Activation Function](#learn-hard-sigmoid-activation-function)
- [Solutions](#solutions)
  - [Custom Implementation](#custom-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

## Problem Statement

### [Implement the Hard Sigmoid Activation Function](https://www.deep-ml.com/problems/96)

Implement the Hard Sigmoid activation function.

Hard Sigmoid is a computationally efficient approximation of the standard sigmoid activation function. Instead of using an exponential operation, it uses a piecewise linear function.

The function takes a single input value `x` and returns the corresponding Hard Sigmoid output.

The implementation should follow the Keras Hard Sigmoid definition:

$$
HardSigmoid(x) =
\begin{cases}
0, & x \leq -2.5 \\
0.2x + 0.5, & -2.5 < x < 2.5 \\
1, & x \geq 2.5
\end{cases}
$$

## Example

### Input

```python
hard_sigmoid(0.0)
```

### Output

```text
0.5
```

### Reasoning

The input value:

$$ x = 0.0 $$

falls inside the linear region:

$$ -2.5 < x < 2.5 $$

Therefore, the linear formula is applied:

$$ HardSigmoid(0.0) = 0.2(0.0) + 0.5 $$

$$ HardSigmoid(0.0) = 0.5 $$

## Learn: Hard Sigmoid Activation Function

### What is it?

Hard Sigmoid is a piecewise linear approximation of the sigmoid activation function.

The standard sigmoid function uses an exponential calculation:

$$ Sigmoid(x) = \frac{1}{1+e^{-x}} $$

Although sigmoid provides a smooth output between 0 and 1, the exponential operation is computationally expensive.

Hard Sigmoid replaces the exponential curve with three simple regions:

- A constant region returning 0.
- A linear region.
- A constant region returning 1.

This makes it faster and easier to compute while maintaining sigmoid-like behavior.

### Mathematical Definition / Formula

The Hard Sigmoid function is defined as:

$$
HardSigmoid(x) =
\begin{cases}
0, & x \leq -2.5 \
0.2x + 0.5, & -2.5 < x < 2.5 \
1, & x \geq 2.5
\end{cases}
$$

The linear section has:

- Slope = 0.2
- Bias = 0.5

For example:

$$ x = 1 $$

$$ HardSigmoid(1) = 0.2(1) + 0.5 = 0.7 $$

### Relationship with Sigmoid

Standard sigmoid:

$$ Sigmoid(x) = \frac{1}{1+e^{-x}} $$

Hard sigmoid approximates sigmoid by replacing the smooth curve with a straight line.

Comparison:

| Activation   | Computation               | Output Range |
| ------------ | ------------------------- | ------------ |
| Sigmoid      | Exponential function      | [0, 1]       |
| Hard Sigmoid | Piecewise linear function | [0, 1]       |

### Characteristics / Key Points

- Output is always bounded between 0 and 1.

$$ 0 \leq HardSigmoid(x) \leq 1 $$

- The function has three regions:
  - Saturation at 0.
  - Linear transition region.
  - Saturation at 1.

- The gradient is constant in the linear region.
- The gradient becomes zero in saturated regions.
- It avoids expensive exponential calculations.

### Gradient Behavior

The derivative of Hard Sigmoid is:

$$
HardSigmoid'(x) =
\begin{cases}
0, & x \leq -2.5 \
0.2, & -2.5 < x < 2.5 \
0, & x \geq 2.5
\end{cases}
$$

The gradient only exists in the linear region.

### Advantages in Neural Networks

Hard Sigmoid provides:

- Faster computation compared to sigmoid.
- Simple gradient calculation.
- Bounded output similar to sigmoid.
- Lower computational cost on hardware with limited resources.

### Limitations

Hard Sigmoid also has drawbacks:

- The gradient becomes zero in saturated regions.
- It is not smooth like sigmoid.
- It may lose information because of hard clipping.

### Why is it used? / Applications

Hard Sigmoid is useful in situations where computational efficiency is important.

Applications:

- Recurrent neural networks:
  - Used in some lightweight LSTM variants.

- Mobile and embedded neural networks.
- Resource-constrained deep learning systems.
- Activation functions where approximate sigmoid behavior is sufficient.

> 💡 **Important Note**
>
> Hard Sigmoid is mainly used when computational efficiency is more important than the smooth gradient behavior of standard sigmoid. In modern deep networks, ReLU variants are often preferred for hidden layers, while sigmoid-like functions are mainly used for output probabilities and gates.

## Solutions

### Custom Implementation

```python id="1m4q7w"
def hard_sigmoid(x: float) -> float:
    if x <= -2.5:
        return 0
    elif -2.5 < x < 2.5:
        return 0.2 * x + 0.5
    else:
        return 1
```

## Code Explanation

### Step 1: Check Lower Saturation Region

```python id="3y9x8p"
if x <= -2.5:
    return 0
```

For values less than or equal to `-2.5`, the function outputs zero.

Example:

$$ x = -3 $$

$$ HardSigmoid(-3) = 0 $$

### Step 2: Apply Linear Region

```python id="k7d2mc"
elif -2.5 < x < 2.5:
    return 0.2 * x + 0.5
```

For values inside the range:

$$ -2.5 < x < 2.5 $$

the linear approximation is used.

Example:

$$ x = 1 $$

$$ HardSigmoid(1) = 0.2(1)+0.5 = 0.7 $$

### Step 3: Check Upper Saturation Region

```python id="4v9fkm"
else:
    return 1
```

For values greater than or equal to `2.5`, the function outputs one.

Example:

$$ x = 4 $$

$$ HardSigmoid(4) = 1 $$

## Time & Space Complexity

| Complexity | Value    |
| ---------- | -------- |
| Time       | **O(1)** |
| Space      | **O(1)** |

Where:

- The function performs a fixed number of comparisons and arithmetic operations.
- No additional memory is required.
