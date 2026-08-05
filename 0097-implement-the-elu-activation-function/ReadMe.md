# Implement the ELU Activation Function (Easy, Deep Learning)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: ELU Activation Function](#learn-elu-activation-function)
- [Solutions](#solutions)
  - [Custom Implementation](#custom-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

## Problem Statement

### [Implement the ELU Activation Function](https://www.deep-ml.com/problems/97)

Implement a Python function `elu(x, alpha=1.0)` that calculates the ELU (Exponential Linear Unit) activation function.

ELU is an activation function designed to overcome some limitations of ReLU by allowing negative outputs for negative inputs.

The function takes:

- `x`: Input value.
- `alpha`: Hyperparameter controlling the saturation value for negative inputs.

The ELU function is defined as:

$$
ELU(x) =
\begin{cases}
x, & x > 0 \\
\alpha(e^x-1), & x \leq 0
\end{cases}
$$

The function should return the ELU activation value rounded to four decimal places.

## Example

### Input

```python
elu(-1)
```

### Output

```text
-0.6321
```

### Reasoning

For:

$$ x = -1 $$

and:

$$ \alpha = 1 $$

Since:

$$ x \leq 0 $$

the negative branch is used:

$$ ELU(-1) = 1(e^{-1}-1) $$

$$ ELU(-1) = 0.3679-1 $$

$$ ELU(-1) = -0.6321 $$

## Learn: ELU Activation Function

### What is it?

ELU (Exponential Linear Unit) is an activation function that improves upon ReLU by allowing negative values instead of replacing all negative inputs with zero.

ReLU is defined as:

$$ ReLU(x) = max(0,x) $$

For negative inputs, ReLU produces:

$$ ReLU(x)=0 $$

This can cause the **dying ReLU problem**, where neurons stop learning because their gradients become zero.

ELU solves this by using an exponential function for negative values, creating a smooth transition around zero.

### Mathematical Definition / Formula

The ELU function is:

$$
ELU(x) =
\begin{cases}
x, & x > 0 \
\alpha(e^x-1), & x \leq 0
\end{cases}
$$

where:

- $x$ is the input value.
- $\alpha$ is a hyperparameter controlling the minimum output value.
- $e$ is Euler's number.

For the commonly used value:

$$ \alpha = 1 $$

the function becomes:

$$
ELU(x) =
\begin{cases}
x, & x > 0 \
e^x-1, & x \leq 0
\end{cases}
$$

### Characteristics / Key Points

- Output range:

$$ [-\alpha,\infty) $$

- Positive inputs behave like the identity function.

$$ ELU(x)=x \text{ for } x>0 $$

- Negative inputs produce smooth negative values.
- Unlike ReLU, ELU allows negative outputs.
- The function is smooth at:

$$ x=0 $$

- Negative outputs push activations closer to zero mean.

### Gradient Behavior

The derivative of ELU is:

$$
ELU'(x) =
\begin{cases}
1, & x > 0 \
\alpha e^x, & x \leq 0
\end{cases}
$$

For positive values:

$$ ELU'(x)=1 $$

For negative values:

$$ ELU'(x)=\alpha e^x $$

Unlike ReLU, ELU maintains a non-zero gradient for negative inputs.

### ELU vs ReLU

| Property          | ReLU   | ELU                  |
| ----------------- | ------ | -------------------- |
| Negative Output   | No     | Yes                  |
| Smooth at zero    | No     | Yes                  |
| Negative Gradient | 0      | Non-zero             |
| Computation       | Simple | Requires exponential |

ReLU:

$$ ReLU(-2)=0 $$

ELU:

$$ ELU(-2)=e^{-2}-1 $$

$$ ELU(-2)\approx -0.8647 $$

### Advantages

ELU provides:

- Reduced dying ReLU problem.
- Better gradient flow for negative inputs.
- Smoother optimization.
- Negative activations that help move neuron outputs closer to zero.

### Limitations

- Requires exponential computation.
- Slightly slower compared to ReLU.
- Introduces an additional hyperparameter $\alpha$.

### Why is it used? / Applications

ELU is used in deep neural networks where smoother optimization and better gradient flow are important.

Applications:

- Deep feed-forward neural networks.
- Computer vision models.
- Regression networks.
- Models where ReLU neurons frequently die.

> 💡 **Important Note**
>
> ELU is an alternative to ReLU when negative activations are beneficial. However, modern architectures often prefer ReLU variants like Leaky ReLU, GELU, or SiLU depending on the task.

## Solutions

### Custom Implementation

```python id="n6x8qh"
import math

def elu(x: float, alpha: float = 1.0) -> float:
    elu = alpha * (math.exp(x) - 1) if x <= 0.0 else x
    return round(elu, 4)
```

## Code Explanation

### Step 1: Handle Positive Inputs

```python id="f3p8yz"
x > 0
```

For positive values, ELU behaves like the identity function.

Example:

$$ ELU(3)=3 $$

The function directly returns the input.

### Step 2: Handle Negative Inputs

```python id="c6w1kp"
alpha * (math.exp(x) - 1)
```

For negative inputs, the exponential transformation is applied.

Example:

$$ ELU(-1)=1(e^{-1}-1) $$

This creates a smooth negative curve.

### Step 3: Round the Output

```python id="v2s7ra"
round(elu, 4)
```

The result is rounded to four decimal places as required.

## Time & Space Complexity

| Complexity | Value    |
| ---------- | -------- |
| Time       | **O(1)** |
| Space      | **O(1)** |

Where:

- The function performs a constant number of arithmetic operations.
- No additional memory is required.
