# Leaky ReLU Activation Function (Easy, Deep Learning)

## Problem Statement

[Leaky ReLU Activation Function](https://www.deep-ml.com/problems/44)

Write a Python function `leaky_relu` that implements the **Leaky Rectified Linear Unit (Leaky ReLU)** activation function. The function should take a float `z` as input and an optional float `alpha` (default `0.01`) representing the slope for negative inputs. Return the value after applying the Leaky ReLU function.

---

## Example

```python
print(leaky_relu(0))
print(leaky_relu(1))
print(leaky_relu(-1))
print(leaky_relu(-2, alpha=0.1))
```

**Output**

```python
0
1
-0.01
-0.2
```

**Explanation**

- `relu(0)` returns `0`.
- Positive values remain unchanged.
- Negative values are multiplied by `alpha`.
- With `alpha=0.1`, `-2` becomes `-0.2`.

---

# Learn About the Topic

## Understanding Leaky ReLU

Leaky ReLU (Leaky Rectified Linear Unit) is an improved version of the standard ReLU activation function. While ReLU sets all negative inputs to zero, Leaky ReLU allows a small, non-zero output for negative values. This helps avoid the **dying ReLU problem**, where neurons become permanently inactive because they always output zero and stop learning.

Instead of completely blocking negative values, Leaky ReLU scales them by a small constant `α` (alpha), usually `0.01`. This ensures gradients continue to flow during backpropagation, allowing the neuron to keep updating its weights even when receiving negative inputs.

### Mathematical Definition

\[
f(z)=
\begin{cases}
z, & z > 0 \\
\alpha z, & z \le 0
\end{cases}
\]

where:

- **z** = input value
- **α (alpha)** = small positive constant (typically `0.01`)

### Properties

- **Output Range:** \((-\infty,\infty)\)
- **Positive Inputs:** Returned unchanged.
- **Negative Inputs:** Scaled by `α` instead of becoming zero.
- **Derivative:**
  - `1` for positive inputs
  - `α` for negative inputs

### Why Use Leaky ReLU?

Compared to ReLU, Leaky ReLU provides several advantages:

- Prevents neurons from "dying" by keeping gradients non-zero for negative inputs.
- Improves gradient flow during training.
- Often leads to faster and more stable convergence in deep neural networks.
- Requires almost no additional computational cost compared to ReLU.

### Comparison with ReLU

| ReLU | Leaky ReLU |
|------|------------|
| Negative inputs become `0` | Negative inputs become `αz` |
| Can suffer from dying ReLU | Greatly reduces dying ReLU |
| Gradient is `0` for negatives | Gradient is `α` for negatives |
| Most commonly used | Useful when dead neurons become an issue |

> **Note:** Choosing a very large value of `α` weakens the sparsity advantage of ReLU, while a very small value behaves almost identically to standard ReLU. In practice, `α = 0.01` works well for most applications.

---

# Solution

```python
def leaky_relu(z: float, alpha: float = 0.01) -> float | int:
    return z if z >= 0 else alpha * z
```

---

# Code Explanation

1. The function accepts an input value `z` and an optional slope `alpha`.
2. If `z` is positive (or zero), it returns `z` unchanged.
3. Otherwise, it returns `alpha * z`, allowing a small negative output instead of zero.
4. This simple modification keeps gradients flowing during training and helps prevent dead neurons.