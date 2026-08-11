# Implement the Softplus Activation Function (Easy, Deep Learning)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Understanding the Softplus Activation Function](#learn-understanding-the-softplus-activation-function)
- [Solution](#solution)
  - [Custom Implementation](#custom-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

---

## Problem Statement

### [Implement the Softplus Activation Function](https://www.deep-ml.com/problems/99)

Implement the **Softplus activation function**, a smooth approximation of the ReLU activation function.

The function should take a numerical input $x$ and return the Softplus value rounded to **4 decimal places**.

The implementation should also handle large positive and negative inputs safely to avoid numerical overflow or underflow.

---

## Example

### Input

```python
softplus(2)
```

### Output

```text
2.1269
```

### Reasoning

The Softplus function is

$$
Softplus(x)=\log(1+e^x)
$$

For $x=2$:

$$
Softplus(2)=\log(1+e^2)
$$

Since

$$
e^2\approx7.3891
$$

we get

$$
\log(1+7.3891)\approx2.1269
$$

Therefore, the output is

```text
2.1269
```

---

## Learn: Understanding the Softplus Activation Function

### What is it?

**Softplus** is a smooth, continuously differentiable activation function used in neural networks.

It is closely related to **ReLU (Rectified Linear Unit)** and can be viewed as a smooth approximation of ReLU.

ReLU is defined as

$$
ReLU(x)=\max(0,x)
$$

Softplus replaces the sharp transition of ReLU around zero with a smooth curve.

The standard definition is

$$
Softplus(x)=\log(1+e^x)
$$

For large positive values, Softplus behaves approximately like $x$.

For large negative values, Softplus approaches $0$.

Therefore, its overall behavior resembles ReLU while remaining smooth everywhere.

---

### Mathematical Definition

The Softplus function is

$$
f(x)=\log(1+e^x)
$$

where

- $x$ is the input.
- $e$ is Euler's number, approximately $2.71828$.
- $\log$ is the natural logarithm.
- $f(x)$ is the Softplus output.

The function is always positive:

$$
f(x)>0
$$

for every finite value of $x$.

---

### Behavior for Large Negative Inputs

When $x$ is very negative,

$$
e^x\rightarrow0
$$

Therefore,

$$
\log(1+e^x)\rightarrow\log(1)=0
$$

So,

$$
\lim_{x\rightarrow-\infty}Softplus(x)=0
$$

This is similar to the negative region of ReLU.

For example:

```text
x = -10
Softplus(x) ≈ 0.000045
```

The result is very close to zero but never exactly zero for finite $x$.

---

### Behavior for Large Positive Inputs

When $x$ is very large,

$$
e^x\gg1
$$

Therefore,

$$
\log(1+e^x)\approx\log(e^x)
$$

and hence

$$
Softplus(x)\approx x
$$

Thus,

$$
\lim_{x\rightarrow\infty}\left(Softplus(x)-x\right)=0
$$

This is why Softplus behaves similarly to ReLU for large positive inputs.

---

### Relationship with ReLU

ReLU is

$$
ReLU(x)=\max(0,x)
$$

Softplus is

$$
Softplus(x)=\log(1+e^x)
$$

Their behavior can be summarized as:

| Input Region       | ReLU              | Softplus          |
| ------------------ | ----------------- | ----------------- |
| Large negative $x$ | $0$               | Approximately $0$ |
| $x=0$              | $0$               | $\log(2)$         |
| Large positive $x$ | Approximately $x$ | Approximately $x$ |

At $x=0$:

$$
Softplus(0)=\log(2)\approx0.6931
$$

Unlike ReLU, Softplus never has a sharp corner at zero.

---

### Smoothness

One of the main differences between Softplus and ReLU is smoothness.

ReLU has a sharp transition at $x=0$.

Softplus changes gradually through the same region.

The first derivative of Softplus is

$$
\frac{d}{dx}Softplus(x)=\frac{1}{1+e^{-x}}
$$

which is exactly the **sigmoid function**.

Therefore,

$$
Softplus'(x)={Sigmoid}(x)
$$

Since sigmoid is continuous, Softplus is continuously differentiable.

---

### Derivative

Starting with

$$
f(x)=\log(1+e^x)
$$

using the chain rule,

$$
f'(x)=\frac{1}{1+e^x}\cdot e^x
$$

which simplifies to

$$
f'(x)=\frac{e^x}{1+e^x}
$$

Dividing the numerator and denominator by $e^x$ gives

$$
f'(x)=\frac{1}{1+e^{-x}}
$$

Therefore,

$$
f'(x)={Sigmoid}(x)
$$

The derivative is always between `0` and `1`:

$$
0<f'(x)<1
$$

---

### Second Derivative

The second derivative of Softplus is

$$
f''(x)={Sigmoid}(x)(1-{Sigmoid}(x))
$$

Since sigmoid lies between `0` and `1`,

$$
f''(x)>0
$$

Therefore, Softplus is a **convex function**.

The second derivative reaches its maximum at $x=0$.

---

### Numerical Stability

The direct implementation

```python
math.log(1 + math.exp(x))
```

can cause numerical problems for very large positive values.

For example, if $x$ is extremely large, calculating

$$
e^x
$$

can exceed the largest value representable by a floating-point number.

This causes an overflow.

For very negative values, $e^x$ can become extremely close to zero, resulting in underflow.

A numerically stable form of Softplus is

$$
Softplus(x)=\max(0,x)+\log(1+e^{-|x|})
$$

This expression avoids directly calculating an unnecessarily large exponential.

---

### Stable Form for Positive Inputs

If $x\geq0$,

$$
Softplus(x)=x+\log(1+e^{-x})
$$

Because $-x\leq0$, the exponential remains bounded:

$$
0<e^{-x}\leq1
$$

This prevents overflow.

---

### Stable Form for Negative Inputs

If $x<0$,

$$
Softplus(x)=\log(1+e^x)
$$

Since $x$ is negative,

$$
0<e^x<1
$$

so the exponential remains small and safe to compute.

The combined stable expression is

$$
Softplus(x)=\max(0,x)+\log(1+e^{-|x|})
$$

In Python, `math.log1p(z)` is preferred over `math.log(1 + z)` for small values of $z$ because it provides better numerical precision.

Therefore, a stable implementation is

```python
math.log1p(math.exp(-abs(x))) + max(0, x)
```

---

### Output Range

For every finite real input,

$$
Softplus(x)>0
$$

As $x$ approaches negative infinity,

$$
Softplus(x)\rightarrow0
$$

As $x$ approaches positive infinity,

$$
Softplus(x)\rightarrow\infty
$$

Therefore, the range is

$$
(0,\infty)
$$

Softplus does not produce negative outputs.

---

### Important Values

Some useful values are:

$$
Softplus(-2)\approx0.1269
$$

$$
Softplus(0)\approx0.6931
$$

$$
Softplus(2)\approx2.1269
$$

Notice that

$$
Softplus(-x)=\log(1+e^{-x})
$$

and positive and negative inputs are not symmetric around zero.

---

### Characteristics / Key Points

- Softplus is a smooth approximation of ReLU.
- Its formula is $\log(1+e^x)$.
- Its output is always positive.
- It never produces exactly zero for finite inputs.
- It is continuously differentiable.
- Its derivative is the sigmoid function.
- It approaches zero for large negative inputs.
- It approaches $x$ for large positive inputs.
- It has no sharp corner at zero.
- It is convex.
- It can suffer from numerical overflow if implemented naively.
- A stable implementation should avoid directly computing $e^x$ for large positive $x$.
- `log1p()` improves numerical precision for small values.

---

### Softplus vs ReLU

| Property                    | ReLU        | Softplus                           |
| --------------------------- | ----------- | ---------------------------------- |
| Formula                     | $\max(0,x)$ | $\log(1+e^x)$                      |
| Negative output             | No          | No                                 |
| Smooth                      | No          | Yes                                |
| Differentiable at $0$       | No          | Yes                                |
| Positive-region behavior    | $x$         | Approximately $x$                  |
| Negative-region behavior    | $0$         | Approximately $0$                  |
| Trainable parameters        | None        | None                               |
| Numerical stability concern | Very low    | Important for naive implementation |

Softplus is therefore useful when a smooth approximation of ReLU is desirable.

---

### Why is it used? / Applications

Softplus can be used in:

- Neural network activation functions.
- Models requiring smooth activation functions.
- Optimization problems where differentiability is useful.
- Positive-valued output transformations.
- Probabilistic neural networks.
- Smooth approximations to ReLU.
- Some generative and density-based models.

Although ReLU is generally more common in standard deep neural networks, Softplus remains useful when smooth gradients are desirable.

---

> 💡 **Important Note**
>
> Do not implement Softplus naively as `math.log(1 + math.exp(x))` when arbitrary input values are allowed. For large positive $x$, `math.exp(x)` can overflow. The numerically stable expression $\max(0,x)+\log(1+e^{-|x|})$ avoids this problem.

> 💡 **Interview Tip**
>
> Remember the key relationship: **Softplus is the smooth version of ReLU, and its derivative is sigmoid**. The two most important formulas are $Softplus(x)=\log(1+e^x)$ and $Softplus'(x)=\frac{1}{1+e^{-x}}$.

---

## Solution

### Custom Implementation

```python
import math

def softplus(x: float) -> float:
    return round(max(0, x) + math.log1p(math.exp(-abs(x))), 4)
```

---

## Code Explanation

### Step 1: Handle the Sign of the Input

```python
max(0, x)
```

This contributes:

- `0` when $x<0$.
- $x$ when $x\geq0$.

This is the first part of the numerically stable Softplus formula.

---

### Step 2: Compute the Stable Exponential Term

```python
math.exp(-abs(x))
```

The absolute value ensures that the exponent is never positive.

Therefore,

$$
-|x|\leq0
$$

and consequently,

$$
0<e^{-|x|}\leq1
$$

This prevents the exponential from becoming extremely large.

---

### Step 3: Use `log1p`

```python
math.log1p(math.exp(-abs(x)))
```

`log1p(z)` computes

$$
\log(1+z)
$$

more accurately than directly evaluating `log(1 + z)` when $z$ is very small.

This is useful for large positive inputs where

$$
e^{-x}\rightarrow0
$$

---

### Step 4: Combine the Two Terms

The implementation evaluates

```python
max(0, x) + math.log1p(math.exp(-abs(x)))
```

which is mathematically equivalent to

$$
Softplus(x)=\max(0,x)+\log(1+e^{-|x|})
$$

For $x\geq0$:

$$
x+\log(1+e^{-x})
$$

For $x<0$:

$$
\log(1+e^x)
$$

Both cases are equivalent to the original Softplus definition.

---

### Step 5: Round the Result

```python
round(..., 4)
```

The result is rounded to four decimal places as required by the problem.

For example,

$$
Softplus(2)\approx2.126928
$$

becomes

```text
2.1269
```

---

### Why the Stable Formula Works

Start with the original definition:

$$
Softplus(x)=\log(1+e^x)
$$

For $x\geq0$, factor $e^x$:

$$
\log(1+e^x)=\log(e^x(1+e^{-x}))
$$

Using the logarithm product rule:

$$
\log(e^x)+\log(1+e^{-x})=x+\log(1+e^{-x})
$$

For $x<0$, the original expression is already numerically safe because $e^x<1$.

Therefore, the two cases can be combined into

$$
Softplus(x)=\max(0,x)+\log(1+e^{-|x|})
$$

This is the form used in the implementation.

---

## Time & Space Complexity

The function performs a constant number of arithmetic, logarithmic, and exponential operations.

Therefore, the time complexity is

$$
O(1)
$$

Only a constant number of scalar variables are used.

Therefore, the auxiliary space complexity is

$$
O(1)
$$

| Complexity | Value      |
| ---------- | ---------- |
| Time       | **$O(1)$** |
| Space      | **$O(1)$** |

The complexity is constant because the function operates on a single scalar input rather than an array of values.
