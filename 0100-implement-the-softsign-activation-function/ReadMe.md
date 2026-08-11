# Implement the Softsign Activation Function (Easy, Deep Learning)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Understanding the Softsign Activation Function](#learn-understanding-the-softsign-activation-function)
- [Solution](#solution)
  - [Custom Implementation](#custom-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

---

## Problem Statement

### [Implement the Softsign Activation Function](https://www.deep-ml.com/problems/100)

Implement the **Softsign activation function**, a smooth non-linear activation function used in neural networks.

The function should take a numerical input $x$ and return the Softsign value rounded to **4 decimal places**.

The output must remain bounded between `-1` and `1`.

---

## Example

### Input

```python
softsign(1)
```

### Output

```text
0.5
```

### Reasoning

The Softsign function is

$$
{Softsign}(x)=\frac{x}{1+|x|}
$$

For $x=1$:

$$
{Softsign}(1)=\frac{1}{1+|1|}
$$

$$
{Softsign}(1)=\frac{1}{2}=0.5
$$

Therefore, the output is `0.5`.

---

## Learn: Understanding the Softsign Activation Function

### What is it?

**Softsign** is a smooth, non-linear activation function used in neural networks.

It transforms an input value into a bounded output between `-1` and `1`.

The function is similar to **tanh**, but its approach toward the limits is slower. This gives Softsign heavier tails and a more gradual transition for large positive and negative inputs.

The Softsign function is defined as

$$
f(x)=\frac{x}{1+|x|}
$$

where $x$ is the input and $|x|$ is its absolute value.

---

### Mathematical Definition

The Softsign activation function is

$$
{Softsign}(x)=\frac{x}{1+|x|}
$$

The absolute value makes the denominator positive for every input.

For positive values, $|x|=x$, so

$$
{Softsign}(x)=\frac{x}{1+x}
$$

For negative values, $|x|=-x$, so

$$
{Softsign}(x)=\frac{x}{1-x}
$$

This produces a smooth S-shaped activation function.

---

### Output Range

Softsign is bounded between `-1` and `1`.

For large positive values,

$$
\lim_{x\rightarrow\infty}\frac{x}{1+x}=1
$$

For large negative values,

$$
\lim_{x\rightarrow-\infty}\frac{x}{1-x}=-1
$$

Therefore,

$$
-1<{Softsign}(x)<1
$$

for every finite $x$.

Unlike a hard clipping operation, the function approaches `-1` and `1` asymptotically rather than reaching them for finite inputs.

---

### Behavior Around Zero

At $x=0$:

$$
{Softsign}(0)=\frac{0}{1+|0|}=0
$$

Therefore, Softsign is **zero-centered**.

For small values of $x$, the denominator is close to `1`, so

$$
{Softsign}(x)\approx x
$$

This means the function behaves approximately linearly around the origin.

---

### Symmetry

Softsign is an **odd function**.

This means

$$
f(-x)=-f(x)
$$

To see this, consider

$$
f(-x)=\frac{-x}{1+|-x|}
$$

Since

$$
|-x|=|x|
$$

we obtain

$$
f(-x)=\frac{-x}{1+|x|}=-f(x)
$$

Therefore, the function is symmetric around the origin.

---

### Derivative

For $x>0$:

$$
f(x)=\frac{x}{1+x}
$$

Using the quotient rule:

$$
f'(x)=\frac{1}{(1+x)^2}
$$

For $x<0$:

$$
f(x)=\frac{x}{1-x}
$$

Therefore,

$$
f'(x)=\frac{1}{(1-x)^2}
$$

These two cases can be combined into

$$
f'(x)=\frac{1}{(1+|x|)^2}
$$

This derivative is always positive.

At $x=0$:

$$
f'(0)=1
$$

Thus, Softsign has its largest gradient around the origin.

---

### Gradient Behavior

For large positive or negative values,

$$
|x|\rightarrow\infty
$$

and therefore

$$
f'(x)=\frac{1}{(1+|x|)^2}\rightarrow0
$$

The gradient approaches zero gradually.

Compared with sigmoid and tanh, Softsign approaches its saturation regions more slowly.

For example, its derivative decreases polynomially rather than exponentially.

---

### Softsign vs Tanh

Softsign and tanh have similar shapes and both produce zero-centered outputs.

| Property          | Softsign     | Tanh     |     |            |
| ----------------- | ------------ | -------- | --- | ---------- |
| Formula           | $\frac{x}{1+ | x        | }$  | $\tanh(x)$ |
| Output Range      | $(-1,1)$     | $(-1,1)$ |     |            |
| Zero-Centered     | Yes          | Yes      |     |            |
| Smooth            | Yes          | Yes      |     |            |
| Differentiable    | Yes          | Yes      |     |            |
| Tail Behavior     | Slower       | Faster   |     |            |
| Derivative at $0$ | $1$          | $1$      |     |            |

The important difference is their behavior in the tails.

Softsign approaches its limits more gradually, while tanh saturates more quickly.

---

### Softsign vs ReLU

ReLU is defined as

$$
{ReLU}(x)=\max(0,x)
$$

Softsign is

$$
{Softsign}(x)=\frac{x}{1+|x|}
$$

Their main differences are:

| Property        | ReLU         | Softsign        |
| --------------- | ------------ | --------------- |
| Output Range    | $[0,\infty)$ | $(-1,1)$        |
| Zero-Centered   | No           | Yes             |
| Smooth          | No           | Yes             |
| Negative Inputs | $0$          | Negative values |
| Bounded         | No           | Yes             |

Softsign is useful when bounded and zero-centered activations are desirable.

---

### Characteristics / Key Points

- Softsign is a smooth non-linear activation function.
- Its formula is $\frac{x}{1+|x|}$.
- Its output lies strictly between `-1` and `1`.
- It is continuous everywhere.
- It is differentiable everywhere.
- It is zero-centered.
- It is an odd function.
- It behaves approximately linearly around zero.
- Its derivative is $\frac{1}{(1+|x|)^2}$.
- Its gradient approaches zero gradually for large inputs.
- It approaches `1` for large positive inputs.
- It approaches `-1` for large negative inputs.
- It has heavier tails than tanh.

---

### Important Values

Some useful values are:

$$
{Softsign}(-2)=\frac{-2}{3}\approx-0.6667
$$

$$
{Softsign}(0)=0
$$

$$
{Softsign}(1)=\frac{1}{2}=0.5
$$

$$
{Softsign}(2)=\frac{2}{3}\approx0.6667
$$

As the magnitude of $x$ increases, the output moves closer to its corresponding asymptotic limit.

---

### Why is it used? / Applications

Softsign can be useful in:

- Neural network activation functions.
- Models requiring bounded outputs.
- Zero-centered neural network representations.
- Situations where smooth activation is preferred.
- Models where slower saturation is desirable.
- Experimental deep learning architectures.

Although ReLU and its variants are more commonly used in modern deep networks, Softsign remains an important activation function for understanding smooth and bounded nonlinearities.

---

> 💡 **Important Note**
>
> Softsign does not actually reach `-1` or `1` for any finite input. These values are approached asymptotically as $x$ approaches negative or positive infinity.

> 💡 **Interview Tip**
>
> Remember the formula ${Softsign}(x)=\frac{x}{1+|x|}$ and its derivative $f'(x)=\frac{1}{(1+|x|)^2}$. Softsign is **zero-centered, smooth, bounded, and has slower saturation than tanh**.

---

## Solution

### Custom Implementation

```python
def softsign(x: float) -> float:
    return round(x / (1 + abs(x)), 4)
```

---

## Code Explanation

### Step 1: Calculate the Absolute Value

```python
abs(x)
```

The Softsign denominator contains $|x|$.

This ensures that the denominator is always positive:

$$
1+|x|>0
$$

---

### Step 2: Apply the Softsign Formula

```python
x / (1 + abs(x))
```

This directly implements

$$
{Softsign}(x)=\frac{x}{1+|x|}
$$

For example, when $x=1$:

$$
\frac{1}{1+|1|}=\frac{1}{2}=0.5
$$

When $x=-1$:

$$
\frac{-1}{1+|-1|}=\frac{-1}{2}=-0.5
$$

---

### Step 3: Round the Result

```python
round(..., 4)
```

The result is rounded to four decimal places as required by the problem.

For example:

$$
\frac{2}{3}=0.666666\ldots
$$

becomes

```text
0.6667
```

---

### Why the Output Stays Bounded

For positive $x$:

$$
0<\frac{x}{1+x}<1
$$

For negative $x$:

$$
-1<\frac{x}{1-x}<0
$$

Therefore, for every finite input,

$$
-1<{Softsign}(x)<1
$$

The denominator grows with the magnitude of the input, preventing the output from growing without bound.

---

## Time & Space Complexity

The function performs a constant number of arithmetic operations.

Therefore, the time complexity is

$$
O(1)
$$

Only the input value and a constant number of intermediate values are stored.

Therefore, the auxiliary space complexity is
w
$$
O(1)
$$

| Complexity | Value      |
| ---------- | ---------- |
| Time       | **$O(1)$** |
| Space      | **$O(1)$** |

The complexity is constant because the function processes a single scalar input.
