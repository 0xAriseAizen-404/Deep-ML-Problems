# Derivative of a Polynomial (Easy, Calculus)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Derivative of a Polynomial](#learn-derivative-of-a-polynomial)
- [Solution](#solution)
  - [Custom Implementation](#custom-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

---

## Problem Statement

### [Derivative of a Polynomial](https://www.deep-ml.com/problems/116)

Implement a function that computes the derivative of a polynomial term of the form `c * x^n` at a given point $x$.

The function should account for both the coefficient $c$ and exponent $n$ using the power rule.

For

$$
f(x)=cx^n
$$

the derivative is

$$
f'(x)=cnx^{n-1}
$$

The function should return the value of this derivative at the given point.

---

## Example

### Input

```python
poly_term_derivative(2.0, 3.0, 2.0)
```

### Output

```text
12.0
```

### Reasoning

The polynomial term is

$$
f(x)=2x^2
$$

Using the power rule:

$$
f'(x)=2\cdot2x^{2-1}
$$

Therefore:

$$
f'(x)=4x
$$

At $x=3$:

$$
f'(3)=4\cdot3=12
$$

Hence, the result is `12.0`.

---

## Learn: Derivative of a Polynomial

### What is it?

A **derivative** measures the instantaneous rate of change of a function at a particular point.

Geometrically, it represents the **slope of the tangent line** to the function at that point.

If the derivative is positive, the function is increasing at that point. If it is negative, the function is decreasing. If it is zero, the function is locally flat and may represent a minimum, maximum, or another stationary point.

Derivatives are fundamental in machine learning because optimization algorithms such as **Gradient Descent** use derivatives to determine how model parameters should change to reduce a loss function.

---

### Mathematical Definition

The derivative of a function $f(x)$ at a point $x$ can be defined using the limit:

$$
f'(x)=\lim_{h\to0}\frac{f(x+h)-f(x)}{h}
$$

Here:

- $x$ is the point where the derivative is evaluated.
- $h$ is a small step.
- $f(x+h)-f(x)$ measures the change in the function.
- Dividing by $h$ gives the average rate of change.
- Taking the limit as $h$ approaches zero gives the instantaneous rate of change.

The derivative therefore generalizes the familiar slope formula between two points.

---

### Power Rule

For a polynomial term

$$
f(x)=x^n
$$

the power rule states that

$$
f'(x)=nx^{n-1}
$$

For a term containing a coefficient:

$$
f(x)=cx^n
$$

the coefficient remains unchanged:

$$
f'(x)=cnx^{n-1}
$$

This is the rule directly used in this problem.

---

### Applying the Power Rule

Suppose:

$$
f(x)=5x^3
$$

Using the power rule:

$$
f'(x)=5\cdot3x^{3-1}
$$

Therefore:

$$
f'(x)=15x^2
$$

At $x=2$:

$$
f'(2)=15(2^2)=60
$$

The derivative tells us that the instantaneous slope of the function at $x=2$ is $60$.

---

### Constant Terms

A constant term has exponent zero:

$$
f(x)=c
$$

Since

$$
x^0=1
$$

the function does not depend on $x$.

Therefore, its derivative is zero:

$$
\frac{d}{dx}c=0
$$

Using the power rule:

$$
\frac{d}{dx}cx^0=c\cdot0\cdot x^{-1}=0
$$

Thus, when $n=0$, the derivative is zero.

---

### Derivative as Slope

Consider a simple function:

$$
f(x)=x^2
$$

Its derivative is:

$$
f'(x)=2x
$$

At $x=1$:

$$
f'(1)=2
$$

At $x=3$:

$$
f'(3)=6
$$

Therefore, the function becomes increasingly steep as $x$ increases.

For negative $x$, the derivative becomes negative, indicating that the function is decreasing.

---

### Characteristics / Key Points

- The derivative measures the instantaneous rate of change.
- The derivative represents the tangent slope at a point.
- The power rule differentiates polynomial powers efficiently.
- For $f(x)=cx^n$, the derivative is $cnx^{n-1}$.
- The coefficient $c$ is multiplied by the exponent.
- The exponent decreases by one after differentiation.
- The derivative of a constant is zero.
- A positive derivative indicates increasing behavior.
- A negative derivative indicates decreasing behavior.
- A zero derivative indicates a stationary point.
- Derivatives are the foundation of gradient-based optimization.
- Polynomial differentiation can be computed directly without numerical approximation.

---

### Why is it used? / Applications

Derivatives are fundamental throughout machine learning and mathematics.

Applications include:

- Gradient Descent.
- Backpropagation.
- Optimization of loss functions.
- Linear Regression.
- Logistic Regression.
- Neural Network training.
- Finding minima and maxima.
- Sensitivity analysis.
- Calculus-based optimization.

In machine learning, a model's parameters are updated according to the gradient of the loss:

$$
\theta=\theta-\alpha\nabla L(\theta)
$$

where:

- $\theta$ represents model parameters.
- $\alpha$ is the learning rate.
- $L(\theta)$ is the loss function.
- $\nabla L(\theta)$ is the gradient of the loss.

The derivative tells the optimizer which direction increases or decreases the objective.

---

> 💡 **Important Note**
>
> The derivative of a polynomial should not be confused with evaluating the original polynomial. For `c * x^n`, the derivative is `c * n * x^(n-1)`. The exponent decreases by one and the original exponent becomes a multiplier.

> 💡 **Interview Tip**
>
> Memorize the power rule: bring the exponent down as a multiplier, then reduce the exponent by one. For `c * x^n`, the result is `c * n * x^(n-1)`.

---

## Solution

### Custom Implementation

```python
def poly_term_derivative(c: float, x: float, n: float) -> float:
    return c * n * (x ** (n - 1))
```

---

## Code Explanation

### Step 1: Represent the Polynomial

The function represents the polynomial term:

$$
f(x)=cx^n
$$

The inputs `c`, `x`, and `n` represent the coefficient, evaluation point, and exponent respectively.

---

### Step 2: Apply the Power Rule

The derivative is:

$$
f'(x)=cnx^{n-1}
$$

The implementation directly translates this mathematical expression:

```python
c * n * (x ** (n - 1))
```

The exponent $n$ becomes a multiplier and the new exponent becomes $n-1$.

---

### Step 3: Evaluate at the Given Point

The expression is evaluated using the supplied value of `x`.

For example, with:

```python
c = 2.0
x = 3.0
n = 2.0
```

the function computes:

$$
2\cdot2\cdot3^{2-1}
$$

which gives:

$$
4\cdot3=12
$$

---

### Edge Case: Zero Exponent

When:

$$
n=0
$$

the derivative becomes:

$$
cnx^{n-1}=c\cdot0\cdot x^{-1}=0
$$

Therefore, the implementation naturally returns zero because the exponent $n$ itself is multiplied into the result.

---

### Algorithm

1. Multiply the coefficient $c$ by the exponent $n$.
2. Decrease the exponent from $n$ to $n-1$.
3. Evaluate $x^{n-1}$.
4. Multiply the two results.
5. Return the derivative value.

No iteration or numerical approximation is required.

---

## Time & Space Complexity

The implementation performs a constant number of arithmetic operations and one exponentiation operation.

| Complexity | Value      |
| ---------- | ---------- |
| Time       | **$O(1)$** |
| Space      | **$O(1)$** |

The algorithm uses constant auxiliary space because it only stores a few scalar values.
