# Implement Adam Optimization Algorithm (Medium, Deep Learning)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Understanding the Adam Optimization Algorithm](#learn-understanding-the-adam-optimization-algorithm)
  - [What is it?](#what-is-it)
  - [Gradient Descent](#gradient-descent)
  - [Momentum](#momentum)
  - [RMSProp](#rmsprop)
  - [Adam](#adam)
  - [Mathematical Definition](#mathematical-definition)
  - [First Moment Estimate](#first-moment-estimate)
  - [Second Moment Estimate](#second-moment-estimate)
  - [Bias Correction](#bias-correction)
  - [Parameter Update](#parameter-update)
  - [Step-by-Step Example](#step-by-step-example)
  - [Characteristics / Key Points](#characteristics--key-points)
  - [Why is it used? / Applications](#why-is-it-used--applications)
- [Solutions](#solutions)
  - [Custom Implementation](#custom-implementation)
  - [NumPy / PyTorch Equivalent](#numpy--pytorch-equivalent)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

---

## Problem Statement

### [Implement Adam Optimization Algorithm](https://www.deep-ml.com/problems/49)

Implement the **Adam (Adaptive Moment Estimation)** optimization algorithm in Python.

The function `adam_optimizer` should update a set of parameters to minimize an objective function using the gradient supplied by a gradient function.

The function receives:

- `f`: The objective function.
- `grad`: A function that computes the gradient.
- `x0`: Initial parameter values.
- `learning_rate`: Step size, with default value `0.001`.
- `beta1`: Exponential decay rate for the first moment, with default value `0.9`.
- `beta2`: Exponential decay rate for the second moment, with default value `0.999`.
- `epsilon`: Small constant for numerical stability, with default value `1e-8`.
- `num_iterations`: Number of optimization steps, with default value `10`.

The function should return the optimized parameter values after performing the specified number of Adam updates.

Adam maintains two running statistics of the gradients:

- The **first moment**, which tracks the mean of gradients.
- The **second moment**, which tracks the mean of squared gradients.

It then applies bias correction before updating the parameters.

---

## Example

### Input

```python
import numpy as np

def objective_function(x):
    return x[0]**2 + x[1]**2

def gradient(x):
    return np.array([
        2*x[0],
        2*x[1]
    ])

x0 = np.array([1.0, 1.0])

x_opt = adam_optimizer(
    objective_function,
    gradient,
    x0
)

print("Optimized parameters:", x_opt)
```

### Output

```text
Optimized parameters: [0.99000325 0.99000325]
```

### Reasoning

The objective function is

$$
f(x) = x_1^2 + x_2^2
$$

Its gradient is

$$
\nabla f(x) =
\begin{bmatrix}
2x_1\
2x_2
\end{bmatrix}
$$

Starting from

$$
x_0 =
\begin{bmatrix}
1\
1
\end{bmatrix}
$$

Adam repeatedly computes the gradient, updates its first and second moment estimates, corrects their initialization bias, and moves the parameters in the direction that reduces the objective function.

The minimum of this objective function occurs at

$$
x^* =
\begin{bmatrix}
0\
0
\end{bmatrix}
$$

After only ten iterations with the default learning rate, the parameters have moved slightly toward this optimum.

---

## Learn: Understanding the Adam Optimization Algorithm

### What is it?

**Adam (Adaptive Moment Estimation)** is a gradient-based optimization algorithm widely used for training neural networks and other differentiable models.

It combines ideas from **Momentum** and **RMSProp**.

Adam maintains two exponentially decaying averages:

1. The average of past gradients.
2. The average of past squared gradients.

The first estimate provides a momentum-like direction, while the second estimate provides an adaptive scaling for each parameter.

The resulting update allows different parameters to effectively use different step sizes.

The main Adam update can be summarized as

$$
\theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
$$

where

- $\theta_t$ is the updated parameter vector.
- $\alpha$ is the learning rate.
- $\hat{m}_t$ is the bias-corrected first moment.
- $\hat{v}_t$ is the bias-corrected second moment.
- $\epsilon$ prevents division by zero.

---

## Gradient Descent

Before understanding Adam, it is useful to understand standard gradient descent.

Suppose we want to minimize

$$
f(\theta)
$$

The gradient

$$
\nabla_\theta f(\theta)
$$

points in the direction of greatest increase of the function.

Therefore, to minimize the function, gradient descent moves in the opposite direction.

The standard update is

$$
\theta_t = \theta_{t-1} - \alpha \nabla_\theta f(\theta_{t-1})
$$

where $\alpha$ is the learning rate.

The same learning rate is applied to every parameter.

This can become inefficient when different parameters have gradients with very different scales.

Adam addresses this by adapting the update using historical gradient information.

---

## Momentum

Momentum keeps an exponentially weighted average of previous gradients.

Instead of using only the current gradient, momentum maintains

$$
m_t = \beta_1m_{t-1} + (1-\beta_1)g_t
$$

where

- $g_t$ is the current gradient.
- $m_t$ is the first moment estimate.
- $\beta_1$ controls how much previous gradients influence the estimate.

The resulting update uses the accumulated direction rather than the raw gradient alone.

This can reduce oscillations and help optimization move consistently along useful directions.

Adam incorporates this idea through its first moment estimate.

---

## RMSProp

RMSProp maintains an exponentially weighted average of squared gradients.

The second moment estimate is

$$
v_t = \beta_2v_{t-1} + (1-\beta_2)g_t^2
$$

where the square is applied element-wise.

The update is scaled according to the magnitude of the historical gradients.

Parameters with consistently large gradients receive smaller effective updates, while parameters with smaller gradients can receive relatively larger updates.

Adam incorporates this adaptive scaling through its second moment estimate.

---

## Adam

Adam combines the two ideas.

The first moment tracks the average direction of the gradient:

$$
m_t = \beta_1m_{t-1} + (1-\beta_1)g_t
$$

The second moment tracks the average magnitude of the squared gradient:

$$
v_t = \beta_2v_{t-1} + (1-\beta_2)g_t^2
$$

The raw moment estimates are then corrected for their initialization at zero.

Finally, the parameters are updated using the corrected moments.

The complete process is

$$
g_t = \nabla_\theta f(\theta_{t-1})
$$

$$
m_t = \beta_1m_{t-1} + (1-\beta_1)g_t
$$

$$
v_t = \beta_2v_{t-1} + (1-\beta_2)g_t^2
$$

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}
$$

$$
\hat{v}_t = \frac{v_t}{1-\beta_2^t}
$$

$$
\theta_t = \theta_{t-1} - \alpha\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
$$

---

## Mathematical Definition

Let

$$
g_t = \nabla_\theta f(\theta_{t-1})
$$

be the gradient of the objective function at iteration $t$.

Adam initializes

$$
m_0 = 0
$$

and

$$
v_0 = 0
$$

The first moment estimate is

$$
m_t = \beta_1m_{t-1} + (1-\beta_1)g_t
$$

The second moment estimate is

$$
v_t = \beta_2v_{t-1} + (1-\beta_2)g_t^2
$$

Because both moment estimates start at zero, they are biased toward zero during the first few iterations.

Adam corrects this bias using

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}
$$

and

$$
\hat{v}_t = \frac{v_t}{1-\beta_2^t}
$$

The final parameter update is

$$
\theta_t = \theta_{t-1} - \alpha\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
$$

All vector operations are applied element-wise where appropriate.

---

## First Moment Estimate

The first moment estimate is the exponentially weighted average of the gradients.

$$
m_t = \beta_1m_{t-1} + (1-\beta_1)g_t
$$

With the commonly used value

$$
\beta_1 = 0.9
$$

the previous estimate receives `90%` of the weight, while the current gradient receives `10%`.

Expanding the recurrence shows that older gradients continue to influence the estimate, but their influence decays exponentially.

The first moment therefore behaves similarly to momentum.

It provides a smoother estimate of the direction in which the parameters should move.

---

## Second Moment Estimate

The second moment estimate tracks the exponentially weighted average of squared gradients.

$$
v_t = \beta_2v_{t-1} + (1-\beta_2)g_t^2
$$

The square is applied element-wise.

With the common value

$$
\beta_2 = 0.999
$$

the second moment changes slowly and retains information about historical gradient magnitudes.

The second moment is used to scale the update for each parameter.

---

## Adaptive Learning Rate

Adam divides the first moment by the square root of the second moment.

$$
\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
$$

This normalization makes the effective update depend on the historical magnitude of the gradient.

If a parameter consistently receives large gradients, its denominator tends to become larger.

If a parameter receives smaller gradients, its denominator tends to be smaller.

Therefore, Adam provides a parameter-wise adaptive step size.

The global learning rate $\alpha$ still controls the overall scale of the updates.

---

## Bias Correction

The first and second moment estimates are initialized as zero.

For example,

$$
m_0 = 0
$$

and

$$
v_0 = 0
$$

During the first few iterations, these estimates are biased toward zero.

Adam corrects this using

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}
$$

and

$$
\hat{v}_t = \frac{v_t}{1-\beta_2^t}
$$

At early iterations, the correction factors are significant.

As $t$ increases,

$$
\beta_1^t \rightarrow 0
$$

and

$$
\beta_2^t \rightarrow 0
$$

so the correction factors approach `1`.

Consequently, the effect of bias correction becomes smaller later in optimization.

---

## Why Bias Correction Is Necessary

Consider the first moment at the first iteration.

Since

$$
m_0 = 0
$$

we have

$$
m_1 = (1-\beta_1)g_1
$$

For $\beta_1 = 0.9$,

$$
m_1 = 0.1g_1
$$

The raw moment is therefore much smaller than the current gradient.

The bias-corrected estimate is

$$
\hat{m}_1 = \frac{m_1}{1-\beta_1}
$$

which gives

$$
\hat{m}_1 = g_1
$$

This correction prevents the initialization at zero from excessively shrinking the early updates.

---

## Role of Epsilon

The Adam update contains

$$
\sqrt{\hat{v}_t}+\epsilon
$$

where $\epsilon$ is a very small positive constant.

The default value commonly used is

$$
\epsilon = 10^{-8}
$$

Its primary purpose is numerical stability.

Without $\epsilon$, the denominator could potentially become zero or extremely close to zero.

The addition of $\epsilon$ prevents an unstable division.

It does not serve as a meaningful learning-rate term.

---

## Hyperparameters

Adam commonly uses the following default values.

| Hyperparameter | Default | Purpose                      |
| -------------- | ------- | ---------------------------- |
| $\alpha$       | `0.001` | Overall learning rate        |
| $\beta_1$      | `0.9`   | Decay rate for first moment  |
| $\beta_2$      | `0.999` | Decay rate for second moment |
| $\epsilon$     | `1e-8`  | Numerical stability          |

The learning rate is usually the most important hyperparameter to tune.

The default values of $\beta_1$ and $\beta_2$ work well across many problems.

---

## Step-by-Step Example

Consider

$$
f(x) = x^2
$$

with initial parameter

$$
x_0 = 1
$$

The gradient is

$$
g(x) = 2x
$$

Assume

$$
\alpha = 0.001
$$

$$
\beta_1 = 0.9
$$

$$
\beta_2 = 0.999
$$

and

$$
\epsilon = 10^{-8}
$$

---

### Step 1: Compute the Gradient

At the first iteration,

$$
g_1 = 2x_0
$$

Since

$$
x_0 = 1
$$

we obtain

$$
g_1 = 2
$$

---

### Step 2: Update the First Moment

Initially,

$$
m_0 = 0
$$

Therefore,

$$
m_1 = 0.9(0)+0.1(2)
$$

which gives

$$
m_1 = 0.2
$$

---

### Step 3: Update the Second Moment

Initially,

$$
v_0 = 0
$$

Therefore,

$$
v_1 = 0.999(0)+0.001(2^2)
$$

which gives

$$
v_1 = 0.004
$$

---

### Step 4: Apply Bias Correction

The first moment is corrected using

$$
\hat{m}_1 = \frac{0.2}{1-0.9^1}
$$

Therefore,

$$
\hat{m}_1 = 2
$$

The second moment is corrected using

$$
\hat{v}_1 = \frac{0.004}{1-0.999^1}
$$

Therefore,

$$
\hat{v}_1 = 4
$$

---

### Step 5: Update the Parameter

The Adam update is

$$
x_1 = x_0 - \alpha\frac{\hat{m}_1}{\sqrt{\hat{v}_1}+\epsilon}
$$

Substituting the values gives

$$
x_1 = 1 - 0.001\frac{2}{2+10^{-8}}
$$

Therefore, approximately

$$
x_1 \approx 0.999
$$

The parameter moves toward the minimum at

$$
x^* = 0
$$

---

## Adam Update Intuition

The Adam update can be viewed as three stages.

```text
Gradient
   │
   ▼
First Moment ──────► Direction
   │
   │
   ▼
Second Moment ─────► Scale
   │
   ▼
Bias Correction
   │
   ▼
Parameter Update
```

The first moment determines the smoothed direction.

The second moment determines how strongly the update should be scaled.

Bias correction compensates for the zero initialization of the moments.

---

## Adam vs Standard Gradient Descent

Standard gradient descent uses

$$
\theta_t = \theta_{t-1}-\alpha g_t
$$

Adam uses

$$
\theta_t = \theta_{t-1}-\alpha\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
$$

The difference is that Adam uses historical gradient information to modify the update.

Gradient descent has one global learning rate.

Adam effectively produces parameter-specific scaling through the second moment.

---

## Adam vs Momentum

Momentum maintains a moving average of gradients.

$$
m_t = \beta_1m_{t-1}+(1-\beta_1)g_t
$$

Adam also maintains this first moment.

However, Adam additionally tracks

$$
v_t = \beta_2v_{t-1}+(1-\beta_2)g_t^2
$$

The second moment allows Adam to adapt the update magnitude separately for different parameters.

Thus, Adam can be viewed as combining momentum-like behavior with adaptive gradient scaling.

---

## Adam vs RMSProp

RMSProp maintains an exponential average of squared gradients.

$$
v_t = \beta_2v_{t-1}+(1-\beta_2)g_t^2
$$

Adam also uses this second moment.

However, Adam additionally maintains the first moment

$$
m_t = \beta_1m_{t-1}+(1-\beta_1)g_t
$$

and applies explicit bias correction.

Therefore, Adam combines the main ideas of momentum and RMSProp into a single optimizer.

---

## Characteristics / Key Points

- Adam stands for **Adaptive Moment Estimation**.
- It is a first-order gradient-based optimizer.
- It does not require second-order derivatives.
- It maintains a first moment of the gradients.
- It maintains a second moment of the squared gradients.
- The first moment behaves similarly to momentum.
- The second moment provides adaptive parameter-wise scaling.
- Adam uses bias correction because moment estimates start at zero.
- The learning rate controls the overall update magnitude.
- $\beta_1$ controls first-moment decay.
- $\beta_2$ controls second-moment decay.
- $\epsilon$ provides numerical stability.
- Adam can handle parameters with different gradient magnitudes.
- Adam often converges quickly in the early stages of training.
- Adam is widely used for deep neural networks.
- The objective function itself does not need to be evaluated during a step if the gradient is supplied separately.
- The gradient function must provide derivatives with respect to the current parameters.
- The same update rule can operate on scalar, vector, or tensor parameters.
- The implementation performs element-wise operations for the squared gradient and adaptive scaling.
- Adam is not guaranteed to outperform every other optimizer on every problem.
- Learning-rate selection can still have a significant effect on final performance.

---

## Why is it used? / Applications

Adam is widely used for optimizing differentiable machine learning models.

Common applications include

- Neural Network Training
- Image Classification
- Object Detection
- Natural Language Processing
- Recurrent Neural Networks
- Transformers
- Generative Models
- Autoencoders
- Computer Vision
- Speech Processing
- Representation Learning
- Large-Scale Deep Learning

Adam is particularly common in deep learning because it generally requires little manual tuning of the basic optimizer hyperparameters.

---

> 💡 **Important Note**
>
> Adam's adaptive learning rate does not mean that the global `learning_rate` becomes irrelevant. The learning rate $\alpha$ still controls the overall scale of every update and is often the most important Adam hyperparameter to tune.

---

> 💡 **Interview Tip**
>
> Remember the three main ideas behind Adam: **first moment for momentum, second moment for adaptive scaling, and bias correction for zero initialization**. If you understand these three components, the complete Adam update becomes much easier to derive.

---

> 💡 **Common Mistake**
>
> Do not forget the square root in the denominator. Adam uses $\sqrt{\hat{v}_t}$, not $\hat{v}_t$ directly. Also, the small $\epsilon$ term is added after taking the square root.

---

## Solutions

### Custom Implementation

```python
import numpy as np

def adam_optimizer(
    f,
    grad,
    x0,
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    num_iterations=10
):
    first_moment = np.zeros_like(
        x0,
        dtype=float
    )

    second_moment = np.zeros_like(
        x0,
        dtype=float
    )

    parameters = np.array(
        x0,
        dtype=float,
        copy=True
    )

    for t in range(
        1,
        num_iterations + 1
    ):
        gradient = grad(parameters)

        first_moment = (
            beta1 * first_moment
            + (1 - beta1) * gradient
        )

        second_moment = (
            beta2 * second_moment
            + (1 - beta2) * gradient**2
        )

        bias_first_moment = (
            first_moment
            / (1 - beta1**t)
        )

        bias_second_moment = (
            second_moment
            / (1 - beta2**t)
        )

        parameters = parameters - (
            learning_rate
            * bias_first_moment
            / (
                np.sqrt(bias_second_moment)
                + epsilon
            )
        )

    return parameters
```

### NumPy / PyTorch Equivalent

In PyTorch, Adam is available as a built-in optimizer.

```python
import torch

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8
)
```

A standard training step is

```python
optimizer.zero_grad()

output = model(x)

loss = loss_function(
    output,
    target
)

loss.backward()

optimizer.step()
```

The `optimizer.step()` operation performs the parameter update using Adam's internal moment estimates.

The custom implementation in this problem exposes those internal calculations explicitly.

---

## Code Explanation

### Step 1: Initialize the First Moment

The first moment is initialized to zeros.

```python
first_moment = np.zeros_like(
    x0,
    dtype=float
)
```

This represents

$$
m_0 = 0
$$

It has the same shape as the parameter vector.

---

### Step 2: Initialize the Second Moment

The second moment is also initialized to zeros.

```python
second_moment = np.zeros_like(
    x0,
    dtype=float
)
```

This represents

$$
v_0 = 0
$$

It also has the same shape as the parameters.

---

### Step 3: Copy the Initial Parameters

The initial parameter values are copied into a floating-point NumPy array.

```python
parameters = np.array(
    x0,
    dtype=float,
    copy=True
)
```

This ensures that the original `x0` is not modified directly.

The current parameter vector represents

$$
\theta_0
$$

---

### Step 4: Iterate Through Optimization Steps

The optimizer performs the requested number of iterations.

```python
for t in range(
    1,
    num_iterations + 1
):
```

The iteration counter starts at `1` because the Adam bias-correction terms use

$$
\beta_1^t
$$

and

$$
\beta_2^t
$$

---

### Step 5: Compute the Gradient

The gradient function is evaluated at the current parameters.

```python
gradient = grad(parameters)
```

This corresponds to

$$
g_t = \nabla_\theta f(\theta_{t-1})
$$

The objective function `f` is accepted by the interface but is not directly evaluated because the gradient function already supplies the required derivative.

---

### Step 6: Update the First Moment

The first moment is updated using

```python
first_moment = (
    beta1 * first_moment
    + (1 - beta1) * gradient
)
```

This implements

$$
m_t = \beta_1m_{t-1}+(1-\beta_1)g_t
$$

The previous first moment and current gradient are combined using exponential decay.

---

### Step 7: Update the Second Moment

The squared gradient is used to update the second moment.

```python
second_moment = (
    beta2 * second_moment
    + (1 - beta2) * gradient**2
)
```

This implements

$$
v_t = \beta_2v_{t-1}+(1-\beta_2)g_t^2
$$

The `**2` operation is element-wise.

---

### Step 8: Correct the First Moment Bias

The first moment estimate is corrected using

```python
bias_first_moment = (
    first_moment
    / (1 - beta1**t)
)
```

This implements

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}
$$

The correction is especially important during the first few iterations.

---

### Step 9: Correct the Second Moment Bias

The second moment is corrected using

```python
bias_second_moment = (
    second_moment
    / (1 - beta2**t)
)
```

This implements

$$
\hat{v}_t = \frac{v_t}{1-\beta_2^t}
$$

Now both moment estimates are corrected for their zero initialization.

---

### Step 10: Update the Parameters

The final Adam update is

```python
parameters = parameters - (
    learning_rate
    * bias_first_moment
    / (
        np.sqrt(bias_second_moment)
        + epsilon
    )
)
```

This implements

$$
\theta_t = \theta_{t-1} - \alpha\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
$$

The update moves the parameters opposite to the estimated gradient direction.

---

### Step 11: Repeat

The updated parameters are used to compute the gradient during the next iteration.

The complete flow is

```text
θt-1
 │
 ▼
Compute gradient gt
 │
 ▼
Update first moment mt
 │
 ▼
Update second moment vt
 │
 ▼
Bias correction
 │
 ▼
Compute Adam update
 │
 ▼
θt
 │
 └──────────► repeat
```

This continues until `num_iterations` has been completed.

---

### Why `first_moment` and `second_moment` Have the Same Shape as Parameters

Suppose

```text
parameters = [θ1, θ2, θ3]
```

Then the first moment is

```text
m = [m1, m2, m3]
```

and the second moment is

```text
v = [v1, v2, v3]
```

Each parameter maintains its own historical gradient statistics.

Therefore, Adam can adapt the update separately for every parameter.

---

### Element-Wise Adaptive Scaling

The update contains

```python
bias_first_moment / (
    np.sqrt(bias_second_moment)
    + epsilon
)
```

This operation is element-wise.

For parameter $\theta_i$,

$$
\theta_{t,i} = \theta_{t-1,i} - \alpha\frac{\hat{m}*{t,i}}{\sqrt{\hat{v}*{t,i}}+\epsilon}
$$

Thus, every parameter receives its own normalized update.

This is the central reason Adam is called an **adaptive** optimizer.

---

### Role of the Objective Function

The function accepts

```python
f
```

but does not evaluate it.

The implementation uses

```python
gradient = grad(parameters)
```

to obtain the derivative directly.

This is sufficient for the Adam update because Adam needs the gradient, not the objective value itself.

In an autograd-based implementation, the optimizer may instead receive gradients generated from the objective function automatically.

---

### Complete Adam Algorithm

The implementation follows the standard Adam algorithm:

```text
ADAM(θ0, α, β1, β2, ε, T)

1. m ← 0
2. v ← 0
3. θ ← θ0

4. for t = 1 to T
5.     g ← gradient(θ)
6.     m ← β1 m + (1 - β1) g
7.     v ← β2 v + (1 - β2) g²
8.     m̂ ← m / (1 - β1^t)
9.     v̂ ← v / (1 - β2^t)
10.    θ ← θ - α m̂ / (sqrt(v̂) + ε)
11. return θ
```

The custom implementation is a direct numerical translation of this algorithm.

---

## Time & Space Complexity

Let

- $T$ be the number of optimization iterations.
- $n$ be the number of parameters.
- $f$ be the objective function.
- $g$ be the gradient function.

At every iteration, the implementation computes the gradient and performs element-wise operations over all $n$ parameters.

Assuming the gradient computation costs $O(n)$, each Adam update costs

$$
O(n)
$$

Therefore, over $T$ iterations, the total optimizer complexity is

$$
O(Tn)
$$

This excludes any additional computational cost required internally by a complex gradient function.

The optimizer maintains:

- The parameter vector.
- The first moment vector.
- The second moment vector.
- Temporary gradient and corrected moment arrays.

Each has size proportional to $n$.

Therefore, the auxiliary space complexity is

$$
O(n)
$$

| Complexity | Value     |
| ---------- | --------- |
| Time       | **O(Tn)** |
| Space      | **O(n)**  |

where

- $T$ is the number of optimization iterations.
- $n$ is the number of parameters.

The dominant memory requirement comes from storing the first and second moment estimates, which have the same shape as the parameter vector.
