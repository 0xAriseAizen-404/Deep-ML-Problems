# Implementation of Log Softmax Function (Easy, Deep Learning)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Log Softmax Function](#learn-log-softmax-function)
- [Solutions](#solutions)
  - [NumPy Implementation](#numpy-implementation)
- [Code Explanation](#code-explanation)

## Problem Statement

[Implementation of Log Softmax Function](https://www.deep-ml.com/problems/39)

Implement a Python function that computes the **log-softmax** of a 1D array of scores. Use a numerically stable implementation to avoid overflow when exponentiating large values.

## Example

```python
A = np.array([1, 2, 3])

print(log_softmax(A))

# Output
array([-2.4076, -1.4076, -0.4076])
```

## Learn: Log Softmax Function

The **log-softmax** function converts logits into **log-probabilities**.

**Formula**

$$
\log(\text{softmax}(x_i)) =
x_i - \log\left(\sum_j e^{x_j}\right)
$$

For numerical stability:

$$
\log(\text{softmax}(x_i)) =
x_i - \left(m + \log\left(\sum_j e^{x_j-m}\right)\right)
$$

where

$$
m = \max(x)
$$

Subtracting the maximum value prevents overflow during exponentiation.

## Solutions

### NumPy Implementation

```python
import numpy as np

def log_softmax(scores: list) -> np.ndarray:
    scores = np.asarray(scores)
    mx = np.max(scores)
    log_sum_exp = mx + np.log(np.sum(np.exp(scores - mx)))
    return scores - log_sum_exp
```

## Code Explanation

The function:

1. Converts the input into a NumPy array.
2. Finds the maximum score for numerical stability.
3. Computes the **log-sum-exp** term.
4. Returns `scores - log_sum_exp`, which is the log-softmax of each element.