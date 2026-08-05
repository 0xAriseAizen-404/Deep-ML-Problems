# Implementation of Log Softmax Function (Easy, Deep Learning)

## Table of Contents

- Problem Statement
- Example
- Learn: Understanding Log Softmax Function
- Solution
- Code Explanation
- Time & Space Complexity

---

## Problem Statement

### [Implementation of Log Softmax Function](https://www.deep-ml.com/problems/39)

Write a Python function that computes the **Log Softmax** of a given vector of scores (logits).

The function should:

- Accept a 1-D NumPy array of scores.
- Compute the log-softmax values using a numerically stable approach.
- Return the transformed values as a NumPy array.

Log Softmax is widely used in deep learning because it is more numerically stable than computing Softmax first and then taking the logarithm.

---

## Example

### Input

```python
A = np.array([1, 2, 3])

print(log_softmax(A))
```

### Output

```python
array([-2.4076, -1.4076, -0.4076])
```

### Reasoning

First compute the softmax probabilities

$$
\text{Softmax}(A) =
[0.0900,\;0.2447,\;0.6652]
$$

Taking the natural logarithm of each probability gives

$$
\log(\text{Softmax}(A)) =
[-2.4076,\;-1.4076,\;-0.4076]
$$

---

## Learn: Understanding Log Softmax Function

### What is it?

The **Softmax** function converts a vector of arbitrary real-valued scores (called **logits**) into a probability distribution.

Every probability satisfies

- It lies between **0** and **1**.
- The probabilities sum to **1**.

The **Log Softmax** function simply computes the logarithm of these probabilities.

Instead of evaluating

$$
\log(\text{Softmax}(x))
$$

directly, we compute an equivalent but numerically stable expression that avoids overflow when the input contains very large values.

This stable formulation is used internally by nearly every deep learning framework.

---

### Mathematical Definition

Given a vector

$$
x = [x_1,\;x_2,\;\ldots,\;x_n]
$$

the Softmax function is

$$
\text{Softmax}(x_i) =
\frac{e^{x_i}}
{\sum_{j=1}^{n} e^{x_j}}
$$

Taking the logarithm,

$$
\log(\text{Softmax}(x_i)) =
\log\left(
\frac{e^{x_i}}
{\sum_{j=1}^{n} e^{x_j}}
\right)
$$

Using logarithm properties,

$$
\log(\text{Softmax}(x_i)) =
x_i -
\log\left(
\sum_{j=1}^{n} e^{x_j}
\right)
$$

To improve numerical stability, subtract the maximum score before exponentiation.

Let

$$
m = \max(x)
$$

Then

$$
\log(\text{Softmax}(x_i)) =
x_i -
\left(
m +
\log\left(
\sum_{j=1}^{n}
e^{x_j-m}
\right)
\right)
$$

This expression is known as the **Log-Sum-Exp Trick**.

---

### Why is Numerical Stability Important?

Suppose the input scores are

```text
[1000, 1001, 1002]
```

Computing

$$
e^{1002}
$$

causes floating-point overflow.

Instead, subtract the maximum value first.

```text
[1000, 1001, 1002]

↓

[-2, -1, 0]
```

Now,

$$
e^{-2},\;
e^{-1},\;
e^{0}
$$

are all safe to compute.

Since every value is shifted by the same constant, the final probabilities remain unchanged.

---

### Log-Sum-Exp Trick

The quantity

$$
\log\left(
\sum_{i=1}^{n}
e^{x_i}
\right)
$$

appears frequently in machine learning.

Instead of computing it directly, use

$$
\log\left(
\sum_{i=1}^{n}
e^{x_i}
\right) =
m +
\log\left(
\sum_{i=1}^{n}
e^{x_i-m}
\right)
$$

where

$$
m = \max(x)
$$

This prevents overflow while producing exactly the same result.

---

### Example Calculation

Given

$$
x = [1,\;2,\;3]
$$

Subtract the maximum value.

$$
x - 3 =
[-2,\;-1,\;0]
$$

Exponentiate.

$$
e^{x-3} =
[0.1353,\;0.3679,\;1]
$$

Compute the normalization constant.

$$
0.1353 + 0.3679 + 1 =
1.5032
$$

Softmax becomes

$$
\frac{[0.1353,\;0.3679,\;1]}{1.5032} =
[0.0900,\;0.2447,\;0.6652]
$$

Taking the logarithm,

$$
[-2.4076,\;-1.4076,\;-0.4076]
$$

---

### Characteristics / Key Points

- Produces log probabilities instead of probabilities.
- More numerically stable than `log(softmax(x))`.
- Uses the Log-Sum-Exp trick.
- Output values are always less than or equal to zero.
- Larger logits produce larger log probabilities.
- Commonly paired with Cross-Entropy Loss.
- Used in nearly every deep learning classification model.

---

### Why is it used? / Applications

Log Softmax is fundamental in modern deep learning.

Applications include

- Multi-class Classification
- Neural Networks
- Natural Language Processing (NLP)
- Image Classification
- Sequence Models
- Transformers
- Speech Recognition
- Language Modeling
- Cross-Entropy Loss

Frameworks such as **PyTorch** and **TensorFlow** implement optimized versions of Log Softmax for stable training.

---

> 💡 **Important Note**
>
> In PyTorch, `CrossEntropyLoss` **already combines** `LogSoftmax` and `Negative Log-Likelihood (NLL) Loss` into a single numerically stable operation. Therefore, you should pass **raw logits** to `CrossEntropyLoss` instead of applying `softmax()` or `log_softmax()` manually.

---

## Solution

### Custom Implementation

```python
import numpy as np

def log_softmax(scores):
    scores = np.asarray(scores)

    mx = np.max(scores)

    log_sum_exp = mx + np.log(
        np.sum(np.exp(scores - mx))
    )

    return scores - log_sum_exp
```

### Alternative Implementation

```python
import numpy as np

def log_softmax(scores):
    scores = np.asarray(scores)

    mx = np.max(scores)

    exp_scores = np.exp(scores - mx)

    return np.log(exp_scores / exp_scores.sum())
```

---

## Code Explanation

### Step 1

Convert the input into a NumPy array.

```python
scores = np.asarray(scores)
```

This enables efficient vectorized computations.

---

### Step 2

Find the maximum score.

```python
mx = np.max(scores)
```

The maximum value is used for numerical stability.

---

### Step 3

Compute the Log-Sum-Exp term.

```python
log_sum_exp = mx + np.log(
    np.sum(np.exp(scores - mx))
)
```

Subtracting the maximum score prevents overflow during exponentiation.

---

### Step 4

Subtract the normalization constant.

```python
scores - log_sum_exp
```

The result is the Log Softmax vector.

Each element represents the logarithm of its corresponding softmax probability.

---

## Time & Space Complexity

| Complexity | Value    |
| ---------- | -------- |
| Time       | **O(n)** |
| Space      | **O(n)** |

where

- $n$ is the number of input scores.

Each score is visited a constant number of times while computing the maximum, exponentials, and normalization constant.
