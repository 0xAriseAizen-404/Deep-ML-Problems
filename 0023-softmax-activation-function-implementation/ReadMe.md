# Softmax Activation Function Implementation (Easy, Deep Learning)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Softmax Activation Function](#learn-softmax-activation-function)
- [Solutions](#solutions)
  - [Python Implementation](#python-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

---

## Problem Statement

[Softmax Activation Function Implementation](https://www.deep-ml.com/problems/23)

Write a Python function that computes the **Softmax Activation Function** for a given list of scores. Handle **numerical stability** by preventing overflow when exponentiating large values.

---

## Example

```python
Input:

scores = [1, 2, 3]

Output:

[0.09, 0.2447, 0.6652]
```

**Reasoning**

Softmax converts the input scores into a probability distribution, where all probabilities sum to **1**.

---

## Learn: Softmax Activation Function

The **Softmax** function converts a vector of scores into probabilities.

Formula:

```text
softmax(zᵢ) = eᶻⁱ / Σeᶻʲ
```

For numerical stability, subtract the maximum score before exponentiating:

```text
softmax(zᵢ) = eᶻⁱ⁻ᵐᵃˣ⁽ᶻ⁾ / Σeᶻʲ⁻ᵐᵃˣ⁽ᶻ⁾
```

**Properties**

- Output values are between **0 and 1**
- All probabilities sum to **1**
- Used for **multi-class classification**

---

## Solutions

### Python Implementation

```python
import math


def softmax(scores: list[float]) -> list[float]:

    max_score = max(scores)
    scores = [x - max_score for x in scores]

    exp_scores = [math.exp(x) for x in scores]
    total = sum(exp_scores)

    return [round(x / total, 4) for x in exp_scores]


# Example Usage
scores = [1, 2, 3]
print(softmax(scores))
```

---

## Code Explanation

- Subtract the maximum value for numerical stability.

```python
scores = [x - max_score for x in scores]
```

- Compute the exponentials.

```python
exp_scores = [math.exp(x) for x in scores]
```

- Divide each exponential by the total sum.

```python
x / total
```

- Round the probabilities to **4 decimal places**.

---

## Time & Space Complexity

| Complexity | Value |
|------------|-------|
| Time | **O(n)** |
| Space | **O(n)** |

Where **n** is the number of input scores.