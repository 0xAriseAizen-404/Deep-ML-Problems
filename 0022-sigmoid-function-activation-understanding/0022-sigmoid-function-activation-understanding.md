# Sigmoid Activation Function Understanding (Easy, Deep Learning)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Sigmoid Activation Function](#learn-sigmoid-activation-function)
- [Solutions](#solutions)
  - [Python Implementation](#python-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

---

## Problem Statement

[Sigmoid Activation Function Understanding](https://www.deep-ml.com/problems/22)

Write a Python function that computes the output of the **Sigmoid Activation Function** for a given input `z`. Return the result rounded to **4 decimal places**.

---

## Example

```python
Input:

z = 0

Output:

0.5
```

**Reasoning**

```text
σ(0) = 1 / (1 + e⁰) = 1 / 2 = 0.5
```

---

## Learn: Sigmoid Activation Function

The **Sigmoid** function maps any real number to the range **(0, 1)**.

Formula:

```text
σ(z) = 1 / (1 + e⁻ᶻ)
```

**Properties**

- Output range: **(0, 1)**
- Commonly used for **binary classification**
- Converts logits into probabilities

---

## Solutions

### Python Implementation

```python
import math


def sigmoid(z: float) -> float:
    return round(1 / (1 + math.exp(-z)), 4)


# Example Usage
print(sigmoid(0))
```

---

## Code Explanation

- Compute `e⁻ᶻ` using `math.exp()`.
- Apply the sigmoid formula.

```python
1 / (1 + math.exp(-z))
```

- Round the result to **4 decimal places**.

---

## Time & Space Complexity

| Complexity | Value |
|------------|-------|
| Time | **O(1)** |
| Space | **O(1)** |