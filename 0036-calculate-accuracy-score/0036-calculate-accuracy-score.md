# Calculate Accuracy Score (Easy, Machine Learning)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Accuracy Score](#learn-accuracy-score)
- [Solutions](#solutions)
  - [Custom Implementation](#custom-implementation)
  - [NumPy Implementation](#numpy-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

---

## Problem Statement

[Calculate Accuracy Score](https://www.deep-ml.com/problems/36)

Write a Python function to calculate the **Accuracy Score** given the true labels (`y_true`) and predicted labels (`y_pred`).

---

## Example

```python
Input:

y_true = np.array([1, 0, 1, 1, 0, 1])

y_pred = np.array([1, 0, 0, 1, 0, 1])

Output:

0.8333333333333334
```

**Reasoning**

Out of **6** predictions, **5** are correct.

```text
Accuracy = 5 / 6 = 0.8333
```

---

## Learn: Accuracy Score

**Accuracy** measures the proportion of correctly classified samples.

Formula:

```text
Accuracy = Correct Predictions / Total Predictions
```

For binary classification:

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

It is simple and effective, but may be misleading for **imbalanced datasets**.

---

## Solutions

### Custom Implementation

```python
import numpy as np


def accuracy_score(y_true, y_pred):

    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    return (TP + TN) / (TP + TN + FP + FN)


# Example Usage
y_true = np.array([1, 0, 1, 1, 0, 1])
y_pred = np.array([1, 0, 0, 1, 0, 1])

print(accuracy_score(y_true, y_pred))
```

### NumPy Implementation

```python
import numpy as np


def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)
```

---

## Code Explanation

- Compute the values of **TP**, **TN**, **FP**, and **FN**.
- Apply the accuracy formula.

```python
(TP + TN) / (TP + TN + FP + FN)
```

- Alternatively, compute the mean of matching labels.

```python
np.mean(y_true == y_pred)
```

---

## Time & Space Complexity

| Complexity | Value |
|------------|-------|
| Time | **O(n)** |
| Space | **O(1)** |

Where **n** is the number of samples.