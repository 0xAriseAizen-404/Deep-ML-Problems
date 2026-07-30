# Implement F-Score Calculation for Binary Classification (Easy, Machine Learning)

## Table of Contents

- Problem Statement
- Example
- Learn: F-Score
- Solution
- Code Explanation
- Time & Space Complexity

---

## Problem Statement

[Implement F-Score Calculation for Binary Classification](https://www.deep-ml.com/problems/61)

Write a Python function `f_score(y_true, y_pred, beta)` that computes the **F-Score** for a binary classification problem.

The function takes:

- `y_true` — True binary labels.
- `y_pred` — Predicted binary labels.
- `beta` — A parameter that controls the relative importance of **Recall** versus **Precision**.

When `beta = 1`, the function computes the **F1-Score**, which gives equal importance to Precision and Recall.

The final F-Score should be rounded to **three decimal places**.

---

## Example

```python
import numpy as np

y_true = np.array([1, 0, 1, 1, 0, 1])
y_pred = np.array([1, 0, 1, 0, 0, 1])

beta = 1

print(f_score(y_true, y_pred, beta))
```

### Output

```text
0.857
```

### Explanation

First compute the confusion matrix values:

- True Positives (TP) = 3
- False Positives (FP) = 0
- False Negatives (FN) = 1

Calculate Precision:

\[
\text{Precision}
=
\frac{TP}{TP+FP}
=
\frac{3}{3}
=
1.0
\]

Calculate Recall:

\[
\text{Recall}
=
\frac{TP}{TP+FN}
=
\frac{3}{4}
=
0.75
\]

Since \( \beta = 1 \),

\[
F_1
=
2
\times
\frac{\text{Precision}\times\text{Recall}}
{\text{Precision}+\text{Recall}}
\]

\[
=
2
\times
\frac{1.0\times0.75}
{1.0+0.75}
=
0.857
\]

---

# Learn: F-Score

## What is it?

The **F-Score** (or **F-Measure**) is an evaluation metric that combines **Precision** and **Recall** into a single value.

A good classification model should:

- Predict positive samples correctly (**High Precision**).
- Find as many positive samples as possible (**High Recall**).

However, improving one metric often decreases the other. The F-Score provides a **balanced evaluation** by considering both simultaneously.

The most commonly used version is the **F1-Score**, where Precision and Recall are weighted equally.

---

## Mathematical Definition / Formula

The general **Fβ-Score** is defined as

\[
F_{\beta}
=
\frac{(1+\beta^2)\times(\text{Precision}\times\text{Recall})}
{(\beta^2\times\text{Precision})+\text{Recall}}
\]

where

- **Precision**

\[
\text{Precision}
=
\frac{TP}{TP+FP}
\]

- **Recall**

\[
\text{Recall}
=
\frac{TP}{TP+FN}
\]

---

### Effect of β

The value of **β** determines which metric is given more importance.

- **β = 1**

\[
F_1
=
2
\times
\frac{PR}{P+R}
\]

Precision and Recall are equally important.

---

- **β > 1**

Recall is weighted **more heavily** than Precision.

Useful when **missing positive samples is expensive**.

---

- **β < 1**

Precision is weighted **more heavily** than Recall.

Useful when **false positives are expensive**.

---

## Characteristics / Key Points

- Combines Precision and Recall into a single metric.
- Values range from **0 to 1**.
- Higher values indicate better classification performance.
- Useful for **imbalanced datasets**.
- Penalizes models that have high Precision but low Recall (or vice versa).
- Uses the **harmonic mean**, making it more sensitive to low values than the arithmetic mean.

---

## Why Harmonic Mean?

Instead of averaging Precision and Recall directly, the F-Score uses the **harmonic mean**:

\[
\text{Harmonic Mean}
=
\frac{2ab}{a+b}
\]

The harmonic mean ensures that both Precision and Recall must be high to achieve a high F-Score.

For example:

| Precision | Recall | Arithmetic Mean | F1-Score |
| ---------- | ------ | --------------- | -------- |
| 1.0 | 0.0 | 0.5 | 0.0 |
| 0.8 | 0.8 | 0.8 | 0.8 |
| 0.9 | 0.3 | 0.6 | 0.45 |

Notice how the harmonic mean heavily penalizes imbalance between the two metrics.

---

## Why is it used? / Applications

The F-Score is widely used when both false positives and false negatives matter.

Common applications include:

- Medical diagnosis
- Spam email detection
- Fraud detection
- Search engines
- Recommendation systems
- Information retrieval
- Sentiment analysis
- Credit risk prediction

> 💡 **Important Note**
>
> Accuracy can be misleading for imbalanced datasets. For example, if only 1% of patients have a disease, a model predicting everyone as healthy achieves **99% accuracy**, but its Recall and F1-Score are **0**. This is why Precision, Recall, and F-Score are preferred for evaluating imbalanced classification problems.

---

# Solution

## Custom Implementation

```python
import numpy as np

def f_score(y_true, y_pred, beta):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    return np.round(
        (1 + beta**2) *
        (
            (precision * recall) /
            (((beta**2) * precision) + recall)
        ),
        3
    )
```

---

# Code Explanation

### Step 1: Compute the Confusion Matrix

```python
TP = np.sum((y_true == 1) & (y_pred == 1))
FN = np.sum((y_true == 1) & (y_pred == 0))
FP = np.sum((y_true == 0) & (y_pred == 1))
TN = np.sum((y_true == 0) & (y_pred == 0))
```

These values summarize the classification results.

---

### Step 2: Compute Precision

```python
precision = TP / (TP + FP)
```

Precision measures:

> Out of all predicted positive samples, how many were actually positive?

---

### Step 3: Compute Recall

```python
recall = TP / (TP + FN)
```

Recall measures:

> Out of all actual positive samples, how many were correctly identified?

---

### Step 4: Compute the F-Score

Using

\[
F_{\beta}
=
\frac{(1+\beta^2)\times PR}
{(\beta^2P)+R}
\]

the implementation calculates the weighted harmonic mean of Precision and Recall.

Finally,

```python
np.round(..., 3)
```

rounds the result to **three decimal places**.

---

## Time & Space Complexity

Let **n** be the number of samples.

| Complexity | Value |
| ---------- | ----- |
| Time | **O(n)** |
| Space | **O(1)** |

The algorithm performs a single pass over the arrays to compute the confusion matrix values and then evaluates the F-Score using constant extra memory.

---
## Note

The provided implementation assumes that the denominators for **Precision**, **Recall**, and the **F-Score** formula are non-zero. In a production implementation, you should handle these edge cases to avoid division-by-zero errors:

```python
if TP + FP == 0 or TP + FN == 0:
    return 0.0
```

This ensures the function behaves correctly even when there are no predicted positives or no actual positives.