# Implement Precision Metric (Easy, Machine Learning)

## Table of Contents

- Problem Statement
- Example
- Learn: Precision Metric
- Solution
- Code Explanation
- Time & Space Complexity

---

## Problem Statement

[Implement Precision Metric](https://www.deep-ml.com/problems/46)

Write a Python function `precision` that calculates the **precision score** for a binary classification problem given the true labels (`y_true`) and predicted labels (`y_pred`).

Precision measures the proportion of predicted positive samples that are actually positive.

---

## Example

```python
import numpy as np

y_true = np.array([1, 0, 1, 1, 0, 1])
y_pred = np.array([1, 0, 1, 0, 0, 1])

result = precision(y_true, y_pred)
print(result)
```

### Output

```text
1.0
```

### Explanation

The confusion matrix values are:

- True Positives (TP) = 3
- False Positives (FP) = 0

Using the precision formula:

\[
\text{Precision}
=
\frac{TP}{TP+FP}
\]

# \[

# \frac{3}{3+0}

1.0
\]

The model correctly classified every predicted positive sample.

---

# Learn: Precision Metric

## What is it?

**Precision** is one of the most important evaluation metrics for classification models, especially in **binary classification**.

It measures **how reliable the model's positive predictions are**. In other words, when the model predicts a sample as positive, precision tells us how often that prediction is correct.

Unlike accuracy, precision focuses **only on predicted positive samples**, making it particularly useful when false positives are costly.

---

## Mathematical Definition / Formula

Precision is defined as:

\[
\text{Precision}
=
\frac{TP}{TP+FP}
\]

where:

- **TP (True Positives):** Positive samples correctly predicted as positive.
- **FP (False Positives):** Negative samples incorrectly predicted as positive.

---

### Confusion Matrix

| Actual \\ Predicted | Positive | Negative |
| ------------------- | -------- | -------- |
| **Positive**        | TP       | FN       |
| **Negative**        | FP       | TN       |

Precision only depends on **TP** and **FP**.

---

## Characteristics / Key Points

- Precision ranges from **0 to 1**.
- **1.0** indicates every predicted positive sample is actually positive.
- High precision means **few false positives**.
- Does **not** consider false negatives.
- Useful when positive predictions must be highly trustworthy.

### Interpretation

- **High Precision:** Very few incorrect positive predictions.
- **Low Precision:** Many negative samples are incorrectly classified as positive.

---

## Why is it used? / Applications

Precision is commonly used in applications where **false positives are expensive**.

Examples include:

- Medical diagnosis (incorrectly diagnosing healthy patients)
- Spam email detection
- Fraud detection
- Malware detection
- Recommendation systems
- Search engines (relevance of retrieved results)

> 💡 **Important Note**
>
> A model can achieve **100% precision** simply by making very few positive predictions. However, it may miss many actual positive samples. Therefore, precision should usually be considered together with **Recall**, and both are often combined using the **F1 Score**.

---

# Solution

## Custom Implementation

```python
import numpy as np

def precision(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))

    return TP / (TP + FP) if (TP + FP) > 0 else 0.0
```

---

# Code Explanation

### Step 1: Count True Positives

```python
TP = np.sum((y_true == 1) & (y_pred == 1))
```

This counts all samples where:

- the actual label is **1**, and
- the predicted label is also **1**.

---

### Step 2: Count False Positives

```python
FP = np.sum((y_true == 0) & (y_pred == 1))
```

This counts samples where:

- the actual label is **0**, but
- the model incorrectly predicts **1**.

---

### Step 3: Compute Precision

Using the formula:

\[
\text{Precision}
=
\frac{TP}{TP+FP}
\]

The implementation safely handles the edge case where the model predicts **no positive samples**.

```python
return TP / (TP + FP) if (TP + FP) > 0 else 0.0
```

Returning `0.0` prevents a division-by-zero error.

---

## Time & Space Complexity

Let **n** be the number of samples.

| Complexity | Value    |
| ---------- | -------- |
| Time       | **O(n)** |
| Space      | **O(1)** |

The algorithm scans the arrays once to compute the counts of **True Positives** and **False Positives**, while using only constant extra memory.
