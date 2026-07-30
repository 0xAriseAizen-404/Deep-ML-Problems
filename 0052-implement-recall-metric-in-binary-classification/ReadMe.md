# Implement Recall Metric in Binary Classification (Easy, Machine Learning)

## Table of Contents

- Problem Statement
- Example
- Learn: Recall Metric
- Solution
- Code Explanation
- Time & Space Complexity

---

## Problem Statement

[Implement Recall Metric in Binary Classification](https://www.deep-ml.com/problems/52)

Write a Python function `recall` that calculates the **recall score** for a binary classification problem given the true labels (`y_true`) and predicted labels (`y_pred`).

Recall measures the proportion of actual positive samples that are correctly identified by the model.

If there are no actual positive samples (i.e., `TP + FN = 0`), the function should return `0.0` to avoid division by zero.

---

## Example

```python
import numpy as np

y_true = np.array([1, 0, 1, 1, 0, 1])
y_pred = np.array([1, 0, 1, 0, 0, 1])

print(recall(y_true, y_pred))
```

### Output

```text
0.75
```

### Explanation

The confusion matrix values are:

- True Positives (TP) = 3
- False Negatives (FN) = 1

Using the recall formula:

\[
\text{Recall}
=
\frac{TP}{TP+FN}
\]

\[
=
\frac{3}{3+1}
=
0.75
\]

The model successfully identified **3 out of the 4 actual positive samples**.

---

# Learn: Recall Metric

## What is it?

**Recall** is one of the most important evaluation metrics in machine learning, especially for **binary classification** problems.

It measures **how many of the actual positive samples were correctly identified by the model**.

Unlike precision, which focuses on the correctness of positive predictions, recall focuses on **finding as many positive instances as possible**.

Recall is also known as **Sensitivity** or **True Positive Rate (TPR)**.

---

## Mathematical Definition / Formula

Recall is defined as:

\[
\text{Recall}
=
\frac{TP}{TP+FN}
\]

where:

- **TP (True Positives):** Positive samples correctly predicted as positive.
- **FN (False Negatives):** Positive samples incorrectly predicted as negative.

---

### Confusion Matrix

| Actual \\ Predicted | Positive | Negative |
| ------------------- | -------- | -------- |
| **Positive** | TP | FN |
| **Negative** | FP | TN |

Recall only depends on **TP** and **FN**.

---

## Characteristics / Key Points

- Recall ranges from **0 to 1**.
- **1.0** indicates every actual positive sample was correctly identified.
- High recall means **very few false negatives**.
- Does **not** consider false positives.
- Useful when missing positive samples is costly.

### Interpretation

- **High Recall:** The model captures most positive instances.
- **Low Recall:** The model misses many actual positive samples.

---

## Why is it used? / Applications

Recall is critical in applications where **false negatives are more expensive than false positives**.

Examples include:

- Disease and cancer detection
- Fraud detection
- Intrusion detection systems
- Manufacturing defect detection
- Disaster warning systems
- Search and information retrieval

> 💡 **Important Note**
>
> A model can achieve **100% recall** by predicting every sample as positive. However, this usually results in many false positives and low precision. In practice, recall is often evaluated together with **Precision**, and both are balanced using the **F1 Score**.

---

# Solution

## Custom Implementation

```python
import numpy as np

def recall(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    return TP / (TP + FN) if (TP + FN) > 0 else 0.0
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

### Step 2: Count False Negatives

```python
FN = np.sum((y_true == 1) & (y_pred == 0))
```

This counts samples where:

- the actual label is **1**, but
- the model incorrectly predicts **0**.

These are the positive samples that the model failed to identify.

---

### Step 3: Compute Recall

Using the formula:

\[
\text{Recall}
=
\frac{TP}{TP+FN}
\]

The implementation safely handles the edge case where there are **no actual positive samples**.

```python
return TP / (TP + FN) if (TP + FN) > 0 else 0.0
```

Returning `0.0` prevents a division-by-zero error.

---

## Time & Space Complexity

Let **n** be the number of samples.

| Complexity | Value |
| ---------- | ----- |
| Time | **O(n)** |
| Space | **O(1)** |

The algorithm scans the arrays once to compute the counts of **True Positives** and **False Negatives**, while using only constant extra memory.