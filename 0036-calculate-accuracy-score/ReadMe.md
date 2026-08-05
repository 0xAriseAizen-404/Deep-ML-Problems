# Calculate Accuracy Score (Easy, Machine Learning)

## Table of Contents

- Problem Statement
- Example
- Learn: Understanding Accuracy Score
- Solution
- Code Explanation
- Time & Space Complexity

---

## Problem Statement

### [Calculate Accuracy Score](https://www.deep-ml.com/problems/36)

Write a Python function that computes the **Accuracy Score** of a classification model.

The function should:

- Accept two 1-D NumPy arrays:
  - `y_true` containing the true labels.
  - `y_pred` containing the predicted labels.
- Compare both arrays.
- Return the proportion of correctly classified samples as a floating-point value.

---

## Example

### Input

```python
y_true = np.array([1, 0, 1, 1, 0, 1])

y_pred = np.array([1, 0, 0, 1, 0, 1])

output = accuracy_score(y_true, y_pred)

print(output)
```

### Output

```python
0.8333333333333334
```

### Reasoning

Comparing each prediction with its true label,

```text
True : 1 0 1 1 0 1
Pred : 1 0 0 1 0 1
        ✓ ✓ ✗ ✓ ✓ ✓
```

There are **5 correct predictions** out of **6** total predictions.

Therefore,

$$
\text{Accuracy} = \frac{5}{6} = 0.8333333333333334
$$

---

## Learn: Understanding Accuracy Score

### What is it?

Accuracy is one of the simplest and most commonly used evaluation metrics for **classification** problems.

It measures the proportion of predictions that are classified correctly.

A model with high accuracy predicts most samples correctly, whereas a model with low accuracy makes many incorrect predictions.

Accuracy is intuitive and easy to interpret, making it the first metric many practitioners examine after training a classification model.

However, it should only be relied upon when the classes are reasonably balanced.

---

### Mathematical Definition

Accuracy is defined as

$$
\text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}}
$$

For binary classification, the confusion matrix contains four possible outcomes:

- **True Positive (TP):** Predicted Positive and actually Positive.
- **True Negative (TN):** Predicted Negative and actually Negative.
- **False Positive (FP):** Predicted Positive but actually Negative.
- **False Negative (FN):** Predicted Negative but actually Positive.

Using these values,

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

where

- $TP$ is the number of True Positives.
- $TN$ is the number of True Negatives.
- $FP$ is the number of False Positives.
- $FN$ is the number of False Negatives.

---

### How Accuracy is Calculated

The algorithm follows these steps:

1. Compare every predicted label with the true label.
2. Count the number of correct predictions.
3. Divide the number of correct predictions by the total number of samples.

For example,

True labels

```text
[1, 0, 1, 1, 0]
```

Predicted labels

```text
[1, 1, 1, 0, 0]
```

Correct predictions

```text
✓ ✗ ✓ ✗ ✓
```

Number of correct predictions

$$
3
$$

Total predictions

$$
5
$$

Accuracy

$$
\frac{3}{5} = 0.6
$$

---

### Confusion Matrix Example

Suppose we have

```text
True Labels      : 1 0 1 1 0 0

Predicted Labels : 1 0 0 1 1 0
```

The confusion matrix becomes

| Actual / Predicted | Positive | Negative |
| ------------------ | -------: | -------: |
| Positive           |   TP = 2 |   FN = 1 |
| Negative           |   FP = 1 |   TN = 2 |

Accuracy is

$$
\frac{TP + TN}{TP + TN + FP + FN} =
\frac{2 + 2}{2 + 2 + 1 + 1} =
\frac{4}{6} =
0.667
$$

---

### Characteristics / Key Points

- Simple and easy to interpret.
- Measures overall prediction correctness.
- Values range between **0** and **1**.
- Higher values indicate better classification performance.
- Suitable for balanced datasets.
- Can be computed directly using Boolean comparisons.
- Uses every prediction equally.

---

### Why is it used? / Applications

Accuracy is widely used as a baseline evaluation metric in classification tasks.

Applications include:

- Binary Classification
- Multi-class Classification
- Image Classification
- Text Classification
- Spam Detection
- Handwritten Digit Recognition
- Medical Diagnosis
- Fraud Detection (when classes are balanced)

It is often reported alongside other metrics such as Precision, Recall, and F1-Score.

---

> 💡 **Important Note**
>
> Accuracy can be misleading for **imbalanced datasets**. For example, if 99% of samples belong to one class, a model that always predicts that class achieves **99% accuracy** while completely failing to detect the minority class. In such cases, metrics like **Precision**, **Recall**, **F1-Score**, **ROC-AUC**, or **Balanced Accuracy** provide a much better evaluation.

---

## Solution

### Custom Implementation (Using Confusion Matrix)

```python
import numpy as np

def accuracy_score(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))

    return (TP + TN) / (TP + TN + FP + FN)
```

### NumPy Implementation

```python
import numpy as np

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)
```

---

## Code Explanation

### Step 1

Compute the number of **True Positives**.

```python
TP = np.sum((y_true == 1) & (y_pred == 1))
```

These are samples correctly classified as the positive class.

---

### Step 2

Compute the number of **False Negatives**.

```python
FN = np.sum((y_true == 1) & (y_pred == 0))
```

These are positive samples incorrectly classified as negative.

---

### Step 3

Compute the number of **False Positives**.

```python
FP = np.sum((y_true == 0) & (y_pred == 1))
```

These are negative samples incorrectly classified as positive.

---

### Step 4

Compute the number of **True Negatives**.

```python
TN = np.sum((y_true == 0) & (y_pred == 0))
```

These are negative samples correctly classified.

---

### Step 5

Calculate the accuracy.

```python
(TP + TN) / (TP + TN + FP + FN)
```

The numerator represents the total number of correct predictions, while the denominator represents the total number of predictions.

---

### Alternative NumPy Solution

The simplest implementation is

```python
np.mean(y_true == y_pred)
```

The comparison

```python
y_true == y_pred
```

creates a Boolean array.

```text
[True, True, False, True, True]
```

NumPy treats

- `True` as `1`
- `False` as `0`

Therefore,

```python
np.mean(...)
```

directly computes the proportion of correct predictions.

---

## Time & Space Complexity

| Complexity | Value    |
| ---------- | -------- |
| Time       | **O(n)** |
| Space      | **O(1)** |

where

- $n$ is the number of samples.

Each sample is examined exactly once, and only a constant amount of additional memory is required.
