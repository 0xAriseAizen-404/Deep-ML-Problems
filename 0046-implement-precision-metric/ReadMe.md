# Implement Precision Metric (Easy, Machine Learning)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Understanding Precision Metric](#learn-understanding-precision-metric)
- [Solutions](#solutions)
  - [Custom Implementation](#custom-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

---

## Problem Statement

### [Implement Precision Metric](https://www.deep-ml.com/problems/46)

Write a Python function that calculates the **Precision** metric for a binary classification model.

The function should take:

- `y_true`: A 1D NumPy array containing the actual binary labels.
- `y_pred`: A 1D NumPy array containing the predicted binary labels.

The function should calculate the ratio of correctly predicted positive samples to all samples predicted as positive.

---

## Example

### Input

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

### Reasoning

The confusion matrix values are:

- True Positives:

$$
TP=3
$$

- False Positives:

$$
FP=0
$$

Precision is calculated as:

$$
Precision=\frac{TP}{TP+FP}
$$

Substituting the values:

$$
Precision=\frac{3}{3+0}=1.0
$$

The model has perfect precision because all predicted positive samples are correct.

---

## Learn: Understanding Precision Metric

### What is it?

Precision is a classification evaluation metric that measures the quality of positive predictions made by a model.

It answers the question:

> "Out of all samples predicted as positive, how many were actually positive?"

Precision focuses on reducing **False Positives**, where the model incorrectly predicts negative samples as positive.

A high precision score means that the model produces fewer incorrect positive predictions.

---

### Mathematical Definition

Precision is defined as:

$$
Precision=\frac{TP}{TP+FP}
$$

where:

- $TP$ represents True Positives.
- $FP$ represents False Positives.

True Positive:

- Actual class is positive.
- Model predicts positive.

False Positive:

- Actual class is negative.
- Model predicts positive.

---

### Confusion Matrix

For binary classification:

|                 | Predicted Positive  | Predicted Negative  |
| --------------- | ------------------- | ------------------- |
| Actual Positive | True Positive (TP)  | False Negative (FN) |
| Actual Negative | False Positive (FP) | True Negative (TN)  |

Precision only considers the samples predicted as positive:

$$
Predicted\ Positive=TP+FP
$$

---

### Example

Suppose an email classifier predicts:

```text
50 emails as spam
```

Among them:

```text
45 emails are actually spam
5 emails are not spam
```

Then:

$$
Precision=\frac{45}{45+5}
$$

$$
Precision=0.9
$$

This means 90% of spam predictions were correct.

---

### Precision vs Recall

Precision and Recall measure different aspects of classification performance.

| Metric    | Measures                            | Focus                    |
| --------- | ----------------------------------- | ------------------------ |
| Precision | Correctness of positive predictions | Reducing False Positives |
| Recall    | Ability to find actual positives    | Reducing False Negatives |

Precision is important when false alarms are costly.

Recall is important when missing positive cases is dangerous.

---

### Precision vs Accuracy

Accuracy measures total correct predictions:

$$
Accuracy=\frac{TP+TN}{TP+TN+FP+FN}
$$

However, accuracy can be misleading for imbalanced datasets.

For example, in fraud detection, most transactions are legitimate.

A model predicting every transaction as legitimate can achieve high accuracy while failing to detect fraud.

Precision provides a better measurement of positive predictions.

---

### Characteristics / Key Points

- Measures correctness of positive predictions.
- Range is between 0 and 1.
- High precision means fewer false positives.
- Useful when false positives are expensive.
- Does not directly consider false negatives.
- Requires careful handling when there are no positive predictions.

---

### Why is it used? / Applications

Precision is commonly used in:

- Medical diagnosis
- Fraud detection
- Spam filtering
- Search engines
- Recommendation systems
- Object detection
- Security systems

For example, in fraud detection, falsely marking a normal transaction as fraudulent can create unnecessary investigation costs.

---

> 💡 **Important Note**
>
> Precision alone does not measure the complete performance of a classifier. A model predicting only a few highly confident positives can achieve high precision while missing many positive samples. Precision is usually evaluated together with Recall using the F1 Score.

---

## Solutions

### Custom Implementation

```python
import numpy as np

def precision(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))

    return TP / (TP + FP) if (TP + FP) > 0.0 else 0.0
```

---

## Code Explanation

### Step 1

Calculate True Positives.

```python
TP = np.sum((y_true == 1) & (y_pred == 1))
```

This counts samples where:

- Actual label is positive.
- Predicted label is positive.

---

### Step 2

Calculate False Positives.

```python
FP = np.sum((y_true == 0) & (y_pred == 1))
```

This counts samples where:

- Actual label is negative.
- Predicted label is positive.

---

### Step 3

Apply the precision formula.

$$
Precision=\frac{TP}{TP+FP}
$$

The denominator represents all positive predictions made by the model.

---

### Step 4

Handle division by zero.

If the model does not predict any positive samples:

$$
TP+FP=0
$$

the function returns `0.0` instead of causing a division error.

---

## Time & Space Complexity

| Complexity | Value    |
| ---------- | -------- |
| Time       | **O(n)** |
| Space      | **O(1)** |

where:

- $n$ is the number of samples.

The algorithm checks every prediction once and stores only the TP and FP counters.
