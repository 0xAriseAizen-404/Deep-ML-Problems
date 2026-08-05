# Implement Recall Metric in Binary Classification (Easy, Machine Learning)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Understanding Recall in Classification](#learn-understanding-recall-in-classification)
- [Solution](#solution)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

---

## Problem Statement

### [Implement Recall Metric in Binary Classification](https://www.deep-ml.com/problems/52)

Write a Python function that computes the **Recall** metric for a binary classification task.

The function should:

- Accept the true binary labels.
- Accept the predicted binary labels.
- Count the number of correctly predicted positive samples.
- Count the number of actual positive samples that were missed.
- Compute the Recall score.
- Return `0.0` when there are no actual positive samples to avoid division by zero.

---

## Example

### Input

```python
import numpy as np

y_true = np.array([1, 0, 1, 1, 0, 1])
y_pred = np.array([1, 0, 1, 0, 0, 1])

print(recall(y_true, y_pred))
```

### Output

```python
0.75
```

### Reasoning

There are **4** actual positive samples.

The model correctly predicts **3** of them while missing **1** positive sample.

Therefore,

$$ \text{Recall} = \frac{3}{3+1} = 0.75 $$

The classifier successfully identifies **75%** of all actual positive instances.

---

## Learn: Understanding Recall in Classification

### What is it?

**Recall** is one of the most important evaluation metrics in binary classification. It measures how effectively a model identifies all actual positive samples in a dataset.

It answers the question:

> **Out of every sample that is truly positive, how many did the model correctly predict as positive?**

Recall is also called **Sensitivity** or the **True Positive Rate (TPR)**.

A high Recall means the model misses very few positive samples, making it especially useful when false negatives are expensive.

---

### Mathematical Definition

Recall is defined as

$$ \text{Recall} = \frac{TP}{TP+FN} $$

where

- $TP$ is the number of **True Positives**.
- $FN$ is the number of **False Negatives**.

The denominator represents the total number of actual positive samples.

---

### Confusion Matrix

Binary classification predictions can be summarized using a confusion matrix.

| Actual \ Predicted | Positive | Negative |
| ------------------ | -------- | -------- |
| Positive           | TP       | FN       |
| Negative           | FP       | TN       |

For Recall, only **True Positives** and **False Negatives** are used.

---

### True Positives

A True Positive occurs when

- The actual label is positive.
- The predicted label is also positive.

Mathematically,

$$ TP = \sum (y*{true}=1 \land y*{pred}=1) $$

Higher TP directly increases Recall.

---

### False Negatives

A False Negative occurs when

- The actual label is positive.
- The model predicts it as negative.

Mathematically,

$$ FN = \sum (y*{true}=1 \land y*{pred}=0) $$

False Negatives decrease Recall because they represent missed positive samples.

---

### Why Recall Ignores False Positives

Recall measures only how many actual positives were successfully detected.

Predicting additional positive samples affects **Precision**, but does not directly affect Recall.

For this reason, Recall focuses entirely on minimizing missed positive cases.

---

### Relationship with Precision

Precision and Recall are often used together.

Precision measures

> Out of all predicted positives, how many are actually positive?

Recall measures

> Out of all actual positives, how many are detected?

Their formulas are

$$ \text{Precision} = \frac{TP}{TP+FP} $$

$$ \text{Recall} = \frac{TP}{TP+FN} $$

Improving Recall often lowers Precision because predicting more positives may increase False Positives.

---

### Relationship with F1 Score

The **F1 Score** combines Precision and Recall into a single metric.

$$ F_1 = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision}+\text{Recall}} $$

It provides a balanced evaluation when both false positives and false negatives are important.

---

### Edge Case

If there are no actual positive samples,

$$ TP+FN = 0 $$

the Recall formula becomes undefined.

To avoid division by zero, this problem specifies returning

$$ \text{Recall} = 0.0 $$

---

### Characteristics / Key Points

- Measures positive class coverage.
- Values range from **0** to **1**.
- Higher Recall indicates fewer missed positive samples.
- Penalizes False Negatives.
- Independent of True Negatives.
- Commonly used alongside Precision.
- Particularly useful for imbalanced datasets.
- Easy to compute from the confusion matrix.

---

### Why is it used? / Applications

Recall is preferred whenever missing a positive case has serious consequences.

Common applications include

- Disease diagnosis
- Cancer detection
- Fraud detection
- Credit card fraud monitoring
- Intrusion detection
- Spam filtering
- Defect detection
- Medical imaging
- Search and information retrieval

In these problems, identifying as many positive samples as possible is more important than avoiding extra positive predictions.

---

### Practical Example

Suppose a disease screening model evaluates **200** patients.

- 50 patients actually have the disease.
- The model correctly detects 47.
- It misses 3 patients.

The Recall becomes

$$ \text{Recall} = \frac{47}{47+3} = 0.94 $$

This means the model identifies **94%** of all infected patients.

---

### Common Mistakes

- Confusing Recall with Precision.
- Ignoring False Negatives.
- Assuming high Recall always means high Accuracy.
- Forgetting to handle division by zero.
- Evaluating Recall without considering Precision.

---

> 💡 **Important Note**
>
> A model can achieve **100% Recall** by predicting every sample as positive. Although no positive samples are missed, this usually produces many False Positives. Therefore, Recall should almost always be evaluated together with Precision or the F1 Score.

---

## Solution

### NumPy Implementation

```python
import numpy as np

def recall(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return TP / (TP + FN) if (TP + FN) > 0.0 else 0.0
```

---

## Code Explanation

### Step 1

Identify every sample where both the actual label and predicted label are positive.

```python
TP = np.sum((y_true == 1) & (y_pred == 1))
```

This computes the total number of **True Positives**.

---

### Step 2

Identify every sample where the actual label is positive but the prediction is negative.

```python
FN = np.sum((y_true == 1) & (y_pred == 0))
```

These are the **False Negatives**, representing positive samples missed by the model.

---

### Step 3

Compute the Recall score.

```python
TP / (TP + FN)
```

This gives the fraction of actual positive samples correctly identified.

---

### Step 4

Handle the edge case.

```python
if (TP + FN) > 0.0:
```

If there are no actual positive samples, return `0.0` instead of performing an invalid division.

---

## Time & Space Complexity

| Complexity | Value    |
| ---------- | -------- |
| Time       | **O(n)** |
| Space      | **O(1)** |

where

- $n$ is the number of samples.
- Each sample is inspected exactly once.
- Only two counters (`TP` and `FN`) are maintained, requiring constant extra memory.
