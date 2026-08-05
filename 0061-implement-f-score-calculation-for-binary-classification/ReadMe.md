# Implement F-Score Calculation for Binary Classification (Easy, Machine Learning)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Understanding F-Score in Classification](#learn-understanding-f-score-in-classification)
- [Solution](#solution)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

---

## Problem Statement

### [Implement F-Score Calculation for Binary Classification](https://www.deep-ml.com/problems/61)

Write a Python function that computes the **F-Score** for a binary classification task.

The function should:

- Accept the true binary labels.
- Accept the predicted binary labels.
- Accept a configurable **beta** value.
- Compute the Precision and Recall.
- Calculate the corresponding **F-Score**.
- Return the result rounded to **three decimal places**.

---

## Example

### Input

```python
import numpy as np

y_true = np.array([1, 0, 1, 1, 0, 1])
y_pred = np.array([1, 0, 1, 0, 0, 1])

beta = 1

print(f_score(y_true, y_pred, beta))
```

### Output

```python
0.857
```

### Reasoning

From the predictions,

- True Positives (TP) = **3**
- False Positives (FP) = **0**
- False Negatives (FN) = **1**

Therefore,

$$ \text{Precision} = \frac{3}{3+0} = 1.0 $$

$$ \text{Recall} = \frac{3}{3+1} = 0.75 $$

Since **β = 1**, the metric becomes the **F1-Score**.

$$ F_1 = \frac{2\times1.0\times0.75}{1.0+0.75} = 0.857 $$

---

## Learn: Understanding F-Score in Classification

### What is it?

The **F-Score** (or **F-Measure**) is a performance metric that combines **Precision** and **Recall** into a single value.

Instead of evaluating these metrics independently, the F-Score provides a balanced assessment of a classification model, especially for **imbalanced datasets**.

The parameter **β (beta)** determines whether Precision or Recall should receive more importance.

- β = 1 → Equal importance (**F1-Score**)
- β > 1 → Recall is emphasized.
- β < 1 → Precision is emphasized.

---

### Mathematical Definition

The general F-Score is defined as

$$ F\_\beta = \frac{(1+\beta^2)\times\text{Precision}\times\text{Recall}}{(\beta^2\times\text{Precision})+\text{Recall}} $$

where

- **Precision** measures prediction correctness.
- **Recall** measures positive class coverage.
- **β** controls the trade-off between Precision and Recall.

---

### Precision

Precision measures how many predicted positive samples are actually positive.

$$ \text{Precision} = \frac{TP}{TP+FP} $$

where

- $TP$ is the number of True Positives.
- $FP$ is the number of False Positives.

Higher Precision means fewer incorrect positive predictions.

---

### Recall

Recall measures how many actual positive samples were correctly identified.

$$ \text{Recall} = \frac{TP}{TP+FN} $$

where

- $FN$ is the number of False Negatives.

Higher Recall means fewer missed positive samples.

---

### F1-Score

When

$$ \beta = 1 $$

the formula becomes

$$ F_1 = \frac{2\times\text{Precision}\times\text{Recall}}{\text{Precision}+\text{Recall}} $$

The F1-Score is the harmonic mean of Precision and Recall.

Unlike the arithmetic mean, the harmonic mean penalizes models that perform poorly on either metric.

---

### Confusion Matrix

The required quantities are obtained from the confusion matrix.

| Actual \ Predicted | Positive | Negative |
| ------------------ | -------- | -------- |
| Positive           | TP       | FN       |
| Negative           | FP       | TN       |

Only **TP**, **FP**, and **FN** contribute to the F-Score calculation.

---

### Choosing β

Different values of β prioritize different objectives.

| β Value | Priority  |
| ------- | --------- |
| β < 1   | Precision |
| β = 1   | Balanced  |
| β > 1   | Recall    |

For example,

- Spam detection often prefers Precision.
- Disease diagnosis usually prefers Recall.
- General classification commonly uses the F1-Score.

---

### Characteristics / Key Points

- Combines Precision and Recall into one metric.
- Values range from **0** to **1**.
- Higher values indicate better classification performance.
- Useful for imbalanced datasets.
- Penalizes poor Precision or poor Recall.
- Independent of True Negatives.
- Common evaluation metric in machine learning competitions.

---

### Why is it used? / Applications

The F-Score is widely used in

- Medical diagnosis
- Fraud detection
- Spam filtering
- Search engines
- Information retrieval
- Sentiment analysis
- Document classification
- Object detection
- NLP classification tasks

It provides a better measure than Accuracy when the classes are imbalanced.

---

### Practical Example

Suppose a classifier predicts

- TP = 90
- FP = 10
- FN = 20

Then,

$$ \text{Precision} = \frac{90}{90+10} = 0.90 $$

$$ \text{Recall} = \frac{90}{90+20} \approx 0.818 $$

The F1-Score becomes

$$ F_1 = \frac{2\times0.90\times0.818}{0.90+0.818} \approx 0.857 $$

The classifier achieves a balanced performance between Precision and Recall.

---

### Common Mistakes

- Confusing the F1-Score with Accuracy.
- Ignoring the effect of β.
- Forgetting to compute Precision before Recall.
- Not handling zero denominators.
- Using the F-Score alone without considering class distribution.

---

> 💡 **Important Note**
>
> The F1-Score is the most commonly reported version of the F-Score because it gives equal importance to Precision and Recall. However, in applications where missing positives or false alarms have different costs, choosing an appropriate β value provides a more meaningful evaluation.

---

## Solution

### NumPy Implementation

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
        (1 + beta**2) * (
            (precision * recall) /
            (((beta**2) * precision) + recall)
        ),
        3
    )
```

---

## Code Explanation

### Step 1

Count the values in the confusion matrix.

```python
TP = np.sum((y_true == 1) & (y_pred == 1))
FN = np.sum((y_true == 1) & (y_pred == 0))
FP = np.sum((y_true == 0) & (y_pred == 1))
TN = np.sum((y_true == 0) & (y_pred == 0))
```

These values summarize the classifier's predictions.

---

### Step 2

Compute Precision.

```python
precision = TP / (TP + FP)
```

This measures how many predicted positive samples are actually correct.

---

### Step 3

Compute Recall.

```python
recall = TP / (TP + FN)
```

This measures how many actual positive samples are correctly identified.

---

### Step 4

Compute the F-Score.

```python
(1 + beta**2) * (
    (precision * recall) /
    (((beta**2) * precision) + recall)
)
```

The β parameter determines the relative importance of Precision and Recall.

---

### Step 5

Round the result.

```python
np.round(..., 3)
```

The final score is rounded to **three decimal places** as required.

---

## Time & Space Complexity

| Complexity | Value    |
| ---------- | -------- |
| Time       | **O(n)** |
| Space      | **O(1)** |

where

- $n$ is the number of samples.
- The arrays are scanned once to compute the confusion matrix.
- Only a constant number of variables are used, resulting in constant auxiliary space.
