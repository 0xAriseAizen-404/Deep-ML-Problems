# Compute TPR and FPR from Classifications (Easy, Machine Learning)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: TPR and FPR](#learn-tpr-and-fpr)
- [Solutions](#solutions)
  - [Custom Implementation](#custom-implementation)
  - [NumPy Implementation](#numpy-implementation)
- [Code Explanation](#code-explanation)

## Problem Statement

[Compute TPR and FPR from Classifications](https://www.deep-ml.com/problems/1217)

Given the true labels `y_true` and predicted labels `y_pred`, compute the **True Positive Rate (TPR)** and **False Positive Rate (FPR)**. If a denominator is zero, return `0.0` for that rate.

## Example

```python
Input:
y_true = np.array([1, 1, 0, 0])
y_pred = np.array([1, 0, 1, 0])

Output:
(0.5, 0.5)
```

## Learn: TPR and FPR

The confusion matrix consists of:

- **TP**: Predicted Positive & Actually Positive
- **FP**: Predicted Positive & Actually Negative
- **FN**: Predicted Negative & Actually Positive
- **TN**: Predicted Negative & Actually Negative

Formulas:

```text
TPR = TP / (TP + FN)

FPR = FP / (FP + TN)
```

- **TPR (Recall/Sensitivity)** measures how many actual positives are correctly identified.
- **FPR** measures how many actual negatives are incorrectly classified as positive.

## Solutions

### Custom Implementation

```python
def compute_tpr_fpr(y_true, y_pred):
    TP = FP = TN = FN = 0

    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            TP += 1
        elif yt == 1 and yp == 0:
            FN += 1
        elif yt == 0 and yp == 1:
            FP += 1
        else:
            TN += 1

    tpr = TP / (TP + FN) if TP + FN else 0.0
    fpr = FP / (FP + TN) if FP + TN else 0.0

    return float(tpr), float(fpr)
```

### NumPy Implementation

```python
import numpy as np

def compute_tpr_fpr(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))

    tpr = TP / (TP + FN) if (TP + FN) else 0.0
    fpr = FP / (FP + TN) if (FP + TN) else 0.0

    return float(tpr), float(fpr)
```

## Code Explanation

- Compute **TP**, **FP**, **FN**, and **TN**.
- Calculate **TPR** using `TP / (TP + FN)`.
- Calculate **FPR** using `FP / (FP + TN)`.
- If a denominator is zero, return `0.0` to avoid division by zero.
- Return both values as Python floats.