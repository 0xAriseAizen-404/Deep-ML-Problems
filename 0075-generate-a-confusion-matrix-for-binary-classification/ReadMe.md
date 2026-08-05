# Generate a Confusion Matrix for Binary Classification (Easy, Machine Learning)

## Table of Contents

- Problem Statement
- Example
- Learn: Confusion Matrix
- Solution
- Code Explanation
- Time & Space Complexity

---

## Problem Statement

[Generate a Confusion Matrix for Binary Classification](https://www.deep-ml.com/problems/75)

Write a Python function `confusion_matrix(data)` that generates the **confusion matrix** for a binary classification problem.

The input is a list of lists, where each inner list contains:

- `y_true` — the actual class label.
- `y_pred` — the predicted class label.

The function should return a **2 × 2 confusion matrix** represented as a list of lists.

The confusion matrix summarizes the prediction results by counting correctly and incorrectly classified samples.

---

## Example

```python
data = [
    [1, 1],
    [1, 0],
    [0, 1],
    [0, 0],
    [0, 1]
]

print(confusion_matrix(data))
```

### Output

```text
[[1, 1], [2, 1]]
```

### Explanation

The predictions can be categorized as follows:

| Actual | Predicted | Category            |
| ------ | --------- | ------------------- |
| 1      | 1         | True Positive (TP)  |
| 1      | 0         | False Negative (FN) |
| 0      | 1         | False Positive (FP) |
| 0      | 0         | True Negative (TN)  |
| 0      | 1         | False Positive (FP) |

Therefore,

```text
TP = 1
FN = 1
FP = 2
TN = 1
```

The confusion matrix is

$$
\begin{bmatrix}
TP & FN \\
FP & TN
\end{bmatrix} =
\begin{bmatrix}
1 & 1 \\
2 & 1
\end{bmatrix}
$$

---

# Learn: Confusion Matrix

## What is it?

A **Confusion Matrix** is one of the most commonly used tools for evaluating **classification models**.

It summarizes how many predictions were:

- Correctly classified.
- Incorrectly classified.

For binary classification, the confusion matrix is a **2 × 2** table that counts predictions based on their actual and predicted classes.

It serves as the foundation for many other evaluation metrics, including **Accuracy**, **Precision**, **Recall**, **Specificity**, and the **F1-score**.

---

## Matrix Structure

For binary classification, the confusion matrix is

$$
M =
\begin{bmatrix}
TP & FN\\
FP & TN
\end{bmatrix}
$$

where

- **TP (True Positives):** Actual positive samples predicted as positive.
- **FN (False Negatives):** Actual positive samples predicted as negative.
- **FP (False Positives):** Actual negative samples predicted as positive.
- **TN (True Negatives):** Actual negative samples predicted as negative.

---

## Mathematical Representation

Given

- Actual labels

$$
y_{\text{true}} =
[y_1,y_2,\ldots,y_n]
$$

- Predicted labels

$$
y_{\text{pred}} =
[\hat{y}_1,\hat{y}_2,\ldots,\hat{y}_n]
$$

the confusion matrix counts

$$
TP =
\sum_{i=1}^{n}
\mathbf{1}(y_i=1,\hat{y}_i=1)
$$

$$
FN =
\sum_{i=1}^{n}
\mathbf{1}(y_i=1,\hat{y}_i=0)
$$

$$
FP =
\sum_{i=1}^{n}
\mathbf{1}(y_i=0,\hat{y}_i=1)
$$

$$
TN =
\sum_{i=1}^{n}
\mathbf{1}(y_i=0,\hat{y}_i=0)
$$

where

$$
\mathbf{1}(\cdot)
$$

is the indicator function that equals **1** when the condition is true and **0** otherwise.

---

## Characteristics / Key Points

- Used for **Classification** problems.
- Applicable to binary and multi-class classification.
- Summarizes prediction outcomes.
- Forms the basis for many evaluation metrics.
- Easy to interpret and visualize.

---

## Why is it used? / Applications

The confusion matrix is widely used in:

- Binary Classification
- Medical Diagnosis
- Fraud Detection
- Spam Detection
- Image Classification
- Machine Learning Model Evaluation
- Information Retrieval

It provides a complete picture of classification performance rather than a single score like accuracy.

> 💡 **Important Note**
>
> A confusion matrix by itself is not a performance metric. Instead, it provides the counts needed to compute metrics such as **Accuracy**, **Precision**, **Recall**, **Specificity**, and the **F1-score**, which offer deeper insights into model performance.

---

# Solution

## Custom Implementation

```python
def confusion_matrix(data):
    TP = FN = FP = TN = 0

    for actual, pred in data:
        TP += actual & pred
        FN += actual & (1 - pred)
        FP += (1 - actual) & pred
        TN += (1 - actual) & (1 - pred)

    return [[TP, FN], [FP, TN]]
```

---

# Code Explanation

### Step 1: Initialize the Counters

```python
TP = FN = FP = TN = 0
```

Each variable stores the count of one category in the confusion matrix.

---

### Step 2: Process Each Observation

```python
for actual, pred in data:
```

Each element of `data` contains:

- `actual` — the true label.
- `pred` — the predicted label.

---

### Step 3: Update the Counts

```python
TP += actual & pred
```

Counts observations where both the actual and predicted labels are `1`.

---

```python
FN += actual & (1 - pred)
```

Counts positive samples that were predicted as negative.

---

```python
FP += (1 - actual) & pred
```

Counts negative samples that were predicted as positive.

---

```python
TN += (1 - actual) & (1 - pred)
```

Counts negative samples that were predicted correctly.

---

### Step 4: Return the Confusion Matrix

```python
return [[TP, FN], [FP, TN]]
```

The confusion matrix is returned in the form

$$
\begin{bmatrix}
TP & FN\\
FP & TN
\end{bmatrix}
$$

---

## Time & Space Complexity

Let

- $n$ = Number of observations.

| Complexity | Value    |
| ---------- | -------- |
| Time       | **O(n)** |
| Space      | **O(1)** |

The algorithm scans the input exactly once, updating four counters for each observation. Since it uses only a constant amount of additional memory regardless of the input size, the auxiliary space complexity is **O(1)**.
