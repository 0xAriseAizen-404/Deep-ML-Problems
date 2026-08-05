# Calculate Dice Score for Classification (Easy, Machine Learning)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Dice Score](#learn-dice-score)
  - [What is it?](#what-is-it)
  - [Mathematical Definition / Formula](#mathematical-definition--formula)
  - [Relationship with F1-Score](#relationship-with-f1-score)
  - [Characteristics / Key Points](#characteristics--key-points)
  - [Why is it used? / Applications](#why-is-it-used--applications)
- [Solution](#solution)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

---

## Problem Statement

[Calculate Dice Score for Classification](https://www.deep-ml.com/problems/73)

Write a Python function `dice_score(y_true, y_pred)` that computes the **Dice Score**, also known as the **Sørensen-Dice coefficient** or **F1-score**, for binary classification.

The function should:

- Calculate the Dice Score between `y_true` and `y_pred`.
- Return the result as a floating-point number rounded to **three decimal places**.
- Handle edge cases such as when there are no true or predicted positive labels.

The Dice Score measures the similarity between the predicted positive labels and the true positive labels. Its value ranges from **0** (no overlap) to **1** (perfect overlap).

---

## Example

```python
import numpy as np

y_true = np.array([1, 1, 0, 1, 0, 1])
y_pred = np.array([1, 1, 0, 0, 0, 1])

print(dice_score(y_true, y_pred))
```

### Output

```text
0.857
```

### Explanation

The overlap between the true and predicted labels is summarized below.

| Index | y_true | y_pred | True Positive? |
| :---: | :----: | :----: | :------------: |
|   0   |   1    |   1    |       ✓        |
|   1   |   1    |   1    |       ✓        |
|   2   |   0    |   0    |       ✗        |
|   3   |   1    |   0    |       ✗        |
|   4   |   0    |   0    |       ✗        |
|   5   |   1    |   1    |       ✓        |

Therefore,

- True Positives (TP) = **3**
- False Positives (FP) = **0**
- False Negatives (FN) = **1**

The Dice Score is

$$
\begin{aligned}
\text{Dice Score}
&=
\frac{2\times3}{2\times3+0+1} \\
&=
\frac{6}{7} \\
&=
0.857
\end{aligned}
$$

---

# Learn: Dice Score

## What is it?

The **Dice Score**, also called the **Sørensen-Dice coefficient**, measures the similarity between two sets.

In binary classification, it measures how well the predicted positive labels overlap with the actual positive labels.

A Dice Score of:

- **1** indicates perfect agreement.
- **0** indicates no overlap.

Compared to the **Jaccard Index**, the Dice Score places more emphasis on the overlapping elements by doubling the intersection.

---

## Mathematical Definition / Formula

For two sets \(A\) and \(B\),

$$
\boxed{
\text{Dice Score} =
\frac{2|A\cap B|}
{|A|+|B|}
}
$$

In binary classification,

$$
\boxed{
\text{Dice Score} =
\frac{2TP}
{2TP+FP+FN}
}
$$

where

- **TP** = True Positives
- **FP** = False Positives
- **FN** = False Negatives

---

## Relationship with F1-Score

The Dice Score is mathematically identical to the **F1-score**.

The F1-score is defined as

$$
\boxed{
F_1 =
\frac{2\times\text{Precision}\times\text{Recall}}
{\text{Precision}+\text{Recall}}
}
$$

which simplifies to the Dice Score:

$$
F_1=\text{Dice Score}
$$

---

## Characteristics / Key Points

- Values range from **0** to **1**.
- Higher values indicate greater similarity.
- Symmetric:

$$
\text{DSC}(A,B)=\text{DSC}(B,A)
$$

- More sensitive to overlap than the Jaccard Index.
- Widely used in image segmentation and binary classification.

### Interpretation

| Dice Score | Meaning          |
| :--------: | :--------------- |
|   **1**    | Perfect overlap  |
|   **0**    | No overlap       |
|  **0.5**   | Moderate overlap |
| **0.857**  | High similarity  |

---

## Why is it used? / Applications

The Dice Score is commonly used in:

- Medical Image Segmentation
- Semantic Segmentation
- Binary Classification
- Object Detection Evaluation
- Image Analysis
- Text Similarity
- Computer Vision

It is particularly popular in medical imaging because it emphasizes correctly overlapping regions.

> 💡 **Important Note**
>
> When both the predicted and true sets are empty, the denominator becomes zero. A common convention (followed by libraries such as scikit-learn) is to return **0.0** in this case.

---

# Solution

## Custom Implementation

```python
import numpy as np

def dice_score(y_true, y_pred):
    intersection = np.sum(y_true & y_pred)
    A = np.sum(y_true)
    B = np.sum(y_pred)

    if A + B == 0:
        return 0.0

    return round((2 * intersection) / (A + B), 3)
```

---

# Code Explanation

### Step 1: Compute the Intersection

```python
intersection = np.sum(y_true & y_pred)
```

The bitwise AND (`&`) identifies positions where both arrays contain `1`.

The sum gives

$$
|A \cap B|
$$

which equals the number of True Positives.

---

### Step 2: Count Positive Labels

```python
A = np.sum(y_true)
B = np.sum(y_pred)
```

`A` counts the number of actual positive labels, while `B` counts the number of predicted positive labels.

These correspond to

$$
|A|
\quad\text{and}\quad
|B|
$$

respectively.

---

### Step 3: Handle the Empty Set Case

```python
if A + B == 0:
    return 0.0
```

If both arrays contain only zeros, there are no positive labels, so the Dice Score is defined as `0.0`.

---

### Step 4: Compute the Dice Score

```python
return round((2 * intersection) / (A + B), 3)
```

This applies

$$
\boxed{
\text{Dice Score} =
\frac{2|A\cap B|}
{|A|+|B|}
}
$$

and rounds the result to three decimal places.

---

## Time & Space Complexity

Let **n** be the number of elements.

| Complexity |  Value   |
| :--------: | :------: |
|    Time    | **O(n)** |
|   Space    | **O(1)** |

The algorithm performs a constant number of passes over the input arrays. Aside from a few scalar variables, it does not allocate additional memory proportional to the input size.
