# Calculate Jaccard Index for Binary Classification (Easy, Machine Learning)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Jaccard Index](#learn-jaccard-index)
  - [What is it?](#what-is-it)
  - [Mathematical Definition / Formula](#mathematical-definition--formula)
  - [Characteristics / Key Points](#characteristics--key-points)
  - [Why is it used? / Applications](#why-is-it-used--applications)
- [Solution](#solution)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

---

## Problem Statement

[Calculate Jaccard Index for Binary Classification](https://www.deep-ml.com/problems/72)

Write a Python function `jaccard_index(y_true, y_pred)` that computes the **Jaccard Index**, a measure of similarity between two binary label arrays.

The function should:

- Calculate the Jaccard Index between `y_true` and `y_pred`.
- Return the result as a floating-point number.
- Correctly handle edge cases such as:
  - No overlap between the arrays.
  - Both arrays containing only zeros.

The Jaccard Index measures the overlap between the predicted positive labels and the true positive labels. Its value ranges from **0** (no overlap) to **1** (perfect overlap).

---

## Example

```python
import numpy as np

y_true = np.array([1, 0, 1, 1, 0, 1])
y_pred = np.array([1, 0, 1, 0, 0, 1])

print(jaccard_index(y_true, y_pred))
```

### Output

```text
0.75
```

### Explanation

The intersection contains the positions where both arrays are `1`.

| Index | y_true | y_pred | Included in Intersection? | Included in Union? |
| :---: | :----: | :----: | :-----------------------: | :----------------: |
|   0   |   1    |   1    |             ✓             |         ✓          |
|   1   |   0    |   0    |             ✗             |         ✗          |
|   2   |   1    |   1    |             ✓             |         ✓          |
|   3   |   1    |   0    |             ✗             |         ✓          |
|   4   |   0    |   0    |             ✗             |         ✗          |
|   5   |   1    |   1    |             ✓             |         ✓          |

Intersection:

$$
|A \cap B| = 3
$$

Union:

$$
|A \cup B| = 4
$$

Therefore,

$$
\begin{aligned}
\text{Jaccard Index}
&=
\frac{|A \cap B|}{|A \cup B|} \\
&=
\frac{3}{4} \\
&=
0.75
\end{aligned}
$$

---

# Learn: Jaccard Index

## What is it?

The **Jaccard Index**, also called the **Jaccard Similarity Coefficient**, measures the similarity between two sets.

For binary classification, the sets correspond to the indices where the labels are **1**.

A higher Jaccard Index indicates greater agreement between the predicted positive labels and the true positive labels.

The metric ranges from **0** to **1**, where:

- **0** indicates no overlap.
- **1** indicates a perfect match.

---

## Mathematical Definition / Formula

For two sets \(A\) and \(B\),

$$
\boxed{
\text{Jaccard Index} =
\frac{|A \cap B|}{|A \cup B|} =
\frac{|A \cap B|}
{|A|+|B|-|A \cap B|}
}
$$

For binary classification,

- \(A \cap B\) represents the **True Positives (TP)**.
- \(A \cup B\) represents all positions where either the true label or predicted label is positive.

Equivalently,

$$
\boxed{
\text{Jaccard Index} =
\frac{\text{TP}}
{\text{TP}+\text{FP}+\text{FN}}
}
$$

where

- **TP** = True Positives
- **FP** = False Positives
- **FN** = False Negatives

---

## Characteristics / Key Points

- Values range from **0** to **1**.
- Higher values indicate greater similarity.
- Symmetric:

$$
J(A,B)=J(B,A)
$$

- Ignores True Negatives.
- Commonly used for binary classification and segmentation tasks.

### Interpretation

| Jaccard Index | Meaning          |
| :-----------: | :--------------- |
|     **1**     | Perfect overlap  |
|     **0**     | No overlap       |
|    **0.5**    | Moderate overlap |
|   **0.75**    | High similarity  |

---

## Why is it used? / Applications

The Jaccard Index is commonly used in:

- Binary Classification
- Image Segmentation
- Object Detection Evaluation
- Clustering
- Document Similarity
- Recommendation Systems
- Information Retrieval

It is especially useful when the positive class is much rarer than the negative class.

> 💡 **Important Note**
>
> When both the predicted and true sets are empty, the union is zero, resulting in division by zero. A common convention is to return **0.0** in this case, although some libraries may define the similarity as **1.0** since both sets are identical.

---

# Solution

## Custom Implementation

```python
import numpy as np

def jaccard_index(y_true, y_pred):
    intersection = np.sum(y_true & y_pred)
    union = np.sum(y_true | y_pred)

    if union == 0:
        return 0.0

    return round(float(intersection / union), 3)
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

### Step 2: Compute the Union

```python
union = np.sum(y_true | y_pred)
```

The bitwise OR (`|`) identifies positions where at least one array contains `1`.

The sum gives

$$
|A \cup B|
$$

---

### Step 3: Handle the Empty Union Case

```python
if union == 0:
    return 0.0
```

If both arrays contain only zeros, there are no positive labels, so the function returns `0.0`.

---

### Step 4: Compute the Jaccard Index

```python
return round(float(intersection / union), 3)
```

This applies

$$
\boxed{
\text{Jaccard Index} =
\frac{|A \cap B|}
{|A \cup B|}
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

The algorithm performs one pass to compute the intersection and one pass to compute the union. Aside from a few scalar variables, no additional memory proportional to the input size is allocated.
