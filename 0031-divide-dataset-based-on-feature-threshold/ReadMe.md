# Divide Dataset Based on Feature Threshold (Medium, Machine Learning)

## Table of Contents

- Problem Statement
- Example
- Learn: Understanding Dataset Division Based on Feature Threshold
- Solution
- Code Explanation
- Time & Space Complexity

---

## Problem Statement

### [Divide Dataset Based on Feature Threshold](https://www.deep-ml.com/problems/31)

Write a Python function that divides a dataset into two subsets based on the value of a specified feature and a threshold.

The function should separate the dataset into:

- Samples whose selected feature value is **greater than or equal to** the threshold.
- Samples whose selected feature value is **less than** the threshold.

The returned subsets should preserve the original ordering of the samples.

---

## Example

### Input

```python
X = np.array([
    [1, 2],
    [3, 4],
    [5, 6],
    [7, 8],
    [9, 10]
])

feature_i = 0
threshold = 5
```

### Output

```python
[
    array([
        [5, 6],
        [7, 8],
        [9, 10]
    ]),
    array([
        [1, 2],
        [3, 4]
    ])
]
```

### Reasoning

The dataset is split based on the first feature (`feature_i = 0`).

The first column contains:

```text
1, 3, 5, 7, 9
```

Rows whose first feature is **greater than or equal to 5** are placed in the first subset, while the remaining rows are placed in the second subset.

---

## Learn: Understanding Dataset Division Based on Feature Threshold

### What is it?

Dataset division based on a feature threshold is one of the most fundamental operations in machine learning. It partitions a dataset into two groups according to the value of a selected feature.

Each sample is evaluated independently. If the chosen feature satisfies a threshold condition, the sample is assigned to one subset; otherwise, it is assigned to another.

This simple operation is the foundation of many machine learning algorithms, particularly **Decision Trees**, **Random Forests**, and other tree-based methods where the dataset is recursively partitioned into smaller, more homogeneous groups.

Instead of modifying the original dataset, the algorithm creates two new subsets that can be processed independently.

---

### Mathematical Definition

Suppose the dataset is

$$
X \in \mathbb{R}^{n \times d}
$$

where

- $n$ is the number of samples.
- $d$ is the number of features.

Let

- $i$ be the selected feature index.
- $t$ be the threshold.

The resulting subsets are

$$
X_{\ge t} = \{x \in X \mid x_i \ge t\}
$$

and

$$
X_{< t} = \{x \in X \mid x_i < t\}
$$

where

- $x_i$ is the value of the selected feature.
- $t$ is the threshold.

Each sample belongs to exactly one subset.

---

### How Dataset Division Works

The algorithm performs the following steps:

1. Select the feature column.
2. Compare every value with the threshold.
3. Create a Boolean mask indicating whether each sample satisfies the condition.
4. Use the mask to extract samples meeting the condition.
5. Invert the mask to obtain the remaining samples.

For example,

Dataset

$$
X =
\begin{bmatrix}
1 & 2\\
3 & 4\\
5 & 6\\
7 & 8\\
9 & 10
\end{bmatrix}
$$

Feature index

$$
i = 0
$$

Threshold

$$
t = 5
$$

Selected feature values

$$
[1,\;3,\;5,\;7,\;9]
$$

Boolean mask

$$
[False,\;False,\;True,\;True,\;True]
$$

First subset

$$
\begin{bmatrix}
5 & 6\\
7 & 8\\
9 & 10
\end{bmatrix}
$$

Second subset

$$
\begin{bmatrix}
1 & 2\\
3 & 4
\end{bmatrix}
$$

---

### Characteristics / Key Points

- Divides a dataset into two mutually exclusive subsets.
- Every sample appears exactly once.
- Preserves the original sample order.
- Does not modify the original dataset.
- Can be implemented using loops or vectorized NumPy operations.
- Boolean indexing is significantly faster than Python loops.
- Works with datasets containing any number of features.
- Forms the basic splitting operation used in Decision Trees.

---

### Why is it used? / Applications

Threshold-based partitioning is widely used in machine learning because it separates data into meaningful regions.

Common applications include:

- Decision Tree training
- Random Forests
- Gradient Boosting algorithms
- Feature engineering
- Rule-based classifiers
- Data filtering
- Outlier separation
- Exploratory Data Analysis (EDA)
- Conditional sampling during preprocessing

For example, a Decision Tree may split data using conditions such as

$$
\text{Age} \ge 18
$$

or

$$
\text{Income} \ge 50000
$$

to create groups with similar characteristics before learning further decision rules.

---

> 💡 **Important Note**
>
> Although this operation can be implemented using Python loops, NumPy Boolean indexing is the preferred approach because it executes the filtering using optimized C code. Since tree-based algorithms perform thousands of such splits during training, vectorized implementations provide significantly better performance.

---

## Solution

### Custom Implementation

```python
def divide_on_feature(X, feature_i, threshold):
    left, right = [], []

    for row in X:
        if row[feature_i] >= threshold:
            left.append(row)
        else:
            right.append(row)

    return [left, right]
```

### NumPy Implementation

```python
import numpy as np

def divide_on_feature(X, feature_i, threshold):
    mask = X[:, feature_i] >= threshold
    return [X[mask], X[~mask]]
```

---

## Code Explanation

### Step 1

Select the required feature column.

```python
X[:, feature_i]
```

This extracts all values corresponding to the chosen feature.

---

### Step 2

Compare every feature value with the threshold.

```python
mask = X[:, feature_i] >= threshold
```

This produces a Boolean array where each value is either `True` or `False`.

---

### Step 3

Extract all samples satisfying the condition.

```python
X[mask]
```

Only rows whose selected feature value is greater than or equal to the threshold are returned.

---

### Step 4

Extract the remaining samples.

```python
X[~mask]
```

The `~` operator inverts the Boolean mask, selecting all rows that do not satisfy the condition.

---

### Step 5

Return both subsets.

```python
return [X[mask], X[~mask]]
```

The function returns the dataset divided into two parts while preserving the original sample order.

---

## Time & Space Complexity

| Complexity | Value    |
| ---------- | -------- |
| Time       | **O(n)** |
| Space      | **O(n)** |

where

- $n$ is the number of samples.
- Each sample is examined exactly once to create the Boolean mask, and the returned subsets together contain all original samples.
