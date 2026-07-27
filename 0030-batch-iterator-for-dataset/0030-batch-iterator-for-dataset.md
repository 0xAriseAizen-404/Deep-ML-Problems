# Batch Iterator for Dataset (Easy, Machine Learning)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Batch Iterator](#learn-batch-iterator)
- [Solutions](#solutions)
  - [Python Implementation](#python-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

---

## Problem Statement

[Batch Iterator for Dataset](https://www.deep-ml.com/problems/30)

Write a Python function that returns batches from a dataset **X** and an optional label array **y**. If **y** is provided, each batch should contain both features and labels; otherwise, return batches of **X** only.

---

## Example

```python
Input:

X = np.array([[1, 2],
              [3, 4],
              [5, 6],
              [7, 8],
              [9, 10]])

y = np.array([1, 2, 3, 4, 5])

batch_size = 2

Output:

[[[[1, 2], [3, 4]], [1, 2]],
 [[[5, 6], [7, 8]], [3, 4]],
 [[[9, 10]], [5]]]
```

**Reasoning**

The dataset is divided into batches of size **2**. The last batch contains the remaining samples.

---

## Learn: Batch Iterator

Batch iteration processes data in **small chunks** instead of loading the entire dataset at once.

**Steps**

1. Iterate through the dataset using the batch size.
2. Slice the feature array **X**.
3. Slice **y** (if provided) using the same indices.
4. Return all batches.

---

## Solutions

### Python Implementation

```python
import numpy as np


def batch_iterator(X, y=None, batch_size=64):

    batches = []

    for i in range(0, len(X), batch_size):

        if y is not None:
            batches.append([
                list(X[i:i + batch_size]),
                list(y[i:i + batch_size])
            ])
        else:
            batches.append(
                list(X[i:i + batch_size])
            )

    return batches


# Example Usage
X = np.array([[1, 2],
              [3, 4],
              [5, 6],
              [7, 8],
              [9, 10]])

y = np.array([1, 2, 3, 4, 5])

print(batch_iterator(X, y, batch_size=2))
```

---

## Code Explanation

- Iterate over the dataset in steps of `batch_size`.

```python
for i in range(0, len(X), batch_size):
```

- Slice **X** and **y** for each batch.

```python
X[i:i + batch_size]
y[i:i + batch_size]
```

- Return only **X** if labels are not provided.

---

## Time & Space Complexity

| Complexity | Value |
|------------|-------|
| Time | **O(n)** |
| Space | **O(n)** |

Where **n** is the number of samples.