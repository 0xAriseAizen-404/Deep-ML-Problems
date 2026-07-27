# One-Hot Encoding of Nominal Values (Easy, Machine Learning)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: One-Hot Encoding](#learn-one-hot-encoding)
- [Solutions](#solutions)
  - [Python Implementation](#python-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

---

## Problem Statement

[One-Hot Encoding of Nominal Values](https://www.deep-ml.com/problems/34)

Write a Python function to perform **One-Hot Encoding** on a 1D NumPy array of integer labels. If the number of columns is not provided, determine it automatically from the input.

---

## Example

```python
Input:

x = np.array([0, 1, 2, 1, 0])

Output:

[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]
 [0. 1. 0.]
 [1. 0. 0.]]
```

**Reasoning**

Each integer label is converted into a binary vector where only its corresponding index is **1** and all others are **0**.

---

## Learn: One-Hot Encoding

One-Hot Encoding converts **categorical values** into binary vectors.

For categories **0, 1, 2**:

```text
0 → [1, 0, 0]

1 → [0, 1, 0]

2 → [0, 0, 1]
```

It prevents machine learning models from assuming an ordinal relationship between categories.

---

## Solutions

### Python Implementation

```python
import numpy as np


def to_categorical(x, n_col=None):

    if n_col is None:
        n_col = np.max(x) + 1

    one_hot = []

    for value in x:
        row = [0] * n_col
        row[value] = 1
        one_hot.append(row)

    return np.array(one_hot)


# Example Usage
x = np.array([0, 1, 2, 1, 0])

print(to_categorical(x))
```

---

## Code Explanation

- Determine the number of categories if not provided.

```python
n_col = np.max(x) + 1
```

- Create a zero-filled row.

```python
row = [0] * n_col
```

- Set the corresponding category index to **1**.

```python
row[value] = 1
```

- Convert the list of rows into a NumPy array.

---

## Time & Space Complexity

| Complexity | Value |
|------------|-------|
| Time | **O(n × k)** |
| Space | **O(n × k)** |

Where:

- **n** = Number of samples
- **k** = Number of categories