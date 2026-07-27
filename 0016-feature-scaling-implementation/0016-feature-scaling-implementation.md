# Feature Scaling Implementation (Easy, Machine Learning)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Feature Scaling](#learn-feature-scaling)
- [Solutions](#solutions)
  - [NumPy Implementation](#numpy-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

---

## Problem Statement

[Feature Scaling Implementation](https://www.deep-ml.com/problems/16)

Write a Python function that performs **Standardization** and **Min-Max Normalization** on a dataset. Return both scaled datasets, rounded to **4 decimal places**.

---

## Example

```python
Input:

data = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])

Output:

(
    [[-1.2247, -1.2247],
     [ 0.0000,  0.0000],
     [ 1.2247,  1.2247]],

    [[0.0, 0.0],
     [0.5, 0.5],
     [1.0, 1.0]]
)
```

**Reasoning**

- **Standardization** scales each feature to have mean **0** and standard deviation **1**.
- **Min-Max Normalization** scales each feature to the range **[0, 1]**.

---

## Learn: Feature Scaling

Feature scaling ensures that all features contribute equally during training.

**Standardization (Z-score)**

```text
z = (x - μ) / σ
```

**Min-Max Normalization**

```text
x' = (x - min) / (max - min)
```

**When to Use**

- **Standardization** → Linear Regression, Logistic Regression, SVM, PCA, Neural Networks.
- **Min-Max Normalization** → KNN, K-Means, Neural Networks (when bounded inputs are preferred).

---

## Solutions

### NumPy Implementation

```python
import numpy as np


def feature_scaling(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    standardized_data = (
        (data - np.mean(data, axis=0))
        / np.std(data, axis=0)
    ).round(4)

    normalized_data = (
        (data - np.min(data, axis=0))
        / (np.max(data, axis=0) - np.min(data, axis=0))
    ).round(4)

    return standardized_data, normalized_data


# Example Usage
data = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])

standardized, normalized = feature_scaling(data)

print(standardized)
print(normalized)
```

---

## Code Explanation

- Compute the **mean** and **standard deviation** for each feature.

```python
(data - np.mean(data, axis=0)) / np.std(data, axis=0)
```

- Compute the **minimum** and **maximum** for each feature.

```python
(data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
```

- Round both outputs to **4 decimal places**.

---

## Time & Space Complexity

| Complexity | Value |
|------------|-------|
| Time | **O(m × n)** |
| Space | **O(m × n)** |

Where:

- **m** = Number of samples
- **n** = Number of features