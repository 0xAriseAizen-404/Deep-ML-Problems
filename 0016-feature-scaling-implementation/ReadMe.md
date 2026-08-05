# Feature Scaling Implementation (Easy, Machine Learning)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Feature Scaling](#learn-feature-scaling)
- [Solutions](#solutions)
  - [Custom Implementation](#custom-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

## Problem Statement

### [Feature Scaling Implementation](https://www.deep-ml.com/problems/16)

Write a Python function that performs **feature scaling** on a dataset using both **Standardization (Z-score Normalization)** and **Min-Max Normalization**.

The input is a 2D NumPy array where:

- Each row represents a data sample.
- Each column represents a feature.

The function should return two NumPy arrays:

1. The standardized dataset.
2. The min-max normalized dataset.

All values should be rounded to **4 decimal places**.

---

## Example

### Input

```python
data = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])
```

### Output

```python
(
    [
        [-1.2247, -1.2247],
        [ 0.0000,  0.0000],
        [ 1.2247,  1.2247]
    ],
    [
        [0.0, 0.0],
        [0.5, 0.5],
        [1.0, 1.0]
    ]
)
```

### Reasoning

For **Standardization**, each feature is transformed so that it has

- Mean = 0
- Standard Deviation = 1

For **Min-Max Normalization**, each feature is rescaled into the range

$$
[0,1]
$$

where the smallest value becomes 0 and the largest becomes 1.

---

## Learn: Feature Scaling

### What is Feature Scaling?

Feature Scaling is a preprocessing technique that transforms numerical features so they lie on a similar scale.

Many Machine Learning algorithms assume that all input features have comparable magnitudes. If one feature has much larger values than another, it can dominate the learning process.

Feature scaling helps algorithms learn faster and often improves model performance.

---

### Why is Feature Scaling Needed?

Consider the dataset

| Age |  Salary |
| --: | ------: |
|  20 |  30,000 |
|  35 |  80,000 |
|  45 | 120,000 |

The salary values are much larger than the age values.

Algorithms based on distances or gradients may treat **Salary** as much more important simply because of its larger numerical range.

Feature scaling removes this imbalance.

---

### Standardization (Z-score Normalization)

Standardization transforms a feature so that it has

- Mean = 0
- Standard Deviation = 1

The transformation is

$$
z=\frac{x-\mu}{\sigma}
$$

where

- $x$ is the original value.
- $\mu$ is the feature mean.
- $\sigma$ is the feature standard deviation.

After standardization,

$$
\mu=0
$$

and

$$
\sigma=1
$$

approximately.

---

### Example of Standardization

Suppose

```text
Feature = [1, 3, 5]
```

Mean

$$
\mu=3
$$

Standard deviation

$$
\sigma\approx1.633
$$

Standardized values become

```text
[-1.2247, 0, 1.2247]
```

The feature now has zero mean and unit variance.

---

### Min-Max Normalization

Min-Max Normalization rescales every feature into a fixed interval.

The most common interval is

$$
[0,1]
$$

The transformation is

$$
x'=\frac{x-x_{\min}}{x_{\max}-x_{\min}}
$$

where

- $x_{\min}$ is the minimum feature value.
- $x_{\max}$ is the maximum feature value.

After normalization,

- minimum value becomes 0.
- maximum value becomes 1.

---

### Example of Min-Max Scaling

Given

```text
Feature = [1, 3, 5]
```

The transformed values become

```text
[0, 0.5, 1]
```

The relative ordering of the data remains unchanged.

---

### Standardization vs Min-Max Normalization

| Standardization                                   | Min-Max Normalization                              |
| ------------------------------------------------- | -------------------------------------------------- |
| Mean becomes 0                                    | Minimum becomes 0                                  |
| Standard deviation becomes 1                      | Maximum becomes 1                                  |
| Values are unbounded                              | Values lie in [0,1]                                |
| Less affected by large feature ranges             | Sensitive to outliers                              |
| Often used for Gradient Descent and Linear Models | Common for Neural Networks and image preprocessing |

---

### Which Algorithms Need Feature Scaling?

Feature scaling is particularly important for:

- Linear Regression (Gradient Descent)
- Logistic Regression
- K-Means Clustering
- K-Nearest Neighbors (KNN)
- Support Vector Machines (SVM)
- Neural Networks
- Principal Component Analysis (PCA)

Algorithms such as Decision Trees and Random Forests generally do **not** require feature scaling.

---

### Characteristics / Key Points

- Prevents large-valued features from dominating learning.
- Improves convergence of Gradient Descent.
- Makes distance calculations fair.
- Applied independently to each feature.
- Does not change the relationships between samples.
- Usually performed before model training.

---

### Why is it Used?

Feature scaling is an essential preprocessing step in Machine Learning pipelines.

Applications include:

- Linear Regression
- Logistic Regression
- KNN Classification
- K-Means Clustering
- PCA
- Neural Networks
- Deep Learning
- Image preprocessing
- Recommendation systems

Most real-world datasets undergo feature scaling before training.

---

> 💡 **Important Note**
>
> Always compute the scaling statistics (**mean, standard deviation, minimum, and maximum**) **only on the training dataset**. Apply those same values to the validation and test datasets. Recomputing them separately on test data causes **data leakage**, leading to overly optimistic evaluation results.

---

## Solutions

### Custom Implementation

```python
import numpy as np

def feature_scaling(
    data: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:

    standardized_data = (
        (data - np.mean(data, axis=0))
        / np.std(data, axis=0)
    ).round(4)

    normalized_data = (
        (data - np.min(data, axis=0))
        / (
            np.max(data, axis=0)
            - np.min(data, axis=0)
        )
    ).round(4)

    return standardized_data, normalized_data
```

---

## Code Explanation

### 1. Compute the Mean

```python
np.mean(data, axis=0)
```

The mean is calculated independently for every feature (column).

This computes

$$
\mu=\frac{1}{m}\sum_{i=1}^{m}x_i
$$

---

### 2. Compute the Standard Deviation

```python
np.std(data, axis=0)
```

The standard deviation measures the spread of each feature.

It is used in

$$
z=\frac{x-\mu}{\sigma}
$$

to standardize the data.

---

### 3. Standardize Every Feature

```python
(data - mean) / std
```

Each feature is transformed independently so that it has approximately

- Mean = 0
- Standard Deviation = 1

---

### 4. Compute Feature Minimum and Maximum

```python
np.min(data, axis=0)

np.max(data, axis=0)
```

These values define the lower and upper bounds used for Min-Max scaling.

---

### 5. Normalize Every Feature

```python
(data - minimum) / (maximum - minimum)
```

This implements

$$
x'=\frac{x-x_{\min}}{x_{\max}-x_{\min}}
$$

ensuring every feature lies within

$$
[0,1]
$$

---

### 6. Round the Results

```python
.round(4)
```

The standardized and normalized datasets are rounded to four decimal places before being returned.

---

## Time & Space Complexity

Let

- $m$ = number of samples.
- $n$ = number of features.

Each scaling method processes every element of the dataset once.

| Complexity | Value        |
| ---------- | ------------ |
| Time       | **O(m × n)** |
| Space      | **O(m × n)** |

The additional space is required to store the transformed datasets.
