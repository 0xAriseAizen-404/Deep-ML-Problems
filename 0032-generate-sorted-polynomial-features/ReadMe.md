# Generate Sorted Polynomial Features (Medium, Machine Learning)

## Table of Contents

- Problem Statement
- Example
- Learn: Understanding Polynomial Features
- Solution
- Code Explanation
- Time & Space Complexity

---

## Problem Statement

### [Generate Sorted Polynomial Features](https://www.deep-ml.com/problems/32)

Write a Python function that generates polynomial features for a given dataset up to a specified degree and sorts the generated features of each sample in ascending order.

The function should:

- Accept a 2-D NumPy array `X`.
- Generate all polynomial combinations of the input features from degree `0` up to the specified degree.
- Sort the generated polynomial features for every sample.
- Return the transformed dataset as a new 2-D NumPy array.

---

## Example

### Input

```python
X = np.array([
    [2, 3],
    [3, 4],
    [5, 6]
])

degree = 2
```

### Output

```python
[
    [1., 2., 3., 4., 6., 9.],
    [1., 3., 4., 9., 12., 16.],
    [1., 5., 6., 25., 30., 36.]
]
```

### Reasoning

For the first sample

```text
(2, 3)
```

The polynomial terms up to degree 2 are

$$
[1,\;2,\;3,\;2^2,\;2\times3,\;3^2]
=
[1,\;2,\;3,\;4,\;6,\;9]
$$

Sorting these values in ascending order produces the same sequence.

The same procedure is applied independently to every sample.

---

## Learn: Understanding Polynomial Features

### What is it?

Polynomial feature generation is a feature engineering technique that transforms existing features into higher-degree polynomial combinations. Instead of learning only linear relationships, a model can learn curved or more complex decision boundaries.

Suppose a dataset contains features $x_1$ and $x_2$. Expanding the dataset to degree 2 creates additional features representing squared terms and feature interactions.

This transformation does **not** change the learning algorithm itself. Instead, it changes the feature space so that even a linear model can capture nonlinear relationships.

After generating the polynomial features, this problem additionally requires sorting every sample's expanded features in ascending order.

---

### Mathematical Definition

Suppose

$$
X \in \mathbb{R}^{n \times d}
$$

where

- $n$ is the number of samples.
- $d$ is the number of features.

For a feature vector

$$
x =
[x_1,\;x_2,\;\cdots,\;x_d]
$$

Polynomial expansion generates every monomial whose total degree is less than or equal to $k$.

For two variables and degree 2, the transformed feature vector becomes

$$
[1,\;x_1,\;x_2,\;x_1^2,\;x_1x_2,\;x_2^2]
$$

The total number of generated polynomial features is

$$
\binom{d+k}{k}
$$

where

- $d$ is the number of original features.
- $k$ is the maximum polynomial degree.

---

### How Polynomial Features Work

The algorithm follows these steps:

1. Generate every feature index combination from degree 0 to the specified degree.
2. Compute the product of the selected feature values.
3. Store every generated polynomial term.
4. Repeat for every sample.
5. Sort each row independently.

For example,

Original sample

$$
[2,\;3]
$$

Degree

$$
2
$$

Generated terms

$$
[1,\;2,\;3,\;4,\;6,\;9]
$$

Sorted output

$$
[1,\;2,\;3,\;4,\;6,\;9]
$$

For another sample

$$
[3,\;4]
$$

Generated terms

$$
[1,\;3,\;4,\;9,\;12,\;16]
$$

Again, sorting produces

$$
[1,\;3,\;4,\;9,\;12,\;16]
$$

---

### Feature Combinations

Polynomial terms are created using combinations with replacement.

For two features

```text
x₁, x₂
```

Degree 0

```text
()
```

Degree 1

```text
(x₁)
(x₂)
```

Degree 2

```text
(x₁,x₁)
(x₁,x₂)
(x₂,x₂)
```

These correspond to

$$
1,\;x_1,\;x_2,\;x_1^2,\;x_1x_2,\;x_2^2
$$

Using combinations with replacement avoids generating duplicate interaction terms such as

$$
x_1x_2
$$

and

$$
x_2x_1
$$

which are mathematically identical.

---

### Characteristics / Key Points

- Expands the feature space with polynomial terms.
- Includes a constant bias feature.
- Generates interaction terms automatically.
- Produces unique feature combinations.
- Higher polynomial degrees create exponentially more features.
- Sorting is performed independently for every sample.
- Useful for capturing nonlinear relationships.

---

### Why is it used? / Applications

Polynomial feature expansion is widely used in machine learning for models that cannot naturally learn nonlinear relationships.

Common applications include

- Polynomial Regression
- Linear Regression with nonlinear data
- Logistic Regression
- Feature engineering
- Kernel approximation
- Scientific data modeling
- Financial forecasting
- Engineering prediction tasks

Sorting the generated features can also simplify certain preprocessing pipelines, histogram-based algorithms, or feature comparison tasks.

---

> 💡 **Important Note**
>
> Polynomial expansion can dramatically increase the number of features. While higher degrees allow models to capture more complex patterns, they also increase memory usage, computation time, and the risk of overfitting. In practice, degrees 2 or 3 are usually sufficient for most machine learning tasks.

---

## Solution

### Custom Implementation

```python
import numpy as np
from itertools import combinations_with_replacement

def polynomial_features(X, degree):
    n_samples, n_features = X.shape

    combs = [
        c
        for d in range(degree + 1)
        for c in combinations_with_replacement(range(n_features), d)
    ]

    X_poly = np.empty((n_samples, len(combs)))

    for i, idx in enumerate(combs):
        if len(idx) == 0:
            X_poly[:, i] = 1
        else:
            X_poly[:, i] = np.prod(X[:, idx], axis=1)

    return np.sort(X_poly, axis=1)
```

---

## Code Explanation

### Step 1

Determine the number of samples and features.

```python
n_samples, n_features = X.shape
```

These values are needed to generate every valid polynomial combination.

---

### Step 2

Generate all feature index combinations.

```python
combinations_with_replacement(
    range(n_features),
    d
)
```

The combinations include every polynomial term from degree `0` through the specified degree.

---

### Step 3

Create an output matrix.

```python
X_poly = np.empty(...)
```

Each column stores one polynomial feature, while each row corresponds to one sample.

---

### Step 4

Compute every polynomial feature.

For each combination,

```python
np.prod(X[:, idx], axis=1)
```

multiplies the selected feature values together.

When the combination is empty,

```python
()
```

the corresponding feature is simply

```python
1
```

which represents the bias term.

---

### Step 5

Sort every sample.

```python
np.sort(X_poly, axis=1)
```

Sorting is performed row-wise, ensuring each sample's polynomial features appear in ascending order.

---

### Step 6

Return the transformed dataset.

The final matrix contains all polynomial features sorted independently for every sample.

---

## Time & Space Complexity

| Complexity | Value                |
| ---------- | -------------------- |
| Time       | **O(nC + nC log C)** |
| Space      | **O(nC)**            |

where

- $n$ is the number of samples.
- $d$ is the number of original features.
- $k$ is the polynomial degree.
- $C = \binom{d+k}{k}$ is the total number of generated polynomial features.

The first term corresponds to generating all polynomial features, while the second term comes from sorting each row independently.
