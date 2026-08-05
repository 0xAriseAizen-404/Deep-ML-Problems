# Generate Random Subsets of a Dataset (Medium, Machine Learning)

## Table of Contents

- Problem Statement
- Example
- Learn: Understanding Random Subsets of a Dataset
- Solution
- Code Explanation
- Time & Space Complexity

---

## Problem Statement

### [Generate Random Subsets of a Dataset](https://www.deep-ml.com/problems/33)

Write a Python function that generates multiple random subsets of a dataset for use in ensemble learning algorithms such as Bagging and Random Forests.

The function accepts:

- A feature matrix `X`
- A target vector `y`
- The number of subsets to generate
- A Boolean parameter `replacements` controlling the sampling strategy

The sampling behavior is:

- **With replacement (`True`)**
  - Each subset has the same size as the original dataset.
  - A sample may appear multiple times within the same subset.

- **Without replacement (`False`)**
  - Each subset contains half of the original dataset (using integer division).
  - A sample can appear at most once within the same subset.

Return every subset as a tuple containing `(X_subset, y_subset)`, where both are converted to Python lists.

---

## Example

### Input

```python
np.random.seed(42)

X = np.array([
    [1, 2],
    [3, 4],
    [5, 6],
    [7, 8],
    [9, 10]
])

y = np.array([1, 2, 3, 4, 5])

n_subsets = 3
replacements = False
```

### Output

```python
[
    (
        [[3, 4], [9, 10]],
        [2, 5]
    ),
    (
        [[7, 8], [3, 4]],
        [4, 2]
    ),
    (
        [[3, 4], [1, 2]],
        [2, 1]
    )
]
```

### Reasoning

The dataset contains **5 samples**.

Since `replacements=False`, each subset contains

$$
\left\lfloor \frac{5}{2} \right\rfloor = 2
$$

samples.

Within each subset, every sample appears at most once, although the same sample may appear in different subsets because each subset is generated independently.

---

## Learn: Understanding Random Subsets of a Dataset

### What is it?

Random subset generation is the process of repeatedly selecting samples from a dataset to create multiple smaller datasets. Each subset can then be used to train an independent machine learning model.

This idea is fundamental to **ensemble learning**, where multiple models are trained on different views of the same dataset and their predictions are combined to produce a more accurate and robust final model.

Depending on the sampling strategy, subsets may allow duplicate samples (**sampling with replacement**) or enforce unique samples (**sampling without replacement**).

---

### Mathematical Definition

Suppose the dataset is

$$
X \in \mathbb{R}^{n \times d}
$$

with corresponding labels

$$
y \in \mathbb{R}^{n}
$$

where

- $n$ is the number of samples.
- $d$ is the number of features.

If sampling **with replacement**, each subset size is

$$
m = n
$$

Each selected index satisfies

$$
i_j \in \{0,1,\ldots,n-1\}
$$

where duplicate indices are allowed.

If sampling **without replacement**, the subset size becomes

$$
m = \left\lfloor \frac{n}{2} \right\rfloor
$$

and every selected index is unique.

---

### Sampling With Replacement

Sampling with replacement is also known as **Bootstrap Sampling**.

After selecting a sample, it is placed back into the sampling pool, allowing it to be selected again.

For example,

Original dataset

```text
A  B  C  D  E
```

One bootstrap sample

```text
A  D  B  D  E
```

Notice that

- `D` appears twice.
- `C` does not appear.

This sampling strategy is the foundation of **Bagging** and **Random Forests**.

---

### Sampling Without Replacement

When sampling without replacement, every selected sample is removed from the pool.

Original dataset

```text
A  B  C  D  E
```

Possible subset

```text
B  E
```

Properties

- No duplicates inside the subset.
- Maximum subset size equals the number of available samples.
- Every selected sample is unique within that subset.

---

### How Random Subset Generation Works

The algorithm follows these steps:

1. Determine the subset size.
2. Randomly generate sample indices.
3. Extract the corresponding feature vectors.
4. Extract the corresponding labels.
5. Repeat until the required number of subsets has been generated.

For example,

Dataset

```text
A  B  C  D  E
```

Without replacement

```text
Subset 1 → B E

Subset 2 → D B

Subset 3 → B A
```

With replacement

```text
Subset 1 → A C C E A

Subset 2 → D B D A C

Subset 3 → E E C B D
```

---

### Characteristics / Key Points

- Produces multiple independent subsets.
- Preserves the correspondence between features and labels.
- Supports both replacement and non-replacement sampling.
- Bootstrap sampling allows duplicate samples.
- Sampling without replacement guarantees unique samples within a subset.
- Each subset can be used to train a separate model.
- Randomness improves model diversity.

---

### Why is it used? / Applications

Random subset generation is a core component of many machine learning algorithms.

Applications include:

- Bagging (Bootstrap Aggregating)
- Random Forests
- Ensemble Learning
- Cross-validation
- Bootstrap estimation
- Distributed model training
- Data augmentation through resampling
- Variance reduction in predictive models

Training multiple models on different subsets reduces overfitting and generally improves model generalization.

---

> 💡 **Important Note**
>
> "With replacement" only means that a sample can appear multiple times **within the same subset**. Different subsets are always generated independently, so the same sample may naturally appear in multiple subsets regardless of the sampling strategy. This distinction is a common interview question.

---

## Solution

### Custom Implementation

```python
import numpy as np

def get_random_subsets(X, y, n_subsets, replacements=True):
    n = len(X)

    subset_size = n if replacements else n // 2

    subsets = []

    for _ in range(n_subsets):
        indices = np.random.choice(
            n,
            size=subset_size,
            replace=replacements
        )

        subsets.append(
            (
                X[indices].tolist(),
                y[indices].tolist()
            )
        )

    return subsets
```

### Vectorized NumPy Implementation

```python
import numpy as np

def get_random_subsets(X, y, n_subsets, replacements=True):
    n, _ = X.shape

    subset_size = n if replacements else n // 2

    idx = np.array([
        np.random.choice(
            n,
            subset_size,
            replace=replacements
        )
        for _ in range(n_subsets)
    ])

    return [
        (
            X[idx[i]].tolist(),
            y[idx[i]].tolist()
        )
        for i in range(n_subsets)
    ]
```

---

## Code Explanation

### Step 1

Determine the number of samples.

```python
n = X.shape[0]
```

This is used to compute the subset size and generate valid sample indices.

---

### Step 2

Compute the subset size.

```python
subset_size = n if replacements else n // 2
```

- With replacement, each subset has the same size as the original dataset.
- Without replacement, each subset contains half of the samples.

---

### Step 3

Generate random indices.

```python
np.random.choice(
    n,
    subset_size,
    replace=replacements
)
```

This randomly selects sample indices according to the specified sampling strategy.

---

### Step 4

Store the generated indices.

The list comprehension creates one random index array for every required subset.

```python
idx = np.array([
    ...
])
```

Each row of `idx` represents one subset.

---

### Step 5

Extract feature and label subsets.

```python
X[idx[i]]
```

selects the corresponding feature vectors.

Similarly,

```python
y[idx[i]]
```

retrieves the matching labels, ensuring every feature remains paired with its correct target.

---

### Step 6

Convert the subsets to Python lists.

```python
.tolist()
```

The problem requires each returned subset to contain standard Python lists instead of NumPy arrays.

---

### Step 7

Return all generated subsets.

The function returns a list containing `n_subsets` tuples, where each tuple consists of one feature subset and its corresponding label subset.

---

## Time & Space Complexity

| Complexity | Value        |
| ---------- | ------------ |
| Time       | **O(s × m)** |
| Space      | **O(s × m)** |

where

- $n$ is the number of samples.
- $s$ is the number of subsets.
- $m$ is the size of each subset.

Each subset requires generating `m` random indices and extracting the corresponding samples, while all generated subsets are stored in memory.
