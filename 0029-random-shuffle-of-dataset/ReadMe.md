# Random Shuffle of Dataset (Easy, Machine Learning)

## Table of Contents

- Problem Statement
- Example
- Learn: Understanding Dataset Shuffling
- Solution
- Code Explanation
- Time & Space Complexity

## Problem Statement

### [Random Shuffle of Dataset](https://www.deep-ml.com/problems/29)

Write a Python function that randomly shuffles the samples of a dataset while preserving the correspondence between the feature matrix **X** and target vector **y**.

The function should:

- Accept feature matrix **X** and label vector **y**.
- Randomly reorder the samples.
- Keep each feature vector matched with its correct label.
- Support an optional random seed for reproducible results.

---

## Example

**Input**

```python
X = np.array([
    [1, 2],
    [3, 4],
    [5, 6],
    [7, 8]
])

y = np.array([1, 2, 3, 4])
```

**Output**

```python
(
    array([
        [5, 6],
        [1, 2],
        [7, 8],
        [3, 4]
    ]),
    array([3, 1, 4, 2])
)
```

**Reasoning**

Instead of shuffling **X** and **y** separately, generate one random permutation of indices and use it to reorder both arrays. This preserves the correct mapping between every feature vector and its label.

---

## Learn: Understanding Dataset Shuffling

### What is Dataset Shuffling?

Dataset shuffling is the process of randomly rearranging the order of training samples before feeding them to a machine learning algorithm.

Without shuffling, the data may contain hidden ordering patterns that can negatively affect model training.

For example, if all positive samples appear before all negative samples, the model may learn biased updates during training.

---

### Mathematical Definition

Suppose the dataset contains

$$
m
$$

training samples.

The feature matrix is

$$
X=
\begin{bmatrix}
x_1\\
x_2\\
\vdots\\
x_m
\end{bmatrix}
$$

and the target vector is

$$
y=
\begin{bmatrix}
y_1\\
y_2\\
\vdots\\
y_m
\end{bmatrix}
$$

Generate a random permutation

$$
\pi=(\pi_1,\pi_2,\ldots,\pi_m)
$$

Then reorder both arrays using the same permutation.

$$
X' = X[\pi]
$$

$$
y' = y[\pi]
$$

Since both arrays use identical indices, every sample remains paired with its correct label.

---

### Algorithm

The shuffling process consists of three simple steps.

1. Generate an index array.

```python
[0, 1, 2, ..., m-1]
```

2. Randomly permute the indices.

```python
[2, 0, 3, 1]
```

3. Use these indices to reorder both arrays.

```python
X = X[indices]
y = y[indices]
```

---

### Random Seed

Machine learning experiments often require reproducibility.

Setting a random seed initializes the random number generator to a fixed state.

```python
np.random.seed(42)
```

Using the same seed always produces the same shuffled order.

Without a seed, every execution generates a different permutation.

---

### Characteristics / Key Points

- Randomizes the training order.
- Preserves feature-label correspondence.
- Improves stochastic optimization.
- Prevents learning biases caused by ordered data.
- Supports reproducible experiments using a fixed random seed.
- Uses the same permutation for both **X** and **y**.

---

### Why is it used? / Applications

Dataset shuffling is an essential preprocessing step in many machine learning workflows.

Applications include:

- Gradient Descent
- Stochastic Gradient Descent (SGD)
- Mini-Batch Gradient Descent
- Neural Network training
- Cross-validation
- Train-test splitting
- Data preprocessing pipelines
- Deep learning data loaders

---

> 💡 **Important Note**
>
> Never shuffle **X** and **y** independently. Doing so destroys the relationship between each sample and its label, producing incorrect training data. Always generate one permutation and apply it to both arrays.

---

## Solution

### NumPy Implementation

```python
import numpy as np

def shuffle_data(X, y, seed=None):
    np.random.seed(seed)
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]
```

---

## Code Explanation

### Step 1

Initialize the random number generator.

```python
np.random.seed(seed)
```

If a seed is provided, the shuffle becomes deterministic and reproducible.

---

### Step 2

Generate a random permutation of sample indices.

```python
indices = np.random.permutation(len(X))
```

For example,

```python
[0, 1, 2, 3]
```

may become

```python
[2, 0, 3, 1]
```

---

### Step 3

Reorder the feature matrix.

```python
X[indices]
```

Each row moves to its new shuffled position.

---

### Step 4

Reorder the labels using the same permutation.

```python
y[indices]
```

Since both arrays use identical indices, every feature vector stays paired with its corresponding target value.

---

### Step 5

Return the shuffled dataset.

The function returns the shuffled feature matrix and label vector.

---

## Time & Space Complexity

| Complexity | Value    |
| ---------- | -------- |
| Time       | **O(m)** |
| Space      | **O(m)** |

where

- **m** is the number of samples in the dataset.
- Generating the permutation and indexing both arrays each require linear time.
- The shuffled arrays require additional linear space.
