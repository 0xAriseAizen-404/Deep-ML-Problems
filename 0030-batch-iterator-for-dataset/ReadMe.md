# Batch Iterator for Dataset (Easy, Machine Learning)

## Table of Contents

- Problem Statement
- Example
- Learn: Understanding Batch Iteration
- Solution
- Code Explanation
- Time & Space Complexity

## Problem Statement

### [Batch Iterator for Dataset](https://www.deep-ml.com/problems/30)

Write a Python function that splits a dataset into smaller batches.

The function should accept:

- A feature matrix **X**
- An optional label vector **y**
- A batch size

If **y** is provided, return batches as **(X_batch, y_batch)** pairs. Otherwise, return batches containing only **X**.

The final batch may contain fewer samples if the dataset size is not divisible by the batch size.

---

## Example

**Input**

```python
X = np.array([
    [1, 2],
    [3, 4],
    [5, 6],
    [7, 8],
    [9, 10]
])

y = np.array([1, 2, 3, 4, 5])

batch_size = 2
```

**Output**

```python
[
    [
        [[1, 2], [3, 4]],
        [1, 2]
    ],
    [
        [[5, 6], [7, 8]],
        [3, 4]
    ],
    [
        [[9, 10]],
        [5]
    ]
]
```

**Reasoning**

The dataset contains **5 samples**.

Using a batch size of **2**, the data is divided into:

- Batch 1 → 2 samples
- Batch 2 → 2 samples
- Batch 3 → 1 remaining sample

The labels are sliced using the same indices, preserving the correspondence between features and targets.

---

## Learn: Understanding Batch Iteration

### What is Batch Iteration?

Batch iteration is the process of dividing a dataset into smaller subsets called **batches**.

Instead of processing every training sample at once, machine learning algorithms process one batch at a time.

This reduces memory usage and enables efficient optimization for large datasets.

---

### Mathematical Definition

Suppose the dataset contains

$$
m
$$

training samples.

If the batch size is

$$
b
$$

then the number of batches is

$$
\left\lceil \frac{m}{b} \right\rceil
$$

Each batch contains at most

$$
b
$$

samples.

For the $k^{th}$ batch,

$$
X_k = X[kb:(k+1)b]
$$

Similarly,

$$
y_k = y[kb:(k+1)b]
$$

The last batch may contain fewer than $b$ samples.

---

### How Batch Iteration Works

The algorithm follows three simple steps.

1. Start from the first sample.

2. Slice **batch_size** consecutive samples.

3. Continue until all samples have been processed.

For example,

```text
Dataset:

[1][2][3][4][5][6][7]
```

Using

```text
batch_size = 3
```

produces

```text
Batch 1 → [1 2 3]

Batch 2 → [4 5 6]

Batch 3 → [7]
```

---

### Why Not Process the Entire Dataset?

For small datasets, processing everything at once is possible.

However, modern datasets may contain millions of samples.

Batch processing offers several advantages:

- Lower memory consumption
- Faster training
- GPU-friendly computation
- Stable gradient estimation
- Efficient parallel processing

---

### Batch Gradient Descent vs Mini-Batch Gradient Descent

There are three common optimization strategies.

**Batch Gradient Descent**

Uses every training sample before updating the model.

$$
\text{Batch Size} = m
$$

---

**Mini-Batch Gradient Descent**

Uses a fixed number of samples per update.

$$
1 < b < m
$$

Typical batch sizes are

```text
16
32
64
128
256
```

---

**Stochastic Gradient Descent (SGD)**

Updates the model after every single sample.

$$
b = 1
$$

---

### Characteristics / Key Points

- Splits large datasets into manageable pieces.
- Preserves feature-label correspondence.
- The final batch can be smaller.
- Requires only simple array slicing.
- Commonly used in deep learning training loops.
- Reduces peak memory usage.

---

### Why is it used? / Applications

Batch iteration is fundamental in machine learning.

Applications include:

- Neural Network training
- Mini-Batch Gradient Descent
- Stochastic Gradient Descent
- PyTorch DataLoader
- TensorFlow Dataset API
- GPU training
- Large-scale data processing
- Distributed machine learning

---

> 💡 **Important Note**
>
> A batch iterator only divides the dataset into batches. It does **not** shuffle the data automatically. In practice, datasets are usually shuffled first and then iterated in batches to avoid learning biases caused by ordered samples.

---

## Solution

### Custom Implementation

```python
import numpy as np

def batch_iterator(X, y=None, batch_size=64):
    batches = []

    for ind in range(0, len(X), batch_size):
        if y is not None:
            batches.append([
                list(X[ind: ind + batch_size]),
                list(y[ind: ind + batch_size]),
            ])
        else:
            batches.append(
                list(X[ind: ind + batch_size])
            )

    return batches
```

---

## Code Explanation

### Step 1

Create an empty list to store all batches.

```python
batches = []
```

---

### Step 2

Iterate through the dataset with a step size equal to the batch size.

```python
for ind in range(0, len(X), batch_size):
```

Each iteration marks the starting index of a new batch.

---

### Step 3

Extract a slice of the feature matrix.

```python
X[ind : ind + batch_size]
```

This selects at most **batch_size** samples.

---

### Step 4

If labels are provided, slice them using the same indices.

```python
y[ind : ind + batch_size]
```

Using identical slices keeps every feature vector paired with its correct label.

---

### Step 5

Store each batch.

- If **y** exists, store both feature and label batches.
- Otherwise, store only the feature batch.

---

### Step 6

Continue until the end of the dataset.

The final batch automatically contains the remaining samples, even if its size is smaller than the specified batch size.

---

## Time & Space Complexity

| Complexity | Value    |
| ---------- | -------- |
| Time       | **O(m)** |
| Space      | **O(m)** |

where

- **m** is the number of samples in the dataset.
- Every sample is visited exactly once, and all batches together store the entire dataset.
