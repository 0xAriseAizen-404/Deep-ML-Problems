# Implement K-Fold Cross-Validation (Medium, Machine Learning)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: K-Fold Cross-Validation](#learn-k-fold-cross-validation)
  - [What is it?](#what-is-it)
  - [How K-Fold Cross-Validation Works](#how-k-fold-cross-validation-works)
  - [Mathematical Definition](#mathematical-definition)
  - [Fold Size Calculation](#fold-size-calculation)
  - [Shuffle](#shuffle)
  - [Characteristics / Key Points](#characteristics--key-points)
  - [Why is it used? / Applications](#why-is-it-used--applications)
- [Solutions](#solutions)
  - [Custom Implementation](#custom-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

---

## Problem Statement

### [Implement K-Fold Cross-Validation](https://www.deep-ml.com/problems/18)

Implement a function that generates train-test index splits for **K-Fold Cross-Validation**.

The function takes the total number of samples, the number of folds, and a shuffle option. It should divide the dataset indices into $k$ roughly equal folds and return a list containing one train-test split for every fold.

For each iteration:

- One fold is used as the test set.
- The remaining $k-1$ folds are combined to form the training set.
- Every sample must appear in a test set exactly once.
- If the samples cannot be divided equally, the extra samples are assigned to the first folds.
- When `shuffle=True`, indices are shuffled before splitting.
- When `shuffle=False`, indices remain in their original order.

The function returns indices rather than the actual data.

---

## Example

### Input

```python
k_fold_cross_validation(
    n_samples=10,
    k=5,
    shuffle=False
)
```

### Output

```text
[
    ([2, 3, 4, 5, 6, 7, 8, 9], [0, 1]),
    ([0, 1, 4, 5, 6, 7, 8, 9], [2, 3]),
    ([0, 1, 2, 3, 6, 7, 8, 9], [4, 5]),
    ([0, 1, 2, 3, 4, 5, 8, 9], [6, 7]),
    ([0, 1, 2, 3, 4, 5, 6, 7], [8, 9])
]
```

### Reasoning

There are $10$ samples and $5$ folds:

$$\frac{10}{5} = 2$$

Therefore, every fold contains two samples.

The folds are:

```text
Fold 0 -> [0, 1]
Fold 1 -> [2, 3]
Fold 2 -> [4, 5]
Fold 3 -> [6, 7]
Fold 4 -> [8, 9]
```

During the first iteration:

```text
Test  -> [0, 1]
Train -> [2, 3, 4, 5, 6, 7, 8, 9]
```

During the second iteration:

```text
Test  -> [2, 3]
Train -> [0, 1, 4, 5, 6, 7, 8, 9]
```

This continues until every fold has been used as the test set exactly once.

---

## Learn: K-Fold Cross-Validation

### What is it?

**K-Fold Cross-Validation** is a resampling technique used to evaluate the performance of a machine learning model.

Instead of dividing the dataset into only one training and testing set, the dataset is divided into $k$ smaller subsets called **folds**.

The model is trained and evaluated $k$ times.

During each iteration:

- One fold is reserved for testing.
- All remaining folds are used for training.
- The model is evaluated on the held-out fold.

After all iterations, every sample has been used for testing exactly once and for training approximately $k-1$ times.

The final model performance can then be summarized using the scores obtained across the folds.

---

### How K-Fold Cross-Validation Works

Suppose there are $12$ samples and $k=4$.

The indices can be divided into:

```text
Fold 0 -> [0, 1, 2]
Fold 1 -> [3, 4, 5]
Fold 2 -> [6, 7, 8]
Fold 3 -> [9, 10, 11]
```

The cross-validation process becomes:

| Iteration | Training Folds | Test Fold |
| --------- | -------------- | --------- |
| 1         | 1, 2, 3        | 0         |
| 2         | 0, 2, 3        | 1         |
| 3         | 0, 1, 3        | 2         |
| 4         | 0, 1, 2        | 3         |

Each fold is used exactly once as the test set.

The process therefore provides multiple estimates of model performance rather than relying on a single train-test split.

---

### Mathematical Definition

Let the dataset contain $n$ samples:

$$D = {0,1,2,\ldots,n-1}$$

After optionally shuffling the indices, divide $D$ into $k$ folds:

$$D = F_1 \cup F_2 \cup \cdots \cup F_k$$

where the folds are mutually disjoint:

$$F_i \cap F_j = \emptyset \quad \text{for } i \neq j$$

and together they contain every sample:

$$\bigcup_{i=1}^{k} F_i = D$$

For fold $i$, the test set is:

$$Test_i = F_i$$

The training set is the union of every other fold:

$$Train_i = \bigcup_{j \neq i} F_j$$

Therefore:

$$Train_i \cap Test_i = \emptyset$$

and

$$|Train_i| + |Test_i| = n$$

---

### Fold Size Calculation

Ideally, every fold would contain exactly:

$$\frac{n}{k}$$

samples.

However, when $n$ is not divisible by $k$, the folds cannot all have the same size.

For example, with:

$$n=10,\quad k=3$$

we have:

$$10 = 3(3) + 1$$

Therefore, the fold sizes are:

```text
[4, 3, 3]
```

The extra sample is assigned to the first fold.

More generally, let:

$$q = \left\lfloor\frac{n}{k}\right\rfloor$$

and

$$r = n \bmod k$$

Then:

- The first $r$ folds contain $q+1$ samples.
- The remaining $k-r$ folds contain $q$ samples.

For example:

$$n=10,\quad k=3$$

gives:

$$q=3$$

and

$$r=1$$

Therefore:

```text
Fold 0 -> 4 samples
Fold 1 -> 3 samples
Fold 2 -> 3 samples
```

This keeps the fold sizes as balanced as possible.

---

### Shuffle

The function supports two modes.

With:

```python
shuffle=False
```

the original indices are preserved:

```text
[0, 1, 2, 3, 4, 5, ...]
```

With:

```python
shuffle=True
```

the indices are randomly rearranged before creating the folds.

For example:

```text
Original:
[0, 1, 2, 3, 4, 5]

Shuffled:
[3, 0, 5, 1, 4, 2]
```

The folds are then created from the shuffled ordering.

The problem specifies `np.random.shuffle()` for this operation.

The random seed is expected to be set externally when deterministic results are required.

---

### Why Shuffle Matters

If the original dataset is ordered according to some meaningful property, splitting without shuffling can produce biased folds.

For example, suppose a dataset is sorted by class:

```text
Class 0:
[0, 1, 2, 3]

Class 1:
[4, 5, 6, 7]
```

Without shuffling, individual folds may contain mostly one class.

Shuffling can distribute the samples more randomly across folds.

However, ordinary K-Fold shuffling does not guarantee class balance. For classification problems where class proportions must be preserved, **Stratified K-Fold** is often more appropriate.

---

### Training and Testing Indices

This problem returns indices rather than actual samples.

Suppose:

```text
X = [x0, x1, x2, x3]
y = [y0, y1, y2, y3]
```

and the test indices are:

```text
[1, 3]
```

Then the corresponding data can be selected using:

```python
X_test = X[[1, 3]]
y_test = y[[1, 3]]
```

while the training indices determine the training data.

Returning indices makes the cross-validation function independent of the actual dataset representation.

---

### Why Return Indices?

Returning indices has several advantages.

#### Memory Efficiency

The function does not create multiple copies of the complete dataset.

Instead, it stores only integer indices.

#### Flexibility

The same splitting function can be used with:

- NumPy arrays
- Pandas DataFrames
- Target arrays
- Multiple input arrays
- Custom datasets

#### Separation of Concerns

The splitting algorithm only determines **which samples belong to each set**.

The caller decides how those indices should be applied to the actual data.

---

### Cross-Validation Performance

Suppose a model produces validation scores:

```text
Fold 1 -> 0.91
Fold 2 -> 0.88
Fold 3 -> 0.93
Fold 4 -> 0.90
Fold 5 -> 0.89
```

A common summary is the mean score:

$$CVScore = \frac{1}{k}\sum_{i=1}^{k}Score_i$$

The variance or standard deviation of the fold scores can also be examined to understand how stable the model is across different subsets of the data.

A model with similar scores across folds generally has more consistent performance.

---

### Characteristics / Key Points

- K-Fold divides a dataset into $k$ folds.
- Each fold becomes the test set exactly once.
- The remaining folds form the training set.
- Every sample is used for testing exactly once.
- Every sample is used for training approximately $k-1$ times.
- Fold sizes are as equal as possible.
- Extra samples are assigned to the first folds.
- Shuffling occurs before splitting when requested.
- The random seed should be controlled externally for reproducibility.
- Training and testing sets do not overlap within an iteration.
- The method produces $k$ train-test splits.
- The function returns indices rather than actual data.
- Larger $k$ generally means more training iterations.
- Smaller $k$ generally requires less computation.
- Common choices include $k=5$ and $k=10$.
- K-Fold is mainly useful when the dataset is not large enough for a reliable single holdout split.
- Standard K-Fold does not explicitly preserve class proportions.
- Stratified K-Fold is preferred when class distribution must be preserved.

---

### K-Fold vs Train-Test Split

A traditional train-test split creates one fixed partition:

```text
Dataset
   |
   +-- Training
   |
   +-- Testing
```

K-Fold creates multiple partitions:

```text
Dataset
   |
   +-- Fold 1
   +-- Fold 2
   +-- Fold 3
   +-- Fold 4
   +-- Fold 5
```

Each fold gets an opportunity to serve as the test set.

Therefore, K-Fold generally provides a more robust estimate than relying on one arbitrary split.

The trade-off is increased computational cost because the model must be trained multiple times.

---

### Why is it used? / Applications

K-Fold Cross-Validation is commonly used for:

- Model evaluation.
- Comparing different machine learning algorithms.
- Hyperparameter tuning.
- Feature selection.
- Estimating generalization performance.
- Detecting whether model performance depends heavily on a particular data split.
- Selecting between competing models.

For example, during hyperparameter tuning, several parameter configurations can be evaluated using the same folds.

The configuration with the best average validation score can then be selected.

---

> 💡 **Important Note**
>
> Cross-validation should be performed carefully when preprocessing the data. Operations such as scaling, feature selection, or imputation should generally be fitted using only the training portion of each fold. Fitting them on the complete dataset before cross-validation can cause **data leakage** because information from the test fold influences the training process.

---

> 💡 **Interview Tip**
>
> Remember the core invariant of K-Fold Cross-Validation: **every sample appears in exactly one test fold and in the training set for all other folds**.

---

## Solutions

### Custom Implementation

```python
import numpy as np
from typing import List, Tuple

def k_fold_cross_validation(
    n_samples: int,
    k: int = 5,
    shuffle: bool = True
) -> List[Tuple[List[int], List[int]]]:
    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    folds = np.array_split(indices, k)

    return [
        (
            [int(x) for j, fold in enumerate(folds) if j != i for x in fold],
            [int(x) for x in folds[i]]
        )
        for i in range(k)
    ]
```

---

## Code Explanation

### Step 1: Generate Dataset Indices

```python
indices = np.arange(n_samples)
```

For $n$ samples, this creates:

```text
[0, 1, 2, ..., n_samples - 1]
```

These indices represent the positions of samples in the dataset.

The function does not need the actual feature matrix or target values.

---

### Step 2: Shuffle When Requested

```python
if shuffle:
    np.random.shuffle(indices)
```

If `shuffle=True`, the indices are randomly reordered.

If `shuffle=False`, they remain in their original order.

This is performed before creating the folds so that the shuffled order influences the fold assignments.

---

### Step 3: Create the Folds

```python
folds = np.array_split(indices, k)
```

`np.array_split()` divides the indices into $k$ approximately equal parts.

Importantly, when the division is not exact, it distributes the extra samples among the first folds.

For example:

```text
n_samples = 10
k = 3
```

produces fold sizes:

```text
[4, 3, 3]
```

This directly satisfies the requirement of keeping fold sizes as balanced as possible.

---

### Step 4: Select the Test Fold

For each iteration:

```python
folds[i]
```

is selected as the test fold.

For example:

```text
i = 0
test = folds[0]
```

Then:

```text
i = 1
test = folds[1]
```

and so on.

This guarantees that every fold becomes the test set exactly once.

---

### Step 5: Construct the Training Fold

The list comprehension:

```python
[int(x) for j, fold in enumerate(folds) if j != i for x in fold]
```

iterates through every fold.

The condition:

```python
j != i
```

excludes the current test fold.

All remaining indices are flattened into one training list.

Thus:

$$Train_i = \bigcup_{j \neq i} F_j$$

---

### Step 6: Convert NumPy Integers to Python Integers

The values returned by NumPy may be NumPy integer types.

The implementation uses:

```python
int(x)
```

to convert every index into a standard Python `int`.

Therefore, the returned structure contains ordinary Python lists of integers as required.

---

### Step 7: Return All Splits

The outer list comprehension creates one tuple for every fold:

```text
[
    (train_0, test_0),
    (train_1, test_1),
    ...
    (train_k-1, test_k-1)
]
```

Therefore, the function returns exactly $k$ train-test pairs.

---

### Algorithm

The complete algorithm is:

1. Create indices from $0$ to $n-1$.
2. Shuffle the indices if requested.
3. Divide the indices into $k$ balanced folds.
4. For every fold:
   - Use the current fold as the test set.
   - Combine all other folds into the training set.

5. Convert indices to Python integers.
6. Return all train-test pairs.

---

### Pseudocode

```text
K_FOLD_CROSS_VALIDATION(n, k, shuffle)

    indices = [0, 1, ..., n - 1]

    if shuffle
        randomly shuffle indices

    folds = split indices into k balanced folds

    result = empty list

    for i = 0 to k - 1

        test = folds[i]
        train = empty list

        for j = 0 to k - 1
            if j != i
                append all elements of folds[j] to train

        append (train, test) to result

    return result
```

---

### Example with Unequal Fold Sizes

Consider:

```text
n_samples = 10
k = 3
shuffle = False
```

The indices are:

```text
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

The folds become:

```text
Fold 0 -> [0, 1, 2, 3]
Fold 1 -> [4, 5, 6]
Fold 2 -> [7, 8, 9]
```

The first split is:

```text
Train -> [4, 5, 6, 7, 8, 9]
Test  -> [0, 1, 2, 3]
```

The second split is:

```text
Train -> [0, 1, 2, 3, 7, 8, 9]
Test  -> [4, 5, 6]
```

The third split is:

```text
Train -> [0, 1, 2, 3, 4, 5, 6]
Test  -> [7, 8, 9]
```

Notice that the test fold sizes are not identical, but they differ by at most one sample.

---

### Important Invariant

For every fold $i$:

$$Train_i \cap Test_i = \emptyset$$

Every sample appears in exactly one test fold:

$$\bigcup_{i=1}^{k} Test_i = D$$

and the test folds do not overlap:

$$Test_i \cap Test_j = \emptyset \quad \text{for } i \neq j$$

Therefore, across all $k$ iterations, every sample is evaluated exactly once.

---

## Time & Space Complexity

Let:

- $n$ be the number of samples.
- $k$ be the number of folds.

Creating the index array takes:

$$O(n)$$

Shuffling the indices takes:

$$O(n)$$

Creating the folds also takes approximately:

$$O(n)$$

However, constructing the training list for every fold requires copying almost all $n$ indices $k$ times.

Therefore, the total time complexity is:

$$O(nk)$$

For a fixed common value of $k$, this behaves approximately as:

$$O(n)$$

---

### Space Complexity

The original index array requires:

$$O(n)$$

The $k$ folds together contain all $n$ indices:

$$O(n)$$

The returned train-test splits contain approximately $n$ training indices and $n/k$ test indices for every fold.

Across all folds, the returned structure contains approximately:

$$O(nk)$$

indices.

Therefore:

| Complexity    | Value                               |
| ------------- | ----------------------------------- |
| Time          | **O(nk)**                           |
| Space         | **O(nk)** including returned splits |
| Index Storage | **O(n)** before constructing splits |

where:

- $n$ is the number of samples.
- $k$ is the number of folds.

The dominant cost comes from constructing and storing the training indices separately for each fold.
