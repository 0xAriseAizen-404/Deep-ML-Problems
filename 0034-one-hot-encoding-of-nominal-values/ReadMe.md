# One-Hot Encoding of Nominal Values (Easy, Machine Learning)

## Table of Contents

- Problem Statement
- Example
- Learn: Understanding One-Hot Encoding
- Solution
- Code Explanation
- Time & Space Complexity

---

## Problem Statement

### [One-Hot Encoding of Nominal Values](https://www.deep-ml.com/problems/34)

Write a Python function that performs **One-Hot Encoding** on a 1-D NumPy array containing integer categorical values.

The function should:

- Accept a 1-D NumPy array `x`.
- Optionally accept `n_col`, representing the total number of output columns.
- Automatically determine the number of categories if `n_col` is not provided.
- Return a 2-D NumPy array where each category is represented by a binary vector.

---

## Example

### Input

```python
x = np.array([0, 1, 2, 1, 0])

output = to_categorical(x)
print(output)
```

### Output

```python
[
    [1., 0., 0.],
    [0., 1., 0.],
    [0., 0., 1.],
    [0., 1., 0.],
    [1., 0., 0.]
]
```

### Reasoning

The input contains three categories:

```text
0, 1, 2
```

Each category is converted into a binary vector whose length equals the total number of categories.

Category `0`

```text
[1, 0, 0]
```

Category `1`

```text
[0, 1, 0]
```

Category `2`

```text
[0, 0, 1]
```

Every sample is replaced by its corresponding one-hot encoded vector.

---

## Learn: Understanding One-Hot Encoding

### What is it?

One-Hot Encoding is a feature encoding technique used to convert **categorical (nominal)** variables into numerical vectors that machine learning algorithms can understand.

Unlike numerical features, categorical values do not have a natural mathematical relationship. For example, assigning

```text
Red = 0
Green = 1
Blue = 2
```

incorrectly suggests that

$$
\text{Blue} > \text{Green} > \text{Red}
$$

which has no real meaning.

One-Hot Encoding eliminates this problem by representing every category as a binary vector where only one element is equal to **1**, while all remaining elements are **0**.

This allows machine learning models to treat every category independently without assuming any ordering.

---

### Mathematical Definition

Suppose there are

$$
k
$$

unique categories

$$
\{0,1,\ldots,k-1\}
$$

For a category

$$
x=j
$$

its one-hot encoded vector is

$$
v=[v_0,v_1,\ldots,v_{k-1}]
$$

where

$$
v_i=
\begin{cases}
1,& i=j\\
0,& \text{otherwise}
\end{cases}
$$

Each encoded vector has exactly one element equal to **1**.

---

### How One-Hot Encoding Works

The algorithm performs the following steps:

1. Determine the number of categories.
2. Create a vector of zeros for every sample.
3. Set the position corresponding to the category value to `1`.
4. Repeat for every sample.

For example,

Input

```text
[0, 1, 2, 1]
```

Number of categories

$$
3
$$

Encoded output

```text
0 → [1 0 0]

1 → [0 1 0]

2 → [0 0 1]

1 → [0 1 0]
```

The resulting matrix is

$$
\begin{bmatrix}
1 & 0 & 0\\
0 & 1 & 0\\
0 & 0 & 1\\
0 & 1 & 0
\end{bmatrix}
$$

---

### Why Not Use Integer Labels?

Suppose colors are encoded as

```text
Red = 0

Green = 1

Blue = 2
```

A machine learning model may incorrectly interpret

$$
2 > 1 > 0
$$

or

$$
2 = 2 \times 1
$$

which introduces a false ordinal relationship.

After One-Hot Encoding,

```text
Red   → [1 0 0]

Green → [0 1 0]

Blue  → [0 0 1]
```

all categories become equally distant from each other.

---

### Characteristics / Key Points

- Converts categorical values into numerical vectors.
- Preserves categorical independence.
- Prevents false ordinal relationships.
- Every encoded vector contains exactly one `1`.
- Remaining elements are `0`.
- Output dimension equals the number of categories.
- Common preprocessing step for machine learning models.

---

### Why is it used? / Applications

One-Hot Encoding is one of the most common preprocessing techniques in machine learning.

Applications include

- Classification problems
- Logistic Regression
- Neural Networks
- Decision Trees
- Random Forests
- Support Vector Machines
- Recommendation systems
- Feature engineering
- Deep learning pipelines

It is especially useful whenever categorical variables have **no natural ordering**.

---

> 💡 **Important Note**
>
> One-Hot Encoding works well for features with a small number of categories. However, for high-cardinality features (such as thousands of unique words or user IDs), it creates very large sparse matrices. In such cases, techniques like **Label Encoding**, **Target Encoding**, **Hash Encoding**, or **Embeddings** are usually more efficient.

---

## Solution

### Custom Implementation

```python
import numpy as np

def to_categorical(x, n_col=None):
    if n_col is None:
        n_col = np.max(x) + 1

    one_hot = []

    for val in x:
        row = [0] * n_col
        row[val] = 1
        one_hot.append(row)

    return np.array(one_hot)
```

### NumPy Implementation

```python
import numpy as np

def to_categorical(x, n_col=None):
    if n_col is None:
        n_col = np.max(x) + 1

    return np.eye(n_col)[x]
```

---

## Code Explanation

### Step 1

Determine the total number of categories.

```python
if n_col is None:
    n_col = np.max(x) + 1
```

If the number of columns is not specified, it is computed using the maximum category value.

---

### Step 2

Create an empty list.

```python
one_hot = []
```

Each encoded vector will be stored here.

---

### Step 3

Create a zero vector for every category.

```python
row = [0] * n_col
```

Initially, every position is zero.

---

### Step 4

Activate the correct category.

```python
row[val] = 1
```

The index corresponding to the category value is set to `1`.

---

### Step 5

Store the encoded vector.

```python
one_hot.append(row)
```

This process is repeated for every element in the input array.

---

### Step 6

Convert the result into a NumPy array.

```python
np.array(one_hot)
```

The final output is a 2-D NumPy array where each row represents one encoded sample.

---

## Time & Space Complexity

| Complexity | Value        |
| ---------- | ------------ |
| Time       | **O(n × k)** |
| Space      | **O(n × k)** |

where

- $n$ is the number of samples.
- $k$ is the number of unique categories (or `n_col`).

Each sample requires creating a vector of length $k$, resulting in a total complexity proportional to the size of the output matrix.
