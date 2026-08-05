# Calculate Cosine Similarity Between Vectors (Easy, Linear Algebra)

## Table of Contents

- Problem Statement
- Example
- Learn: Cosine Similarity
- Solution
- Code Explanation
- Time & Space Complexity

---

## Problem Statement

[Calculate Cosine Similarity Between Vectors](https://www.deep-ml.com/problems/76)

Write a Python function `cosine_similarity(v1, v2)` that computes the **cosine similarity** between two vectors.

The function should:

- Calculate the cosine similarity between `v1` and `v2`.
- Return the cosine similarity as a float.
- Handle invalid inputs such as:
  - Vectors with mismatched shapes.
  - Empty vectors.
  - Vectors with zero magnitude.

Cosine similarity measures the cosine of the angle between two vectors, making it a useful measure of **directional similarity** regardless of their magnitudes.

---

## Example

```python
import numpy as np

v1 = np.array([1, 2, 3])
v2 = np.array([2, 4, 6])

print(round(cosine_similarity(v1, v2), 3))
```

### Output

```text
1.0
```

### Explanation

The vectors

```text
v1 = [1, 2, 3]
v2 = [2, 4, 6]
```

point in the same direction.

Their cosine similarity is

$$
\cos(\theta) = 1
$$

indicating perfect directional similarity.

---

# Learn: Cosine Similarity

## What is it?

**Cosine Similarity** measures the similarity between two vectors by computing the cosine of the angle between them.

Unlike Euclidean distance, cosine similarity ignores the magnitudes of the vectors and focuses only on their direction.

A cosine similarity of:

- **1** indicates identical directions.
- **0** indicates orthogonal (perpendicular) vectors.
- **-1** indicates opposite directions.

---

## Mathematical Definition / Formula

Suppose we have two vectors

$$
\mathbf{A} = [A_1, A_2, \ldots, A_n]
$$

and

$$
\mathbf{B} = [B_1, B_2, \ldots, B_n].
$$

The cosine similarity is

$$
\cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\|\|\mathbf{B}\|}
$$

where

$$
\mathbf{A} \cdot \mathbf{B} = \sum_{i=1}^{n} A_iB_i
$$

is the dot product, and

$$
\|\mathbf{A}\| = \sqrt{\sum_{i=1}^{n} A_i^2}
$$

$$
\|\mathbf{B}\| = \sqrt{\sum_{i=1}^{n} B_i^2}
$$

are the Euclidean (L2) norms of the vectors.

---

## Characteristics / Key Points

- Measures **directional similarity**.
- Independent of vector magnitude.
- Symmetric metric.
- Commonly used for high-dimensional data.
- Output ranges from **-1** to **1**.

### Interpretation

| Cosine Similarity | Meaning             |
| ----------------: | ------------------- |
|             **1** | Same direction      |
|             **0** | Orthogonal vectors  |
|            **-1** | Opposite directions |

---

## Why is it used? / Applications

Cosine similarity is widely used in:

- Text Similarity
- Information Retrieval
- Recommendation Systems
- Document Clustering
- Natural Language Processing (NLP)
- Image Retrieval
- Semantic Search

It is particularly useful when the magnitude of the vectors is less important than their orientation.

> 💡 **Important Note**
>
> Cosine similarity measures only the angle between vectors. Two vectors with very different magnitudes can still have a cosine similarity of **1** if they point in the same direction.

---

# Solution

## Custom Implementation

```python
import numpy as np

def cosine_similarity(v1, v2):
    if v1.shape != v2.shape:
        raise ValueError(
            f"Shape mismatch: v1 shape {v1.shape} != v2 shape {v2.shape}"
        )

    if v1.size == 0 or v2.size == 0:
        raise ValueError("Input vectors cannot be empty.")

    v1_mag = np.linalg.norm(v1)
    v2_mag = np.linalg.norm(v2)

    if v1_mag == 0 or v2_mag == 0:
        raise ValueError("Input vectors cannot have zero magnitude.")

    return np.dot(v1, v2) / (v1_mag * v2_mag)
```

---

# Code Explanation

### Step 1: Validate the Inputs

```python
if v1.shape != v2.shape:
```

Ensure both vectors have the same dimensions.

---

```python
if v1.size == 0 or v2.size == 0:
```

Prevent computations on empty vectors.

---

### Step 2: Compute the Vector Magnitudes

```python
v1_mag = np.linalg.norm(v1)
v2_mag = np.linalg.norm(v2)
```

These compute

$$
\|\mathbf{A}\|
$$

and

$$
\|\mathbf{B}\|
$$

respectively.

---

### Step 3: Check for Zero Magnitude

```python
if v1_mag == 0 or v2_mag == 0:
```

Cosine similarity is undefined for zero vectors because the denominator becomes zero.

---

### Step 4: Compute the Dot Product

```python
np.dot(v1, v2)
```

This computes

$$
\mathbf{A} \cdot \mathbf{B}
$$

---

### Step 5: Compute the Cosine Similarity

```python
np.dot(v1, v2) / (v1_mag * v2_mag)
```

This applies

$$
\cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\|\|\mathbf{B}\|}
$$

to obtain the similarity between the two vectors.

---

## Time & Space Complexity

Let

- $n$ = Number of elements in each vector.

| Complexity | Value    |
| ---------- | -------- |
| Time       | **O(n)** |
| Space      | **O(1)** |

The algorithm performs a constant number of passes over the input vectors to compute the dot product and vector magnitudes. Aside from a few scalar variables, it uses no additional memory proportional to the input size, resulting in **O(1)** auxiliary space.
