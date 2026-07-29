# Linear Kernel Function (Easy, Machine Learning)

## Table of Contents

- Problem Statement
- Example
- Learn: Linear Kernel Function
- Solution
- Code Explanation
- Time & Space Complexity

---

## Problem Statement

[Linear Kernel Function](https://www.deep-ml.com/problems/45)

Write a Python function `kernel_function` that computes the linear kernel between two input vectors `x1` and `x2`.

The linear kernel is defined as the dot product (inner product) of two vectors.

---

## Example

```python
import numpy as np

x1 = np.array([1, 2, 3])
x2 = np.array([4, 5, 6])

result = kernel_function(x1, x2)

print(result)
```

### Output

```text
32
```

### Explanation

The linear kernel is calculated using the dot product:

\[
K(x_1,x_2)=x_1 \cdot x_2
\]

\[
=(1 \times 4)+(2 \times 5)+(3 \times 6)
\]

\[
=4+10+18
\]

\[
=32
\]

---

# Learn: Linear Kernel Function

## What is it?

A kernel function is a mathematical function used in machine learning to measure the similarity between two data points.

The **linear kernel** is the simplest kernel function and computes similarity using the dot product between two feature vectors.

Instead of explicitly transforming data into a higher-dimensional feature space, the kernel computes the relationship directly using the original vectors.

Linear kernels are commonly used in Support Vector Machines (SVM) when the data is approximately linearly separable.

---

## Mathematical Definition / Formula

Given two vectors:

\[
x_1=[x_{1,1},x_{1,2},...,x_{1,n}]
\]

\[
x_2=[x_{2,1},x_{2,2},...,x_{2,n}]
\]

The linear kernel is defined as:

\[
K(x_1,x_2)=x_1 \cdot x_2
\]

The dot product can be expanded as:

\[
K(x_1,x_2)=\sum_{i=1}^{n}x_{1,i}x_{2,i}
\]

where:

- \(n\) is the number of features.
- \(x_{1,i}\) represents the ith feature of vector \(x_1\).
- \(x_{2,i}\) represents the ith feature of vector \(x_2\).

---

## Characteristics / Key Points

- Simple implementation using vector multiplication and summation.
- Computationally efficient compared to complex kernels.
- Works well for high-dimensional feature spaces.
- Provides an interpretable similarity measure.
- Does not require explicit feature transformation.

### Advantages

- Low computational cost.
- Easy to understand.
- Effective for linearly separable datasets.

### Limitations

- Cannot capture complex non-linear relationships.
- Performance depends on the original feature representation.

---

## Why is it used? / Applications

Linear kernels are used in:

- Support Vector Machines (SVM).
- Text classification using TF-IDF features.
- High-dimensional sparse datasets.
- Similarity calculations between embeddings.
- Linear classification problems.

> 💡 **Important Note**
>
> In many NLP tasks, linear kernels perform surprisingly well because text representations such as TF-IDF already create very high-dimensional feature spaces.

---

# Solution

## Custom Implementation

```python
import numpy as np

def kernel_function(x1, x2):
    return np.sum(x1 * x2)
```

---

# Code Explanation

1. Perform element-wise multiplication between the two vectors.

Example:

\[
[1,2,3] \times [4,5,6]
\]

produces:

\[
[4,10,18]
\]

2. Add all multiplied values:

\[
4+10+18=32
\]

3. The final sum represents the dot product between the two vectors.

The function:

```python
np.sum(x1 * x2)
```

is equivalent to:

```python
np.dot(x1, x2)
```

---

## Time & Space Complexity

Let \(n\) be the number of features.

| Complexity | Value |
| ---------- | ----- |
| Time | **O(n)** |
| Space | **O(1)** |

The algorithm only stores the final accumulated value and does not require additional memory.