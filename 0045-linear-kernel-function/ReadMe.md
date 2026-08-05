# Linear Kernel Function (Easy, Machine Learning)

## Table of Contents

- Problem Statement
- Example
- Learn: Understanding the Linear Kernel
- Solution
- Code Explanation
- Time & Space Complexity

---

## Problem Statement

### [Linear Kernel Function](https://www.deep-ml.com/problems/45)

Write a Python function that computes the **Linear Kernel** between two vectors.

The function should:

- Accept two input vectors `x1` and `x2`.
- Compute their dot (inner) product.
- Return the resulting scalar value.

The Linear Kernel is one of the simplest kernel functions and is widely used in **Support Vector Machines (SVMs)** and other linear machine learning models.

---

## Example

### Input

```python
import numpy as np

x1 = np.array([1, 2, 3])
x2 = np.array([4, 5, 6])

result = kernel_function(x1, x2)

print(result)
```

### Output

```python
32
```

### Reasoning

The Linear Kernel is simply the dot product of the two vectors.

$$
(1\times4)+(2\times5)+(3\times6)=4+10+18=32
$$

---

## Learn: Understanding the Linear Kernel

### What is it?

A **Kernel Function** measures the similarity between two data points.

Instead of explicitly transforming data into a higher-dimensional feature space, kernel functions allow machine learning algorithms to compute similarities directly in the original space. This idea is known as the **Kernel Trick**.

The **Linear Kernel** is the simplest kernel function. It does not perform any nonlinear transformation—it simply computes the dot product between two vectors.

When the data is approximately linearly separable, the Linear Kernel is often the fastest and most effective choice.

---

### Mathematical Definition

Given two vectors

$$
x_1=
[x_{11},x_{12},...,x_{1n}]
$$

and

$$
x_2=
[x_{21},x_{22},...,x_{2n}]
$$

the Linear Kernel is

$$
K(x_1,x_2)=x_1^Tx_2
$$

or equivalently,

$$
K(x_1,x_2)=
\sum_{i=1}^{n}
x_{1i}x_{2i}
$$

where

- $n$ is the number of features.
- $x_{1i}$ is the $i^{th}$ component of the first vector.
- $x_{2i}$ is the $i^{th}$ component of the second vector.

---

### Dot Product Example

Suppose

$$
x_1=
\begin{bmatrix}
1\\
2\\
3
\end{bmatrix}
$$

and

$$
x_2=
\begin{bmatrix}
4\\
5\\
6
\end{bmatrix}
$$

The Linear Kernel is

$$
\begin{bmatrix}
1 & 2 & 3
\end{bmatrix}
\begin{bmatrix}
4\\
5\\
6
\end{bmatrix}
=32
$$

or,

$$
(1\times4)+(2\times5)+(3\times6)=32
$$

A larger value indicates greater similarity when the vectors point in similar directions.

---

### Geometric Interpretation

The dot product can also be written as

$$
x_1^Tx_2=
||x_1||
||x_2||
\cos\theta
$$

where

- $||x_1||$ is the magnitude of the first vector.
- $||x_2||$ is the magnitude of the second vector.
- $\theta$ is the angle between them.

This means the Linear Kernel measures both

- Magnitude
- Direction

Vectors pointing in similar directions produce larger kernel values.

---

### Linear Kernel vs Other Kernels

| Kernel         | Formula                    | Suitable For                   |
| -------------- | -------------------------- | ------------------------------ | ------- | --- | ---- | ------------------------------- |
| Linear         | $x_1^Tx_2$                 | Linearly separable data        |
| Polynomial     | $(x_1^Tx_2+c)^d$           | Polynomial decision boundaries |
| RBF (Gaussian) | $e^{-\gamma                |                                | x_1-x_2 |     | ^2}$ | Complex nonlinear relationships |
| Sigmoid        | $\tanh(\alpha x_1^Tx_2+c)$ | Neural-network-inspired models |

The Linear Kernel is computationally the simplest among all common kernels.

---

### Characteristics / Key Points

- Computes the dot product between two vectors.
- Measures similarity in the original feature space.
- Does not create nonlinear decision boundaries.
- Very computationally efficient.
- Requires no additional hyperparameters.
- Works well for high-dimensional sparse data.
- Frequently used in linear SVMs.

---

### Why is it used? / Applications

The Linear Kernel is widely used in machine learning.

Applications include

- Support Vector Machines (SVMs)
- Linear Classification
- Linear Regression
- Text Classification
- Spam Detection
- Sentiment Analysis
- Information Retrieval
- High-dimensional Sparse Data

Because text data often contains thousands of features, the Linear Kernel is commonly preferred over more expensive nonlinear kernels.

---

> 💡 **Important Note**
>
> The Linear Kernel is mathematically identical to the **dot product**, so using a Linear Kernel SVM is equivalent to training a linear classifier in the original feature space. Unlike Polynomial or RBF kernels, it does **not** use the Kernel Trick to create nonlinear decision boundaries.

---

## Solution

### Custom Implementation

```python
import numpy as np

def kernel_function(x1, x2):
    return np.sum(x1 * x2)
```

### NumPy Equivalent

```python
import numpy as np

def kernel_function(x1, x2):
    return np.dot(x1, x2)
```

or

```python
np.inner(x1, x2)
```

---

## Code Explanation

### Step 1

Receive the two input vectors.

```python
x1
x2
```

Both vectors must have the same number of features.

---

### Step 2

Multiply corresponding elements.

```python
x1 * x2
```

This performs element-wise multiplication.

For example,

```python
[1, 2, 3] * [4, 5, 6]
```

produces

```python
[4, 10, 18]
```

---

### Step 3

Sum all products.

```python
np.sum(...)
```

Adding the element-wise products computes the dot product, which is the Linear Kernel value.

---

### Step 4

Return the similarity score.

A larger value generally indicates that the two vectors point in a similar direction.

---

## Time & Space Complexity

| Complexity | Value    |
| ---------- | -------- |
| Time       | **O(n)** |
| Space      | **O(1)** |

where

- $n$ is the number of features in each input vector.

The algorithm performs one multiplication and one addition for every feature while using only constant extra memory.
