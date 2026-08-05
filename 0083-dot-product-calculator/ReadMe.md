# Dot Product Calculator (Easy, Linear Algebra)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Dot Product of Vectors](#learn-dot-product-of-vectors)
- [Solutions](#solutions)
  - [Custom Implementation](#custom-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

## Problem Statement

### [Dot Product Calculator](https://www.deep-ml.com/problems/83)

Implement a Python function `calculate_dot_product(vec1, vec2)` that calculates the dot product of two vectors.

The input consists of two 1D NumPy arrays of equal length. The function should multiply corresponding elements of both vectors and return their sum as a single scalar value.

The dot product is defined as:

$$ a \cdot b = \sum\_{i=1}^{n} a_i b_i $$

where:

- $a_i$ represents the elements of the first vector.
- $b_i$ represents the elements of the second vector.
- $n$ represents the vector length.

## Example

### Input

```python
import numpy as np

vec1 = np.array([1, 2, 3])
vec2 = np.array([4, 5, 6])

result = calculate_dot_product(vec1, vec2)
print(result)
```

### Output

```text
32
```

### Reasoning

The dot product multiplies corresponding elements and sums the results:

$$ a \cdot b = (1 \times 4) + (2 \times 5) + (3 \times 6) $$

$$ a \cdot b = 4 + 10 + 18 = 32 $$

## Learn: Dot Product of Vectors

### What is it?

The dot product, also called the scalar product, is a fundamental operation in linear algebra that combines two vectors and produces a single numerical value.

For two vectors of the same dimension, the dot product is calculated by multiplying corresponding elements and adding all the products.

Given two vectors:

$$ a = [a_1, a_2, ..., a_n] $$

$$ b = [b_1, b_2, ..., b_n] $$

The dot product produces:

$$ a \cdot b = a_1b_1 + a_2b_2 + ... + a_nb_n $$

The output is a scalar value, not another vector.

### Mathematical Definition / Formula

The dot product of two vectors is:

$$ a \cdot b = \sum\_{i=1}^{n} a_i b_i $$

Example:

$$ a = [1,2,3] $$

$$ b = [4,5,6] $$

Then:

$$ a \cdot b = (1 \times 4) + (2 \times 5) + (3 \times 6) $$

$$ a \cdot b = 32 $$

### Geometric Interpretation

The dot product can also describe the relationship between two vectors using the angle between them.

$$ a \cdot b = |a||b|\cos(\theta) $$

where:

- $|a|$ represents the magnitude of vector $a$.
- $|b|$ represents the magnitude of vector $b$.
- $\theta$ represents the angle between the vectors.

This interpretation helps determine vector similarity.

If:

$$ a \cdot b > 0 $$

The vectors point in a similar direction.

If:

$$ a \cdot b < 0 $$

The vectors point in opposite directions.

If:

$$ a \cdot b = 0 $$

The vectors are perpendicular.

### Properties of Dot Product

#### Commutative Property

The order of vectors does not affect the result.

$$ a \cdot b = b \cdot a $$

#### Distributive Property

The dot product distributes over vector addition.

$$ a \cdot (b+c) = a \cdot b + a \cdot c $$

#### Scalar Multiplication

A scalar can be moved outside the dot product.

$$ (ka) \cdot b = k(a \cdot b) $$

#### Orthogonal Vectors

Two vectors are orthogonal if their dot product is zero.

$$ a \cdot b = 0 $$

### Characteristics / Key Points

- Input vectors must have the same dimension.
- The output is always a scalar value.
- Dot product measures similarity between vectors.
- It is the foundation of many machine learning operations.
- Vector multiplication in neural networks is based on dot products.
- Computational complexity is linear with vector size.

### Why is it used? / Applications

The dot product is widely used in machine learning, deep learning, and mathematics.

Applications:

- Neural networks:
  - Computing weighted sums of inputs.

- Linear regression:
  - Calculating predictions using feature weights.

- Similarity calculations:
  - Measuring similarity between embeddings.

- Computer graphics:
  - Calculating lighting and angles.

- Physics:
  - Computing projections and work done by forces.

> 💡 **Important Note**
>
> The dot product is one of the most important operations in machine learning. A neuron in a neural network essentially performs a dot product between input features and weights, followed by an activation function.

## Solutions

### Custom Implementation

```python id="9b3j8x"
import numpy as np

def calculate_dot_product(vec1, vec2):
    return np.dot(vec1, vec2)
```

## Code Explanation

### Step 1: Receive Two Vectors

The function accepts two one-dimensional NumPy arrays.

Example:

```python
vec1 = np.array([1, 2, 3])
vec2 = np.array([4, 5, 6])
```

Both vectors must have the same length because each element needs a corresponding element for multiplication.

### Step 2: Perform Element-wise Multiplication

The dot product multiplies matching positions:

$$
[1,2,3] \times [4,5,6]
$$

produces:

$$
[1 \times 4, 2 \times 5, 3 \times 6]
$$

### Step 3: Sum the Products

The resulting values are added together:

$$ 4 + 10 + 18 = 32 $$

The final result is a single scalar value.

### NumPy Implementation

`np.dot()` internally performs the complete dot product operation efficiently.

For two 1D arrays:

```python
np.dot(vec1, vec2)
```

is equivalent to:

```python
sum(vec1[i] * vec2[i] for i in range(n))
```

## Time & Space Complexity

| Complexity | Value    |
| ---------- | -------- |
| Time       | **O(n)** |
| Space      | **O(1)** |

Where:

- **n** is the number of elements in the vectors.
- Each element is multiplied and added once.
- Only the accumulated result is stored.
