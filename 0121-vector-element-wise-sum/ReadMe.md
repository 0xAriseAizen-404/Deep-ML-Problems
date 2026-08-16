# Vector Element-wise Sum (Easy, Linear Algebra)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Understanding Vector Element-wise Sum](#learn-understanding-vector-element-wise-sum)
- [Solution](#solution)
  - [Custom Implementation](#custom-implementation)

- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

---

## Problem Statement

### [Vector Element-wise Sum](https://www.deep-ml.com/problems/121)

Implement a Python function that computes the **element-wise sum** of two vectors.

Two vectors can be added element-wise only when they have the same dimension. If the vectors have different lengths, the operation is invalid and the function should return `-1`.

---

## Example

### Input

```python
a = [1, 3]
b = [4, 5]
```

### Output

```text
[5, 8]
```

### Reasoning

The corresponding elements of both vectors are added:

$$
[1+4,\ 3+5]
$$

Therefore:

$$
[5,8]
$$

---

## Learn: Understanding Vector Element-wise Sum

### What is it?

**Vector addition** is a fundamental operation in linear algebra where corresponding elements of two vectors are added together.

If two vectors have the same dimension, each element in the first vector is paired with the element at the same position in the second vector.

For example:

$$
\mathbf{a}=[1,2,3]
$$

and

$$
\mathbf{b}=[4,5,6]
$$

then:

$$
\mathbf{a}+\mathbf{b}=[1+4,\ 2+5,\ 3+6]
$$

which gives:

$$
\mathbf{a}+\mathbf{b}=[5,7,9]
$$

The result is another vector with the same dimension as the input vectors.

---

### Mathematical Definition

Let two vectors be:

$$
\mathbf{a}=[a_1,a_2,\ldots,a_n]
$$

and

$$
\mathbf{b}=[b_1,b_2,\ldots,b_n]
$$

Their element-wise sum is:

$$
\mathbf{a}+\mathbf{b}=[a_1+b_1,\ a_2+b_2,\ \ldots,\ a_n+b_n]
$$

Equivalently, each element of the resulting vector can be written as:

$$
c_i=a_i+b_i
$$

for:

$$
i=1,2,\ldots,n
$$

The resulting vector is:

$$
\mathbf{c}=\mathbf{a}+\mathbf{b}
$$

---

### Dimension Requirement

Vector addition is defined only when both vectors have the same dimension.

For example:

$$
\mathbf{a}=[1,2,3]
$$

and

$$
\mathbf{b}=[4,5,6]
$$

can be added because both vectors have dimension $3$.

However:

$$
\mathbf{a}=[1,2,3]
$$

and

$$
\mathbf{b}=[4,5]
$$

cannot be added element-wise because their dimensions are different.

There is no corresponding element in $\mathbf{b}$ for the third element of $\mathbf{a}$.

---

### Geometric Interpretation

Vectors can also be viewed as points or directed arrows in space.

When two vectors are added, the result can be interpreted geometrically using the **parallelogram rule**.

If $\mathbf{a}$ and $\mathbf{b}$ are represented as arrows starting from the same point, their sum points toward the opposite corner of the parallelogram formed by the two vectors.

For example:

$$
\mathbf{c}=\mathbf{a}+\mathbf{b}
$$

The operation therefore has both an algebraic and geometric interpretation.

---

### Properties of Vector Addition

Vector addition satisfies several important properties.

#### Commutativity

The order of addition does not matter:

$$
\mathbf{a}+\mathbf{b}=\mathbf{b}+\mathbf{a}
$$

For example:

$$
[1,2]+[3,4]=[4,6]
$$

and:

$$
[3,4]+[1,2]=[4,6]
$$

---

#### Associativity

When adding three vectors, the grouping does not affect the result:

$$
(\mathbf{a}+\mathbf{b})+\mathbf{c}=\mathbf{a}+(\mathbf{b}+\mathbf{c})
$$

This allows multiple vector additions to be performed in different groupings.

---

#### Additive Identity

The zero vector acts as the additive identity:

$$
\mathbf{a}+\mathbf{0}=\mathbf{a}
$$

For example:

$$
[1,2,3]+[0,0,0]=[1,2,3]
$$

---

#### Additive Inverse

Every vector has an additive inverse.

For:

$$
\mathbf{a}=[a_1,a_2,\ldots,a_n]
$$

its inverse is:

$$
-\mathbf{a}=[-a_1,-a_2,\ldots,-a_n]
$$

Therefore:

$$
\mathbf{a}+(-\mathbf{a})=\mathbf{0}
$$

---

### Vector Addition vs Scalar Addition

Vector addition should not be confused with adding a scalar to every vector element.

For two vectors:

$$
\mathbf{a}=[1,2,3]
$$

and:

$$
\mathbf{b}=[4,5,6]
$$

vector addition gives:

$$
\mathbf{a}+\mathbf{b}=[5,7,9]
$$

A scalar operation would instead be something like:

$$
\mathbf{a}+2
$$

which is a different operation and depends on the mathematical or programming context.

---

### Characteristics / Key Points

- Vector addition is performed element by element.
- Both vectors must have the same dimension.
- The output has the same dimension as the input vectors.
- Corresponding elements are added together.
- The operation is commutative.
- The operation is associative.
- The zero vector is the additive identity.
- Every vector has an additive inverse.
- The operation is fundamental to linear algebra.
- Mismatched dimensions make element-wise vector addition invalid.

---

### Why is it used? / Applications

Vector addition is one of the most frequently used operations in machine learning and linear algebra.

Applications include:

- Neural network computations.
- Feature vector manipulation.
- Linear transformations.
- Gradient updates.
- Physics simulations.
- Computer graphics.
- Robotics.
- Signal processing.
- Optimization algorithms.
- Embedding operations.

For example, gradient descent updates model parameters using vector subtraction:

$$
\boldsymbol{\theta}*{new}=\boldsymbol{\theta}*{old}-\alpha\nabla L
$$

Both $\boldsymbol{\theta}$ and $\nabla L$ are vectors of the same dimension, so their elements are combined component-wise.

Vector addition is therefore a fundamental operation behind many optimization and machine learning algorithms.

---

> 💡 **Important Note**
>
> Dimension compatibility is essential for vector addition. Unlike some matrix operations that have specific broadcasting rules in libraries such as NumPy, ordinary mathematical vector addition requires the vectors to have the same dimension.

> 💡 **Interview Tip**
>
> For an element-wise vector operation, always check the dimensions first. If the dimensions are equal, process corresponding elements; otherwise, the operation is invalid.

---

## Solution

### Custom Implementation

```python
def vector_sum(a: list[int | float], b: list[int | float]) -> list[int | float] | int:
    if len(a) != len(b):
        return -1
    return [x + y for x, y in zip(a, b)]
```

---

## Code Explanation

### Step 1: Check Dimensions

The first step is to verify that both vectors have the same length.

```python
if len(a) != len(b):
    return -1
```

If their lengths differ, element-wise addition is not defined, so the function immediately returns `-1`.

---

### Step 2: Pair Corresponding Elements

The `zip()` function pairs elements at the same positions:

```python
zip(a, b)
```

For:

```python
a = [1, 3]
b = [4, 5]
```

the pairs are effectively:

```text
(1, 4)
(3, 5)
```

---

### Step 3: Add Each Pair

A list comprehension computes the sum of every pair:

```python
[x + y for x, y in zip(a, b)]
```

This directly implements:

$$
c_i=a_i+b_i
$$

For the example:

$$
c_1=1+4=5
$$

$$
c_2=3+5=8
$$

Therefore, the final result is:

```text
[5, 8]
```

---

### Algorithm

1. Compare the lengths of the two input vectors.
2. Return `-1` if the lengths are different.
3. Pair corresponding elements using `zip`.
4. Add each pair.
5. Return the resulting vector.

The algorithm performs exactly one operation for each vector element.

---

## Time & Space Complexity

Let $n$ be the dimension of the vectors.

| Complexity | Value    |
| ---------- | -------- |
| Time       | **O(n)** |
| Space      | **O(n)** |

The algorithm visits each pair of corresponding elements once, giving **O(n)** time.

The resulting vector contains $n$ elements, requiring **O(n)** space.
