# Compute the Cross Product of Two 3D Vectors (Easy, Linear Algebra)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Understanding the Cross Product](#learn-understanding-the-cross-product)
- [Solution](#solution)
  - [Custom Implementation](#custom-implementation)
  - [NumPy Implementation](#numpy-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

---

## Problem Statement

### [Compute the Cross Product of Two 3D Vectors](https://www.deep-ml.com/problems/118)

Implement a function that computes the **cross product** of two 3-dimensional vectors.

For two vectors $\mathbf{a}$ and $\mathbf{b}$, their cross product produces a third vector that is perpendicular to both input vectors.

The resulting vector follows the **right-hand rule**, and its magnitude represents the area of the parallelogram formed by the two vectors.

---

## Example

### Input

```python
cross_product([1, 0, 0], [0, 1, 0])
```

### Output

```text
[0, 0, 1]
```

### Reasoning

Let

$$
\mathbf{a}=[1,0,0]
$$

and

$$
\mathbf{b}=[0,1,0]
$$

Using the cross product formula:

$$
\mathbf{a}\times\mathbf{b}=[a_2b_3-a_3b_2,\ a_3b_1-a_1b_3,\ a_1b_2-a_2b_1]
$$

we obtain:

$$
\mathbf{a}\times\mathbf{b}=[0(0)-0(1),\ 0(0)-1(0),\ 1(1)-0(0)]
$$

Therefore:

$$
\mathbf{a}\times\mathbf{b}=[0,0,1]
$$

The result points in the positive $z$-direction and is perpendicular to both input vectors.

---

## Learn: Understanding the Cross Product

### What is it?

The **cross product** is a vector operation defined for vectors in three-dimensional space.

Given two vectors $\mathbf{a}$ and $\mathbf{b}$, their cross product produces a new vector:

$$
\mathbf{c}=\mathbf{a}\times\mathbf{b}
$$

The resulting vector $\mathbf{c}$ is perpendicular to both $\mathbf{a}$ and $\mathbf{b}$.

Unlike the **dot product**, which produces a scalar, the cross product produces another vector.

The cross product is therefore useful whenever both **magnitude and direction** are important.

---

### Mathematical Definition

Let

$$
\mathbf{a}=[a_1,a_2,a_3]
$$

and

$$
\mathbf{b}=[b_1,b_2,b_3]
$$

Then their cross product is:

$$
\mathbf{a}\times\mathbf{b}=[a_2b_3-a_3b_2,\ a_3b_1-a_1b_3,\ a_1b_2-a_2b_1]
$$

The three components can therefore be calculated independently:

$$
c_1=a_2b_3-a_3b_2
$$

$$
c_2=a_3b_1-a_1b_3
$$

$$
c_3=a_1b_2-a_2b_1
$$

Thus:

$$
\mathbf{c}=[c_1,c_2,c_3]
$$

---

### Determinant Representation

The cross product can be remembered using a determinant-like representation:

$$
\mathbf{a}\times\mathbf{b}=
\begin{vmatrix}
\mathbf{i} & \mathbf{j} & \mathbf{k}\
a_1 & a_2 & a_3\
b_1 & b_2 & b_3
\end{vmatrix}
$$

Expanding along the first row gives:

$$
\mathbf{a}\times\mathbf{b}=
\mathbf{i}(a_2b_3-a_3b_2)
-\mathbf{j}(a_1b_3-a_3b_1)
+\mathbf{k}(a_1b_2-a_2b_1)
$$

Rearranging the middle component:

$$
\mathbf{a}\times\mathbf{b}=[a_2b_3-a_3b_2,\ a_3b_1-a_1b_3,\ a_1b_2-a_2b_1]
$$

This determinant form is a convenient way to remember the component formula.

---

### Magnitude of the Cross Product

The magnitude of the cross product is:

$$
|\mathbf{a}\times\mathbf{b}|=|\mathbf{a}||\mathbf{b}|\sin(\theta)
$$

where $\theta$ is the angle between $\mathbf{a}$ and $\mathbf{b}$.

This gives an important geometric interpretation.

The magnitude of the cross product equals the **area of the parallelogram** formed by the two vectors.

If the base has length $|\mathbf{a}|$ and the perpendicular height is $|\mathbf{b}|\sin(\theta)$, then:

$$
\text{Area}=|\mathbf{a}||\mathbf{b}|\sin(\theta)
$$

Therefore:

$$
\text{Area}=|\mathbf{a}\times\mathbf{b}|
$$

---

### Direction and the Right-Hand Rule

The direction of $\mathbf{a}\times\mathbf{b}$ is determined by the **right-hand rule**.

Point the index finger of your right hand in the direction of $\mathbf{a}$ and rotate it toward $\mathbf{b}$. Your thumb points in the direction of:

$$
\mathbf{a}\times\mathbf{b}
$$

Reversing the order reverses the direction:

$$
\mathbf{b}\times\mathbf{a}=-(\mathbf{a}\times\mathbf{b})
$$

Therefore, the cross product is **anti-commutative**.

---

### Perpendicularity

The resulting vector is perpendicular to both original vectors.

Therefore:

$$
(\mathbf{a}\times\mathbf{b})\cdot\mathbf{a}=0
$$

and

$$
(\mathbf{a}\times\mathbf{b})\cdot\mathbf{b}=0
$$

This provides a useful way to verify a cross product implementation.

For example:

$$
[0,0,1]\cdot[1,0,0]=0
$$

and

$$
[0,0,1]\cdot[0,1,0]=0
$$

Hence, $[0,0,1]$ is perpendicular to both vectors.

---

### Parallel and Collinear Vectors

If two vectors point in the same or opposite direction, they are parallel.

For parallel vectors:

$$
\theta=0
$$

or

$$
\theta=\pi
$$

Since:

$$
\sin(0)=0
$$

and

$$
\sin(\pi)=0
$$

their cross product is zero:

$$
\mathbf{a}\times\mathbf{b}=\mathbf{0}
$$

This means the parallelogram formed by the vectors has zero area.

---

### Orthogonal Vectors

If two non-zero vectors are perpendicular:

$$
\theta=\frac{\pi}{2}
$$

Since:

$$
\sin\left(\frac{\pi}{2}\right)=1
$$

the magnitude becomes:

$$
|\mathbf{a}\times\mathbf{b}|=|\mathbf{a}||\mathbf{b}|
$$

For example:

$$
[1,0,0]\times[0,1,0]=[0,0,1]
$$

Both vectors have magnitude $1$, so the resulting vector also has magnitude $1$.

---

### Cross Product vs Dot Product

The dot product and cross product provide different types of information.

| Operation     | Result | Main Interpretation                    |
| ------------- | ------ | -------------------------------------- |
| Dot Product   | Scalar | Measures alignment                     |
| Cross Product | Vector | Gives perpendicular direction and area |

The dot product is:

$$
\mathbf{a}\cdot\mathbf{b}=|\mathbf{a}||\mathbf{b}|\cos(\theta)
$$

The cross product magnitude is:

$$
|\mathbf{a}\times\mathbf{b}|=|\mathbf{a}||\mathbf{b}|\sin(\theta)
$$

Thus, the dot product is closely related to the cosine of the angle, while the cross product magnitude is related to the sine.

---

### Characteristics / Key Points

- The cross product is defined for 3D vectors.
- It produces a vector perpendicular to both input vectors.
- Its direction follows the right-hand rule.
- Its magnitude equals the area of the parallelogram formed by the vectors.
- It is anti-commutative.
- Reversing the operands changes the sign.
- Parallel vectors produce the zero vector.
- Orthogonal vectors produce the maximum magnitude for fixed vector lengths.
- The cross product is not commutative.
- The zero vector is produced if either input vector is zero.
- The operation is bilinear.
- Cross products are commonly represented using determinants.

Important identities include:

$$
\mathbf{a}\times\mathbf{b}=-(\mathbf{b}\times\mathbf{a})
$$

$$
\mathbf{a}\times\mathbf{a}=\mathbf{0}
$$

$$
\mathbf{a}\times(\mathbf{b}+\mathbf{c})=\mathbf{a}\times\mathbf{b}+\mathbf{a}\times\mathbf{c}
$$

---

### Why is it used? / Applications

Cross products are important in several areas involving 3D geometry and vectors.

Applications include:

- Computing surface normals in 3D graphics.
- Calculating torque in physics.
- Computing angular momentum.
- Robotics and spatial transformations.
- Computer graphics.
- Game development.
- Geometry and computational geometry.
- Determining orientation in 3D space.
- Calculating areas of parallelograms.
- Constructing perpendicular vectors.

For example, in 3D graphics, two vectors lying on a surface can be crossed to obtain a vector normal to that surface:

$$
\mathbf{n}=\mathbf{a}\times\mathbf{b}
$$

The normal vector can then be normalized:

$$
\hat{\mathbf{n}}=\frac{\mathbf{n}}{|\mathbf{n}|}
$$

Surface normals are important for lighting and shading calculations.

---

> 💡 **Important Note**
>
> In 3D, the cross product of two vectors is **not their determinant**. A determinant is defined for square matrices, while two 3D vectors form a $3\times2$ matrix. The cross product is constructed using $2\times2$ determinant-like expressions, and its magnitude represents the parallelogram area.

> 💡 **Interview Tip**
>
> Remember the component pattern: the first component uses indices `(2,3)`, the second uses `(3,1)`, and the third uses `(1,2)`. The middle component has the opposite sign in the determinant expansion.

---

## Solution

### Custom Implementation

```python
def cross_product(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    ]
```

### NumPy Implementation

```python
import numpy as np

def cross_product(a, b):
    return np.cross(a, b)
```

---

## Code Explanation

### Step 1: Compute the First Component

The first component is:

$$
c_1=a_2b_3-a_3b_2
$$

The implementation computes:

```python
a[1] * b[2] - a[2] * b[1]
```

Python uses zero-based indexing, so mathematical indices $1,2,3$ correspond to Python indices `0,1,2`.

---

### Step 2: Compute the Second Component

The second component is:

$$
c_2=a_3b_1-a_1b_3
$$

The implementation computes:

```python
a[2] * b[0] - a[0] * b[2]
```

The sign of this component is important because of the alternating signs in the determinant expansion.

---

### Step 3: Compute the Third Component

The third component is:

$$
c_3=a_1b_2-a_2b_1
$$

The implementation computes:

```python
a[0] * b[1] - a[1] * b[0]
```

---

### Step 4: Construct the Result

The three calculated components are returned as one vector:

```python
[
    c1,
    c2,
    c3
]
```

Therefore:

$$
\mathbf{a}\times\mathbf{b}=[c_1,c_2,c_3]
$$

For the example:

```python
a = [1, 0, 0]
b = [0, 1, 0]
```

the components become:

$$
c_1=0(0)-0(1)=0
$$

$$
c_2=0(0)-1(0)=0
$$

$$
c_3=1(1)-0(0)=1
$$

Therefore:

$$
\mathbf{a}\times\mathbf{b}=[0,0,1]
$$

---

### Geometric Interpretation

The implementation computes more than three arithmetic expressions. Together, those expressions encode two important properties:

1. The magnitude represents the area of the parallelogram formed by the vectors.
2. The direction is perpendicular to both vectors and follows the right-hand rule.

The implementation therefore directly represents the geometric definition of the cross product.

---

## Time & Space Complexity

The cross product always requires exactly three component calculations.

| Complexity | Value    |
| ---------- | -------- |
| Time       | **O(1)** |
| Space      | **O(1)** |

The input vectors always contain exactly three elements, so the number of arithmetic operations does not grow with the input size.

The returned vector contains three elements, which also requires constant space.
