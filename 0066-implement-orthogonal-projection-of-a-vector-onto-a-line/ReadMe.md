# Implement Orthogonal Projection of a Vector onto a Line (Easy, Linear Algebra)

## Table of Contents

- Problem Statement
- Example
- Learn: Orthogonal Projection
- Solution
- Code Explanation
- Time & Space Complexity

---

## Problem Statement

[Implement Orthogonal Projection of a Vector onto a Line](https://www.deep-ml.com/problems/66)

Write a Python function `orthogonal_projection(v, L)` that computes the **orthogonal projection** of a vector **v** onto another vector **L**, which defines a line.

The function should return the projection vector as a list, with each component rounded to **three decimal places**.

The projection represents the **closest point on the line** defined by **L** to the vector **v**.

---

## Example

```python
v = [3, 4]
L = [1, 0]

print(orthogonal_projection(v, L))
```

### Output

```text
[3.0, 0.0]
```

### Explanation

The vector

\[
v=
\begin{bmatrix}
3\\
4
\end{bmatrix}
\]

is projected onto the x-axis represented by

\[
L=
\begin{bmatrix}
1\\
0
\end{bmatrix}
\]

Since **L** points entirely in the x-direction, only the x-component of **v** remains.

Therefore,

\[
\operatorname{proj}_L(v)
=
\begin{bmatrix}
3\\
0
\end{bmatrix}
\]

---

# Learn: Orthogonal Projection

## What is it?

An **orthogonal projection** is the process of finding the component of one vector that lies **along the direction of another vector**.

Imagine shining a flashlight directly above a vector.

The shadow cast onto another vector represents its **projection**.

Only the component **parallel** to the line remains, while the perpendicular component is discarded.

The projection is called **orthogonal** because the remaining error vector is **perpendicular** to the projection direction.

---

## Geometric Intuition

Suppose we have two vectors

- **v** → the vector being projected.
- **L** → the direction (line).

```text
                v
               ●
              /|
             / |
            /  |
           /   |
          /θ   |
         /     |
--------●------●-----------------> L
       Origin  Projection
```

The projection is simply the point on **L** that is **closest** to **v**.

The leftover vector

\[
v-\operatorname{proj}_L(v)
\]

is always perpendicular to **L**.

---

## Mathematical Definition / Formula

The orthogonal projection of **v** onto **L** is

\[
\operatorname{proj}_L(v)
=
\frac{v\cdot L}{L\cdot L}L
\]

where

- \(v\cdot L\) is the dot product.
- \(L\cdot L=||L||^2\).
- The scalar

\[
\frac{v\cdot L}{L\cdot L}
\]

determines **how far** to travel along **L**.

---

### Using a Unit Vector

If

\[
\hat{L}
=
\frac{L}{||L||}
\]

is a unit vector, then the projection becomes

\[
\operatorname{proj}_L(v)
=
(v\cdot\hat{L})\hat{L}
\]

Both formulas produce the same result.

---

## Why Does This Formula Work?

The projection vector needs two things:

- **Correct Magnitude**
- **Correct Direction**

The direction is easy.

It must point in the same direction as **L**.

So we use the vector **L**.

The remaining task is determining **how much of L** we need.

---

### Step 1: Find the Projection Length

From trigonometry,

\[
\cos\theta
=
\frac{\text{Projection Length}}{||v||}
\]

Therefore,

\[
\text{Projection Length}
=
||v||\cos\theta
\]

---

### Step 2: Connect with the Dot Product

Recall that

\[
v\cdot L
=
||v||
\,||L||
\cos\theta
\]

Rearranging,

\[
||v||
\cos\theta
=
\frac{v\cdot L}{||L||}
\]

Thus,

\[
\text{Projection Length}
=
\frac{v\cdot L}{||L||}
\]

---

### Step 3: Convert Length into a Vector

A magnitude alone is not enough.

We need a **vector**.

The unit vector along **L** is

\[
\hat{L}
=
\frac{L}{||L||}
\]

Multiplying the projection length by the unit vector gives

\[
\operatorname{proj}_L(v)
=
\frac{v\cdot L}{||L||}
\times
\frac{L}{||L||}
\]

Combining the denominators,

\[
\operatorname{proj}_L(v)
=
\frac{v\cdot L}{||L||^2}L
\]

Since

\[
||L||^2
=
L\cdot L
\]

the final formula becomes

\[
\boxed{
\operatorname{proj}_L(v)
=
\frac{v\cdot L}{L\cdot L}L
}
\]

---

## Characteristics / Key Points

- Projection is always parallel to **L**.
- The residual vector

\[
v-\operatorname{proj}_L(v)
\]

is perpendicular to **L**.

- If **v** is already parallel to **L**, the projection equals **v**.
- If **v** is perpendicular to **L**, the projection is the zero vector.
- The direction of the projection always matches the direction of **L**.

---

## Special Cases

### Parallel Vectors

If

\[
v=kL
\]

then

\[
\operatorname{proj}_L(v)=v
\]

---

### Perpendicular Vectors

If

\[
v\cdot L=0
\]

then

\[
\operatorname{proj}_L(v)=0
\]

---

### Zero Vector

If

\[
L=0
\]

then

\[
L\cdot L=0
\]

and the projection is **undefined** because division by zero occurs.

---

## Why is it used? / Applications

Orthogonal projection is fundamental in many areas of mathematics and machine learning.

Applications include:

- Least Squares Regression
- Principal Component Analysis (PCA)
- Gram-Schmidt Orthogonalization
- Computer Graphics
- Computer Vision
- Robotics
- Signal Processing
- Physics
- Optimization
- Recommendation Systems

Projection is one of the most frequently used operations in **Linear Algebra**.

> 💡 **Important Note**
>
> Orthogonal projection is the foundation of **Linear Regression**. In Ordinary Least Squares (OLS), the predicted output is the orthogonal projection of the target vector onto the column space of the feature matrix. Understanding this concept makes many machine learning algorithms much easier to visualize.

---

# Solution

## Custom Implementation

```python
import numpy as np

def orthogonal_projection(v, L):
    v = np.asarray(v)
    L = np.asarray(L)

    return np.round(
        (v.dot(L) / L.dot(L)) * L,
        3
    )
```

---

# Code Explanation

### Step 1: Convert Inputs into NumPy Arrays

```python
v = np.asarray(v)
L = np.asarray(L)
```

This enables efficient vector operations such as dot products and scalar multiplication.

---

### Step 2: Compute the Dot Product

```python
v.dot(L)
```

This computes

\[
v\cdot L
\]

which measures how much of **v** points in the direction of **L**.

---

### Step 3: Compute the Squared Magnitude of L

```python
L.dot(L)
```

This computes

\[
L\cdot L
=
||L||^2
\]

which is the squared length of **L**.

---

### Step 4: Compute the Projection Scalar

```python
v.dot(L) / L.dot(L)
```

This determines **how far along L** the projection lies.

---

### Step 5: Compute the Projection Vector

```python
(v.dot(L) / L.dot(L)) * L
```

Multiplying the scalar by **L** produces the projection vector.

Finally,

```python
np.round(..., 3)
```

rounds each component to three decimal places.

---

## Time & Space Complexity

Let

- \(n\) = Dimension of the vectors.

| Complexity | Value |
| ---------- | ----- |
| Time | **O(n)** |
| Space | **O(n)** |

The algorithm performs two dot products and one scalar-vector multiplication, each requiring **O(n)** time. The returned projection vector also requires **O(n)** space.