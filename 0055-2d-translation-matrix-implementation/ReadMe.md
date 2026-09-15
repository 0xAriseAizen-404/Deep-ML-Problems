# [2D Translation Matrix Implementation](https://www.deep-ml.com/problems/55) (Medium, Linear Algebra)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: 2D Translation Matrix](#learn-2d-translation-matrix)
  - [Concept Overview](#concept-overview)
  - [Homogeneous Coordinates](#homogeneous-coordinates)
  - [Translation Matrix](#translation-matrix)
  - [Applying the Translation](#applying-the-translation)
  - [Why a 3x3 Matrix](#why-a-3x3-matrix)
  - [Geometric Interpretation](#geometric-interpretation)
  - [Matrix Multiplication](#matrix-multiplication)
  - [Multiple Points](#multiple-points)
  - [Important Properties](#important-properties)
  - [Practical Applications](#practical-applications)
- [Solutions](#solutions)
  - [Custom Implementation](#custom-implementation)
  - [NumPy Implementation](#numpy-implementation)
- [Code Explanation](#code-explanation)
  - [Building the Translation Matrix](#building-the-translation-matrix)
  - [Converting to Homogeneous Coordinates](#converting-to-homogeneous-coordinates)
  - [Applying the Transformation](#applying-the-transformation)
  - [Removing the Homogeneous Coordinate](#removing-the-homogeneous-coordinate)
  - [Returning the Result](#returning-the-result)
- [Step-by-Step Example](#step-by-step-example)
- [Common Mistakes](#common-mistakes)
- [Interview and Practical Notes](#interview-and-practical-notes)
- [Time and Space Complexity](#time-and-space-complexity)

## Problem Statement

The task is to implement a function that applies a 2D translation
to a collection of points.

The function is:

```python
translate_object(points, tx, ty)
```

where:

- `points` is a list of `[x, y]` coordinates.
- `tx` is the translation distance along the x-axis.
- `ty` is the translation distance along the y-axis.
- The function returns a new list containing the translated points.

A translation moves an object without changing its:

- Shape
- Size
- Orientation
- Angles
- Relative distances between points

For a point $(x, y)$, translation by $(t_x, t_y)$ gives:

\((x', y') = (x + t_x, y + t_y)\)

The problem asks us to implement this transformation using a
translation matrix.

## Example

### Input

```python
points = [[0, 0], [1, 0], [0.5, 1]]
tx, ty = 2, 3

print(translate_object(points, tx, ty))
```

### Output

```text
[[2.0, 3.0], [3.0, 3.0], [2.5, 4.0]]
```

### Reasoning

The translation is:

\(t_x = 2,\quad t_y = 3\)

Therefore, every point receives `2` in its x-coordinate and
`3` in its y-coordinate.

For `[0, 0]`:

\((0 + 2,\ 0 + 3) = (2,\ 3)\)

For `[1, 0]`:

\((1 + 2,\ 0 + 3) = (3,\ 3)\)

For `[0.5, 1]`:

\((0.5 + 2,\ 1 + 3) = (2.5,\ 4)\)

Thus:

```text
[[2.0, 3.0], [3.0, 3.0], [2.5, 4.0]]
```

## Learn: 2D Translation Matrix

## Concept Overview

A translation is a geometric transformation that moves every point
of an object by the same amount.

Suppose we have a point:

\(P = (x, y)\)

and want to move it by:

\((t_x, t_y)\)

The translated point is:

\(P' = (x + t_x,\ y + t_y)\)

Translation is different from transformations such as scaling and
rotation because the transformation does not change the shape of
the object.

Every point is displaced by exactly the same vector:

\(\Delta P = (t_x, t_y)\)

This means that the distance between any two points remains unchanged.

For example, consider:

```text
A = (1, 2)
B = (4, 5)
```

The vector from A to B is:

\(B-A = (3,3)\)

After translating both points by $(5,2)$:

```text
A' = (6, 4)
B' = (9, 7)
```

The new difference is:

\(B'-A' = (3,3)\)

Therefore, translation preserves relative geometry.

## Homogeneous Coordinates

A normal 2D point contains two coordinates:

\(P = (x,y)\)

A 2D translation cannot be represented directly by multiplying
a standard 2x2 matrix with this point.

A 2x2 matrix can represent transformations such as:

- Rotation
- Scaling
- Reflection
- Shearing

But translation requires addition:

\(x' = x+t_x\)

\(y' = y+t_y\)

Matrix multiplication naturally represents linear operations.
Translation is affine rather than purely linear.

To represent translation using matrix multiplication, we introduce
a third coordinate.

The Cartesian point:

\(P\_{Cartesian} = (x,y)\)

is represented in homogeneous coordinates as:

\(P\_{Homogeneous} = (x,y,1)\)

The additional coordinate allows translation to be encoded inside
a matrix.

More generally, any non-zero scalar multiple represents the same
Cartesian point:

\((kx,ky,k) \sim (x,y,1)\)

for:

\(k \neq 0\)

For this problem, we use the standard representation with the
third coordinate equal to `1`.

## Translation Matrix

The 2D translation matrix is:

\(T = \begin{bmatrix}1&0&t_x\\0&1&t_y\\0&0&1\end{bmatrix}\)

The original point is represented as:

\(P = \begin{bmatrix}x\\y\\1\end{bmatrix}\)

The transformed point is:

\(P' = TP\)

Therefore:

\(\begin{bmatrix}1&0&t_x\\0&1&t_y\\0&0&1\end{bmatrix}\begin{bmatrix}x\\y\\1\end{bmatrix} = \begin{bmatrix}x+t_x\\y+t_y\\1\end{bmatrix}\)

The final homogeneous coordinate remains `1`.

We can therefore convert the result back to Cartesian coordinates
by removing the third coordinate.

## Applying the Translation

Matrix multiplication gives three equations.

For the first coordinate:

\(x' = 1x + 0y + t_x(1) = x+t_x\)

For the second coordinate:

\(y' = 0x + 1y + t_y(1) = y+t_y\)

For the third coordinate:

\(w' = 0x + 0y + 1(1) = 1\)

Therefore:

\(P' = (x+t_x,\ y+t_y)\)

The matrix representation and the direct coordinate representation
produce exactly the same result.

## Why a 3x3 Matrix

The third row and third column are not arbitrary.

The matrix:

\(\begin{bmatrix}1&0&t_x\\0&1&t_y\\0&0&1\end{bmatrix}\)

contains:

- `1` on the x-axis scale position.
- `1` on the y-axis scale position.
- `tx` as the x translation.
- `ty` as the y translation.
- The final `1` preserves the homogeneous coordinate.

Without the third coordinate, a standard linear transformation
cannot directly express:

\(x' = x+t_x\)

because matrix multiplication of a 2D vector only produces
linear combinations of `x` and `y`.

Homogeneous coordinates solve this by adding a constant `1`:

\(t_x(1) = t_x\)

and:

\(t_y(1) = t_y\)

This effectively turns translation into matrix multiplication.

## Geometric Interpretation

Think of translation as moving an entire object without modifying
its internal geometry.

For example:

```text
Before:

(0,1)
  /\
 /  \
(0,0)--(1,0)
```

Applying:

```text
tx = 2
ty = 3
```

moves every point by the same vector.

The triangle becomes:

```text
(2,4)
  /\
 /  \
(2,3)--(3,3)
```

The triangle is not rotated or scaled.

Its:

- Width remains the same.
- Height remains the same.
- Angles remain the same.
- Side lengths remain the same.

Only its position changes.

## Matrix Multiplication

For a matrix:

\(A = \begin{bmatrix}a&b&c\\d&e&f\\g&h&i\end{bmatrix}\)

and vector:

\(v = \begin{bmatrix}x\\y\\z\end{bmatrix}\)

the matrix-vector product is:

\(Av = \begin{bmatrix}ax+by+cz\\dx+ey+fz\\gx+hy+iz\end{bmatrix}\)

For the translation matrix:

\(T = \begin{bmatrix}1&0&t_x\\0&1&t_y\\0&0&1\end{bmatrix}\)

we obtain:

\(TP = \begin{bmatrix}x+t_x\\y+t_y\\1\end{bmatrix}\)

This is the mathematical core of the implementation.

## Multiple Points

The transformation matrix does not change when we process different
points.

For every point:

```text
[x, y]
```

we construct:

```text
[x, y, 1]
```

and apply the same matrix:

\(P_i' = TP_i\)

For points:

```text
P1 = [0, 0]
P2 = [1, 0]
P3 = [0.5, 1]
```

we have:

```text
P1 = [0, 0, 1]
P2 = [1, 0, 1]
P3 = [0.5, 1, 1]
```

The same translation matrix is applied to all three points.

This is important in computer graphics because an entire object
can contain hundreds, thousands, or millions of points.

Instead of designing a separate transformation for every point,
we construct one transformation matrix and apply it repeatedly.

## Important Properties

Translation preserves distances.

For two points $P_1$ and $P_2$:

\(||P_1-P_2|| = ||(P_1+t)-(P_2+t)||\)

because:

\((P_1+t)-(P_2+t)=P_1-P_2\)

Translation also preserves angles.

If two vectors are translated by the same amount, their difference
does not change.

Translation is therefore an example of a rigid transformation.

A rigid transformation preserves:

- Distances
- Angles
- Shape
- Area
- Orientation

A translation does not preserve absolute coordinates because every
coordinate changes by the translation vector.

## Translation Composition

Translations can be combined.

Suppose we first translate by:

\((t*{x1},t*{y1})\)

and then translate by:

\((t*{x2},t*{y2})\)

The total translation is:

\((t*{x1}+t*{x2},\ t*{y1}+t*{y2})\)

Using matrices:

\(T_2T_1P\)

produces the same result as one translation matrix:

\(T*{total} = \begin{bmatrix}1&0&t*{x1}+t*{x2}\\0&1&t*{y1}+t\_{y2}\\0&0&1\end{bmatrix}\)

This also demonstrates that translations are composable.

## Inverse Translation

Every translation has an inverse.

If a point is translated by:

\((t_x,t_y)\)

the inverse translation is:

\((-t_x,-t_y)\)

The inverse matrix is:

\(T^{-1} = \begin{bmatrix}1&0&-t_x\\0&1&-t_y\\0&0&1\end{bmatrix}\)

Applying both transformations returns the original point.

Therefore:

\(T^{-1}TP=P\)

This is useful when transformations need to be undone.

## Practical Applications

Translation matrices are heavily used in computer graphics.

Typical applications include:

- Moving objects in a scene.
- Positioning sprites in 2D games.
- Moving GUI elements.
- Transforming image coordinates.
- Robotics and geometric transformations.
- Computer vision.
- Animation.
- CAD systems.
- Coordinate-system transformations.

In a graphics pipeline, translation is usually combined with
other transformations such as rotation and scaling.

A common transformation pipeline is:

\(P' = TRSP\)

where the matrices represent different transformations.

The order of matrix multiplication matters because matrix
multiplication is generally not commutative.

## Solutions

## Custom Implementation

The submitted solution constructs the homogeneous translation matrix
using NumPy and then applies it to every point.

```python
import numpy as np

def translate_object(points, tx, ty):
    translation_values = [tx, ty]
    transformation_matrix = np.eye(len(translation_values) + 1)
    for axis_index in range(len(translation_values)):
        transformation_matrix[axis_index][-1] = translation_values[axis_index]

    translated_points = []
    for point in points:
        homogeneous_point = np.append(point, 1)
        transformed_point = transformation_matrix @ homogeneous_point
        transformed_point = transformed_point.tolist()
        transformed_point.pop()
        translated_points.append(transformed_point)
    return translated_points
```

The implementation follows the mathematical definition directly.

The important sequence is:

```text
translation values
        ↓
3x3 transformation matrix
        ↓
convert point to homogeneous coordinates
        ↓
matrix-vector multiplication
        ↓
remove homogeneous coordinate
        ↓
translated point
```

## NumPy Implementation

The same transformation can be expressed more compactly using
a fixed 3x3 translation matrix.

```python
import numpy as np

def translate_object(points, tx, ty):
    transformation_matrix = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
    translated_points = []
    for point in points:
        homogeneous_point = np.append(point, 1)
        transformed_point = transformation_matrix @ homogeneous_point
        translated_points.append(transformed_point[:2].tolist())
    return translated_points
```

This version is specifically written for 2D translation.

The submitted solution instead constructs the matrix from the number
of translation values, which makes its construction more generic.

## Code Explanation

## Building the Translation Matrix

The first step is:

```python
translation_values = [tx, ty]
```

This stores the translation amount for each spatial axis.

For a 2D problem:

```text
translation_values = [tx, ty]
```

Therefore, the required homogeneous matrix has dimension:

```text
3 x 3
```

The code creates it using:

```python
transformation_matrix = np.eye(len(translation_values) + 1)
```

Since:

```text
len([tx, ty]) = 2
```

the expression becomes:

```python
np.eye(3)
```

which initially produces:

```text
[[1, 0, 0],
 [0, 1, 0],
 [0, 0, 1]]
```

The final column is then populated with the translation values.

## Inserting Translation Values

The loop is:

```python
for axis_index in range(len(translation_values)):
    transformation_matrix[axis_index][-1] = translation_values[axis_index]
```

For `tx = 2` and `ty = 3`, the first iteration places `2` in:

```text
matrix[0][-1]
```

and the second iteration places `3` in:

```text
matrix[1][-1]
```

The resulting matrix is:

```text
[[1, 0, 2],
 [0, 1, 3],
 [0, 0, 1]]
```

This is exactly the standard 2D translation matrix.

## Converting to Homogeneous Coordinates

A normal point is represented as:

```python
point = [x, y]
```

The transformation matrix requires a three-dimensional vector.

The code therefore uses:

```python
homogeneous_point = np.append(point, 1)
```

For:

```python
point = [0.5, 1]
```

we obtain:

```text
[0.5, 1, 1]
```

The final `1` is the homogeneous coordinate.

This allows the translation matrix to perform addition through
matrix multiplication.

## Applying the Transformation

The actual transformation is:

```python
transformed_point = transformation_matrix @ homogeneous_point
```

The `@` operator performs matrix multiplication.

For:

```text
T = [[1, 0, 2],
     [0, 1, 3],
     [0, 0, 1]]
```

and:

```text
P = [0.5, 1, 1]
```

we obtain:

```text
T @ P = [2.5, 4, 1]
```

The first coordinate becomes:

\(1(0.5)+0(1)+2(1)=2.5\)

The second coordinate becomes:

\(0(0.5)+1(1)+3(1)=4\)

The third coordinate remains:

\(1\)

Therefore:

```text
[2.5, 4, 1]
```

is the transformed homogeneous point.

## Removing the Homogeneous Coordinate

The result still contains the third coordinate:

```text
[2.5, 4, 1]
```

But the required output format is:

```text
[2.5, 4]
```

The code converts the NumPy array to a Python list:

```python
transformed_point = transformed_point.tolist()
```

Then removes the final coordinate:

```python
transformed_point.pop()
```

The resulting point is:

```text
[2.5, 4.0]
```

This is then added to the output list.

## Returning the Result

The output container is initialized with:

```python
translated_points = []
```

Every transformed point is appended:

```python
translated_points.append(transformed_point)
```

After all points have been processed:

```python
return translated_points
```

The function therefore returns a new list instead of modifying the
original `points` list.

This is useful because the original geometry remains available.

## Step-by-Step Example

Consider:

```python
points = [[0, 0], [1, 0], [0.5, 1]]
tx = 2
ty = 3
```

The transformation matrix is:

```text
[[1, 0, 2],
 [0, 1, 3],
 [0, 0, 1]]
```

### Point 1

Original:

```text
[0, 0]
```

Homogeneous:

```text
[0, 0, 1]
```

Matrix multiplication:

\(\begin{bmatrix}1&0&2\\0&1&3\\0&0&1\end{bmatrix}\begin{bmatrix}0\\0\\1\end{bmatrix} = \begin{bmatrix}2\\3\\1\end{bmatrix}\)

After removing the final coordinate:

```text
[2, 3]
```

### Point 2

Original:

```text
[1, 0]
```

Homogeneous:

```text
[1, 0, 1]
```

Transformation:

\(\begin{bmatrix}1&0&2\\0&1&3\\0&0&1\end{bmatrix}\begin{bmatrix}1\\0\\1\end{bmatrix} = \begin{bmatrix}3\\3\\1\end{bmatrix}\)

Final point:

```text
[3, 3]
```

### Point 3

Original:

```text
[0.5, 1]
```

Homogeneous:

```text
[0.5, 1, 1]
```

Transformation:

\(\begin{bmatrix}1&0&2\\0&1&3\\0&0&1\end{bmatrix}\begin{bmatrix}0.5\\1\\1\end{bmatrix} = \begin{bmatrix}2.5\\4\\1\end{bmatrix}\)

Final point:

```text
[2.5, 4]
```

Therefore:

```text
[[2.0, 3.0], [3.0, 3.0], [2.5, 4.0]]
```

## Direct Interpretation

Although the problem asks for a matrix implementation, it is
important to recognize the underlying operation.

The matrix multiplication is mathematically equivalent to:

```python
x_new = x + tx
y_new = y + ty
```

For every point.

The matrix formulation becomes valuable when translation is combined
with other geometric transformations.

For example, a graphics engine may represent several transformations
as matrices and combine them into one transformation matrix.

The direct formula is simpler for translation alone, while the matrix
form is more powerful as part of a larger transformation pipeline.

## Common Mistakes

### Forgetting Homogeneous Coordinates

A common mistake is trying to multiply:

```text
2x2 matrix × [x, y]
```

to perform translation.

A normal 2x2 linear transformation cannot directly represent:

\(x' = x+t_x\)

Homogeneous coordinates solve this problem.

### Using the Wrong Matrix

The standard 2D translation matrix is:

\(\begin{bmatrix}1&0&t_x\\0&1&t_y\\0&0&1\end{bmatrix}\)

The translation values belong in the final column when points
are represented as column vectors.

### Forgetting the Third Coordinate

The input:

```text
[x, y]
```

must become:

```text
[x, y, 1]
```

before multiplication.

### Returning the Homogeneous Coordinate

The transformed result contains:

```text
[x + tx, y + ty, 1]
```

The final `1` is not part of the requested 2D coordinate.

It should be removed before returning the result.

### Modifying the Original Points

The function should return a new collection.

Changing the original list directly can create unexpected side effects
for code that still needs the original geometry.

### Confusing Translation with Scaling

Scaling uses a matrix such as:

\(\begin{bmatrix}s_x&0&0\\0&s_y&0\\0&0&1\end{bmatrix}\)

Translation instead changes the final column:

\(\begin{bmatrix}1&0&t_x\\0&1&t_y\\0&0&1\end{bmatrix}\)

### Confusing Translation with Rotation

Rotation changes the orientation of the object:

\(R = \begin{bmatrix}\cos\theta&-\sin\theta&0\\\sin\theta&\cos\theta&0\\0&0&1\end{bmatrix}\)

Translation does not change orientation.

## Interview and Practical Notes

### Is Translation a Linear Transformation?

Strictly speaking, translation is not a linear transformation in
ordinary Cartesian coordinates because a linear transformation must
map the zero vector to the zero vector.

For a non-zero translation:

\(T(0,0)=(t_x,t_y)\)

which is not the zero vector.

Therefore, translation is an affine transformation.

Homogeneous coordinates allow affine transformations to be represented
using matrix multiplication.

### Why Are Homogeneous Coordinates Useful?

Homogeneous coordinates allow multiple transformations to use the
same matrix-based representation.

For example:

- Translation
- Rotation
- Scaling
- Shearing
- Reflection

can all be represented using 3x3 matrices in 2D.

This makes it possible to compose transformations through matrix
multiplication.

### Does Translation Change Distances?

No.

For points $P_1$ and $P_2$:

\(||(P_1+t)-(P_2+t)|| = ||P_1-P_2||\)

Therefore, all pairwise distances remain unchanged.

### Does Translation Change Angles?

No.

Because translation moves every point by the same vector, the
direction of vectors between points remains unchanged.

Therefore, angles are preserved.

### Does Translation Change Area?

No.

A translated object has exactly the same area as the original.

For a triangle, rectangle, polygon, or any other planar shape,
translation only changes its position.

### Can Translation Be Reversed?

Yes.

If:

```text
tx = 2
ty = 3
```

the inverse translation is:

```text
tx = -2
ty = -3
```

Applying both returns every point to its original position.

### Why Use NumPy?

NumPy provides efficient matrix and vector operations.

The expression:

```python
transformation_matrix @ homogeneous_point
```

directly corresponds to the mathematical matrix-vector product.

This makes the implementation close to the mathematical definition
and avoids manually computing every matrix multiplication term.

## Numerical Considerations

The translation itself is numerically simple because it mainly
involves addition.

For example:

```python
0.5 + 2
```

produces:

```text
2.5
```

Floating-point representation can still introduce tiny numerical
errors for some decimal values.

For this problem, these errors are normally negligible.

The implementation also starts with:

```python
np.eye(...)
```

which creates floating-point values when translation values are
assigned.

Therefore, outputs such as:

```text
2.0
```

are expected even when the mathematical answer is simply `2`.

## Generalization

The submitted implementation determines the matrix size from the
number of translation values:

```python
translation_values = [tx, ty]
```

Then:

```python
len(translation_values) + 1
```

produces the homogeneous dimension.

For a 2D point:

```text
2 coordinates + 1 homogeneous coordinate = 3
```

For a 3D point:

```text
3 coordinates + 1 homogeneous coordinate = 4
```

A 3D translation would therefore use a 4x4 matrix.

The 2D case is:

\(T\_{2D} = \begin{bmatrix}1&0&t_x\\0&1&t_y\\0&0&1\end{bmatrix}\)

The 3D case is:

\(T\_{3D} = \begin{bmatrix}1&0&0&t_x\\0&1&0&t_y\\0&0&1&t_z\\0&0&0&1\end{bmatrix}\)

The underlying idea remains identical.

## Why the Matrix Is Reused

The transformation matrix depends only on:

```text
tx
ty
```

It does not depend on the individual point.

Therefore, it is constructed once before the loop.

This is algorithmically important.

We should avoid rebuilding the same matrix for every point.

The structure is:

```python
transformation_matrix = ...
for point in points:
    ...
```

rather than:

```python
for point in points:
    transformation_matrix = ...
```

The first approach avoids unnecessary repeated work.

## Vectorized NumPy Approach

If the points are already stored as a NumPy array, the transformation
can be vectorized.

For example:

```python
import numpy as np

def translate_object(points, tx, ty):
    points = np.asarray(points, dtype=float)
    homogeneous_points = np.hstack((points, np.ones((len(points), 1))))
    transformation_matrix = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
    transformed_points = homogeneous_points @ transformation_matrix.T
    return transformed_points[:, :2].tolist()
```

This approach processes all points through array operations rather
than explicitly looping through them in Python.

The mathematical operation remains the same.

## Direct Coordinate Implementation

If the matrix representation is not required, translation can be
implemented much more simply:

```python
def translate_object(points, tx, ty):
    return [[x + tx, y + ty] for x, y in points]
```

This is computationally simpler for this specific task.

However, the matrix implementation is more educational because it
demonstrates homogeneous coordinates and the standard representation
used in computer graphics.

## Time and Space Complexity

Let:

- $N$ = number of points.

Each point requires:

- Creating a homogeneous vector.
- Multiplying a 3x3 matrix by a 3x1 vector.
- Converting the result to a list.
- Removing the homogeneous coordinate.
- Appending the result.

Because the matrix dimension is fixed at 3x3, the work per point
is constant.

Therefore:

\(T(N) = O(N)\)

The transformation matrix itself requires constant space:

\(O(1)\)

Each transformed point requires constant space.

The output contains $N$ points, so the output space is:

\(O(N)\)

Therefore, the total auxiliary/output space used by the implementation
is:

\(O(N)\)

If the returned output is included in space complexity, the total
space complexity is:

\(O(N)\)

The important distinction is that the algorithm does not require
space proportional to the number of points for intermediate matrix
storage; the $O(N)$ space comes primarily from storing the output.

## Complexity Summary

| Operation                    | Complexity |
| ---------------------------- | ---------: |
| Construct translation matrix |     $O(1)$ |
| Transform one point          |     $O(1)$ |
| Transform N points           |     $O(N)$ |
| Output storage               |     $O(N)$ |
| Auxiliary matrix storage     |     $O(1)$ |
| Total space including output |     $O(N)$ |

## Key Takeaways

1. Translation moves every point by the same vector.

2. The direct mathematical transformation is:

\((x,y) \rightarrow (x+t_x,\ y+t_y)\)

3. Translation is an affine transformation rather than a linear
   transformation in ordinary Cartesian coordinates.

4. Homogeneous coordinates add a third coordinate:

\((x,y) \rightarrow (x,y,1)\)

5. The standard 2D translation matrix is:

\(T = \begin{bmatrix}1&0&t_x\\0&1&t_y\\0&0&1\end{bmatrix}\)

6. Matrix multiplication produces:

\(TP = \begin{bmatrix}x+t_x\\y+t_y\\1\end{bmatrix}\)

7. The final homogeneous coordinate is removed to recover the
   ordinary 2D point.

8. Translation preserves distances, angles, shape, area, and
   orientation.

9. The inverse of a translation by $(t_x,t_y)$ is a translation by
   $(-t_x,-t_y)$.

10. Multiple translations can be combined by adding their translation
    vectors.

11. Translation matrices are especially useful when combined with
    rotation, scaling, and other geometric transformations.

12. For $N$ points, the implementation runs in:

\(O(N)\)

and requires:

\(O(N)\)

space when the output is included.
