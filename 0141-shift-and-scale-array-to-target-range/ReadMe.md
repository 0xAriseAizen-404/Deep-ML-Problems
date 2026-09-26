# Shift and Scale Array to Target Range (Easy, Machine Learning)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Shifting and Scaling a Range](#learn-shifting-and-scaling-a-range)
  - [What is Range Scaling?](#what-is-range-scaling)
  - [Mathematical Definition](#mathematical-definition)
  - [Deriving the Formula](#deriving-the-formula)
  - [Properties of the Mapping](#properties-of-the-mapping)
  - [Constant Arrays](#constant-arrays)
  - [1D and 2D Arrays](#1d-and-2d-arrays)
  - [Applications](#applications)
  - [Important Considerations](#important-considerations)

- [Solutions](#solutions)
  - [Custom NumPy Implementation](#custom-numpy-implementation)
  - [Equivalent Min-Max Scaling](#equivalent-min-max-scaling)

- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

---

## Problem Statement

### [Shift and Scale Array to Target Range](https://www.deep-ml.com/problems/141)

Implement a function `convert_range` that maps the values of a NumPy array from its original range `[a,b]` to a target range `[c,d]`.

The original range is determined automatically:

\(a = \min(x)\)

\(b = \max(x)\)

The target range is provided through `c` and `d`.

The function must:

- Work with 1D arrays.
- Work with 2D arrays.
- Preserve the original shape.
- Return floating-point values.
- Use only NumPy.
- Correctly map the original minimum to `c`.
- Correctly map the original maximum to `d`.
- Handle constant arrays where `min(x) = max(x)`.

The required mapping is:

\(f(x)=c+\frac{d-c}{b-a}(x-a)\)

---

## Example

### Input

```python
import numpy as np

x = np.array([0, 5, 10])
c, d = 2, 4

print(convert_range(x, c, d))
```

### Output

```text
[2. 3. 4.]
```

### Reasoning

The input minimum is:

\(a=0\)

The input maximum is:

\(b=10\)

The target range is:

\([c,d]=[2,4]\)

The transformation is:

\(f(x)=2+\frac{4-2}{10-0}(x-0)\)

For `x = 0`:

\(f(0)=2\)

For `x = 5`:

\(f(5)=3\)

For `x = 10`:

\(f(10)=4\)

Therefore:

```text
[2. 3. 4.]
```

---

# Learn: Shifting and Scaling a Range

## What is Range Scaling?

Range scaling is a preprocessing technique that transforms numerical values from one interval into another.

Suppose the original values lie in:

\([a,b]\)

and we want them to lie in:

\([c,d]\)

The transformation shifts and scales the original values so that the endpoints correspond exactly:

\(a \rightarrow c\)

and:

\(b \rightarrow d\)

This is commonly called min-max scaling when the source interval is determined using the minimum and maximum values of a dataset.

---

## Mathematical Definition

Let:

\(a=\min(x)\)

and:

\(b=\max(x)\)

The target range is:

\([c,d]\)

The required transformation is:

\(f(x)=c+\frac{d-c}{b-a}(x-a)\)

Each component has a specific purpose.

The term:

\(x-a\)

shifts the minimum from `a` to zero.

The term:

\(\frac{x-a}{b-a}\)

normalizes the value into the unit interval.

The term:

\((d-c)\frac{x-a}{b-a}\)

scales the unit interval to the target range length.

Finally:

\(c+(d-c)\frac{x-a}{b-a}\)

shifts the result so that the lower endpoint becomes `c`.

---

## Deriving the Formula

The transformation can be understood as three operations.

### Step 1: Shift

Start with:

\(x-a\)

The original interval:

\([a,b]\)

becomes:

\([0,b-a]\)

The minimum is now zero.

---

### Step 2: Normalize

Divide by the original range:

\(\frac{x-a}{b-a}\)

The interval becomes:

\([0,1]\)

This removes the original scale.

The quantity:

\(t=\frac{x-a}{b-a}\)

is sometimes called the normalized position of `x` inside the original interval.

For the endpoints:

\(x=a \Rightarrow t=0\)

and:

\(x=b \Rightarrow t=1\)

---

### Step 3: Scale and Shift

The target interval has length:

\(d-c\)

Therefore, scale the normalized value:

\(t(d-c)\)

This gives the interval:

\([0,d-c]\)

Then add `c`:

\(c+t(d-c)\)

Substituting the definition of `t`:

\(f(x)=c+\frac{d-c}{b-a}(x-a)\)

This is the complete range transformation.

---

## Verifying the Endpoints

A useful way to verify any range transformation is to substitute the original endpoints.

For the minimum:

\(f(a)=c+\frac{d-c}{b-a}(a-a)\)

Therefore:

\(f(a)=c\)

For the maximum:

\(f(b)=c+\frac{d-c}{b-a}(b-a)\)

Therefore:

\(f(b)=c+d-c\)

and:

\(f(b)=d\)

So the transformation maps:

\(a\rightarrow c\)

and:

\(b\rightarrow d\)

exactly.

---

## Properties of the Mapping

The transformation is affine:

\(f(x)=mx+k\)

where:

\(m=\frac{d-c}{b-a}\)

and:

\(k=c-ma\)

The slope `m` determines how the original interval is stretched or compressed.

If:

\(|m|>1\)

the values are expanded relative to their original scale.

If:

\(|m|<1\)

the values are compressed.

If:

\(m<0\)

the target interval is reversed.

For example, if:

```text
[c, d] = [1, 0]
```

then the original minimum maps to `1` and the original maximum maps to `0`.

Therefore, the formula works even when the target range is descending, provided `c != d`.

---

## Preserving Relative Position

Range scaling preserves the relative position of values inside the interval.

Suppose:

\(x=\frac{a+b}{2}\)

Then `x` is halfway between the original endpoints.

Its normalized position is:

\(\frac{x-a}{b-a}=\frac{1}{2}\)

Therefore, its transformed value is halfway between `c` and `d`:

\(f(x)=\frac{c+d}{2}\)

This demonstrates that the transformation changes scale and location without changing the relative ordering of values when `d > c`.

---

## Example with Negative Values

Consider:

```python
x = np.array([-10, 0, 10])
c, d = 0, 1
```

The original interval is:

\([-10,10]\)

The transformation becomes:

\(f(x)=0+\frac{1-0}{10-(-10)}(x+10)\)

Therefore:

```text
-10 → 0
  0 → 0.5
 10 → 1
```

The negative values do not require special treatment.

The minimum and maximum automatically determine the transformation.

---

## 1D and 2D Arrays

The function uses:

```python
np.min(values)
```

and:

```python
np.max(values)
```

without specifying an axis.

For a 1D array:

```python
x = np.array([1, 2, 3])
```

the minimum and maximum are computed over all elements.

For a 2D array:

```python
x = np.array([
    [1, 2],
    [3, 4]
])
```

the same operations compute the global minimum and maximum.

Therefore:

\(a=\min(x)\)

and:

\(b=\max(x)\)

refer to the entire array.

The arithmetic then operates element-wise through NumPy broadcasting.

The output has exactly the same shape as the input.

---

## Constant Arrays

A special case occurs when every element has the same value.

For example:

```python
x = np.array([5, 5, 5])
```

Then:

\(a=b=5\)

The standard formula contains:

\(\frac{1}{b-a}\)

which would result in division by zero.

There is no unique linear scaling in this situation because the original array contains no variation.

The implementation handles this case explicitly:

```python
if arr_max == arr_min:
    return np.full_like(values, c, dtype=float)
```

Every element is mapped to the lower target bound `c`.

Thus:

```text
[5, 5, 5] → [c, c, c]
```

This is a practical convention for constant input data.

---

## Data Type Considerations

The output should contain floating-point values.

This matters because NumPy arrays can contain integer values.

For example:

```python
x = np.array([0, 5, 10])
```

has an integer dtype.

The transformation involves division:

\(\frac{d-c}{b-a}\)

so the result should not be forced back into an integer representation.

The constant-array branch explicitly uses:

```python
dtype=float
```

The normal branch naturally produces floating-point output when the scaling factor is floating-point.

---

## Applications

Range scaling is commonly used in machine learning and data processing.

### Feature Engineering

Different numerical features may have very different scales.

For example:

```text
Age       → [18, 80]
Income    → [20000, 200000]
```

Scaling can bring features into a common numerical range.

---

### Image Processing

Pixel intensities can be transformed between ranges such as:

\([0,255]\rightarrow[0,1]\)

This is common when preparing image data for neural networks.

---

### Score Conversion

Scores from one grading system can be mapped to another scale.

For example:

\([0,100]\rightarrow[0,10]\)

The relative ordering is preserved.

---

### Numerical Optimization

Some optimization algorithms behave better when features have comparable scales.

Range scaling can reduce large differences in feature magnitudes.

However, scaling does not automatically guarantee that an optimization problem becomes well-conditioned.

---

## Min-Max Scaling

When the target interval is `[0,1]`, the formula becomes:

\(x'=\frac{x-a}{b-a}\)

This is the standard min-max normalization formula.

For a general target interval `[c,d]`, the formula is:

\(x'=c+(d-c)\frac{x-a}{b-a}\)

Therefore, standard min-max scaling is simply a special case of the more general range transformation.

---

## Important Considerations

- The minimum and maximum determine the source interval.
- The transformation is applied element-wise.
- The output shape is identical to the input shape.
- The transformation is linear up to a shift.
- Values between `a` and `b` map between `c` and `d`.
- A constant array requires special handling.
- Floating-point output avoids unintended integer truncation.
- Outliers can strongly affect the minimum and maximum.
- Scaling does not remove outliers; it changes their numerical scale.
- If new data falls outside `[a,b]`, applying the same transformation can produce values outside `[c,d]`.

---

> 💡 **Important Note**
>
> Min-max scaling is sensitive to outliers because the minimum and maximum determine the entire transformation.
>
> If one feature contains an extreme outlier, most of the remaining observations may be compressed into a small portion of the target interval.
>
> In such situations, alternatives such as standardization or robust scaling may be more appropriate depending on the model and data distribution.

---

# Solutions

## Custom NumPy Implementation

```python
import numpy as np

def convert_range(
    values: np.ndarray,
    c: float,
    d: float
) -> np.ndarray:
    """
    Shift and scale values from their original range
    [min, max] to the target [c, d] range.
    """
    arr_max = np.max(values)
    arr_min = np.min(values)

    if arr_max == arr_min:
        return np.full_like(values, c, dtype=float)

    return c + ((d - c) / (arr_max - arr_min)) * (values - arr_min)
```

---

## Equivalent Min-Max Scaling

The implementation can be written by first calculating the normalized values.

```python
import numpy as np

def convert_range(
    values: np.ndarray,
    c: float,
    d: float
) -> np.ndarray:
    arr_min = np.min(values)
    arr_max = np.max(values)

    if arr_min == arr_max:
        return np.full(values.shape, c, dtype=float)

    normalized = (values - arr_min) / (arr_max - arr_min)
    return c + normalized * (d - c)
```

The intermediate quantity:

```python
normalized = (values - arr_min) / (arr_max - arr_min)
```

maps the original values to:

\([0,1]\)

The final transformation:

```python
c + normalized * (d - c)
```

maps `[0,1]` to `[c,d]`.

Both implementations are mathematically equivalent.

---

# Code Explanation

## 1. Find the Global Maximum

```python
arr_max = np.max(values)
```

This calculates:

\(b=\max(x)\)

For a 2D array, the default behavior considers all elements rather than computing one maximum per row or column.

---

## 2. Find the Global Minimum

```python
arr_min = np.min(values)
```

This calculates:

\(a=\min(x)\)

Together:

```python
arr_min
arr_max
```

define the original interval.

---

## 3. Detect a Constant Array

```python
if arr_max == arr_min:
```

If both values are equal:

\(b-a=0\)

The standard transformation would divide by zero.

Therefore, the function handles this case before performing the normal calculation.

---

## 4. Handle the Constant Case

```python
return np.full_like(values, c, dtype=float)
```

This creates an array with the same shape as `values`.

Every element is assigned:

\(c\)

The explicit `dtype=float` ensures that the output is floating-point.

---

## 5. Calculate the Target-Range Scale

The normal transformation contains:

```python
(d - c) / (arr_max - arr_min)
```

This is the ratio:

\(\frac{\text{target range length}}{\text{source range length}}\)

The target range length is:

\(d-c\)

The source range length is:

\(b-a\)

Therefore:

\(\text{scale}=\frac{d-c}{b-a}\)

---

## 6. Shift the Input

The expression:

```python
values - arr_min
```

subtracts the original minimum.

Therefore:

\(x-a\)

maps the original minimum to zero.

For example:

```text
Original: [0, 5, 10]
Shifted:  [0, 5, 10]
```

If the original range were `[-5, 5]`:

```text
Original: [-5, 0, 5]
Shifted:  [ 0, 5, 10]
```

The source interval has now been shifted to start at zero.

---

## 7. Apply the Scale

The complete scaling operation is:

```python
((d - c) / (arr_max - arr_min)) * (values - arr_min)
```

This maps:

\([0,b-a]\rightarrow[0,d-c]\)

The relative position of every value is preserved.

---

## 8. Shift into the Target Range

Finally:

```python
c + ...
```

adds the target lower bound.

The complete transformation is:

\(f(x)=c+\frac{d-c}{b-a}(x-a)\)

Therefore:

\(a\rightarrow c\)

and:

\(b\rightarrow d\)

---

## 9. NumPy Handles the Array Element-Wise

Suppose:

```python
values = np.array([
    [0, 5],
    [10, 15]
])
```

The scalar values:

```python
arr_min = 0
arr_max = 15
```

are broadcast against the entire matrix.

The expression:

```python
values - arr_min
```

therefore operates on every element.

The output remains:

```text
(2, 2)
```

The same principle works for arbitrary NumPy array dimensions, even though the problem specifically requires 1D and 2D arrays.

---

## 10. Why the Output Shape Is Preserved

Every operation in the main expression is element-wise after the scalar minimum and maximum are computed.

The expression:

```python
values - arr_min
```

has the same shape as `values`.

Multiplication by the scalar scale factor also preserves shape.

Adding scalar `c` preserves shape again.

Therefore:

\(\operatorname{shape}(f(x))=\operatorname{shape}(x)\)

---

## 11. Example with a 2D Array

Consider:

```python
x = np.array([
    [0, 5],
    [10, 15]
])

convert_range(x, 2, 4)
```

The original range is:

\([0,15]\)

The transformation is:

\(f(x)=2+\frac{2}{15}x\)

Therefore:

```text
0  → 2.0000
5  → 2.6667
10 → 3.3333
15 → 4.0000
```

The output is:

```text
[[2.         2.66666667]
 [3.33333333 4.        ]]
```

The matrix shape remains:

```text
(2, 2)
```

---

## 12. Values Outside the Original Range

The formula itself does not prevent values outside `[a,b]`.

Suppose the transformation was fitted using:

\([0,10]\rightarrow[0,1]\)

but a new value `20` is passed through the same transformation.

Then:

\(f(20)=\frac{20}{10}=2\)

The result is outside the target range.

Therefore, the function performs an affine transformation; it does not automatically clip values into `[c,d]`.

This distinction is important in machine-learning preprocessing.

---

# Time & Space Complexity

Let `N` be the total number of elements in the input array.

Computing the minimum requires:

\(O(N)\)

Computing the maximum also requires:

\(O(N)\)

The element-wise transformation requires:

\(O(N)\)

Therefore, the total time complexity is:

\(O(N)\)

The output array contains `N` floating-point values.

Therefore, storing the result requires:

\(O(N)\)

additional space.

The scalar values `arr_min`, `arr_max`, and the scaling factor require:

\(O(1)\)

extra scalar space.

| Complexity             | Value    |
| ---------------------- | -------- |
| Time                   | **O(N)** |
| Output Space           | **O(N)** |
| Auxiliary Scalar Space | **O(1)** |

Where **N** is the total number of elements in the input array.
