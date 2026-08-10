# Min-Max Scaling of Feature Values (Easy, Data Preprocessing)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Min-Max Normalization](#learn-min-max-normalization)
- [Solution](#solution)
  - [Custom Implementation](#custom-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

---

## Problem Statement

### [Min-Max Scaling of Feature Values](https://www.deep-ml.com/problems/112)

Implement a function that performs **Min-Max Normalization** on a list of numerical values.

Min-Max normalization rescales values to a common range, typically $[0,1]$, using the minimum and maximum values of the input.

The function should return the normalized values rounded to **4 decimal places**.

---

## Example

### Input

```python
min_max([1, 2, 3, 4, 5])
```

### Output

```text
[0.0, 0.25, 0.5, 0.75, 1.0]
```

### Reasoning

The minimum value `1` is mapped to `0.0`, while the maximum value `5` is mapped to `1.0`.

The remaining values are scaled proportionally between these two boundaries.

For example, for `3`:

$$
\frac{3-1}{5-1}=0.5
$$

Therefore, `3` becomes `0.5`.

---

## Learn: Min-Max Normalization

### What is it?

**Min-Max Normalization**, also called **Min-Max Scaling**, is a feature scaling technique that transforms numerical values into a specified range, commonly $[0,1]$.

The transformation preserves the relative ordering of the original values while changing their scale.

For example,

```text
Original:  [10, 20, 30, 40, 50]
Scaled:    [0, 0.25, 0.5, 0.75, 1]
```

The smallest value always becomes `0` and the largest value becomes `1`.

---

### Mathematical Definition

For a value $X$, the standard Min-Max transformation to $[0,1]$ is

$$
X'=\frac{X-X_{min}}{X_{max}-X_{min}}
$$

where

- $X$ is the original value.
- $X'$ is the normalized value.
- $X_{min}$ is the minimum value in the feature.
- $X_{max}$ is the maximum value in the feature.

The resulting values satisfy

$$
0\leq X'\leq1
$$

The formula can also be generalized to any target range $[a,b]$:

$$
X'=a+\frac{(X-X_{min})(b-a)}{X_{max}-X_{min}}
$$

The standard $[0,1]$ case is obtained by setting $a=0$ and $b=1$.

---

### Example

Given

```text
X = [1, 2, 3, 4, 5]
```

we have

$$
X_{min}=1
$$

and

$$
X_{max}=5
$$

Therefore,

$$
X_{max}-X_{min}=4
$$

For each value:

$$
\frac{1-1}{4}=0
$$

$$
\frac{2-1}{4}=0.25
$$

$$
\frac{3-1}{4}=0.5
$$

$$
\frac{4-1}{4}=0.75
$$

$$
\frac{5-1}{4}=1
$$

Thus,

```text
[0.0, 0.25, 0.5, 0.75, 1.0]
```

---

### Why Normalize?

Different features can have completely different numerical scales.

For example:

```text
Age:       [18, 25, 40, 60]
Salary:    [20000, 50000, 100000, 200000]
```

If these features are used directly, the larger numerical scale of salary can dominate calculations in algorithms that depend on distances or gradients.

Min-Max scaling puts features into a comparable range.

It is particularly useful for:

- K-Nearest Neighbors
- Neural Networks
- Gradient Descent
- K-Means Clustering
- Support Vector Machines
- Distance-based algorithms

---

### Characteristics / Key Points

- Scales values to a fixed range.
- The standard range is $[0,1]$.
- Preserves the ordering of values.
- The minimum becomes `0`.
- The maximum becomes `1`.
- Intermediate values are scaled proportionally.
- Does not center the data around zero.
- Sensitive to outliers because minimum and maximum determine the scaling range.
- Requires the minimum and maximum values of the feature.
- Constant-valued features require special handling.

---

### Special Case: Constant Values

If all values are identical, then

$$
X_{max}=X_{min}
$$

and the denominator becomes zero:

$$
X_{max}-X_{min}=0
$$

The standard formula is therefore undefined.

For this problem, the expected behavior is to return an array of zeros.

For example,

```python
min_max([5, 5, 5, 5])
```

produces

```text
[0.0, 0.0, 0.0, 0.0]
```

This treats a constant feature as containing no relative variation.

---

### Min-Max Scaling vs Standardization

Min-Max scaling and standardization are different techniques.

Min-Max scaling transforms values according to their minimum and maximum:

$$
X'=\frac{X-X_{min}}{X_{max}-X_{min}}
$$

Standardization uses the mean and standard deviation:

$$
Z=\frac{X-\mu}{\sigma}
$$

Min-Max scaling usually produces values in $[0,1]$, while standardization produces values centered around zero and does not have a fixed upper or lower bound.

---

### Why is it used? / Applications

Min-Max normalization is commonly used during machine learning preprocessing.

Applications include:

- Preparing numerical features for neural networks.
- Scaling inputs for gradient-based optimization.
- Distance-based machine learning algorithms.
- Clustering.
- Feature engineering.
- Image and signal preprocessing.
- Any model where comparable feature scales are beneficial.

---

> 💡 **Important Note**
>
> Min-Max scaling is sensitive to outliers. If a feature contains an extreme value, the minimum or maximum can change significantly, compressing most other observations into a small portion of the $[0,1]$ range.

> 💡 **Practical Tip**
>
> When working with a train/test split, compute the minimum and maximum using the **training data only**. Then use those same values to transform the validation and test data. This prevents information from the test set from leaking into the training process.

---

## Solution

### Custom Implementation

```python
def min_max(x: list[float]) -> list[float]:
    min_val, max_val = min(x), max(x)
    if min_val == max_val:
        return [0.0] * len(x)
    return [round((val - min_val) / (max_val - min_val), 4) for val in x]
```

---

## Code Explanation

### Step 1: Find the Minimum and Maximum

```python
min_val, max_val = min(x), max(x)
```

The minimum and maximum values define the scaling range.

---

### Step 2: Handle Constant Input

```python
if min_val == max_val:
    return [0.0] * len(x)
```

If the minimum and maximum are equal, every value is identical and the normalizing denominator would be zero.

The implementation therefore returns zeros.

---

### Step 3: Apply the Formula

```python
(val - min_val) / (max_val - min_val)
```

For every value, subtract the minimum and divide by the total range.

This directly implements

$$
X'=\frac{X-X_{min}}{X_{max}-X_{min}}
$$

---

### Step 4: Round the Result

```python
round((val - min_val) / (max_val - min_val), 4)
```

Each normalized value is rounded to four decimal places as required.

---

### Step 5: Build the Output

The list comprehension applies the transformation independently to every element and produces the final scaled list.

---

## Time & Space Complexity

Let $n$ be the number of values in the input list.

Finding the minimum and maximum each requires a linear scan, and scaling the values requires another linear scan.

Therefore,

| Complexity | Value      |
| ---------- | ---------- |
| Time       | **$O(n)$** |
| Space      | **$O(n)$** |

The output list contains $n$ normalized values, so the space required for the returned result is $O(n)$.