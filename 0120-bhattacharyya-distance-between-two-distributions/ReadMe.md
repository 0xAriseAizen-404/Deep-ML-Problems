# [Bhattacharyya Distance Between Two Distributions](https://www.deep-ml.com/problems/120) (Easy, Statistics)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Understanding Bhattacharyya Distance](#understanding-bhattacharyya-distance)
  - [Bhattacharyya Coefficient](#bhattacharyya-coefficient)
  - [Bhattacharyya Distance](#bhattacharyya-distance)
  - [Interpretation](#interpretation)
  - [Key Properties](#key-properties)
  - [Bhattacharyya Distance vs KL Divergence](#bhattacharyya-distance-vs-kl-divergence)
- [Solution](#solution)
  - [Approach](#approach)
  - [NumPy Implementation](#numpy-implementation)
- [Code Explanation](#code-explanation)
- [Complexity Analysis](#complexity-analysis)
- [Key Takeaways](#key-takeaways)

## Problem Statement

Implement a function to calculate the Bhattacharyya distance between two discrete probability distributions.

Given two lists $p$ and $q$ representing probability distributions:

1. If the distributions have different lengths, return `0.0`.
2. If either distribution is empty, return `0.0`.
3. Calculate the Bhattacharyya coefficient.
4. Calculate the Bhattacharyya distance.
5. Return the result rounded to 4 decimal places.

The Bhattacharyya coefficient is:

$$BC(P,Q)=\sum_{i=1}^{n}\sqrt{P_iQ_i}$$

The Bhattacharyya distance is:

$$BD(P,Q)=-\ln(BC(P,Q))$$

## Example

### Input

```python
p = [0.1, 0.2, 0.3, 0.4]
q = [0.4, 0.3, 0.2, 0.1]
```

### Output

```text
0.1166
```

### Calculation

The Bhattacharyya coefficient is:

\(BC(P,Q)=\sqrt{0.1(0.4)}+\sqrt{0.2(0.3)}+\sqrt{0.3(0.2)}+\sqrt{0.4(0.1)}\)

\(BC(P,Q)\approx0.8899\)

Therefore:

\(BD(P,Q)=-\ln(0.8899)\approx0.1166\)

The coefficient measures the overlap between the two distributions, while the distance converts this overlap into a non-negative dissimilarity measure.

## Understanding Bhattacharyya Distance

Bhattacharyya Distance is a statistical measure used to quantify the similarity or overlap between two probability distributions.

For two discrete distributions $P$ and $Q$, corresponding probabilities are compared element by element.

Unlike a directional divergence such as KL divergence, Bhattacharyya distance is symmetric:

\(BD(P,Q)=BD(Q,P)\)

A smaller distance means the distributions have greater overlap, while a larger distance means they are more different.

### Bhattacharyya Coefficient

The first step is calculating the Bhattacharyya coefficient:

\(BC(P,Q)=\sum\_{i=1}^{n}\sqrt{P_iQ_i}\)

For each corresponding pair $(P_i,Q_i)$:

1. Multiply the probabilities.
2. Take the square root.
3. Add the result to the coefficient.

For example:

\(\sqrt{0.1\times0.4}=\sqrt{0.04}=0.2\)

The coefficient is therefore the sum of all such terms.

For valid probability distributions, the coefficient lies in:

\(0\leq BC(P,Q)\leq1\)

When the distributions are identical:

\(BC(P,Q)=1\)

### Bhattacharyya Distance

Once the coefficient has been calculated, the distance is:

\(BD(P,Q)=-\ln(BC(P,Q))\)

If the distributions are identical:

\(BC(P,Q)=1\)

and therefore:

\(BD(P,Q)=-\ln(1)=0\)

As the overlap decreases, the coefficient becomes smaller and the distance increases.

### Interpretation

The Bhattacharyya coefficient directly measures distributional overlap.

- $BC=1$ indicates identical distributions.
- $BC$ close to $1$ indicates strong overlap.
- $BC$ close to $0$ indicates very little overlap.
- Larger Bhattacharyya distance indicates greater dissimilarity.

The logarithm transforms the coefficient into a distance-like quantity where zero corresponds to identical distributions.

### Key Properties

#### 1. Non-negative

For valid probability distributions:

\(BD(P,Q)\geq0\)

Because:

\(BC(P,Q)\leq1\)

and therefore:

\(-\ln(BC(P,Q))\geq0\)

#### 2. Symmetric

The distance is symmetric because:

\(\sqrt{P_iQ_i}=\sqrt{Q_iP_i}\)

Therefore:

\(BD(P,Q)=BD(Q,P)\)

#### 3. Zero for Identical Distributions

If:

\(P=Q\)

then:

\(BC(P,Q)=1\)

and:

\(BD(P,Q)=0\)

#### 4. Requires Corresponding Probabilities

The two distributions must represent probabilities over the same discrete outcomes. Therefore, corresponding elements are compared element by element.

### Bhattacharyya Distance vs KL Divergence

Bhattacharyya Distance and KL Divergence both compare probability distributions, but they behave differently.

| Property                         | Bhattacharyya Distance      | KL Divergence           |
| -------------------------------- | --------------------------- | ----------------------- |
| Symmetric                        | Yes                         | No                      |
| Measures overlap                 | Yes                         | Indirectly              |
| Direction matters                | No                          | Yes                     |
| Zero for identical distributions | Yes                         | Yes                     |
| Formula                          | $-\ln(\sum_i\sqrt{P_iQ_i})$ | $\sum_iP_i\ln(P_i/Q_i)$ |
| Interpretation                   | Distributional similarity   | Information difference  |

KL divergence measures the information lost when using $Q$ to approximate $P$.

Bhattacharyya distance instead focuses on how much the two distributions overlap.

## Solution

### Approach

The algorithm is straightforward:

1. Check whether `p` and `q` have the same length.
2. Return `0.0` if either distribution is empty.
3. Convert both lists into NumPy arrays.
4. Multiply corresponding probabilities.
5. Take the square root of every product.
6. Sum the values to obtain the Bhattacharyya coefficient.
7. Apply `-log` to obtain the distance.
8. Round the result to 4 decimal places.

Mathematically:

\(BC=\sum_i\sqrt{p_iq_i}\)

Then:

\(BD=-\ln(BC)\)

### NumPy Implementation

```python
import numpy as np

def bhattacharyya_distance(p: list[float], q: list[float]) -> float:
    if len(p) != len(q):
        return 0.0
    if len(p) == 0 or len(q) == 0:
        return 0.0

    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)

    bhattacharyya_coefficient = np.sum(np.sqrt(p * q))
    return np.round(-np.log(bhattacharyya_coefficient), 4)
```

## Code Explanation

### Input Validation

```python
if len(p) != len(q):
    return 0.0
if len(p) == 0 or len(q) == 0:
    return 0.0
```

The problem explicitly requires `0.0` when the distributions have different lengths or are empty.

### Convert to NumPy Arrays

```python
p = np.array(p, dtype=float)
q = np.array(q, dtype=float)
```

NumPy allows the element-wise operations to be performed efficiently.

### Calculate the Bhattacharyya Coefficient

```python
bhattacharyya_coefficient = np.sum(np.sqrt(p * q))
```

The expression:

```python
p * q
```

computes:

\([P_1Q_1,P_2Q_2,\ldots,P_nQ_n]\)

Then:

```python
np.sqrt(p * q)
```

computes:

\([\sqrt{P_1Q_1},\sqrt{P_2Q_2},\ldots,\sqrt{P_nQ_n}]\)

Finally:

```python
np.sum(...)
```

computes:

\(BC=\sum_i\sqrt{P_iQ_i}\)

### Calculate the Distance

```python
-np.log(bhattacharyya_coefficient)
```

implements:

\(BD=-\ln(BC)\)

The negative sign ensures that a coefficient between $0$ and $1$ produces a non-negative distance.

### Rounding

```python
np.round(-np.log(bhattacharyya_coefficient), 4)
```

returns the distance rounded to four decimal places as required.

## Complexity Analysis

Let $n$ be the number of elements in each distribution.

### Time Complexity

The element-wise multiplication, square root, and summation each require:

\(O(n)\)

Therefore:

\(\boxed{O(n)}\)

### Space Complexity

The NumPy operations create arrays proportional to the number of elements:

\(\boxed{O(n)}\)

for auxiliary storage.

## Key Takeaways

- Bhattacharyya distance measures the overlap between two probability distributions.
- The Bhattacharyya coefficient is:

\(BC(P,Q)=\sum_i\sqrt{P_iQ_i}\)

- The Bhattacharyya distance is:

\(BD(P,Q)=-\ln(BC(P,Q))\)

- Identical distributions have distance $0$.
- The distance is symmetric.
- For valid probability distributions, the distance is non-negative.
- The coefficient lies between $0$ and $1$.
- Smaller distance means greater distributional overlap.
- The implementation uses vectorized NumPy operations instead of explicitly iterating through the distributions.
- The two distributions must have matching lengths because corresponding outcomes are compared element by element.
