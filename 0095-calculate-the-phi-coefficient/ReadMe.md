# Calculate the Phi Coefficient (Easy, Statistics)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: The Phi Coefficient](#learn-the-phi-coefficient)
- [Solution](#solution)
  - [Custom Implementation](#custom-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

---

## Problem Statement

### [Calculate the Phi Coefficient](https://www.deep-ml.com/problems/95)

Implement a function to calculate the **Phi coefficient**, a measure of correlation between two binary variables.

The function takes two lists containing only `0` and `1` values and returns the Phi coefficient rounded to **4 decimal places**.

The Phi coefficient is useful when both variables are binary, such as:

- Gender and presence of a disease.
- Treatment received and recovery.
- Prediction outcome and actual outcome.
- Presence or absence of a particular feature.

---

## Example

### Input

```python
phi_corr([1, 1, 0, 0], [0, 0, 1, 1])
```

### Output

```text
-1.0
```

### Reasoning

The two binary variables have a perfect negative relationship.

When the first variable is `1`, the second variable is always `0`, and when the first variable is `0`, the second variable is always `1`.

Therefore, the Phi coefficient is

$$
\phi=-1
$$

A value of `-1` represents perfect negative correlation.

---

## Learn: The Phi Coefficient

### What is it?

The **Phi coefficient** is a correlation coefficient used to measure the association between **two binary variables**.

A binary variable can take only two possible values, commonly represented as `0` and `1`.

For example, suppose:

- $X$ represents gender.
- $Y$ represents whether a person has heart disease.

If both $X$ and $Y$ are binary, the Phi coefficient can be used to measure how strongly they are associated.

The Phi coefficient is closely related to the **Pearson correlation coefficient**. In fact, for two binary variables, the Phi coefficient is equivalent to their Pearson correlation when the binary values are represented numerically as `0` and `1`.

---

### Contingency Table

The relationship between two binary variables can be represented using a $2\times2$ contingency table.

|       | $Y=0$ | $Y=1$ |
| ----- | ----: | ----: |
| $X=0$ |  $TN$ |  $FN$ |
| $X=1$ |  $FP$ |  $TP$ |

The four values represent the number of observations falling into each possible combination.

- $TN$ or $x_{00}$: Number of cases where $X=0$ and $Y=0$.
- $FN$ or $x_{01}$: Number of cases where $X=0$ and $Y=1$.
- $FP$ or $x_{10}$: Number of cases where $X=1$ and $Y=0$.
- $TP$ or $x_{11}$: Number of cases where $X=1$ and $Y=1$.

The names `TP`, `TN`, `FP`, and `FN` come from the confusion matrix interpretation, while $x_{ij}$ notation directly describes the values of the two variables.

---

### Mathematical Definition

Using the four contingency-table counts, the Phi coefficient is

$$
\phi=\frac{x_{11}x_{00}-x_{10}x_{01}}{\sqrt{(x_{11}+x_{10})(x_{11}+x_{01})(x_{00}+x_{10})(x_{00}+x_{01})}}
$$

Using the confusion-matrix notation, the same formula becomes

$$
\phi=\frac{TP\times TN-FP\times FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}
$$

The numerator measures the difference between the two diagonal products.

The denominator normalizes this value so that the resulting coefficient lies between `-1` and `1`.

---

### Interpretation

The Phi coefficient has the range

$$
-1\leq\phi\leq1
$$

Its value can be interpreted as follows:

| Phi Value | Interpretation               |
| --------- | ---------------------------- |
| $\phi=1$  | Perfect positive association |
| $\phi>0$  | Positive association         |
| $\phi=0$  | No linear association        |
| $\phi<0$  | Negative association         |
| $\phi=-1$ | Perfect negative association |

A value close to `1` means that the two variables tend to take the same binary values.

A value close to `-1` means that when one variable is `1`, the other tends to be `0`.

A value close to `0` indicates little or no linear association between the two binary variables.

---

### Example of Perfect Positive Correlation

Consider

```text
X = [1, 1, 0, 0]
Y = [1, 1, 0, 0]
```

The contingency counts are

$$
TP=2,\quad TN=2,\quad FP=0,\quad FN=0
$$

Therefore,

$$
\phi=\frac{(2)(2)-(0)(0)}{\sqrt{(2)(2)(2)(2)}}=1
$$

Thus, the variables have perfect positive association.

---

### Example of Perfect Negative Correlation

Consider

```text
X = [1, 1, 0, 0]
Y = [0, 0, 1, 1]
```

The counts are

$$
TP=0,\quad TN=0,\quad FP=2,\quad FN=2
$$

The numerator becomes

$$
(0)(0)-(2)(2)=-4
$$

The denominator becomes

$$
\sqrt{(2)(2)(2)(2)}=4
$$

Therefore,

$$
\phi=\frac{-4}{4}=-1
$$

This represents perfect negative association.

---

### Relationship with Pearson Correlation

For two binary variables encoded as `0` and `1`, the Phi coefficient is equivalent to the **Pearson correlation coefficient**.

Pearson correlation is generally defined as

$$
r=\frac{Cov(X,Y)}{\sigma_X\sigma_Y}
$$

When $X$ and $Y$ are binary, simplifying this expression produces the same result as the Phi coefficient formula.

Therefore, Phi can be viewed as a convenient way to calculate correlation specifically for binary variables.

---

### Characteristics / Key Points

- Phi is designed for two binary variables.
- Both variables should contain only two possible categories.
- The coefficient ranges from `-1` to `1`.
- Positive values indicate positive association.
- Negative values indicate negative association.
- A value of `0` indicates no linear association.
- A value of `1` represents perfect positive association.
- A value of `-1` represents perfect negative association.
- The calculation is based on a $2\times2$ contingency table.
- Phi is symmetric, so exchanging the two variables does not change the result.
- The denominator can become zero when one variable has no variation.

---

### Edge Case: Zero Denominator

The denominator is

$$
\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}
$$

If one of the variables contains only a single value, one of these terms can become zero.

For example,

```text
X = [1, 1, 1, 1]
Y = [0, 1, 0, 1]
```

Here, $X$ has no variation because every value is `1`.

In this situation, the Phi coefficient is mathematically undefined because the denominator is zero.

For this problem, the implementation handles this case by returning `0.0`.

---

### Why is it used? / Applications

The Phi coefficient is useful whenever the relationship between two binary variables needs to be measured.

Applications include:

- Medical statistics.
- Epidemiology.
- Binary feature analysis.
- Classification analysis.
- Survey data analysis.
- Feature selection.
- Association analysis.
- Evaluating relationships between binary events.

For example, in medical research, Phi can measure the association between a binary risk factor and a binary disease outcome.

---

> 💡 **Important Note**
>
> Do not interpret a high Phi coefficient as proof of causation. It measures statistical association between two binary variables, not whether one variable causes the other.

> 💡 **Interview Tip**
>
> Remember the structure of the formula: the numerator is the difference between the diagonal products, $TP\times TN-FP\times FN$, while the denominator normalizes this value. Also remember that a zero denominator means that at least one binary variable has no variation.

---

## Solution

### Custom Implementation

```python
from math import sqrt

def phi_corr(x: list[int], y: list[int]) -> float:
    TP = FP = FN = TN = 0
    for xi, yi in zip(x, y):
        if xi == 1 and yi == 1:
            TP += 1
        elif xi == 1 and yi == 0:
            FP += 1
        elif xi == 0 and yi == 1:
            FN += 1
        else:
            TN += 1
    numerator = (TP * TN) - (FP * FN)
    denominator = sqrt(
        (TP + FP) *
        (TP + FN) *
        (TN + FP) *
        (TN + FN)
    )
    if denominator == 0.0:
        return 0.0
    return round(numerator / denominator, 4)
```

---

## Code Explanation

### Step 1: Initialize the Four Counts

```python
TP = FP = FN = TN = 0
```

The algorithm maintains four counters corresponding to the four possible combinations of binary values.

For each pair $(x_i,y_i)$:

| $x_i$ | $y_i$ | Counter |
| ----- | ----- | ------- |
| `1`   | `1`   | `TP`    |
| `1`   | `0`   | `FP`    |
| `0`   | `1`   | `FN`    |
| `0`   | `0`   | `TN`    |

This converts the two input lists into the required $2\times2$ contingency table.

---

### Step 2: Count the Combinations

```python
for xi, yi in zip(x, y):
```

`zip()` processes corresponding elements from both lists simultaneously.

For example,

```text
x = [1, 1, 0, 0]
y = [0, 0, 1, 1]
```

produces the pairs

```text
(1, 0)
(1, 0)
(0, 1)
(0, 1)
```

The appropriate counter is incremented for every pair.

---

### Step 3: Calculate the Numerator

```python
numerator = (TP * TN) - (FP * FN)
```

The numerator is the difference between the products of the two diagonals of the contingency table.

Mathematically,

$$
N=TP\times TN-FP\times FN
$$

A positive numerator indicates that the matching combinations are stronger, while a negative numerator indicates that the mismatching combinations dominate.

---

### Step 4: Calculate the Denominator

```python
denominator = sqrt(
    (TP + FP) *
    (TP + FN) *
    (TN + FP) *
    (TN + FN)
)
```

The denominator normalizes the numerator.

It is

$$
D=\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}
$$

Without this normalization, the magnitude of the result would depend on the number of observations.

---

### Step 5: Handle the Undefined Case

```python
if denominator == 0.0:
    return 0.0
```

If the denominator is zero, the Phi coefficient is mathematically undefined.

The implementation returns `0.0` to provide a defined result for this edge case.

---

### Step 6: Calculate and Round the Coefficient

```python
return round(numerator / denominator, 4)
```

The final coefficient is obtained by dividing the numerator by the denominator.

$$
\phi=\frac{N}{D}
$$

The result is rounded to four decimal places as required by the problem.

---

## Time & Space Complexity

Let $n$ be the number of observations in each input list.

The algorithm makes a single pass through the input lists to calculate `TP`, `TN`, `FP`, and `FN`.

The remaining arithmetic operations use only a constant number of values.

| Complexity | Value      |
| ---------- | ---------- |
| Time       | **$O(n)$** |
| Space      | **$O(1)$** |

Where **$n$** is the number of paired observations.

The algorithm uses constant auxiliary space because it stores only four counters regardless of the input size.
