# Outlier Detection and Removal Using IQR Method (Medium, Data Preprocessing)

## Table of Contents

* [Problem Statement](#problem-statement)
* [Example](#example)
* [Learn: Outlier Detection Using the IQR Method](#learn-outlier-detection-using-the-iqr-method)

  * [What is an Outlier?](#what-is-an-outlier)
  * [Quartiles](#quartiles)
  * [Interquartile Range](#interquartile-range)
  * [IQR Outlier Boundaries](#iqr-outlier-boundaries)
  * [Choosing the Multiplier](#choosing-the-multiplier)
  * [Worked Example](#worked-example)
  * [Why IQR is Robust](#why-iqr-is-robust)
  * [IQR vs Mean and Standard Deviation](#iqr-vs-mean-and-standard-deviation)
  * [Outlier Removal](#outlier-removal)
  * [Outlier Detection vs Outlier Removal](#outlier-detection-vs-outlier-removal)
  * [Limitations](#limitations)
  * [Applications](#applications)
* [Solutions](#solutions)

  * [Custom Implementation](#custom-implementation)
  * [NumPy Implementation](#numpy-implementation)
* [Code Explanation](#code-explanation)
* [Time & Space Complexity](#time--space-complexity)

## Problem Statement

### [Outlier Detection and Removal Using IQR Method](https://www.deep-ml.com/problems/355)

Implement a function that detects and removes outliers from a
one-dimensional numerical dataset using the **Interquartile Range
(IQR)** method.

The function receives:

* a list of numerical values
* an IQR multiplier $k$

It must return a dictionary containing:

* `cleaned_data`: the input values with detected outliers removed
* `outlier_indices`: indices of detected outliers in the original data
* `lower_bound`: lower threshold used for detection
* `upper_bound`: upper threshold used for detection

The bounds are defined using the first and third quartiles.

The first quartile is:

\(Q_1=P_{25}\)

The third quartile is:

\(Q_3=P_{75}\)

The interquartile range is:

\(IQR=Q_3-Q_1\)

The lower and upper detection thresholds are:

\(LowerBound=Q_1-k(IQR)\)

\(UpperBound=Q_3+k(IQR)\)

A value $x$ is classified as an outlier when:

\(x<LowerBound\quad\text{or}\quad x>UpperBound\)

Values exactly equal to either boundary are **not** outliers because
the problem uses strict inequalities.

The returned numerical values for `cleaned_data`, `lower_bound`, and
`upper_bound` must be rounded to four decimal places.

---

## Example

### Input

```python
data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 50.0]
k = 1.5
```

### Step 1: Compute $Q_1$

Using the specified percentile calculation:

\(Q_1=3.25\)

### Step 2: Compute $Q_3$

The third quartile is:

\(Q_3=7.75\)

### Step 3: Compute IQR

\(IQR=Q_3-Q_1=7.75-3.25=4.5\)

### Step 4: Compute Lower Bound

\(LowerBound=3.25-1.5(4.5)=-3.5\)

### Step 5: Compute Upper Bound

\(UpperBound=7.75+1.5(4.5)=14.5\)

### Step 6: Detect Outliers

Every value lies inside the interval except:

```text
50.0
```

because:

\(50>14.5\)

Therefore, index `9` is an outlier.

### Output

```python
{
    "cleaned_data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    "outlier_indices": [9],
    "lower_bound": -3.5,
    "upper_bound": 14.5
}
```

---

# Learn: Outlier Detection Using the IQR Method

## What is an Outlier?

An outlier is an observation that is unusually far from the central
pattern of a dataset.

For example:

```text
[10, 11, 12, 13, 14, 100]
```

The value `100` is substantially larger than the other observations.

An outlier can occur because of:

* measurement errors
* data-entry mistakes
* sensor failures
* unusual events
* rare but genuine observations
* natural variability in the population

Outliers are not automatically errors.

A genuine extreme observation can contain useful information.

Therefore, outlier detection should be separated conceptually from the
decision of whether an observation should actually be removed.

---

## Quartiles

Quartiles divide ordered observations into approximately four parts.

The three main quartiles are:

### First Quartile

$Q_1$ is the 25th percentile.

Approximately 25% of observations lie below it.

\(Q_1=P_{25}\)

### Second Quartile

$Q_2$ is the 50th percentile.

It is the median.

\(Q_2=P_{50}=Median\)

### Third Quartile

$Q_3$ is the 75th percentile.

Approximately 75% of observations lie below it.

\(Q_3=P_{75}\)

The IQR method only needs $Q_1$ and $Q_3$.

---

## Interquartile Range

The interquartile range measures the spread of the middle 50% of the
dataset.

It is defined as:

\(IQR=Q_3-Q_1\)

For example, if:

\(Q_1=10\)

and:

\(Q_3=30\)

then:

\(IQR=30-10=20\)

The interval:

\([Q_1,Q_3]\)

contains the central portion of the distribution.

Unlike the full range, the IQR ignores the extreme lower and upper
tails when measuring this central spread.

---

## IQR Outlier Boundaries

The standard IQR rule extends the interval beyond $Q_1$ and $Q_3$.

The lower boundary is:

\(L=Q_1-k(IQR)\)

The upper boundary is:

\(U=Q_3+k(IQR)\)

where $k$ controls how aggressively observations are classified as
outliers.

A value $x$ is an outlier when:

\(x<L\quad\text{or}\quad x>U\)

A value inside the interval:

\(L\leq x\leq U\)

is retained.

---

## Why Extend Beyond the Quartiles?

The middle 50% alone is not the definition of an outlier.

If we classified everything below $Q_1$ or above $Q_3$ as an outlier,
half of the observations would potentially be removed.

Instead, the IQR is used to estimate the typical spread of the central
data and the boundaries are extended beyond the quartiles.

For the standard choice $k=1.5$:

\(L=Q_1-1.5(IQR)\)

\(U=Q_3+1.5(IQR)\)

Only observations sufficiently far beyond the central range are
flagged.

---

## Choosing the Multiplier

The multiplier $k$ controls sensitivity.

### $k=1.5$

This is the conventional Tukey-style rule for identifying potential
outliers.

It provides a relatively sensitive threshold.

### $k=3.0$

A larger multiplier creates wider boundaries:

\(L=Q_1-3(IQR)\)

\(U=Q_3+3(IQR)\)

Fewer observations will be classified as outliers.

This is often used to identify more extreme observations.

### Effect of Increasing $k$

As $k$ increases:

```text
Lower bound moves downward
Upper bound moves upward
```

Therefore:

```text
larger k -> fewer detected outliers
smaller k -> more detected outliers
```

---

## Worked Example

Consider:

```text
[1, 2, 3, 4, 5, 6, 7, 8, 9, 50]
```

with:

```text
k = 1.5
```

Using NumPy's percentile calculation:

\(Q_1=3.25\)

\(Q_3=7.75\)

Therefore:

\(IQR=7.75-3.25=4.5\)

The lower bound is:

\(L=3.25-1.5(4.5)=-3.5\)

The upper bound is:

\(U=7.75+1.5(4.5)=14.5\)

Every value from `1` through `9` satisfies:

\(-3.5\leq x\leq14.5\)

The value `50` satisfies:

\(50>14.5\)

Therefore:

```text
index 9 -> outlier
```

---

## Understanding the Bounds

The bounds are not necessarily actual observations.

For the example:

```text
data = [1,2,3,4,5,6,7,8,9,50]
```

the bounds are:

```text
lower = -3.5
upper = 14.5
```

Neither `-3.5` nor `14.5` exists in the dataset.

They are statistical thresholds derived from the quartiles.

This distinction is important.

---

## Outlier Detection vs Outlier Removal

The IQR rule detects observations.

It does not automatically tell us whether those observations should be
deleted.

For example:

```text
[20, 21, 22, 23, 24, 100]
```

If `100` is detected as an outlier, it could represent:

```text
A data-entry error
```

or:

```text
A legitimate rare observation
```

Removing it without understanding the data-generating process can
discard useful information.

Therefore, a robust preprocessing workflow often looks like:

```text
Detect
  |
  v
Investigate
  |
  v
Decide
  |
  v
Remove / transform / retain
```

The problem specifically asks us to remove detected outliers, but real
machine learning pipelines should not blindly apply this step.

---

## Why IQR is Robust

The IQR depends on quartiles rather than the arithmetic mean and
standard deviation.

Suppose:

```text
[10, 11, 12, 13, 14, 1000]
```

The extreme value `1000` can significantly increase the mean.

It also greatly increases the standard deviation.

The quartiles are much less affected because they are determined by the
ordered positions of observations.

This makes IQR-based detection relatively robust to extreme values.

---

## IQR vs Mean and Standard Deviation

Another common outlier detection technique uses the z-score.

The z-score is:

\(z=\frac{x-\mu}{\sigma}\)

where:

* $\mu$ is the mean
* $\sigma$ is the standard deviation

A common rule is to flag observations whose absolute z-score exceeds a
chosen threshold.

The IQR method instead uses:

\(Q_1,\ Q_3,\ IQR\)

This difference matters because mean and standard deviation are
themselves sensitive to extreme observations.

---

## Example of Mean Sensitivity

Consider:

```text
[10, 11, 12, 13, 14, 1000]
```

The value `1000` substantially changes:

\(\mu\)

and:

\(\sigma\)

Therefore, a method based on mean and standard deviation can have its
own detection threshold distorted by the extreme observation.

The quartiles are more resistant to this effect.

---

## IQR Does Not Require Normality

The IQR method does not require the feature to follow a Gaussian
distribution.

This makes it useful for skewed or non-normal numerical data.

However, this does not mean it is equally appropriate for every
distribution.

For heavily skewed data, the symmetric lower and upper extension:

\(Q_1-k(IQR)\)

and:

\(Q_3+k(IQR)\)

may classify observations differently from what a domain-specific
definition of unusual behavior would suggest.

---

## Important Limitation: Small Datasets

Quartiles estimated from very small datasets can be unstable.

For example:

```text
[1, 2, 3, 100]
```

contains very few observations.

The percentile calculation depends strongly on the chosen interpolation
convention and the positions of the observations.

Therefore, IQR-based decisions should be interpreted carefully for small
datasets.

---

## Percentile Definitions Matter

There is more than one convention for computing sample quantiles.

Different statistical libraries can use different interpolation or
quantile methods.

For this problem, the implementation uses:

```python
np.percentile(data, 25)
np.percentile(data, 75)
```

Therefore, the exact output should follow NumPy's percentile convention.

This matters especially for small datasets.

For the supplied example, NumPy produces:

\(Q_1=3.25\)

and:

\(Q_3=7.75\)

which leads to the expected boundaries.

---

## Symmetry Assumption

The standard IQR rule uses the same multiplier on both sides:

\(Q_1-k(IQR)\)

and:

\(Q_3+k(IQR)\)

This implicitly treats the distance from the quartiles symmetrically.

For strongly skewed distributions, the lower and upper tails may behave
very differently.

In such cases, blindly applying the standard rule may not represent the
domain's definition of an unusual observation.

---

## Univariate Detection

The problem operates on one numerical feature:

```text
x_1, x_2, ..., x_n
```

Each observation is evaluated independently.

This is called **univariate outlier detection**.

The method only considers the distribution of the selected variable.

It does not consider relationships between multiple features.

---

## Multivariate Outliers

Consider two features:

```text
Age
Income
```

A point may look normal when each feature is considered independently
but unusual when the two features are considered together.

For example:

```text
Age = 20
Income = extremely high
```

Neither feature necessarily needs to be an IQR outlier individually.

Their combination can still be unusual.

Multivariate methods can detect such patterns.

Examples include:

* Mahalanobis distance
* Isolation Forest
* Local Outlier Factor
* One-Class SVM
* robust covariance methods

The standard IQR method does not capture these relationships.

---

## Removing Outliers

Once the bounds are calculated, filtering is straightforward.

Keep:

\(L\leq x\leq U\)

Remove:

\(x<L\quad\text{or}\quad x>U\)

For:

```text
data = [1, 2, 3, 50]
```

if:

```text
upper_bound = 20
```

then:

```text
1 -> keep
2 -> keep
3 -> keep
50 -> remove
```

The cleaned dataset preserves the original order of retained values.

---

## Preserving Original Indices

The problem requires:

```text
outlier_indices
```

to refer to indices in the original dataset.

Therefore, when iterating through the data:

```python
for index, value in enumerate(data):
```

the index should be recorded before removing anything.

For example:

```text
Original:
index:  0  1  2  3
value: [1, 2, 3, 50]
```

If `50` is an outlier:

```text
outlier_indices = [3]
```

The index should not be recalculated after filtering.

---

## Rounding Requirements

The problem requires:

* cleaned values rounded to 4 decimal places
* lower bound rounded to 4 decimal places
* upper bound rounded to 4 decimal places

Python provides:

```python
round(value, 4)
```

For a NumPy scalar:

```python
round(np_value.item(), 4)
```

can be used.

The rounding requirement is primarily about the returned representation.

The actual detection should preferably be performed using the
unrounded bounds.

---

## Detection Before Rounding

Suppose the exact upper bound is:

```text
14.50006
```

Rounding first gives:

```text
14.5001
```

These are not mathematically identical.

Therefore, the safer sequence is:

```text
Compute exact bounds
       |
       v
Detect outliers using exact bounds
       |
       v
Round returned bounds
```

This prevents rounding from affecting classification.

---

## Negative Values

The IQR method works with negative values without modification.

For example:

```text
[-10, -8, -7, -6, -5, 20]
```

The quartiles and bounds can be negative.

The method does not assume that values are positive.

---

## Constant Data

Consider:

```text
[5, 5, 5, 5, 5]
```

Then:

\(Q_1=5\)

and:

\(Q_3=5\)

Therefore:

\(IQR=0\)

The bounds become:

\(L=5\)

\(U=5\)

Every value satisfies:

\(5\leq x\leq5\)

so no value is classified as an outlier.

This is a valid consequence of the IQR definition.

---

## Duplicate Values

Repeated values are handled naturally.

For example:

```text
[1, 1, 1, 2, 2, 3, 3, 3, 100]
```

The quartiles are computed from the complete dataset, including
duplicates.

The algorithm does not need to remove duplicates before calculating the
IQR.

Duplicates represent repeated observations and therefore contribute to
the empirical distribution.

---

## Empty Data

An empty dataset has no meaningful quartiles.

Calling percentile operations on an empty array is therefore invalid for
the purpose of this problem.

A production implementation can explicitly validate:

```python
if len(data) == 0:
    ...
```

The exact behavior for empty input depends on the required API.

The supplied problem does not specify an expected output for this case.

---

## Negative Multiplier

The IQR multiplier is normally non-negative.

For:

\(k<0\)

the interpretation of lower and upper bounds breaks down because the
interval can become inverted.

Therefore, a production implementation should generally validate:

\(k\geq0\)

The problem uses positive values such as:

```text
k = 1.5
k = 3.0
```

---

## Applications

IQR-based outlier detection can be useful in preprocessing:

* tabular machine learning
* exploratory data analysis
* quality control
* financial data cleaning
* sensor data analysis
* experimental measurements
* anomaly screening
* feature preprocessing

It is especially useful when a quick, interpretable univariate rule is
needed.

---

> 💡 **Important Note**
>
> Outlier does not mean incorrect. The IQR rule identifies observations
> that are statistically unusual relative to the feature's central
> distribution. Whether to delete, cap, transform, or retain them is a
> separate modeling decision.

---

# Solutions

The core algorithm is:

```text
1. Compute Q1.
2. Compute Q3.
3. Compute IQR.
4. Compute lower and upper bounds.
5. Scan the original data.
6. Record indices outside the bounds.
7. Keep values inside the bounds.
8. Round the returned numerical values.
```

The supplied solution already follows this algorithm correctly.

A small improvement is to explicitly round the returned values as
required by the problem.

---

## Custom Implementation

```python
import numpy as np

def detect_outliers_iqr(data: list[float], k: float = 1.5) -> dict:
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr

    cleaned_data = []
    outlier_indices = []

    for index, value in enumerate(data):
        if value < lower_bound or value > upper_bound:
            outlier_indices.append(index)
        else:
            cleaned_data.append(round(value, 4))

    return {
        "cleaned_data": cleaned_data,
        "outlier_indices": outlier_indices,
        "lower_bound": round(float(lower_bound), 4),
        "upper_bound": round(float(upper_bound), 4)
    }
```

---

## NumPy Implementation

The filtering step can also be vectorized with NumPy.

```python
import numpy as np

def detect_outliers_iqr_numpy(data: list[float], k: float = 1.5) -> dict:
    values = np.asarray(data, dtype=float)

    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1

    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr

    outlier_mask = (values < lower_bound) | (values > upper_bound)
    outlier_indices = np.flatnonzero(outlier_mask).tolist()
    cleaned_data = np.round(values[~outlier_mask], 4).tolist()

    return {
        "cleaned_data": cleaned_data,
        "outlier_indices": outlier_indices,
        "lower_bound": round(float(lower_bound), 4),
        "upper_bound": round(float(upper_bound), 4)
    }
```

The vectorized version expresses the filtering operation directly as a
Boolean mask.

---

# Code Explanation

## Step 1: Compute the First Quartile

```python
q1 = np.percentile(data, 25)
```

This calculates the 25th percentile.

Mathematically:

\(Q_1=P_{25}\)

The value represents the lower quartile of the empirical distribution.

---

## Step 2: Compute the Third Quartile

```python
q3 = np.percentile(data, 75)
```

This calculates:

\(Q_3=P_{75}\)

Together, $Q_1$ and $Q_3$ describe the central portion of the data.

---

## Step 3: Compute IQR

```python
iqr = q3 - q1
```

The interquartile range is:

\(IQR=Q_3-Q_1\)

It measures the spread between the first and third quartiles.

---

## Step 4: Compute the Lower Boundary

```python
lower_bound = q1 - k * iqr
```

This implements:

\(L=Q_1-k(IQR)\)

For the default:

\(k=1.5\)

the boundary becomes:

\(L=Q_1-1.5(IQR)\)

---

## Step 5: Compute the Upper Boundary

```python
upper_bound = q3 + k * iqr
```

This implements:

\(U=Q_3+k(IQR)\)

The two boundaries define the region in which observations are
considered non-outlying according to this rule.

---

## Step 6: Iterate Through Original Data

```python
for index, value in enumerate(data):
```

`enumerate` provides both:

```text
index
value
```

This is necessary because the problem requires the indices of outliers
from the original dataset.

---

## Step 7: Detect an Outlier

```python
if value < lower_bound or value > upper_bound:
```

This directly implements:

\(x<L\quad\text{or}\quad x>U\)

If either condition is true, the observation is an outlier.

Notice that equality is not included.

Therefore:

```text
value == lower_bound
```

is retained.

Likewise:

```text
value == upper_bound
```

is retained.

---

## Step 8: Store the Outlier Index

```python
outlier_indices.append(index)
```

The original index is preserved.

For:

```text
[1, 2, 3, 50]
```

if `50` is the outlier:

```text
outlier_indices = [3]
```

---

## Step 9: Keep Non-Outliers

```python
cleaned_data.append(round(value, 4))
```

Values inside the interval are retained.

The value is rounded to satisfy the output specification.

The original ordering is preserved.

---

## Step 10: Return the Result

The final dictionary contains four pieces of information:

```python
{
    "cleaned_data": ...,
    "outlier_indices": ...,
    "lower_bound": ...,
    "upper_bound": ...
}
```

This makes the result useful for both downstream preprocessing and
inspection of which observations were removed.

---

# Analysis of the Supplied Implementation

The user's supplied implementation is algorithmically correct for the
specified task.

The important calculations:

```python
q1 = np.percentile(data, 25)
q3 = np.percentile(data, 75)
iqr = q3 - q1
```

correctly implement the IQR method.

The detection condition:

```python
if val < lower_bound or upper_bound < val:
```

is equivalent to:

```python
if val < lower_bound or val > upper_bound:
```

Both correctly implement strict boundary comparisons.

The original indices are preserved using:

```python
enumerate(data)
```

which is exactly what the problem requires.

---

## Improvement: Rounding

The problem explicitly requires `cleaned_data` to be rounded to four
decimal places.

The supplied implementation currently does:

```python
cleaned_data.append(val)
```

rather than:

```python
cleaned_data.append(round(val, 4))
```

Therefore, the implementation should be adjusted to satisfy the exact
output contract.

---

## Improvement: Dictionary Construction

The original code uses:

```python
result = dict({
    ...
})
```

This is unnecessary.

A dictionary literal is simpler:

```python
return {
    "cleaned_data": cleaned_data,
    "outlier_indices": outlier_indices,
    "lower_bound": ...,
    "upper_bound": ...
}
```

Both approaches are functionally equivalent.

---

## Improvement: Explicit Float Conversion

The original code uses:

```python
lower_bound.item()
```

and:

```python
upper_bound.item()
```

This works because the values returned by the NumPy operations are
typically NumPy scalar types.

A more general implementation can use:

```python
float(lower_bound)
```

This makes the intention explicit and does not depend on `.item()`.

---

## Important: Do Not Round Before Detection

A tempting implementation would be:

```python
lower_bound = round(lower_bound, 4)
upper_bound = round(upper_bound, 4)
```

and then use those rounded values for classification.

That can slightly change the decision near a boundary.

The better sequence is:

```text
Calculate exact bounds
        |
        v
Detect outliers
        |
        v
Round returned bounds
```

The classification should use the unrounded mathematical thresholds.

---

# Vectorized Detection

The NumPy implementation creates a Boolean mask:

```python
outlier_mask = (values < lower_bound) | (values > upper_bound)
```

For:

```text
values = [1, 2, 3, 50]
```

and:

```text
upper_bound = 14.5
```

the mask becomes conceptually:

```text
[False, False, False, True]
```

The mask can then be used directly:

```python
values[~outlier_mask]
```

to select non-outliers.

Likewise:

```python
np.flatnonzero(outlier_mask)
```

returns the original indices of outliers.

---

# Comparison of Implementations

| Approach         | Main Idea                    | Advantage              |
| ---------------- | ---------------------------- | ---------------------- |
| Loop             | Inspect each value           | Simple and explicit    |
| Boolean masking  | Build an outlier mask        | Compact and vectorized |
| NumPy percentile | Library quartile calculation | Reliable and efficient |

The loop-based version is especially clear for learning the algorithm.

The vectorized version is generally more natural when already working
with NumPy arrays.

---

# Mathematical Summary

Given observations:

\(X=\{x_1,x_2,\ldots,x_n\}\)

compute:

\(Q_1=P_{25}(X)\)

and:

\(Q_3=P_{75}(X)\)

Then:

\(IQR=Q_3-Q_1\)

Calculate:

\(L=Q_1-k(IQR)\)

and:

\(U=Q_3+k(IQR)\)

For each observation $x_i$:

\(x_i\text{ is an outlier}\iff x_i<L\lor x_i>U\)

Otherwise:

\(L\leq x_i\leq U\)

and the observation is retained.

The cleaned dataset is therefore:

\(X_{clean}=\{x_i\in X:L\leq x_i\leq U\}\)

while the outlier index set is:

\(I_{outlier}=\{i:x_i<L\lor x_i>U\}\)

---

# Practical ML Perspective

IQR-based filtering is useful as a preprocessing technique, but it
should not automatically be applied to every feature.

Consider a feature representing annual income.

A very high income may be statistically unusual but completely valid.

Removing it could make the training data less representative.

Therefore, the practical question is not simply:

```text
"Is this point an outlier?"
```

but:

```text
"Is this observation an error or an observation that should be
excluded for this particular modeling objective?"
```

The IQR method provides a statistical screening rule.

Domain knowledge determines what should happen afterward.

---

# IQR and Feature Scaling

Outlier removal and feature scaling are separate operations.

For example:

```text
Raw feature
    |
    v
Outlier analysis
    |
    v
Potential removal / transformation
    |
    v
Standardization or normalization
```

IQR detection itself is based on the feature's ordered values and does
not require standardization first.

For positive linear scaling:

\(x'=ax+b\)

the ordering of observations is preserved when $a>0$, and quartile-based
boundaries transform correspondingly.

However, nonlinear transformations can change the distribution and
therefore the resulting IQR-based thresholds.

---

# Alternative to Removing Outliers: Capping

Instead of deleting an outlier, another strategy is **winsorization**,
where extreme values are capped at selected thresholds.

For example:

\(x'=\min(\max(x,L),U)\)

This transforms:

```text
x < L
```

into:

```text
L
```

and:

```text
x > U
```

into:

```text
U
```

This preserves the number of observations while limiting the influence
of extreme values.

It is conceptually different from the problem, which explicitly asks
for removal.

---

# Alternative: Log Transformation

For heavily right-skewed positive data, a logarithmic transformation
can reduce the influence of very large values.

For example:

\(x'=\log(1+x)\)

This does not explicitly remove observations.

Instead, it changes the scale of the feature.

This can sometimes be preferable when large values are genuine rather
than erroneous.

---

# IQR and Robust Statistics

The IQR is a **robust statistic** because it depends on quartiles rather
than directly on extreme observations.

Other robust statistics include:

* median
* median absolute deviation
* trimmed mean
* robust covariance estimates

Robust methods are useful when the dataset contains contamination or
heavy tails.

---

> 💡 **Interview Tip**
>
> Remember the distinction:
>
> \(IQR=Q_3-Q_1\)
>
> but the outlier boundaries are:
>
> \(Q_1-1.5(IQR)\)
>
> and:
>
> \(Q_3+1.5(IQR)\)
>
> Do not confuse the quartile interval $[Q_1,Q_3]$ with the full IQR
> outlier-detection interval.

---

# Time & Space Complexity

Let:

* $n$ = number of observations

Computing the two percentiles is the dominant statistical operation.

A typical percentile implementation requires sorting or an equivalent
selection procedure.

If sorting is used, the complexity is approximately:

\(O(n\log n)\)

The subsequent scan through the data requires:

\(O(n)\)

Therefore, the total complexity is dominated by percentile computation:

\(O(n\log n)\)

The output contains the cleaned data and potentially the indices of all
outliers.

In the worst case, both can contain $O(n)$ elements.

Therefore:

\(O(n)\)

additional output space is required.

For the loop-based implementation, the complexity is:

| Complexity | Value          |
| ---------- | -------------- |
| Time       | **O(n log n)** |
| Space      | **O(n)**       |

Here, $n$ is the number of numerical observations.

If a percentile implementation uses a selection algorithm rather than
full sorting, percentile computation can potentially be reduced toward
linear expected time, but the practical complexity depends on the
library implementation.

The important algorithmic structure remains:

```text
Percentiles
    |
    v
IQR
    |
    v
Bounds
    |
    v
Single scan for filtering
```

The filtering itself is only:

\(O(n)\)

once the quartiles and bounds have been computed.