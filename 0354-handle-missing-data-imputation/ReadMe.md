# Handle Missing Data with Imputation (Medium, Data Preprocessing)

## Table of Contents

* [Problem Statement](#problem-statement)
* [Example](#example)
* [Learn: Missing Data Imputation](#learn-missing-data-imputation)

  * [What is Missing Data?](#what-is-missing-data)
  * [Types of Missing Data](#types-of-missing-data)
  * [Imputation](#imputation)
  * [Mean Imputation](#mean-imputation)
  * [Median Imputation](#median-imputation)
  * [Mode Imputation](#mode-imputation)
  * [Column-wise Processing](#column-wise-processing)
  * [All-Missing Columns](#all-missing-columns)
  * [Data Leakage](#data-leakage)
  * [Missing Indicators](#missing-indicators)
  * [Choosing a Strategy](#choosing-a-strategy)
  * [Important Implementation Details](#important-implementation-details)
  * [Applications](#applications)
* [Solutions](#solutions)

  * [Custom Implementation](#custom-implementation)
  * [NumPy Implementation](#numpy-implementation)
* [Code Explanation](#code-explanation)
* [Time & Space Complexity](#time--space-complexity)

## Problem Statement

### [Handle Missing Data with Imputation](https://www.deep-ml.com/problems/354)

Implement a function that handles missing values represented by `np.nan`
in a two-dimensional dataset.

The function must support three imputation strategies:

* `mean`
* `median`
* `mode`

Each missing value must be replaced using a statistic computed from
the non-missing values in the same column.

The imputation statistic must be computed independently for every column.

For a column containing values

```text
[2.0, NaN, 6.0]
```

mean imputation gives

```text
[2.0, 4.0, 6.0]
```

because

\(x_{missing}=\frac{2+6}{2}=4\)

Median imputation uses the middle observed value after sorting.

Mode imputation uses the most frequently occurring observed value.

If multiple values have the same maximum frequency, the smallest value
must be selected.

If an entire column contains missing values, those `NaN` values must
remain unchanged because there is no observed information from which
to compute an imputation statistic.

The function should return the imputed two-dimensional NumPy array.

## Example

### Input

```python
data = [[1.0, 2.0], [3.0, np.nan], [5.0, 6.0]]
strategy = "mean"
```

### Output

```text
[[1.0, 2.0],
 [3.0, 4.0],
 [5.0, 6.0]]
```

### Reasoning

Column `0` contains no missing values.

Therefore, it remains unchanged:

```text
[1.0, 3.0, 5.0]
```

Column `1` contains:

```text
[2.0, NaN, 6.0]
```

The observed values are:

```text
[2.0, 6.0]
```

Their mean is:

\(\bar{x}=\frac{2+6}{2}=4\)

Therefore, the missing value becomes `4.0`.

The final matrix is:

```text
[[1.0, 2.0],
 [3.0, 4.0],
 [5.0, 6.0]]
```

---

## Learn: Missing Data Imputation

### What is Missing Data?

Missing data occurs when an observation is unavailable for one or more
features in a dataset.

In NumPy, missing numerical values are commonly represented using:

```python
np.nan
```

For example:

```text
[[10.0, 20.0],
 [15.0, NaN],
 [12.0, 25.0]]
```

The second feature is missing for the second observation.

Many machine learning algorithms cannot directly operate on `NaN`
values.

Therefore, missing values often need to be handled before model training.

One common approach is **imputation**.

---

## Types of Missing Data

The mechanism responsible for missingness matters because it affects
whether simple imputation is appropriate.

There are three standard categories.

### MCAR

**Missing Completely At Random** means that the probability of a value
being missing is unrelated to both observed and unobserved data.

For example, suppose some measurements are lost because of a random
hardware failure.

The missingness does not depend on the value itself or another feature.

Formally, if $M$ represents the missingness indicator:

\(P(M\mid X_{observed},X_{missing})=P(M)\)

MCAR is the strongest missingness assumption.

---

### MAR

**Missing At Random** means that missingness depends on variables that
are observed, but not directly on the missing value after accounting
for the observed information.

For example, suppose income values are more frequently missing for
people in a particular observed age group.

The missingness can be related to age even though income itself is not
directly used to determine missingness.

Conceptually:

\(P(M\mid X_{observed},X_{missing})=P(M\mid X_{observed})\)

under the relevant modeling assumptions.

---

### MNAR

**Missing Not At Random** means that missingness depends on the
unobserved value itself or on information not adequately captured by
the observed variables.

For example, people with unusually high expenses may be less likely to
report their expenses.

The probability of missingness can therefore depend on the missing
value.

MNAR is more difficult to handle because the missingness mechanism
itself must often be modeled or investigated.

---

## Imputation

Imputation replaces missing observations with estimated values.

Instead of leaving:

```text
[10, NaN, 20]
```

the dataset becomes something such as:

```text
[10, 15, 20]
```

depending on the selected strategy.

The important principle in this problem is that imputation is performed
**column by column**.

For column $j$, only its observed values are used:

\(X_{observed,j}=\{x_{ij}\mid x_{ij}\neq NaN\}\)

The selected statistic is then calculated from this set.

Every missing value in that column receives the same statistic.

---

## Mean Imputation

Mean imputation replaces every missing value with the arithmetic mean
of the observed values in that column.

For observed values $x_1,x_2,\ldots,x_n$:

\(\bar{x}=\frac{1}{n}\sum_{i=1}^{n}x_i\)

A missing value is then replaced with:

\(x_{missing}=\bar{x}\)

### Example

Consider:

```text
[10, 20, NaN, 30, NaN]
```

The observed values are:

```text
[10, 20, 30]
```

Therefore:

\(\bar{x}=\frac{10+20+30}{3}=20\)

The imputed column becomes:

```text
[10, 20, 20, 30, 20]
```

### Advantages

Mean imputation is:

* simple
* fast
* easy to implement
* useful for approximately symmetric numerical data

It also preserves the sample mean of the observed values when replacing
missing observations.

### Disadvantages

Mean imputation can:

* reduce variance
* make the data distribution artificially concentrated
* weaken relationships between variables
* be strongly affected by outliers

For example, in:

```text
[10, 11, 12, 1000]
```

the mean is much larger than most observations.

Median imputation may be more appropriate in such a case.

---

## Median Imputation

Median imputation replaces missing values with the median of the
observed values.

First, sort the observed values.

For an odd number of observations, the median is the middle value.

For an even number of observations, the median is usually the average
of the two middle values.

### Odd Number of Observations

For:

```text
[3, 7, 9]
```

the median is:

\(median=7\)

### Even Number of Observations

For:

```text
[2, 4, 8, 10]
```

the median is:

\(median=\frac{4+8}{2}=6\)

Therefore:

```text
[2, 4, NaN, 8, 10]
```

becomes:

```text
[2, 4, 6, 8, 10]
```

### Advantages

Median imputation is:

* robust to outliers
* useful for skewed numerical distributions
* simple to compute
* less sensitive to extreme observations than the mean

### Disadvantages

Median imputation still:

* reduces variance
* ignores relationships with other features
* can create an artificial concentration around the median

---

## Mode Imputation

Mode imputation replaces missing values with the most frequently
occurring observed value.

For a value $v$, define its frequency as:

\(count(v)=|\{x_i:x_i=v\}|\)

The mode is:

\(mode=\arg\max_v count(v)\)

### Example

Consider:

```text
[2, 3, 3, 5, NaN]
```

The frequencies are:

```text
2 -> 1
3 -> 2
5 -> 1
```

Therefore:

```text
mode = 3
```

The result becomes:

```text
[2, 3, 3, 5, 3]
```

### Advantages

Mode imputation:

* works naturally for categorical data
* preserves the most common category
* does not require arithmetic operations
* can also be used for discrete numerical features

### Disadvantages

Mode imputation can:

* increase the frequency of the dominant value
* create artificial spikes
* distort class or category distributions
* ignore relationships between features

---

## Tie Handling for Mode

The problem specifies an important rule:

> If multiple values have the same maximum frequency, choose the
> smallest value.

Consider:

```text
[2, 2, 5, 5, NaN]
```

Both `2` and `5` occur twice.

Therefore:

```text
mode = 2
```

because:

\(2<5\)

NumPy's `np.unique` returns unique values in sorted order.

Therefore, if the counts are examined in that order and `np.argmax`
selects the first maximum, the smallest tied value is automatically
chosen.

For example:

```python
unique, counts = np.unique(values, return_counts=True)
mode = unique[np.argmax(counts)]
```

This satisfies the problem's tie-breaking rule.

---

## Column-wise Processing

The problem requires every column to be processed independently.

Suppose the dataset is:

```text
[[1, 100],
 [2, NaN],
 [3, 300]]
```

The missing value is in column `1`.

The statistic must be calculated from:

```text
[100, 300]
```

rather than from every value in the matrix.

For mean imputation:

\(mean_{column1}=\frac{100+300}{2}=200\)

The result becomes:

```text
[[1, 100],
 [2, 200],
 [3, 300]]
```

Different columns can have completely different distributions and
therefore different imputation values.

---

## Why Not Compute One Global Statistic?

A global mean mixes unrelated features.

Consider:

```text
Age       Salary
20        30000
30        NaN
40        70000
```

The numerical scales and meanings of the columns are different.

Computing one statistic across all values would have no meaningful
interpretation.

Instead:

```text
Age    -> column-specific statistic
Salary -> column-specific statistic
```

This is why column-wise processing is fundamental to the problem.

---

## All-Missing Columns

A special case occurs when an entire column contains `NaN`.

For example:

```text
[[1.0, NaN],
 [2.0, NaN],
 [3.0, NaN]]
```

The second column has no observed values.

Therefore, there is no valid mean:

\(mean(\emptyset)\)

There is also no median or mode.

The problem explicitly requires such columns to remain unchanged.

Therefore:

```text
[[1.0, NaN],
 [2.0, NaN],
 [3.0, NaN]]
```

must remain:

```text
[[1.0, NaN],
 [2.0, NaN],
 [3.0, NaN]]
```

This case must be checked before attempting median or mode computation.

---

## Data Leakage

Imputation statistics must be computed carefully when working with
machine learning datasets.

Suppose a dataset is split into:

```text
Training data
Validation data
Test data
```

The mean, median, or mode should be learned from the training set.

For example:

\(\mu_{train}=\frac{1}{n}\sum_{i=1}^{n}x_i\)

That statistic is then applied to missing values in validation and test
data.

The test set should not be used to calculate the training imputation
statistic.

### Why?

Using test data to calculate preprocessing statistics allows information
from the evaluation set to influence the training pipeline.

This is a form of **data leakage**.

A correct workflow is:

```text
Training data
      |
      v
Compute imputation statistics
      |
      v
Transform training data
      |
      +------> Transform validation data
      |
      +------> Transform test data
```

The same learned statistic is applied to all later datasets.

---

## Missing Indicators

Imputation removes the `NaN`, but the fact that the value was missing
may itself contain information.

For example, suppose a medical feature is missing more often for a
particular group.

Replacing the value with a mean hides the original missingness pattern.

A missing indicator can preserve this information.

For a feature $X$, define:

\(M_i=\begin{cases}1 & \text{if }X_i\text{ was missing}\\0 & \text{otherwise}\end{cases}\)

The dataset can then contain both:

```text
imputed_feature
missing_indicator
```

For example:

```text
Original:
[10, NaN, 20]

Imputed:
[10, 15, 20]

Indicator:
[0, 1, 0]
```

The model can therefore use both the estimated value and the fact that
the original value was missing.

---

## Choosing a Strategy

The correct strategy depends on the data.

### Mean

Often useful when:

* the feature is numerical
* the distribution is approximately symmetric
* outliers are limited
* a simple baseline is sufficient

### Median

Often useful when:

* the feature is numerical
* the distribution is skewed
* outliers are present
* robustness is important

### Mode

Often useful when:

* the feature is categorical
* the feature is discrete
* the most common category is a reasonable replacement

No single imputation strategy is universally optimal.

---

## Imputation and Distribution

Imputation changes the data distribution.

Suppose:

```text
[10, 12, 14, NaN, NaN]
```

The observed mean is:

\(\bar{x}=12\)

Mean imputation produces:

```text
[10, 12, 14, 12, 12]
```

The two new observations are exactly equal to the mean.

This increases the frequency of one value and generally reduces variance.

Therefore, imputation should not be thought of as recovering the
unknown true values.

It is an approximation used to create a dataset that downstream
algorithms can process.

---

## Imputation Does Not Recover Ground Truth

A critical distinction is:

```text
Missing value
       |
       v
Estimated replacement
```

The imputed value is not necessarily the actual value that was missing.

For example:

```text
Observed:
[10, 20, NaN, 40]
```

Mean imputation gives:

\(\frac{10+20+40}{3}=23.33\)

But the actual missing value could have been `25`, `100`, or `5`.

The imputed value is simply an estimate based on the selected strategy.

---

## More Advanced Imputation

Mean, median, and mode are simple univariate methods.

More advanced techniques can use relationships between features.

Examples include:

* K-nearest-neighbor imputation
* regression imputation
* iterative imputation
* multiple imputation
* model-based imputation

These methods can exploit correlations between variables.

For example, if height and weight are strongly related, a missing weight
value could potentially be estimated using height and other features.

However, more sophisticated methods also introduce additional modeling
assumptions and computational cost.

---

## Applications

Missing-data imputation appears in many machine learning workflows.

Common applications include:

* customer datasets
* financial records
* medical datasets
* sensor measurements
* survey responses
* transaction data
* scientific experiments
* recommendation systems
* time-series preprocessing

A typical preprocessing pipeline may look like:

```text
Raw Dataset
    |
    v
Detect Missing Values
    |
    v
Analyze Missingness
    |
    v
Fit Imputation Statistics on Training Data
    |
    v
Apply Imputation
    |
    v
Scale / Encode Features
    |
    v
Train Model
```

---

> 💡 **Important Note**
>
> The most important practical rule is not simply "replace NaN with the
> mean." The imputation statistic must be learned from the training
> data only. Otherwise, preprocessing can leak information from the
> validation or test set into the model.

---

## Solutions

The implementation can be viewed as two stages:

1. Determine the statistic for every column.
2. Replace the missing entries using that statistic.

The supplied solution computes the statistic separately for every
missing element.

That approach is correct for ordinary columns, but it recomputes the
same statistic repeatedly when a column contains multiple missing
values.

A more efficient implementation computes the statistic once per
column and then fills all missing positions in that column.

---

## Custom Implementation

```python
import numpy as np

def impute_missing_data(data: np.ndarray, strategy: str = "mean") -> np.ndarray:
    data = data.copy().astype(float)

    if strategy not in {"mean", "median", "mode"}:
        raise ValueError("strategy must be 'mean', 'median', or 'mode'")

    n_samples, n_cols = data.shape

    for col in range(n_cols):
        values = data[:, col]
        valid = values[~np.isnan(values)]

        if len(valid) == 0:
            continue

        if strategy == "mean":
            fill_value = np.mean(valid)
        elif strategy == "median":
            fill_value = np.median(valid)
        else:
            unique, counts = np.unique(valid, return_counts=True)
            fill_value = unique[np.argmax(counts)]

        data[np.isnan(data[:, col]), col] = fill_value

    return data
```

---

## NumPy Implementation

NumPy already provides efficient implementations for mean and median
operations while ignoring `NaN` values.

```python
import numpy as np

def impute_missing_data_numpy(data: np.ndarray, strategy: str = "mean") -> np.ndarray:
    data = data.copy().astype(float)

    if strategy not in {"mean", "median", "mode"}:
        raise ValueError("strategy must be 'mean', 'median', or 'mode'")

    for col in range(data.shape[1]):
        missing = np.isnan(data[:, col])

        if not missing.any():
            continue

        valid = data[~missing, col]

        if len(valid) == 0:
            continue

        if strategy == "mean":
            fill_value = np.mean(valid)
        elif strategy == "median":
            fill_value = np.median(valid)
        else:
            unique, counts = np.unique(valid, return_counts=True)
            fill_value = unique[np.argmax(counts)]

        data[missing, col] = fill_value

    return data
```

The two implementations use essentially the same algorithm.

The second version makes explicit use of NumPy operations for the
statistical calculations.

---

## Code Explanation

### Step 1: Copy the Input

```python
data = data.copy().astype(float)
```

The copy prevents the function from unexpectedly modifying the caller's
original array.

This is an improvement over directly modifying `data`.

The conversion to floating point is important because imputation may
produce non-integer values.

For example, the median of:

```text
[1, 2, 4, 5]
```

is:

\(median=\frac{2+4}{2}=3\)

If the original array were integer-only, floating-point imputation could
otherwise become problematic.

---

### Step 2: Validate the Strategy

```python
if strategy not in {"mean", "median", "mode"}:
    raise ValueError("strategy must be 'mean', 'median', or 'mode'")
```

Only three strategies are supported.

Accepting an unknown strategy would make the function's behavior
ambiguous.

The validation is performed before processing the matrix.

---

### Step 3: Iterate Over Columns

```python
for col in range(n_cols):
```

The problem explicitly requires column-wise processing.

For every column, the function independently determines which values
are observed and which values are missing.

---

### Step 4: Extract the Current Column

```python
values = data[:, col]
```

This selects every row from the current column.

For example:

```text
data =
[[1, 2],
 [3, NaN],
 [5, 6]]
```

For `col = 1`:

```text
values =
[2, NaN, 6]
```

---

### Step 5: Remove NaN Values

```python
valid = values[~np.isnan(values)]
```

`np.isnan(values)` produces a Boolean mask:

```text
[False, True, False]
```

Applying `~` gives:

```text
[True, False, True]
```

Therefore:

```text
valid =
[2, 6]
```

Only these values are used for calculating the imputation statistic.

This directly satisfies the requirement that missing values must not
participate in the statistic.

---

### Step 6: Handle an All-Missing Column

```python
if len(valid) == 0:
    continue
```

If no observed values exist, the column contains only `NaN`.

There is no valid mean, median, or mode.

The problem requires these values to remain unchanged.

Therefore, the function skips the column.

---

### Step 7: Mean Strategy

```python
fill_value = np.mean(valid)
```

For:

```text
valid = [2, 6]
```

we obtain:

\(mean=\frac{2+6}{2}=4\)

Every missing value in the column receives `4`.

---

### Step 8: Median Strategy

```python
fill_value = np.median(valid)
```

NumPy computes the median from the observed values.

For:

```text
valid = [2, 6]
```

the result is:

\(median=\frac{2+6}{2}=4\)

For:

```text
valid = [1, 3, 8]
```

the median is:

\(median=3\)

---

### Step 9: Mode Strategy

```python
unique, counts = np.unique(valid, return_counts=True)
```

Suppose:

```text
valid = [2, 5, 5, 7, 7, NaN]
```

After removing `NaN`:

```text
valid = [2, 5, 5, 7, 7]
```

`np.unique` produces sorted unique values:

```text
unique = [2, 5, 7]
```

and their counts:

```text
counts = [1, 2, 2]
```

Then:

```python
np.argmax(counts)
```

returns the index of the first maximum.

The first maximum is the count associated with `5`.

Therefore:

```text
mode = 5
```

This also implements the required smallest-value tie breaker.

---

### Step 10: Replace Missing Values

```python
data[np.isnan(data[:, col]), col] = fill_value
```

The mask identifies every missing value in the current column.

For:

```text
[2, NaN, 6, NaN]
```

the mask is:

```text
[False, True, False, True]
```

If:

```text
fill_value = 4
```

the result becomes:

```text
[2, 4, 6, 4]
```

All missing values in the column are therefore replaced in one
operation.

---

## Analysis of the Supplied Implementation

The original implementation follows the correct overall idea.

It:

* processes columns independently
* ignores `NaN` values
* supports mean, median, and mode
* uses the smallest value for mode ties through `np.unique`
* modifies the missing values in place

However, there are several implementation details worth improving.

---

### Recomputing the Mean

The original code processes every missing coordinate:

```python
for row, col in zip(*indices):
```

Then for every missing value it calculates:

```python
np.nanmean(data[:, col])
```

If a column contains $m$ missing values, the same mean is calculated
$m$ times.

For example:

```text
[NaN, NaN, NaN, 10, 20]
```

The mean:

\(mean=15\)

is calculated three times.

It is more efficient to calculate the statistic once per column.

---

### Median Empty-Column Problem

The original median branch does:

```python
nums = nums[~np.isnan(nums)]
nums = sorted(nums)
mid = len(nums) // 2
```

If the column contains only `NaN`, then:

```text
nums = []
```

and:

```python
nums[mid]
```

causes an indexing error.

The all-missing case must therefore be handled before calculating
the median.

---

### Mode Empty-Column Problem

The original mode branch uses:

```python
unique, counts = np.unique(nums, return_counts=True)
data[row, col] = unique[np.argmax(counts)]
```

If the column is entirely missing:

```text
nums = []
```

Then there are no unique values and no counts.

`np.argmax(counts)` therefore cannot select a valid mode.

Again, the all-missing case must be handled explicitly.

---

### In-Place Mutation

The original function modifies the input array directly.

For example:

```python
result = impute_missing_data(data)
```

also changes `data`.

This may be intentional, but preprocessing functions are often safer
when they return a transformed copy.

The improved implementation therefore uses:

```python
data = data.copy().astype(float)
```

---

### Strategy Validation

The original function validates the strategy inside the loop.

For example, the invalid strategy is only detected when the function
reaches a missing value.

If the dataset contains no missing values, an invalid strategy may never
be detected.

Validating before processing is cleaner:

```python
if strategy not in {"mean", "median", "mode"}:
    raise ValueError(...)
```

---

## Worked Example: Mean

Consider:

```python
data = np.array([
    [1.0, 2.0],
    [3.0, np.nan],
    [5.0, 6.0]
])
```

Column `0`:

```text
[1, 3, 5]
```

contains no missing values.

Therefore, it remains unchanged.

Column `1`:

```text
[2, NaN, 6]
```

has observed values:

```text
[2, 6]
```

The mean is:

\(\bar{x}=\frac{2+6}{2}=4\)

The final result is:

```text
[[1.0, 2.0],
 [3.0, 4.0],
 [5.0, 6.0]]
```

---

## Worked Example: Median

Consider:

```text
[[10, 100],
 [20, NaN],
 [30, 300],
 [40, NaN]]
```

For column `1`:

```text
[100, NaN, 300, NaN]
```

the observed values are:

```text
[100, 300]
```

The median is:

\(median=\frac{100+300}{2}=200\)

Therefore:

```text
[[10, 100],
 [20, 200],
 [30, 300],
 [40, 200]]
```

---

## Worked Example: Mode

Consider:

```text
[[1, 10],
 [2, 10],
 [3, NaN],
 [4, 20],
 [5, 10]]
```

The second column contains:

```text
[10, 10, NaN, 20, 10]
```

Observed values:

```text
[10, 10, 20, 10]
```

Frequencies:

```text
10 -> 3
20 -> 1
```

Therefore:

```text
mode = 10
```

The result is:

```text
[[1, 10],
 [2, 10],
 [3, 10],
 [4, 20],
 [5, 10]]
```

---

## Worked Example: Mode Tie

Consider:

```text
[[1, 2],
 [2, 5],
 [3, 2],
 [4, 5],
 [5, NaN]]
```

Column `1` has:

```text
[2, 5, 2, 5, NaN]
```

Observed frequencies:

```text
2 -> 2
5 -> 2
```

There is a tie.

The problem requires the smallest value.

Therefore:

```text
mode = 2
```

The final column becomes:

```text
[2, 5, 2, 5, 2]
```

---

## Worked Example: All-Missing Column

Consider:

```text
[[1, NaN],
 [2, NaN],
 [3, NaN]]
```

Column `1` contains no observed values.

Therefore:

```text
valid = []
```

There is no statistic that can be calculated.

The required result is:

```text
[[1, NaN],
 [2, NaN],
 [3, NaN]]
```

The function deliberately leaves this column unchanged.

---

## Mean vs Median vs Mode

| Strategy | Statistic           | Main Strength              | Main Weakness                 |
| -------- | ------------------- | -------------------------- | ----------------------------- |
| Mean     | Arithmetic average  | Simple and fast            | Sensitive to outliers         |
| Median   | Middle observation  | Robust to outliers         | Ignores feature relationships |
| Mode     | Most frequent value | Works for categorical data | Can create artificial spikes  |

The appropriate method depends on the feature's type and distribution.

---

## Important Implementation Details

### `NaN` Is Not an Ordinary Number

A key property of `NaN` is:

```python
np.nan == np.nan
```

is `False`.

Therefore, missing values should be detected with:

```python
np.isnan(x)
```

rather than:

```python
x == np.nan
```

This is important whenever implementing numerical missing-value
handling manually.

---

### `np.nanmean`

NumPy provides:

```python
np.nanmean(values)
```

which calculates the mean while ignoring `NaN`.

Similarly:

```python
np.nanmedian(values)
```

calculates the median while ignoring `NaN`.

However, both require care when all values are missing because there is
no valid statistic to compute.

---

### Why `np.unique` Works for Mode

`np.unique` returns sorted unique values.

For:

```text
[5, 2, 5, 2, 8]
```

it produces:

```text
unique = [2, 5, 8]
counts = [2, 2, 1]
```

`np.argmax(counts)` returns the first maximum.

Therefore, the mode becomes:

```text
2
```

which is exactly the smallest value among tied modes.

---

## Applications in Machine Learning

Imputation is usually part of a larger preprocessing pipeline.

For tabular data, a common workflow is:

```text
Raw Data
   |
   v
Missing Value Analysis
   |
   v
Train / Validation / Test Split
   |
   v
Fit Imputer on Training Data
   |
   v
Transform All Splits
   |
   v
Feature Scaling
   |
   v
Model Training
```

The order matters.

In particular, the imputation statistic should be learned after
splitting the data.

---

## Statistical Perspective

Mean, median, and mode are all forms of univariate imputation.

They estimate a missing value using only the observed values of the same
feature.

For feature $X_j$:

\(\hat{x}_{ij}=f(X_j^{observed})\)

where $f$ may be:

```text
mean
median
mode
```

This means the method does not directly use information from other
columns.

More advanced imputation methods can instead estimate:

\(\hat{x}_{ij}=f(X_{i,-j})\)

where $X_{i,-j}$ represents other features for the same observation.

This distinction explains why simple imputation is computationally easy
but may lose useful relationships between features.

---

> 💡 **Interview Tip**
>
> When implementing column-wise imputation, explicitly think about the
> all-`NaN` case. Mean and median functions may return `NaN`, while mode
> computation on an empty set can fail. A robust solution handles this
> case before calculating the statistic.

---

## Time & Space Complexity

Let:

* $m$ = number of rows
* $n$ = number of columns
* $N=mn$ = total number of elements

The improved implementation scans each column and computes its
statistics once.

### Mean

Computing the mean for one column takes:

\(O(m)\)

Across all $n$ columns:

\(O(mn)\)

### Median

A median generally requires sorting or an equivalent selection
operation.

Sorting one column costs:

\(O(m\log m)\)

Across $n$ columns:

\(O(nm\log m)\)

### Mode

`np.unique` requires processing and sorting unique values.

In the worst case, mode computation can be approximately:

\(O(m\log m)\)

per column.

Therefore, across all columns:

\(O(nm\log m)\)

in the general implementation used here.

### Space

The input contains:

\(O(mn)\)

elements.

The copied output array also requires:

\(O(mn)\)

space.

The temporary column arrays and masks require up to:

\(O(m)\)

additional auxiliary space for one column at a time.

The overall memory usage is therefore:

| Complexity      | Value           |
| --------------- | --------------- |
| Mean Time       | **O(mn)**       |
| Median Time     | **O(nm log m)** |
| Mode Time       | **O(nm log m)** |
| Output Space    | **O(mn)**       |
| Auxiliary Space | **O(m)**        |

If the input array is modified in place rather than copied, the output
copy cost can be avoided, although this changes the function's mutation
behavior.

The key optimization compared with the original implementation is that
each column's statistic is computed once rather than once for every
missing element.