# Descriptive Statistics Calculator (Easy, Statistics)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Descriptive Statistics](#learn-descriptive-statistics)
- [Solutions](#solutions)
  - [Custom Implementation](#custom-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

## Problem Statement

### [Descriptive Statistics Calculator](https://www.deep-ml.com/problems/78)

Implement a Python function `descriptive_statistics(data)` that calculates important descriptive statistics metrics for a given dataset.

The input can be a list or NumPy array containing numerical values.

The function should return a dictionary containing:

- Mean
- Median
- Mode
- Population Variance
- Standard Deviation
- 25th Percentile
- 50th Percentile
- 75th Percentile
- Interquartile Range (IQR)

The function should calculate population variance using:

$$ Variance = \frac{\sum\_{i=1}^{n}(x_i - \bar{x})^2}{n} $$

where $n$ represents the total number of samples.

## Example

### Input

```python
data = [1, 2, 2, 3, 4, 4, 4, 5]

result = descriptive_statistics(data)
print(result)
```

### Output

```text
{
'mean': 3.125,
'median': 3.5,
'mode': 4,
'variance': 1.6094,
'standard_deviation': 1.2686,
...
}
```

### Reasoning

Given:

```text
[1, 2, 2, 3, 4, 4, 4, 5]
```

Mean:

$$ Mean = \frac{1+2+2+3+4+4+4+5}{8} = 3.125 $$

Median:

The sorted dataset has 8 elements, so the middle value is the average of the 4th and 5th elements:

$$ Median = \frac{3+4}{2} = 3.5 $$

Mode:

The value `4` appears three times, which is the highest frequency.

Variance and standard deviation measure how spread out the values are from the mean.

Percentiles divide the sorted dataset into different sections, and IQR represents the spread of the middle 50% of the data.

## Learn: Descriptive Statistics

### What is it?

Descriptive statistics are methods used to summarize and understand the basic characteristics of a dataset.

They provide information about:

- Central tendency: Where data values are centered.
- Dispersion: How spread out the values are.
- Distribution: How values are distributed across ranges.

Before applying machine learning algorithms, descriptive statistics help understand the dataset structure, detect anomalies, and identify important patterns.

The major categories are:

- Measures of central tendency.
- Measures of variability.
- Distribution measurements.

## Central Tendency

Central tendency describes the typical or central value of a dataset.

### Mean

Mean is the average value of all samples.

$$ Mean = \bar{x} = \frac{\sum\_{i=1}^{n}x_i}{n} $$

where:

- $x_i$ represents each data point.
- $n$ represents the total number of samples.

Mean is sensitive to extreme values because outliers can significantly shift the average.

### Median

Median is the middle value after sorting the dataset.

For an odd number of samples:

$$ Median = x\_{\frac{n+1}{2}} $$

For an even number of samples:

$$ Median = \frac{x*{\frac{n}{2}} + x*{\frac{n}{2}+1}}{2} $$

Median is more robust to outliers compared to mean.

### Mode

Mode is the value that occurs most frequently in the dataset.

If multiple values have the same highest frequency, the smallest value is selected.

Example:

```text
[1, 2, 2, 3, 4, 4, 4, 5]

Mode = 4
```

## Spread / Dispersion

Dispersion measures how much values differ from the center of the dataset.

### Population Variance

Variance measures the average squared distance of each value from the mean.

$$ Variance = \sigma^2 = \frac{\sum\_{i=1}^{n}(x_i-\bar{x})^2}{n} $$

A larger variance means values are more spread out.

This implementation uses population variance by dividing by $n$.

Sample variance uses $n-1$ instead.

### Standard Deviation

Standard deviation is the square root of variance.

$$ Standard\ Deviation = \sigma = \sqrt{\sigma^2} $$

It represents the average distance of values from the mean.

Since it uses the same units as the original data, it is easier to interpret than variance.

## Distribution Statistics

### Percentiles

Percentiles divide sorted data into portions.

A percentile represents the value below which a given percentage of data falls.

Common percentiles:

- 25th percentile (Q1):
  - 25% of values are below this point.

- 50th percentile (Q2):
  - Same as median.

- 75th percentile (Q3):
  - 75% of values are below this point.

### Interquartile Range (IQR)

IQR measures the spread of the middle 50% of the dataset.

$$ IQR = Q3 - Q1 $$

It is commonly used for detecting outliers.

Outlier boundaries:

$$ Lower\ Bound = Q1 - 1.5 \times IQR $$

$$ Upper\ Bound = Q3 + 1.5 \times IQR $$

Values outside these ranges are potential outliers.

## Characteristics / Key Points

- Mean, median, and mode describe the center of data.
- Variance and standard deviation describe data spread.
- Percentiles describe the distribution structure.
- IQR is robust against extreme values.
- Population variance divides by $n$ instead of $n-1$.
- Sorting is required for median and percentile calculations.

## Why is it used? / Applications

Descriptive statistics are widely used in machine learning workflows.

Applications:

- Exploratory Data Analysis (EDA):
  - Understanding dataset distributions before modeling.

- Outlier Detection:
  - Identifying unusual values using IQR.

- Feature Engineering:
  - Understanding feature ranges before scaling.

- Data Quality Analysis:
  - Detecting incorrect or abnormal data.

- Model Preparation:
  - Selecting appropriate preprocessing techniques.

> 💡 **Important Note**
>
> Descriptive statistics are usually the first step in a machine learning pipeline. Understanding the distribution of features helps choose better preprocessing methods such as normalization, scaling, or outlier removal.

## Solutions

### Custom Implementation

```python
import numpy as np

def descriptive_statistics(data: list | np.ndarray) -> dict:
    if not data:
        raise ValueError("Data cannot be empty")

    nums = sorted(data)
    n = len(nums)

    result = {}

    mean = sum(nums) / n
    result["mean"] = mean

    if n % 2:
        median = nums[n // 2]
    else:
        median = (nums[n // 2 - 1] + nums[n // 2]) / 2

    result["median"] = median

    frequency = {}

    for value in nums:
        frequency[value] = frequency.get(value, 0) + 1

    max_frequency = max(frequency.values())

    modes = [
        value
        for value, count in frequency.items()
        if count == max_frequency
    ]

    result["mode"] = modes[0]

    variance = sum((x - mean) ** 2 for x in nums) / n

    result["variance"] = variance
    result["standard_deviation"] = variance ** 0.5

    arr = np.array(nums)

    p25, p50, p75 = np.percentile(
        arr,
        [25, 50, 75]
    )

    result["25th_percentile"] = p25.item()
    result["50th_percentile"] = p50.item()
    result["75th_percentile"] = p75.item()
    result["interquartile_range"] = (p75 - p25).item()

    return result
```

## Code Explanation

### Step 1: Validate Input

The function first checks whether the dataset is empty.

An empty dataset cannot have meaningful statistical values.

### Step 2: Sort the Dataset

```python
nums = sorted(data)
```

Sorting helps calculate:

- Median.
- Percentiles.
- Quartiles.

### Step 3: Calculate Mean

```python
mean = sum(nums) / n
```

The average is calculated by dividing the sum of all values by the number of samples.

### Step 4: Calculate Median

For odd-sized datasets:

- Select the middle element.

For even-sized datasets:

- Average the two middle elements.

### Step 5: Calculate Mode

A frequency dictionary stores the count of each value.

The value with the maximum frequency becomes the mode.

If multiple values have the same frequency, the smallest value is selected because the sorted order is maintained.

### Step 6: Calculate Variance and Standard Deviation

Variance is calculated by finding the squared distance of every value from the mean.

Then:

$$ Standard\ Deviation = \sqrt{Variance} $$

### Step 7: Calculate Percentiles

NumPy percentile calculation is used to find:

- Q1 (25th percentile)
- Q2 (50th percentile)
- Q3 (75th percentile)

Finally:

$$ IQR = Q3 - Q1 $$

## Time & Space Complexity

| Complexity | Value          |
| ---------- | -------------- |
| Time       | **O(n log n)** |
| Space      | **O(n)**       |

Where:

- **n** is the number of samples.
- Sorting requires **O(n log n)** time.
- The sorted list requires **O(n)** additional space.
