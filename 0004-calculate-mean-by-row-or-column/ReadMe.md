# Calculate Mean by Row or Column (Easy, Linear Algebra)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Calculate Mean by Row or Column](#learn-calculate-mean-by-row-or-column)
- [Solutions](#solutions)
  - [Custom Implementation](#custom-implementation)
  - [NumPy Implementation](#numpy-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

## Problem Statement

### [Calculate Mean by Row or Column](https://www.deep-ml.com/problems/4)

Write a Python function that computes the **mean** of a matrix either **row-wise** or **column-wise**, depending on the given mode.

- If the mode is `"row"`, return the mean of every row.
- If the mode is `"column"`, return the mean of every column.

The function should return the resulting means as a list of floating-point numbers.

---

## Example

### Input

```python
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

mode = "column"
```

### Output

```python
[4.0, 5.0, 6.0]
```

### Reasoning

The mean of each column is calculated independently.

First column:

$$
\frac{1+4+7}{3}=4
$$

Second column:

$$
\frac{2+5+8}{3}=5
$$

Third column:

$$
\frac{3+6+9}{3}=6
$$

Therefore,

```python
[4.0, 5.0, 6.0]
```

---

## Learn: Calculate Mean by Row or Column

### What is Mean?

The **mean** (or arithmetic average) is one of the most fundamental statistical measures. It represents the central value of a collection of numbers.

For a dataset containing $n$ values,

$$
\mu=\frac{1}{n}\sum_{i=1}^{n}x_i
$$

where

- $\mu$ is the mean.
- $x_i$ is the $i^{th}$ value.
- $n$ is the total number of values.

The mean provides a simple summary of the data and is widely used in statistics and machine learning.

---

### Row Mean

A row mean computes the average of all elements within a single row.

For the $i^{th}$ row,

$$
\mu_{row,i}=\frac{1}{n}\sum_{j=1}^{n}a_{ij}
$$

where

- $a_{ij}$ is the element in row $i$ and column $j$.
- $n$ is the number of columns.

Each row produces one average value.

For example,

```text
[1, 2, 3]
```

has mean

$$
\frac{1+2+3}{3}=2
$$

---

### Column Mean

A column mean computes the average of all elements within a single column.

For the $j^{th}$ column,

$$
\mu_{column,j}=\frac{1}{m}\sum_{i=1}^{m}a_{ij}
$$

where

- $m$ is the number of rows.
- $a_{ij}$ is the element in row $i$ and column $j$.

Each column contributes one value to the output.

For example,

```text
1
4
7
```

has mean

$$
\frac{1+4+7}{3}=4
$$

---

### Mean Along an Axis

When computing means in a matrix, one dimension is **collapsed** while the other is preserved.

For a matrix of shape

$$
m \times n
$$

- **Row mean** collapses the columns, producing **m** values.
- **Column mean** collapses the rows, producing **n** values.

For example,

```text
1 2 3
4 5 6
7 8 9
```

Row means:

```text
2
5
8
```

Column means:

```text
4
5
6
```

Although both operations use the same elements, they summarize the data along different dimensions.

---

### Characteristics / Key Points

- Computes the arithmetic average.
- Row mode returns one value for each row.
- Column mode returns one value for each column.
- Preserves one dimension while reducing the other.
- Produces floating-point results even when inputs are integers.
- Assumes all rows have the same number of columns.

---

### Why is it Used?

Mean computation is one of the most common preprocessing operations in Machine Learning.

Some applications include:

- Feature normalization
- Data standardization
- Exploratory Data Analysis (EDA)
- Missing value imputation
- Computing dataset statistics
- Image processing
- Signal processing
- Batch statistics in Deep Learning

For example, feature scaling often subtracts the column mean before dividing by the standard deviation:

$$
x'=\frac{x-\mu}{\sigma}
$$

where

- $x$ is the original value.
- $\mu$ is the feature mean.
- $\sigma$ is the standard deviation.

---

> 💡 **Important Note**
>
> The row mean and column mean summarize the same matrix from different perspectives. A common mistake is confusing the axis of computation. In NumPy, `axis=0` computes **column means**, while `axis=1` computes **row means** because `axis` specifies the dimension being reduced.

---

## Solutions

### Custom Implementation

```python
def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    res = []

    if mode == "row":
        for vec in matrix:
            res.append(sum(vec) / len(vec))
    else:
        for ind in range(len(matrix[0])):
            summ = sum(vec[ind] for vec in matrix)
            res.append(summ / len(matrix))

    return res
```

### NumPy Implementation

```python
import numpy as np

def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    if mode == "column":
        return np.asarray(matrix).mean(axis=0).tolist()
    else:
        return np.asarray(matrix).mean(axis=1).tolist()
```

---

## Code Explanation

### 1. Check the Mode

The function first determines whether the mean should be computed across rows or columns.

```python
if mode == "row":
```

---

### 2. Compute Row Means

```python
for vec in matrix:
    res.append(sum(vec) / len(vec))
```

Each row is processed independently.

The average is calculated using

$$
\text{Mean}=\frac{\text{Sum of Row}}{\text{Number of Columns}}
$$

One mean value is produced for every row.

---

### 3. Compute Column Means

```python
for ind in range(len(matrix[0])):
```

Each iteration selects one column index.

For that column,

```python
summ = sum(vec[ind] for vec in matrix)
```

adds together the corresponding element from every row.

The column average is then

```python
summ / len(matrix)
```

which is equivalent to

$$
\frac{\text{Sum of Column}}{\text{Number of Rows}}
$$

---

### 4. Return the Result

After computing all averages, the resulting list is returned.

The output length depends on the chosen mode:

- `"row"` → number of rows.
- `"column"` → number of columns.

---

## Time & Space Complexity

Let

- $m$ = number of rows
- $n$ = number of columns

Every element of the matrix is visited exactly once.

| Complexity | Value |
| ---------- | ----- |
| Time | **O(m × n)** |
| Space | **O(m + n)** |

The output stores either **m** row means or **n** column means, depending on the selected mode.