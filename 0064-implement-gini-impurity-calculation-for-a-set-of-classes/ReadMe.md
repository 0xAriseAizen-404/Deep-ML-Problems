# Implement Gini Impurity Calculation for a Set of Classes (Easy, Machine Learning)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Understanding Gini Impurity](#learn-understanding-gini-impurity)
  - [What is it?](#what-is-it)
  - [Mathematical Definition](#mathematical-definition)
  - [Class Probabilities](#class-probabilities)
  - [Binary Classification Example](#binary-classification-example)
  - [Multi-Class Gini Impurity](#multi-class-gini-impurity)
  - [Gini Impurity in Decision Trees](#gini-impurity-in-decision-trees)
  - [Choosing a Split](#choosing-a-split)
  - [Characteristics / Key Points](#characteristics--key-points)
  - [Why is it used? / Applications](#why-is-it-used--applications)
- [Solutions](#solutions)
  - [Custom Implementation](#custom-implementation)
  - [NumPy Implementation](#numpy-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

---

## Problem Statement

### [Implement Gini Impurity Calculation for a Set of Classes](https://www.deep-ml.com/problems/64)

Implement a Python function that calculates the **Gini Impurity** for a given set of class labels.

Gini impurity measures the amount of impurity or disorder within a node of a classification dataset. It is commonly used by **decision tree algorithms** to evaluate the quality of a split.

The function should:

- Accept a list of class labels.
- Calculate the probability of each unique class.
- Compute the Gini impurity using the class probabilities.
- Return the result rounded to three decimal places.

---

## Example

### Input

```python
y = [0, 1, 1, 1, 0]

print(gini_impurity(y))
```

### Output

```text
0.48
```

### Reasoning

The dataset contains five observations.

Class `0` appears twice:

$$
p_0=\frac{2}{5}=0.4
$$

Class `1` appears three times:

$$
p_1=\frac{3}{5}=0.6
$$

The Gini impurity is

$$
G=1-(p_0^2+p_1^2)
$$

Substituting the probabilities gives

$$
G=1-(0.4^2+0.6^2)=0.48
$$

Therefore, the Gini impurity is **0.48**.

---

## Learn: Understanding Gini Impurity

### What is it?

**Gini impurity** is a measure of how mixed the classes are within a dataset or decision tree node.

For a classification node, it measures the probability that a randomly selected sample would be incorrectly classified if it were assigned a class according to the class distribution of that node.

A node containing samples from only one class has zero impurity.

For example,

```text
[1, 1, 1, 1, 1]
```

is completely pure because every sample belongs to the same class.

Its Gini impurity is

$$
G=1-1^2=0
$$

A node containing an even mixture of classes has higher impurity.

For binary classification, the maximum occurs when both classes have probability `0.5`.

$$
G=1-(0.5^2+0.5^2)=0.5
$$

Therefore, for binary classification, Gini impurity lies between `0` and `0.5`.

---

## Mathematical Definition

Suppose a node contains samples belonging to $C$ different classes.

Let $p_i$ be the proportion of samples belonging to class $i$.

The Gini impurity is defined as

$$
G=1-\sum_{i=1}^{C}p_i^2
$$

where

- $G$ is the Gini impurity.
- $C$ is the number of classes.
- $p_i$ is the probability or proportion of samples belonging to class $i$.

Since the class probabilities satisfy

$$
\sum_{i=1}^{C}p_i=1
$$

the Gini impurity measures how far the class distribution is from being completely concentrated in one class.

---

## Class Probabilities

The probability of a class is calculated from its frequency.

If class $i$ occurs $n_i$ times in a node containing $N$ total samples, then

$$
p_i=\frac{n_i}{N}
$$

For example,

```text
y = [0, 1, 1, 1, 0]
```

The total number of samples is

$$
N=5
$$

The frequency of class `0` is `2` and the frequency of class `1` is `3`.

Therefore,

$$
p_0=\frac{2}{5}
$$

and

$$
p_1=\frac{3}{5}
$$

The probabilities always sum to one:

$$
p_0+p_1=1
$$

---

## Binary Classification Example

Consider the labels

```text
[0, 0, 0, 0]
```

There is only one class.

Therefore,

$$
p_0=1
$$

and

$$
G=1-1^2=0
$$

The node is completely pure.

Now consider

```text
[0, 0, 1, 1]
```

The class probabilities are

$$
p_0=\frac{2}{4}=0.5
$$

$$
p_1=\frac{2}{4}=0.5
$$

Therefore,

$$
G=1-(0.5^2+0.5^2)
$$

$$
G=0.5
$$

This is the maximum possible Gini impurity for binary classification.

---

## Multi-Class Gini Impurity

Gini impurity also works with more than two classes.

Suppose a node contains

```text
[0, 1, 2, 2, 2, 0]
```

The class counts are

```text
Class 0 → 2
Class 1 → 1
Class 2 → 3
```

There are six total samples.

Therefore,

$$
p_0=\frac{2}{6}
$$

$$
p_1=\frac{1}{6}
$$

$$
p_2=\frac{3}{6}
$$

The Gini impurity becomes

$$
G=1-\left[\left(\frac{2}{6}\right)^2+\left(\frac{1}{6}\right)^2+\left(\frac{3}{6}\right)^2\right]
$$

$$
G=1-\left[\frac{4}{36}+\frac{1}{36}+\frac{9}{36}\right]
$$

$$
G=\frac{22}{36}\approx0.611
$$

For $C$ classes, the maximum possible Gini impurity occurs when every class is equally represented.

Then,

$$
p_i=\frac{1}{C}
$$

and

$$
G_{max}=1-\frac{1}{C}
$$

Therefore, for three classes,

$$
G_{max}=1-\frac{1}{3}=\frac{2}{3}
$$

---

## Gini Impurity in Decision Trees

Gini impurity is primarily used in **classification decision trees**.

A decision tree repeatedly divides the training data into smaller groups called **nodes**.

At every node, the algorithm evaluates possible splits and tries to find one that produces purer child nodes.

For example, suppose a parent node contains

```text
[0, 0, 1, 1]
```

Its impurity is

$$
G_{parent}=0.5
$$

Suppose a split produces two child nodes:

```text
Left  → [0, 0]
Right → [1, 1]
```

Both child nodes are completely pure.

Therefore,

$$
G_{left}=0
$$

and

$$
G_{right}=0
$$

The split has transformed an impure parent node into pure child nodes.

This is exactly the type of split a decision tree prefers.

---

## Choosing a Split

Decision trees generally compare candidate splits using the **weighted Gini impurity** of the resulting child nodes.

Suppose a parent node is divided into two children.

The weighted impurity is

$$
G_{split}=\frac{N_L}{N}G_L+\frac{N_R}{N}G_R
$$

where

- $N$ is the number of samples in the parent node.
- $N_L$ is the number of samples in the left child.
- $N_R$ is the number of samples in the right child.
- $G_L$ is the Gini impurity of the left child.
- $G_R$ is the Gini impurity of the right child.

A decision tree generally prefers the split with the **lowest weighted Gini impurity**.

The corresponding reduction in impurity can be expressed as

$$
\Delta G=G_{parent}-G_{split}
$$

A larger reduction indicates a better split.

---

## Gini Impurity vs Entropy

Both **Gini impurity** and **entropy** measure class impurity and can be used for decision tree splitting.

Gini impurity is

$$
G=1-\sum_{i=1}^{C}p_i^2
$$

Entropy is

$$
H=-\sum_{i=1}^{C}p_i\log_2(p_i)
$$

Gini impurity does not require logarithms, making it computationally simpler.

In practice, both criteria often produce similar decision trees, although the selected splits may differ.

---

## Characteristics / Key Points

- Gini impurity measures class disorder within a node.
- A value of `0` represents a completely pure node.
- Lower Gini impurity means greater class purity.
- Higher Gini impurity means greater class mixing.
- It works with binary and multi-class classification.
- The formula uses squared class probabilities.
- For $C$ classes, the maximum impurity is $1-\frac{1}{C}$.
- For binary classification, the maximum impurity is `0.5`.
- Decision trees use Gini impurity to evaluate classification splits.
- The best split generally minimizes weighted child-node impurity.
- Gini impurity is computationally efficient because it does not require logarithms.
- Class probabilities are obtained from class frequencies within the node.
- A node containing only one class has zero impurity.
- Balanced class distributions produce higher impurity than highly imbalanced distributions.
- Gini impurity is different from classification accuracy.
- Gini impurity measures the distribution of classes within a node rather than the final predictive performance of a model.

---

## Why is it used? / Applications

Gini impurity is commonly used in classification problems involving tree-based models.

Applications include

- Decision Tree Classification
- Random Forests
- Gradient-boosted tree implementations
- Customer classification
- Fraud detection
- Medical diagnosis
- Credit risk classification
- Spam detection
- Customer churn prediction
- Image classification
- Feature-based rule generation

It is especially useful when a model needs to determine which feature and threshold create the purest child nodes.

---

> 💡 **Important Note**
>
> Gini impurity should not be interpreted as model accuracy. A low Gini impurity means that the samples inside a particular node are dominated by one class. It does not directly tell us how accurate the complete trained model will be on unseen data.

---

> 💡 **Interview Tip**
>
> Remember the core idea as **"one minus the squared class probabilities."** For a pure node, one probability is `1`, so Gini becomes `0`. For a balanced binary node, both probabilities are `0.5`, producing the maximum value of `0.5`.

---

## Solutions

### Custom Implementation

```python
def gini_impurity(y):
    """
    Calculate Gini Impurity for a list of class labels.

    :param y: List of class labels
    :return: Gini Impurity rounded to three decimal places
    """
    classes = {}

    for label in y:
        classes[label] = classes.get(label, 0) + 1

    total = len(y)
    impurity = 1.0

    for count in classes.values():
        probability = count / total
        impurity -= probability ** 2

    return round(impurity, 3)
```

### NumPy Implementation

```python
import numpy as np

def gini_impurity(y):
    """
    Calculate Gini Impurity using NumPy.

    :param y: List of class labels
    :return: Gini Impurity rounded to three decimal places
    """
    _, counts = np.unique(
        y,
        return_counts=True
    )

    probabilities = counts / counts.sum()

    return round(
        1 - np.sum(probabilities ** 2),
        3
    )
```

---

## Code Explanation

### Step 1: Count Class Frequencies

The custom implementation stores the frequency of every unique class in a dictionary.

```python
classes = {}

for label in y:
    classes[label] = classes.get(label, 0) + 1
```

For

```text
[0, 1, 1, 1, 0]
```

the resulting frequency table is conceptually

```text
0 → 2
1 → 3
```

The frequencies are enough to calculate the class probabilities.

---

### Step 2: Calculate Total Samples

```python
total = len(y)
```

The total number of observations is used as the denominator when calculating each class probability.

For the example,

$$
N=5
$$

---

### Step 3: Initialize the Impurity

```python
impurity = 1.0
```

The Gini formula starts with

$$
G=1-\sum_i p_i^2
$$

Therefore, the implementation starts at `1` and subtracts the squared probability of every class.

---

### Step 4: Calculate Class Probabilities

```python
for count in classes.values():
    probability = count / total
```

For each class,

$$
p_i=\frac{n_i}{N}
$$

where $n_i$ is the class frequency and $N$ is the total number of observations.

---

### Step 5: Subtract Squared Probabilities

```python
impurity -= probability ** 2
```

This directly implements

$$
G=1-\sum_i p_i^2
$$

For the example,

$$
G=1-(0.4^2+0.6^2)
$$

which gives

$$
G=0.48
$$

---

### Step 6: NumPy Class Counting

The NumPy implementation uses

```python
_, counts = np.unique(
    y,
    return_counts=True
)
```

`np.unique()` identifies the unique labels and `return_counts=True` returns how many times each label occurs.

For example,

```text
y = [0, 1, 1, 1, 0]
```

produces counts equivalent to

```text
[2, 3]
```

---

### Step 7: Calculate All Probabilities at Once

```python
probabilities = counts / counts.sum()
```

NumPy performs element-wise division, producing

```text
[0.4, 0.6]
```

These values represent the class probabilities.

---

### Step 8: Compute Gini Impurity

```python
1 - np.sum(probabilities ** 2)
```

The squared probabilities are calculated element-wise and then summed.

This directly corresponds to

$$
G=1-\sum_{i=1}^{C}p_i^2
$$

---

### Step 9: Round the Result

```python
round(..., 3)
```

The problem requires the Gini impurity to be returned rounded to three decimal places.

For the example, the result is

```text
0.48
```

---

## Complete Algorithm Flow

```text
Gini Impurity(y)

1. Count the occurrences of every unique class.
2. Find the total number of samples.
3. For each class:
       Calculate its probability.
       Square the probability.
4. Sum all squared probabilities.
5. Subtract the sum from 1.
6. Round the result to three decimal places.
7. Return the Gini impurity.
```

The central computation is therefore

$$
G=1-\sum_{i=1}^{C}\left(\frac{n_i}{N}\right)^2
$$

where $n_i$ is the number of samples belonging to class $i$.

---

## Time & Space Complexity

Let

- $n$ be the number of samples.
- $C$ be the number of unique classes.

The algorithm must inspect every sample to determine class frequencies.

Therefore, counting the classes requires

$$
O(n)
$$

time.

The subsequent calculation iterates over the unique classes and requires

$$
O(C)
$$

time.

Since $C\leq n$, the overall time complexity is

$$
O(n)
$$

The frequency dictionary stores one entry per unique class, requiring

$$
O(C)
$$

additional space.

| Complexity | Value    |
| ---------- | -------- |
| Time       | **O(n)** |
| Space      | **O(C)** |

where

- $n$ is the number of samples.
- $C$ is the number of unique classes.

In the worst case, every sample belongs to a different class, so $C=n$ and the space complexity becomes **O(n)**.
