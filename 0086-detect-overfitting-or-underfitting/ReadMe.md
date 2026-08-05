# Detect Overfitting or Underfitting (Easy, Machine Learning)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Overfitting and Underfitting](#learn-overfitting-and-underfitting)
- [Solutions](#solutions)
  - [Custom Implementation](#custom-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

## Problem Statement

### [Detect Overfitting or Underfitting](https://www.deep-ml.com/problems/86)

Implement a Python function `model_fit_quality(training_accuracy, test_accuracy)` to determine whether a machine learning model is:

- Overfitting
- Underfitting
- Performing well (Good Fit)

The function takes:

- `training_accuracy`: Accuracy of the model on training data.
- `test_accuracy`: Accuracy of the model on unseen test data.

The function should return:

- `1` if the model is overfitting.
- `-1` if the model is underfitting.
- `0` if the model has a good fit.

The decision rules are:

### Overfitting

A model is overfitting when the training accuracy is significantly higher than test accuracy.

$$ Training\ Accuracy - Test\ Accuracy > 0.2 $$

### Underfitting

A model is underfitting when both training and test accuracy values are low.

$$ Training\ Accuracy < 0.7 \ and \ Test\ Accuracy < 0.7 $$

### Good Fit

If neither condition is true, the model is considered to have a good fit.

## Example

### Input

```python
training_accuracy = 0.95
test_accuracy = 0.65

result = model_fit_quality(training_accuracy, test_accuracy)
print(result)
```

### Output

```text
1
```

### Reasoning

The difference between training and test accuracy is:

$$ 0.95 - 0.65 = 0.30 $$

Since:

$$ 0.30 > 0.2 $$

the model is overfitting.

The model has learned the training data too specifically and does not generalize well to unseen data.

## Learn: Overfitting and Underfitting

### What is it?

Overfitting and underfitting are two common problems in machine learning that affect how well a model generalizes to new data.

A machine learning model should learn useful patterns from training data while avoiding memorizing noise.

The goal is to find a balance between model complexity and generalization ability.

## Overfitting

Overfitting occurs when a model learns the training data too well, including noise and unnecessary patterns.

The model performs very well on training data but performs poorly on unseen test data.

Typical behavior:

- Training accuracy is high.
- Test accuracy is significantly lower.
- Large gap between training and test performance.

Example:

```text id="h4wq7m"
Training Accuracy = 98%
Test Accuracy = 65%
```

The model has memorized training examples instead of learning general patterns.

### Causes of Overfitting

Common causes include:

- Model is too complex.
- Too many parameters compared to available data.
- Training for too many epochs.
- No regularization.
- Dataset is too small.

### Solutions for Overfitting

Methods to reduce overfitting:

- Add regularization:
  - L1 regularization.
  - L2 regularization.

- Reduce model complexity.
- Remove unnecessary features.
- Add more training data.
- Use dropout in neural networks.
- Apply early stopping.

## Underfitting

Underfitting occurs when a model is too simple to learn the underlying patterns in the dataset.

The model performs poorly on both training and test data.

Typical behavior:

- Low training accuracy.
- Low test accuracy.
- Model fails to capture important relationships.

Example:

```text id="6plm8r"
Training Accuracy = 55%
Test Accuracy = 50%
```

The model has high bias and cannot represent the complexity of the problem.

### Causes of Underfitting

Common causes include:

- Model is too simple.
- Insufficient training.
- Poor feature selection.
- Excessive regularization.
- Not enough model capacity.

### Solutions for Underfitting

Methods to reduce underfitting:

- Increase model complexity.
- Add more useful features.
- Train for more epochs.
- Improve feature engineering.
- Reduce excessive regularization.

## Bias-Variance Tradeoff

Machine learning models balance between:

- Bias:
  - Error caused by overly simple assumptions.
  - Related to underfitting.

- Variance:
  - Error caused by sensitivity to training data.
  - Related to overfitting.

A good model achieves a balance where it learns meaningful patterns without memorizing noise.

## Characteristics / Key Points

- Overfitting means high training performance but poor generalization.
- Underfitting means poor performance on both training and testing data.
- A good fit has similar and reasonably high training and test accuracy.
- Training accuracy alone cannot determine model quality.
- Test performance is important because it measures generalization.

> 💡 **Important Note**
>
> A small difference between training and test accuracy is normal. The goal is not to maximize training accuracy but to build a model that performs well on unseen data.

## Why is it used? / Applications

Detecting model fit quality is important during model evaluation.

Applications:

- Model debugging:
  - Identifying whether a model needs more complexity or regularization.

- Neural network training:
  - Deciding when to use dropout or early stopping.

- Machine learning pipelines:
  - Improving generalization performance.

- Hyperparameter tuning:
  - Selecting appropriate model complexity.

## Solutions

### Custom Implementation

```python
def model_fit_quality(training_accuracy, test_accuracy):
    if training_accuracy - test_accuracy > 0.2:
        return 1
    elif training_accuracy < 0.7 and test_accuracy < 0.7:
        return -1
    else:
        return 0
```

## Code Explanation

### Step 1: Check Overfitting Condition

```python
if training_accuracy - test_accuracy > 0.2:
```

The function calculates the difference between training and test accuracy.

If the training accuracy is much higher than test accuracy, the model is likely memorizing training data.

Example:

```text id="x7p1eg"
Training Accuracy = 0.95
Test Accuracy = 0.65

Difference = 0.30
```

Since the difference is greater than `0.2`, the model is classified as overfitting.

### Step 2: Check Underfitting Condition

```python
elif training_accuracy < 0.7 and test_accuracy < 0.7:
```

If both training and test accuracy are below `0.7`, the model is not learning enough from the data.

The model is classified as underfitting.

### Step 3: Return Good Fit

```python
else:
    return 0
```

If neither overfitting nor underfitting conditions are satisfied, the model is considered to have a good fit.

## Time & Space Complexity

| Complexity | Value    |
| ---------- | -------- |
| Time       | **O(1)** |
| Space      | **O(1)** |

Where:

- The function performs only a fixed number of arithmetic operations and comparisons.
- No additional memory is required.
