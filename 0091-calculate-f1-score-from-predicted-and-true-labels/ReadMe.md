# Calculate F1 Score from Predicted and True Labels (Easy, Machine Learning)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: F1 Score](#learn-f1-score)
- [Solutions](#solutions)
  - [Custom Implementation](#custom-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

## Problem Statement

### [Calculate F1 Score from Predicted and True Labels](https://www.deep-ml.com/problems/91)

Implement a Python function `calculate_f1_score(y_true, y_pred)` that calculates the F1 score for a binary classification problem.

The input contains:

- `y_true`: Actual binary labels.
- `y_pred`: Predicted binary labels from the model.

The F1 score combines precision and recall into a single metric using their harmonic mean.

The function should calculate the F1 score and return the value rounded to three decimal places.

The F1 score is defined as:

$$ F1 = \frac{2 \times Precision \times Recall}{Precision + Recall} $$

where:

$$ Precision = \frac{TP}{TP + FP} $$

$$ Recall = \frac{TP}{TP + FN} $$

where:

- **TP (True Positive)**: Positive samples correctly predicted as positive.
- **FP (False Positive)**: Negative samples incorrectly predicted as positive.
- **FN (False Negative)**: Positive samples incorrectly predicted as negative.

## Example

### Input

```python
y_true = [1, 0, 1, 1, 0]
y_pred = [1, 0, 0, 1, 1]

result = calculate_f1_score(y_true, y_pred)
print(result)
```

### Output

```text
0.667
```

### Reasoning

From the predictions:

- True Positives (TP) = 2
- False Positives (FP) = 1
- False Negatives (FN) = 1

Precision:

$$ Precision = \frac{2}{2+1} = 0.667 $$

Recall:

$$ Recall = \frac{2}{2+1} = 0.667 $$

F1 score:

$$ F1 = \frac{2 \times 0.667 \times 0.667}{0.667 + 0.667} = 0.667 $$

## Learn: F1 Score

### What is it?

The F1 score is a classification evaluation metric that combines precision and recall into a single value.

It is the harmonic mean of precision and recall, which means it gives importance to both metrics and decreases significantly when either precision or recall is low.

The F1 score is especially useful when working with imbalanced datasets where accuracy may not provide meaningful information.

For example, in fraud detection:

- The number of fraudulent transactions may be much smaller than normal transactions.
- Accuracy can appear high even if the model fails to detect fraud.
- F1 score provides a better evaluation by considering positive predictions.

### Mathematical Definition / Formula

The F1 score is calculated as:

$$ F1 = \frac{2 \times Precision \times Recall}{Precision + Recall} $$

Precision measures how many predicted positive samples are actually positive:

$$ Precision = \frac{TP}{TP + FP} $$

Recall measures how many actual positive samples are correctly detected:

$$ Recall = \frac{TP}{TP + FN} $$

Combining both:

$$ F1 = \frac{2TP}{2TP + FP + FN} $$

where:

- $TP$ represents true positives.
- $FP$ represents false positives.
- $FN$ represents false negatives.

### Confusion Matrix View

|                 | Predicted Positive  | Predicted Negative  |
| --------------- | ------------------- | ------------------- |
| Actual Positive | True Positive (TP)  | False Negative (FN) |
| Actual Negative | False Positive (FP) | True Negative (TN)  |

The F1 score only considers:

- True Positives.
- False Positives.
- False Negatives.

True negatives are ignored because F1 focuses on the positive class.

### Characteristics / Key Points

- F1 score ranges from 0 to 1.
- A value close to 1 indicates strong classification performance.
- It balances precision and recall.
- It is useful when classes are imbalanced.
- A model with high precision but low recall will have a lower F1 score.
- A model with high recall but low precision will also have a lower F1 score.

### Precision vs Recall vs F1 Score

Precision focuses on false positives:

> "When the model predicts positive, how often is it correct?"

Recall focuses on false negatives:

> "How many actual positives did the model detect?"

F1 score balances both:

> "How well does the model detect positives while avoiding incorrect positive predictions?"

### Why is it used? / Applications

F1 score is commonly used in:

- Binary classification:
  - Spam detection.
  - Fraud detection.
  - Medical diagnosis.

- Multi-class classification:
  - Evaluating each class separately.
  - Macro and weighted averaging.

- Information retrieval:
  - Search engines.
  - Recommendation systems.

> 💡 **Important Note**
>
> F1 score is useful when both false positives and false negatives matter. However, if one type of error is significantly more important, precision or recall may be a better metric.

## Solutions

### Custom Implementation

```python id="v6r8gx"
def calculate_f1_score(y_true, y_pred):
    TP, FN, FP, TN = 0, 0, 0, 0

    for x, y in zip(y_true, y_pred):
        if x & y:
            TP += 1
        elif x & (1 - y):
            FN += 1
        elif (1 - x) & y:
            FP += 1
        else:
            TN += 1

    if (TP + FP) > 0.0:
        precision = TP / (TP + FP)
    else:
        precision = 0.0

    if (TP + FN) > 0.0:
        recall = TP / (TP + FN)
    else:
        recall = 0.0

    if (precision + recall) > 0.0:
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return round(f1, 3)
```

## Code Explanation

### Step 1: Calculate Confusion Matrix Values

The function iterates through actual and predicted labels together.

For each sample:

- If both values are `1`, it is a true positive.
- If actual is `1` and prediction is `0`, it is a false negative.
- If actual is `0` and prediction is `1`, it is a false positive.
- Otherwise, it is a true negative.

### Step 2: Calculate Precision

```python
precision = TP / (TP + FP)
```

Precision measures the correctness of positive predictions.

The division-by-zero case is handled by returning `0.0`.

### Step 3: Calculate Recall

```python
recall = TP / (TP + FN)
```

Recall measures how many actual positive cases were identified.

If there are no actual positives, recall is set to `0.0`.

### Step 4: Calculate F1 Score

```python
f1 = (2 * precision * recall) / (precision + recall)
```

The harmonic mean combines precision and recall.

If both precision and recall are zero, the function returns `0.0`.

Finally, the result is rounded to three decimal places.

## Time & Space Complexity

| Complexity | Value    |
| ---------- | -------- |
| Time       | **O(n)** |
| Space      | **O(1)** |

Where:

- **n** is the number of samples in `y_true` and `y_pred`.
- Each sample is processed once.
- Only counters and metric variables are stored.
