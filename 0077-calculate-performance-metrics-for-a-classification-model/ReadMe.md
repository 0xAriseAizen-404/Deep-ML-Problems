# Calculate Performance Metrics for a Classification Model (Medium, Machine Learning)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Performance Metrics](#learn-performance-metrics)
  - [What is it?](#what-is-it)
  - [Confusion Matrix](#confusion-matrix)
  - [True Positive](#true-positive)
  - [False Negative](#false-negative)
  - [False Positive](#false-positive)
  - [True Negative](#true-negative)
  - [Accuracy](#accuracy)
  - [Precision](#precision)
  - [Recall](#recall)
  - [Specificity](#specificity)
  - [Negative Predictive Value](#negative-predictive-value)
  - [F1 Score](#f1-score)
  - [Relationship Between Metrics](#relationship-between-metrics)
  - [Characteristics / Key Points](#characteristics--key-points)
  - [Why is it used? / Applications](#why-is-it-used--applications)
- [Solutions](#solutions)
  - [Custom Implementation](#custom-implementation)
  - [NumPy / Scikit-learn Equivalent](#numpy--scikit-learn-equivalent)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

---

## Problem Statement

### [Calculate Performance Metrics for a Classification Model](https://www.deep-ml.com/problems/77)

Implement a function `performance_metrics(actual, predicted)` that calculates several performance metrics for a **binary classification model**.

The function receives two lists:

- `actual`: The ground-truth class labels.
- `predicted`: The labels predicted by the model.

Each label is either `0` or `1`.

The function must calculate and return:

1. Confusion Matrix
2. Accuracy
3. F1 Score
4. Specificity
5. Negative Predictive Value

The metrics must be rounded to three decimal places.

The confusion matrix should have the form

```text
[[TP, FN],
 [FP, TN]]
```

---

## Example

### Input

```python
actual = [1, 0, 1, 0, 1]

predicted = [1, 0, 0, 1, 1]

print(performance_metrics(actual, predicted))
```

### Output

```text
([[2, 1], [1, 1]], 0.6, 0.667, 0.5, 0.5)
```

### Reasoning

Compare each actual label with its corresponding prediction.

| Actual | Predicted | Result |
| ------ | --------- | ------ |
| 1      | 1         | TP     |
| 0      | 0         | TN     |
| 1      | 0         | FN     |
| 0      | 1         | FP     |
| 1      | 1         | TP     |

Therefore,

```text
TP = 2
TN = 1
FP = 1
FN = 1
```

The confusion matrix is

$$
\begin{bmatrix}
2 & 1\
1 & 1
\end{bmatrix}
$$

The total number of observations is

$$
N = TP + TN + FP + FN = 5
$$

Accuracy is

$$
Accuracy = \frac{TP + TN}{TP + TN + FP + FN} = \frac{2+1}{5} = 0.6
$$

Precision is

$$
Precision = \frac{TP}{TP+FP} = \frac{2}{3} \approx 0.667
$$

Recall is

$$
Recall = \frac{TP}{TP+FN} = \frac{2}{3} \approx 0.667
$$

The F1 score is approximately

$$
F1 = \frac{2(0.667)(0.667)}{0.667+0.667} \approx 0.667
$$

Specificity is

$$
Specificity = \frac{TN}{TN+FP} = \frac{1}{2} = 0.5
$$

Negative Predictive Value is

$$
NPV = \frac{TN}{TN+FN} = \frac{1}{2} = 0.5
$$

Thus the final result is

```text
([[2, 1], [1, 1]], 0.6, 0.667, 0.5, 0.5)
```

---

## Learn: Performance Metrics

### What is it?

**Performance metrics** are numerical measures used to evaluate how well a classification model predicts the correct class.

For a binary classification problem, predictions generally belong to one of two classes:

```text
0 → Negative
1 → Positive
```

Simply counting how many predictions are correct is often not enough.

For example, suppose a medical dataset contains `990` healthy patients and only `10` patients with a disease.

A model that predicts every patient as healthy achieves

$$
Accuracy = \frac{990}{1000} = 0.99
$$

Although the accuracy is `99%`, the model completely fails to identify the diseased patients.

Therefore, metrics such as **precision, recall, specificity, F1 score, and negative predictive value** provide additional information about the model's behavior.

The **confusion matrix** is the foundation from which many of these metrics are calculated.

---

## Confusion Matrix

For binary classification, the confusion matrix is a `2 × 2` matrix that summarizes the relationship between actual and predicted classes.

The convention used in this problem is

$$
\begin{bmatrix}
TP & FN\
FP & TN
\end{bmatrix}
$$

It can be visualized as

|                 | Predicted Positive | Predicted Negative |
| --------------- | ------------------ | ------------------ |
| Actual Positive | TP                 | FN                 |
| Actual Negative | FP                 | TN                 |

The four entries are:

- **TP**: True Positive
- **FN**: False Negative
- **FP**: False Positive
- **TN**: True Negative

These four values are the basic building blocks of the classification metrics.

---

### True Positive

A **True Positive (TP)** occurs when the actual class is positive and the model also predicts positive.

```text
Actual = 1
Predicted = 1
```

For example, if a patient actually has a disease and the model predicts that the patient has the disease, the prediction is a true positive.

TP contributes positively to metrics such as accuracy, precision, recall, and F1 score.

---

### False Negative

A **False Negative (FN)** occurs when the actual class is positive but the model predicts negative.

```text
Actual = 1
Predicted = 0
```

This is also called a **Type II error**.

For example, if a patient has a disease but the model predicts that the patient is healthy, the prediction is a false negative.

False negatives are particularly important in applications where failing to detect a positive case is dangerous.

---

### False Positive

A **False Positive (FP)** occurs when the actual class is negative but the model predicts positive.

```text
Actual = 0
Predicted = 1
```

This is also called a **Type I error**.

For example, if a patient is healthy but the model predicts that the patient has a disease, the prediction is a false positive.

False positives directly affect precision and specificity.

---

### True Negative

A **True Negative (TN)** occurs when the actual class is negative and the model also predicts negative.

```text
Actual = 0
Predicted = 0
```

For example, if a patient is healthy and the model correctly predicts that the patient is healthy, the result is a true negative.

TN contributes to accuracy, specificity, and negative predictive value.

---

## Accuracy

**Accuracy** measures the proportion of all predictions that are correct.

The correct predictions are TP and TN.

Therefore,

$$
Accuracy = \frac{TP+TN}{TP+TN+FP+FN}
$$

The denominator represents the total number of observations.

Accuracy is easy to interpret, but it can be misleading when the classes are highly imbalanced.

For example, if `95%` of observations belong to class `0`, a model that always predicts `0` can achieve `95%` accuracy while completely ignoring class `1`.

---

## Precision

**Precision** measures how many observations predicted as positive are actually positive.

It is defined as

$$
Precision = \frac{TP}{TP+FP}
$$

The denominator contains all predicted positive observations.

```text
Predicted Positive = TP + FP
```

Therefore, precision answers

> When the model predicts positive, how often is it correct?

A high precision model produces relatively few false positives.

Precision is particularly important when false positive predictions are expensive or undesirable.

---

## Recall

**Recall** measures how many actual positive observations are correctly identified.

It is defined as

$$
Recall = \frac{TP}{TP+FN}
$$

The denominator contains all actual positive observations.

```text
Actual Positive = TP + FN
```

Therefore, recall answers

> Out of all actual positive cases, how many did the model detect?

A high recall model produces relatively few false negatives.

Recall is also commonly called **Sensitivity** or **True Positive Rate (TPR)**.

---

## Specificity

**Specificity** measures how well the model identifies negative observations.

It is defined as

$$
Specificity = \frac{TN}{TN+FP}
$$

The denominator contains all actual negative observations.

```text
Actual Negative = TN + FP
```

Therefore, specificity answers

> Out of all actual negative cases, how many did the model correctly identify as negative?

Specificity is also called the **True Negative Rate (TNR)**.

A model with high specificity produces relatively few false positives.

---

## Negative Predictive Value

**Negative Predictive Value (NPV)** measures how many observations predicted as negative are actually negative.

It is defined as

$$
NPV = \frac{TN}{TN+FN}
$$

The denominator contains all predicted negative observations.

```text
Predicted Negative = TN + FN
```

Therefore, NPV answers

> When the model predicts negative, how often is that prediction correct?

NPV is different from specificity.

Specificity starts with the **actual negative population**, while NPV starts with the **predicted negative population**.

---

## F1 Score

The **F1 score** combines precision and recall into a single metric.

It is defined as the harmonic mean of precision and recall.

$$
F1 = \frac{2 \times Precision \times Recall}{Precision+Recall}
$$

The harmonic mean penalizes situations where one of the two values is very low.

For example, suppose

$$
Precision = 1.0
$$

and

$$
Recall = 0.1
$$

The arithmetic mean would be relatively high, but the F1 score is

$$
F1 = \frac{2(1.0)(0.1)}{1.0+0.1} \approx 0.182
$$

This reflects the poor recall.

The F1 score is therefore useful when both precision and recall matter.

---

## F1 Score Directly From the Confusion Matrix

Precision and recall can be substituted into the F1 formula.

Since

$$
Precision = \frac{TP}{TP+FP}
$$

and

$$
Recall = \frac{TP}{TP+FN}
$$

the F1 score can also be written as

$$
F1 = \frac{2TP}{2TP+FP+FN}
$$

This form allows the F1 score to be calculated directly from the four confusion-matrix values.

---

## Relationship Between Metrics

The metrics focus on different aspects of classification performance.

| Metric      | Formula                     | Main Question                                        |
| ----------- | --------------------------- | ---------------------------------------------------- |
| Accuracy    | $\frac{TP+TN}{TP+TN+FP+FN}$ | How many predictions are correct overall?            |
| Precision   | $\frac{TP}{TP+FP}$          | When positive is predicted, how often is it correct? |
| Recall      | $\frac{TP}{TP+FN}$          | How many actual positives were detected?             |
| Specificity | $\frac{TN}{TN+FP}$          | How many actual negatives were detected?             |
| NPV         | $\frac{TN}{TN+FN}$          | When negative is predicted, how often is it correct? |
| F1          | $\frac{2TP}{2TP+FP+FN}$     | How well are precision and recall balanced?          |

Understanding which population appears in the denominator is one of the easiest ways to distinguish these metrics.

---

## Actual vs Predicted Perspective

A useful way to remember the metrics is to look at what the denominator represents.

For **recall**,

$$
TP+FN
$$

represents all actual positives.

Therefore, recall asks how many actual positives were found.

For **specificity**,

$$
TN+FP
$$

represents all actual negatives.

Therefore, specificity asks how many actual negatives were correctly identified.

For **precision**,

$$
TP+FP
$$

represents all predicted positives.

Therefore, precision asks how reliable positive predictions are.

For **NPV**,

$$
TN+FN
$$

represents all predicted negatives.

Therefore, NPV asks how reliable negative predictions are.

This distinction is extremely useful when interpreting classification metrics.

---

## Sensitivity vs Specificity

Sensitivity and specificity describe opposite sides of binary classification.

Sensitivity is

$$
Sensitivity = \frac{TP}{TP+FN}
$$

Specificity is

$$
Specificity = \frac{TN}{TN+FP}
$$

Sensitivity focuses on the **positive class**.

Specificity focuses on the **negative class**.

A model can have high sensitivity but low specificity, or high specificity but low sensitivity.

The appropriate balance depends on the application.

---

## False Positive Rate

Specificity is closely related to the **False Positive Rate (FPR)**.

The false positive rate is

$$
FPR = \frac{FP}{FP+TN}
$$

Since

$$
Specificity = \frac{TN}{FP+TN}
$$

we have

$$
FPR = 1-Specificity
$$

Therefore, a model with high specificity has a low false positive rate.

---

## False Negative Rate

Recall is closely related to the **False Negative Rate (FNR)**.

The false negative rate is

$$
FNR = \frac{FN}{FN+TP}
$$

Since

$$
Recall = \frac{TP}{TP+FN}
$$

we have

$$
FNR = 1-Recall
$$

Therefore, a model with high recall has a low false negative rate.

---

## Class Imbalance

Class imbalance occurs when one class contains significantly more observations than another.

Suppose a dataset contains

```text
950 negative observations
50 positive observations
```

A model that predicts every observation as negative obtains

$$
Accuracy = \frac{950}{1000} = 0.95
$$

This looks good from accuracy alone.

However,

$$
Recall = \frac{0}{0+50} = 0
$$

The model detects none of the positive cases.

This demonstrates why accuracy should not always be used as the only evaluation metric.

For imbalanced datasets, metrics such as precision, recall, F1 score, specificity, and the confusion matrix provide much more information.

---

## Characteristics / Key Points

- Classification metrics quantify different aspects of model performance.
- The confusion matrix is the foundation of many binary classification metrics.
- TP represents correctly predicted positive observations.
- FN represents positive observations incorrectly predicted as negative.
- FP represents negative observations incorrectly predicted as positive.
- TN represents correctly predicted negative observations.
- Accuracy measures overall correctness.
- Precision measures the reliability of positive predictions.
- Recall measures the ability to detect actual positives.
- Specificity measures the ability to detect actual negatives.
- NPV measures the reliability of negative predictions.
- F1 balances precision and recall.
- Accuracy can be misleading on imbalanced datasets.
- Precision and NPV depend on the predicted classes.
- Recall and specificity depend on the actual classes.
- High recall means fewer false negatives.
- High specificity means fewer false positives.
- High precision means fewer false positives among predicted positives.
- High NPV means fewer false negatives among predicted negatives.
- The choice of metric should depend on the problem's objective.
- No single metric completely describes a classification model.

---

## Why is it used? / Applications

Performance metrics are used whenever a machine learning model makes classification predictions.

Common applications include

- Medical diagnosis
- Fraud detection
- Spam detection
- Credit-risk classification
- Intrusion detection
- Customer churn prediction
- Disease screening
- Image classification
- Sentiment classification
- Defect detection
- Recommendation systems
- Search systems
- Anomaly detection

Different applications emphasize different metrics.

For example, in disease screening, missing a positive patient may be more dangerous than producing a false alarm.

In such a case, recall may be especially important.

In spam filtering, incorrectly classifying a legitimate email as spam may be undesirable, making precision important.

In safety-critical systems, both sensitivity and specificity may need to be considered carefully.

---

> 💡 **Important Note**
>
> Accuracy alone does not tell you how a classifier behaves on each class. Always inspect the confusion matrix when evaluating a classification model, especially when the dataset is imbalanced.

---

> 💡 **Interview Tip**
>
> A common interview question is the difference between precision and recall. Remember: **precision starts from predicted positives**, while **recall starts from actual positives**. Precision asks whether positive predictions are trustworthy; recall asks whether the model found the positive cases.

---

> 💡 **Practical Insight**
>
> Specificity and recall are both calculated from the actual classes. Recall measures performance on the positive class, while specificity measures performance on the negative class. Together, they provide a more balanced view of how the model handles both classes.

---

> 💡 **Common Mistake**
>
> Do not confuse specificity with negative predictive value. Specificity is $\frac{TN}{TN+FP}$, where the denominator represents all actual negatives. NPV is $\frac{TN}{TN+FN}$, where the denominator represents all predicted negatives.

---

## Solutions

### Custom Implementation

```python
def performance_metrics(
    actual: list[int],
    predicted: list[int]
) -> tuple:

    TP, FN, FP, TN = 0, 0, 0, 0

    for x, y in zip(actual, predicted):

        if x == 1 and y == 1:
            TP += 1

        elif x == 1 and y == 0:
            FN += 1

        elif x == 0 and y == 1:
            FP += 1

        else:
            TN += 1

    confusion_matrix = [
        [TP, FN],
        [FP, TN]
    ]

    total = TP + FN + FP + TN

    if total > 0:
        accuracy = (TP + TN) / total
    else:
        accuracy = 0.0

    if TP + FP > 0:
        precision = TP / (TP + FP)
    else:
        precision = 0.0

    if TP + FN > 0:
        recall = TP / (TP + FN)
    else:
        recall = 0.0

    if precision + recall > 0:
        f1 = (
            2 * precision * recall
        ) / (precision + recall)
    else:
        f1 = 0.0

    if FP + TN > 0:
        specificity = TN / (FP + TN)
    else:
        specificity = 0.0

    if FN + TN > 0:
        negative_predictive = TN / (TN + FN)
    else:
        negative_predictive = 0.0

    return (
        confusion_matrix,
        round(accuracy, 3),
        round(f1, 3),
        round(specificity, 3),
        round(negative_predictive, 3)
    )
```

### NumPy / Scikit-learn Equivalent

The same metrics can be obtained using established machine learning libraries.

```python
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    recall_score
)

cm = confusion_matrix(
    actual,
    predicted
)

accuracy = accuracy_score(
    actual,
    predicted
)

f1 = f1_score(
    actual,
    predicted
)

tn, fp, fn, tp = cm.ravel()

specificity = (
    tn / (tn + fp)
    if tn + fp > 0
    else 0.0
)

negative_predictive_value = (
    tn / (tn + fn)
    if tn + fn > 0
    else 0.0
)
```

The library functions automate standard metric calculations, while the custom implementation demonstrates how the metrics are derived directly from the predictions.

---

## Code Explanation

### Step 1: Initialize the Confusion-Matrix Counts

The implementation starts with four counters.

```python
TP, FN, FP, TN = 0, 0, 0, 0
```

Each counter represents one of the four possible combinations of actual and predicted labels.

Initially, all counts are zero.

---

### Step 2: Process Actual and Predicted Labels Together

The two lists are traversed simultaneously.

```python
for x, y in zip(actual, predicted):
```

Here,

```text
x → actual label
y → predicted label
```

For every observation, the function determines which confusion-matrix category it belongs to.

---

### Step 3: Detect True Positives

```python
if x == 1 and y == 1:
    TP += 1
```

If both the actual and predicted labels are `1`, the prediction is a true positive.

The TP counter is incremented.

---

### Step 4: Detect False Negatives

```python
elif x == 1 and y == 0:
    FN += 1
```

If the actual label is positive but the model predicts negative, the prediction is a false negative.

The FN counter is incremented.

---

### Step 5: Detect False Positives

```python
elif x == 0 and y == 1:
    FP += 1
```

If the actual label is negative but the model predicts positive, the prediction is a false positive.

The FP counter is incremented.

---

### Step 6: Detect True Negatives

```python
else:
    TN += 1
```

The remaining valid binary case is

```text
Actual = 0
Predicted = 0
```

Therefore, the TN counter is incremented.

---

### Step 7: Construct the Confusion Matrix

The four counts are arranged as

```python
confusion_matrix = [
    [TP, FN],
    [FP, TN]
]
```

Therefore, the matrix has the structure

$$
\begin{bmatrix}
TP & FN\
FP & TN
\end{bmatrix}
$$

This ordering is important because different libraries can use different conventions.

---

### Step 8: Calculate Accuracy

The total number of observations is

```python
total = TP + FN + FP + TN
```

Correct predictions are TP and TN.

Therefore,

```python
accuracy = (TP + TN) / total
```

which corresponds to

$$
Accuracy = \frac{TP+TN}{TP+TN+FP+FN}
$$

The implementation checks whether `total` is non-zero to avoid division by zero.

---

### Step 9: Calculate Precision

Precision is calculated using

```python
precision = TP / (TP + FP)
```

when the denominator is non-zero.

This corresponds to

$$
Precision = \frac{TP}{TP+FP}
$$

The code stores precision because it is required to calculate the F1 score.

---

### Step 10: Calculate Recall

Recall is calculated using

```python
recall = TP / (TP + FN)
```

when the denominator is non-zero.

This corresponds to

$$
Recall = \frac{TP}{TP+FN}
$$

Recall measures how many actual positive observations were correctly detected.

---

### Step 11: Calculate the F1 Score

The implementation combines precision and recall.

```python
f1 = (
    2 * precision * recall
) / (precision + recall)
```

This corresponds to

$$
F1 = \frac{2 \times Precision \times Recall}{Precision+Recall}
$$

The denominator is checked before performing the division so that the implementation remains safe when both precision and recall are zero.

---

### Step 12: Calculate Specificity

Specificity uses TN and FP.

```python
specificity = TN / (FP + TN)
```

This corresponds to

$$
Specificity = \frac{TN}{TN+FP}
$$

The denominator represents all actual negative observations.

---

### Step 13: Calculate Negative Predictive Value

NPV uses TN and FN.

```python
negative_predictive = TN / (TN + FN)
```

This corresponds to

$$
NPV = \frac{TN}{TN+FN}
$$

The denominator represents all observations predicted as negative.

---

### Step 14: Round the Results

The required metrics are rounded to three decimal places.

```python
round(accuracy, 3)
round(f1, 3)
round(specificity, 3)
round(negative_predictive, 3)
```

The confusion matrix contains integer counts, so it does not require rounding.

---

### Step 15: Return the Results

The function returns a tuple containing

```python
return (
    confusion_matrix,
    round(accuracy, 3),
    round(f1, 3),
    round(specificity, 3),
    round(negative_predictive, 3)
)
```

The order is exactly

```text
Confusion Matrix
Accuracy
F1 Score
Specificity
Negative Predictive Value
```

---

### Complete Logic

The entire algorithm can be summarized as

```text
1. Initialize TP, FN, FP, TN to zero.
2. Traverse actual and predicted labels together.
3. Classify every observation as TP, FN, FP, or TN.
4. Construct the confusion matrix.
5. Calculate accuracy.
6. Calculate precision.
7. Calculate recall.
8. Calculate F1 score from precision and recall.
9. Calculate specificity.
10. Calculate negative predictive value.
11. Round the floating-point metrics.
12. Return all requested values.
```

The important insight is that all requested metrics can be derived from the four confusion-matrix counts.

---

### Edge Cases

The implementation checks denominators before division.

For example, accuracy requires

$$
TP+TN+FP+FN > 0
$$

If there are no observations, accuracy is set to `0.0`.

Similarly, precision requires

$$
TP+FP > 0
$$

Recall requires

$$
TP+FN > 0
$$

Specificity requires

$$
TN+FP > 0
$$

and NPV requires

$$
TN+FN > 0
$$

These checks prevent division-by-zero errors.

The F1 score additionally requires

$$
Precision+Recall > 0
$$

because otherwise its denominator is zero.

---

### Implementation Detail

The original solution can be simplified slightly by calculating the four confusion-matrix counts first and deriving every metric afterward.

This follows a useful pattern for classification problems:

```text
Predictions
    ↓
Confusion Matrix
    ↓
TP / FN / FP / TN
    ↓
Derived Metrics
```

Once the four counts are known, the remaining calculations become direct applications of their mathematical definitions.

---

## Time & Space Complexity

Let

- $n$ be the number of observations.
- $TP$ be the number of true positives.
- $TN$ be the number of true negatives.
- $FP$ be the number of false positives.
- $FN$ be the number of false negatives.

The implementation traverses the input lists once.

Therefore, the time complexity for constructing the confusion matrix is

$$
O(n)
$$

After the traversal, all metrics are calculated using only the four counters.

Each metric calculation takes constant time.

Therefore, the overall time complexity remains

$$
O(n)
$$

Only four integer counters and a constant-size confusion matrix are maintained.

Therefore, the additional auxiliary space is

$$
O(1)
$$

excluding the input lists.

| Complexity      | Value    |
| --------------- | -------- |
| Time            | **O(n)** |
| Auxiliary Space | **O(1)** |

where $n$ is the number of observations.

The algorithm is therefore optimal in time for this problem because every actual/predicted pair must be inspected at least once to determine the confusion-matrix counts.
