# Compute Multi-class Cross-Entropy Loss (Easy, Deep Learning)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Multi-class Cross-Entropy Loss](#learn-multi-class-cross-entropy-loss)
  - [What is Cross-Entropy?](#what-is-cross-entropy)
  - [Mathematical Definition](#mathematical-definition)
  - [One-Hot Labels](#one-hot-labels)
  - [Why the Logarithm?](#why-the-logarithm)
  - [Batch Cross-Entropy](#batch-cross-entropy)
  - [Numerical Stability](#numerical-stability)
  - [Probability Requirements](#probability-requirements)
  - [Interpretation](#interpretation)
  - [Cross-Entropy vs Accuracy](#cross-entropy-vs-accuracy)
  - [Applications](#applications)

- [Solutions](#solutions)
  - [Custom NumPy Implementation](#custom-numpy-implementation)
  - [Vectorized Implementation](#vectorized-implementation)

- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

---

## Problem Statement

### [Compute Multi-class Cross-Entropy Loss](https://www.deep-ml.com/problems/134)

Implement a function that computes the average cross-entropy loss for a batch of predictions in a multi-class classification problem.

The function receives:

- Predicted class probabilities.
- One-hot encoded true labels.

It returns the average categorical cross-entropy loss across the batch.

The predicted probabilities should represent a probability distribution over the classes.

For each sample:

\(\sum\_{c=1}^{C} p_c = 1\)

The implementation must also handle numerical instability by clipping probabilities with a small positive value `epsilon`.

---

## Example

### Input

```python
predicted_probs = [
    [0.7, 0.2, 0.1],
    [0.3, 0.6, 0.1]
]

true_labels = [
    [1, 0, 0],
    [0, 1, 0]
]
```

### Output

```text
0.4338
```

### Reasoning

For the first sample, the correct class has predicted probability:

\(p = 0.7\)

For the second sample, the correct class has predicted probability:

\(p = 0.6\)

Therefore, the batch loss is:

\(L = -\frac{\log(0.7)+\log(0.6)}{2}\)

Using the natural logarithm:

\(L \approx 0.4338\)

The loss is lower when the model assigns high probability to the correct class.

If the correct class receives a probability close to zero, the loss becomes very large.

---

# Learn: Multi-class Cross-Entropy Loss

## What is Cross-Entropy?

Cross-entropy is a loss function commonly used for classification problems.

It measures how different the predicted probability distribution is from the target distribution.

For multi-class classification, categorical cross-entropy compares:

- The predicted probability assigned to every class.
- The one-hot encoded target distribution.

The model is rewarded for assigning high probability to the correct class.

It is strongly penalized when the correct class receives a very small probability.

For a classification problem with `C` classes, the prediction for one sample can be represented as:

```text
[p(class 1), p(class 2), ..., p(class C)]
```

The probabilities normally satisfy:

\(p_c \geq 0\)

and:

\(\sum\_{c=1}^{C}p_c = 1\)

---

## Mathematical Definition

For one sample, categorical cross-entropy is:

\(L = -\sum\_{c=1}^{C}y_c\log(p_c)\)

where:

- `C` is the number of classes.
- `y_c` is the target value for class `c`.
- `p_c` is the predicted probability for class `c`.

For one-hot encoded labels:

\(y_c \in \{0,1\}\)

Exactly one class normally has:

\(y_c = 1\)

and all other classes have:

\(y_c = 0\)

Therefore, only the probability assigned to the correct class contributes to the loss.

If class `k` is correct:

\(L = -\log(p_k)\)

This is the key simplification behind the implementation.

---

## One-Hot Labels

Suppose there are three classes:

```text
cat
dog
bird
```

If the correct class is `dog`, the one-hot target is:

```text
[0, 1, 0]
```

Suppose the model predicts:

```text
[0.2, 0.7, 0.1]
```

The loss is:

\(L = -(0\log(0.2)+1\log(0.7)+0\log(0.1))\)

Therefore:

\(L = -\log(0.7)\)

The probabilities for the incorrect classes disappear from the final sum because their target values are zero.

---

## Why the Logarithm?

The logarithm creates a strong penalty for confidently incorrect predictions.

Consider the loss:

\(L = -\log(p)\)

Some examples are:

| Correct-class probability | Loss     |
| ------------------------- | -------- |
| `1.0`                     | `0.0000` |
| `0.9`                     | `0.1054` |
| `0.7`                     | `0.3567` |
| `0.5`                     | `0.6931` |
| `0.1`                     | `2.3026` |
| `0.01`                    | `4.6052` |
| `0.001`                   | `6.9078` |

As the correct-class probability approaches zero, the loss increases without bound.

This encourages a classifier not only to predict the correct class, but also to assign it a meaningful probability.

---

## Batch Cross-Entropy

For `N` samples and `C` classes, the average loss is:

\(L*{batch} = -\frac{1}{N}\sum*{n=1}^{N}\sum*{c=1}^{C}y*{n,c}\log(p\_{n,c})\)

where:

- `N` is the number of samples.
- `C` is the number of classes.
- `y_{n,c}` is the target value for sample `n` and class `c`.
- `p_{n,c}` is the predicted probability for sample `n` and class `c`.

Because the labels are one-hot encoded, each sample contributes only one logarithmic term.

If the correct class for sample `n` is `k`:

\(L*n = -\log(p*{n,k})\)

The batch loss is therefore:

\(L*{batch} = -\frac{1}{N}\sum*{n=1}^{N}\log(p\_{n,k_n})\)

---

## Numerical Stability

The logarithm is undefined at zero:

\(\log(0) \rightarrow -\infty\)

Therefore:

\(-\log(0) \rightarrow +\infty\)

A model can produce extremely small probabilities, so directly computing:

```python
np.log(predicted_probs)
```

can create numerical problems.

A common solution is to clip probabilities to a small positive value.

For example:

```python
epsilon = 1e-15
```

Then:

```python
np.clip(predicted_probs, epsilon, 1.0)
```

ensures that every probability passed to the logarithm satisfies:

\(p \geq \epsilon\)

The resulting computation is:

\(L = -\log(\max(p,\epsilon))\)

This prevents `log(0)`.

---

## Why Clip Only the Selected Probabilities?

Because one-hot labels contain exactly one `1` per sample, only the probability associated with the true class contributes to the loss.

For example:

```text
predicted = [0.7, 0.2, 0.1]
label     = [1,   0,   0]
```

The loss is:

\(-[1\log(0.7)+0\log(0.2)+0\log(0.1)]\)

Therefore, the implementation can first select the probabilities corresponding to the `1`s in the labels.

This gives:

```text
[0.7]
```

for the first sample.

For the complete batch:

```text
[0.7, 0.6]
```

These are the only values needed to calculate the loss.

---

## Probability Requirements

The predicted values should represent probabilities.

For every sample:

\(0 \leq p\_{n,c} \leq 1\)

and:

\(\sum*{c=1}^{C}p*{n,c}=1\)

Typically, a neural network produces logits first.

A softmax operation converts logits into probabilities:

\(p*c = \frac{e^{z_c}}{\sum*{j=1}^{C}e^{z_j}}\)

The cross-entropy function in this problem expects the probabilities after this normalization.

In many deep learning frameworks, however, the preferred API accepts logits directly because the framework can combine softmax and cross-entropy into a numerically stable operation.

---

## Cross-Entropy with Logits

Suppose the model produces logits:

```text
z = [2.0, 1.0, 0.1]
```

Softmax converts them into probabilities.

Conceptually:

```text
logits
  ↓
softmax
  ↓
probabilities
  ↓
cross-entropy
```

In PyTorch, one commonly uses:

```python
torch.nn.CrossEntropyLoss()
```

with raw logits rather than manually applying softmax first.

This is preferable because the implementation combines the necessary operations in a numerically stable way.

For this Deep-ML problem, however, the input is explicitly defined as predicted probabilities.

---

## Interpretation

Cross-entropy can be interpreted as the negative log-likelihood of the correct classes.

For a sample whose correct class probability is `p`:

\(L=-\log(p)\)

Therefore:

- High correct-class probability → low loss.
- Low correct-class probability → high loss.
- Correct probability of `1` → zero loss.
- Correct probability approaching `0` → very large loss.

A perfect prediction has:

\(p\_{correct}=1\)

and therefore:

\(L=-\log(1)=0\)

---

## Cross-Entropy vs Accuracy

Accuracy only checks whether the predicted class is correct.

Cross-entropy also considers confidence.

Consider two predictions for the same correct class:

```text
Prediction A: [0.51, 0.49]
Prediction B: [0.99, 0.01]
```

Both predict the first class correctly.

Their accuracies are identical for this sample.

But their losses differ:

\(L_A=-\log(0.51)\)

\(L_B=-\log(0.99)\)

Therefore, cross-entropy distinguishes between a barely confident correct prediction and a highly confident correct prediction.

This makes it useful during model training, where gradients need a continuous measure of prediction quality.

---

## Loss and Confidence

Cross-entropy strongly penalizes confident mistakes.

Suppose the true class is class `1`.

A prediction such as:

```text
[0.99, 0.01, 0.00]
```

has very small loss if class `1` is the first class.

But if the correct class is the second class:

```text
[0.99, 0.01, 0.00]
```

then the loss becomes:

\(-\log(0.01) \approx 4.6052\)

The model is not merely wrong.

It is highly confident that it is wrong.

Cross-entropy therefore provides a much stronger training signal for confident incorrect predictions than simple classification error.

---

## Important Characteristics

- Cross-entropy is commonly used for classification.
- Categorical cross-entropy is appropriate for multi-class one-hot targets.
- The loss is based on predicted probabilities.
- Only the correct-class probability contributes for one-hot labels.
- The logarithm penalizes small correct-class probabilities strongly.
- Perfect confidence in the correct class gives zero loss.
- Probabilities should form a valid distribution.
- Clipping prevents numerical issues from `log(0)`.
- The average batch loss is usually used during training.
- Cross-entropy is differentiable with respect to the prediction probabilities away from the clipping boundary.

---

## Applications

Cross-entropy is widely used in:

- Multi-class image classification.
- Text classification.
- Neural language models.
- Object recognition.
- Speech classification.
- Document classification.
- Sequence classification.
- Neural network training.
- Multi-class probability estimation.

For language modeling, the same fundamental loss is often used at every token position.

If a language model predicts the next token distribution:

```text
P(token_1), P(token_2), ..., P(token_V)
```

cross-entropy measures how much probability the model assigned to the actual next token.

---

> 💡 **Important Note**
>
> Do not confuse cross-entropy with classification accuracy.
>
> Accuracy depends only on the final predicted class.
>
> Cross-entropy uses the entire probability assigned to the correct class and therefore provides a richer training signal.

---

# Solutions

## Custom NumPy Implementation

```python
import numpy as np

def compute_cross_entropy_loss(
    predicted_probs: np.ndarray,
    true_labels: np.ndarray,
    epsilon=1e-15
) -> float:
    selected = predicted_probs[true_labels.astype(bool)]
    selected = np.clip(selected, epsilon, 1.0)
    return -np.mean(np.log(selected))
```

---

## Vectorized Implementation

The same calculation can be expressed directly using element-wise multiplication.

```python
import numpy as np

def compute_cross_entropy_loss(
    predicted_probs: np.ndarray,
    true_labels: np.ndarray,
    epsilon=1e-15
) -> float:
    clipped_probs = np.clip(predicted_probs, epsilon, 1.0)
    return -np.mean(np.sum(true_labels * np.log(clipped_probs), axis=1))
```

The second implementation follows the mathematical definition more directly:

\(L*{batch} = -\frac{1}{N}\sum*{n=1}^{N}\sum*{c=1}^{C}y*{n,c}\log(p\_{n,c})\)

The first implementation takes advantage of one-hot encoding and directly extracts the correct-class probabilities.

Both approaches produce the same result when the labels are valid one-hot vectors.

---

# Code Explanation

## 1. Identify the Correct-Class Probabilities

The key observation is that the labels are one-hot encoded.

For:

```python
true_labels = np.array([
    [1, 0, 0],
    [0, 1, 0]
])
```

the boolean representation becomes:

```text
[True, False, False]
[False, True, False]
```

The expression:

```python
true_labels.astype(bool)
```

creates this boolean mask.

---

## 2. Select the Correct Predictions

The code uses:

```python
selected = predicted_probs[true_labels.astype(bool)]
```

Suppose:

```text
predicted_probs =
[[0.7, 0.2, 0.1],
 [0.3, 0.6, 0.1]]
```

and:

```text
true_labels =
[[1, 0, 0],
 [0, 1, 0]]
```

The selected values are:

```text
[0.7, 0.6]
```

These are exactly the probabilities assigned to the true classes.

---

## 3. Clip the Probabilities

The selected probabilities are clipped using:

```python
selected = np.clip(selected, epsilon, 1.0)
```

For:

```python
epsilon = 1e-15
```

any value below `1e-15` becomes `1e-15`.

The upper bound is `1.0`.

Therefore:

\(p\_{selected}\in[\epsilon,1]\)

This guarantees that the logarithm does not receive zero.

---

## 4. Apply the Logarithm

The loss for each sample is:

\(-\log(p\_{correct})\)

The code computes:

```python
np.log(selected)
```

For the example:

```text
selected = [0.7, 0.6]
```

the logarithms are approximately:

```text
[-0.3567, -0.5108]
```

---

## 5. Negate and Average

The final operation is:

```python
return -np.mean(np.log(selected))
```

First, the logarithms are averaged:

\(\frac{\log(0.7)+\log(0.6)}{2}\)

Then the sign is reversed:

\(-\frac{\log(0.7)+\log(0.6)}{2}\)

giving approximately:

```text
0.4338
```

---

## 6. Why `np.mean()` Works Here

The selected array contains exactly one correct-class probability per sample.

Therefore, if there are `N` samples:

```text
selected.shape = (N,)
```

Taking the mean is equivalent to:

\(\frac{1}{N}\sum\_{n=1}^{N}L_n\)

This gives the average loss across the batch.

---

## 7. Understanding the Vectorized Alternative

The direct mathematical implementation is:

```python
clipped_probs = np.clip(predicted_probs, epsilon, 1.0)
return -np.mean(
    np.sum(
        true_labels * np.log(clipped_probs),
        axis=1
    )
)
```

The multiplication:

```python
true_labels * np.log(clipped_probs)
```

zeroes out the contributions from incorrect classes.

For example:

```text
true_labels = [1, 0, 0]
log_probs   = [a, b, c]
```

produces:

```text
[a, 0, 0]
```

Summing across classes gives:

```text
a
```

which is exactly:

\(\sum\_{c=1}^{C}y_c\log(p_c)\)

The outer mean then averages the loss across samples.

---

## 8. Why the Two Implementations Are Equivalent

The custom implementation uses:

```python
predicted_probs[true_labels.astype(bool)]
```

to explicitly select the correct probabilities.

The vectorized mathematical implementation uses:

```python
true_labels * np.log(predicted_probs)
```

to eliminate incorrect-class contributions.

Because the target is one-hot:

\(\sum*{c=1}^{C}y_c\log(p_c)=\log(p*{correct})\)

Therefore, both approaches calculate the same quantity.

The first is concise.

The second makes the mathematical definition more explicit.

---

## 9. Important Edge Case: Probability of Zero

Consider:

```python
predicted_probs = [[0.0, 1.0, 0.0]]
true_labels = [[1, 0, 0]]
```

Without clipping:

```python
np.log(0.0)
```

produces negative infinity.

The loss would therefore become positive infinity.

With:

```python
epsilon = 1e-15
```

the selected probability becomes:

```text
1e-15
```

and the loss remains finite:

\(-\log(10^{-15}) \approx 34.5388\)

The value is extremely large, correctly reflecting the model's near-zero confidence in the correct class, while avoiding an infinite numerical result.

---

## 10. Shape Requirements

For `N` samples and `C` classes:

```text
predicted_probs.shape = (N, C)
true_labels.shape     = (N, C)
```

Both arrays must have the same shape.

Each row of `true_labels` should normally contain exactly one `1`.

For example:

```text
[1, 0, 0]
[0, 1, 0]
[0, 0, 1]
```

The corresponding probability rows should contain the model's probability distribution over the same classes.

---

## 11. Important Difference from Binary Cross-Entropy

Multi-class categorical cross-entropy uses:

\(L = -\sum_c y_c\log(p_c)\)

For one-hot labels, only one class contributes.

Binary cross-entropy has a different form:

\(L = -[y\log(p)+(1-y)\log(1-p)]\)

Binary cross-entropy accounts for both the positive and negative class terms.

Categorical cross-entropy instead handles a probability distribution over multiple mutually exclusive classes.

---

## 12. Practical Deep Learning Equivalent

In PyTorch, if the model outputs logits rather than probabilities, the standard approach is:

```python
import torch
import torch.nn as nn

criterion = nn.CrossEntropyLoss()

logits = torch.tensor([
    [2.0, 0.5, -1.0],
    [0.2, 2.0, -0.5]
])

targets = torch.tensor([0, 1])

loss = criterion(logits, targets)
```

Notice that the targets are class indices rather than one-hot vectors.

The PyTorch loss internally performs the numerically stable equivalent of log-softmax followed by negative log-likelihood.

Therefore, this:

```python
criterion(logits, targets)
```

is generally preferred over manually doing:

```python
probabilities = torch.softmax(logits, dim=1)
loss = -torch.log(probabilities)
```

for model training.

---

# Time & Space Complexity

Let:

- `N` be the number of samples.
- `C` be the number of classes.

The predicted probability matrix contains:

\(N \times C\)

elements.

Selecting the correct-class probabilities requires examining the batch:

\(O(NC)\)

The clipping operation is:

\(O(N)\)

after selection.

The logarithm and mean operate on `N` selected probabilities:

\(O(N)\)

Therefore, the total time complexity is:

\(O(NC)\)

For the direct vectorized implementation, the entire probability matrix is processed, so the complexity is also:

\(O(NC)\)

The input itself occupies:

\(O(NC)\)

space.

The selected correct-class probabilities require:

\(O(N)\)

additional space.

The total auxiliary space of the selection-based implementation is:

\(O(N)\)

excluding the input arrays.

For the fully vectorized mathematical implementation, intermediate arrays can contain the entire `(N, C)` matrix, resulting in:

\(O(NC)\)

temporary space depending on how NumPy evaluates the expressions.

| Complexity                 | Value                                            |
| -------------------------- | ------------------------------------------------ |
| Time                       | **O(NC)**                                        |
| Auxiliary Space            | **O(N)** for selected-probability implementation |
| Vectorized Temporary Space | **O(NC)**                                        |

Where:

- **N** = number of samples.
- **C** = number of classes.
