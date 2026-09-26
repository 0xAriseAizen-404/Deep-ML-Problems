# Implement Binary Cross-Entropy Loss (Easy, Deep Learning)

## Table of Contents

* [Problem Statement](#problem-statement)
* [Example](#example)
* [Learn: Binary Cross-Entropy Loss](#learn-binary-cross-entropy-loss)

  * [What is Binary Cross-Entropy?](#what-is-binary-cross-entropy)
  * [Mathematical Definition](#mathematical-definition)
  * [Understanding the Two Cases](#understanding-the-two-cases)
  * [Intuition](#intuition)
  * [Effect of Prediction Confidence](#effect-of-prediction-confidence)
  * [Mean Binary Cross-Entropy](#mean-binary-cross-entropy)
  * [Numerical Stability](#numerical-stability)
  * [Why Clipping is Necessary](#why-clipping-is-necessary)
  * [Connection to Maximum Likelihood](#connection-to-maximum-likelihood)
  * [Connection to Information Theory](#connection-to-information-theory)
  * [BCE and Classification](#bce-and-classification)
  * [BCE vs Accuracy](#bce-vs-accuracy)
  * [BCE and Sigmoid](#bce-and-sigmoid)
  * [Use Cases](#use-cases)
  * [Important Considerations](#important-considerations)
* [Solutions](#solutions)

  * [Custom Python Implementation](#custom-python-implementation)
  * [NumPy Implementation](#numpy-implementation)
  * [PyTorch Implementation](#pytorch-implementation)
* [Code Explanation](#code-explanation)
* [Time & Space Complexity](#time--space-complexity)

---

## Problem Statement

### [Implement Binary Cross-Entropy Loss](https://www.deep-ml.com/problems/263)

Implement the binary cross-entropy loss function for binary classification.

The function receives:

* A list of true binary labels.
* A list of predicted probabilities.
* An optional `epsilon` used for numerical stability.

Each true label must be either:

\(y\in\{0,1\}\)

Each prediction represents the model's estimated probability of the positive class:

\(0\leq p\leq1\)

For each sample, binary cross-entropy is:

\(L(y,p)=-[y\log(p)+(1-y)\log(1-p)]\)

For `N` samples, the required output is the mean loss:

\(L=-\frac{1}{N}\sum_{i=1}^{N}[y_i\log(p_i)+(1-y_i)\log(1-p_i)]\)

The implementation must clip predictions before taking logarithms to avoid numerical problems at exactly `0` and `1`.

---

## Example

### Input

```python id="p5i3qo"
y_true = [1, 0, 1, 0]
y_pred = [0.9, 0.1, 0.8, 0.2]
```

### Output

```text id="x3r5ku"
0.1643
```

### Reasoning

For a positive label:

\(y=1\)

the loss becomes:

\(L=-\log(p)\)

For the first sample:

\(L_1=-\log(0.9)\approx0.1054\)

For the second sample:

\(y=0\)

so:

\(L_2=-\log(1-0.1)=-\log(0.9)\approx0.1054\)

For the third sample:

\(L_3=-\log(0.8)\approx0.2231\)

For the fourth sample:

\(L_4=-\log(1-0.2)=-\log(0.8)\approx0.2231\)

Therefore:

\(L=\frac{0.1054+0.1054+0.2231+0.2231}{4}\)

and:

\(L\approx0.1643\)

---

# Learn: Binary Cross-Entropy Loss

## What is Binary Cross-Entropy?

Binary Cross-Entropy, commonly abbreviated as BCE, is a loss function used when the target represents a binary outcome.

Examples include:

```text
Spam       → 0 or 1
Fraud      → 0 or 1
Disease    → 0 or 1
Pass       → 0 or 1
```

The model produces a probability rather than simply producing a hard class.

For example:

```text
p = 0.90
```

can be interpreted as the model assigning a `90%` probability to the positive class.

BCE evaluates how well that probability agrees with the actual label.

---

# Mathematical Definition

For one sample:

\(L(y,p)=-[y\log(p)+(1-y)\log(1-p)]\)

where:

* $y$ is the true binary label.
* $p$ is the predicted probability of class `1`.
* $\log$ is usually the natural logarithm.

Because $y$ is either `0` or `1`, one of the two terms becomes zero.

This allows the same formula to represent both possible classes.

---

## Case 1: Positive Class

Suppose:

\(y=1\)

Then:

\(L(1,p)=-[1\log(p)+0\log(1-p)]\)

Therefore:

\(L(1,p)=-\log(p)\)

The loss depends only on the probability assigned to the positive class.

If:

\(p\rightarrow1\)

then:

\(-\log(p)\rightarrow0\)

If:

\(p\rightarrow0\)

then:

\(-\log(p)\rightarrow\infty\)

Therefore, confidently predicting a positive example as negative produces a very large penalty.

---

## Case 2: Negative Class

Suppose:

\(y=0\)

Then:

\(L(0,p)=-[0\log(p)+1\log(1-p)]\)

Therefore:

\(L(0,p)=-\log(1-p)\)

If:

\(p\rightarrow0\)

then:

\(-\log(1-p)\rightarrow0\)

If:

\(p\rightarrow1\)

then:

\(-\log(1-p)\rightarrow\infty\)

Thus, confidently predicting a negative example as positive produces a very large penalty.

---

# Intuition

BCE rewards predictions that assign high probability to the correct class.

Consider a positive example:

\(y=1\)

The loss is:

\(L=-\log(p)\)

Some values are:

| Predicted $p$ |   Loss |
| ------------: | -----: |
|          0.99 | 0.0101 |
|          0.90 | 0.1054 |
|          0.70 | 0.3567 |
|          0.50 | 0.6931 |
|          0.10 | 2.3026 |
|          0.01 | 4.6052 |

The important property is that the penalty grows rapidly as the model becomes confidently wrong.

---

# Effect of Prediction Confidence

BCE does not merely ask whether the predicted class is correct.

It also evaluates the confidence of the prediction.

Suppose the true label is:

\(y=1\)

Consider two predictions:

```text
p = 0.51
p = 0.99
```

Both produce the correct class after thresholding at `0.5`.

However:

\(-\log(0.51)\approx0.6733\)

while:

\(-\log(0.99)\approx0.0101\)

The second prediction receives much lower loss because it assigns much higher probability to the correct class.

---

## Confident Mistakes

Suppose:

\(y=1\)

but:

\(p=0.001\)

Then:

\(L=-\log(0.001)\)

and:

\(L\approx6.9078\)

The loss is large because the model was extremely confident in the wrong direction.

This is one of the major differences between BCE and simple classification accuracy.

---

# Mean Binary Cross-Entropy

For `N` samples:

\(L=-\frac1N\sum_{i=1}^{N}[y_i\log(p_i)+(1-y_i)\log(1-p_i)]\)

The individual sample losses are computed first.

Then they are averaged.

For:

```text
Losses = [0.1, 0.2, 0.3, 0.4]
```

the mean is:

\(L=\frac{0.1+0.2+0.3+0.4}{4}\)

Therefore:

\(L=0.25\)

Taking the mean ensures that the magnitude of the loss does not directly depend on the number of samples in the batch.

---

# Numerical Stability

The logarithm creates an important numerical problem.

Mathematically:

\(\log(0)=-\infty\)

Therefore, if:

\(p=0\)

the positive-class term can become infinite.

Similarly:

\(\log(1-p)=\log(0)=-\infty\)

when:

\(p=1\)

This can produce infinite losses or unstable numerical behavior.

---

## Clipping the Prediction

The solution is to restrict predictions to a safe interval:

\([\epsilon,1-\epsilon]\)

The implementation uses:

\(\epsilon=10^{-15}\)

The clipped probability is:

\(p_{\text{clipped}}=\operatorname{clip}(p,\epsilon,1-\epsilon)\)

Therefore:

```text
p < epsilon
```

becomes:

```text
epsilon
```

and:

```text
p > 1 - epsilon
```

becomes:

```text
1 - epsilon
```

---

## Why Both Ends Must Be Clipped

It is not enough to protect only:

\(\log(p)\)

because BCE also contains:

\(\log(1-p)\)

If:

\(p=1\)

then:

\(1-p=0\)

and therefore:

\(\log(1-p)=\log(0)\)

So predictions must be restricted away from both endpoints.

The correct clipping interval is:

\([\epsilon,1-\epsilon]\)

---

# Example of Clipping

Suppose:

\(\epsilon=10^{-15}\)

and:

\(p=0\)

Then:

\(p_{\text{clipped}}=10^{-15}\)

instead of zero.

Likewise, if:

\(p=1\)

then:

\(p_{\text{clipped}}=1-10^{-15}\)

This prevents the logarithm from receiving an exact zero.

---

# Connection to Maximum Likelihood

BCE has a direct statistical interpretation.

A Bernoulli random variable has probability mass function:

\(P(y|p)=p^y(1-p)^{1-y}\)

For `N` independent observations:

\(P(y_1,\ldots,y_N)=\prod_{i=1}^{N}p_i^{y_i}(1-p_i)^{1-y_i}\)

Taking the logarithm gives the log-likelihood:

\(\log P=\sum_{i=1}^{N}[y_i\log(p_i)+(1-y_i)\log(1-p_i)]\)

Machine learning commonly minimizes a loss rather than maximizing likelihood.

Therefore, taking the negative average log-likelihood gives BCE:

\(L=-\frac1N\sum_{i=1}^{N}[y_i\log(p_i)+(1-y_i)\log(1-p_i)]\)

Thus, minimizing BCE is equivalent to maximizing the likelihood under a Bernoulli model.

---

# Connection to Information Theory

BCE is closely related to cross-entropy from information theory.

For a binary target distribution:

\(q=[y,1-y]\)

and predicted distribution:

\(p=[p,1-p]\)

the cross-entropy is:

\(H(q,p)=-\sum_iq_i\log(p_i)\)

Substituting the two binary probabilities produces:

\(H(q,p)=-[y\log(p)+(1-y)\log(1-p)]\)

which is exactly BCE.

Therefore, BCE measures the discrepancy between the true binary distribution and the predicted probability distribution.

---

# BCE and Classification

BCE is commonly used when the model predicts one binary outcome.

A typical neural-network pipeline is:

```text
Input
  ↓
Neural Network
  ↓
Logit
  ↓
Sigmoid
  ↓
Probability
  ↓
BCE Loss
```

The sigmoid converts an unrestricted real-valued logit into a probability:

\(p=\sigma(z)=\frac{1}{1+e^{-z}}\)

The resulting value satisfies:

\(0<p<1\)

and can therefore be used in the BCE formula.

---

# BCE and Sigmoid

Suppose the neural network produces:

\(z\in\mathbb{R}\)

The sigmoid function produces:

\(p=\sigma(z)\)

where:

\(\sigma(z)=\frac{1}{1+e^{-z}}\)

Then BCE is:

\(L=-[y\log(\sigma(z))+(1-y)\log(1-\sigma(z))]\)

This combination is extremely common in binary classification.

However, implementations should preferably combine the sigmoid and BCE computations into a numerically stable operation when possible.

---

# BCE With Logits

Deep-learning frameworks often provide a loss that accepts logits directly.

For PyTorch, this is:

```python
torch.nn.BCEWithLogitsLoss()
```

Instead of explicitly calculating:

```python
p = sigmoid(z)
loss = BCE(y, p)
```

the combined loss computes the equivalent operation more stably.

This avoids unnecessary numerical problems caused by explicitly materializing probabilities close to zero or one.

The conceptual pipeline remains:

\(z\rightarrow\sigma(z)\rightarrow\text{BCE}\)

but the implementation combines the operations.

---

# BCE Gradient

For a predicted probability $p$, BCE is:

\(L=-[y\log(p)+(1-y)\log(1-p)]\)

Differentiating with respect to $p$:

\(\frac{\partial L}{\partial p}=-\frac{y}{p}+\frac{1-y}{1-p}\)

Combining the terms:

\(\frac{\partial L}{\partial p}=\frac{p-y}{p(1-p)}\)

This gradient becomes very large when the prediction approaches the wrong endpoint.

When BCE is combined with a sigmoid output, the derivative with respect to the logit $z$ simplifies significantly:

\(\frac{\partial L}{\partial z}=p-y\)

This simple gradient is one reason sigmoid plus BCE is such a fundamental combination in binary classification.

---

# BCE vs Accuracy

Accuracy and BCE measure different things.

Accuracy considers only whether the predicted class is correct.

For a threshold of `0.5`:

\(\hat y=\mathbb{1}[p\geq0.5]\)

BCE instead uses the actual probability.

Consider a positive example:

```text
Prediction A = 0.51
Prediction B = 0.99
```

Both produce:

```text
Predicted class = 1
```

and therefore the same accuracy contribution.

But their BCE losses are very different.

\(-\log(0.51)\approx0.6733\)

while:

\(-\log(0.99)\approx0.0101\)

Thus, BCE provides much more information about model confidence.

---

# BCE and Class Imbalance

BCE treats each sample according to its contribution to the average loss.

If one class occurs much more frequently than another, the majority class can dominate the overall loss.

For example:

```text
Positive samples → 5%
Negative samples → 95%
```

A model that predicts almost everything as negative can achieve high accuracy while performing poorly on positive examples.

In imbalanced datasets, alternatives or extensions may be appropriate, such as:

* Weighted BCE
* Focal Loss
* Resampling
* Class-balanced training

The appropriate method depends on the task and evaluation objective.

---

# Use Cases

## Binary Classification

BCE is widely used for binary classification tasks.

Examples include:

* Spam detection.
* Fraud detection.
* Disease classification.
* Customer churn prediction.
* Defect detection.
* Click prediction.

---

## Multi-Label Classification

BCE can also be used for multi-label classification.

Suppose an image can contain multiple independent labels:

```text
cat = 1
dog = 0
car = 1
tree = 0
```

Each label can have its own sigmoid probability.

The BCE losses for individual labels can then be averaged.

This differs from ordinary multi-class classification, where exactly one class is typically selected.

---

# Important Considerations

* BCE expects binary targets.
* Predictions represent probabilities of the positive class.
* Predictions should lie in `[0,1]`.
* The logarithm requires protection against zero.
* Both `p` and `1-p` must be protected.
* Clipping should use `[epsilon,1-epsilon]`.
* BCE heavily penalizes confident incorrect predictions.
* Mean BCE is independent of batch size in scale.
* Accuracy and BCE measure different properties.
* For neural networks, logits plus a numerically stable BCE-with-logits implementation are generally preferred.
* For heavily imbalanced data, standard BCE may require weighting or another loss strategy.

---

> 💡 **Important Note**
>
> Do not confuse probability clipping with rescaling.
>
> Clipping only restricts extreme values:
>
> \(p\rightarrow\operatorname{clip}(p,\epsilon,1-\epsilon)\)
>
> It does not redistribute or normalize the predictions. This is exactly what is needed for numerical stability before taking logarithms.

---

# Solutions

## Custom Python Implementation

```python id="v8y3fk"
import math

def binary_cross_entropy(
    y_true: list[float],
    y_pred: list[float],
    epsilon: float = 1e-15
) -> float:
    """
    Compute binary cross-entropy loss.
    """
    loss = 0.0

    for y, p in zip(y_true, y_pred):
        p = max(epsilon, min(p, 1 - epsilon))
        loss += y * math.log(p) + (1 - y) * math.log(1 - p)

    return -loss / len(y_true)
```

---

## NumPy Implementation

A vectorized NumPy version performs the same calculation without an explicit Python loop.

```python id="x7f1pq"
import numpy as np

def binary_cross_entropy(
    y_true: list[float],
    y_pred: list[float],
    epsilon: float = 1e-15
) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    loss = -(
        y_true * np.log(y_pred)
        + (1 - y_true) * np.log(1 - y_pred)
    )

    return float(np.mean(loss))
```

The vectorized expression directly implements:

\(L=-\frac1N\sum_{i=1}^{N}[y_i\log(p_i)+(1-y_i)\log(1-p_i)]\)

---

# PyTorch Implementation

For a neural-network implementation, PyTorch provides BCE directly.

```python id="e9n7wu"
import torch

loss_fn = torch.nn.BCELoss()

y_pred = torch.tensor([0.9, 0.1, 0.8, 0.2])
y_true = torch.tensor([1.0, 0.0, 1.0, 0.0])

loss = loss_fn(y_pred, y_true)
```

When working with logits rather than probabilities, the preferred formulation is:

```python id="1w2qz8"
loss_fn = torch.nn.BCEWithLogitsLoss()

logits = model(x)
loss = loss_fn(logits, y_true)
```

The second approach combines sigmoid and BCE internally for improved numerical stability.

---

# Code Explanation

## 1. Initialize the Accumulator

The implementation starts with:

```python id="c0s6sv"
loss = 0.0
```

This variable stores the sum of the log-likelihood terms across all samples.

Because BCE is defined as the negative mean log-likelihood, the final result will negate and divide this value.

---

## 2. Iterate Through Labels and Predictions

```python id="2x5qbc"
for y, p in zip(y_true, y_pred):
```

Each iteration processes one training example.

The two values are:

```text
y → true binary label
p → predicted probability
```

For example:

```text
y = 1
p = 0.9
```

---

## 3. Clip the Probability

The implementation uses:

```python id="m4qk6n"
p = max(epsilon, min(p, 1 - epsilon))
```

This is equivalent to:

\(p\leftarrow\operatorname{clip}(p,\epsilon,1-\epsilon)\)

The inner `min` ensures that:

\(p\leq1-\epsilon\)

The outer `max` ensures that:

\(p\geq\epsilon\)

Therefore:

\(\epsilon\leq p\leq1-\epsilon\)

---

## 4. Compute the Log-Likelihood Term

The implementation calculates:

```python id="8x5gqa"
y * math.log(p) + (1 - y) * math.log(1 - p)
```

This corresponds directly to:

\(y\log(p)+(1-y)\log(1-p)\)

When:

\(y=1\)

the second term disappears.

When:

\(y=0\)

the first term disappears.

---

## 5. Accumulate the Samples

Each sample contributes its log-likelihood term to:

```python id="y3c3lh"
loss
```

After processing all `N` samples:

\(\text{loss}=\sum_{i=1}^{N}[y_i\log(p_i)+(1-y_i)\log(1-p_i)]\)

At this point, the value is negative or zero because logarithms of probabilities in `(0,1)` are non-positive.

---

## 6. Negate the Sum

BCE is defined as the negative log-likelihood.

Therefore:

```python id="e9s6md"
-loss
```

converts the accumulated log-likelihood into a positive loss.

---

## 7. Compute the Mean

Finally:

```python id="2p5c2w"
return -loss / len(y_true)
```

divides by the number of samples.

Therefore:

\(L=-\frac{\text{loss}}{N}\)

which produces the mean BCE.

---

# Why the Commented-Out Approach is Incorrect

The original solution also contained a commented-out `clip` function that rescales predictions according to their own minimum and maximum.

That approach is not appropriate for BCE.

The intended operation is clipping:

\(p_{\text{clipped}}=\operatorname{clip}(p,\epsilon,1-\epsilon)\)

not min-max rescaling:

\(p'=\frac{p-\min(p)}{\max(p)-\min(p)}\)

These operations have fundamentally different meanings.

---

## Why Rescaling Changes the Model's Probabilities

Suppose the model predicts:

```text
[0.2, 0.4, 0.6, 0.8]
```

Min-max rescaling to `[epsilon, 1-epsilon]` would approximately produce:

```text
[epsilon, 0.333, 0.667, 1-epsilon]
```

This changes the actual probability values.

The model's original confidence information has been altered.

BCE should evaluate the probabilities produced by the model, not normalize them based on the current batch.

Therefore, clipping is the correct operation.

---

# Example With Extreme Predictions

Consider:

```python id="6w0e1h"
y_true = [1, 0]
y_pred = [0.0, 1.0]
```

Without clipping:

```text
log(0)
```

appears in the BCE calculation.

Mathematically:

\(\log(0)=-\infty\)

With:

\(\epsilon=10^{-15}\)

the predictions become approximately:

```text
[1e-15, 1 - 1e-15]
```

The resulting loss is finite.

The loss remains extremely large because both predictions are confidently wrong, but it no longer produces an undefined logarithm.

---

# Empty Input Consideration

The implementation assumes that `y_true` and `y_pred` contain at least one sample.

If:

```python id="0u1x2h"
y_true = []
y_pred = []
```

then:

```python id="5x9s1j"
len(y_true)
```

is zero.

The final division would therefore divide by zero.

For production code, an explicit validation could be added:

```python id="y3l7fc"
if len(y_true) == 0:
    raise ValueError("Input lists must not be empty.")
```

For the Deep-ML problem, the expected input is assumed to contain valid samples.

---

# Time & Space Complexity

Let `N` be the number of samples.

The implementation processes each sample exactly once.

For every sample it performs:

* One probability clipping operation.
* Two logarithm evaluations in the general expression.
* A constant number of arithmetic operations.

Therefore:

\(T(N)=O(N)\)

The implementation stores only the accumulated loss and a few scalar values.

Therefore:

\(S(N)=O(1)\)

excluding the input lists.

| Complexity      | Value    |
| --------------- | -------- |
| Time            | **O(N)** |
| Auxiliary Space | **O(1)** |

For the NumPy vectorized implementation, the arithmetic is still:

\(O(N)\)

but NumPy may allocate temporary arrays containing the per-sample losses, giving:

\(O(N)\)

additional memory depending on the expression evaluation and implementation.