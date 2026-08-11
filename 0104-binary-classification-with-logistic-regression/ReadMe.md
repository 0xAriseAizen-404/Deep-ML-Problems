# Binary Classification with Logistic Regression (Easy, Machine Learning)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Binary Classification with Logistic Regression](#learn-binary-classification-with-logistic-regression)
- [Solution](#solution)
  - [Custom Implementation](#custom-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

---

## Problem Statement

### [Binary Classification with Logistic Regression](https://www.deep-ml.com/problems/104)

Implement the prediction function for **binary classification using Logistic Regression**.

The function receives a batch of input samples, model weights, and a bias. It should:

- Compute the linear combination of features and weights.
- Apply the sigmoid function to obtain class probabilities.
- Convert probabilities into binary predictions using a threshold of `0.5`.
- Return predictions as `0` or `1`.

---

## Example

### Input

```python
predict_logistic(
    np.array([[1, 1], [2, 2], [-1, -1], [-2, -2]]),
    np.array([1, 1]),
    0
)
```

### Output

```text
[1 1 0 0]
```

### Reasoning

For each sample, Logistic Regression first computes

$$
z=Xw+b
$$

For the first sample $[1,1]$:

$$
z=(1)(1)+(1)(1)+0=2
$$

The sigmoid probability is

$$
\sigma(2)=\frac{1}{1+e^{-2}}\approx0.8808
$$

Since the probability is greater than `0.5`, the prediction is `1`.

For $[2,2]$:

$$
z=(2)(1)+(2)(1)+0=4
$$

$$
\sigma(4)\approx0.9820
$$

Therefore, the prediction is `1`.

For $[-1,-1]$:

$$
z=(-1)(1)+(-1)(1)+0=-2
$$

$$
\sigma(-2)\approx0.1192
$$

Therefore, the prediction is `0`.

For $[-2,-2]$:

$$
z=(-2)(1)+(-2)(1)+0=-4
$$

$$
\sigma(-4)\approx0.0180
$$

Therefore, the final predictions are

```text
[1 1 0 0]
```

---

## Learn: Binary Classification with Logistic Regression

### What is it?

**Logistic Regression** is a supervised machine learning algorithm commonly used for **binary classification**.

Instead of directly predicting a class label, Logistic Regression first computes a real-valued score and transforms it into a probability using the **sigmoid function**.

The probability represents how likely the sample belongs to class `1`.

A threshold is then applied to convert the probability into a binary prediction.

The basic flow is:

```text
Input Features
      ↓
Linear Combination
      ↓
Sigmoid
      ↓
Probability
      ↓
Threshold
      ↓
Class 0 or 1
```

---

### Linear Model

Given an input vector $x$ and weight vector $w$, the model first computes

$$
z=w^Tx+b
$$

For $D$ features, this can be written as

$$
z=\sum_{j=1}^{D}w_jx_j+b
$$

where

- $x_j$ is the $j$-th feature.
- $w_j$ is the corresponding model weight.
- $b$ is the bias.
- $z$ is called the **logit** or linear score.

For a batch of samples, the computation can be vectorized as

$$
z=Xw+b
$$

where $X$ has shape $N\times D$.

---

### Sigmoid Function

The linear score $z$ can take any value from negative infinity to positive infinity.

To convert this score into a probability, Logistic Regression applies the sigmoid function:

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

The sigmoid maps every real-valued input into the interval

$$
0<\sigma(z)<1
$$

Therefore, its output can be interpreted as the estimated probability of class `1`.

---

### Sigmoid Behavior

For a large positive value of $z$:

$$
z\rightarrow\infty
$$

and

$$
\sigma(z)\rightarrow1
$$

For a large negative value of $z$:

$$
z\rightarrow-\infty
$$

and

$$
\sigma(z)\rightarrow0
$$

At $z=0$:

$$
\sigma(0)=\frac{1}{2}=0.5
$$

Therefore, `0.5` naturally becomes the standard classification threshold.

---

### From Probability to Prediction

After calculating the probability, a threshold is used to determine the class.

For the standard threshold of `0.5`:

$$
\hat{y}=
\begin{cases}
1 & \text{if } \sigma(z)\geq0.5\
0 & \text{if } \sigma(z)<0.5
\end{cases}
$$

In the implementation:

```python
(probs >= 0.5).astype(int)
```

converts the probabilities directly into integer class predictions.

---

### Why the Threshold is 0.5

The sigmoid equals `0.5` when the logit is zero:

$$
\sigma(z)=0.5
$$

which occurs at

$$
z=0
$$

Therefore,

$$
z\geq0\Rightarrow\hat{y}=1
$$

and

$$
z<0\Rightarrow\hat{y}=0
$$

For a standard Logistic Regression classifier, the `0.5` probability threshold corresponds directly to the decision boundary $w^Tx+b=0$.

---

### Decision Boundary

The decision boundary occurs when

$$
\sigma(z)=0.5
$$

which means

$$
z=0
$$

Since

$$
z=w^Tx+b
$$

the decision boundary is

$$
w^Tx+b=0
$$

For two features, this represents a line.

For higher-dimensional feature spaces, it represents a hyperplane.

---

### Vectorized Prediction

Suppose there are $N$ samples and $D$ features.

Instead of calculating each sample separately, NumPy can compute all logits at once:

```python
logits = X @ weights + bias
```

If

- $X$ has shape $(N,D)$
- $w$ has shape $(D,)$

then

$$
Xw
$$

has shape

$$
(N,)
$$

The same bias is added to every sample.

This avoids explicit Python loops and allows NumPy to perform the computation efficiently.

---

### Numerical Stability

The sigmoid contains an exponential term:

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

For extremely large values of $|z|$, directly computing the exponential can cause numerical overflow.

For example, when $z$ is a very large negative number:

$$
-z\rightarrow\infty
$$

and

$$
e^{-z}
$$

can become too large for floating-point representation.

A common simple approach is to clip the logits:

```python
z = np.clip(z, -500, 500)
```

This limits the range before calculating the exponential.

A more robust implementation can use a numerically stable sigmoid formulation that treats positive and negative logits separately.

---

### Probability Interpretation

The sigmoid output can be interpreted as

$$
P(y=1\mid x)=\sigma(w^Tx+b)
$$

Consequently,

$$
P(y=0\mid x)=1-\sigma(w^Tx+b)
$$

For example, if

$$
P(y=1\mid x)=0.88
$$

the model predicts class `1` using the standard `0.5` threshold.

If

$$
P(y=1\mid x)=0.12
$$

the model predicts class `0`.

---

### Characteristics / Key Points

- Logistic Regression is primarily used for binary classification.
- It computes a linear score before applying sigmoid.
- The sigmoid converts logits into probabilities.
- Probability values lie strictly between `0` and `1`.
- A threshold converts probabilities into class labels.
- The standard threshold is `0.5`.
- The decision boundary occurs at $w^Tx+b=0$.
- Vectorized matrix multiplication makes prediction efficient.
- Numerical stability is important when computing exponentials.
- The model produces probabilities as well as class predictions.
- The weights determine the influence of individual features.
- The bias shifts the decision boundary.

---

### Logistic Regression Prediction Pipeline

The complete prediction process can be summarized as

$$
z=Xw+b
$$

followed by

$$
p=\sigma(z)
$$

and finally

$$
\hat{y}=\mathbb{1}(p\geq0.5)
$$

where $\mathbb{1}$ is the indicator function.

Therefore,

```text
X
↓
X @ weights + bias
↓
Sigmoid
↓
Probability
↓
Threshold at 0.5
↓
Binary Prediction
```

---

### Why is it used? / Applications

Logistic Regression is widely used for:

- Binary classification.
- Spam detection.
- Disease classification.
- Customer churn prediction.
- Fraud detection.
- Credit risk prediction.
- Click-through prediction.
- Customer conversion prediction.
- Text classification.
- Medical diagnosis.

It is especially useful when interpretability and a simple probabilistic classification model are important.

---

> 💡 **Important Note**
>
> Logistic Regression does not directly predict `0` or `1` internally. It first predicts a probability using sigmoid. The threshold is applied afterward to obtain the final class label.

> 💡 **Interview Tip**
>
> Remember the complete pipeline: **logit → sigmoid → probability → threshold**. The decision boundary for a `0.5` threshold is simply $w^Tx+b=0$ because $\sigma(0)=0.5$.

---

## Solution

### Custom Implementation

```python
import numpy as np

def predict_logistic(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    logits = X @ weights + bias
    probs = 1 / (1 + np.exp(-logits))
    return (probs >= 0.5).astype(int)
```

---

## Code Explanation

### Step 1: Compute the Logits

```python
logits = X @ weights + bias
```

The matrix multiplication computes the linear combination for every sample:

$$
z=Xw+b
$$

For $N$ samples and $D$ features, this is performed simultaneously for the entire batch.

---

### Step 2: Apply the Sigmoid Function

```python
probs = 1 / (1 + np.exp(-logits))
```

This applies

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

to every logit.

The resulting values represent the predicted probability of class `1`.

---

### Step 3: Apply the Classification Threshold

```python
probs >= 0.5
```

Every probability greater than or equal to `0.5` becomes `True`, while values below `0.5` become `False`.

Mathematically,

$$
\hat{y}=
\begin{cases}
1 & p\geq0.5\
0 & p<0.5
\end{cases}
$$

---

### Step 4: Convert Boolean Values to Integers

```python
.astype(int)
```

NumPy converts

```text
True  → 1
False → 0
```

producing the final binary prediction array.

---

### Vectorization

No explicit loop is required.

The expression

```python
X @ weights + bias
```

computes the logits for every sample at once.

This is more concise and efficient than manually iterating through every sample and feature.

---

## Time & Space Complexity

Let

- $N$ be the number of samples.
- $D$ be the number of features.

The matrix-vector multiplication

$$
Xw
$$

requires

$$
O(ND)
$$

operations.

The sigmoid and threshold operations each process $N$ values:

$$
O(N)
$$

Therefore, the overall time complexity is

$$
O(ND)
$$

The input matrix already occupies $O(ND)$ space. The logits and probabilities require additional $O(N)$ space.

Thus, the auxiliary space complexity is

$$
O(N)
$$

| Complexity | Value       |
| ---------- | ----------- |
| Time       | **$O(ND)$** |
| Space      | **$O(N)$**  |

where **$N$** is the number of samples and **$D$** is the number of features.
