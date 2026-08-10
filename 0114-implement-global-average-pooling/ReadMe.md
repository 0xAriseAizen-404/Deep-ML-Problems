# Implement Global Average Pooling (Easy, Deep Learning)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Understanding Global Average Pooling](#learn-understanding-global-average-pooling)
- [Solution](#solution)
  - [Custom Implementation](#custom-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

---

## Problem Statement

### [Implement Global Average Pooling](https://www.deep-ml.com/problems/114)

Implement a function that performs **Global Average Pooling (GAP)** on a 3D NumPy array representing feature maps produced by a convolutional layer.

The input has shape

$$
(H,W,C)
$$

where $H$ is the height, $W$ is the width, and $C$ is the number of channels.

The function should return a 1D array of shape $(C,)$, where each element is the average of all spatial values in the corresponding channel.

---

## Example

### Input

```python
import numpy as np

x = np.array([
    [[1, 2, 3], [4, 5, 6]],
    [[7, 8, 9], [10, 11, 12]]
])
```

### Output

```text
[5.5, 6.5, 7.5]
```

### Reasoning

The input has shape $(2,2,3)$, meaning:

- Height = `2`
- Width = `2`
- Channels = `3`

For channel `0`:

$$
\frac{1+4+7+10}{4}=5.5
$$

For channel `1`:

$$
\frac{2+5+8+11}{4}=6.5
$$

For channel `2`:

$$
\frac{3+6+9+12}{4}=7.5
$$

Therefore, the output is

```text
[5.5, 6.5, 7.5]
```

---

## Learn: Understanding Global Average Pooling

### What is it?

**Global Average Pooling (GAP)** is a pooling operation commonly used in **Convolutional Neural Networks (CNNs)** to reduce the spatial dimensions of feature maps.

A convolutional layer typically produces a 3D tensor containing:

- Height
- Width
- Channels

For an input of shape

$$
(H,W,C)
$$

Global Average Pooling averages every spatial position within each channel.

Instead of keeping an entire $H\times W$ feature map, GAP reduces it to a single scalar.

Therefore,

$$
(H,W,C)\rightarrow(C,)
$$

Each channel produces exactly one output value.

---

### How It Works

For a feature map with height $H$, width $W$, and channel $c$, GAP calculates

$$
GAP(x)*c=\frac{1}{H\times W}\sum*{i=1}^{H}\sum_{j=1}^{W}x_{i,j,c}
$$

where

- $x_{i,j,c}$ is the value at height $i$, width $j$, and channel $c$.
- $H\times W$ is the number of spatial positions.
- $c$ identifies the channel.
- $GAP(x)_c$ is the resulting value for channel $c$.

The operation is repeated independently for every channel.

---

### Example

Consider a $2\times2\times3$ tensor:

$$
x=
\begin{bmatrix}
[1,2,3] & [4,5,6]\
[7,8,9] & [10,11,12]
\end{bmatrix}
$$

The first channel contains

$$
\begin{bmatrix}
1 & 4\
7 & 10
\end{bmatrix}
$$

Its average is

$$
\frac{1+4+7+10}{4}=5.5
$$

The second channel contains

$$
\begin{bmatrix}
2 & 5\
8 & 11
\end{bmatrix}
$$

Its average is

$$
\frac{2+5+8+11}{4}=6.5
$$

The third channel contains

$$
\begin{bmatrix}
3 & 6\
9 & 12
\end{bmatrix}
$$

Its average is

$$
\frac{3+6+9+12}{4}=7.5
$$

Thus,

```text
[5.5, 6.5, 7.5]
```

---

### Global Average Pooling vs Average Pooling

Both operations calculate averages, but they operate differently.

**Average Pooling** usually applies a fixed-size window, such as $2\times2$, across the feature map.

For example:

$$
(H,W,C)\rightarrow(H',W',C)
$$

The spatial dimensions are reduced but generally remain greater than $1$.

**Global Average Pooling** uses the entire spatial feature map as one pooling window:

$$
(H,W,C)\rightarrow(1,1,C)
$$

After removing the singleton spatial dimensions, this becomes

$$
(H,W,C)\rightarrow(C,)
$$

Therefore, GAP completely removes the spatial dimensions.

---

### Parameter Reduction

A common CNN architecture can contain a large fully connected layer after convolutional feature extraction.

Suppose a feature map has shape

$$
7\times7\times512
$$

Flattening it produces

$$
7\times7\times512=25088
$$

features.

Connecting these to a dense layer with `1000` neurons would require

$$
25088\times1000=25,088,000
$$

weights, excluding biases.

GAP instead reduces the feature map to only

$$
512
$$

values.

This can drastically reduce the number of parameters in the classification portion of the network.

---

### Spatial Information

GAP summarizes the entire spatial feature map into one value per channel.

This means detailed spatial information is lost.

For example, if a feature is strongly activated in the top-left corner or bottom-right corner, GAP may produce a similar result if the overall average activation is the same.

Therefore, GAP provides a form of **spatial aggregation** rather than preserving exact feature locations.

---

### Spatial Invariance

Because GAP averages over every spatial location, the resulting value is less dependent on the exact position of a feature.

If the same feature moves across the image while maintaining similar activation strength, the global average can remain relatively stable.

This makes GAP useful when the model needs to determine **whether a feature exists** rather than exactly **where the feature is located**.

---

### Variable Input Sizes

One useful property of GAP is that its output depends only on the number of channels.

For example,

$$
224\times224\times64\rightarrow64
$$

and

$$
128\times128\times64\rightarrow64
$$

Both produce a vector with `64` values.

This makes GAP particularly useful in architectures where the spatial input dimensions can vary.

A traditional flattening layer would produce different numbers of features for these two spatial sizes.

---

### Benefits of Global Average Pooling

#### Parameter Reduction

GAP can replace large fully connected layers and significantly reduce the number of trainable parameters.

#### Reduced Overfitting

Fewer parameters can reduce the model's capacity to memorize training data, potentially improving generalization.

#### Spatial Aggregation

GAP summarizes the entire spatial feature map into one value per channel.

#### No Pooling Hyperparameters

Unlike ordinary pooling, GAP does not require choosing a window size or stride because the entire spatial feature map is used.

#### Flexible Spatial Dimensions

The output size depends on the number of channels rather than the input height and width.

---

### Use in Modern Architectures

Global Average Pooling became especially important in modern CNN architectures.

For example, **ResNet** commonly uses global average pooling near the end of the network before the final classification layer.

A simplified architecture is

```text
Input
  ↓
Convolutional Layers
  ↓
Feature Maps
  ↓
Global Average Pooling
  ↓
Feature Vector
  ↓
Fully Connected Layer
  ↓
Class Scores
```

If the final convolutional feature maps have shape

$$
H\times W\times C
$$

GAP converts them into

$$
C
$$

values, which can then be passed to the classifier.

---

### Characteristics / Key Points

- GAP operates independently on every channel.
- It averages across height and width.
- It produces one scalar per channel.
- It converts $(H,W,C)$ into $(C,)$.
- It contains no trainable parameters.
- It has no learned weights or biases.
- It removes spatial dimensions completely.
- It reduces the number of values passed to later layers.
- It can reduce the need for large fully connected layers.
- It loses detailed spatial information.
- Its output size depends only on the number of channels.
- It can support varying spatial input dimensions.

---

### Why is it used? / Applications

Global Average Pooling is commonly used in:

- Image Classification
- Convolutional Neural Networks
- Transfer Learning
- Feature Extraction
- Object Recognition
- Image Embedding Generation
- Mobile and lightweight CNN architectures
- Classification heads of modern CNNs

It is particularly useful when the final stage of a CNN needs a compact representation of the learned feature maps.

---

> 💡 **Important Note**
>
> GAP has **no trainable parameters**. It simply computes an average over the spatial dimensions. This makes it fundamentally different from a fully connected layer, where the mapping contains learned weights and biases.

> 💡 **Interview Tip**
>
> Remember the shape transformation: Global Average Pooling converts $(H,W,C)$ into $(C,)$ by averaging across axes `0` and `1`. In NumPy, this can be written directly as `np.mean(x, axis=(0, 1))`.

---

## Solution

### Custom Implementation

```python
import numpy as np

def global_avg_pool(x: np.ndarray) -> np.ndarray:
    # pool = [0] * len(x[0][0])
    # pool = np.asarray(pool)
    # cnt = 0
    # for row in x:
        # for pixel in row:
            # pool = pool + pixel
            # cnt += 1
    # return pool / cnt
    return np.sum(x, axis=(0, 1)) / (x.shape[0] * x.shape[1])
```

### NumPy Equivalent

```python
def global_avg_pool(x: np.ndarray) -> np.ndarray:
    return np.mean(x, axis=(0, 1))
```

Both implementations produce the same result.

The first implementation explicitly computes the sum and divides by the number of spatial elements, while `np.mean()` performs the same operation directly.

---

## Code Explanation

### Step 1: Identify the Spatial Dimensions

The input has shape

$$
(H,W,C)
$$

The first two axes correspond to height and width, while the third axis represents the channels.

For the example,

$$
(2,2,3)
$$

there are `4` spatial positions for every channel.

---

### Step 2: Sum Across Height and Width

```python
np.sum(x, axis=(0, 1))
```

The axes `0` and `1` represent height and width.

Therefore, summing over these axes combines all spatial positions while preserving the channel dimension.

For the example, the result is

```text
[22, 26, 30]
```

because

$$
1+4+7+10=22
$$

$$
2+5+8+11=26
$$

$$
3+6+9+12=30
$$

The resulting shape is

$$
(C,)
$$

---

### Step 3: Divide by the Number of Spatial Elements

The number of spatial positions is

$$
H\times W
$$

In the example,

$$
H\times W=2\times2=4
$$

Therefore,

```python
np.sum(x, axis=(0, 1)) / (x.shape[0] * x.shape[1])
```

computes the average for every channel.

The final result is

```text
[5.5, 6.5, 7.5]
```

---

### Step 4: Using `np.mean`

NumPy provides a direct implementation of the same mathematical operation:

```python
np.mean(x, axis=(0, 1))
```

The `axis=(0,1)` argument tells NumPy to average over height and width while preserving the channel axis.

Therefore,

$$
(H,W,C)\xrightarrow{\text{mean over }H,W}(C,)
$$

---

## Time & Space Complexity

Let:

- $H$ be the height.
- $W$ be the width.
- $C$ be the number of channels.

Every element of the input must be considered once to calculate the average.

Therefore, the time complexity is

$$
O(HWC)
$$

The output contains one value for each channel, so the output space is

$$
O(C)
$$

The implementation uses only a constant amount of additional working memory apart from the output.

| Complexity | Value        |
| ---------- | ------------ |
| Time       | **$O(HWC)$** |
| Space      | **$O(C)$**   |

Where **$H$** and **$W$** are the spatial dimensions and **$C$** is the number of channels.
