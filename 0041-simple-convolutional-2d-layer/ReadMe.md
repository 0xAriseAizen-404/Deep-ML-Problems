# Simple Convolutional 2D Layer (Medium, Deep Learning)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Understanding 2D Convolution](#learn-understanding-2d-convolution)
- [Solution](#solution)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

---

## Problem Statement

### [Simple Convolutional 2D Layer](https://www.deep-ml.com/problems/41)

Write a Python function that performs a **2D convolution** on an input matrix.

The function should:

- Accept a 2-D input matrix.
- Accept a 2-D convolution kernel (filter).
- Support configurable **padding**.
- Support configurable **stride**.
- Slide the kernel across the padded input matrix.
- Compute the element-wise multiplication followed by summation at every valid position.
- Return the resulting output feature map.

---

## Example

### Input

```python
import numpy as np

input_matrix = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9,10,11,12],
    [13,14,15,16]
])

kernel = np.array([
    [1, 0],
    [-1, 1]
])

padding = 1
stride = 2

output = simple_conv2d(input_matrix, kernel, padding, stride)

print(output)
```

### Output

```python
[
    [ 1.,  1., -4.],
    [ 9.,  7., -4.],
    [ 0., 14., 16.]
]
```

### Reasoning

The input matrix is first padded with one layer of zeros.

The kernel is then moved across the padded matrix with a stride of **2**.

At every position,

- Multiply the kernel and the corresponding input window element-wise.
- Sum all products.
- Store the result in the output feature map.

Repeating this process over the entire matrix produces the final convolution output.

---

## Learn: Understanding 2D Convolution

### What is it?

A **Convolutional Layer** is the fundamental building block of **Convolutional Neural Networks (CNNs)**.

Instead of connecting every input pixel to every neuron (as in a fully connected layer), convolution uses a small matrix called a **kernel** (or **filter**) that slides across the input.

At each location, the kernel extracts useful local patterns such as

- Edges
- Corners
- Lines
- Textures
- Shapes

As multiple convolutional layers are stacked, the network gradually learns increasingly complex features.

---

### Mathematical Definition

Suppose

- Input matrix has dimensions

$$
H \times W
$$

- Kernel has dimensions

$$
K_h \times K_w
$$

For every valid location,

$$
O(i,j) =
\sum_{m=0}^{K_h-1}
\sum_{n=0}^{K_w-1}
I(i+m,\;j+n)\times K(m,n)
$$

where

- $I$ is the input matrix.
- $K$ is the kernel.
- $O$ is the output feature map.

---

### Output Size Formula

If

- Input height = $H$
- Input width = $W$
- Kernel size = $K_h \times K_w$
- Padding = $P$
- Stride = $S$

then the output dimensions are

$$
H_{out} =
\left\lfloor
\frac{H+2P-K_h}{S}
\right\rfloor + 1
$$

$$
W_{out} =
\left\lfloor
\frac{W+2P-K_w}{S}
\right\rfloor + 1
$$

These formulas determine how many times the kernel fits across the padded input.

---

### Padding

Padding adds zeros around the border of the input.

Without padding

```text
1 2 3
4 5 6
7 8 9
```

With

```text
padding = 1
```

becomes

```text
0 0 0 0 0
0 1 2 3 0
0 4 5 6 0
0 7 8 9 0
0 0 0 0 0
```

Padding helps

- Preserve spatial dimensions.
- Allow border pixels to contribute equally.
- Reduce information loss near edges.

---

### Stride

Stride determines how far the kernel moves after each convolution.

For

```text
Stride = 1
```

the kernel moves

```text
→ → →
↓
→ → →
```

For

```text
Stride = 2
```

it skips every other position.

Larger strides reduce the output size and computational cost.

---

### Convolution Example

Suppose

Input

$$
\begin{bmatrix}
1 & 2 & 3\\
4 & 5 & 6\\
7 & 8 & 9
\end{bmatrix}
$$

Kernel

$$
\begin{bmatrix}
1 & 0\\
-1 & 1
\end{bmatrix}
$$

The first window is

$$
\begin{bmatrix}
1 & 2\\
4 & 5
\end{bmatrix}
$$

The convolution is

$$
(1\times1) + (2\times0) + (4\times-1) + (5\times1) = 2
$$

The kernel continues sliding until every valid position has been processed.

---

### Convolution vs Cross-Correlation

Mathematically, convolution flips the kernel before applying it.

$$
K(i,j)
\rightarrow
K(-i,-j)
$$

However, most deep learning libraries such as **PyTorch**, **TensorFlow**, and **Keras** actually perform **cross-correlation**, where the kernel is **not flipped**.

The implementation in this problem also performs cross-correlation, which is the standard operation used in modern CNNs.

---

### Characteristics / Key Points

- Extracts local spatial features.
- Uses shared weights across the image.
- Greatly reduces the number of trainable parameters.
- Supports configurable padding and stride.
- Produces feature maps instead of scalar outputs.
- Learns hierarchical visual representations.
- Translation-equivariant, making it effective for image tasks.

---

### Why is it used? / Applications

2D convolutions are used throughout deep learning.

Applications include

- Image Classification
- Object Detection
- Face Recognition
- Medical Image Analysis
- Image Segmentation
- OCR (Optical Character Recognition)
- Autonomous Driving
- Video Processing
- Satellite Image Analysis

Nearly every modern computer vision model is built upon convolutional layers.

---

> 💡 **Important Note**
>
> Although the operation is commonly called **convolution**, deep learning libraries such as **PyTorch** and **TensorFlow** actually implement **cross-correlation** (without flipping the kernel). During training, the network learns the appropriate filter values, so flipping the kernel is unnecessary in practice.

---

## Solution

### Custom Implementation

```python
import numpy as np

def simple_conv2d(input_matrix, kernel, padding, stride):
    input_height, input_width = input_matrix.shape
    kernel_height, kernel_width = kernel.shape

    matrix = np.pad(
        input_matrix,
        ((padding, padding), (padding, padding)),
        mode="constant"
    )

    mat_height, mat_width = matrix.shape

    output_matrix = []

    for i in range(0, mat_height - kernel_height + 1, stride):
        row = []

        for j in range(0, mat_width - kernel_width + 1, stride):
            window = matrix[i:i+kernel_height, j:j+kernel_width]

            row.append(np.sum(window * kernel))

        output_matrix.append(row)

    return np.round(output_matrix, 4)
```

### NumPy / Deep Learning Equivalent

```python
import numpy as np
from scipy.signal import correlate2d

output = correlate2d(
    input_matrix,
    kernel,
    mode="same"
)
```

In deep learning frameworks:

```python
torch.nn.Conv2d(...)
```

or

```python
tf.keras.layers.Conv2D(...)
```

performs the same underlying operation while learning the kernel weights automatically.

---

## Code Explanation

### Step 1

Determine the dimensions of the input matrix and kernel.

```python
input_height, input_width = input_matrix.shape
kernel_height, kernel_width = kernel.shape
```

These dimensions are required to compute the valid sliding positions.

---

### Step 2

Pad the input matrix.

```python
matrix = np.pad(
    input_matrix,
    ((padding, padding),
     (padding, padding)),
    mode="constant"
)
```

Zeros are added around the border according to the specified padding.

---

### Step 3

Slide the kernel across the padded matrix.

```python
for i in range(
    0,
    mat_height - kernel_height + 1,
    stride
):
```

The outer loop moves vertically, while the inner loop moves horizontally.

---

### Step 4

Extract the current window.

```python
window = matrix[
    i:i+kernel_height,
    j:j+kernel_width
]
```

The extracted region has exactly the same dimensions as the kernel.

---

### Step 5

Perform the convolution.

```python
np.sum(window * kernel)
```

First perform element-wise multiplication, then sum all values to produce a single output element.

---

### Step 6

Store every computed value.

Each convolution result is appended to the current output row.

After processing all windows, the rows form the final feature map.

---

### Step 7

Round the output.

```python
np.round(output_matrix, 4)
```

This produces cleaner numerical output for display.

---

## Time & Space Complexity

| Complexity | Value                          |
| ---------- | ------------------------------ |
| Time       | **O(Hₒ × Wₒ × Kₕ × K𝓌)**       |
| Space      | **O((H + 2P)(W + 2P) + HₒWₒ)** |

where

- $H, W$ are the input height and width.
- $K_h, K_w$ are the kernel dimensions.
- $P$ is the padding size.
- $H_o, W_o$ are the output height and width.

Each output element requires multiplying every kernel value with its corresponding input value, resulting in the stated time complexity.
