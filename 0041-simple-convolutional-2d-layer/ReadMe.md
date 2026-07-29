# Simple Convolutional 2D Layer (Medium, Deep Learning)

## Problem Statement

[Simple Convolutional 2D Layer](https://www.deep-ml.com/problems/41)

Implement a Python function that performs a **2D convolution** on an input matrix using a given convolution kernel, padding, and stride. The function should return the resulting output feature map after applying the convolution operation.

---

## Example

```python
import numpy as np

input_matrix = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])

kernel = np.array([
    [1, 0],
    [-1, 1]
])

padding = 1
stride = 2

print(simple_conv2d(input_matrix, kernel, padding, stride))
```

**Output**

```python
[[ 1.  1. -4.]
 [ 9.  7. -4.]
 [ 0. 14. 16.]]
```

**Explanation**

The input is first padded with zeros. The kernel then slides across the padded matrix with a stride of `2`. At each location, element-wise multiplication is performed between the kernel and the current window, and the sum of these products becomes one element in the output feature map.

---

# Learn About the Topic

## Understanding the 2D Convolution Layer

A **Convolutional Layer (Conv2D)** is one of the core building blocks of Convolutional Neural Networks (CNNs). Instead of connecting every input pixel to every neuron like a fully connected layer, a convolution layer scans the input using a small matrix called a **kernel** (or filter). This allows the network to efficiently detect local patterns such as edges, corners, textures, and eventually complex objects.

Each application of the kernel produces one value in the output feature map. By sliding the kernel across the input, the network extracts spatial features while using far fewer parameters than dense layers.

### Convolution Operation

Given an input matrix \(X\) and a kernel \(K\), the output at each position is computed as

\[
Y(i,j)=\sum_{m}\sum_{n}X(i+m,j+n)\times K(m,n)
\]

This is simply:

1. Select a window from the input.
2. Multiply corresponding elements of the window and kernel.
3. Sum the products.
4. Store the result in the output.

This process repeats until the kernel has covered the entire input.

### Important Parameters

#### Kernel (Filter)

A kernel is a small matrix (e.g., 3×3 or 5×5) whose values determine what feature it detects.

Example:

```text
[ 1  0 -1 ]
[ 1  0 -1 ]
[ 1  0 -1 ]
```

This kernel highlights **vertical edges** in an image.

During neural network training, these kernel values are **learned automatically**.

---

#### Padding

Padding adds zeros around the border of the input before convolution.

Without padding:

```text
1 2 3
4 5 6
7 8 9
```

With padding = 1:

```text
0 0 0 0 0
0 1 2 3 0
0 4 5 6 0
0 7 8 9 0
0 0 0 0 0
```

Padding helps:

- Preserve spatial dimensions.
- Allow border pixels to influence the output.
- Prevent excessive shrinking after multiple convolution layers.

---

#### Stride

Stride determines how many cells the kernel moves after each convolution.

- **Stride = 1** → Kernel moves one pixel at a time.
- **Stride = 2** → Kernel skips every other pixel.

Larger strides produce **smaller output feature maps** and reduce computation.

---

### Output Size Formula

If

- Input size = \(N\)
- Kernel size = \(K\)
- Padding = \(P\)
- Stride = \(S\)

then the output size is

\[
\text{Output}=
\left\lfloor
\frac{N-K+2P}{S}
\right\rfloor+1
\]

This formula is applied separately to the height and width.

---

### Why Convolution Works So Well

Compared to fully connected layers, convolution offers several advantages:

- **Local connectivity:** learns nearby patterns instead of the entire image at once.
- **Parameter sharing:** one kernel is reused across the whole image, greatly reducing parameters.
- **Translation awareness:** the same feature can be detected regardless of where it appears.
- **Hierarchical learning:** early layers learn edges, middle layers learn textures, and deeper layers learn objects.

Because of these properties, convolution layers form the foundation of modern computer vision models such as **LeNet, AlexNet, VGG, ResNet, YOLO,** and many others.

> **Note:** In deep learning libraries like PyTorch and TensorFlow, the operation called "convolution" is technically **cross-correlation**, since the kernel is not flipped before multiplication. During training this distinction does not matter because the kernel values are learned automatically.

---

# Solution

```python
import numpy as np

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
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

---

# Code Explanation

1. Read the dimensions of the input matrix and convolution kernel.
2. Add zero padding around the input using `np.pad()`.
3. Slide the kernel over the padded input using the specified stride.
4. At each position, extract a window equal to the kernel size.
5. Perform element-wise multiplication between the window and the kernel.
6. Sum all multiplied values to obtain a single output value.
7. Repeat until the kernel has covered the entire input.
8. Round the final output matrix to four decimal places before returning it.