# Implement a Simple Residual Block with Shortcut Connection (Easy, Deep Learning)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Understanding Residual Blocks in ResNet](#learn-understanding-residual-blocks-in-resnet)
- [Solution](#solution)
  - [Custom Implementation](#custom-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

---

## Problem Statement

### [Implement a Simple Residual Block with Shortcut Connection](https://www.deep-ml.com/problems/113)

Implement a simple **residual block** using NumPy.

The block takes a 1D input array and processes it through two weight layers with ReLU activations. The original input is then added back through a **shortcut connection**, followed by a final ReLU activation.

The computation follows:

$$
F(x)=W_2{ReLU}(W_1x)
$$

and

$$
y={ReLU}(F(x)+x)
$$

---

## Example

### Input

```python
x = np.array([1.0, 2.0])
w1 = np.array([[1.0, 0.0], [0.0, 1.0]])
w2 = np.array([[0.5, 0.0], [0.0, 0.5]])

residual_block(x, w1, w2)
```

### Output

```text
[1.5, 3.0]
```

---

## Learn: Understanding Residual Blocks in ResNet

### What is it?

A **Residual Block** is the fundamental building block of **Residual Networks (ResNet)**.

As neural networks become deeper, training can become difficult because gradients may become very small during backpropagation. Residual blocks address this problem by providing a **shortcut connection** that allows the input to bypass one or more layers.

Instead of forcing the layers to learn a complete transformation $H(x)$, the block learns a residual transformation $F(x)$ and adds the original input back:

$$
H(x)=F(x)+x
$$

This simple addition creates a direct path from the input to the output.

---

### Residual Learning

In a traditional neural network, the desired mapping can be represented as

$$
H(x)
$$

A residual block instead learns

$$
F(x)=H(x)-x
$$

Therefore,

$$
H(x)=F(x)+x
$$

The network only needs to learn the difference between the desired transformation and the identity mapping.

If the optimal transformation is close to the identity function, the residual function can become approximately zero:

$$
F(x)\approx0
$$

which gives

$$
H(x)\approx x
$$

This is much easier for the network to represent.

---

### Structure of a Residual Block

A simple residual block can be represented as:

```text
        ┌──────────────────────────┐
        │                          │
        │          Shortcut        │
        │                          │
        │                          ▼
x ──────┼──> W1 ──> ReLU ──> W2 ──> (+) ──> ReLU ──> y
        │                          ▲
        └──────────────────────────┘
```

The main transformation is

$$
F(x)=W_2{ReLU}(W_1x)
$$

The shortcut provides the original input:

$$
S(x)=x
$$

The final output is

$$
y={ReLU}(F(x)+S(x))
$$

For this problem, the shortcut is simply the identity function $S(x)=x$.

---

### ReLU Activation

The ReLU activation function is

$$
{ReLU}(z)=\max(0,z)
$$

It keeps positive values unchanged and converts negative values to zero.

For example:

$$
{ReLU}
\begin{pmatrix}
-2\
3
\end{pmatrix} ============
\begin{pmatrix}
0\
3
\end{pmatrix}
$$

ReLU introduces non-linearity while remaining computationally simple.

---

### Mathematical Structure

For input $x$ and weight matrices $W_1$ and $W_2$, the first transformation is

$$
z_1=W_1x
$$

Apply ReLU:

$$
a_1={ReLU}(z_1)
$$

Apply the second weight matrix:

$$
z_2=W_2a_1
$$

The residual transformation is therefore

$$
F(x)=W_2{ReLU}(W_1x)
$$

The shortcut adds the original input:

$$
z_3=F(x)+x
$$

Finally:

$$
y={ReLU}(z_3)
$$

Combining everything:

$$
y={ReLU}(W_2{ReLU}(W_1x)+x)
$$

This is the complete computation implemented by the problem.

---

### Shortcut Connection

The shortcut connection directly passes the input to the addition operation.

Without the shortcut, the network would only compute

$$
y=F(x)
$$

With the shortcut:

$$
y=F(x)+x
$$

The input is therefore preserved even when the learned transformation is small.

The shortcut does not necessarily need to be an identity mapping in more general ResNet architectures. If the dimensions of the input and residual transformation differ, a projection layer can be used:

$$
S(x)=W_sx
$$

The output then becomes

$$
y={ReLU}(F(x)+W_sx)
$$

In this problem, the dimensions match, so the identity shortcut is sufficient.

---

### Why Shortcut Connections Help

#### Ease of Learning

Suppose the desired mapping is approximately the identity:

$$
H(x)\approx x
$$

The residual block only needs to learn

$$
F(x)\approx0
$$

instead of learning an entire identity transformation through multiple layers.

This makes optimization easier.

#### Gradient Flow

During backpropagation, the addition operation provides a direct path through the shortcut.

Given

$$
y=F(x)+x
$$

the derivative with respect to $x$ contains the direct identity contribution:

$$
\frac{\partial y}{\partial x}=
\frac{\partial F(x)}{\partial x}+I
$$

where $I$ is the identity matrix.

This direct gradient path helps reduce the difficulty of training very deep networks.

---

### Identity Mapping

One of the most important ideas behind residual learning is that the block can represent an identity transformation.

If

$$
F(x)=0
$$

then

$$
y=x
$$

This means adding more residual blocks does not necessarily force the network to learn a harmful transformation.

The network can effectively make a block behave like an identity mapping when that is useful.

---

### Residual Block vs Traditional Block

| Property              | Traditional Block | Residual Block       |
| --------------------- | ----------------- | -------------------- |
| Main Transformation   | $H(x)$            | $F(x)$               |
| Shortcut              | No                | Yes                  |
| Output                | $H(x)$            | $F(x)+x$             |
| Identity Mapping      | Harder            | Easy                 |
| Gradient Path         | Through layers    | Direct shortcut path |
| Deep Network Training | More difficult    | Easier               |

The main architectural difference is the addition of the shortcut connection.

---

### Characteristics / Key Points

- Residual blocks are the core components of ResNet.
- They learn residual mappings instead of complete transformations.
- The shortcut adds the original input to the transformed output.
- The identity shortcut requires matching input and output dimensions.
- ReLU provides non-linearity inside the block.
- The residual transformation in this problem contains two weight layers.
- The shortcut creates a direct path for information and gradients.
- If $F(x)=0$, the block becomes an identity mapping.
- Projection shortcuts can handle mismatched dimensions.
- Residual connections make very deep networks easier to optimize.
- The basic residual equation is $y=F(x)+x$.

---

### Why is it used? / Applications

Residual blocks are widely used in:

- Image classification.
- Object detection.
- Image segmentation.
- Computer vision.
- Medical image analysis.
- Feature extraction.
- Deep convolutional neural networks.
- ResNet and its variants.
- Many modern architectures that use skip connections.

The central benefit is the ability to construct much deeper networks while maintaining effective gradient flow.

---

> 💡 **Important Note**
>
> A residual block does not simply "skip learning." The shortcut provides an identity path while the weight layers learn a residual correction. The final representation combines both paths through addition.

> 💡 **Interview Tip**
>
> Remember the core equation: **$y=F(x)+x$**. If the desired mapping is close to identity, the network only needs to learn $F(x)\approx0$. This is the central intuition behind ResNet.

---

## Solution

### Custom Implementation

```python
import numpy as np

def residual_block(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    z1 = np.maximum(0, x @ w1)
    z2 = z1 @ w2 + x
    return np.maximum(0, z2)
```

---

## Code Explanation

### Step 1: First Weight Transformation

```python
z1 = x @ w1
```

The input is multiplied by the first weight matrix:

$$
z_1=xW_1
$$

This produces the first intermediate representation.

---

### Step 2: Apply ReLU

```python
z1 = np.maximum(0, z1)
```

The ReLU function removes negative values:

$$
a_1={ReLU}(z_1)
$$

The NumPy expression

```python
np.maximum(0, z1)
```

performs ReLU element-wise.

---

### Step 3: Apply the Second Weight Layer

```python
z2 = z1 @ w2
```

The activated representation is passed through the second weight matrix:

$$
F(x)=W_2{ReLU}(W_1x)
$$

The code uses row-vector notation, so the actual NumPy operation is equivalent to

$$
F(x)={ReLU}(xW_1)W_2
$$

---

### Step 4: Add the Shortcut

```python
z2 = z2 + x
```

The original input is added to the residual transformation:

$$
z_2=F(x)+x
$$

This is the key operation that makes the block residual.

---

### Step 5: Final ReLU

```python
return np.maximum(0, z2)
```

The summed result is passed through a final ReLU:

$$
y={ReLU}(F(x)+x)
$$

This produces the final output of the residual block.

---

### Complete Computation

The implementation performs the following sequence:

```text
x
↓
x @ W1
↓
ReLU
↓
@ W2
↓
+ x
↓
ReLU
↓
Output
```

Mathematically, using the row-vector convention of the implementation:

$$
y={ReLU}({ReLU}(xW_1)W_2+x)
$$

This is equivalent to the residual structure described by the problem.

---

## Time & Space Complexity

Let

- $D$ be the input dimension.
- $H$ be the hidden dimension.
- $O$ be the output dimension.
- $W_1$ have shape $D\times H$.
- $W_2$ have shape $H\times O$.

The first matrix multiplication requires

$$
O(DH)
$$

The second matrix multiplication requires

$$
O(HO)
$$

The element-wise ReLU and shortcut addition require

$$
O(H+O)
$$

Therefore, the total time complexity is

$$
O(DH+HO)
$$

For the problem's same-dimensional case where all vectors and matrices are of size $D$:

$$
O(D^2)
$$

The intermediate vectors require $O(H+O)$ additional space.

| Complexity | Value          |
| ---------- | -------------- |
| Time       | **$O(DH+HO)$** |
| Space      | **$O(H+O)$**   |

For the specific square case where the input, hidden representation, and output all have dimension $D$, the complexity becomes **$O(D^2)$ time** and **$O(D)$ auxiliary space**.
