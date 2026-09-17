# Implement an LSTM Cell from Scratch (Medium, Deep Learning)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: The LSTM Cell](#learn-the-lstm-cell)
  - [What is an LSTM?](#what-is-an-lstm)
  - [The Four Gates](#the-four-gates)
  - [Mathematical Definition](#mathematical-definition)
  - [State Updates](#state-updates)
  - [Why Two States?](#why-two-states)
  - [Why Two Biases?](#why-two-biases)
  - [Parameter Count](#parameter-count)
  - [Important Characteristics](#important-characteristics)
  - [Applications](#applications)

- [Solutions](#solutions)
  - [Custom Implementation](#custom-implementation)
  - [PyTorch Equivalent](#pytorch-equivalent)

- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

---

## Problem Statement

### [Implement an LSTM Cell from Scratch](https://www.deep-ml.com/problems/907)

Implement an `LSTMCell(input_size, hidden_size)` as a PyTorch `nn.Module`.

The constructor must create the following parameters:

- `W_ih`: Shape `(4 * hidden_size, input_size)`
- `W_hh`: Shape `(4 * hidden_size, hidden_size)`
- `b_ih`: Shape `(4 * hidden_size,)`
- `b_hh`: Shape `(4 * hidden_size,)`

The four times multiplier represents the four LSTM components:

1. Input gate
2. Forget gate
3. Cell candidate
4. Output gate

The `forward()` method receives:

- `x`: Input tensor of shape `(N, input_size)`
- `h_prev`: Previous hidden state of shape `(N, hidden_size)`
- `c_prev`: Previous cell state of shape `(N, hidden_size)`

It must return:

- `h_new`: New hidden state of shape `(N, hidden_size)`
- `c_new`: New cell state of shape `(N, hidden_size)`

The gate order must match PyTorch's layout:

\(i,\ f,\ \tilde{c},\ o\)

---

## Example

### Input

```python
input_size = 3
hidden_size = 4

x = torch.randn(2, 3)
state = (
    torch.zeros(2, 4),
    torch.zeros(2, 4)
)
```

### Output

```python
(h_new, c_new)
```

Both tensors have shape:

```python
torch.Size([2, 4])
```

### Reasoning

The batch size is `2`, and the hidden size is `4`.

Each input example produces four gate vectors of size `4`.

The cell state and hidden state therefore maintain the shape:

```text
(batch_size, hidden_size)
```

The initial hidden state and cell state contain zeros.

Consequently, the first update depends on the input tensor and the input-side weights, together with the biases.

---

# Learn: The LSTM Cell

## What is an LSTM?

A Long Short-Term Memory network is a recurrent neural network architecture designed to preserve information across multiple time steps.

Vanilla RNNs repeatedly apply a nonlinear transformation to their hidden state.

This repeated transformation can cause gradients to become extremely small or extremely large during backpropagation through time.

LSTMs address this issue using a separate cell state and learned gates.

An LSTM maintains two states:

- Hidden state `h`: The short-term representation and output.
- Cell state `c`: The long-term memory carried across time steps.

At every time step, the LSTM decides:

- Which information should be forgotten.
- Which new information should be written.
- Which information should be exposed as output.

The decisions are controlled by learned gates.

---

## The Four Gates

An LSTM cell contains three sigmoid gates and one candidate state.

The four components are:

| Component      | Activation | Purpose                                            |
| -------------- | ---------- | -------------------------------------------------- |
| Input gate     | Sigmoid    | Controls new information written to memory         |
| Forget gate    | Sigmoid    | Controls information retained from previous memory |
| Cell candidate | Tanh       | Generates potential new memory content             |
| Output gate    | Sigmoid    | Controls information exposed as hidden state       |

The sigmoid function produces values between `0` and `1`.

\(\sigma(x) = \frac{1}{1 + e^{-x}}\)

A sigmoid gate behaves like a soft filter.

- Value near `0`: Block most information.
- Value near `1`: Allow most information.
- Intermediate value: Allow part of the information.

The hyperbolic tangent function produces values between `-1` and `1`.

\(\tanh(x) \in [-1,1]\)

The cell candidate uses `tanh` because new memory can contain positive and negative values.

---

## Mathematical Definition

Instead of performing four separate matrix multiplications, the implementation combines the input and hidden-state transformations.

The complete pre-activation is:

\(g = xW*{ih}^{T} + b*{ih} + h*{prev}W*{hh}^{T} + b\_{hh}\)

The tensor `g` has shape:

```text
(batch_size, 4 * hidden_size)
```

The tensor is divided into four equal sections:

\(i*{pre},\ f*{pre},\ \tilde{c}_{pre},\ o_{pre} = \operatorname{chunk}(g,4)\)

The input gate is:

\(i = \sigma(i\_{pre})\)

The forget gate is:

\(f = \sigma(f\_{pre})\)

The cell candidate is:

\(\tilde{c} = \tanh(\tilde{c}\_{pre})\)

The output gate is:

\(o = \sigma(o\_{pre})\)

These four tensors all have shape:

```text
(batch_size, hidden_size)
```

---

## State Updates

The cell state is updated first.

\(c*{new} = f \odot c*{prev} + i \odot \tilde{c}\)

Here, `⊙` represents element-wise multiplication.

The first term controls retained memory:

\(f \odot c\_{prev}\)

The second term controls newly written memory:

\(i \odot \tilde{c}\)

The hidden state is then computed from the new cell state.

\(h*{new} = o \odot \tanh(c*{new})\)

The output gate determines how much of the transformed cell state becomes visible in the hidden state.

The cell state is not directly passed through a recurrent `tanh` operation during its update.

This additive structure helps gradients propagate through multiple time steps.

---

## Forget Gate

The forget gate determines how much of the previous cell state should be retained.

\(f = \sigma(W*{if}x + W*{hf}h\_{prev} + b_f)\)

The cell state contribution is:

\(f \odot c\_{prev}\)

If a forget gate element is close to `1`, the corresponding memory value is mostly preserved.

If it is close to `0`, the corresponding memory value is mostly removed.

The forget gate is content-dependent because its value changes according to the input and previous hidden state.

---

## Input Gate

The input gate determines how much new information should be written.

\(i = \sigma(W*{ii}x + W*{hi}h\_{prev} + b_i)\)

The candidate memory is calculated separately:

\(\tilde{c} = \tanh(W*{ic}x + W*{hc}h\_{prev} + b_c)\)

The amount written into the cell state is:

\(i \odot \tilde{c}\)

The input gate does not generate the new information itself.

It controls the amount of information generated by the cell candidate.

---

## Cell Candidate

The candidate state represents possible new content that can be added to memory.

\(\tilde{c} = \tanh(W*{ic}x + W*{hc}h\_{prev} + b_c)\)

The candidate is bounded between `-1` and `1`.

The candidate is combined with the previous cell state through the input and forget gates.

This separation allows the LSTM to distinguish between:

- Information that may be written.
- The amount of information written.
- Information that should remain in memory.

---

## Output Gate

The output gate controls how much information is exposed through the hidden state.

\(o = \sigma(W*{io}x + W*{ho}h\_{prev} + b_o)\)

The hidden state is:

\(h*{new} = o \odot \tanh(c*{new})\)

The hidden state is therefore a filtered version of the current cell state.

The cell state can retain information that is not immediately exposed through the hidden state.

---

## Why Two States?

A vanilla RNN maintains one hidden state.

An LSTM maintains both:

```text
h: Hidden state
c: Cell state
```

The cell state acts as a memory pathway.

The hidden state is used as the current output and as an input to the next time step.

This separation allows the model to preserve information while controlling what is exposed.

The cell state is updated using an additive operation:

\(c*{new} = f \odot c*{prev} + i \odot \tilde{c}\)

The additive path is important because it can allow gradients to flow more effectively than repeatedly applying nonlinear transformations to the entire state.

LSTMs reduce vanishing-gradient problems but do not eliminate all optimization difficulties.

---

## Why Two Biases?

The implementation contains:

```python
b_ih
b_hh
```

The two bias vectors are added together during the gate computation.

\(b = b*{ih} + b*{hh}\)

Mathematically, these two biases could be combined into a single bias.

However, PyTorch maintains separate input-hidden and hidden-hidden biases to match its parameter structure and backend compatibility requirements.

The implementation therefore follows the same convention as `torch.nn.LSTMCell`.

---

## Parameter Count

Let:

- `I` be the input size.
- `H` be the hidden size.

The input-hidden weight matrix contains:

\(4H \cdot I\)

parameters.

The hidden-hidden weight matrix contains:

\(4H \cdot H\)

parameters.

Each bias contains:

\(4H\)

parameters.

There are two bias vectors, so the total parameter count is:

\(4HI + 4H^2 + 8H\)

Factoring the expression:

\(4H(I + H) + 8H\)

For `input_size = 3` and `hidden_size = 4`:

\(4(4)(3 + 4) + 8(4) = 144\)

Therefore, the cell contains `144` trainable parameters.

---

## Important Characteristics

- LSTMs process sequential data one time step at a time.
- The cell state carries long-term information.
- The hidden state represents the current output.
- Sigmoid gates produce values between `0` and `1`.
- The cell candidate uses `tanh`.
- The cell state uses additive memory updates.
- The four gate computations can be combined into one matrix operation.
- The implementation supports batched inputs.
- The gate order must remain consistent with the parameter layout.
- The hidden state and cell state must have compatible shapes.

The gate order used in this problem is:

```text
input, forget, cell candidate, output
```

A different gate order would change the behavior of the cell unless the parameter layout and splitting logic were changed consistently.

---

## Applications

LSTMs have historically been used for:

- Language modeling.
- Sentiment analysis.
- Speech recognition.
- Time-series forecasting.
- Sequence classification.
- Anomaly detection.
- Sensor data processing.
- Streaming inference.
- Sequence-to-sequence models.

Transformers are widely used for modern large-scale sequence modeling.

However, LSTMs remain useful when:

- The model must have a small memory footprint.
- Data arrives continuously.
- Online inference is required.
- The sequence is relatively short.
- A recurrent architecture is easier to deploy.
- The task requires maintaining a fixed-size state.

---

> 💡 **Important Note**
>
> LSTM gates are not hard binary switches.
>
> Their values are continuous and learned through gradient descent.
>
> A gate value of `0.8` means that approximately 80% of the corresponding information is passed through the element-wise operation, not that the model makes a strict yes-or-no decision.

---

# Solutions

## Custom Implementation

The implementation below follows the standard PyTorch parameter layout.

```python
import math
import torch
import torch.nn as nn

class LSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        k = 1 / math.sqrt(hidden_size)

        self.W_ih = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.W_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        self.b_ih = nn.Parameter(torch.empty(4 * hidden_size))
        self.b_hh = nn.Parameter(torch.empty(4 * hidden_size))

        with torch.no_grad():
            self.W_ih.uniform_(-k, k)
            self.W_hh.uniform_(-k, k)
            self.b_ih.uniform_(-k, k)
            self.b_hh.uniform_(-k, k)

    def forward(self, x, state):
        h_prev, c_prev = state

        gates = (
            x @ self.W_ih.T
            + self.b_ih
            + h_prev @ self.W_hh.T
            + self.b_hh
        )

        i, f, c_dash, o = gates.chunk(4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        c_dash = torch.tanh(c_dash)
        o = torch.sigmoid(o)

        c_new = f * c_prev + i * c_dash
        h_new = o * torch.tanh(c_new)

        return h_new, c_new
```

---

## PyTorch Equivalent

PyTorch provides a built-in implementation through `torch.nn.LSTMCell`.

```python
import torch
import torch.nn as nn

cell = nn.LSTMCell(
    input_size=3,
    hidden_size=4
)

x = torch.randn(2, 3)

h_prev = torch.zeros(2, 4)
c_prev = torch.zeros(2, 4)

h_new, c_new = cell(
    x,
    (h_prev, c_prev)
)
```

The built-in module uses parameter names such as:

```python
cell.weight_ih
cell.weight_hh
cell.bias_ih
cell.bias_hh
```

The custom implementation uses:

```python
W_ih
W_hh
b_ih
b_hh
```

The parameter dimensions and gate order are designed to match the PyTorch convention.

---

# Code Explanation

## 1. Store the Dimensions

The constructor stores the input and hidden sizes.

```python
self.input_size = input_size
self.hidden_size = hidden_size
```

These values are useful for defining parameter shapes and understanding the expected input dimensions.

---

## 2. Determine the Initialization Range

The initialization range is:

\(k = \frac{1}{\sqrt{H}}\)

where `H` is the hidden size.

The parameters are initialized from:

\(U(-k,k)\)

This produces values uniformly distributed between `-k` and `k`.

The test cases override the parameters, so the exact initial random values do not affect the test results.

---

## 3. Create the Input-Hidden Weights

```python
self.W_ih = nn.Parameter(
    torch.empty(4 * hidden_size, input_size)
)
```

The matrix transforms the input into four gate pre-activation vectors.

Its shape is:

```text
(4 * hidden_size, input_size)
```

During matrix multiplication, the input is transposed relative to the weight matrix:

```python
x @ self.W_ih.T
```

If `x` has shape `(N, I)`, the result has shape:

```text
(N, 4 * H)
```

---

## 4. Create the Hidden-Hidden Weights

```python
self.W_hh = nn.Parameter(
    torch.empty(4 * hidden_size, hidden_size)
)
```

This matrix transforms the previous hidden state into four gate pre-activation vectors.

The resulting expression is:

```python
h_prev @ self.W_hh.T
```

The output shape is:

```text
(N, 4 * H)
```

---

## 5. Create the Bias Vectors

```python
self.b_ih = nn.Parameter(torch.empty(4 * hidden_size))
self.b_hh = nn.Parameter(torch.empty(4 * hidden_size))
```

Each bias contains four sections.

Each section corresponds to one of the four LSTM components.

The two bias vectors are added separately in the complete gate equation.

---

## 6. Compute the Combined Gates

```python
gates = (
    x @ self.W_ih.T
    + self.b_ih
    + h_prev @ self.W_hh.T
    + self.b_hh
)
```

This combines the input and recurrent transformations into a single tensor.

The resulting tensor has shape:

```text
(N, 4 * H)
```

Broadcasting automatically adds each bias vector to every batch example.

This approach avoids manually concatenating the input and hidden state.

---

## 7. Split the Gate Tensor

```python
i, f, c_dash, o = gates.chunk(4, dim=1)
```

The `chunk()` operation divides the second dimension into four equal parts.

Each resulting tensor has shape:

```text
(N, H)
```

The order is important:

```text
i       Input gate
f       Forget gate
c_dash  Cell candidate
o       Output gate
```

---

## 8. Apply the Gate Activations

```python
i = torch.sigmoid(i)
f = torch.sigmoid(f)
c_dash = torch.tanh(c_dash)
o = torch.sigmoid(o)
```

The input, forget, and output gates use sigmoid.

The cell candidate uses hyperbolic tangent.

This ensures that:

```text
Input gate      ∈ [0, 1]
Forget gate     ∈ [0, 1]
Output gate     ∈ [0, 1]
Cell candidate  ∈ [-1, 1]
```

---

## 9. Update the Cell State

```python
c_new = f * c_prev + i * c_dash
```

This line combines retained memory and new candidate information.

The forget gate controls the previous cell state:

```python
f * c_prev
```

The input gate controls the new candidate:

```python
i * c_dash
```

The operations are element-wise.

---

## 10. Update the Hidden State

```python
h_new = o * torch.tanh(c_new)
```

The new cell state is transformed using `tanh`.

The output gate controls how much of this transformed state is exposed.

The hidden state is then returned together with the new cell state.

```python
return h_new, c_new
```

---

## 11. Why the Alternative Implementation Is Better

The original implementation manually:

- Splits the input-hidden weights.
- Splits the hidden-hidden weights.
- Concatenates the corresponding gate weights.
- Splits both bias vectors.
- Concatenates the input and hidden state.

That approach is mathematically valid, but it performs additional tensor operations.

The optimized implementation computes:

```python
x @ W_ih.T + b_ih + h_prev @ W_hh.T + b_hh
```

This is:

- Easier to read.
- Closer to PyTorch's implementation.
- More efficient in terms of unnecessary concatenation.
- Simpler to verify against the reference implementation.

Both approaches represent the same mathematical operation.

---

# Time & Space Complexity

Let:

- `N` be the batch size.
- `I` be the input size.
- `H` be the hidden size.

| Operation                    | Complexity |
| ---------------------------- | ---------- |
| Input-hidden multiplication  | **O(NIH)** |
| Hidden-hidden multiplication | **O(NH²)** |
| Gate activation functions    | **O(NH)**  |
| Cell state update            | **O(NH)**  |
| Hidden state update          | **O(NH)**  |

The total time complexity is:

\(O(NIH + NH^2)\)

Factoring the expression:

\(O(NH(I + H))\)

The parameter space complexity is:

\(O(HI + H^2 + H)\)

The intermediate gate tensor requires:

\(O(NH)\)

additional space.

Therefore, the total working space for one forward pass is:

\(O(NH + HI + H^2)\)

For a fixed model architecture, the input-dependent activation space is:

\(O(NH)\)

The implementation processes one time step.

For a sequence containing `T` time steps, the total computation becomes:

\(O(TNH(I + H))\)

The recurrent state itself remains fixed-size for each batch example, which makes LSTMs suitable for streaming and online inference.
