# Implementing a Simple RNN (Medium, Deep Learning)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Understanding Recurrent Neural Networks](#learn-understanding-recurrent-neural-networks)
  - [What is it?](#what-is-it)
  - [Mathematical Definition](#mathematical-definition)
  - [Hidden State](#hidden-state)
  - [How an RNN Processes a Sequence](#how-an-rnn-processes-a-sequence)
  - [Step-by-Step Example](#step-by-step-example)
  - [Characteristics / Key Points](#characteristics--key-points)
  - [Why is it used? / Applications](#why-is-it-used--applications)
- [Solutions](#solutions)
  - [Custom Implementation](#custom-implementation)
  - [NumPy Implementation](#numpy-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

---

## Problem Statement

### [Implementing a Simple RNN](https://www.deep-ml.com/problems/54)

Write a Python function that implements the forward pass of a simple **Recurrent Neural Network (RNN)**.

The function should:

- Accept a sequence of input vectors.
- Accept an initial hidden state.
- Accept the input-to-hidden weight matrix.
- Accept the hidden-to-hidden weight matrix.
- Accept a bias vector.
- Process the complete sequence one time step at a time.
- Update the hidden state using the `tanh` activation function.
- Return the final hidden state.
- Round the final result to four decimal places.

The RNN must use the previous hidden state when processing each new input, allowing information from earlier elements of the sequence to influence later hidden states.

---

## Example

### Input

```python
input_sequence = [[1.0], [2.0], [3.0]]

initial_hidden_state = [0.0]

Wx = [[0.5]]

Wh = [[0.8]]

b = [0.0]

output = rnn_forward(
    input_sequence,
    initial_hidden_state,
    Wx,
    Wh,
    b
)
```

### Output

```text
[0.9759]
```

### Reasoning

The RNN processes the input sequence sequentially.

At each time step, the current input is combined with the previous hidden state.

The resulting value is passed through the `tanh` activation function.

For the first input, the RNN uses the initial hidden state.

For every later input, the hidden state produced at the previous time step becomes the input's recurrent information.

The computation therefore follows

$$h_t = \tanh(W_xx_t + W_hh_{t-1} + b)$$

After the final input has been processed, the resulting hidden state is returned.

---

## Learn: Understanding Recurrent Neural Networks

### What is it?

A **Recurrent Neural Network (RNN)** is a neural network architecture designed to process **sequential data**.

Unlike a standard feed-forward neural network, an RNN maintains a **hidden state** that carries information from previous time steps.

This allows the network to model dependencies between elements of a sequence.

For a sequence

```text
x1 → x2 → x3 → ... → xT
```

the RNN does not process every input independently.

Instead, the computation forms a chain:

```text
x1 → h1
      ↓
x2 → h2
      ↓
x3 → h3
      ↓
...
      ↓
xT → hT
```

Each hidden state depends on both:

- The current input.
- The previous hidden state.

Therefore, information can flow through the sequence over time.

---

### Mathematical Definition

At time step `t`, a simple RNN computes its hidden state as

$$h_t = \tanh(W_xx_t + W_hh_{t-1} + b)$$

where

- $x_t$ is the input vector at time step $t$.
- $h_t$ is the current hidden state.
- $h_{t-1}$ is the previous hidden state.
- $W_x$ is the input-to-hidden weight matrix.
- $W_h$ is the hidden-to-hidden weight matrix.
- $b$ is the bias vector.
- $\tanh$ is the hyperbolic tangent activation function.

The initial hidden state is usually represented as

$$h_0$$

and is supplied before the sequence is processed.

---

### Matrix Dimensions

Suppose the input vector has dimension $D$ and the hidden state has dimension $H$.

Then

$$x_t \in \mathbb{R}^{D}$$

and

$$h_t \in \mathbb{R}^{H}$$

The input-to-hidden weight matrix has dimensions

$$W_x \in \mathbb{R}^{H \times D}$$

The hidden-to-hidden weight matrix has dimensions

$$W_h \in \mathbb{R}^{H \times H}$$

The bias vector has dimensions

$$b \in \mathbb{R}^{H}$$

The resulting hidden state also has dimension $H$.

The matrix multiplication therefore works as

$$W_xx_t \in \mathbb{R}^{H}$$

and

$$W_hh_{t-1} \in \mathbb{R}^{H}$$

so the terms can be added together.

---

### Hidden State

The **hidden state** is the main memory mechanism of a simple RNN.

At each time step, the hidden state is updated using the current input and the previous hidden state.

The recurrence can be represented as

$$h_0 \rightarrow h_1 \rightarrow h_2 \rightarrow \cdots \rightarrow h_T$$

Each state contains information produced from the sequence processed so far.

For example,

$$h_1 = f(x_1, h_0)$$

$$h_2 = f(x_2, h_1)$$

$$h_3 = f(x_3, h_2)$$

and generally

$$h_t = f(x_t, h_{t-1})$$

where the function $f$ is the RNN update rule.

For the simple RNN in this problem,

$$f(x_t,h_{t-1}) = \tanh(W_xx_t + W_hh_{t-1} + b)$$

Therefore, the final hidden state contains information that has been propagated through all previous time steps.

---

### The Role of the Input-to-Hidden Weights

The matrix $W_x$ controls how the current input affects the hidden state.

The input contribution is

$$W_xx_t$$

A larger magnitude in the weights can cause particular input features to have a stronger influence on the hidden state.

For every time step, the current input is transformed using the same matrix $W_x$.

This is an important property of RNNs.

The network does not need a separate input weight matrix for every position in the sequence.

The same parameters are reused across time.

---

### The Role of the Hidden-to-Hidden Weights

The matrix $W_h$ controls how information from the previous hidden state is carried forward.

The recurrent contribution is

$$W_hh_{t-1}$$

This term is responsible for the recurrent behavior of the network.

Without this term, each input would be processed independently.

With the recurrent term, the current hidden state depends on previous hidden states.

Therefore,

$$h_t = \tanh(W_xx_t + W_hh_{t-1} + b)$$

contains both:

- Information from the current input.
- Information propagated from earlier inputs.

---

### The Role of the Bias

The bias vector $b$ provides an additional learnable offset.

The complete pre-activation value is

$$z_t = W_xx_t + W_hh_{t-1} + b$$

The activation function is then applied to this value

$$h_t = \tanh(z_t)$$

The bias allows the neuron to shift the activation independently of the input and previous hidden state.

---

### The Tanh Activation Function

The simple RNN in this problem uses the **hyperbolic tangent** activation function.

It is defined as

$$\tanh(x) = \frac{e^x-e^{-x}}{e^x+e^{-x}}$$

The output range is

$$-1 < \tanh(x) < 1$$

Therefore, every hidden-state component is bounded between approximately `-1` and `1`.

For large positive values,

$$\tanh(x) \rightarrow 1$$

For large negative values,

$$\tanh(x) \rightarrow -1$$

For values close to zero,

$$\tanh(x) \approx x$$

This makes `tanh` useful for producing bounded hidden-state representations.

---

### Why Tanh is Used in a Simple RNN

The hidden state can potentially grow without bound if repeated linear transformations are applied.

The `tanh` function keeps the hidden state within a limited numerical range.

For example,

$$\tanh(0) = 0$$

$$\tanh(1) \approx 0.7616$$

$$\tanh(2) \approx 0.9640$$

$$\tanh(5) \approx 0.9999$$

This allows the RNN to maintain a controlled hidden representation.

However, the saturation behavior of `tanh` also contributes to the **vanishing gradient problem** during training.

---

### How an RNN Processes a Sequence

Suppose the input sequence is

```text
[x1, x2, x3]
```

and the initial hidden state is

```text
h0
```

The first input produces

$$h_1 = \tanh(W_xx_1 + W_hh_0 + b)$$

The second input uses the first hidden state

$$h_2 = \tanh(W_xx_2 + W_hh_1 + b)$$

The third input uses the second hidden state

$$h_3 = \tanh(W_xx_3 + W_hh_2 + b)$$

The final result is

$$h_3$$

More generally, for a sequence of length $T$,

$$h_T = \tanh(W_xx_T + W_hh_{T-1} + b)$$

The function in this problem returns this final state.

---

### Unrolling the RNN

An RNN can be visualized as a single recurrent cell that is **unrolled through time**.

The same cell is reused at every time step.

```text
              ┌─────────────┐
x1 ──────────►│             │
h0 ──────────►│   RNN Cell  │──► h1
              │             │
              └─────────────┘
                     │
                     ▼
              ┌─────────────┐
x2 ──────────►│             │
h1 ──────────►│   RNN Cell  │──► h2
              │             │
              └─────────────┘
                     │
                     ▼
              ┌─────────────┐
x3 ──────────►│             │
h2 ──────────►│   RNN Cell  │──► h3
              │             │
              └─────────────┘
```

Although the diagram shows multiple cells, the parameters are shared.

The same $W_x$, $W_h$, and $b$ are used at every time step.

---

### Parameter Sharing

One of the most important characteristics of RNNs is **parameter sharing across time**.

For every time step,

$$W_x^{(1)} = W_x^{(2)} = \cdots = W_x^{(T)}$$

and

$$W_h^{(1)} = W_h^{(2)} = \cdots = W_h^{(T)}$$

The bias is also shared

$$b^{(1)} = b^{(2)} = \cdots = b^{(T)}$$

This allows an RNN to process sequences of different lengths without requiring a different set of parameters for every position.

---

### Step-by-Step Example

Consider the input sequence

```text
x1 = 1.0
x2 = 2.0
x3 = 3.0
```

with

```text
h0 = 0.0
Wx = 0.5
Wh = 0.8
b = 0.0
```

The recurrence is

$$h_t = \tanh(0.5x_t + 0.8h_{t-1})$$

#### Step 1: First Input

For $x_1 = 1.0$,

$$h_1 = \tanh(0.5(1.0) + 0.8(0.0) + 0.0)$$

Therefore,

$$h_1 = \tanh(0.5) \approx 0.4621$$

The first hidden state is approximately

```text
h1 = 0.4621
```

---

#### Step 2: Second Input

For $x_2 = 2.0$, the RNN uses $h_1$.

$$h_2 = \tanh(0.5(2.0) + 0.8(0.4621) + 0.0)$$

The value inside `tanh` is approximately

$$1.0 + 0.3697 = 1.3697$$

Therefore,

$$h_2 = \tanh(1.3697) \approx 0.8781$$

The second hidden state is approximately

```text
h2 = 0.8781
```

---

#### Step 3: Third Input

For $x_3 = 3.0$, the RNN uses $h_2$.

$$h_3 = \tanh(0.5(3.0) + 0.8(0.8781) + 0.0)$$

The value inside `tanh` is approximately

$$1.5 + 0.7025 = 2.2025$$

Therefore,

$$h_3 = \tanh(2.2025) \approx 0.9759$$

The final hidden state is approximately

```text
[0.9759]
```

The exact result may vary slightly depending on intermediate rounding.

---

### Forward Pass Algorithm

The forward pass can be expressed as the following algorithm.

```text
RNN-FORWARD(X, h0, Wx, Wh, b)

1. h ← h0
2. for each input xt in X
3.     z ← Wx xt + Wh h + b
4.     h ← tanh(z)
5. return h
```

The important idea is that `h` is updated in place at every time step.

The next input always receives the hidden state produced by the previous input.

---

### Characteristics / Key Points

- RNNs are designed for sequential data.
- The hidden state acts as a form of recurrent memory.
- The current hidden state depends on the current input.
- The current hidden state also depends on the previous hidden state.
- The same parameters are reused across all time steps.
- The input-to-hidden transformation uses $W_x$.
- The hidden-to-hidden transformation uses $W_h$.
- The bias is represented by $b$.
- The `tanh` activation is applied element-wise.
- The hidden state is bounded approximately between `-1` and `1`.
- The final hidden state summarizes information propagated through the sequence.
- The sequence length can vary without changing the model parameters.
- The computation is inherently sequential across time steps.
- RNNs can model temporal dependencies.
- Simple RNNs can struggle with long-term dependencies.
- Long sequences can lead to vanishing or exploding gradients during training.
- Parameter sharing makes RNNs more parameter-efficient than using independent networks for every time step.

---

### Recurrent Dependency

The defining characteristic of an RNN is the dependency

$$h_t = f(x_t,h_{t-1})$$

This means that changing an earlier input can potentially change every later hidden state.

For example,

$$x_1 \rightarrow h_1 \rightarrow h_2 \rightarrow h_3$$

Therefore, information from $x_1$ can influence the final state $h_3$.

This recurrent dependency is what distinguishes an RNN from a standard feed-forward network.

---

### Many-to-One Sequence Processing

The problem returns only the final hidden state.

This corresponds to a common **many-to-one** sequence-processing pattern.

For example,

```text
x1 ─┐
x2 ─┤
x3 ─┤──► RNN ──► final hidden state
x4 ─┤
x5 ─┘
```

The complete sequence is processed, but a single representation is produced at the end.

Many-to-one RNN architectures can be useful for tasks such as sequence classification.

For example,

```text
Words in a sentence → RNN → sentiment prediction
```

The final hidden state can act as a representation of the sequence.

---

### Many-to-Many Processing

RNNs can also produce an output at every time step.

Instead of returning only

```text
hT
```

the network can produce

```text
h1, h2, ..., hT
```

This is useful for tasks such as sequence labeling.

For example,

```text
Input words → RNN → output for every word
```

Applications include named entity recognition and part-of-speech tagging.

---

### RNNs and Variable-Length Sequences

Because the same recurrence is applied repeatedly, the RNN can process sequences of different lengths.

For a sequence of length $T$,

$$h_T = f(x_T,h_{T-1})$$

For another sequence of length $T+5$,

$$h_{T+5} = f(x_{T+5},h_{T+4})$$

The parameter matrices remain unchanged.

Only the number of recurrence steps changes.

This is an important advantage when working with real sequential data.

---

### Vanishing Gradient Problem

During training, gradients are propagated backward through many time steps.

Because the hidden state depends recursively on previous states, gradients can involve repeated multiplication.

If these factors have magnitude smaller than one, gradients can become extremely small.

This is called the **vanishing gradient problem**.

As a result, a simple RNN may have difficulty learning long-range dependencies.

For example, an RNN may struggle to remember information from the beginning of a very long sequence when processing the end.

---

### Exploding Gradient Problem

The opposite can also happen.

If repeated transformations amplify the gradient, its magnitude can grow very large.

This is called the **exploding gradient problem**.

Gradient clipping is a common technique used to limit excessively large gradients during RNN training.

The forward pass implemented in this problem does not perform gradient computation, so gradient clipping is not required here.

---

### RNN vs Feed-Forward Network

A feed-forward network processes each input without maintaining recurrent state.

An RNN instead maintains

$$h_t$$

and uses it when processing the next input.

A feed-forward computation can be viewed as

```text
x → network → output
```

while an RNN follows

```text
x1 → h1
      ↓
x2 → h2
      ↓
x3 → h3
```

The recurrent connection gives the RNN access to previous information.

---

### RNN vs LSTM / GRU

A simple RNN uses a single hidden state update

$$h_t = \tanh(W_xx_t + W_hh_{t-1} + b)$$

**LSTM** and **GRU** architectures introduce additional mechanisms for controlling information flow.

LSTMs use gates such as

- Input gate
- Forget gate
- Output gate

GRUs use

- Update gate
- Reset gate

These architectures were developed partly to handle long-term dependencies more effectively than a simple RNN.

The RNN in this problem is intentionally simplified so that the fundamental recurrence is clear.

---

### Why is it used? / Applications

RNNs were widely used for sequential machine learning problems.

Common applications include

- Natural Language Processing
- Sentiment Analysis
- Language Modeling
- Speech Recognition
- Time-Series Forecasting
- Sequence Classification
- Sequence Labeling
- Machine Translation
- Handwriting Recognition
- Event Prediction
- Sensor Data Analysis
- Financial Time-Series Modeling

Modern systems often use architectures such as Transformers, LSTMs, or GRUs instead of basic RNNs for demanding sequence tasks.

However, the simple RNN remains an important concept because it introduces the fundamental idea of maintaining state across time.

---

> 💡 **Important Note**
>
> An RNN does not store the entire sequence explicitly inside the hidden state. Instead, it repeatedly transforms the previous hidden state together with the current input. The hidden state therefore acts as a learned compressed representation of information from the sequence processed so far.

---

> 💡 **Interview Tip**
>
> If asked what makes an RNN different from a feed-forward neural network, the key answer is **recurrent state**. The hidden state from the previous time step is fed into the computation of the current hidden state, allowing information to flow across the sequence.

---

> 💡 **Practical Insight**
>
> The parameters $W_x$, $W_h$, and $b$ are shared across time steps. This means an RNN does not learn a separate set of parameters for position 1, position 2, position 3, and so on. The same recurrent cell is reused throughout the sequence.

---

## Solutions

### Custom Implementation

```python
import numpy as np

def rnn_forward(
    input_sequence: list[list[float]],
    initial_hidden_state: list[float],
    Wx: list[list[float]],
    Wh: list[list[float]],
    b: list[float]
) -> list[float]:

    output = np.array(
        initial_hidden_state,
        dtype=float
    )

    for word_vector in input_sequence:
        wv = np.array(
            word_vector,
            dtype=float
        )

        output = np.tanh(
            np.dot(Wh, output)
            + np.dot(Wx, wv)
            + b
        )

    return np.round(
        output,
        4
    ).tolist()
```

### NumPy Implementation

The solution directly uses NumPy for the matrix-vector operations.

```python
import numpy as np

def rnn_forward(
    input_sequence,
    initial_hidden_state,
    Wx,
    Wh,
    b
):
    h = np.asarray(
        initial_hidden_state,
        dtype=float
    )

    Wx = np.asarray(Wx, dtype=float)
    Wh = np.asarray(Wh, dtype=float)
    b = np.asarray(b, dtype=float)

    for x in input_sequence:
        x = np.asarray(x, dtype=float)

        h = np.tanh(
            Wx @ x
            + Wh @ h
            + b
        )

    return np.round(h, 4).tolist()
```

The `@` operator performs matrix multiplication.

Therefore,

```python
Wx @ x
```

represents the input contribution

$$W_xx_t$$

while

```python
Wh @ h
```

represents the recurrent contribution

$$W_hh_{t-1}$$

The final update is equivalent to

```python
h = np.tanh(Wx @ x + Wh @ h + b)
```

---

## Code Explanation

### Step 1: Initialize the Hidden State

The initial hidden state is converted into a NumPy array.

```python
output = np.array(
    initial_hidden_state,
    dtype=float
)
```

At the beginning,

```text
output = h0
```

This is the state that will be used when processing the first input.

---

### Step 2: Iterate Through the Sequence

The function processes each input vector sequentially.

```python
for word_vector in input_sequence:
```

If the sequence contains `T` input vectors, this loop executes exactly `T` times.

The order of the inputs matters because each iteration modifies the hidden state used by the next iteration.

---

### Step 3: Convert the Input to a NumPy Array

Each input vector is converted into a floating-point NumPy array.

```python
wv = np.array(
    word_vector,
    dtype=float
)
```

This allows NumPy to perform the required vector and matrix operations.

---

### Step 4: Compute the Input Contribution

The current input is multiplied by the input-to-hidden weight matrix.

```python
np.dot(Wx, wv)
```

This corresponds to

$$W_xx_t$$

It determines how the current input affects the hidden state.

---

### Step 5: Compute the Recurrent Contribution

The previous hidden state is multiplied by the hidden-to-hidden weight matrix.

```python
np.dot(Wh, output)
```

This corresponds to

$$W_hh_{t-1}$$

It carries information from previous time steps into the current computation.

---

### Step 6: Add the Bias

The bias vector is added to the two weighted contributions.

```python
np.dot(Wh, output) + np.dot(Wx, wv) + b
```

The complete pre-activation state is therefore

$$z_t = W_xx_t + W_hh_{t-1} + b$$

---

### Step 7: Apply Tanh

The pre-activation state is passed through `np.tanh`.

```python
output = np.tanh(...)
```

This implements

$$h_t = \tanh(z_t)$$

or equivalently

$$h_t = \tanh(W_xx_t + W_hh_{t-1} + b)$$

The resulting vector becomes the hidden state for the next time step.

---

### Step 8: Reuse the Hidden State

After processing one input,

```python
output
```

contains the new hidden state.

On the next iteration, it becomes

```text
previous hidden state
```

This creates the recurrent dependency

```text
h0 → h1 → h2 → h3 → ... → hT
```

This is the central operation of the RNN.

---

### Step 9: Process the Complete Sequence

The loop continues until every input vector has been processed.

For a sequence

```text
[x1, x2, x3]
```

the implementation performs

```text
h0 → h1 → h2 → h3
```

where

$$h_1 = \tanh(W_xx_1 + W_hh_0 + b)$$

$$h_2 = \tanh(W_xx_2 + W_hh_1 + b)$$

$$h_3 = \tanh(W_xx_3 + W_hh_2 + b)$$

---

### Step 10: Return the Final Hidden State

After the loop finishes, `output` contains the final hidden state.

```python
return np.round(
    output,
    4
).tolist()
```

The result is rounded to four decimal places as required by the problem.

The NumPy array is then converted back into a Python list.

---

### Complete Computation Flow

The implementation can be summarized as

```text
Initial hidden state
        │
        ▼
      h0
        │
        ▼
   ┌─────────┐
x1 │  RNN    │ → h1
   └─────────┘
        │
        ▼
   ┌─────────┐
x2 │  RNN    │ → h2
   └─────────┘
        │
        ▼
   ┌─────────┐
x3 │  RNN    │ → h3
   └─────────┘
        │
        ▼
 Final hidden state
```

At every RNN cell, the same operation is performed:

$$h_t = \tanh(W_xx_t + W_hh_{t-1} + b)$$

---

### Why the Loop Is Necessary

The RNN recurrence depends on the previous hidden state.

Therefore, the computation cannot simply process every input independently.

The second hidden state requires the first hidden state:

$$h_2 = f(x_2,h_1)$$

The third hidden state requires the second:

$$h_3 = f(x_3,h_2)$$

This dependency is why the sequence is processed in order.

---

### Why Parameters Are Not Updated Inside the Loop

The loop performs the **forward pass** only.

The matrices

```python
Wx
Wh
```

and the bias

```python
b
```

are used as fixed parameters during this computation.

During actual neural network training, these parameters would be updated by an optimizer after computing gradients through **Backpropagation Through Time (BPTT)**.

This problem only asks us to implement the forward recurrence.

---

### Backpropagation Through Time

During training, the recurrent computation is conceptually unrolled across time.

For example,

```text
h0 → h1 → h2 → h3
     ↑    ↑    ↑
     x1   x2   x3
```

The gradients are propagated backward through these recurrent connections.

This process is called **Backpropagation Through Time**.

The same shared parameters receive gradient contributions from multiple time steps.

Although BPTT is not required for this problem, understanding it explains why RNNs can suffer from vanishing and exploding gradients.

---

## Time & Space Complexity

Let

- $T$ be the number of time steps.
- $D$ be the input dimension.
- $H$ be the hidden-state dimension.
- $W_x$ have dimensions $H \times D$.
- $W_h$ have dimensions $H \times H$.

At each time step, the implementation performs two matrix-vector multiplications.

The input transformation costs

$$O(HD)$$

The recurrent transformation costs

$$O(H^2)$$

Therefore, for a sequence of length $T$, the total time complexity is

$$O(T(HD + H^2))$$

The implementation only needs the current hidden state while processing the sequence.

Therefore, for the forward pass itself, the working space is

$$O(H)$$

excluding the input sequence and parameter storage.

The parameter storage is

$$O(HD + H^2 + H)$$

where

- $HD$ stores $W_x$.
- $H^2$ stores $W_h$.
- $H$ stores the bias vector.

| Complexity      | Value              |
| --------------- | ------------------ |
| Time            | **O(T(HD + H²))**  |
| Working Space   | **O(H)**           |
| Parameter Space | **O(HD + H² + H)** |

where

- $T$ is the sequence length.
- $D$ is the input-vector dimension.
- $H$ is the hidden-state dimension.

For the specific scalar example,

```text
T = 3
D = 1
H = 1
```

so every time step performs only constant-sized operations.

The important general result is that the computational cost grows **linearly with the sequence length** $T$, while the recurrent matrix multiplication contributes the $H^2$ term.
