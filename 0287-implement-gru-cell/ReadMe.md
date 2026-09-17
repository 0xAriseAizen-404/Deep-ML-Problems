# Implement GRU Cell (Medium, Deep Learning)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Gated Recurrent Unit](#learn-gated-recurrent-unit)
  - [What is a GRU?](#what-is-a-gru)
  - [GRU Architecture](#gru-architecture)
  - [Mathematical Definition](#mathematical-definition)
  - [Update Gate](#update-gate)
  - [Reset Gate](#reset-gate)
  - [Candidate Hidden State](#candidate-hidden-state)
  - [New Hidden State](#new-hidden-state)
  - [Gate Intuition](#gate-intuition)
  - [GRU vs. Vanilla RNN](#gru-vs-vanilla-rnn)
  - [GRU vs. LSTM](#gru-vs-lstm)
  - [Parameter Count](#parameter-count)
  - [Applications](#applications)

- [Solutions](#solutions)
  - [Custom NumPy Implementation](#custom-numpy-implementation)
  - [Vectorized Implementation](#vectorized-implementation)

- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

---

## Problem Statement

### [Implement GRU Cell](https://www.deep-ml.com/problems/287)

Implement the forward pass of a Gated Recurrent Unit (GRU) cell.

A GRU is a recurrent neural network architecture that uses gates to control information flow through time.

The cell receives:

- An input vector.
- The previous hidden state.
- Input-to-hidden weight matrices.
- Hidden-to-hidden weight matrices.
- Bias vectors.

It returns the next hidden state.

### Input Parameters

| Parameter | Shape                        | Description                   |
| --------- | ---------------------------- | ----------------------------- |
| `x`       | `(input_size,)`              | Current input vector          |
| `h_prev`  | `(hidden_size,)`             | Previous hidden state         |
| `W_z`     | `(hidden_size, input_size)`  | Update gate input weights     |
| `W_r`     | `(hidden_size, input_size)`  | Reset gate input weights      |
| `W_h`     | `(hidden_size, input_size)`  | Candidate input weights       |
| `U_z`     | `(hidden_size, hidden_size)` | Update gate recurrent weights |
| `U_r`     | `(hidden_size, hidden_size)` | Reset gate recurrent weights  |
| `U_h`     | `(hidden_size, hidden_size)` | Candidate recurrent weights   |
| `b_z`     | `(hidden_size,)`             | Update gate bias              |
| `b_r`     | `(hidden_size,)`             | Reset gate bias               |
| `b_h`     | `(hidden_size,)`             | Candidate bias                |

### Output

The function returns:

```text
h_next
```

The output has shape:

```text
(hidden_size,)
```

The GRU uses two gates:

1. Update gate.
2. Reset gate.

The update gate controls the mixture of the previous hidden state and the candidate hidden state.

The reset gate controls how much of the previous hidden state is used when calculating the candidate.

---

## Example

### Input

```python
x = np.array([1.0, 0.5])
h_prev = np.array([0.0, 0.0, 0.0])
```

Assume:

- All input weight matrices contain `0.1`.
- All recurrent weight matrices contain `0.2`.
- All biases are zero.
- The hidden state contains zeros.

### Output

```text
[0.1565, 0.1565, 0.1565]
```

### Reasoning

Since the previous hidden state is zero, the recurrent terms contribute nothing to the gate calculations.

For the update gate:

\(z_t = \sigma(0.1(1.0) + 0.1(0.5))\)

Therefore:

\(z_t = \sigma(0.15) \approx 0.5374\)

The reset gate has the same value in this example:

\(r_t = \sigma(0.15) \approx 0.5374\)

However, the reset gate does not affect the candidate because:

\(r*t \odot h*{prev} = 0\)

The candidate hidden state is:

\(\tilde{h}\_t = \tanh(0.1(1.0) + 0.1(0.5))\)

Thus:

\(\tilde{h}\_t = \tanh(0.15) \approx 0.1489\)

The new hidden state is:

\(h*t = (1-z_t)\odot h*{prev} + z_t\odot\tilde{h}\_t\)

Because the previous hidden state is zero:

\(h_t = z_t\odot\tilde{h}\_t\)

Therefore:

\(h_t \approx 0.5374 \times 0.1489 \approx 0.0800\)

The exact numerical output depends on the weight values used for each gate.

> **Important:** The stated output of `0.1565` corresponds to a candidate pre-activation of approximately `0.3`, not `0.15`. Always verify the supplied weight assumptions against the equations before accepting a numerical example.

---

# Learn: Gated Recurrent Unit

## What is a GRU?

A Gated Recurrent Unit (GRU) is a recurrent neural network architecture introduced by Cho and collaborators in 2014.

It was designed to improve the ability of recurrent networks to preserve information over long sequences.

A vanilla RNN repeatedly updates its hidden state using a nonlinear transformation.

This repeated transformation can cause:

- Vanishing gradients.
- Exploding gradients.
- Difficulty learning long-term dependencies.
- Loss of earlier information.

A GRU introduces gates that regulate the hidden-state update.

Unlike an LSTM, a GRU does not maintain a separate cell state.

It uses one hidden state to represent both memory and output.

The hidden state is updated using:

- The previous hidden state.
- The current input.
- The update gate.
- The reset gate.
- The candidate hidden state.

---

## GRU Architecture

A GRU contains two gates.

| Gate        | Symbol | Main Function                                                       |
| ----------- | ------ | ------------------------------------------------------------------- |
| Update gate | `z_t`  | Controls how much new candidate information is used                 |
| Reset gate  | `r_t`  | Controls how much previous information contributes to the candidate |

The candidate hidden state is not a gate.

It is a proposed new hidden-state representation generated from the input and a gated version of the previous hidden state.

The complete sequence of operations is:

```text
Input and previous state
          |
          v
   Update gate z_t
   Reset gate r_t
          |
          v
 Candidate hidden state
          |
          v
 Interpolate old and new states
          |
          v
      h_next
```

The GRU performs a learned interpolation between the previous hidden state and the candidate hidden state.

---

## Mathematical Definition

Let:

- `x_t` be the current input.
- `h_{t-1}` be the previous hidden state.
- `z_t` be the update gate.
- `r_t` be the reset gate.
- `\tilde{h}_t` be the candidate hidden state.
- `h_t` be the new hidden state.

The update gate is:

\(z*t = \sigma(W_zx_t + U_zh*{t-1} + b_z)\)

The reset gate is:

\(r*t = \sigma(W_rx_t + U_rh*{t-1} + b_r)\)

The candidate hidden state is:

\(\tilde{h}_t = \tanh(W_hx_t + U_h(r_t \odot h_{t-1}) + b_h)\)

The new hidden state is:

\(h*t = (1-z_t)\odot h*{t-1} + z_t\odot\tilde{h}\_t\)

The symbol `⊙` represents element-wise multiplication.

The sigmoid function is:

\(\sigma(a) = \frac{1}{1+e^{-a}}\)

The sigmoid output is in the range:

\(\sigma(a) \in (0,1)\)

The hyperbolic tangent function is:

\(\tanh(a) \in (-1,1)\)

---

## Update Gate

The update gate determines how strongly the candidate hidden state influences the next hidden state.

\(z*t = \sigma(W_zx_t + U_zh*{t-1} + b_z)\)

The update gate has one value for every hidden-state dimension.

The final state uses:

\((1-z*t)\odot h*{t-1}\)

for the old state, and:

\(z_t\odot\tilde{h}\_t\)

for the candidate state.

If an update gate value is close to zero:

- The previous hidden state is mostly retained.
- The candidate contributes very little.
- The model performs a small update.

If an update gate value is close to one:

- The candidate hidden state is mostly selected.
- The previous hidden state contributes very little.
- The model performs a stronger update.

The update gate therefore controls the balance between memory preservation and new information.

---

## Reset Gate

The reset gate controls how much of the previous hidden state is used to calculate the candidate.

\(r*t = \sigma(W_rx_t + U_rh*{t-1} + b_r)\)

The gated previous hidden state is:

\(r*t \odot h*{t-1}\)

This quantity is passed through the recurrent candidate transformation:

\(U*h(r_t \odot h*{t-1})\)

If the reset gate is close to zero:

\(r*t \odot h*{t-1} \approx 0\)

The candidate depends mainly on the current input and candidate bias.

If the reset gate is close to one:

\(r*t \odot h*{t-1} \approx h\_{t-1}\)

The candidate can use most of the previous hidden state.

The reset gate therefore controls the amount of historical information considered during candidate generation.

---

## Candidate Hidden State

The candidate hidden state represents a possible new state.

\(\tilde{h}_t = \tanh(W_hx_t + U_h(r_t \odot h_{t-1}) + b_h)\)

The candidate combines:

- The current input.
- The reset-gated previous hidden state.
- The candidate weights.
- The candidate bias.

The candidate is passed through `tanh`.

This bounds its values between `-1` and `1`.

The candidate does not automatically replace the previous hidden state.

The update gate decides how much of it is used.

This separation allows the GRU to generate new information without necessarily overwriting the entire hidden state.

---

## New Hidden State

The final hidden state is calculated as:

\(h*t = (1-z_t)\odot h*{t-1} + z_t\odot\tilde{h}\_t\)

This equation is a convex interpolation when the gate values lie between zero and one.

The coefficients for the previous state and candidate state sum to one:

\((1-z_t) + z_t = 1\)

For each hidden dimension, the GRU chooses a mixture of:

- The previous hidden state.
- The candidate hidden state.

This operation is element-wise.

Different hidden dimensions can update at different rates.

Some dimensions may preserve information while others replace it with newly computed values.

---

## Gate Intuition

### Update Gate Near Zero

If:

\(z_t \approx 0\)

Then:

\(h*t \approx h*{t-1}\)

The previous state is preserved.

This behavior can help the model maintain information across many time steps.

---

### Update Gate Near One

If:

\(z_t \approx 1\)

Then:

\(h_t \approx \tilde{h}\_t\)

The candidate state becomes the primary source of the new hidden state.

This allows the model to incorporate new information.

---

### Reset Gate Near Zero

If:

\(r_t \approx 0\)

Then:

\(r*t \odot h*{t-1} \approx 0\)

The candidate mostly ignores the previous hidden state.

This can be useful when the model needs to process a new segment or context.

---

### Reset Gate Near One

If:

\(r_t \approx 1\)

Then:

\(r*t \odot h*{t-1} \approx h\_{t-1}\)

The candidate fully uses the previous hidden state.

This allows historical information to influence the candidate.

---

## GRU vs. Vanilla RNN

A vanilla RNN commonly uses:

\(h*t = \tanh(W_xx_t + W_hh*{t-1} + b)\)

The entire previous hidden state is transformed at every time step.

A GRU introduces separate control mechanisms.

| Feature                | Vanilla RNN         | GRU                      |
| ---------------------- | ------------------- | ------------------------ |
| Number of states       | One hidden state    | One hidden state         |
| Gates                  | None                | Two                      |
| Memory control         | Implicit            | Explicit through gates   |
| Candidate computation  | Direct state update | Reset-gated state update |
| Long-term dependencies | Often difficult     | Generally easier         |
| Parameter count        | Lower               | Higher                   |

The GRU adds parameters and computation but provides more control over information retention.

The gates do not guarantee that every long-term dependency will be learned successfully.

Training behavior still depends on the data, architecture, initialization, and optimization process.

---

## GRU vs. LSTM

Both GRUs and LSTMs use gating mechanisms.

The main structural difference is that an LSTM maintains separate hidden and cell states, while a GRU uses a single hidden state.

| Feature             | GRU           | LSTM                                   |
| ------------------- | ------------- | -------------------------------------- |
| Gates               | Two           | Three main gates                       |
| Hidden state        | Yes           | Yes                                    |
| Separate cell state | No            | Yes                                    |
| Parameter count     | Usually lower | Usually higher                         |
| State update        | Interpolation | Cell-state update and output filtering |
| Architecture        | Simpler       | More expressive state structure        |

A standard GRU has three sets of weight matrices:

- Update gate.
- Reset gate.
- Candidate state.

A standard LSTM has four sets of gate-related parameters:

- Input gate.
- Forget gate.
- Cell candidate.
- Output gate.

For input size `I` and hidden size `H`, the GRU parameter count is:

\(3HI + 3H^2 + 3H\)

The standard LSTM parameter count with two bias vectors is:

\(4HI + 4H^2 + 8H\)

The exact count depends on the implementation and whether one or two bias vectors are used.

---

## Parameter Count

Let:

- `I` represent the input size.
- `H` represent the hidden size.

The three input weight matrices each contain:

\(HI\)

parameters.

The total input-weight parameters are:

\(3HI\)

The three recurrent weight matrices each contain:

\(H^2\)

parameters.

The total recurrent-weight parameters are:

\(3H^2\)

Each bias vector contains `H` parameters.

The total bias parameters are:

\(3H\)

Therefore, the total number of parameters is:

\(3HI + 3H^2 + 3H\)

Factoring the expression:

\(3H(I+H+1)\)

---

## Applications

GRUs can be used for:

- Sequence classification.
- Sentiment analysis.
- Language modeling.
- Speech processing.
- Time-series forecasting.
- Sensor data modeling.
- Anomaly detection.
- Sequence prediction.
- Streaming inference.
- Recurrent encoder-decoder models.

GRUs can be useful when a recurrent model is required but a simpler architecture than an LSTM is preferred.

The choice between GRU, LSTM, and Transformer architectures depends on:

- Sequence length.
- Dataset size.
- Latency requirements.
- Available memory.
- Training resources.
- Need for streaming inference.
- Model accuracy requirements.

---

> 💡 **Important Note**
>
> The update-gate convention used here is:
>
> \(h*t = (1-z_t)\odot h*{t-1} + z_t\odot\tilde{h}\_t\)
>
> Some GRU implementations use a different naming convention where the update gate represents the amount of old information retained:
>
> \(h*t = z_t\odot h*{t-1} + (1-z_t)\odot\tilde{h}\_t\)
>
> These equations are equivalent only when the gate definition is changed accordingly. Always follow the convention specified by the problem.

---

# Solutions

## Custom NumPy Implementation

```python id="f7k2za"
import numpy as np

def gru_cell(x: np.ndarray, h_prev: np.ndarray,
             W_z: np.ndarray, U_z: np.ndarray, b_z: np.ndarray,
             W_r: np.ndarray, U_r: np.ndarray, b_r: np.ndarray,
             W_h: np.ndarray, U_h: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def tanh(z):
        return np.tanh(z)

    r_t = sigmoid(W_r @ x + U_r @ h_prev + b_r)
    z_t = sigmoid(W_z @ x + U_z @ h_prev + b_z)
    h_tilde = tanh(W_h @ x + U_h @ (r_t * h_prev) + b_h)
    h_next = (1 - z_t) * h_prev + z_t * h_tilde

    return h_next
```

---

## Vectorized Implementation

The matrix multiplications can be combined by concatenating the input and hidden state.

Define:

\(q*t = [x_t;h*{t-1}]\)

The concatenated vector has shape:

```text
(input_size + hidden_size,)
```

The corresponding combined weights can be constructed for each GRU component.

```python id="q8g5t1"
import numpy as np

def gru_cell_vectorized(x, h_prev, W_z, U_z, b_z,
                       W_r, U_r, b_r, W_h, U_h, b_h):
    def sigmoid(value):
        return 1 / (1 + np.exp(-value))

    combined = np.concatenate([x, h_prev])

    Wz_combined = np.concatenate([W_z, U_z], axis=1)
    Wr_combined = np.concatenate([W_r, U_r], axis=1)
    Wh_combined = np.concatenate([W_h, U_h], axis=1)

    z_t = sigmoid(Wz_combined @ combined + b_z)
    r_t = sigmoid(Wr_combined @ combined + b_r)
    h_tilde = np.tanh(
        Wh_combined @ np.concatenate([x, r_t * h_prev]) + b_h
    )

    return (1 - z_t) * h_prev + z_t * h_tilde
```

The first implementation is simpler because the candidate uses a reset-gated hidden state.

The second implementation requires special handling for the candidate because the reset gate is applied only to the hidden-state portion.

---

# Code Explanation

## 1. Define the Activation Functions

The sigmoid function is used for both gates.

```python id="x2l8wa"
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

The sigmoid function maps each input to a value between zero and one.

The candidate uses NumPy's hyperbolic tangent implementation:

```python id="7g7yqz"
def tanh(z):
    return np.tanh(z)
```

The candidate output is bounded between negative one and one.

---

## 2. Compute the Reset Gate

```python id="3k8v9p"
r_t = sigmoid(W_r @ x + U_r @ h_prev + b_r)
```

The reset gate combines:

- The current input.
- The previous hidden state.
- Input-to-hidden weights.
- Hidden-to-hidden weights.
- A bias vector.

The matrix multiplication:

```python id="q3h5c0"
W_r @ x
```

produces a vector of shape:

```text
(hidden_size,)
```

The recurrent term:

```python id="j2k5x8"
U_r @ h_prev
```

has the same shape.

The two vectors and the bias are added before applying sigmoid.

---

## 3. Compute the Update Gate

```python id="a0o7ms"
z_t = sigmoid(W_z @ x + U_z @ h_prev + b_z)
```

The update gate uses the same structure as the reset gate but has independent parameters.

The gate controls the interpolation between the old hidden state and the candidate hidden state.

The update gate is computed before the final state update.

---

## 4. Apply the Reset Gate

```python id="h7w3qk"
r_t * h_prev
```

This is an element-wise multiplication.

Each hidden-state component is scaled by the corresponding reset-gate value.

The result is passed into the candidate recurrent transformation:

```python id="f0d4x1"
U_h @ (r_t * h_prev)
```

This operation allows the model to control how much previous information contributes to candidate generation.

---

## 5. Compute the Candidate Hidden State

```python id="n4x6qa"
h_tilde = tanh(
    W_h @ x
    + U_h @ (r_t * h_prev)
    + b_h
)
```

The candidate combines two sources:

```text
Current input contribution
Reset-gated hidden-state contribution
```

The `tanh` activation generates the proposed new hidden representation.

The candidate is not directly returned.

It is combined with the previous hidden state through the update gate.

---

## 6. Compute the New Hidden State

```python id="u1s9pl"
h_next = (1 - z_t) * h_prev + z_t * h_tilde
```

The first term retains information from the previous state:

```python id="k9w5dt"
(1 - z_t) * h_prev
```

The second term incorporates new candidate information:

```python id="r2c6vm"
z_t * h_tilde
```

The two terms are added element by element.

The resulting vector has shape:

```text
(hidden_size,)
```

---

## 7. Why the Matrix Multiplication Uses `@`

The `@` operator performs matrix multiplication in NumPy.

For the input transformation:

```python id="g5t8zv"
W_z @ x
```

the dimensions are:

```text
(hidden_size, input_size) @ (input_size,)
```

The result is:

```text
(hidden_size,)
```

For the recurrent transformation:

```python id="d6q1mk"
U_z @ h_prev
```

the dimensions are:

```text
(hidden_size, hidden_size) @ (hidden_size,)
```

The result is also:

```text
(hidden_size,)
```

The shapes are compatible with element-wise addition and activation functions.

---

## 8. Why the Reset Gate Is Applied Before the Recurrent Transformation

The candidate equation is:

\(\tilde{h}_t = \tanh(W_hx_t + U_h(r_t\odot h_{t-1}) + b_h)\)

The reset gate is applied to the previous hidden state before multiplication by `U_h`.

This is different from applying the reset gate to the result of the recurrent matrix multiplication.

The specified operation is:

```python id="j7w0cn"
U_h @ (r_t * h_prev)
```

It is not:

```python id="b1d4ye"
r_t * (U_h @ h_prev)
```

These operations generally produce different results because matrix multiplication mixes hidden-state dimensions.

Following the equation exactly is necessary for correctness.

---

## 9. Why the Update Gate Uses Complementary Coefficients

The final state equation is:

\(h*t = (1-z_t)\odot h*{t-1} + z_t\odot\tilde{h}\_t\)

The coefficient for the previous state is:

\(1-z_t\)

The coefficient for the candidate is:

\(z_t\)

Their sum is one.

This creates an interpolation between the old and candidate states.

When `z_t` is small, the old state receives more weight.

When `z_t` is large, the candidate receives more weight.

---

## 10. Numerical Stability Considerations

The sigmoid implementation:

```python id="x6v1py"
1 / (1 + np.exp(-z))
```

is mathematically correct.

However, very large positive or negative values can cause numerical overflow in `np.exp`.

A numerically safer implementation can use a piecewise formulation or a library-provided stable sigmoid function.

For ordinary educational inputs, the direct implementation is usually sufficient.

In production code, numerical stability should be considered when activations may become very large.

---

## 11. Common Implementation Mistakes

### Mistake 1: Swapping the Gate Convention

Different sources may define the update gate differently.

Always use the equation provided by the problem.

---

### Mistake 2: Applying Reset Gate to the Wrong Quantity

The reset gate should be applied to the previous hidden state:

```python id="w6h9yr"
U_h @ (r_t * h_prev)
```

It should not generally be applied after the matrix multiplication.

---

### Mistake 3: Using the Wrong Activation

The gates use sigmoid:

```python id="u9k3cs"
z_t = sigmoid(...)
r_t = sigmoid(...)
```

The candidate uses `tanh`:

```python id="k8r5xd"
h_tilde = np.tanh(...)
```

Using sigmoid for the candidate changes its value range and behavior.

---

### Mistake 4: Replacing the Previous State Completely

The new hidden state should combine both sources:

```python id="y3x7vp"
(1 - z_t) * h_prev + z_t * h_tilde
```

Returning only the candidate would remove the update-gate mechanism.

---

### Mistake 5: Using Matrix Multiplication for Gate Products

The gate-state operation is element-wise:

```python id="e6n2qw"
r_t * h_prev
```

It should not use matrix multiplication.

---

# Time & Space Complexity

Let:

- `I` be the input size.
- `H` be the hidden size.

Each input-to-hidden matrix multiplication has complexity:

\(O(HI)\)

There are three input-to-hidden transformations:

\(O(3HI) = O(HI)\)

Each hidden-to-hidden matrix multiplication has complexity:

\(O(H^2)\)

There are three recurrent transformations:

\(O(3H^2) = O(H^2)\)

The element-wise gate operations have complexity:

\(O(H)\)

The total time complexity is:

\(O(HI + H^2)\)

The constant factor of three is omitted in Big-O notation.

The parameter storage includes:

- Three input weight matrices.
- Three recurrent weight matrices.
- Three bias vectors.

The total parameter space is:

\(O(HI + H^2 + H)\)

The intermediate vectors have size `H`.

Therefore, the working space for one forward pass is:

\(O(HI + H^2 + H)\)

If the parameters are excluded and only temporary activations are counted, the additional forward-pass space is:

\(O(H)\)

For a sequence containing `T` time steps, the forward computation requires:

\(O(TH(I+H))\)

The recurrent state remains fixed-size per time step, although training with backpropagation through time may require storing intermediate states.
