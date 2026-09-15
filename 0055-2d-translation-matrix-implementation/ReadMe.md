# Implement Long Short-Term Memory (LSTM) Network (Medium, Deep Learning)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Understanding Long Short-Term Memory Networks](#learn-understanding-long-short-term-memory-networks)
  - [What is an LSTM?](#what-is-an-lstm)
  - [Why RNNs Need LSTMs](#why-rnns-need-lstms)
  - [LSTM Architecture](#lstm-architecture)
  - [Forget Gate](#forget-gate)
  - [Input Gate](#input-gate)
  - [Candidate Cell State](#candidate-cell-state)
  - [Cell State Update](#cell-state-update)
  - [Output Gate](#output-gate)
  - [Hidden State](#hidden-state)
  - [Complete LSTM Equations](#complete-lstm-equations)
  - [Role of the Cell State](#role-of-the-cell-state)
  - [Sigmoid and Tanh](#sigmoid-and-tanh)
  - [Implementation Flow](#implementation-flow)
  - [Step-by-Step Example](#step-by-step-example)
  - [Characteristics / Key Points](#characteristics--key-points)
  - [Why is it used? / Applications](#why-is-it-used--applications)
- [Solutions](#solutions)
  - [Custom Implementation](#custom-implementation)
  - [NumPy / Deep Learning Equivalent](#numpy--deep-learning-equivalent)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

---

## Problem Statement

### [Implement Long Short-Term Memory (LSTM) Network](https://www.deep-ml.com/problems/59)

Implement an **LSTM network** that processes a sequence of input vectors and produces the hidden states for every time step along with the final hidden state and cell state.

The implementation should define an `LSTM` class with:

- `__init__(input_size, hidden_size)` to initialize the gate weights and biases.
- `forward(x, initial_hidden_state, initial_cell_state)` to process the complete input sequence.

At every time step, the LSTM should calculate:

- Forget gate
- Input gate
- Candidate cell state
- Updated cell state
- Output gate
- Updated hidden state

The final result should contain the hidden state at every time step, the final hidden state, and the final cell state.

---

## Example

### Input

```python
import numpy as np

input_sequence = np.array([
    [1.0],
    [2.0],
    [3.0]
])

initial_hidden_state = np.zeros((1, 1))
initial_cell_state = np.zeros((1, 1))

lstm = LSTM(
    input_size=1,
    hidden_size=1
)

outputs, final_h, final_c = lstm.forward(
    input_sequence,
    initial_hidden_state,
    initial_cell_state
)

print(final_h)
```

### Output

```text
[[0.73698596]]
```

The exact output can vary because the LSTM weights are randomly initialized.

### Reasoning

The LSTM receives one input vector at each time step.

At the beginning,

```text
h0 = 0
c0 = 0
```

For every input, the previous hidden state and the current input are concatenated and passed through four different transformations.

The gates control how information flows through the network:

1. The **forget gate** determines which previous cell-state information should be retained.
2. The **input gate** determines which new information should be written.
3. The **candidate state** generates possible new information.
4. The **cell state** combines retained information with new information.
5. The **output gate** determines which information from the cell state becomes the new hidden state.

After processing `[1.0]`, `[2.0]`, and `[3.0]`, the final hidden state represents information accumulated from the complete sequence.

---

## Learn: Understanding Long Short-Term Memory Networks

### What is an LSTM?

A **Long Short-Term Memory (LSTM)** network is a specialized type of **Recurrent Neural Network (RNN)** designed to model sequential data while handling long-term dependencies more effectively.

A standard RNN maintains a hidden state that is repeatedly updated as new inputs arrive.

The basic RNN update is

$$
h_t=\tanh(W_xx_t+W_hh_{t-1}+b)
$$

The problem is that repeatedly applying the same transformation can make gradients become extremely small or extremely large during backpropagation through many time steps.

LSTMs address this problem by introducing a separate **cell state** and several gates that control how information is retained, removed, and exposed.

Instead of directly replacing the previous state, an LSTM uses learned gates to control information flow.

The two main states are:

- **Hidden state $h_t$**: the output representation exposed to the next layer and next time step.
- **Cell state $c_t$**: the internal memory that carries information through the sequence.

The LSTM therefore has two recurrent states rather than only one.

---

## Why RNNs Need LSTMs

A standard RNN repeatedly applies transformations to its hidden state.

For a long sequence, information from early time steps must pass through many recurrent operations before influencing a later output.

During backpropagation, gradients are repeatedly multiplied by weight matrices and derivatives of activation functions.

This can result in:

- Vanishing gradients.
- Exploding gradients.
- Difficulty learning long-term dependencies.
- Difficulty remembering information over many time steps.

LSTM introduces an explicit cell state with controlled additive updates.

The cell state is updated using

$$
c_t=f_t\circ c_{t-1}+i_t\circ\tilde{c}_t
$$

This structure allows information to be carried forward while the gates decide what should be forgotten or added.

The cell state is therefore often viewed as the LSTM's **long-term memory**.

---

## LSTM Architecture

At time step $t$, the LSTM receives:

- Current input $x_t$.
- Previous hidden state $h_{t-1}$.
- Previous cell state $c_{t-1}$.

The hidden state and input are concatenated:

$$
z_t=[h_{t-1},x_t]
$$

This combined vector is passed through four transformations.

The four main components are:

```text
                 x_t
                  |
                  v
          +---------------+
h_(t-1) ->|  Concatenate  |
          +---------------+
                  |
        +---------+---------+---------+
        |         |         |         |
        v         v         v         v
      Forget     Input   Candidate  Output
       Gate       Gate     State      Gate
        |          |         |         |
        +----------+----+----+---------+
                       |
                       v
                 Cell State c_t
                       |
                       v
                 Hidden State h_t
```

The four transformations are:

$$
f_t=\sigma(W_fz_t+b_f)
$$

$$
i_t=\sigma(W_iz_t+b_i)
$$

$$
\tilde{c}_t=\tanh(W_cz_t+b_c)
$$

$$
o_t=\sigma(W_oz_t+b_o)
$$

The cell state is then updated:

$$
c_t=f_t\circ c_{t-1}+i_t\circ\tilde{c}_t
$$

Finally, the hidden state is calculated:

$$
h_t=o_t\circ\tanh(c_t)
$$

---

## Forget Gate

The **forget gate** determines how much information from the previous cell state should be retained.

It is calculated as

$$
f_t=\sigma(W_f[h_{t-1},x_t]+b_f)
$$

The sigmoid function produces values between `0` and `1`.

Therefore, each element of the forget gate can be interpreted as a soft decision:

- Value close to `0` → forget most of the corresponding information.
- Value close to `1` → retain most of the corresponding information.
- Value around `0.5` → partially retain the information.

The forget gate is applied element-wise to the previous cell state.

$$
f_t\circ c_{t-1}
$$

For example, if

$$
c_{t-1}=[0.8,-0.5]
$$

and

$$
f_t=[1.0,0.1]
$$

then

$$
f_t\circ c_{t-1}=[0.8,-0.05]
$$

The first component is retained while most of the second component is forgotten.

---

## Input Gate

The **input gate** determines how strongly new candidate information should be written into the cell state.

It is calculated as

$$
i_t=\sigma(W_i[h_{t-1},x_t]+b_i)
$$

Like the forget gate, the sigmoid restricts its values to the interval `[0,1]`.

The input gate does not generate the new information itself.

Instead, it controls how much of the candidate cell state should be added.

$$
i_t\circ\tilde{c}_t
$$

Therefore, the input gate acts as a learned filter for new information.

---

## Candidate Cell State

The **candidate cell state** contains new information that could potentially be stored in the cell state.

It is calculated using the hyperbolic tangent activation:

$$
\tilde{c}_t=\tanh(W_c[h_{t-1},x_t]+b_c)
$$

Unlike the sigmoid gates, tanh produces values approximately between `-1` and `1`.

Therefore, candidate values can represent both positive and negative changes to the memory.

The candidate state is not automatically stored.

It must first be filtered by the input gate.

$$
i_t\circ\tilde{c}_t
$$

This distinction is important:

- Input gate → controls **how much** new information enters.
- Candidate state → determines **what** new information could enter.

---

## Cell State Update

The cell state is the central memory mechanism of an LSTM.

The update equation is

$$
c_t=f_t\circ c_{t-1}+i_t\circ\tilde{c}_t
$$

It contains two separate contributions.

### Previous Memory

The previous cell state is filtered by the forget gate:

$$
f_t\circ c_{t-1}
$$

This determines what old information survives.

### New Information

The candidate cell state is filtered by the input gate:

$$
i_t\circ\tilde{c}_t
$$

This determines what new information is added.

The two parts are then added together.

$$
c_t=(f_t\circ c_{t-1})+(i_t\circ\tilde{c}_t)
$$

This additive update is one of the most important differences between an LSTM and a basic RNN.

---

## Output Gate

The **output gate** determines how much of the internal cell state should be exposed as the hidden state.

It is calculated as

$$
o_t=\sigma(W_o[h_{t-1},x_t]+b_o)
$$

The output gate produces values between `0` and `1`.

However, the cell state itself is first transformed by tanh:

$$
\tanh(c_t)
$$

The result is then filtered by the output gate:

$$
h_t=o_t\circ\tanh(c_t)
$$

The hidden state therefore represents the information that the LSTM chooses to expose from its internal memory.

---

## Hidden State

The hidden state $h_t$ serves two purposes.

First, it is the output of the LSTM for the current time step.

Second, it is passed into the next time step:

$$
h_t\rightarrow h_{t+1}
$$

The next time step therefore receives the previous hidden state along with the next input.

The hidden state is calculated from the updated cell state:

$$
h_t=o_t\circ\tanh(c_t)
$$

The cell state and hidden state are therefore related but serve different purposes.

The **cell state** acts as internal memory, while the **hidden state** acts as the exposed representation.

---

## Complete LSTM Equations

At each time step, concatenate the previous hidden state and current input:

$$
z_t=[h_{t-1},x_t]
$$

Compute the forget gate:

$$
f_t=\sigma(W_fz_t+b_f)
$$

Compute the input gate:

$$
i_t=\sigma(W_iz_t+b_i)
$$

Compute the candidate cell state:

$$
\tilde{c}_t=\tanh(W_cz_t+b_c)
$$

Update the cell state:

$$
c_t=f_t\circ c_{t-1}+i_t\circ\tilde{c}_t
$$

Compute the output gate:

$$
o_t=\sigma(W_oz_t+b_o)
$$

Compute the hidden state:

$$
h_t=o_t\circ\tanh(c_t)
$$

These equations are repeated for every input in the sequence.

---

## Weight Matrix Dimensions

Suppose:

- Input size = $D$
- Hidden size = $H$

The concatenated vector contains both the previous hidden state and current input.

Therefore,

$$
z_t\in\mathbb{R}^{H+D}
$$

Each gate maps this vector to the hidden dimension.

Therefore, each weight matrix has shape

$$
W_f,W_i,W_c,W_o\in\mathbb{R}^{H\times(H+D)}
$$

Each bias has shape

$$
b_f,b_i,b_c,b_o\in\mathbb{R}^{H}
$$

In the implementation, the biases are stored as column vectors with shape

$$
(H,1)
$$

For four gates, the total number of weight parameters is

$$
4H(H+D)
$$

The total number of bias parameters is

$$
4H
$$

Therefore, the total number of trainable parameters in this implementation is

$$
4H(H+D)+4H
$$

---

## Sigmoid and Tanh

LSTMs use two important activation functions.

### Sigmoid

The sigmoid function is

$$
\sigma(x)=\frac{1}{1+e^{-x}}
$$

Its output lies between `0` and `1`.

This makes it suitable for gates because the output can be interpreted as a soft retention factor.

Sigmoid is used for:

- Forget gate.
- Input gate.
- Output gate.

---

### Hyperbolic Tangent

The tanh function is

$$
\tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}
$$

Its output lies between `-1` and `1`.

Tanh is used for:

- Candidate cell state.
- Transforming the cell state before producing the hidden state.

The candidate state needs negative and positive values because it represents potential changes to memory.

---

## Element-Wise Multiplication

The symbol $\circ$ represents element-wise multiplication.

For example,

$$
a\circ b=[a_1b_1,a_2b_2,\ldots,a_nb_n]
$$

In NumPy, this is performed using the `*` operator for arrays of compatible shapes.

For example:

```python
cell_state = forget_gate * previous_cell_state
```

This is different from matrix multiplication.

Matrix multiplication is performed using operations such as:

```python
np.dot(W, x)
```

or

```python
W @ x
```

Understanding this distinction is essential when implementing an LSTM from scratch.

---

## Implementation Flow

For every input vector in the sequence:

```text
1. Read current input x_t
2. Combine h_(t-1) and x_t
3. Calculate forget gate
4. Calculate input gate
5. Calculate candidate cell state
6. Update cell state
7. Calculate output gate
8. Update hidden state
9. Store hidden state
10. Continue to next input
```

The previous hidden state and cell state are carried forward between time steps.

The sequence therefore creates a chain:

```text
x1 ──> h1, c1
       |
x2 ──> h2, c2
       |
x3 ──> h3, c3
       |
...
       |
xT ──> hT, cT
```

The final states are

$$
h_T
$$

and

$$
c_T
$$

where $T$ is the number of time steps.

---

## Step-by-Step Example

Consider a simplified one-dimensional LSTM.

Given:

```text
x1 = 1.0
x2 = 2.0
x3 = 3.0
```

Initial states:

```text
h0 = 0
c0 = 0
```

Assume for demonstration:

```text
Wf = Wi = Wc = Wo = 0.5
bf = bi = bc = bo = 0.1
```

### Time Step 1

The first input is

$$
x_1=1.0
$$

The previous hidden state is

$$
h_0=0
$$

The previous cell state is

$$
c_0=0
$$

The forget gate is

$$
f_1=\sigma(0.5\times1.0+0.1)\approx0.6457
$$

The input gate is

$$
i_1=\sigma(0.5\times1.0+0.1)\approx0.6457
$$

The candidate state is

$$
\tilde{c}_1=\tanh(0.5\times1.0+0.1)\approx0.5370
$$

The cell state becomes

$$
c_1=f_1c_0+i_1\tilde{c}_1
$$

Since $c_0=0$,

$$
c_1\approx0.6457\times0+0.6457\times0.5370\approx0.3467
$$

The output gate is

$$
o_1=\sigma(0.5\times1.0+0.1)\approx0.6457
$$

The hidden state becomes

$$
h_1=o_1\times\tanh(c_1)\approx0.2152
$$

The first time step therefore produces new memory and a new hidden representation.

---

### Time Step 2

The second input is

$$
x_2=2.0
$$

The previous states are now $h_1$ and $c_1$.

The concatenated input contains

$$
[h_1,x_2]
$$

The four gates are recalculated using these updated states.

The forget gate decides how much of $c_1$ survives.

The input gate decides how much new candidate information enters.

The candidate state generates the potential new memory.

The cell state is updated using

$$
c_2=f_2c_1+i_2\tilde{c}_2
$$

The output gate then determines the exposed hidden state:

$$
h_2=o_2\tanh(c_2)
$$

---

### Time Step 3

The third input is

$$
x_3=3.0
$$

The LSTM now uses

$$
h_2
$$

and

$$
c_2
$$

from the previous time step.

The gates are calculated again.

The cell state becomes

$$
c_3=f_3c_2+i_3\tilde{c}_3
$$

The hidden state becomes

$$
h_3=o_3\tanh(c_3)
$$

The final hidden state is therefore

$$
h_T=h_3
$$

and the final cell state is

$$
c_T=c_3
$$

This demonstrates how information flows through the complete sequence rather than treating every input independently.

---

## LSTM vs Simple RNN

A simple RNN maintains one recurrent state.

$$
h_t=\tanh(W_xx_t+W_hh_{t-1}+b)
$$

An LSTM maintains two states:

$$
(h_t,c_t)
$$

The cell state provides a dedicated memory pathway, while gates control the flow of information.

| Feature                       | Simple RNN | LSTM         |
| ----------------------------- | ---------- | ------------ |
| Hidden state                  | Yes        | Yes          |
| Cell state                    | No         | Yes          |
| Forget gate                   | No         | Yes          |
| Input gate                    | No         | Yes          |
| Output gate                   | No         | Yes          |
| Long-term dependency handling | Difficult  | Better       |
| Parameters                    | Fewer      | More         |
| Computation                   | Simpler    | More complex |

The additional gates make an LSTM more computationally expensive than a simple RNN, but they provide substantially better control over long-term information.

---

## LSTM vs GRU

Another gated recurrent architecture is the **Gated Recurrent Unit (GRU)**.

LSTM and GRU both use gates to control information flow, but their structures differ.

LSTM has:

- Cell state.
- Hidden state.
- Forget gate.
- Input gate.
- Candidate state.
- Output gate.

GRU generally has:

- Hidden state.
- Update gate.
- Reset gate.

GRUs have fewer components and therefore fewer parameters.

LSTMs provide a more explicit separation between internal memory and exposed hidden state.

The choice between LSTM and GRU depends on the problem, architecture, computational budget, and empirical performance.

---

## Characteristics / Key Points

- LSTM is a gated recurrent neural network.
- It is designed for sequential and temporal data.
- It maintains both a hidden state and a cell state.
- The cell state acts as a long-term memory pathway.
- The forget gate controls retained information.
- The input gate controls newly written information.
- The candidate state generates potential new memory.
- The output gate controls exposed information.
- Sigmoid is used for the three gates.
- Tanh is used for candidate memory and hidden-state generation.
- Gate values are between `0` and `1`.
- Candidate values are between `-1` and `1`.
- Cell-state updates contain an additive pathway.
- LSTMs can model dependencies across many time steps.
- The same parameters are reused at every time step.
- Sequence length affects computation but not the number of recurrent parameters.
- Larger hidden sizes increase both parameter count and computation.
- LSTMs contain more parameters than simple RNNs.
- LSTMs are still susceptible to optimization difficulties, although they substantially reduce the classic vanishing-gradient problem.
- Random initialization means repeated executions can produce different outputs.

---

## Why is it used? / Applications

LSTMs are useful when the order and history of observations matter.

Common applications include:

- Natural Language Processing.
- Language Modeling.
- Machine Translation.
- Sentiment Analysis.
- Speech Recognition.
- Time-Series Forecasting.
- Stock and financial sequence modeling.
- Sensor-data analysis.
- Activity Recognition.
- Text Generation.
- Sequence Classification.
- Anomaly Detection.
- Predictive Maintenance.
- Handwriting Recognition.

For example, in language modeling, the meaning of a word can depend on words that appeared much earlier in the sentence.

An LSTM can preserve useful information through its cell state and use gates to decide which information should remain relevant.

---

> 💡 **Important Note**
>
> The cell state and hidden state are not the same thing. The cell state $c_t$ is the LSTM's internal memory, while the hidden state $h_t$ is the information exposed by the output gate. Both states must be carried between time steps.

---

> 💡 **Interview Tip**
>
> Remember the LSTM update as **Forget → Input → Candidate → Cell Update → Output**. The forget gate controls old memory, the input gate controls new memory, the cell state stores the combined memory, and the output gate controls what becomes the hidden state.

---

> 💡 **Common Mistake**
>
> Do not calculate the output gate from the updated hidden state. All four gates are calculated from the same concatenation of the previous hidden state and current input. The cell state is updated afterward, and only then is the new hidden state calculated.

---

## Solutions

### Custom Implementation

```python
import numpy as np

class LSTM:
	def __init__(self, input_size, hidden_size):
		self.input_size = input_size
		self.hidden_size = hidden_size

		# Initialize weights and biases
		self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
		self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
		self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
		self.Wo = np.random.randn(hidden_size, input_size + hidden_size)

		self.bf = np.zeros((hidden_size, 1))
		self.bi = np.zeros((hidden_size, 1))
		self.bc = np.zeros((hidden_size, 1))
		self.bo = np.zeros((hidden_size, 1))

	def forward(self, X, initial_hidden_state, initial_cell_state):
		"""
		Processes a sequence of inputs and returns the hidden states, final hidden state, and final cell state.
		"""
		def sigmoid(x):
			return 1 / (1 + np.exp(-x))
		def tanh(x):
			return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

		H = initial_hidden_state
		C = initial_cell_state
		hidden_states = []
		for word in X:
			word = word.reshape(-1, 1)
			combined = np.vstack((H, word))
			# Forget Gate
			FG = sigmoid(np.dot(self.Wf, combined) + self.bf)
			# Input Gate
			IG = sigmoid(np.dot(self.Wi, combined) + self.bi)
			C_dash = tanh(np.dot(self.Wc, combined) + self.bc)
			# Cell State Updation
			C = C * FG + IG * C_dash
			# Output Gate
			OG = sigmoid(np.dot(self.Wo, combined) + self.bo)
			H = OG * tanh(C)
			hidden_states.append(H)
		return hidden_states, H, C
```

### NumPy / Deep Learning Equivalent

NumPy does not provide a built-in high-level LSTM layer.

A framework such as PyTorch provides an optimized implementation:

```python
import torch
import torch.nn as nn

lstm = nn.LSTM(
    input_size=1,
    hidden_size=1,
    batch_first=True
)

x = torch.tensor([
    [[1.0]],
    [[2.0]],
    [[3.0]]
])

output, (final_h, final_c) = lstm(x)
```

The framework implementation internally performs the same conceptual operations:

```text
Forget Gate
Input Gate
Candidate State
Cell State Update
Output Gate
Hidden State Update
```

The main difference is that production deep learning frameworks combine and optimize these operations rather than implementing each gate separately in Python.

---

## Code Explanation

### Step 1: Initialize the LSTM

```python
self.input_size = input_size
self.hidden_size = hidden_size
```

The input size determines the number of features in each input vector.

The hidden size determines the number of values maintained by the hidden and cell states.

If

```text
input_size = D
hidden_size = H
```

then the concatenated vector has size

```text
D + H
```

---

### Step 2: Initialize the Four Weight Matrices

The implementation creates four independent weight matrices:

```python
self.Wf
self.Wi
self.Wc
self.Wo
```

Each has shape

```text
(hidden_size, input_size + hidden_size)
```

The matrices correspond to:

| Matrix | Purpose              |
| ------ | -------------------- |
| `Wf`   | Forget gate          |
| `Wi`   | Input gate           |
| `Wc`   | Candidate cell state |
| `Wo`   | Output gate          |

Random initialization gives each gate its own learnable transformation.

---

### Step 3: Initialize Biases

```python
self.bf = np.zeros((hidden_size, 1))
self.bi = np.zeros((hidden_size, 1))
self.bc = np.zeros((hidden_size, 1))
self.bo = np.zeros((hidden_size, 1))
```

There is one bias vector for each transformation.

All biases initially contain zeros.

---

### Step 4: Initialize the States

Inside `forward`, the initial states are assigned:

```python
H = initial_hidden_state
C = initial_cell_state
```

`H` represents the hidden state from the previous time step.

`C` represents the cell state from the previous time step.

At the first time step, these correspond to $h_0$ and $c_0$.

---

### Step 5: Create Storage for Hidden States

```python
hidden_states = []
```

The problem requires the hidden state from every time step.

Therefore, each newly calculated hidden state is stored in this list.

The list eventually contains:

```text
[h1, h2, h3, ..., hT]
```

---

### Step 6: Iterate Through the Sequence

```python
for word in X:
```

Each `word` represents the current input vector.

If the sequence has $T$ inputs, this loop executes $T$ times.

Each iteration represents one time step of the LSTM.

---

### Step 7: Reshape the Input

```python
word = word.reshape(-1, 1)
```

The input is converted into a column vector.

For example,

```text
[1.0]
```

becomes a vector with shape

```text
(1, 1)
```

This makes its dimensions compatible with the matrix operations.

---

### Step 8: Combine Hidden State and Input

```python
combined = np.vstack((H, word))
```

The previous hidden state and current input are vertically stacked.

Mathematically:

$$
z_t=[h_{t-1},x_t]
$$

If the hidden size is `H` and input size is `D`, the resulting vector has shape

```text
(H + D, 1)
```

This same combined vector is used by all four gate transformations.

---

### Step 9: Calculate the Forget Gate

```python
FG = sigmoid(
    np.dot(self.Wf, combined)
    + self.bf
)
```

This implements

$$
f_t=\sigma(W_fz_t+b_f)
$$

The forget gate determines how much of the previous cell state should survive.

---

### Step 10: Calculate the Input Gate

```python
IG = sigmoid(
    np.dot(self.Wi, combined)
    + self.bi
)
```

This implements

$$
i_t=\sigma(W_iz_t+b_i)
$$

The input gate determines how much candidate information should be added to the cell state.

---

### Step 11: Calculate the Candidate State

```python
C_dash = tanh(
    np.dot(self.Wc, combined)
    + self.bc
)
```

This implements

$$
\tilde{c}_t=\tanh(W_cz_t+b_c)
$$

The candidate state represents possible new information that could be stored.

It is controlled by the input gate before being added to the cell state.

---

### Step 12: Update the Cell State

```python
C = (
    C * FG
    + IG * C_dash
)
```

This implements the core LSTM memory equation:

$$
c_t=f_t\circ c_{t-1}+i_t\circ\tilde{c}_t
$$

The first term preserves useful old information.

The second term adds selected new information.

This is the main mechanism through which the LSTM maintains memory.

---

### Step 13: Calculate the Output Gate

```python
OG = sigmoid(
    np.dot(self.Wo, combined)
    + self.bo
)
```

This implements

$$
o_t=\sigma(W_oz_t+b_o)
$$

The output gate determines which parts of the updated cell state should be exposed.

---

### Step 14: Update the Hidden State

```python
H = OG * tanh(C)
```

This implements

$$
h_t=o_t\circ\tanh(c_t)
$$

The cell state is first passed through tanh and then filtered using the output gate.

The resulting `H` becomes the hidden state for the current time step.

---

### Step 15: Store the Hidden State

```python
hidden_states.append(H)
```

The current hidden state is saved.

After processing the complete sequence, the list contains every hidden state.

---

### Step 16: Carry States Forward

At the end of an iteration:

```python
H
```

and

```python
C
```

contain the current hidden and cell states.

During the next iteration, they become:

```text
h_(t-1)
c_(t-1)
```

This creates the recurrent connection between time steps.

---

### Step 17: Return the Results

```python
return hidden_states, H, C
```

The function returns three values:

1. Hidden states at every time step.
2. Final hidden state.
3. Final cell state.

If the sequence contains $T$ inputs:

$$
hidden\_states=[h_1,h_2,\ldots,h_T]
$$

and the final states are

$$
final\_h=h_T
$$

and

$$
final\_c=c_T
$$

---

## Complete Forward-Pass Logic

The implementation can be summarized as:

```text
Initialize H = h0
Initialize C = c0

For each input x_t:

    Combine H and x_t

    Forget Gate:
        f_t = sigmoid(Wf * combined + bf)

    Input Gate:
        i_t = sigmoid(Wi * combined + bi)

    Candidate:
        c~_t = tanh(Wc * combined + bc)

    Cell State:
        C = f_t * C + i_t * c~_t

    Output Gate:
        o_t = sigmoid(Wo * combined + bo)

    Hidden State:
        H = o_t * tanh(C)

    Store H

Return:
    hidden_states
    H
    C
```

The critical ordering is that the cell state must be updated before calculating the new hidden state.

---

## Time & Space Complexity

Let

- $T$ be the sequence length.
- $D$ be the input size.
- $H$ be the hidden size.

At every time step, four gate transformations are performed.

Each matrix-vector multiplication has complexity

$$
O(H(D+H))
$$

Since there are four transformations,

$$
O(4H(D+H))
$$

Since `4` is a constant, this simplifies to

$$
O(H(D+H))
$$

per time step.

For a sequence of length $T$, the total forward-pass complexity is

$$
O(TH(D+H))
$$

The implementation stores all hidden states.

Therefore, storing the hidden-state sequence requires

$$
O(TH)
$$

The model parameters require

$$
O(H(D+H))
$$

space.

The recurrent states themselves require

$$
O(H)
$$

additional space.

Therefore, including the model parameters and stored hidden states, the overall space requirement is

$$
O(H(D+H)+TH)
$$

| Complexity | Value                |
| ---------- | -------------------- |
| Time       | **O(TH(D + H))**     |
| Space      | **O(H(D + H) + TH)** |

where

- $T$ is the number of time steps.
- $D$ is the input feature size.
- $H$ is the hidden-state size.

For a single time step, the computation is

$$
O(H(D+H))
$$

and for a sequence of $T$ time steps, it becomes

$$
O(TH(D+H))
$$

The four gates increase the constant factor of the computation, but they do not change the asymptotic complexity.

```

```
