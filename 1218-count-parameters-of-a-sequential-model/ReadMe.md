# Count Parameters of a Sequential Model (Easy, PyTorch)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Counting Model Parameters](#learn-counting-model-parameters)
- [Solutions](#solutions)
  - [PyTorch Implementation](#pytorch-implementation)
- [Code Explanation](#code-explanation)

## Problem Statement

[Count Parameters of a Sequential Model](https://www.deep-ml.com/problems/1218)

Implement `count_params()` that builds the following model:

```python
nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 2)
)
```

Return the total number of **trainable parameters** as a Python integer.

## Example

```python
count_params()

# Output
58
```

## Learn: Counting Model Parameters

A layer `nn.Linear(in_features, out_features)` contains:

$$
\text{Parameters} = (\text{in\_features} \times \text{out\_features}) + \text{out\_features}
$$

For this model:

- `Linear(4, 8)` → `4 × 8 + 8 = 40`
- `ReLU()` → `0`
- `Linear(8, 2)` → `8 × 2 + 2 = 18`

**Total Parameters**

$$
40 + 18 = 58
$$

## Solutions

### PyTorch Implementation

```python
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )

    def forward(self, x):
        return self.mlp(x)

def count_params():
    model = MyModel()
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```

## Code Explanation

The function:

1. Builds the required sequential model.
2. Iterates over all parameter tensors using `model.parameters()`.
3. Counts only trainable parameters (`requires_grad=True`).
4. Uses `p.numel()` to count the number of elements in each tensor.
5. Returns the total number of trainable parameters (`58`).