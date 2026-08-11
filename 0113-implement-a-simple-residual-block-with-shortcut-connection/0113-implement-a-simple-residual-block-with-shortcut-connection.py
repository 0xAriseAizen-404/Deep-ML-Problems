import numpy as np

def residual_block(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    z1 = x @ w1
    # z1_relu = np.maximum(0, z1)
    z1_relu = np.where(z1 > 0, z1, 0)
    z2 = z1_relu @ w2
    # Skip connection
    z2 = z2 + x
    # z2_relu = np.maximum(0, z2)
    z2_relu = np.where(z2 > 0, z2, 0)
    return z2_relu

# * $D$ be the input dimension.
# * $H$ be the hidden dimension.
# * $O$ be the output dimension.
# * $W_1$ have shape $D\times H$.
# * $W_2$ have shape $H\times O$.
# TC: O(DH+HO)
# SC: O(H+O)