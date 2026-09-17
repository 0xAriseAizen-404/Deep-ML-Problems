import numpy as np

def gru_cell(x: np.ndarray, h_prev: np.ndarray,
             W_z: np.ndarray, U_z: np.ndarray, b_z: np.ndarray,
             W_r: np.ndarray, U_r: np.ndarray, b_r: np.ndarray,
             W_h: np.ndarray, U_h: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    """
    Implements a single GRU cell forward pass.
    
    Args:
        x: Input vector of shape (input_size,)
        h_prev: Previous hidden state of shape (hidden_size,)
        W_z, W_r, W_h: Weight matrices for input
        U_z, U_r, U_h: Weight matrices for hidden state
        b_z, b_r, b_h: Bias vectors
    
    Returns:
        h_next: New hidden state of shape (hidden_size,)
    """
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    def tanh(z):
        # return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
        return np.tanh(z)
    
    Rt = sigmoid(W_r @ x + U_r @ h_prev + b_r)
    Zt = sigmoid(W_z @ x + U_z @ h_prev + b_z)
    ht_dash = tanh(W_h @ x + U_h @ (Rt * h_prev) + b_h)
    h_next = (1 - Zt) * h_prev + Zt * ht_dash

    return h_next