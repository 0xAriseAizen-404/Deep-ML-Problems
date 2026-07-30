import numpy as np
def rmse(y_true, y_pred):
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("y_true and y_pred must be NumPy arrays.")
    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("Input arrays cannot be empty.")
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true shape {y_true.shape} != y_pred shape {y_pred.shape}"
        )
    return np.round(np.sqrt(np.mean((y_true - y_pred) ** 2)), 3)