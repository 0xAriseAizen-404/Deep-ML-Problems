import numpy as np

def dice_score(y_true, y_pred):
    intersection = np.sum(y_true & y_pred)
    A = np.sum(y_true)
    B = np.sum(y_pred)
    if A + B == 0.0:
        return 0.0
    return ((2 * intersection) / (A + B))

# TC: O(n)
# SC: O(1)