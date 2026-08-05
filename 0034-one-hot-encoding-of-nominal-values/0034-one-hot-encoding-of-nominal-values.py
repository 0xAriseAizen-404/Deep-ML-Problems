import numpy as np

def to_categorical(x, n_col=None):
    if n_col is None:
        n_col = np.max(x) + 1
    one_hot = []
    for val in x:
        row = [0] * n_col
        row[val] = 1
        one_hot.append(row)
    return np.array(one_hot)

# TC: O(n * k)
# SC: O(n * k)