import numpy as np

def batch_iterator(X, y=None, batch_size=64):
    # Your code here
    batches = []
    for ind in range(0, len(X), batch_size):
        if y is not None:
            batches.append([
            list(X[ind: ind+batch_size]),
            list(y[ind: ind+batch_size]),
            ])
        else:
            batches.append(list(X[ind: ind+batch_size]))
    return batches