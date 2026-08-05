import numpy as np

def divide_on_feature(X, feature_i, threshold):
    # return [
    # [row for row in X if row[feature_i] >= threshold],
    # [row for row in X if row[feature_i] < threshold]
    # ]

    # left, right = [], []
    # for row in X:
        # if row[feature_i] >= threshold:
            # left.append(row)
        # else:
            # right.append(row)
    # return [left, right]
    
    mask = X[:, feature_i] >= threshold
    return [X[mask], X[~mask]]

# TC: O(n)
# SC: O(n)