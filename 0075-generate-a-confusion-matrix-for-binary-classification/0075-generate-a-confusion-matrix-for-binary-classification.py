import numpy as np
from collections import Counter

def confusion_matrix(data):
    TP = FN = FP = TN = 0
    for actual, pred in data:
        TP += actual & pred
        FN += actual & (1 - pred)
        FP += (1 - actual) & pred
        TN += (1 - actual) & (1 - pred)
    return [[TP, FN], [FP, TN]]

# def confusion_matrix(data):
    # data = np.asarray(data)
    # actual = data[:, 0]
    # pred = data[:, 1]
    # TP = np.sum((actual == 1) & (pred == 1))
    # FN = np.sum((actual == 1) & (pred == 0))
    # FP = np.sum((actual == 0) & (pred == 1))
    # TN = np.sum((actual == 0) & (pred == 0))
    # return [[TP, FN], [FP, TN]]

# def confusion_matrix(data):
    # c = Counter(map(tuple, data))
    # return [
        # [c[(1, 1)], c[(1, 0)]],  # TP, FN
        # [c[(0, 1)], c[(0, 0)]]   # FP, TN
    # ]

# TC: O(n)
# SC: O(1)