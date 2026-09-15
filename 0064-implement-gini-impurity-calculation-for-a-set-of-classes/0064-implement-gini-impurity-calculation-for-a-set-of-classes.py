import numpy as np

def gini_impurity(y):
    """
    Calculate Gini Impurity for a list of class labels.

    :param y: List of class labels
    :return: Gini Impurity rounded to three decimal places
    """
    # cls = np.array(y, dtype=int)
    # freq = {}
    # for val in cls:
    #     freq[val] = freq.get(val, 0) + 1
    # total_probs = 0.0
    # for cnt in freq.values():
    #     p = cnt / len(cls)
    #     total_probs += p ** 2
    # gini_impurity_val = 1 - total_probs
    # return round(gini_impurity_val, 3)

    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return round(1 - np.sum(p ** 2), 3)