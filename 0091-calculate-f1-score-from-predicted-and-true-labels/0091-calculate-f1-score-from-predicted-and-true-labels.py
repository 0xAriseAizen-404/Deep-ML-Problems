# import numpy as np
def calculate_f1_score(y_true, y_pred):
    """
    Calculate the F1 score based on true and predicted labels.

    Args:
        y_true (list): True labels (ground truth).
        y_pred (list): Predicted labels.

    Returns:
        float: The F1 score rounded to three decimal places.
    """
    # y_true = np.asarray(y_true)
    # y_pred = np.asarray(y_pred)
    # TP = np.sum(y_true & y_pred)
    # FN = np.sum(y_true & (1 - y_pred))
    # FP = np.sum((1 - y_true) & y_pred)
    # TN = np.sum((1 - y_true) & (1 - y_pred))
    
    TP, FN, FP, TN = 0, 0, 0, 0
    for x, y in zip(y_true, y_pred):
        if x & y:
            TP += 1
        elif x & (1 - y):
            FN += 1
        elif (1 - x) & y:
            FP += 1
        else:
            TN += 1

    if (TP + FP) > 0.0:
        precision = TP / (TP + FP)
    else:
        precision = 0.0
    
    if (TP + FN) > 0.0:
        recall = TP / (TP + FN)
    else:
        recall = 0.0
            
    if (precision + recall) > 0.0:
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0.0
    
    return round(f1, 3)

# TC: O(n)
# SC: O(1)