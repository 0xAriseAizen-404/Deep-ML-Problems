import numpy as np

def compute_tpr_fpr(y_true, y_pred):
    """
    Compute TPR and FPR from true and predicted binary labels.

    Args:
        y_true (array-like): Ground-truth labels (0 or 1).
        y_pred (array-like): Predicted labels (0 or 1).

    Returns:
        tuple: (tpr, fpr) as Python floats.
    """
    # TP, FP, FN, TN = 0, 0, 0, 0
    # for y_t, y_p in zip(y_true, y_pred):
    #     if y_t == 1:
    #         if y_p == 1:
    #             TP += 1
    #         else:
    #             FN += 1
    #     else:
    #         if y_p == 0:
    #             TN += 1
    #         else:
    #             FP += 1
    # tpr = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    # fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    # return float(tpr), float(fpr)

    TP = np.sum((y_true == 1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))

    tpr = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0

    return float(tpr), float(fpr)