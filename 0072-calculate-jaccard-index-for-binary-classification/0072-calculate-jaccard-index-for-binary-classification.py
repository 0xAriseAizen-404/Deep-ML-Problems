import numpy as np

def jaccard_index(y_true, y_pred):
    # Jaccard_index = A(Inter)B / A(Union)B
    # A(Inter)B = A==1 && B==1 = (A&B)
    # A(Union)B = A==1 or B==1 or A&B==1 = (A + B - A&B)

    # AiB = np.sum((y_true == 1) & (y_pred == 1))
    # A = np.sum((y_true == 1))
    # B = np.sum((y_pred == 1))
    # AuB = A + B - AiB
    # return round(AiB / AuB, 3)

    intersection = np.sum(y_true & y_pred)
    union = np.sum(y_true | y_pred)
    if union == 0:
        return 0.0
    return round(float(intersection / union), 3)

# TC: O(n)
# SC: O(1)