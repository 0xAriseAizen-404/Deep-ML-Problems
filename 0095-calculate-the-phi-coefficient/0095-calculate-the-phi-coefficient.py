from math import sqrt
def phi_corr(x: list[int], y: list[int]) -> float:
    """
    Calculate the Phi coefficient between two binary variables.

    Args:
    x (list[int]): A list of binary values (0 or 1).
    y (list[int]): A list of binary values (0 or 1).

    Returns:
    float: The Phi coefficient rounded to 4 decimal places.
    """
    TP = FP = FN = TN = 0
    for xi, yi in zip(x, y):
        if xi == 1 and yi == 1:
            TP += 1
        elif xi == 1 and yi == 0:
            FP += 1
        elif xi == 0 and yi == 1:
            FN += 1
        else:
            TN += 1
    numerator = (TP * TN) - (FP * FN)
    denominator = sqrt((TP + FP) * (TP + FN) * 
                        (TN + FP) * (TN + FN))
    if denominator == 0.0:
        return 0.0
    return round(numerator / denominator, 4)

# TC: O(n)
# SC: O(1)