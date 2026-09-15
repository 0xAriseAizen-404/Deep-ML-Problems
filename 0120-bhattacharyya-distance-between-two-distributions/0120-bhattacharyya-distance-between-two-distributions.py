import numpy as np

def bhattacharyya_distance(p: list[float], q: list[float]) -> float:

    if len(p) != len(q):
        return 0.0
    if len(p) == 0 or len(q) == 0:
        return 0.0

    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)

    # bhattacharyya_coefficient = 0
    # for x, y in zip(p, q):
    #     bhattacharyya_coefficient += (np.sqrt(x * y))

    bhattacharyya_coefficient = np.sum(np.sqrt(p * q))
    return np.round(-np.log(bhattacharyya_coefficient), 4)