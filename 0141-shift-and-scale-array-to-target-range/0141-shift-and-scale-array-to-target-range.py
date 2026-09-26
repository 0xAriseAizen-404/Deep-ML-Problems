import numpy as np

def convert_range(values: np.ndarray, c: float, d: float) -> np.ndarray:
    """
    Shift and scale values from their original range [min, max] to a target [c, d] range.
    """
    # Formula to turn a range into a target range
    # f(x) = t_min + ((t_max - t_min) / (arr_max - arr_min)) * (x - arr_min)

    arr_max = np.max(values)
    arr_min = np.min(values)

    if arr_max == arr_min:
        return np.full_like(values, c, dtype=float)

    return c + ((d - c) / (arr_max - arr_min)) * (values - arr_min)