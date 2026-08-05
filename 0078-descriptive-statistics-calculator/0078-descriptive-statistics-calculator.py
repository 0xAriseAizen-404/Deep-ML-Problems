import numpy as np

def descriptive_statistics(data: list | np.ndarray) -> dict:
    """
    Calculate various descriptive statistics metrics for a given dataset.
    
    Args:
        data: List or numpy array of numerical values
    
    Returns:
        Dictionary containing mean, median, mode, variance, standard deviation,
        percentiles (25th, 50th, 75th), and interquartile range (IQR)
    """
    if not data:
        raise ValueError("Data cannot be empty")
    nums = sorted(data)
    n = len(nums)
    res = {}
    mean = sum(nums) / n
    res["mean"] = mean
    if n % 2:
        median = nums[n // 2]
    else:
        median = (nums[n // 2 - 1] + nums[n // 2]) / 2
    res["median"] = median
    freq = {}
    for x in nums:
        freq[x] = freq.get(x, 0) + 1
    max_freq = max(freq.values())
    modes = [k for k, v in freq.items() if v == max_freq]
    res["mode"] = modes[0]
    variance = sum((x - mean) ** 2 for x in nums) / n
    res["variance"] = variance
    std = variance ** 0.5
    res["standard_deviation"] = std
    # p25 = nums[round((n - 1) * 0.25)]
    # p50 = nums[round((n - 1) * 0.50)]
    # p75 = nums[round((n - 1) * 0.75)]
    arr = np.array(nums)
    p25, p50, p75 = np.percentile(arr, [25, 50, 75])
    res["25th_percentile"] = p25.item()
    res["50th_percentile"] = p50.item()
    res["75th_percentile"] = p75.item()
    res["interquartile_range"] = (p75 - p25).item()
    return res
    
    # arr = np.asarray(data, dtype=float)
    # if arr.size == 0:
        # raise ValueError("Data cannot be empty")
    # values, counts = np.unique(arr, return_counts=True)
    # max_count = counts.max()
    # modes = values[counts == max_count]
    # p25, p50, p75 = np.percentile(arr, [25, 50, 75])
    # return {
        # "mean": np.mean(arr),
        # "median": np.median(arr),
        # "mode": modes.tolist(),
        # "variance": np.var(arr),
        # "standard_deviation": np.std(arr),
        # "25th_percentile": p25,
        # "50th_percentile": p50,
        # "75th_percentile": p75,
        # "interquartile_range": p75 - p25,
    # }

# TC: O(n*logn)
# SC: O(n)