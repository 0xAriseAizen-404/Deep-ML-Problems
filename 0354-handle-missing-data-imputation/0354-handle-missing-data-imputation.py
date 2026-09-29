import numpy as np

def impute_missing_data(data: np.ndarray, strategy: str = 'mean') -> np.ndarray:
    """
    Impute missing values in a 2D array using the specified strategy.
    
    Args:
        data: 2D numpy array with missing values represented as np.nan
        strategy: Imputation strategy - 'mean', 'median', or 'mode'
        
    Returns:
        2D numpy array with missing values imputed
    """
    n_samples, n_cols = data.shape
    indices = np.where(np.isnan(data))
    for row, col in zip(*indices):
        if strategy == "mean":
            data[row, col] = np.nanmean(data[:, col])
        elif strategy == "median":
            nums = data[:, col]
            nums = nums[~np.isnan(nums)]
            nums = sorted(nums)
            mid = len(nums) // 2
            if len(nums) % 2 == 1:
                data[row, col] = nums[mid]
            else:
                data[row, col] = (nums[mid - 1] + nums[mid]) / 2
        elif strategy == "mode":
            nums = data[:, col]
            nums = nums[~np.isnan(nums)]
            unique, counts = np.unique(nums, return_counts=True)
            data[row, col] = unique[np.argmax(counts)]
        else:
            raise ValueError("strategy must be 'mean', 'median', or 'mode'")
    return data