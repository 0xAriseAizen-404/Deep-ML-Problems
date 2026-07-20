import numpy as np
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
    # Your code here
    # standardized_data = StandardScaler().fit_transform(data).round(4)
    # normalized_data = MinMaxScaler().fit_transform(data).round(4)
    standardized_data = (
        (data - np.mean(data, axis=0))
        / np.std(data, axis=0)
    ).round(4)
    normalized_data = (
        (data - np.min(data, axis=0))
        / (np.max(data, axis=0) - np.min(data, axis=0))
    ).round(4)

    return standardized_data, normalized_data