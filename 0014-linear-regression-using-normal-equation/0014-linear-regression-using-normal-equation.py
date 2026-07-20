import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
    # Your code here, make sure to round
    # θ=(X^T.X)^−1.X^T.y
    npX = np.asarray(X)
    npy = np.asarray(y).reshape(-1, 1)
    theta = np.linalg.inv(npX.T.dot(X)).dot(npX.T).dot(y)
    return np.round(theta, 4).flatten().tolist()