import numpy as np

def transform_matrix(A: list[list[int|float]], T: list[list[int|float]], S: list[list[int|float]]) -> list[list[int|float]]:
    # Formula = Inv(T) @ A @ S
    detT = np.linalg.det(np.asarray(T))
    detS = np.linalg.det(np.asarray(S))
    if detT == 0 or detS == 0:
        return -1
        
    result: numpy.ndarray = np.linalg.inv(np.asarray(T)) @ np.asarray(A) @ np.asarray(S)
    return np.round(result, 10).tolist()

    # return np.linalg.solve(T, np.asarray(A) @ np.asarray(S)).tolist()