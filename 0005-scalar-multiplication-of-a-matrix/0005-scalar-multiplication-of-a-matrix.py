import numpy as np
def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
    # for i in range(len(matrix)):
        # for j in range(len(matrix[i])):
            # matrix[i][j] = matrix[i][j] * scalar
    # return matrix
    return (np.asarray(matrix) * scalar).tolist()