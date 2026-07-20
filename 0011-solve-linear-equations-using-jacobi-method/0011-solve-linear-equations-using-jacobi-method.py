import numpy as np
def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
    x = np.zeros(len(A))
    for _ in range(n):
        x_new = np.zeros_like(x)
        for i in range(len(A)):
            x_new[i] = (b[i] - sum([A[i][j] * x[j] for j in range(len(A)) if i != j])) / A[i][i]
        x = np.round(x_new, 4)
        # x = x_new
    return x
    # return np.round(x, 4).tolist()