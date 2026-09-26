import numpy as np

def gaussian_elimination(A, b):
    A = A.astype(float)
    b = b.astype(float)
    n = len(b)
    for pivot in range(n):
        max_row = pivot + np.argmax(np.abs(A[pivot:, pivot]))
        A[[pivot, max_row]] = A[[max_row, pivot]]
        b[[pivot, max_row]] = b[[max_row, pivot]]
        for row in range(pivot + 1, n):
            factor = A[row, pivot] / A[pivot, pivot]
            A[row] -= factor * A[pivot]
            b[row] -= factor * b[pivot]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]
    return x