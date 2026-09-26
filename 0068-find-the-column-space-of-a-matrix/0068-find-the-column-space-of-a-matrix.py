import numpy as np

def rref(mat):
    matrix = mat.copy().astype(float)
    rows, cols = matrix.shape
    pivot = 0
    col = 0
    while pivot < rows and col < cols:
        if matrix[pivot][col] == 0:
            row_ind = -1
            for temp in range(pivot + 1, rows):
                if matrix[temp][col] != 0:
                    row_ind = temp
                    break
            if row_ind == -1:
                col += 1
                continue
            matrix[[pivot, row_ind]] = matrix[[row_ind, pivot]]
        matrix[pivot] = matrix[pivot] / matrix[pivot][col]
        for row in range(rows):
            if row != pivot:
                matrix[row] = matrix[row] - (matrix[row][col] * matrix[pivot])
        pivot += 1
        col += 1
    return matrix

def matrix_image(A):
    r = rref(A)
    pivot_cols = []
    for j in range(r.shape[1]):
        col = r[:, j]
        if np.count_nonzero(np.abs(col) > 1e-10) == 1 and np.isclose(col.max(), 1):
            pivot_cols.append(j)
    return A[:, pivot_cols]