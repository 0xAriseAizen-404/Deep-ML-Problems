def determinant_4x4(matrix: list[list[int | float]]) -> float:
    def det_help(mat: list[list[int | float]]) -> float:
        n = len(mat)
        if n == 1:
            return mat[0][0]
        if n == 2:
            return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]
        det = 0
        for j in range(n):
            minor = [row[:j] + row[j + 1:] for row in mat[1:]]
            det += (-1) ** j * mat[0][j] * det_help(minor)
        return det
    return det_help(matrix)