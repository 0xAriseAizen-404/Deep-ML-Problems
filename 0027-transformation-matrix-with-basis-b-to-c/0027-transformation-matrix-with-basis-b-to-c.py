def transform_basis(B: list[list[int]], C: list[list[int]]) -> list[list[float]]:

    def det_help(mat: list[list[int]]) -> float:
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
    
    def adjoint_help(mat: list[list[int]]) -> list[list[float]]:
        n = len(mat)
        if n == 1:
            return [[1]]
        cof = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                minor = [row[:j] + row[j+1:] for r, row in enumerate(mat) if r != i]
                cof[i][j] = ((-1) ** (i + j)) * det_help(minor)
        # Transpose = Adjoint
        adj = [[cof[j][i] for j in range(n)] for i in range(n)]
        return adj

    det_C = det_help(C)
    adjoint_mat_C = adjoint_help(C)

    # inverse_C = list(
    #     map(
    #         lambda row: list(map(lambda x: x / det_C, row)),
    #         adjoint_mat_C
    #     )
    # )
    inverse_C = [[x / det_C for x in row] for row in adjoint_mat_C]
    # P = C^{-1} @ B
    n = len(C)
    P = [[sum(inverse_C[i][k] * B[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
    return P
    
    
    # return (np.linalg.inv(C) @ np.array(B)).tolist()