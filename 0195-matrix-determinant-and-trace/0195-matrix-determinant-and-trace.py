def matrix_determinant_and_trace(matrix: list[list[float]]) -> tuple[float, float]:
	"""
	Compute the determinant and trace of a square matrix.
	
	Args:
		matrix: A square matrix (n x n) represented as list of lists
	
	Returns:
		Tuple of (determinant, trace)
	"""
	def det_help(mat: list[list[float]]) -> float:
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
    
	def trace(mat: list[list[float]]) -> float:
		n = len(mat)
		trace_mat = 0.0
		for ind in range(n):
			trace_mat += mat[ind][ind]
		return trace_mat
	
	return (det_help(matrix), trace(matrix))