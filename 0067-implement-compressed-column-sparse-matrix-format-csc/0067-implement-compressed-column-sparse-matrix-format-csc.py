def compressed_col_sparse_matrix(dense_matrix):
    """
    Convert a dense matrix into its Compressed Column Sparse (CSC) representation.

    :param dense_matrix: List of lists representing the dense matrix
    :return: Tuple of (values, row indices, column pointer)
    """
    values = []
    row_indices = []
    col_ptr = [0]
    for col in range(len(dense_matrix[0])):
        for row in range(len(dense_matrix)):
            if dense_matrix[row][col] != 0:
                values.append(dense_matrix[row][col])
                row_indices.append(row)
        col_ptr.append(len(values))
    return values, row_indices, col_ptr

	# 1 0 0 0
	# 0 2 0 0
	# 3 0 4 0
	# 1 0 0 5
	# values = 1 3 1 2 4 5
	# row_indices = 0 2 3 1 2 3
	# col_ptr = 0 3 4 5 6
	
	# col 0 -> [0, 3)
	# col 1 -> [3, 4)
	# col 2 -> [4, 5)
	# col 3 -> [5, 6)

	# 1 0 0 0
	# 0 0 0 0
	# 3 0 0 0
	# 1 0 0 5
	# values = 1 3 1 5
	# row_indices = 0 2 3 3
	# col_ptr = 0 3 3 3 4

	# col 0 -> [0, 3)
	# col 1 -> [3, 3)
	# col 2 -> [3, 3)
	# col 3 -> [3, 4)

# TC: O(m * n)
# SC: O(k + n)