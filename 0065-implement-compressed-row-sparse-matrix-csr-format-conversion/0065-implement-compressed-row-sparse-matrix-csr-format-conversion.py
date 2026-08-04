import numpy as np

def compressed_row_sparse_matrix(dense_matrix):
    """
    Convert a dense matrix to its Compressed Row Sparse (CSR) representation.

    :param dense_matrix: 2D list representing a dense matrix
    :return: A tuple containing (values array, column indices array, row pointer array)
    """
    values = []
    col_indices = []
    row_ptr = [0]
    for row in range(len(dense_matrix)):
        for col in range(len(dense_matrix[0])):
            if dense_matrix[row][col] != 0:
                values.append(dense_matrix[row][col])
                col_indices.append(col)
        row_ptr.append(len(values))
    return values, col_indices, row_ptr

    # 1 0 0 0
    # 0 2 0 0
    # 3 0 4 0
    # 1 0 0 5
    # values = 1 2 3 4 1 5
    # col_indices = 0 1 0 2 0 3
    # row_ptr = 0 1 2 4 6
    
    # col 0 -> [0, 1)
    # col 1 -> [1, 2)
    # col 2 -> [2, 4)
    # col 3 -> [4, 6)

    # 1 0 0 0
    # 0 0 0 0
    # 3 0 0 0
    # 1 0 0 5
    # values = 1 3 1 5
    # col_indices = 0 0 0 3
    # row_ptr = 0 1 1 2 4

    # col 0 -> [0, 1)
    # col 1 -> [1, 1)
    # col 2 -> [1, 2)
    # col 3 -> [2, 4)