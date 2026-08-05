import numpy as np

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
    input_height, input_width = input_matrix.shape
    kernel_height, kernel_width = kernel.shape

    # mat_height = input_height + padding + padding
    # mat_width = input_width + padding + padding
    # matrix = [[0] * mat_width for _ in range(mat_height)]
    # for i in range(input_height):
    #     for j in range(input_width):
    #         matrix[padding+i][padding+j] = input_matrix[i][j]
    # matrix = np.asarray(matrix)
    
    matrix = np.pad(
        input_matrix,
        ((padding, padding), (padding, padding)),
        mode="constant"
    )
    mat_height, mat_width = matrix.shape
    
    output_matrix = []
    for i in range(0, mat_height - kernel_height + 1, stride):
        row = []
        for j in range(0, mat_width - kernel_width + 1, stride):
            window = matrix[i:i+kernel_height, j:j+kernel_width]
            row.append(np.sum(window * kernel))
        output_matrix.append(row)
    return np.round(output_matrix, 4)

# TC: O(Hₒ × Wₒ × Kₕ × K𝓌)
# SC: O((H + 2P)(W + 2P) + HₒWₒ)