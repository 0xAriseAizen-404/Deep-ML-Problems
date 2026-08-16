import numpy as np

def rgb_to_grayscale(image):
    """
    Convert an RGB image to grayscale using luminosity method.

    Returns:
        Grayscale image as 2D list with integer values,
        or -1 if input is invalid.
    """
    if not isinstance(image, (list, np.ndarray)):
        return -1
    if len(image) == 0:
        return -1
    if not isinstance(image[0], (list, np.ndarray)) or len(image[0]) == 0:
        return -1
    row_len = len(image[0])
    for row in image:
        if not isinstance(row, (list, np.ndarray)) or len(row) != row_len:
            return -1
        for pixel in row:
            if not isinstance(pixel, (list, np.ndarray)) or len(pixel) != 3:
                return -1
            if not all(0 <= value <= 255 for value in pixel):
                return -1
    
    GrayImage = []
    for row in image:
        new_row = []
        for pixel in row:
            GrayScale = (0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2])
            new_row.append(round(GrayScale))
        GrayImage.append(new_row)
    return GrayImage

# TC: O(H * W)
# SC: O(W)