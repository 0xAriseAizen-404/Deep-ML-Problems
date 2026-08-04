import numpy as np
def calculate_brightness(img):
    # if not img:
        # return -1
    # cols = len(img[0])
    # total_pixels = 0.0
    # for row in img:
        # if len(row) != cols:
            # return -1
        # for val in row:
            # if val < 0 or val > 255:
                # return -1
            # total_pixels += val
    # return total_pixels / (len(img) * cols)
    
    try:
        img = np.asarray(img, dtype=float)
    except ValueError:
        return -1
    if img.size == 0:
        return -1
    if np.any((img < 0) | (img > 255)):
        return -1
    return np.mean(img)