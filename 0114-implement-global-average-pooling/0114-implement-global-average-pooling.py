import numpy as np

def global_avg_pool(x: np.ndarray) -> np.ndarray:
    # pool = [0] * len(x[0][0])
    # pool = np.asarray(pool)
    # cnt = 0
    # for row in x:
        # for pixel in row:
            # pool = pool + pixel
            # cnt += 1
    # return pool / cnt
    
    # return np.mean(axis=(0, 1))
    
    return np.sum(x, axis=(0, 1)) / (x.shape[0] * x.shape[1])

# TC: O(H * W * C)
# SC: O(C)