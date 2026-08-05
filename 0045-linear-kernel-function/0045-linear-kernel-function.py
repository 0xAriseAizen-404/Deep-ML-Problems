import numpy as np

def kernel_function(x1, x2):
	return np.sum(x1 * x2)

# TC: O(n)
# SC: O(1)