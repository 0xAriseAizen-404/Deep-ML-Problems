import numpy as np

def make_diagonal(x):
	# Your code here
	sz = len(x)
	mat = [[0 for _ in range(sz)] for _ in range(sz)]
	for ind in range(sz):
		mat[ind][ind] = x[ind]
	return np.array(mat)

	# return np.diag(x)