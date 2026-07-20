import numpy as np

def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
	res = []
	if mode == "row":
		for vec in matrix:
			res.append(sum(vec)/len(vec))
	else:
		for ind in range(len(matrix[0])):
			summ = sum(vec[ind] for vec in matrix)
			res.append(summ/len(matrix))
	return res

	# if mode == "column":
	# 	return np.asarray(matrix).mean(axis=0).tolist()
	# else:
	# 	return np.asarray(matrix).mean(axis=1).flatten().tolist()