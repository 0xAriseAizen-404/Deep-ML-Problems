import math
def binary_cross_entropy(y_true: list[float], y_pred: list[float], epsilon: float = 1e-15) -> float:
	"""
	Compute binary cross-entropy loss.
	
	Args:
		y_true: True binary labels (0 or 1)
		y_pred: Predicted probabilities (between 0 and 1)
		epsilon: Small value for numerical stability
	
	Returns:
		Mean binary cross-entropy loss
	"""
    loss = 0.0
    for y, p in zip(y_true, y_pred):
        p = max(epsilon, min(p, 1 - epsilon))
        loss += (y * math.log(p) + (1 - y) * math.log(1 - p))
    return -loss / len(y_true)


	# def clip(y_pred: list[float], min_val: float, max_val: float):
	# 	min_y_pred = min(y_pred)
	# 	max_y_pred = max(y_pred)
	# 	for ind in range(len(y_pred)):
	# 		y_pred[ind] = min_val + ((max_val - min_val) / (max_y_pred - min_y_pred)) * (y_pred[ind] - min_y_pred)
	# 	return y_pred

	# y_pred = clip(y_pred, epsilon, 1 - epsilon)
	# bce_error = 1.0
	# for y, p in zip(y_true, y_pred):
	# 	bce_error += ((y * math.log(p)) + ((1 - y) * math.log(1 - p)))
	# return bce_error/len(y_true)