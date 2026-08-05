def hard_sigmoid(x: float) -> float:
	"""
	Implements the Hard Sigmoid activation function.

	Args:
		x (float): Input value

	Returns:
		float: The Hard Sigmoid of the input
	"""
	# HardSigmoid = 0  if x <= -2.5
	# HardSigmoid = 0.2x + 0.5 if x -2.5 < x < 2.5
	# HardSigmoid = 1 if x >= 2.5

	if x <= -2.5:
		return 0
	elif -2.5 < x < 2.5:
		return 0.2 * x + 0.5
	else:
		return 1

# TC: O(1)
# SC: O(1)