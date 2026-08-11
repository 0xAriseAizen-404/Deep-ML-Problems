import math

def selu(x: float) -> float:
	"""
	Implements the SELU (Scaled Exponential Linear Unit) activation function.

	Args:
		x: Input value

	Returns:
		SELU activation value
	"""
	scale = 1.0507009873554804
	alpha = 1.6732632423543772
	
	if x > 0:
		return scale * x
	else:
		return scale * alpha * (math.exp(x) - 1)

# TC: O(1)
# SC: O(1)