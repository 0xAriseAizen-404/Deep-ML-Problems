import math

def swish(x: float) -> float:
	"""
	Implements the Swish activation function.

	Args:
		x: Input value

	Returns:
		The Swish activation value
	"""
	return round(x * (1 / (1 + math.exp(-x))), 4)

# TC: O(1)
# SC: O(1)