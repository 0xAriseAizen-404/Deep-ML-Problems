import numpy as np

def cosine_similarity(v1, v2):
	"""
	Calculate the cosine_similarity of two vectors.
	Args:
		vec1 (numpy.ndarray): 1D array representing the first vector.
		vec2 (numpy.ndarray): 1D array representing the second vector.
	Returns:
		The cosine_similarity of the two vectors.
	"""
	if v1.shape != v2.shape:
		raise ValueError(f"Shape Mismatch: v1 shape {v1.shape} != v2 shape {v2.shape}")
	if v1.size == 0 or v2.size == 0:
		raise ValueError("Input Vectors cannot be empty")
	
	v1_mag = np.sqrt(np.sum(v1**2))
	v2_mag = np.sqrt(np.sum(v2**2))

	if v1_mag == 0 or v2_mag == 0:
		raise ValueError("Input Vectors cannot have Zero Magnitude")
	
	# A@B = |A| * |B| * cos(theta)
	# cos(theta) = A@B / (|A|*|B|)

	# return v1.dot(v2) / (v1_mag * v2_mag)
	return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# TC: O(n)
# SC: O(1)