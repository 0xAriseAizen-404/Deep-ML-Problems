import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	#Write your code here and return a python list after reshaping by using numpy's tolist() method
	# a_numpy_array = np.asarray(a)
	# reshaped_a_np_arr = a_numpy_array.reshape(new_shape)
	# return reshaped_a_np_arr.tolist()
    try:
        return np.asarray(a).reshape(new_shape).tolist()
    except ValueError:
        return []

# TC: O(n)
# SC: O(n)