import numpy as np

def detect_outliers_iqr(data: list[float], k: float = 1.5) -> dict:
	"""
	Detect and remove outliers using the IQR method.
	
	Args:
		data: List of numerical values
		k: IQR multiplier for determining outlier bounds (default 1.5)
	
	Returns:
		Dictionary with 'cleaned_data', 'outlier_indices', 'lower_bound', 'upper_bound'
	"""
	q1 = np.percentile(data, 25)
	q3 = np.percentile(data, 75)
	iqr = q3 - q1
	lower_bound = q1 - (k * iqr)
	upper_bound = q3 + (k * iqr)
	cleaned_data = []
	outlier_indices = []
	for ind, val in enumerate(data):
		if val < lower_bound or upper_bound < val:
			outlier_indices.append(ind)
		else:
			cleaned_data.append(val)
	result = dict({
		"cleaned_data": cleaned_data,
		"outlier_indices": outlier_indices,
		"lower_bound": lower_bound.item(),
		"upper_bound": upper_bound.item()
	})
	return result