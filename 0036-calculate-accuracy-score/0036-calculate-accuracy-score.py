import numpy as np

def accuracy_score(y_true, y_pred):
    # Your code here
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    return ((TP + TN) / (TP + FN + FP + TN))
    
    # return np.mean(y_true == y_pred)

	# correct = np.sum(y_true == y_pred)
	# total = len(y_true)
	# accuracy = correct / total
	# return accuracy

# TC: O(n)
# SC: O(1)