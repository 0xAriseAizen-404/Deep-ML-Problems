import numpy as np

def log_softmax(scores: list) -> np.ndarray:
    scores = np.asarray(scores)
    mx = np.max(scores)
    log_sum_exp = mx + np.log(np.sum(np.exp(scores - mx)))
    return scores - log_sum_exp
    
    # scores = np.asarray(scores)
    # mx = np.max(scores)
    # exp_scores = np.exp(scores - mx)
    # return np.log(exp_scores/exp_scores.sum())

# TC: O(n)
# SC: O(n)