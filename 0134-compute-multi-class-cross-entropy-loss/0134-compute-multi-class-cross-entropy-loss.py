import numpy as np

def compute_cross_entropy_loss(predicted_probs: np.ndarray, true_labels: np.ndarray, epsilon = 1e-15) -> float:
    # return -np.mean(np.log(predicted_probs[true_labels.astype(bool)]))

    selected = predicted_probs[true_labels.astype(bool)]
    selected = np.clip(selected, epsilon, 1.0)
    return -np.mean(np.log(selected))