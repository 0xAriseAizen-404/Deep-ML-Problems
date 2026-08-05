import numpy as np

def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
    weights = initial_weights[:].reshape(-1, 1)
    bias = initial_bias
    mse_values = []
    labels = labels.reshape(-1, 1)
    for _ in range(epochs):
        # Forward Pass, // ndarray's right so below operations will work, [1, 2, 3] + 3 && exp([1, 2, 3]) -> numpy operations are element-wise
        z = features.dot(weights) + bias
        probs = 1 / (1 + np.exp(-z))
        # Loss
        mse = np.mean((probs - labels) ** 2)
        mse_values.append(mse)
        # Backpropogation
        d_mse = 1  # d_loss/d_loss = 1
        d_probs = (2 / len(probs)) * (probs - labels)  # d_loss/d_probs
        d_z = d_probs * (probs * (1 - probs))  # d_loss/d_z = d_loss/d_probs * d_probs/d_z
        d_weights = features.T.dot(d_z)  # d_loss/d_weights = d_loss/d_z * d_z/d_weights
        d_bias = np.sum(d_z) * 1  # d_loss/d_bias = d_loss/d_z * d_z/d_bias
        # Update Step
        weights = weights - learning_rate * d_weights
        bias = bias - learning_rate * d_bias
    return (
    np.round(weights.ravel(), 4),
    round(float(bias), 4),
    [round(float(x), 4) for x in mse_values]
    )

# TC: O(epochs * m * n)
# SC: O(m + n)