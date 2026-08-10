import numpy as np

class GradientDescent:
    def __init__(self, X, y, weights, learning_rate, n_epochs):
        self.X = X
        self.m = self.X.shape[0]
        self.y = y.reshape(-1, 1)
        self.weights = np.array(weights, dtype=float).reshape(-1, 1)
        self.learning_rate = learning_rate
        self.epochs = n_epochs
        
    def batchGD(self):
        for _ in range(self.epochs):
            y_pred = self.X.dot(self.weights)
            errors = y_pred - self.y
            gradients = (2 / self.m) * self.X.T.dot(errors)
            self.weights -= self.learning_rate * gradients
        return self.weights.flatten()
    def stochasticGD(self):
        for _ in range(self.epochs):
            for i in range(self.m):
                sample = self.X[i].reshape(1, -1)
                target = self.y[i].reshape(1, -1)
                y_pred = sample.dot(self.weights)
                errors = y_pred - target
                gradients = 2 * sample.T.dot(errors)
                self.weights -= self.learning_rate * gradients
        return self.weights.flatten()
    def miniBatchGD(self, batch_size):
        for _ in range(self.epochs):
            for ind in range(0, self.m, batch_size):
                samples = self.X[ind : ind + batch_size]
                targets = self.y[ind : ind + batch_size]
                y_pred = samples.dot(self.weights)
                errors = y_pred - targets
                gradients = (2 / samples.shape[0]) * samples.T.dot(errors)
                self.weights -= self.learning_rate * gradients
        return self.weights.flatten()

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization.
    
    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')
    
    Returns:
        Optimized weights
    """
    obj = GradientDescent(X, y, weights, learning_rate, n_epochs)
    if method == 'batch':
        return obj.batchGD()
    elif method == 'stochastic':
        return obj.stochasticGD()
    elif method == 'mini_batch':
        return obj.miniBatchGD(batch_size)
    else:
        raise ValueError("method must be 'batch', 'stochastic', or 'mini_batch'")

# Batch GD
# TC: O(E*m*n)
# SC: O(m*n)

# SGD
# TC: O(E*m*n)
# SC: O(m*n)

# Mini-Batch GD
# TC: O(E*m*n)
# SC: O(m*n)