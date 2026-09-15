import numpy as np

def soft_threshold(w: np.ndarray, threshold: float) -> np.ndarray:
    """Apply soft-thresholding operator element-wise.
    
    S(w, Î») = sign(w) * max(|w| - Î», 0)
    
    Args:
        w: Input array
        threshold: Threshold value Î»
    
    Returns:
        Soft-thresholded array where:
        - Values with |w| > Î» are shrunk toward zero by Î»
        - Values with |w| â¤ Î» become exactly zero
    """
    sign_w = np.where(w < 0, -1, 1)
    # sign_w = np.sign(w)
    return sign_w * np.maximum(np.abs(w) - threshold, 0)

def l1_regularization_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float = 0.1, learning_rate: float = 0.01, max_iter: int = 1000, tol: float = 1e-4) -> tuple:
    """
    Implement Lasso Regression using ISTA (Iterative Shrinkage-Thresholding Algorithm).
    
    ISTA alternates between:
    1. Gradient step on MSE loss: w_temp = w - lr * gradient_mse
    2. Proximal step (soft-thresholding): w_new = soft_threshold(w_temp, lr * alpha)
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Target vector of shape (n_samples,)
        alpha: L1 regularization strength
        learning_rate: Step size for gradient descent
        max_iter: Maximum iterations
        tol: Convergence tolerance on weight change
    
    Returns:
        tuple: (weights, bias)
    
    Note: The bias term is NOT regularized.
    """
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0.0

    for _ in range(max_iter):
        error = np.dot(X, weights) + bias - y
        grad_w = (1 / n_samples) * np.dot(X.T, error)
        grad_b = np.mean(error)
        
        w_temp = weights - learning_rate * grad_w
        weights_new = soft_threshold(w_temp, learning_rate * alpha)
        bias -= learning_rate * grad_b
        if np.linalg.norm(weights_new - weights) < tol:
            weights = weights_new
            break
        weights = weights_new

    return weights, bias