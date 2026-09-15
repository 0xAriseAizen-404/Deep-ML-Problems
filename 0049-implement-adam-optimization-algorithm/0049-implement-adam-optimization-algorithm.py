import numpy as np

def adam_optimizer(f, grad, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=10):
    # Note: `f` is accepted only for interface parity with the PyTorch/Tinygrad
    # variants, which derive gradients from it via autograd. This version uses
    # `grad` only — the objective value `f` is never evaluated.
    # Your code here
    
    first_moment = np.zeros_like(x0, dtype=float)
    second_moment = np.zeros_like(x0, dtype=float)
    parameters = np.array(x0, dtype=float, copy=True)

    for t in range(1, num_iterations + 1):
        gradient = grad(parameters)
        first_moment = beta1 * first_moment + (1 - beta1) * gradient
        second_moment = beta2 * second_moment + (1 - beta2) * gradient**2
        bias_first_moment = first_moment / (1 - beta1**t)
        bias_second_moment = second_moment / (1 - beta2**t)
        parameters = parameters - (
            learning_rate
            * bias_first_moment
            / (np.sqrt(bias_second_moment) + epsilon)
        )
    return parameters

# TC: O(Tn)
# SC: O(1)