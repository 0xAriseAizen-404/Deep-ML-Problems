import math

def softplus(x: float) -> float:
    """
    Compute the softplus activation function.
    Args:
        x: Input value

    Returns:
        The softplus value: log(1 + e^x)
    """
    return round(math.log(1 + math.exp(x)), 4)

# TC: O(1)
# SC: O(1)