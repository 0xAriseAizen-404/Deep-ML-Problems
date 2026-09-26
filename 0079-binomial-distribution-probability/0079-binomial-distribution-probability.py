import math
def binomial_probability(n: int, k: int, p: float) -> float:
    """
    Calculate the probability of exactly k successes in n Bernoulli trials.
    
    Args:
        n: Total number of trials
        k: Number of successes
        p: Probability of success on each trial
    
    Returns:
        Probability of k successes
    """
    def fact(n: int) -> int:
        if n == 0 or n == 1:
            return 1
        return n * fact(n-1)

    def nck(n: int, k: int) -> float:
        return fact(n) / (fact(k) * fact(n-k))

    # bionomial_coefficient = nck(n, k) * (p ** k) * ((1 - p) ** (n - k))
    bionomial_coefficient = math.comb(n, k) * math.pow(p, k) * math.pow((1 - p), (n - k))
    return bionomial_coefficient