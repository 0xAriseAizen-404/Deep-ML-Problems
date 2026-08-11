def poly_term_derivative(c: float, x: float, n: float) -> float:
    # Polynomial term at X point = c * x^n
    # Derivative of it, c * n * x^(n-1)
    return c * n * (x**(n-1))

# TC: O(1)
# SC: O(1)